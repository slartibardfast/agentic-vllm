#!/usr/bin/env python3
"""campaign-runner: supervision for long unattended weco-local campaigns.

Wraps ``weco local run`` (the local-only fork, opencode harness) with the
supervision the fork lacks, per plan/0008 and DOCTRINE.md:

- three-valued eval outcomes (VALUE / CANDIDATE_CRASH / INFRA_CRASH)
  with retry-once on harness failure and a circuit breaker (three
  consecutive infra crashes, or a 30% infra rate over ten or more
  steps, pauses the campaign instead of burning steps);
- an eval timeout: the eval command runs under a session-isolated guard
  (TERM, grace, KILL to the whole process group); a timed-out eval is
  INFRA, never a silently-null metric;
- commit-before-append resume: every improved step is committed in the
  workdir and recorded with its sha in a side-car journal; ``--resume``
  reconstructs step count, best value, and the incumbent tree from that
  journal, discards the torn tail of a killed run, and never re-uses a
  cached metric as a fresh measurement;
- a stale-append watchdog: no step appended within --watchdog seconds
  means a wedged harness - kill the tree, write a post-mortem, exit
  resumable;
- the opencode harness contract: stdin from the void, PATH extended
  with ~/.opencode/bin, autoupdate disabled, and an exit-0 lie check
  (the harness must deliver the steps it promised);
- a per-workdir lock, and a champion-ledger bridge carrying intent
  fields (question / mechanism / disposition / provenance).

Exit codes: 0 complete; 2 terminal (harness failed twice, or an exit-0
lie); 3 circuit breaker (resumable); 4 watchdog (resumable).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

RUNNER_DIR = ".weco/local-loop"
SIDECAR = "runner-journal.jsonl"
STATE = "runner-state.json"
LOCK = "runner.lock"
EVAL_GUARD = "eval-guard.sh"
EVAL_STATUS = "eval-status.json"
BEST_REF = "weco/best"

VALUE, CANDIDATE_CRASH, INFRA_CRASH = "VALUE", "CANDIDATE_CRASH", "INFRA_CRASH"

EVAL_GUARD_SRC = """#!/usr/bin/env bash
# campaign-runner eval guard: session-isolated eval, TERM -> grace -> KILL
# to the whole process group on timeout. Passes output through, exits with
# the eval's own exit code, records what happened in $EVAL_STATUS.
T="$1"; shift
setsid "$@" & pid=$!
flag="$EVAL_STATUS.timed"
rm -f "$flag"
timed=0
# the watcher must not hold the caller's stdout/stderr pipe open, or a
# capturing parent blocks until the timeout expires even on success;
# it signals a timeout via a flag file (a subshell variable would not
# propagate to this shell)
( sleep "$T"
  touch "$flag"
  kill -TERM -- -"$pid" 2>/dev/null
  sleep 5
  kill -KILL -- -"$pid" 2>/dev/null ) >/dev/null 2>&1 & wpid=$!
wait "$pid"; rc=$?
kill -KILL "$wpid" 2>/dev/null
wait "$wpid" 2>/dev/null
if [ -f "$flag" ]; then timed=1; rm -f "$flag"; fi
printf '{"rc": %s, "timed_out": %s}\\n' "$rc" "$timed" > "$EVAL_STATUS"
exit "$rc"
"""


def log(msg: str) -> None:
    print(f"[campaign-runner] {msg}", flush=True)


def harness_env() -> None:
    """The opencode harness contract: PATH, no autoupdate, no stdin."""
    opencode = shutil.which("opencode") or os.path.expanduser(
        "~/.opencode/bin/opencode")
    if not os.access(opencode, os.X_OK):
        sys.exit("campaign-runner: opencode not found (~/.opencode/bin)")
    os.environ["PATH"] = (os.path.dirname(opencode) + os.pathsep
                          + os.environ.get("PATH", ""))
    os.environ["OPENCODE_DISABLE_AUTOUPDATE"] = "1"


def resolve_weco() -> str:
    weco = shutil.which("weco") or os.path.expanduser("~/.local/bin/weco")
    if not (os.path.isfile(weco) and os.access(weco, os.X_OK)):
        sys.exit("campaign-runner: weco not found (PATH or ~/.local/bin)")
    out = subprocess.run([weco, "--version"], capture_output=True,
                         text=True).stdout
    if "connollydavid/weco-cli" not in out:
        sys.exit(f"campaign-runner: refusing non-fork weco: {out.strip()!r}")
    return weco


def git(workdir: pathlib.Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(workdir), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"campaign-runner: git {args[0]} failed: {r.stderr.strip()}")
    return r.stdout


def load_state(d: pathlib.Path) -> dict:
    f = d / STATE
    if f.exists():
        return json.loads(f.read_text())
    return {"consumed_offset": 0, "steps_done": 0, "best": None,
            "best_step": None, "invocations": 0}


def journal_state(d: pathlib.Path, st: dict) -> dict:
    """Resume authority is the journal, not the state file: the kill that
    makes a resume necessary can land between a journal append and the
    state save, so the state file may lag by one step."""
    f = d / SIDECAR
    if not f.exists():
        return st
    last = None
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("event") == "step":
            last = rec
        elif rec.get("event") == "watchdog_postmortem":
            pass
    if last is None or "offset" not in last:
        return st
    st["steps_done"] = last["step"]
    st["consumed_offset"] = last["offset"]
    best = None
    best_step = None
    for line in f.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") == "step" and rec.get("improved_best"):
            best, best_step = rec.get("value"), rec["step"]
    st["best"], st["best_step"] = best, best_step
    return st


def save_state(d: pathlib.Path, st: dict) -> None:
    (d / STATE).write_text(json.dumps(st, indent=1) + "\n")


def journal(d: pathlib.Path, rec: dict) -> None:
    rec["t"] = round(time.time(), 1)
    with (d / SIDECAR).open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_new_lines(steps_file: pathlib.Path, offset: int):
    if not steps_file.exists():
        return [], offset
    data = steps_file.read_bytes()
    if len(data) <= offset:
        return [], offset
    chunk = data[offset:]
    cut = len(chunk) if chunk.endswith(b"\n") else chunk.rfind(b"\n") + 1
    lines = [l for l in chunk[:cut].decode("utf-8", "replace").splitlines()
             if l.strip()]
    return lines, offset + cut


def classify(rec: dict, d: pathlib.Path) -> str:
    if rec.get("value") is not None:
        return VALUE
    status = {}
    try:
        status = json.loads((d / EVAL_STATUS).read_text())
    except (OSError, json.JSONDecodeError):
        pass
    if status.get("timed_out"):
        return INFRA_CRASH  # eval hung: infra, never the candidate's fault
    if status.get("rc"):
        return CANDIDATE_CRASH  # eval ran and failed
    return INFRA_CRASH  # null with no guard record: eval never ran


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 15)
        time.sleep(3)
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def build_guard(d: pathlib.Path) -> None:
    g = d / EVAL_GUARD
    g.write_text(EVAL_GUARD_SRC)
    g.chmod(0o755)
    (d / EVAL_STATUS).unlink(missing_ok=True)


def wrap_eval(cmd: str, timeout_s: int, d: pathlib.Path) -> str:
    return (f"EVAL_STATUS={shlex.quote(str(d / EVAL_STATUS))} "
            f"bash {shlex.quote(str(d / EVAL_GUARD))} {timeout_s} "
            f"bash -c {shlex.quote(cmd)}")


def commit_best(workdir: pathlib.Path, step: int, value: float, st: dict,
                args) -> str:
    git(workdir, "add", "-A", "--", ".", ":(exclude).weco")
    git(workdir, "commit", "-q", "--allow-empty",
        "-m", f"campaign: step {step} improves {args.metric} to {value}")
    sha = git(workdir, "rev-parse", "HEAD").strip()
    git(workdir, "update-ref", BEST_REF, sha)
    st["best"], st["best_step"] = value, step
    if args.ledger:
        rec = {"event": "improved_best", "step": step, "metric": args.metric,
               "value": value, "sha": sha, "goal": args.goal,
               "intent": args.intent}
        with open(args.ledger, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return sha


def harness_command(args, d: pathlib.Path, remaining: int) -> list[str]:
    if args.harness_cmd:
        s = args.harness_cmd.replace("{{STEPS}}", str(remaining))
        if "{{EVAL}}" in s:
            s = s.replace("{{EVAL}}",
                          shlex.quote(wrap_eval(args.eval_cmd,
                                                args.eval_timeout, d)))
        return shlex.split(s)
    weco = resolve_weco()
    return [weco, "local", "run", "--harness", "opencode",
            "-e", wrap_eval(args.eval_cmd, args.eval_timeout, d),
            "--metric", args.metric,
            "-g", "maximize" if args.maximize else "minimize",
            "--steps", str(remaining),
            "--workdir", str(args.workdir_path)]


def supervise(args, d: pathlib.Path, st: dict,
              harness_cmd: list[str]) -> int:
    steps_file = d / "steps.jsonl"
    st["invocations"] += 1
    save_state(d, st)
    logf = open(d / "harness.log", "ab")
    proc = subprocess.Popen(harness_cmd, stdin=subprocess.DEVNULL,
                            start_new_session=True, stdout=logf,
                            stderr=subprocess.STDOUT)
    offset = st["consumed_offset"]
    last_append = time.time()
    infra_streak = 0

    def consume(lines, offset):
        nonlocal infra_streak, last_append
        for line in lines:
            rec = json.loads(line)
            outcome = classify(rec, d)
            st["steps_done"] += 1
            value = rec.get("value")
            entry = {"event": "step", "step": st["steps_done"],
                     "value": value, "outcome": outcome}
            improved = outcome == VALUE and (
                st["best"] is None
                or (value > st["best"]) == args.maximize)
            if improved:
                entry["sha"] = commit_best(args.workdir_path,
                                           st["steps_done"], value, st, args)
            entry["improved_best"] = bool(improved)
            infra_streak = infra_streak + 1 if outcome == INFRA_CRASH else 0
            last_append = time.time()
            entry["offset"] = offset
            journal(d, entry)
            save_state(d, st)
            # circuit breaker: wedged evals burn walltime, stop early
            if infra_streak >= 3 or (
                    len(outcomes_hist(d)) >= 10
                    and outcomes_hist(d).count(INFRA_CRASH)
                    / len(outcomes_hist(d)) >= 0.30):
                log("circuit breaker: evals infra-broken; pausing")
                journal(d, {"event": "circuit_breaker",
                            "steps_done": st["steps_done"]})
                save_state(d, st)
                return 3, offset
        return None, offset

    rc = None
    try:
        while st["steps_done"] < args.steps and proc.poll() is None:
            lines, offset = read_new_lines(steps_file, offset)
            if lines:
                stopped, offset = consume(lines, offset)
                if stopped == 3:
                    kill_tree(proc)
                    return 3
            elif time.time() - last_append > args.watchdog:
                log(f"watchdog: no step appended in {args.watchdog}s; "
                    "killing the harness tree")
                kill_tree(proc)
                journal(d, {"event": "watchdog_postmortem",
                            "steps_done": st["steps_done"]})
                save_state(d, st)
                return 4
            else:
                time.sleep(0.5)
        rc = proc.wait(timeout=120)
        # drain the tail the harness wrote before exiting
        while True:
            lines, offset = read_new_lines(steps_file, offset)
            if not lines:
                break
            stopped, offset = consume(lines, offset)
            if stopped == 3:
                return 3
            if st["steps_done"] >= args.steps:
                break
    finally:
        if proc.poll() is None:
            kill_tree(proc)
        logf.close()
        save_state(d, st)

    if st["steps_done"] >= args.steps:
        return 0
    if rc == 0:
        log(f"harness exited 0 after {st['steps_done']}/{args.steps} "
            "steps - exit-status lie")
        journal(d, {"event": "exit_zero_lie", "promised": args.steps,
                    "delivered": st["steps_done"]})
        return 2
    journal(d, {"event": "harness_exit", "rc": rc,
                "steps_done": st["steps_done"]})
    return 1


def outcomes_hist(d: pathlib.Path) -> list[str]:
    f = d / SIDECAR
    if not f.exists():
        return []
    return [json.loads(l).get("outcome")
            for l in f.read_text().splitlines()
            if l.strip() and json.loads(l).get("event") == "step"]


def main() -> None:
    ap = argparse.ArgumentParser(prog="campaign-runner")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--eval", required=True, dest="eval_cmd")
    ap.add_argument("--metric", required=True)
    ap.add_argument("--goal", default="max",
                    choices=["max", "maximize", "min", "minimize"])
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-timeout", type=int, default=600)
    ap.add_argument("--watchdog", type=int, default=1800)
    ap.add_argument("--harness-cmd", default=None,
                    help="test seam: override the weco local run command; "
                         "{{STEPS}} receives the remaining step count")
    ap.add_argument("--ledger", default=None,
                    help="champion-ledger bridge: append improved_best "
                         "records with intent fields here")
    ap.add_argument("--intent", default="{}",
                    help="JSON intent fields (question/mechanism/"
                         "disposition) carried onto ledger records")
    args = ap.parse_args()
    args.maximize = args.goal in ("max", "maximize")
    args.intent = args.intent if args.intent else "{}"
    args.workdir_path = pathlib.Path(args.workdir).expanduser().resolve()
    if not args.workdir_path.is_dir():
        sys.exit(f"campaign-runner: workdir {args.workdir_path} missing")
    if not ((args.workdir_path / ".git").exists()):
        sys.exit(f"campaign-runner: {args.workdir_path} is not a git "
                 "worktree; commit-before-append resume needs one")
    if not args.harness_cmd:
        harness_env()
        resolve_weco()

    d = args.workdir_path / RUNNER_DIR
    d.mkdir(parents=True, exist_ok=True)
    lock = open(d / LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"campaign-runner: {args.workdir_path} is already "
                 "supervised (lock held)")

    st = load_state(d)
    if args.resume:
        st = journal_state(d, st)
        # discard the torn tail: the tree returns to the incumbent
        # snapshot (weco/best if it exists, else the current HEAD)
        ref_sha = git(args.workdir_path, "rev-parse", "--verify",
                      "--quiet", BEST_REF, check=False).strip() \
            or git(args.workdir_path, "rev-parse", "HEAD").strip()
        git(args.workdir_path, "reset", "--hard", ref_sha)
    else:
        steps_file = d / "steps.jsonl"
        st["consumed_offset"] = (steps_file.stat().st_size
                                 if steps_file.exists() else 0)
        st["steps_done"] = 0
        st["best"] = st["best_step"] = None
        git(args.workdir_path, "update-ref", "-d", BEST_REF, check=False)

    build_guard(d)
    rc = 1
    for attempt in (1, 2):
        rc = supervise(args, d, st,
                       harness_command(args, d,
                                       args.steps - st["steps_done"]))
        if rc != 1 or attempt == 2:
            break
        log("harness failed once; retrying the remaining steps")
    if rc == 1:
        rc = 2
        log("harness failed twice; terminal INFRA")
    st["final_rc"] = rc
    save_state(d, st)
    print("CAMPAIGN_SUMMARY " + json.dumps(
        {"rc": rc, "steps_done": st["steps_done"], "steps": args.steps,
         "best": st["best"], "best_step": st["best_step"]}))
    sys.exit(rc)


if __name__ == "__main__":
    main()
