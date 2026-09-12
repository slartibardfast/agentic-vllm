#!/usr/bin/env python3
"""campaign-runner selftest: the plan/0008 smoke battery, hermetic.

T1  three green steps on the toy eval (line count of a file the steps
    mutate); improved steps commit and move the weco/best ref.
T2  kill -9 the runner mid-campaign, then --resume: reconstruction from
    the side-car journal, torn tail discarded, campaign completes.
T3  a hung eval hits the eval timeout, is classified INFRA, and the
    campaign continues past it.
T4  an eval that hangs every time trips the circuit breaker (exit 3).
T5  an exit-0 lie (harness promises steps, delivers one) is terminal.
T6  a harness crash is retried once; the retry completes.

Exit 0 iff every case passes.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = sys.executable + " " + str(HERE / "campaign_runner.py")
FAKE = sys.executable + " " + str(HERE / "fake_harness.py")

LINE_EVAL = ("wc -l < progress.txt | awk '{printf \"lines: %d\\n\", $1}'")
HANG_ONCE_EVAL = ("if [ -e .hung ]; then wc -l < progress.txt | "
                  "awk '{printf \"lines: %d\\n\", $1}'; "
                  "else touch .hung; sleep 999; fi")
HANG_ALWAYS_EVAL = "sleep 999"


def fresh_workdir(tmp: Path, name: str) -> Path:
    w = tmp / name
    w.mkdir()
    (w / "progress.txt").write_text("seed\n")
    def git(*a):
        subprocess.run(["git", "-C", str(w), *a], check=True,
                       capture_output=True)
    subprocess.run(["git", "init", "-q", str(w)], check=True,
                   capture_output=True)
    git("config", "user.email", "selftest@localhost")
    git("config", "user.name", "selftest")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    return w


def run_runner(w: Path, extra: list[str]) -> subprocess.Popen:
    cmd = RUNNER.split() + ["--workdir", str(w), "--metric", "lines",
                            "--goal", "max"] + extra
    return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def journal_steps(w: Path) -> list[dict]:
    f = w / ".weco/local-loop/runner-journal.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines()
            if l.strip() and json.loads(l).get("event") == "step"]


def wait_for(cond, timeout=60, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.3)
    sys.exit(f"SELFTEST TIMEOUT waiting for {what}")


def t1_green(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t1")
    p = run_runner(w, ["--eval", LINE_EVAL, "--steps", "3",
                       "--harness-cmd",
                       f"{FAKE} --steps {{{{STEPS}}}} --workdir {w} "
                       f"-e {{{{EVAL}}}} --metric lines -g max"])
    out = p.communicate(timeout=120)[0]
    assert p.returncode == 0, f"T1 rc={p.returncode}: {out[-800:]}"
    steps = journal_steps(w)
    assert [s["outcome"] for s in steps] == ["VALUE"] * 3, steps
    assert any(s["improved_best"] for s in steps), steps
    ref = subprocess.run(["git", "-C", str(w), "rev-parse", "weco/best"],
                         capture_output=True, text=True)
    assert ref.returncode == 0, "weco/best ref missing"
    print("T1 PASS (3 green steps, best committed, weco/best moved)",
          flush=True)


def t2_kill9_resume(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t2")
    p = run_runner(w, ["--eval", LINE_EVAL, "--steps", "4",
                       "--watchdog", "60",
                       "--harness-cmd",
                       f"env FAKE_STEP_DELAY=2 {FAKE} --steps {{{{STEPS}}}} "
                       f"--workdir {w} -e {{{{EVAL}}}} --metric lines -g max"])
    try:
        wait_for(lambda: len(journal_steps(w)) >= 1, 90,
                 "first step journal line")
        time.sleep(0.4)
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)  # session dies hard
    finally:
        p.wait(timeout=30)
        subprocess.run(["pkill", "-9", "-f",
                        f"fake_harness.py.*{w.name}"], capture_output=True)
    steps_before = len(journal_steps(w))
    assert 1 <= steps_before < 4, steps_before

    p2 = run_runner(w, ["--eval", LINE_EVAL, "--steps", "4", "--resume",
                        "--harness-cmd",
                        f"{FAKE} --steps {{{{STEPS}}}} --workdir {w} "
                        f"-e {{{{EVAL}}}} --metric lines -g max"])
    out = p2.communicate(timeout=180)[0]
    assert p2.returncode == 0, f"T2 resume rc={p2.returncode}: {out[-800:]}"
    steps = journal_steps(w)
    assert len(steps) == 4, steps
    # values count progress-file lines: the resume reset to the incumbent
    # snapshot (2 lines) and the remaining steps appended 3 more
    assert steps[-1]["value"] == 5.0, steps
    print("T2 PASS (kill -9 mid-campaign; --resume reconstructed and "
          f"finished; {steps_before} pre-kill steps carried)", flush=True)


def t3_hung_eval_once(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t3")
    p = run_runner(w, ["--eval", HANG_ONCE_EVAL, "--steps", "3",
                       "--eval-timeout", "3",
                       "--harness-cmd",
                       f"{FAKE} --steps {{{{STEPS}}}} --workdir {w} "
                       f"-e {{{{EVAL}}}} --metric lines -g max"])
    out = p.communicate(timeout=180)[0]
    assert p.returncode == 0, f"T3 rc={p.returncode}: {out[-800:]}"
    outcomes = [s["outcome"] for s in journal_steps(w)]
    assert outcomes[0] == "INFRA_CRASH", outcomes
    assert outcomes.count("VALUE") == 2, outcomes
    print("T3 PASS (hung eval -> INFRA after timeout; campaign continued)", flush=True)


def t4_circuit_breaker(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t4")
    p = run_runner(w, ["--eval", HANG_ALWAYS_EVAL, "--steps", "5",
                       "--eval-timeout", "2",
                       "--harness-cmd",
                       f"{FAKE} --steps {{{{STEPS}}}} --workdir {w} "
                       f"-e {{{{EVAL}}}} --metric lines -g max"])
    out = p.communicate(timeout=180)[0]
    assert p.returncode == 3, f"T4 rc={p.returncode}: {out[-800:]}"
    outcomes = [s["outcome"] for s in journal_steps(w)]
    assert outcomes == ["INFRA_CRASH"] * 3, outcomes
    print("T4 PASS (3 consecutive infra evals -> circuit breaker, exit 3)", flush=True)


def t5_exit0_lie(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t5")
    env_cmd = f"env FAKE_LIE=1 {FAKE} --steps {{{{STEPS}}}} --workdir {w} " \
              f"-e {{{{EVAL}}}} --metric lines -g max"
    p = run_runner(w, ["--eval", LINE_EVAL, "--steps", "3",
                       "--harness-cmd", env_cmd])
    out = p.communicate(timeout=120)[0]
    assert p.returncode == 2, f"T5 rc={p.returncode}: {out[-800:]}"
    print("T5 PASS (exit-0 after 1 of 3 steps -> classified, exit 2)", flush=True)


def t6_crash_retry(tmp: Path) -> None:
    w = fresh_workdir(tmp, "t6")
    env_cmd = f"env FAKE_CRASH_ONCE=1 {FAKE} --steps {{{{STEPS}}}} " \
              f"--workdir {w} -e {{{{EVAL}}}} --metric lines -g max"
    p = run_runner(w, ["--eval", LINE_EVAL, "--steps", "3",
                       "--harness-cmd", env_cmd])
    out = p.communicate(timeout=180)[0]
    assert p.returncode == 0, f"T6 rc={p.returncode}: {out[-800:]}"
    assert len(journal_steps(w)) == 3
    print("T6 PASS (harness crash retried once; retry completed)", flush=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="campaign-runner-selftest") as t:
        tmp = Path(t)
        t1_green(tmp)
        t2_kill9_resume(tmp)
        t3_hung_eval_once(tmp)
        t4_circuit_breaker(tmp)
        t5_exit0_lie(tmp)
        t6_crash_retry(tmp)
    print("CAMPAIGN-RUNNER SELFTEST: ALL PASS", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
