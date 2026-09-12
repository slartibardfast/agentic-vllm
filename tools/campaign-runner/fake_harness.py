#!/usr/bin/env python3
"""Test-seam harness for campaign-runner's selftest.

Mimics the ``weco local run`` contract the runner supervises: improve the
workspace (append a line to progress.txt), run the eval command, append a
step record to <workdir>/.weco/local-loop/steps.jsonl. Hooks via env:

FAKE_HANG_STEP=<n>   sleep forever during step n, before appending its line
                     (watchdog / kill -9 tests)
FAKE_CRASH_ONCE=1    exit 3 at step 0 on the first invocation only (retry test)
FAKE_LIE=1           exit 0 after one step regardless of --steps (lie test)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

MARKER = ".fake-crashed"


def parse_metric(output: str, metric: str) -> float | None:
    value = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(f"{metric}:"):
            try:
                value = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("-e", "--eval-command", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("-g", "--goal", default="max")
    args = ap.parse_args()
    workdir = pathlib.Path(args.workdir).resolve()
    d = workdir / ".weco" / "local-loop"
    d.mkdir(parents=True, exist_ok=True)
    maximize = args.goal in ("max", "maximize")

    crash_once = "FAKE_CRASH_ONCE" in __import__("os").environ
    if crash_once and not (d / MARKER).exists() and args.steps > 1:
        (d / MARKER).write_text("1")
        __import__("sys").exit(3)

    best = None
    for step in range(args.steps):
        if str(step) == __import__("os").environ.get("FAKE_HANG_STEP", ""):
            time.sleep(999)
        delay = __import__("os").environ.get("FAKE_STEP_DELAY")
        if delay:
            time.sleep(float(delay))
        with (workdir / "progress.txt").open("a") as f:
            f.write(f"step-{step}\n")
        r = subprocess.run(["bash", "-c", args.eval_command],
                           cwd=str(workdir), capture_output=True, text=True)
        value = parse_metric((r.stdout or "") + (r.stderr or ""), args.metric)
        improved = value is not None and (
            best is None or (value > best) == maximize)
        if improved:
            best = value
        with (d / "steps.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "value": value,
                                "improved_best": improved,
                                "output_tail": (r.stdout or "")[-500:]})
                    + "\n")
        if "FAKE_LIE" in __import__("os").environ and step == 0:
            break


if __name__ == "__main__":
    main()
