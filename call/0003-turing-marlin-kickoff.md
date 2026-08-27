# Kick off the Turing Marlin backend program

- Status: accepted
- Scope: software
- Date: 2026-08-27

## Context and Problem Statement

The host now governs a vLLM fork, and the operator has supplied a complete
implementation research plan for a high-performance Turing (sm_75) backend for
the Marlin kernel ecosystem in vLLM. The plan's central rule is to preserve the
Marlin contract and replace only the machinery that TU102 cannot execute
efficiently; interop is the invariant and the kernel architecture is the
variable. The program needs a home: one milestone that carries the plan as a
receipted task graph, recorded hardware facts, and the decisions about where
each deliverable lives.

## Decision

Adopt the research plan as milestone plan/0002 (turing marlin backend), the
first real program of this host. Its tasks, dependencies, and verification
commands live in that milestone's build sequence; this decision records the
frame the tasks operate inside.

- Hardware: this host carries two Quadro RTX 6000 cards (TU102, sm_75, 24 GB),
  driver 610.57.04, CUDA 13.3 at /opt/cuda; graphics clocks are locked to
  1455 MHz on both cards for measurement stability. rope (192.168.178.60,
  reachable as dconnolly) carries an RTX 3060 Ti (sm_86), driver 610.43.02,
  graphics clocks locked to 1665 MHz; rope is the native Ampere reference rig
  the final comparison runs against.
- Software: kernel work happens on a branch of the hosted fork
  (slartibardfast/vllm, pin 79378fe at kickoff), never on the pin itself; the
  recorded pin advances only when a coherent state lands on the branch and is
  pushed.
- Analysis deliverables (the contract reconstruction, the performance model,
  the final report) live as documents in the milestone folder and reference
  the exact paths they trace in the worktree.

## Consequences

- The build sequence below plan/0002 is the authority for ordering and
  verification; receipts for its tasks are checked by the gate.
- Benchmarks are only trusted when run under locked clocks on the recorded
  rigs; any number cited in a report names its rig and clock.
- The incumbent Turing path in vLLM (an explicit sm_75 branch in the Marlin
  sources, two-stage pipeline, FP16 or INT8 activations) is the baseline to
  characterize and beat, not an assumed failure.
