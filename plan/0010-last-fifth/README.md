# The last fifth: closing the bridge-to-baseline decode gap

The 12-hour window opened 2026-09-14 on the operator's grant, with
the operator's process directive: cull options quicker. The standing
numbers this queue works against (paths relative to the sm75-marlin
worktree): the champion 41.3/41.2/195.5 and the TRITON baseline rows
in `turing_lab/results/ab-champion-ledger.md`; the arithmetic that
reframes the gap in `turing_lab/results/inner-loop-surgery/PROFILE-DELTA.md`.

The framing this queue tests: the residual is not bandwidth. At
ctx2048 the gap is ~4.3 ms of a 24.3 ms step while attention's KV
traffic is ~0.12 ms at the DRAM wall - the split architecture's
workspace round-trip, its second launch, and the step's serial
structure are the suspects. Bandwidth-level kernel surgery is
falsified territory (plan/0009); structure is the target.

## The culling discipline (new, operator-directed)

Two tiers. Gate-1 (cull tier): ONE engine start, five in-process
reps per row via the macro-gate arm, ~10 minutes per option;
directional verdicts at the measured in-process band. The committed
protocol (three fresh restarts, medians, labels) is spent only on
gate-1 survivors. Gate-1 is validated once against the known
baseline rows before its first cull. Isolated oracle timing never
culls anything (the plan/0009 inversion lesson); it only proves
numerics.

## Build sequence

### Validate gate-1 against the committed baseline {#gate1-validate}

One macro-gate arm at the champion configuration; its rows must
reproduce the committed medians within the in-process bands. This
buys the cull tier's license for the rest of the window.

- depends: (none)
- verify: attested operator
- inputs: software/vllm/sm75-marlin/turing_lab/thirdparty/vllm_macro_gate.py

### Re-sweep the split knee post-surgery {#pps-resweep}

The PPS knee (1-2) was chosen on the pre-surgery kernel's economics;
the register surgery changed the inner loop, not the split math, and
PPS1 halves the workspace and combine reads. Gate-1 arms at PPS1 and
PPS4 against the validated PPS2 arm; a gate-1 winner earns one
committed arm.

- depends: #gate1-validate
- verify: attested operator

### Decompose the decode step {#step-profile}

The operator's profiling directive: nsys (or chrome-trace) timelines
of BOTH backends' decode steps at ctx512 and ctx2048 plus the short
row - per-kernel durations AND inter-kernel gaps, bridge vs TRITON,
GDN kernels included. Output: a named time ledger (kernel time, gap
time, allreduce time per step) that decides which surgery can pay.
The runbooks and results live under turing_lab/results/step-profile/.

- depends: #gate1-validate
- verify: attested operator

### Run the scheduling corpus {#corpus}

The approved nine-item corpus (research/decode-step-scheduling/):
megakernels, attention-to-GEMM tile fusion, graph-node concurrency,
the step roofline method, the March-September recalibration, the
TRITON kernel source read, GDN step coupling, small-batch
parallelism, and the NVLink allreduce share. CPU-only; fills the
GPU waits. Integration per call/0005.

- depends: (none)
- verify: attested operator
- inputs: research/decode-step-scheduling/outline.yaml

### Cull and commit the surgery shortlist {#surgeries}

From the step-profile ledger and the corpus: the imaginative
scheduling surgeries (combine fusion via the banked single-kernel
reduction patterns, graph-structure fixes, GDN overlap,
small-batch-parallelism restructuring), each authored and culled at
gate-1 cadence; every survivor runs the committed protocol and only
a committed WIN moves the champion ledger. A falsified surgery is a
recorded terminal state with its mechanism.

- depends: #step-profile, #pps-resweep, #corpus
- verify: attested operator

### Close out the window {#close-out}

Ledger and records current, runbook status set, clocks reset,
llama-server restored and health-checked, lane pushed then pin moved
then host pushed, every task receipted.

- depends: #surgeries
- verify: attested operator

## Execution record (2026-09-15)

- #corpus DONE (receipted): 9/9 items validated at full field
  coverage, dated sources, integrated into CONSOLIDATION (the
  last-fifth section). The corpus converges on one arbitration for
  the surgeries task: the step-profile ledger decides
  dependency-chain (fused walk+combine wins: last-CTA combine,
  CUTLASS-semaphore, or FlashDecoding++ unified-max) vs SM-idle
  (graph fork/join, DBO-shaped) vs collective fallback (NCCL ring
  vs one-shot; only P2P ld/st is portable on the NVLink bridge).
  The roofline headline: step floor 18.0-18.3 ms; TRITON ~2 ms
  above it; the champion ~6.2 ms; the residual cannot be byte
  volume - it is latency regime and step structure, ~539 us/layer
  of excess across the 8 full-attn layers. Also banked: the FA
  split-count sizing rule for capture time; TU102 is 72 SMs; no
  cc 7.5-capable SoTA in the Mar-Sep 2026 delta; FlashInfer sm_75
  restore one PR away (#55380).
