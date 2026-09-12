# Campaign plateau: the sm75 bridge campaign driven to its plateau

The working queue for the plateau session opened 2026-09-12, moved here
wholesale from the lane runbook
(`software/vllm/sm75-marlin/turing_lab/thirdparty/NEXT-WINDOW-RUNBOOK.md`,
which now points at this milestone instead of holding the queue). The
reason for the move is the research-integration principle applied one
level down: the task machinery reads plan READMEs, and a work queue
that lives only in the lane is invisible to it — no receipts, no ready
frontier, no input-digest staleness.

The exit contract is the doctrine's (D1 in
`research/longrange-agentic/DOCTRINE.md`): drain the build sequence in
dependency order; close the session at queue exhaustion or on the
operator's word, never at a phase boundary. GPU blocks are the spine;
the CPU tasks fill the waits inside long GPU runs. Out of scope, named
so the exhaustion test stays honest: the register-dequant+repack W4A16
surgery (days-grade; its own window) and the weco-skill pin bump
(blocked on the operator pushing the unpushed weco-skill commit; see
MEMORY).

Standing numbers this queue works against live in the records, not
here: the bridge committed line and its baseline ratio in
`software/vllm/sm75-marlin/turing_lab/results/committed-both-card-noneager/COMMITTED-BRIDGE-SPLIT.md`,
the convergence record beside it, and the champion ledger at
`turing_lab/results/ab-champion-ledger.md` (paths relative to the
`sm75-marlin` worktree).

## Build sequence

### Preflight the window {#preflight}

Stop llama-server through its systemd unit (it runs as the `llm`
service user), lock both cards, and run the six-check battery: GPUs
free, clocks locked, CUDA_HOME carried inline, the flashinfer
version-check bypass, the bridge d256 kernel smoke on both model
shapes, fixtures present. All checks must pass before anything else in
this sequence starts; the battery encodes every environment failure a
previous window paid for.

- verify: cd software/vllm/sm75-marlin && bash turing_lab/thirdparty/preflight_window.sh
- inputs: software/vllm/sm75-marlin/turing_lab/thirdparty/preflight_window.sh

### Root-cause multi-row paged attention {#multirow-paged-rootcause}

The MTP-times-paged failure is narrowed to its traffic class: multi-row
paged decode, K-independent (red at K=1), oracle-blind (the standalone
battery is green at every row width — the failure needs the engine's
cache state), and the draft phase is exonerated. Instrument the paged
arm under real engine traffic — dump page tables, cache state, and the
metadata the arm receives at the failing step — until the mechanism is
named, then re-run gate-0 with the paged arm serving MTP verification
rows. Two acceptable terminal states: the gate passes green (greedy
match across the battery), or a written mechanism diagnosis with the
discriminating evidence named, landed beside the lane's records. Until
this closes, MTP verification rides the gather arm.

- depends: #preflight
- verify: attested operator

### Sweep MTP K on the champion path {#k-sweep-champion}

Measure the draft-token sweep at K in {1, 2} on the champion
configuration — decode on the paged-split kernel, verify and prefill on
the gather arm, CUDA graphs on, the 27B checkpoint — completing the K
tree whose K=3 point is banked in the convergence record. Protocol:
greedy-lossless canaries, medians across three engine restarts, labels
asserted from engine logs, zero preemptions, the champion ledger
appended with the intent fields.

- depends: #preflight
- verify: attested operator
- inputs: software/vllm/sm75-marlin/turing_lab/results/committed-both-card-noneager/CONVERGENCE-RUN.md

### Close the per-token inner-loop residual {#inner-loop}

The named residual between the split kernel's decode and the TRITON
control line is the per-token inner loop. Profile first, then one
surgery at a time, each through the full gate train: the oracle
batteries (standard and long-context, split path) after every step,
champion A/B against the incumbent configuration before any ledger
change. The champion moves only through the ledger gates; a falsified
surgery is recorded with its mechanism and the loop re-plans.

- depends: #preflight
- verify: cd software/vllm/sm75-marlin && SPLIT=1 CUDA_HOME=/opt/cuda .venv/bin/python ../../../.weco/c2-paged-decode/oracle.py
- inputs: software/vllm/sm75-marlin/turing_lab/bridge/bridge_paged_decode_split.cu

### Probe INT8 quantization {#int8-probe}

Quantize a fixture at weight-and-activation 8-bit through the autoround
recipe (the lane runbook's fixture-reproduction section holds the
command pattern and its pacing notes), then measure decode against the
W4A16 incumbent on identical rows under the committed-run protocol:
medians across three restarts, labels asserted, zero preemptions. The
question is headroom: what a narrower activation path buys on this
silicon versus the recorded W4A16 line.

- depends: #preflight
- verify: attested operator

### Build the campaign runner {#campaign-runner}

The approved design's runner: a wrapper around `weco local run`
implementing three-valued eval outcomes with the circuit breaker, eval
timeout via process-group kill, commit-before-append resume with a
never-reuse-cached-metrics rule, the stale-append heartbeat watchdog,
the opencode harness contract (stdin from the void, PATH resolution
preflight, an exit-status lie detected by requiring the assistant text
event), a per-workdir lock, and the champion-ledger bridge with intent
fields. CPU-only; fills the waits inside GPU blocks. The smoke test is
the three-part battery: three green steps on a toy eval, a kill mid
step followed by resume proving reconstruction, and a hung eval hitting
the timeout and continuing.

- depends: (none)
- verify: python3 tools/campaign-runner/selftest.py

### Raise the weco-cli upstream PR {#weco-cli-pr}

On the local-only fork, by the fork-branch-and-PR route the login-guard
bug took: local-mode resume from the step log, an eval timeout knob,
harness-failure continue-with-retry, and the stdin/PATH bridge fixes —
each citing the campaign runner as the local proof of concept. Blocked
on the runner existing, not on any GPU work.

- depends: #campaign-runner
- verify: attested operator

### Close out the session {#close-out}

Ledger and records current, the lane runbook's status line updated,
clocks reset, llama-server back up through its unit and health-checked,
lane and host commits pushed, the pins moved to the closed state, every
task in this sequence carrying its receipt.

- depends: #multirow-paged-rootcause, #k-sweep-champion, #inner-loop, #int8-probe, #campaign-runner, #weco-cli-pr
- verify: attested operator
