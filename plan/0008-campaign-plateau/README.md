# Campaign plateau: the sm75 bridge campaign driven to its plateau

The working queue for the plateau session opened 2026-09-12, moved here
wholesale from the lane runbook
(`software/vllm/sm75-marlin/turing_lab/thirdparty/NEXT-WINDOW-RUNBOOK.md`,
which now points at this milestone instead of holding the queue). The
reason for the move is the research-integration principle applied one
level down: the task machinery reads plan READMEs, and a work queue
that lives only in the lane is invisible to it: no receipts, no ready
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
battery is green at every row width, so the failure needs the engine's
cache state), and the draft phase is exonerated. Instrument the paged
arm under real engine traffic: dump page tables, cache state, and the
metadata the arm receives at the failing step, until the mechanism is
named, then re-run gate-0 with the paged arm serving MTP verification
rows. Two acceptable terminal states: the gate passes green (greedy
match across the battery), or a written mechanism diagnosis with the
discriminating evidence named, landed beside the lane's records. Until
this closes, MTP verification rides the gather arm.

- depends: #preflight
- verify: attested operator

### Sweep MTP K on the champion path {#k-sweep-champion}

Measure the draft-token sweep at K in {1, 2} on the champion
configuration: decode on the paged-split kernel, verify and prefill on
the gather arm, CUDA graphs on, the 27B checkpoint. This completes the K tree whose
K=3 point is banked in the convergence record. Protocol:
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
harness-failure continue-with-retry, and the stdin/PATH bridge fixes,
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

## Execution record (2026-09-12/13, in flight)

- #multirow-paged-rootcause DONE (receipted). Mechanism: the glue's
  `query[:n*qlen].view(n, H, qlen, D)` flat-viewed the engine's
  token-major verify rows into the kernel's head-major contract:
  flawless attention on scrambled q for any q_len > 1; invisible at
  q_len == 1 (all committed numbers), oracle-invisible (the oracle
  feeds natively-ordered q). Found by instrumenting the live engine
  (BRIDGE_PAGED_DUAL dual-arm diff, 774 verify steps: deterministic,
  seq_lens and cache writes clean) plus a one-shot raw-dump
  brute-force layout matrix. Fixed input-side repack and output-side
  regroup (lane 734cd84be4); post-fix discriminator greedy 5/5 with
  max logit diff 0.031 (tie-break class). Record:
  turing_lab/results/committed-both-card-noneager/MULTIROW-PAGED-ROOT-CAUSE.md.
- #campaign-runner DONE (receipted). tools/campaign-runner +
  selftest 6/6 (green+commit, kill-9 resume, hung-eval INFRA,
  breaker, exit-0 lie, crash retry). Host db19bc0.
- #weco-cli-pr DONE (receipted). connollydavid/weco-cli#2: --resume,
  --eval-timeout, harness retry-then-abort, bridge stdin/PATH; 5 new
  tests, 35 regression-green.
- #k-sweep-champion RUNNING (K in {1,2} x 3 reps, 27B, graphs,
  champion path PPS2).
- #k-sweep-champion DONE (receipted). The 27B champion-path K-tree:
  K1 1.06x/0.87x, K2 0.86x/0.93x, K3 1.12x/1.07x (banked): NO
  COMPOUND at any K (draft cost scales with the model; the 4B stays
  the MTP-friendly shape). Greedy 4/5 on all six reps adjudicated
  benign via the stock-backend K1 control (also 4/5): MTP-vs-plain-
  decode tie-break, not the bridge. Ledger entry + summary.json +
  adjudicator in turing_lab/results/mtp-champion-sweep/ (lane
  8b26330508). Side observation for next window: stock K1 speedups
  (1.19/1.14) exceeded champion K1's in a single rep.
- #inner-loop IN PROGRESS: ncu profile of the champion-shape walk
  kernel names the wall: L1/smem 84 pct, 11.2 of 32 lanes active,
  30 pct barrier stalls, occupancy smem-limited at 2 CTAs/SM, all
  traced to the shared-memory sm_o accumulation and lane-0-serialized
  softmax updates. Surgery: register-resident accumulators,
  warp-distributed pairs, one staging sync pair per token. Gates:
  standalone compile clean, oracle 36/36 at PPS2 (rel 2.54e-04, the
  pre-surgery class); committed-run A/B vs the standing 34.3/34.4/213.5
  in flight.
- #inner-loop DONE (receipted). Surgery: register-resident accumulators,
  warp-distributed pairs, warp-uniform softmax, two staging syncs per
  token, 1 KB smem. Gates: compile clean, oracle 36/36 at PPS2, then
  committed-run medians-of-3: ctx512 41.3 (band 9.9), ctx2048 41.2
  (band 0.0), short 195.5 (-8.4 pct, recorded cost). NEW STANDING
  CHAMPION 41.3/41.2/195.5, floor 0.75, 0.82 of the TRITON baseline.
  Mechanism archived: results/inner-loop-surgery/PROFILE-DELTA.md
  (lanes 11.2->32, smem limit 2->32 CTAs, wall moved to DRAM). Lane
  92ea74c791 + 53604dc957.
- #int8-probe DONE (receipted). Autoround W8A8 quantization of the 4B
  fixture succeeds (248/347 layers, g128, 3167 s); the engine load
  fails in the lane's loader (MergedColumnParallelLinear has no
  orig_layer: the AutoRound int8 method does not cover the model's
  merged parallel linears). An integration gap, not silicon: the
  quantized fixture is banked on disk for a window that closes the
  loader gap. Probe infra notes recorded (backend must be pinned on
  sm_75; load attempts need a real script with a __main__ guard).
  Lane results/int8-probe/VERDICT.md, 5f069cfedb.
- #close-out EXECUTED (2026-09-13). Records current; runbook status
  set; clocks reset; llama-server restored and health-checked
  ({"status":"ok"}); lane pushed through bbd7d75702 and the vllm lane
  pin moved to it. Residual, by design: the weco-skill pin stays at
  707e132 (its fix commit 90f531c awaits the operator's push to
  connollydavid: an unpushed sha is never pinned), so the weco-skill
  DRIFT and the verify recheck it re-opens remain visible until that
  push lands. The 14 remap tells surfaced by this session's records
  are dispositioned in .host-lint-allow as measured quantities and
  test labels.
