# Champion bandwidth: the DRAM-bound kernel driven toward the baseline

The re-planned queue for the window that follows the plateau session
(plan/0008, drained; its execution record holds the close state).
The plateau closed with the register-accumulator surgery as the
standing champion and the kernel's wall moved to the DRAM path, which
is where attention belongs: the named successor levers are
bandwidth-level, not control flow. This milestone drives those
levers and closes the residuals the plateau session recorded as
costs or open questions on its way out.

The exit contract is the doctrine's D1
(`research/longrange-agentic/DOCTRINE.md`): drain the build sequence
in dependency order; close the window at queue exhaustion or on the
operator's word, never at a phase boundary. GPU blocks are the spine;
the CPU task fills the waits inside them. Standing rules carried from
the campaign: no external-service auth flow is ever initiated (weco
stays local-only), the GPU pair is serialized, and llama-server stops
and starts only through the preflight and close-out pattern on the
operator's window grant.

Out of scope, named so the exhaustion test stays honest: the
register-dequant plus repack W4A16 surgery (days-grade; its own
future milestone), the operator-side merges (the weco-cli pull
request, the host-lint pronoun-role issue), and weco-skill lane
re-authoring.

Standing numbers this queue works against live in the records, not
in this file: the champion line and its baseline ratio (the
inner-loop entry of `turing_lab/results/ab-champion-ledger.md`), the
mechanism delta and the named levers
(`turing_lab/results/inner-loop-surgery/PROFILE-DELTA.md`), the
committed-run protocol
(`turing_lab/results/committed-both-card-noneager/COMMITTED-BRIDGE-SPLIT.md`),
and the int8 verdict (`turing_lab/results/int8-probe/VERDICT.md`);
paths relative to the `sm75-marlin` worktree.

## Build sequence

### Author the INT8 loader bridge {#int8-loader}

The int8 probe's verdict: quantization succeeds, the engine load
fails in the lane's loader because the AutoRound int8 method does not
cover the model's merged parallel linears. Patch the load path
(qwen3_5) so those linears resolve: authored against the banked W8A8
fixture's checkpoint layout, CPU-only, no GPU required, py-compile
checked. The load itself is measured by the headroom task below.

- depends: (none)
- verify: cd software/vllm/sm75-marlin && .venv/bin/python -m py_compile vllm/model_executor/models/qwen3_5.py
- inputs: software/vllm/sm75-marlin/vllm/model_executor/models/qwen3_5.py

### Preflight the window {#preflight}

Stop llama-server through its systemd unit (it runs as the `llm`
service user), lock both cards, and run the six-check battery: GPUs
free, clocks locked, CUDA_HOME carried inline, the flashinfer
version-check bypass, the bridge d256 kernel smoke on both model
shapes, fixtures present. All checks must pass before anything else
in this sequence starts; the battery encodes every environment
failure a previous window paid for.

- depends: (none)
- verify: cd software/vllm/sm75-marlin && bash turing_lab/thirdparty/preflight_window.sh
- inputs: software/vllm/sm75-marlin/turing_lab/thirdparty/preflight_window.sh

### Recover the short-decode row {#short-row}

The champion surgery recorded a real cost: the fixed per-token
staging and per-pair broadcast overhead dominates at tiny contexts,
and the short row sits below its pre-surgery class. Recover it by a
single-row fast path or a trimmed staging, whichever the profile
names. Gates: the oracle battery, then the committed A/B medians
against the standing champion; the short row must recover toward its
pre-surgery class while the mid and long rows hold their recorded
bands. A falsified attempt is recorded with its mechanism and the
queue moves on.

- depends: #preflight
- verify: cd software/vllm/sm75-marlin && SPLIT=1 CUDA_HOME=/opt/cuda .venv/bin/python ../../../.weco/c2-paged-decode/oracle.py
- inputs: software/vllm/sm75-marlin/turing_lab/bridge/bridge_paged_decode_split.cu, software/vllm/sm75-marlin/vllm/v1/attention/backends/bridge_attn.py

### Stage K and V through half2 {#half2-staging}

The first named bandwidth lever (PROFILE-DELTA): the kernel is
DRAM-bound, so the win, if any, is in how bytes move. Profile first
to confirm the wall stands, then restage the K and V tile reads
through shared memory as half2 pairs arranged against the two-way
bank split, cutting the load instruction count on the hot path. One
surgery through the full gate train: the oracle battery, gate-0
triangulation, committed A/B medians against the incumbent
configuration. The champion moves only through the ledger gates.

- depends: #short-row
- verify: cd software/vllm/sm75-marlin && SPLIT=1 CUDA_HOME=/opt/cuda .venv/bin/python ../../../.weco/c2-paged-decode/oracle.py
- inputs: software/vllm/sm75-marlin/turing_lab/bridge/bridge_paged_decode_split.cu

### Pipeline the KV read {#kv-pipelining}

The second named bandwidth lever: overlap the next page's loads with
the current page's compute through explicit double buffering (Turing
has no async copy; the prefetch is registers and unrolled loads).
Same discipline: profile first, one surgery, the full gate train, the
ledger decides.

- depends: #half2-staging
- verify: cd software/vllm/sm75-marlin && SPLIT=1 CUDA_HOME=/opt/cuda .venv/bin/python ../../../.weco/c2-paged-decode/oracle.py
- inputs: software/vllm/sm75-marlin/turing_lab/bridge/bridge_paged_decode_split.cu

### Settle the K1 inversion {#k1-inversion}

The champion-path K-tree sweep closed with a side observation held
at single-rep weight: the stock backend's K1 speedups exceeded the
champion path's, the inverse of the banked deepest-K ordering.
Replicate at the standing protocol (fresh restarts, both arms,
medians, labels asserted) and adjudicate: a real inversion or
single-rep noise. Record the answer in the sweep tree beside the
K-tree.

- depends: #preflight
- verify: attested operator

### Measure MTP K2 on the 4B champion path {#mtp-4b}

The open family question from the K-tree verdict: the lighter
same-geometry fixture compounds under MTP on the stock path, and the
champion path on that fixture is unmeasured. Measure it: the champion
configuration on the fixture, draft-token sweep at the compounding
depth, fresh restarts, greedy canaries, labels asserted, ledger entry
with the intent fields. Either result is terminal: compounding
transfers to the champion path, or the fixture's deployment shape
stays on its measured stock-path numbers.

- depends: #preflight
- verify: attested operator

### Load W8A8 and measure headroom {#int8-headroom}

With the loader bridge authored, load the banked W8A8 fixture and
answer the probe's deferred question: what a narrower activation path
buys on this silicon versus the W4A16 incumbent. Identical rows, the
committed-run protocol: medians across fresh restarts, labels
asserted, zero preemptions. A blocked load is a recorded terminal
state with its traceback, not a silent skip.

- depends: #int8-loader, #preflight
- verify: attested operator

### Close out the window {#close-out}

Ledger and records current, the lane runbook's status line updated,
clocks reset, llama-server back up through its unit and
health-checked, lane and host commits pushed, the pin moved to the
closed state, every task in this sequence carrying its receipt.

- depends: #short-row, #half2-staging, #kv-pipelining, #k1-inversion, #mtp-4b, #int8-headroom
- verify: attested operator

## Execution record (2026-09-13, in flight)

- #preflight DONE (receipted). Battery all-OK at window open
  (CUDA_HOME inline on the second pass); llama-server stopped via
  its unit, clocks locked.
- #int8-loader DONE (receipted), with its premise corrected by
  measurement: the banked W8A8 fixture is NOT int8. Its index holds
  only orig_layer.weight tensors, BF16 in the safetensors headers,
  with no scales anywhere; autoround 0.15.0's auto_round format
  exports the triton-act wrapper IR when activation quant is on, and
  the tuning never left the process. The engine-side orig_layer
  AttributeError was wrapper naming, not a qwen3_5 loader gap, so no
  model-file patch was authored (the task's py_compile verify passes
  unchanged). The fix is export-side:
  results/int8-probe/requantize_llmcompressor.sh re-runs the
  fixture-grade recipe with the llm_compressor format, which packs
  int8 weights, group scales and activation scales as
  compressed-tensors the engine loads natively. Lane 578d06e717;
  the re-export itself is GPU work queued for #int8-headroom.
- #short-row DONE (receipted), terminal: the recorded cost STANDS. The
  falsification chain: the champion reproduces same-day (baseline
  reRun 195.5/41.6/40.2); a same-day A/B with only the walk kernel
  swapped to the pre-surgery version returns short 216.1 (the
  pre-surgery class) while mid/long fall to 35.8/35.1 - the kernel
  commit is the cause, both directions; yet the post-surgery kernel
  is 3x FASTER isolated at the short shape, so the regression is an
  engine-context inversion (~3.6 ms per 41 ms step, larger than all
  16 attention launches combined). The named staging trim (half2,
  bank-conflict-free, minus 16 pct isolated) produced a committed
  NO_DIFF - the engine's sensitivity floor sits at multi-ms kernel
  deltas (the register surgery: ~16 ms/step = +20 pct), so no
  kernel-internal surgery can close this row. Mechanism open
  (chrome-trace diff is the named follow-up); evidence in
  results/short-row-recovery/. Lane c4909f6bf3.
- #half2-staging DONE (receipted), falsified as an engine mover:
  oracle 36/36 at PPS2, gate-0 8/8 zero-diff triangulation,
  committed A/B 193.1/41.9/39.8 vs same-day baseline
  195.5/41.6/40.2 - every row inside overlapping bands. Isolated
  wins recorded (minus 16 pct short, minus 9 pct long shapes); the
  patch is preserved at results/short-row-recovery/half2-kernel.patch
  and the working tree reverted to the ledger champion. The ledger
  entry carries the mechanism (sensitivity floor). Lane c4909f6bf3.
- #mtp-4b DONE (receipted): the family question closed NEGATIVE. The
  4B fixture on the champion configuration (decode paged-split PPS2,
  verify/prefill gather, graphs, TP2), K-tree medians over 3 fresh
  restarts: K1 0.53/0.74, K2 0.44/0.70, K3 0.59/0.73, K4 0.91/0.84
  (ctx512/ctx2048) - every depth negative by a wide margin, a stronger penalty
  than the 27B's, greedy 3/3 lossless on every rep. The stock path's
  compounding does NOT transfer; recorded hypothesis: the champion
  decode arm is much faster per step on this fixture, so the
  draft-and-verify overhead prices far higher against it. The 4B
  deployment shape stays on its measured stock-path numbers. Summary
  and ledger entry: results/mtp-4b-champion/summary.json.
- #k1-inversion DONE (receipted): REAL. The stock-backend K1 control
  at the standing protocol (3 fresh restarts): speedup medians
  1.32/1.13 (vals 1.27-1.42/1.08-1.21) vs the champion path's banked
  1.06/0.87 - the single-rep observation replicates, greedy 3/3
  throughout. Recorded mechanism (hypothesis, consistent with the 4B
  K-tree): MTP buys back step overhead the champion path no longer
  pays while the draft-and-verify machinery prices higher against its
  faster decode. Spec decode favors the STOCK path at every measured
  scale; the champion deployment stays no-MTP. Summary:
  results/k1-inversion/summary.json.
- #kv-pipelining DONE (receipted), falsified NEGATIVE: register
  prefetch + double-buffered staging + one sync per token. Oracle
  36/36 at PPS2; gate-0 6/8 adjudicated benign by the dual-arm
  discriminator (mean worst-row diff 0.004, max 0.0625, the fp16
  tie-break class; GATE0-ADJUDICATION.md); committed A/B
  186.7/36.8/39.5 vs the same-day baseline 195.5/41.6/40.2 - ctx512
  minus 11.5 pct at band 0.3, decisive. Tree reverted, patch
  preserved (pipelining-kernel.patch). With half2 NO_DIFF and this
  NEGATIVE, the PROFILE-DELTA lever list is exhausted: the champion
  41.3/41.2/195.5 class stands as this kernel generation's
  engine-measured optimum. Lane fe14558eeb.
- #int8-headroom DONE (receipted): NO HEADROOM, lane closed. The
  re-export (llm_compressor, per-channel sym dynamic W8A8 after the
  format check rejected grouped g128) produced a real
  compressed-tensors checkpoint (int8 weights, fp16 channel scales,
  dynamic token activations) and the engine LOADS IT NATIVELY on the
  champion configuration - the 0008 loader gap closed end to end, no
  model patch. Identical rows, 3 fresh restarts per arm: short
  426.9 vs 538.1 (ratio 0.79), ctx512 80.2 vs 94.0 (0.85), ctx2048
  75.9 vs 89.2 (0.85). Decode is weight-bandwidth-bound; W8 doubles
  the weight bytes and the int8 tensor rate cannot buy them back at
  decode shapes. The formats doctrine's last unexplored lane on this
  silicon is measured and closed; W4A16 stands. Lane a4310546ba.
