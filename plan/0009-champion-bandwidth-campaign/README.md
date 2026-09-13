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
here: the champion line and its baseline ratio in
`turing_lab/results/ab-champion-ledger.md` (the inner-loop entry),
the mechanism delta and the named levers in
`turing_lab/results/inner-loop-surgery/PROFILE-DELTA.md`, the
committed-run protocol in
`turing_lab/results/committed-both-card-noneager/COMMITTED-BRIDGE-SPLIT.md`,
and the int8 verdict in `turing_lab/results/int8-probe/VERDICT.md`
(paths relative to the `sm75-marlin` worktree).

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
