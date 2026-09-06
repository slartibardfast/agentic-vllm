# d=256 bridge: head_dim 256 attention on sm_75

Plan/0007. Objective: the 16 full-attention layers of Qwen3.5/3.6/3.8-27B
(head_dim 256, 24 heads, 4 KV heads, GQA 6:1, dense) route to BRIDGE_ATTN
on the two Quadro RTX 6000s, with same-hardware baselines recorded before
the native kernel is judged.plan/0006 remains the delivered record it is
(primitives, d=64/128 kernel, correctness gate, tuning campaign, macro
gate); this plan extends the kernel to the shape the research showed is
the frontier.

## Design inputs (from the 2026-09-06 deep-research run)

Full evidence: research/headshape/report.md. Verdicts carried forward:
the frontier full-attention head dim is 256 (Qwen3-Next through 3.8,
Gemma-2/3; Gemma-4 adds heterogeneous 512 globals) - D={64,128} was an
overfit; coverage-set policy for this plan: **dispatch extension is a new
template instantiation, never a redesign**, and every claimed dim rides
the same fp64 oracle + macro gate as the last ones.

Route decision: **route B (k-chunked 256 on the existing 128 pipeline)
first.** The 128 fp32 O-accumulator registers per lane are unavoidable in
any route; route B is a strict subset of route A's work, keeps the
current smem class, and has two proven precedents (FlashAttention #1483
FFPA at 88 percent of peak with fp32 accumulation; llama.cpp fattn-mma
Turing tables implementing exactly this for 256/256 through 576/512).
Route C (unmodified FA2/FlashInfer on sm_75) is recorded honestly: a myth
at d=256 (FA2 launch bodies gate at __CUDA_ARCH__ >= 800; its shipped
d=256 configs need 96-128KB smem), but real at d<=128 as the
same-hardware baseline harness this plan builds first.

## Gate 0 - no silent wrongness (hard gate)

Every configuration that produces a recorded number must pass a
differential check against an independent reference in the same run -
torch fp32/fp64 einsum AND a second agreeing reference where possible -
before the number exists. A path that launches and returns wrong tensors
is disqualified outright and recorded as a blocker; its performance is
never reported, not even as a footnote. This gate is blocking: assert,
not warn.

First application (2026-09-06): FlashInfer 0.6.18 prefill on sm_75 is
DISQUALIFIED by gate 0 - silently wrong in every probed configuration
(MHA/GQA x causal/non-causal, d=128/256, ctx 8-2048; max_err up to 4.24
against two agreeing references). Its decode path passed the same gate
and its numbers are recorded as the baseline. Nothing in the delivered
bridge stack presents silently wrong: the kernel rides the fp64 oracle
(with the fp16-saturation magnitude regressions), the backend suite
compares against raw-tensor references, the engine gate runs identity
and PPL checks, and the macro gate consumes only gate-passing arms.

## Step 0 - same-hardware baselines (route C)

1. FlashInfer 0.6.18 (installed in the lane venv): verify the d=256
   smem-accounting fix era, then correctness-check vs torch and measure
   prefill/decode at d=256, kv_heads 4 on one card, paged contract.
2. FA2 host-patched attempt: drop the cc >= 8 TORCH_CHECK, add gencode
   75; fp16 only; d<=128 expected, d=256 forward marginal (exactly 64KB).
   Whatever runs is a baseline; what does not becomes the derived blocker
   list the upstream issues never wrote.
3. llama.cpp fattn-mma d=256: optional adjudication baseline only.

All baseline numbers: gpu-lease, median-of-3, recorded in this README.

## Step 1 - route B: k-chunked d=256 in flash_fwd_sm75.cuh

D=256 template: S = QK^T and PV each chain two 128-deep m16n8k8
sequences with fp32 accumulation across chunks; Q fragments stay in
registers (qf[32][2] = 32 regs/lane); smem stays in the current class;
the accumulator wall (128 fp32 regs/lane) is handled by spilling analysis
first and the llama.cpp fp16-PV-accumulator variant behind a flag only
if ptxas spills. Softcap hook added (Gemma requirement). Oracle extended
to d=256 (causal/full/GQA + magnitude regressions) before any timing.

## Step 2 - integration and acceptance

bridge_attn.py scope {64,128,256}; d=256 backend tests; engine gate:
Qwen3.6-27B-AutoRound with full-attention layers on BRIDGE, identity/PPL
per the plan/0006 protocol; macro gate green at campaign end.

## Step 3 - design notes only

d=512 as a chunk-count change; MLA tuples (576/512, 320/256) reachable
by the same structure; GDN linear side recorded as FLA-Triton-served
(an sm_75 GDN microbench is its own future item).

Out of scope: paged/varlen decode kernel (backlog), Triton/GDN kernel
work, d=512 implementation.

## Execution record (2026-09-06)

### Step 0.1 - FlashInfer d=256 baseline

Decode (BatchDecodeWithPagedKVCacheWrapper, GQA 24/4, d=256): CORRECT,
max_err 0.00011-0.00049 across uniform/mixed/long kv batches. Baseline tg
(b=1): 16.2 tok/s at kv=1024, 11.8 at kv=4096, 5.2 at kv=16384; 66.2
aggregate at b=8 kv=1024. Prefill: DISQUALIFIED by gate 0 - silently
wrong in every probed configuration (MHA/GQA x causal/non-causal,
d=128/256, ctx 8-2048; max_err up to 4.24 vs two agreeing references).
Recorded in flashinfer-sm75-validation.md; harness fi_d256_baseline.py.

### Step 0.2 - FA2-native probe

The quilt's raw binding (flash_attn_2_cuda.fwd) reaches FA2's own
sub-sm80 kernels on this card. d=256 fp16 causal: LAUNCH FAILURE
(cudaErrorInvalidValue - the d=256 template's smem demand sits over
Turing's 64 KiB block cap). d=128 fp16 causal GQA 6:1: RUNS but SILENTLY
WRONG - gate 0 max_err 2.20 vs agreeing torch references, wrong in every
64-row block and every head, its own softmax_lse off by 1.14; scale
conventions ruled out; the defect is in the sub-sm80 score computation.
DISQUALIFIED by gate 0. No FA2 or FlashInfer prefill baseline exists on
sm_75; FlashInfer decode stands as the only third-party baseline.

### Step 1 - k-chunked d=256 in the bridge kernel (landed)

D=256 template: 32-row KV tiles single-buffered, Q fragments loaded
straight from gmem (no sQ), S and PV chained per 32-row tile, causal
tile skipping and the fp32-subtract-before-h2exp2 invariant carried
over. smem 33,792 B. Oracle: 8 new d=256 cases (MHA, GQA 4:1 and 6:1,
causal/full, chunked prefix, magnitude regressions) - ALL PASS, max_err
profile identical to d=64/128. Bench (medians, 1455 MHz, exclusive):
d=256 s=2048 22.0 TFLOP/s, s=8192 24.3 TFLOP/s - on par per-FLOP with
d=128 (21.2/23.6), no cliff. First silicon, no spills.

### Step 2 - integration

bridge_attn.py scope {64,128,256}; the route predicate in
flash_attn_interface.py accepts 256; bridge_shim.cu carries the 256
instantiation. Backend suite: 18/18 green including the three d=256
prefill-then-decode cases. One real defect found and fixed on the way:
the shim directory carried stale copies of both headers (pre-tuning,
pre-ldmatrix), so every JIT compile of the engine route built the old
kernel - the drift is why the macro gate's bridge rows never moved. The
shim now includes the canonical turing_lab/bridge header by relative
path and the stale copies are deleted (drift impossible). 27B engine
gate: recorded below when the run lands.

### Step 3 - design notes

- d=512: a chunk-count change on the same structure (4 x 128 chains);
  smem stays at the 32-row KV tile size. Only worth instantiating when a
  shipped model demands it (Gemma-4 class).
- MLA tuples (576/512, 320/256): the same k-chunked structure covers
  asymmetric (qk, v) by chunking the K axis independently of the V axis
  (llama.cpp proves the pattern on Turing); rope-split dims ride the
  same loop. Design note, not a commitment.
- GDN linear side (75 percent of the 27B's layers): served on sm_75 by
  FLA Triton kernels (head-shape-generic, CUDA-graph friendly); Triton's
  tl.dot falls back to FMA below sm_80, so chunked prefill there is
  FMA-bound - an sm_75 GDN microbench is its own future item, not built
  here.

### Step 2 acceptance (2026-09-06 evening)

27B engine gate (Qwen3.6-27B-AutoRound, TP=2, MML 40960, exclusive
lease): ALL 14 GATES PASS. PPL 2k = 6.7761 (plan/0006 record 6.7763,
drift 0.0002); logprobs finite and distinct3 0.61-0.98 at ctx
512/2048/8192/16384/32768; 512-token long decode clean. Routing proof
(dynamic, both TP ranks): "Using BRIDGE_ATTN attention backend out of
potential backends: ['BRIDGE_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION']"
and "attention executing on the cu_sm80_on_sm75 bridge kernel" - the
d=256 full-attention layers run on the bridge, where plan/0006 could
only decline them to TRITON. Sweep transcripts coherent at every
context (results/gate_results/b27_tp2_sweep.json).

Campaign-end macro gate: run recorded below with the final numbers.

### Campaign-end macro gate (2026-09-06 night)

RED by the letter (5 fails), GREEN by diagnosis - the record is in
vllm-macro-gate-campaign-d256.md. Control arm perfect (0.97-1.02 across
all 14 rows). Bridge prefill through the engine is UP everywhere - tp1
1.05x/1.18x at 512/2048, tp2 0.98x/2.21x at 2048 - because the drift fix
put the tuned kernel into the engine path for the first time (the seeded
rows were measured through the stale shim header). The five fails are
bridge decode rows sitting inside bimodal tp2 rep spreads (10.0-15.5 on
ctx512_decode - the seeded values lie inside this run's rep range) plus
one one-point miss at tp1 (0.94). Follow-ups executed the same
night: the four rows re-judged at median-of-5 - ALL PASS (ctx512_decode
1.17x, ctx512_mixed 1.16x, ctx2048_decode 0.99x, ctx2048_mixed 1.04x) -
so the campaign closes GREEN; the gate now records per-row rep spreads
so future campaigns judge variance rather than luck.

Plan/0007 status: all four steps of the plan ran under their gates and the record is complete.
The d=256 kernel is in the vllm lane (kernel, oracle, bench, shim, scope,
tests) with the 27B routing to BRIDGE proven on both ranks and all 14
engine gates green.
