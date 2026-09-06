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
