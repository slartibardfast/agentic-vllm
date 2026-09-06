# cu_sm80_on_sm75: the primitives bridge for FlashAttention and FlashInfer

Plan/0006. flash-attn and FlashInfer — the kernel libraries modern vLLM
is built on — gate to sm_80+. This plan makes them run on sm_75 at state
of the art through a **generic primitives bridge**: raw-PTX sm_75
implementations of the sm_80+ device primitives, plus a mechanical,
re-appliable patchset per upstream release. Adoption is a checker run,
not a merge; no behavioral divergence.

## Substitution map

| sm_80+ primitive | sm_75 bridge |
|---|---|
| `cp.async.ca/cg`, commit-groups, multi-stage pipelines | LDG -> register -> STS behind an N-stage register-staging template |
| `mma.sync.m16n8k16` FP16->FP32 | 2x `mma.sync.m16n8k8` (k-order equivalence: arXiv:2208.11174) |
| `mma.sync.m16n8k32` INT8 | Turing IMMA reshape (4x `m8n8k16`) — INT8 TC at 2x FP16, D8-measured |
| bf16 arithmetic/MMA | fp16 recast (8 mantissa bits subset 10; overflow check at boundaries) |
| `ldmatrix` / `movmatrix` | native on sm_75 — passthrough |
| `cuda::barrier` / `memcpy_async` | bridge-backed equivalents |
| TMA / WGMMA / clusters (sm_90, FA3-only) | out of scope |

## The bridge set

Defined at the PTX/ISA primitive level, per source architecture, not per
consumer:

| Bridge | Unlocks | Emulated | Effort |
|---|---|---|---|
| cu_sm80_on_sm75 | flash-attn 2, FlashInfer sm_80 paths, Marlin, CUTLASS sm_80 kernels, most Triton output | cp.async, m16n8k16, INT8 reshape, bf16, mbarrier | core |
| cu_sm89_on_sm75 | FP8 kernels | FP8 MMA as cvt->FP16 MMA; e4m3/e5m2 cvt | thin delta |
| cu_sm90_on_sm75 | FA3-class, DeepGEMM-class, Machete, FlashInfer Hopper paths | WGMMA->mma.sync with ldmatrix (native), TMA->cp.async/ld+st, clusters->flat | large |
| cu_sm100/120_on_sm75 | NVFP4/MXFP4/FP6-FP8 checkpoints | tcgen05 (functional); real value is format decode: FP4 LUT + block scales | medium |

## Design rules

1. Conformance is against the PTX ISA specification, not against any
   consumer.
2. Deliverables are consumer-agnostic: the device header library, a
   PTX->PTX rewrite pass for the gated instruction families, and the
   per-primitive conformance suite.
3. vLLM is consumer #1 behind a thin dispatch adapter; upstream internal
   changes cannot touch the bridge.
4. Compatibility is proven multi-consumer: flash-attn, FlashInfer, a
   Triton-emitted kernel, and a dequant-GEMM on the same bridge.
5. Limits are recorded per bridge: no cp.async register bypass, no
   smem-direct WGMMA operands, no tensor memory; emulation runs at FP16
   tensor rate.

## Red lines (build-failing conformance)

The bridge is hand-written inline-asm PTX in .cuh headers; instruction
selection, cache hints, store ordering, and pipeline schedules are fixed
by the header. Conformance tests fail the build on regression:

1. cp.async semantics are 1:1 by construction: the sm_75 scoreboard
   enforces per-thread LDG->STS ordering, so commit/wait lower to
   scheduling fences at zero runtime cost. The irreducible delta is
   registers in flight; the occupancy gate prices it.
2. Staged-copy bandwidth >= 90% of plain streaming on identical volumes.
3. MMA adapters: <= +1 ALU op per fragment; issue slots <= 2x native k16.
4. Streaming loads carry .cs/.nc hints; verified by SASS inspection.
5. Register caps from the occupancy rule; verified by the SASS gate.

## Execution path (the bridge is the kernel)

The bridge header is the sm_75 execution path for flash-attn and
FlashInfer — the in-tree dormant sub-80 fallback of flash-attn is the
thing this plan replaces, not the vehicle. Quilts route the libraries'
Turing dispatch at bridge-native kernels; upstream code supplies only
shapes, parameters, and the python API.

`turing_lab/bridge/flash_fwd_sm75.cuh`: bridge-native attention forward
(fp16, d=64/128, causal, GQA) on the bridge's streaming loads, N-stage
staging, MMA adapter, and online softmax — oracle-attested against the
fp64 reference, red-line benched, then routed by the quilt as the Turing
backend of flash-attn/FlashInfer/vLLM.

## Auto-research over the bridge

The .cuh primitives are generated artifacts with a searchable schedule
space (stage count, vector width, swizzle, issue interleaving, cache
hints, unroll), driven by the plan/0003 machinery:

- Bit-exactness, SASS legality, and the red lines gate — never trade.
- Exhaustive enumeration within budget; TPE beyond it.
- Coverage ledger: every legal candidate is measured (transcript
  committed) or model-pruned (reason committed). Nothing is skipped
  silently.
- The shipped header carries the winning schedule and transcript hash.

## Limits

No cp.async register bypass (the LDG+STS substitute pays registers and
issue slots; priced by the gates). No TMA/WGMMA — FA3-style warp
specialization is out of scope. No bf16 tensor rate — bf16 workloads
run the fp16 recast. Emulation never reaches native FP8/FP4 rates.

## Turing assets

`ldmatrix`/`movmatrix` native; FP16 tensor cores at full rate with FP32
accumulate (attention's accumulate mode; no Ampere-consumer halving);
INT8 tensor cores at 2x FP16. Existing program assets: the k16->2xk8
split, register-staged pipelines, SASS/occupancy gates, fp64 oracles,
the ladder, CUDA-graph validation.

## Status and remaining sequence

Done: bridge header and conformance suite green (mma adapter bit-exact,
staged copy byte-exact at 94.8% of plain streaming, mbarrier, redux,
FP8 exhaustive 256/256, bf16 RNE); **the bridge-native attention
forward is oracle-green** (max_err 0.00005 non-causal, 0.00032 causal
vs the fp64 reference, s=512 d=64 — built entirely on bridge
primitives); flash-attn 2.8.3 quilt v1 builds for sm_75 (hdim 64
oracle-exact via the sub-80 path); ptxas blowup on the 256-wide
splitKV tiles root-caused and worked around. See `research.md`.

Remaining, in dependency order:

1. ~~Fragment mapping~~ DONE — the layoutsolve differential froze the
   m16n8k8 register grid; six kernel bugs fixed in sequence (cvta,
   k-step count, staging bounds, running max, rescale scale, and the
   oracle's own 1-D launch).
2. ~~Forward tuning + d=128/GQA~~ DONE — double-buffered K/V register
   pipeline and half2-vectorized softmax exp (native
   ex2.approx.f16x2); d=64 16.3 TFLOP/s causal. Templated over head
   dim with GQA: the oracle suite is ALL GREEN across d=64/d=128 x
   MHA/GQA-8:1 x causal/full (max_err 0.00006..0.00043). d=128 runs
   single-buffered at 10.6 (a 32-row KV tile would restore double
   buffering); the remaining gap to SDPA is scalar LDS fragment
   loads — ldmatrix (native) is the next lever.
3. **hdim 128 tile path**: FA2 2.8.3's own sub-80 d=128 kernel is wrong
   (non-causal err 0.38, causal NaN); port the ssiu fork's kernel-traits
   fix (SmemCopyAtomQ, 16 rows/warp) — same mainloop files as item 1.
4. ~~FlashInfer sm_75 validation~~ RECORDED — 0.6.18: prefill PASSES
   (max_err 0.00094, causal d=128) and batch decode PASSES multi-request
   (b up to 16, mixed lengths, both kernel variants, max_err <= 0.00072)
   — the earlier multi-request fault claim is RETRACTED: our harness fed
   a non-contract paged-cache shape; with `(num_pages, 2, page_size, h,
   d)` there is no sm_75 breakage to file upstream. Throughput recorded
   in flashinfer-sm75-validation.md.
5. ~~vLLM engine run~~ DONE — V1 engine on TU102: **24.2 tok/s** on
   the 27B int4-AutoRound model (W4A16 Turing backend + V1 attention),
   evidence committed (vllm-engine-run.md); TP=2 both cards 76.7 tok/s
   aggregate short-context (9.9/6.4 end-to-end at ctx 512/2048).
   Correction (2026-09-05): on capability (7,5) the V1 selector
   auto-selects TRITON_ATTN (FLASH_ATTN/FLASHINFER are gated >= (8,0),
   and the FLASH_ATTN backend imports the vendored vllm_flash_attn,
   not the quilt-patched package) — so engine attention was Triton,
   not the bridge. CLOSED same day: `BridgeAttentionBackend` (the
   plan's thin dispatch adapter, design rule 3) auto-selects on sm_75
   and calls the real `flash_attn_func` per layer; quilt v3 adds
   bottom-right causal (q0, chunked prefill with prefix) and the
   causal zero-padding route; profiler evidence
   (`vllm-bridge-routing-trace.json.gz`) shows `bridge_fwd_kernel<128>`
   firing per layer per step inside the engine; full A/B vs TRITON_ATTN
   committed (vllm-engine-run.md) — parity at ctx 2048 single-card,
   behind at short context (python dispatch + paged-KV gather; the
   varlen/paged bridge kernel is backlog). head_dim 256 (the 27B)
   declines to TRITON by scope — d=256 needs the 32-row KV tile +
   Q-in-registers restructuring (backlog).
6. ~~Upstream checker~~ DONE — check_upstream.sh: fresh-tag clone ->
   quilt applies clean -> oracle via the real flash_attn_func
   all-pass -> VERDICT GREEN, one command.

## Correctness gate (2026-09-05)

A numerics-before-timing pause validated the stack end to end
(vllm-correctness-gate.md + gate-logs/ in the fork): bridge vs TRITON
PPL within 0.007% (1.5B d=128 and 0.5B d=64), chunked prefill and
prefix caching bit-identical on the bridge (the q0/bottom-right workout),
the 27B hybrid stack coherent to ctx 32768 with stable 512-token decode,
TP1 vs TP2 greedy identity 3/3. The gate caught one real kernel defect
the oracles could not see: real-activation RAW dot products exceed fp16
range, so packing raw S and m into half before subtracting saturated to
inf and inf - inf = NaN. Fixed by subtracting in fp32 before the half
conversion; regression-locked with in_scale=100 magnitude oracle cases
(fwd_oracle.cu), route-level checks, and the engine gate itself. Quilt
re-verified VERDICT GREEN after the fix. Remaining backlog unchanged:
architectural backend speedup (paged/varlen kernel), d=256 tiles,
ldmatrix loads.

## Acceptance

1. flash-attn and FlashInfer releases compile for sm_75 via the quilt
   and pass their oracle suites.
2. The vLLM engine runs on TU102 through them; tokens/s and
   context-length scaling committed against the prior engine attention.
3. Upstream tracking in one command: fresh checkout -> quilt -> green.
4. Every quilt patch is a primitive substitution.

### weco-tuned forward kernel (2026-09-06)

The first local weco run (observe 1c5a7d79, local mode, ZCode as the
intelligence) took the bridge forward kernel from 26.0 to 59.7 TFLOP/s
(2.30x, sum of s=8192 causal TFLOP/s over d64+d128) in the isolated
.weco workspace: causal tile skip (the kernel computed its whole masked
upper triangle and discarded it), ldmatrix x4/x4.trans B-fragments
(native on sm_75; .trans distribution is exactly the mma B-fragment),
and single-buffered d64 for 2 CTAs/SM. Every step ran under gpu-lease
exclusive with the fp64 oracle gating all timing; the failed
transposed-V branch is logged with numbers in the observe store, and
mma.m16n8k16 was confirmed absent on Turing (the primitives' k16 entry
is a 2x k8 wrapper). Applied to the lane at 69e91020 with the lane
oracle, bench and the 15-test backend suite green; the one red probe
was a missing CUDA_HOME in the shell, not the kernel.

### Macro gate formalized (2026-09-06)

The engine-level A/B is now a gate, not an anecdote: vllm_macro_gate.py
(protocol v2: prefix caching disabled, median of 3, gpu-lease per arm,
control-arm invalidation on drift) runs at the end of every A/B
campaign. First formal run came back RED as designed — the 2026-09-05
single-run baselines failed the triton control check (1.50x/1.51x on
short-decode), so they were re-seeded under the clean protocol at kernel
69e91020; the next campaign is judged against the seeded table in
vllm-engine-run.md (bridge prefill-only 13.4k tok/s at ctx2048 on one
card, 3.1x triton; decode-phase 33-35 vs triton 48-51 — the padded-Q
tile and gather backlog, unchanged).

### Head-shape design inputs (2026-09-06, deep-research run)

The deep-research stack's first full loop (outline -> 10-item deep phase,
every item validated 17/17 fields -> report) answered the head-shape
question. Full record: research/headshape/report.md (raw JSONs beside it).

Verdict on {64,128}: an overfit, now evidenced. The 2025-2026 frontier
full-attention dim is 256 (Qwen3-Next blog: "increase the dimension per
attention head from 128 to 256"; the whole Qwen3.5/3.6/3.8 line, Gemma-2/3;
Gemma-4 adds per-layer heterogeneous 512 globals), against a 2023-2025
generation that was uniformly 128. SageAttention #47's "head dim greater
than 128 is quite rare" (2024-11) is the canonical narrow-set statement,
falsified on its own thread by mid-2026. FA2 instantiates
{32,64,96,128,192,256} for exactly this reason. Qwen3.8-27B is
architecturally identical to Qwen3.6-27B (verified three ways: local
configs, local GGUF metadata, public config.json) - head_dim 256, 24/4
heads GQA 6:1, dense, hybrid 3:1 GDN:full.

Route decision for d=256: **route B, k-chunked 256 on the existing 128
pipeline, first.** The 128 fp32 O-accumulator registers are unavoidable in
any route (output is 256 wide), so B's work is a strict subset of A's; B
keeps our current 51KB smem and adds two staging passes + two barriers per
tile. Precedent: FlashAttention #1474/#1483 (FFPA) does MMA-level head-dim
chunking at 88% of peak (105 TFLOPS on L20, D=512, fp32 accumulation), and
llama.cpp's fattn-mma Turing tables implement exactly this structure for
256/256 through 576/512 - including the two tricks worth stealing: fp16
PV accumulation to halve accumulator registers (GeForce Turing halves the
fp32-accumulate mma rate), and time-multiplexed K/V smem tiles with XOR
swizzle. Route A (32-row KV tiles + Q-in-registers) only if profiling
shows B's staging overhead is material. Route C honesty: "unmodified
FA2/FlashInfer on sm_75" is a myth for d=256 (FA2 launch bodies gate at
__CUDA_ARCH__ >= 800 with a fatal printf; its shipped d=256 configs need
96-128KB smem; FlashInfer d=256 fits in exactly 65536B - zero headroom),
but the route works at d=64/128 as the same-hardware baseline harness this
plan always intended, and FlashInfer's sm75 breakages are host-side smem
accounting bugs, not ISA limits (un-gating needs an sm75 CI lane plus a
one-line vLLM capability revert).

Surroundings recorded for the roadmap: the 75% GDN linear-attention side
of the 27B never touches this kernel (served by FLA Triton on sm_75, but
Triton's tl.dot lost tensor cores below sm80 in Triton 3.3); MLA
(576/512) and asymmetric (qk,v) tuples are reachable by the same k-chunked
structure (llama.cpp proves it on Turing); numerics invariants carry over
(fp32 subtract before h2exp2; d=256 widens |S|; softcap needed for the
Gemma family). Next action: design the k-chunked d=256 variant of
flash_fwd_sm75.cuh, acceptance through the fp64 oracle and the macro gate.
