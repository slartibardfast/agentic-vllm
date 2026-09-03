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
2. **hdim 128 tile path**: FA2 2.8.3's own sub-80 d=128 kernel is wrong
   (non-causal err 0.38, causal NaN); port the ssiu fork's kernel-traits
   fix (SmemCopyAtomQ, 16 rows/warp) — same mainloop files as item 1.
3. **FlashInfer sm_75 validation** (v0.6.18; upstream breakage #3620
   and its fix #3621 to check).
4. **vLLM engine run on both cards** through the bridge-backed
   backends; tokens/s committed.
5. **Upstream checker** (fresh checkout -> quilt -> oracle -> green).

## Acceptance

1. flash-attn and FlashInfer releases compile for sm_75 via the quilt
   and pass their oracle suites.
2. The vLLM engine runs on TU102 through them; tokens/s and
   context-length scaling committed against the prior engine attention.
3. Upstream tracking in one command: fresh checkout -> quilt -> green.
4. Every quilt patch is a primitive substitution.
