# plan/0006 research: the cu_sm80_on_sm75 bridge

Research phase findings, 2026-08-28. Three tracks: PTX ISA inventory,
consumer surface + prior art, emulation techniques (track 3 appended
when complete). Sources cited inline; empirical results run on this
host's toolchain.

## Track 1 — PTX ISA inventory (documented, PTX ISA 9.3)

### MMA shapes (mma.sync.aligned), per architecture

| Type/shape | PTX | arch | sm_75 |
|---|---|---|---|
| f16 m8n8k4 | 6.4 | sm_70+ | yes |
| f16 m16n8k8 | 6.5 | sm_75+ | **yes** |
| f16 m16n8k16 | 7.0 | sm_80+ | no -> 2x m16n8k8 |
| s8/u8 m8n8k16 | 6.5 | sm_75+ | **yes** |
| s8/u8 m16n8k16, m16n8k32 | 7.0 | sm_80+ | no |
| s4/u4 m8n8k32 | 6.5 | sm_75+ | shape accepted by ptxas (empirical; vector arity open) |
| s4/u4 m16n8k32, m16n8k64 | 7.0 | **sm_80+** (ptxas: "requires .target sm_80 or higher" - empirical) | no |
| b1 m8n8k128 | 7.0 | sm_75+ | yes |
| f64, bf16, tf32 mma | 7.0+ | sm_80+ | no |
| FP8 e4m3/e5m2 mma | 8.4 | sm_89+ | no |
| block-scaled mxf4/nvf4 (sync) | 8.7 | sm_120a/f | no |
| sparse mma.sp | 7.1 | sm_80+ | no |

Empirical s4 note: ptxas for sm_75 rejects m16n8k32/m16n8k64 .s4 with
"requires .target sm_80 or higher" (verbatim), and rejects m8n8k32 .s4
with "argument vector size mismatch" at every tried arity (2-4 regs) —
the canonical arity and `.satfinite` requirement remain an open
micro-question. Program impact: INT4 tensor MMA on sm_75 exists only as
the tiny m8n8k32 with INT activations, so it cannot serve W4A16 (FP16
activations); D8's conclusion stands in substance, now with precise
shape facts and a recorded W4A8-IMMA possibility for later.

### Other ISA facts

- `ldmatrix` (.m8n8 .x1/x2/x4 .b16, optional .trans): PTX 6.5,
  **sm_75+** — native. `movmatrix`: PTX 7.8, **sm_75+** — native.
  `stmatrix`: sm_90+ (absent on 75/80 — FA2-era code does not use it).
- `cp.async` (.ca/.cg, commit_group, wait_group): PTX 7.0, sm_80+.
  Semantics to preserve in emulation: non-blocking; per-thread group
  FIFO; writes visible to the executing thread only after
  wait_group/wait_all; no cross-op ordering absent explicit sync;
  other threads see data only after a further fence/mbarrier;
  zero-fill on short src-size / ignore-src predicate.
- `mbarrier`: sm_80+ (expect_tx/complete_tx parts sm_90+). Emulation
  spec: 64-bit shared state = pending arrivals + phase parity + tx
  count; `atom.shared.add` for arrive (release), acquire spin for
  test_wait; no suspension on sm_75, try_wait hints ignored.
- `redux.sync` (int min/max/and/or/xor): sm_80+ -> emulate via
  shfl.sync butterfly (shfl/vote/match: sm_30+/sm_70+, native).
- bf16 `cvt`: sm_80+/90+ -> emulate via f32 round + truncation; f16
  cvt/arithmetic: native since sm_53+ (`tanh.f16x2`: sm_75 OK).
- `cvt.rn.f16x2.e4m3x2` (FP8): sm_89+ -> byte LUT or integer rebias.
- Cache hints: `.L1::evict_*` is sm_70+ (usable); `.L2::cache_hint`
  rides on createpolicy (sm_80+) — not available.
- min/max f16/f16x2: sm_80+ -> bit-trick emulation (matters for the
  online-softmax max in attention).
- `fence.proxy.async` (TMA/async-proxy visibility): sm_90 concept; the
  sm_75 substitute for cross-thread visibility of staged data is the
  plain `membar.cta`/barrier discipline.
- TMA (sm_90) / tcgen05 (sm_100/120): descriptor-driven bulk copies
  and tensor-memory MMA — functional emulation only (per-thread
  ld/st loops + explicit fences); the perf story is out of scope.

## Track 2 — consumer surface and prior art

### flash-attn 2: feasible-as-engineering, prior art exists

- Official stance: "Ampere, Ada, or Hopper" — sm_75 dropped at build
  level (`setup.py` arch list has no 75); the host gate is a one-line
  `TORCH_CHECK(is_sm8x_min)` repeated 5x in flash_api.cpp.
- **Critical in-tree fact**: `csrc/flash_attn/src/kernel_traits.h`
  retains a compile-time sub-80 path — `Has_cp_async = false` forces
  the `SM75_16x8x8_F32F16F16F32_TN` MMA atom and
  `SM75_U32x4_LDSM_N`/`SM75_U16x8_LDSM_T` (ldmatrix) copy atoms, and
  the fwd mainloop falls back to sync copy when `Has_cp_async` is
  false. **cp.async is the single sm_80-only ISA dependency, and the
  fallback already exists in-tree.**
- Kernels are templated on (kHeadDim, kBlockM, kBlockN, kNWarps) with
  arch handled by `__CUDA_ARCH__` macros; all instantiations are
  `_sm80.cu` files but nothing semantic blocks compute_75 compilation
  once dispatch/TORCH_CHECKs are patched. Constraint: 64 KiB smem on
  Turing vs 99/163 KiB -> smaller tiles.
- **Prior art**: ssiu/flash-attention-turing — a working FA2 Turing
  port (fwd+bwd, hdim 64/128, causal, GQA, varlen; up to 2.19x/1.95x
  faster than PyTorch attention on T4). Upstream issue #1342 is an
  open RFH.

### FlashInfer: feasible-as-engineering, one tracked breakage

- Officially "SM75 (Turing) and later"; the paper claims Turing..sm90a.
- vLLM's backend comment records the state: "FlashInfer supports SM75+,
  but is currently broken on SM75 (Turing): issue #3620 (fix: PR
  #3621). Temporarily raise the floor to SM80 ... revert to (7,5) once
  it does." So the default-dispatch path is one tracked fix away.
- Kernels are hand-written CUDA with JIT-templated generation (not
  CUTLASS-3.x/TMA-based), so sm_75 rides the FA2-style hand-CUDA paths.

### vLLM engine gating (corrects earlier statements)

- **vLLM V1 runs on Turing today via the Triton attention backend**
  (`TritonAttentionBackend.supports_compute_capability: return True`) —
  the earlier claim that current upstream has "no working sm_75 engine"
  is retracted. V0/xformers is gone (deprecated mid-2025, removed in
  v0.10-v0.11), but V1's dispatch priority
  FLASHINFER -> FLASH_ATTN -> TRITON_ATTN -> FLEX_ATTENTION degrades
  gracefully to Triton on sm_75.
- The real gaps are the faster, default-dispatch paths: FlashAttention
  requires >= (8,0) and FlashInfer is temporarily >= (8,0) pending the
  sm_75 fix. Platform dtype handling already excludes bf16 on
  pre-Ampere.
- Consequence for plan/0006: the bridge's value is upgrading Turing
  from "Triton attention fallback" to "FlashAttention/FlashInfer-class"
  — not rescuing a dead engine.

### CUTLASS, llama.cpp

- CUTLASS: full Turing support (mma_sm75.h, default_mma_core_sm75.h,
  memory_sm75.h ldmatrix) — the canonical 2.x-API prior art for
  Turing collectives.
- llama.cpp: works well on Turing; GGML CUDA kernels are
  arch-agnostic dequant+GEMM — established precedent that
  dequant-in-flight kernels port trivially.

### PTX-translation prior art

- ZLUDA (PTX->AMD), GPU Ocelot (academic CUDA binary translation),
  HetGPU (arXiv 2506.15993), CASS (arXiv 2505.16968): the mechanism
  (PTX-level rewriting) is proven cross-vendor.
- ptxas will not downgrade cp.async to sm_75 (rejected at ptxas), so
  substitution must happen at PTX-before-ptxas or source level.
- Note: NVIDIA's 2024 EULA addition restricting translation layers —
  same-vendor down-leveling is a different case, recorded for legal
  review.
- The genuinely novel work: the PTX-level cp.async substitution layer
  with reconstructed commit/wait semantics — no prior art found.

## Verdicts

| Surface | Verdict |
|---|---|
| FA2 on sm_75 | feasible-as-engineering (in-tree fallback + ssiu prior art) |
| FlashInfer on sm_75 | feasible-as-engineering (nominally shipped; tracked breakage #3620) |
| vLLM V1 on sm_75 | works today via Triton attention (correction to earlier statement) |
| CUTLASS sm_75 | supported (2.x API) |
| llama.cpp sm_75 | works |
| PTX-level cp.async bridge | open-research — the novel deliverable |

## Refinements to the plan/0006 stage plan

1. The bridge's first consumer is not FA2-from-scratch: start from the
   in-tree sub-80 path of flash-attn (kernel_traits.h) and the ssiu
   port, upstream the bridge header beneath them.
2. FlashInfer: track and pull PR #3621's sm_75 fix into our stack; the
   bridge's cp.async layer covers what it needs.
3. vLLM: lift FlashInfer/FlashAttention capability floors via the
   bridge-backed backends; keep Triton attention as the portable
   fallback.
4. The cp.async bridge (commit/wait semantics + zero-fill + visibility)
   is the novel research deliverable and gets the red-line conformance
   treatment (bandwidth/occupancy floors) from the plan.

## Track 3 — emulation techniques (implementation recipes)

### cp.async staging

NVIDIA's own degradation template is `cutlass/arch/memory_sm80.h`:
`CUDA_CP_ASYNC_ACTIVATED` is set only for `__CUDA_ARCH__ >= 800`; below
that the `cp_async`/`cp_async_zfill` intrinsics degrade to a typed
load+store of `Array<uint8_t, N>` — the exact bridge pattern, from
NVIDIA itself. The canonical pre-Ampere pipeline is
`cutlass/gemm/threadblock/mma_pipelined.h` (Volta AND Turing): 2-stage
shared ring, per-K-slice LDG.128 into threadblock fragment registers ->
STS -> bare `__syncthreads()`, next global tile prefetched into the
fragment registers while math runs. Register pressure caps practical
depth at 2 stages on Turing (Ampere uses 3-4) — our ladder may search
2-3 with the occupancy gate pruning beyond. Forum/field finding:
`cuda::pipeline` gave no speedup on Turing, but warp specialization
did.

### mbarrier emulation

CCCL's `cuda::barrier` already dispatches pre-sm80: warp-coalesced
arrivals via `__activemask()` + `__match_any_sync`, one elected lane
arrives, the token is `shfl_sync`ed back, and waiting is a shared-memory
phase-poll spin with backoff. Simpler primitives: `__syncthreads_count`,
named `bar.sync id, count`, `bar.arrive`/`bar.red` (what CUTLASS
pre-Hopper pipelines use). Memory-ordering note: `fence.proxy.async` has
no Turing equivalent and needs none — it exists only for async-proxy
(cp.async) writes; on Turing all writes are generic-proxy, so
`bar.sync`/`__threadfence_block()` suffices.

### 2x m16n8k8 MMA: the accumulate-order question dissolves

Ampere microbenchmarking (arXiv:2208.11174) shows m16n8k16 hardware
processes K in 8-deep slices with sequential fp32 accumulation — so
2x m16n8k8 reproduces the same k0-7 -> k8-15 order; it is within a
ulp-tree reorder of any K-split. Fragment layouts are identical for the
first k=8; the second loads B's alternate k-half into the same fragment
registers and chains mma(D, A1, B1, D). flash-attn 1.x ran on Turing
this way; prior art: xformers CUTLASS FMHA.

### INT8

Ampere m16n8k32's A-fragment (4 x b32) decomposes as two row-groups x
two k-halves -> 4x Turing m8n8k16 with a near-identity register
reshuffle; prior art jundaf2/CUDA-INT8-GEMM composes m16n16k16 from 4x
m8n8k16 on Turing. Layout caveat: CUTLASS Turing int8 requires
ColumnMajorInterleaved<32> + TensorOpMultiplicandCrosswise smem layouts.

### bf16

bf16->fp32 is one <<16 shift (free); fp32->bf16 RNE on sm_75 is a 4-6
ALU-op bit trick (round bit, sticky, mantissa LSB, then += 1<<16) —
sm_80+ gets a single cvt. Dominant practice: recast weights to fp16.

### FP8 without FP8 units — the exact recipe for plan/0005

vLLM's FP8-Marlin (`fp8_marlin.cu`, credited to FasterTransformer's
interleaved converters) is a LUT-free bit trick: per 2 elements,
~4 int ops + 1 packed HMUL2 (RIGHT_SHIFT = 1, MASK = 0x7F007F00, sign
passthrough, bias fix by HMUL2 with 256). Alternatives: 256-entry fp16
LUT (512 B, smem-friendly) or CUDA's software `__nv_cvt_fp8_to_halfraw`.
This is the ready-made dequant recipe for plan/0005's emulated FP8
strategy row.

### PTX-translation prior art

ZLUDA (PTX->LLVM->AMDGPU), CuPBoP, BarraCUDA (CUDA->RDNA3, no LLVM),
CASS (ML NVIDIA->AMD transpilation), HetGPU, NVLift (SASS->LLVM),
GpuOcelot/GPGPU-Sim. Lesson for a same-vendor same-family bridge:
rewriting is far easier than the cross-vendor problem — identical warp
size, memory model, and SASS family; only the capability-gated
instructions need substitution, and NVIDIA's own header guards
(`CUDA_CP_ASYNC_ACTIVATED`, `NV_DISPATCH_TARGET`) show NVIDIA performs
exactly this substitution at the header level.

## Open items

- m8n8k32 .s4 canonical register arity and `.satfinite` requirement
  (empirical open question; does not gate any committed plan).

## Target versions (best available, verified 2026-08-28)

| component | version | role |
|---|---|---|
| flash-attn | v2.8.3 stable line (FA4 v4.0.0-beta28 is Blackwell tcgen05 — out of bridge scope) | quilt target #1 |
| FlashInfer | v0.6.18 (2026-08-10; our venv pins 0.6.17 — bump candidate, both vendor their csrc locally) | quilt target #2 |
| CUTLASS | 4.8.0 — 2.x Turing API headers mma_sm75.h / memory_sm75.h, retention to be verified in the first implementation stage | prior-art reference + header source |
| CCCL | CUDA 13.3 toolkit does NOT ship `cuda/__barrier` headers at the expected path — the bridge vendors the pre-sm80 barrier emulation per the CCCL SM_70 recipe | mbarrier emulation |
| torch | 2.13.0+cu130 | extension ABI |
| vLLM | fork sm75-marlin @ pinned tip | integration host |

## References

- PTX ISA 9.3, NVIDIA:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
  (mma shapes 9.7.15.5; ldmatrix/movmatrix 9.7.15.5.15/17; cp.async
  9.7.9.26; mbarrier 9.7.14.16; cvt 9.7.9.22; TMA 9.7.9.26.4-5;
  tcgen05 9.7.17)
- flash-attn: https://github.com/Dao-AILab/flash-attention
  (kernel_traits.h sub-80 path; setup.py arch list; host gate
  csrc/flash_attn/flash_api.cpp; releases: fa4-v4.0.0.beta28, 2.8.x)
- FA2 Turing port (prior art): https://github.com/ssiu/flash-attention-turing
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
  (releases v0.6.18-20260819; sm_75 breakage: issue #3620, fix PR #3621;
  paper arXiv:2501.01005)
- vLLM engine gating:
  https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/
  (flash_attn.py >= (8,0); flashinfer.py floor comment; triton_attn.py
  returns True); V0 removal RFC #18571
- CUTLASS: https://github.com/NVIDIA/cutlass (4.8.0; arch/mma_sm75.h,
  arch/memory_sm75.h, gemm/threadblock/mma_pipelined.h,
  arch/memory_sm80.h CUDA_CP_ASYNC_ACTIVATED guard; bfloat16.h)
- CCCL pre-sm80 barrier: https://github.com/NVIDIA/cccl
  (libcudacxx cuda/__barrier/barrier_block_scope.h SM_70 dispatch;
  issue #419)
- FP8-Marlin recipe:
  https://github.com/vllm-project/vllm/blob/v0.8.5/csrc/quantization/fp8/fp8_marlin.cu
  (FasterTransformer interleaved converters); vLLM FP8 docs
  https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8/
- MMA microbenchmarking (k-order): arXiv:2208.11174
- FP8 format: arXiv:2209.05433; FP8-Marlin kernel paper: arXiv:2408.11743
- INT8 Turing prior art: https://github.com/jundaf2/CUDA-INT8-GEMM
  (4x m8n8k16 m16n16k16); CUTLASS int8 layouts: discussion #911, issue #1030
- Translation-layer prior art: ZLUDA https://github.com/vosen/ZLUDA ;
  GPU Ocelot https://github.com/gpuocelot ; HetGPU arXiv:2506.15993 ;
  CASS arXiv:2505.16968 ; NVLift/Sass-LLifter ; BarraCUDA ; CuPBoP
  (ACM 10.1145/3624062.3624185)
- xformers FMHA (CUTLASS 2.x Turing attention):
  https://github.com/facebookresearch/xformers
- llama.cpp CUDA on Turing: ggml-org/llama.cpp discussion #15013 ;
  https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/
- NVIDIA EULA translation-layer note:
  https://www.reddit.com/r/programming/comments/1b7go8t/nvidia_bans_using_translation_layers_for_cuda/
