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

## Open items

- m8n8k32 .s4 canonical register arity and `.satfinite` requirement
  (empirical open question; does not gate any committed plan).
- Track 3 (emulation techniques: CUTLASS 2.x staging idioms, mbarrier
  emulation patterns, FP8-conversion bit tricks) — appended when the
  research completes.
