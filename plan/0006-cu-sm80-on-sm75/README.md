# cu_sm80_on_sm75: the primitives bridge for FlashAttention and FlashInfer

Plan/0006. The operator's requirement: flash-attn and FlashInfer — the two
kernel libraries modern vLLM's engine is built on — keep working on sm_75,
state of the art, tracking upstream. Not a wild fork: the deliverable is a
**generic primitives bridge** plus a mechanical, re-appliable patchset, so
upstream releases can be adopted by re-running a checker, not by merging a
divergence.

## The substitution map (the whole project in one table)

| sm_80+ primitive | sm_75 bridge |
|---|---|
| `cp.async.ca/cg` G2S copies, commit-groups, multi-stage pipelines | LDG -> register -> STS behind an N-stage register-staging template (the pipe3 pattern, generalized to N stages and arbitrary tile types) |
| `mma.sync.m16n8k16` FP16->FP32 | 2x `mma.sync.m16n8k8` (accumulate-order difference attested by the bit-exactness oracle) |
| `mma.sync.m16n8k32` INT8 shapes | Turing IMMA reshape (`m16n8k16.s8`, 2 issues) — Turing has INT8 tensor cores at 2x FP16 (D8-measured) |
| bf16 arithmetic/MMA | fp16 recast (8 mantissa bits subset 10; range check at boundaries) |
| `ldmatrix` / `movmatrix` | NATIVE on sm_75 — passthrough (Turing introduced them; this is a genuine asset) |
| `cuda::barrier` / `cuda::memcpy_async` libcu++ abstractions | bridge-backed equivalents |
| TMA / WGMMA / clusters (sm_90, FA3-only) | out of scope by definition |

## The bridge set (what cu_smXX_on_sm75 should exist)

Defined at the PTX/ISA primitive level, per source architecture —
not per consumer:

| Bridge | Unlocks | New primitives to emulate | Effort |
|---|---|---|---|
| cu_sm80_on_sm75 | flash-attn 2, FlashInfer sm_80 paths, Marlin family, CUTLASS sm_80 kernels, most Triton output | cp.async + commit/wait groups, m16n8k16 FP16->FP32, m16n8k32 INT8 reshape, bf16 arithmetic, mbarrier | core |
| cu_sm89_on_sm75 | FP8-targeted kernels (Ada generation) | FP8 MMA as cvt->FP16 MMA; e4m3/e5m2 cvt | thin, delta over sm_80 |
| cu_sm90_on_sm75 | FA3-class kernels, DeepGEMM-class, Machete, FlashInfer Hopper paths | WGMMA->mma.sync loops with ldmatrix (native), TMA->cp.async/ld+st, clusters->flat, mbarrier phases | large |
| cu_sm100/120_on_sm75 | Blackwell-era NVFP4/MXFP4/FP6-FP8 block-scaled checkpoints | tcgen05->mma.sync (functional only); the real value is format decode: FP4 LUT + block scales | medium |

Priority follows the table: sm_80 is the core, sm_89 is a thin FP8
delta, sm_90 buys the Hopper kernel surface, sm_100/120 buys checkpoint
format decode where the machinery (nibble LUTs, block scales) is
already ours.

## Non-overfit design rules

1. The bridge is specified against the PTX ISA documentation's
   semantics — conformance means implements-the-spec, so any consumer
   gets exactly what the spec guarantees.
2. Deliverables are consumer-agnostic: a device header library, a
   PTX->PTX rewrite pass for the instruction families (so upstream
   artifacts can be post-processed without source patches at all), and
   a per-primitive conformance suite.
3. vLLM is consumer #1 behind a thin adapter (backend registration and
   dispatch only). If vLLM's internals change, the bridge and the
   conformance suite are untouched.
4. The generality proof is multi-consumer: flash-attn, FlashInfer, a
   Triton-emitted kernel, and a llama.cpp-style dequant-GEMM must all
   run on the same bridge before it is called compatible.
5. Honest limits stay attached per bridge: no register-file bypass for
   cp.async, no smem-direct WGMMA operands, no tensor memory, and
   emulation runs at FP16 tensor rate, never at native FP8/FP4 rates.

## Red-line performance contract (raw PTX headers, by construction)

The bridge is hand-written inline-asm PTX in .cuh headers — instruction
selection, cache hints, STS ordering and pipeline schedules are fixed by
the header, not left to the compiler. C++ wrappers exist only at the API
surface. Per-primitive red lines are enforced by conformance tests that
FAIL THE BUILD on regression:

1. **cp.async semantics are 1:1 by construction.** On sm_75 the STS
   cannot issue before its LDG retires - the hardware scoreboard is
   wait_group for per-thread ordering, so commit_group/wait_group
   compile to scheduling fences at zero runtime cost. The one
   irreducible delta is registers held in flight; the occupancy gate
   prices it (CUTLASS 2.x Turing GEMMs demonstrated near-peak DRAM
   utilization with exactly this pattern - the line is reachable, not
   aspirational).
2. **Staging bandwidth floor**: the staged-copy pattern sustains >= 90%
   of the bench_memory-measured DRAM peak at reference tile sizes.
3. **MMA adapter overhead**: <= +1 ALU op per fragment over the native
   sequence; issue slots <= 2x the k16 native (the 2x k8 split is by
   definition), verified against bench_mma.
4. **Cache discipline**: streaming loads carry .cs/.nc hints so staging
   never evicts hot L2; verified by SASS inspection.
5. **Register budget**: per-config caps from the occupancy rule;
   violations fail the SASS gate.

## Architecture

1. **The bridge header** (`turing_lab/bridge/sm80_on_sm75.cuh`):
   header-only, namespaced, with an N-stage pipeline template, the MMA
   adapters, and the copy primitives. Per-primitive unit tests live with
   the header.
2. **Per-primitive validation**: bit-exactness oracles against the
   sm_80 semantics (fp64 references for the MMA adapters, sync-copy
   reference for cp.async), the SASS gate with an occupancy budget, and
   microbenchmarks against the plan/0002 bench_mma/bench_memory floors.
3. **Target integration, minimal-fork style**: a versioned quilt of
   mechanical patches against flash-attn and flashinfer release tags
   that swaps their primitive layer for the bridge, plus a checker
   script: fresh upstream checkout -> apply quilt -> compile for sm_75
   -> run the attention oracle + benches. Green means the upstream
   release is adopted; red means the quilt needs one mechanical update.
4. **vLLM integration**: the fork's backend selection recognizes
   FA-via-bridge and FlashInfer-via-bridge as available on sm_75, so
   upstream engine evolution keeps flowing — the point of the whole
   exercise.

## What the bridge cannot provide (recorded honestly)

- cp.async's register-file bypass: the LDG+STS substitute pays registers
  and issue slots; occupancy impact is measured by the SASS/occupancy
  gates and priced into the benches, not wished away.
- No TMA/WGMMA: FA3-style warp specialization stays out of scope until
  someone writes an sm_75-only attention design that beats FA2-on-bridge.
- No bf16 tensor rate: bf16 workloads run the fp16 recast.

## Why Turing is a good host for this

- `ldmatrix`/`movmatrix` are native (introduced with Turing).
- FP16 tensor cores at FULL rate with FP32 accumulate — the accumulate
  mode attention wants; the Ampere-consumer halving does not apply.
- The program's existing assets already cover half the map: the k16->2xk8
  MMA split, register-staged N-buffer pipelines (pipe/pipe3), SASS and
  occupancy gates, float64 oracles, the ladder, and CUDA-graph
  validation.

## Auto-research over the bridge (nothing left on the table)

The .cuh headers are generated artifacts, not hand-written endpoints.
Every primitive is a schedule with a searchable design space, and the
plan/0003 machinery (TPE ladder, oracle, winner gates) drives it:

- **Space per primitive**: the staged-copy bridge searches stage count
  (2-8), vector width, XOR-swizzle pattern, LDG/STS/MMA issue
  interleavings, cache hints (.cs/.nc/plain), and unroll factors; the
  MMA adapters search pair-split order, operand forwarding, and dual
  issue interleaving; converters search LUT sizing vs ALU sequences.
- **Hard constraints, non-negotiable**: bit-exactness (schedules are
  semantically identical by definition - any oracle diff is a defect,
  not a tradeoff), SASS/occupancy legality, and the red-line floors
  (bandwidth, issue, register caps). These gate, never trade.
- **Exhaustive where enumerable, TPE where not**: schedule permutations
  within a stage budget are enumerated outright; only beyond the
  enumeration budget does the TPE sampler take over.
- **The coverage ledger**: every legal point in the space is either
  measured (timing transcript committed) or model-pruned (the pruning
  reason committed). Nothing-on-the-table is therefore auditable: no
  silently skipped candidates, ever.
- **Provenance in the shipped header**: the winning schedule is frozen
  into the .cuh with the search transcript hash, so every primitive
  carries the evidence it was chosen by measurement, and a future
  generation can re-run the search against new upstream releases.

## Stages

1. Bridge header generator (schedule-space definition) + per-primitive
   oracles + red-line conformance + microbenches.
2. flash-attn quilt: sm_75 build of FA2's sm_80 kernel path; attention
   oracle vs the fp64 softmax reference; benches vs the current engine
   attention.
3. FlashInfer quilt: the V1-engine APIs vLLM uses (paged decode/prefill,
   sampling); same gates.
4. vLLM selection wiring + engine smoke on both cards.
5. Upstream-tracking checker committed (quilt-apply + validate script).

## Acceptance

1. A current flash-attn release and a current flashinfer release compile
   for sm_75 via the quilt and pass their attention/oracle suites.
2. vLLM engine runs on TU102 using them, tokens/s and context-length
   scaling committed against the incumbent engine attention.
3. The checker script demonstrates the upstream-tracking contract:
   fresh checkout -> quilt -> green, in one command.
4. No behavioral divergence: every quilt patch is a primitive
   substitution, reviewable as such.
