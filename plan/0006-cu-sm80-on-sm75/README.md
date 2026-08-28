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

## Stages

1. Bridge header + per-primitive oracles + microbenches.
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
