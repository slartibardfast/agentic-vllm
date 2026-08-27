# Marlin contract at the pin

Reconstruction of the Marlin implementation in the hosted vLLM fork at pin
79378fe. Status: draft covering the dense path and the kernel-side Turing
facts; the MoE path and the repack formats get their own sections as they are
traced in full. Every claim below names the file it was read from, so a
reviewer can check it against the worktree at
`software/vllm/main/`.

## Source map (verified at the pin)

| layer | file |
|---|---|
| dense kernel selection | `vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py`, `marlin.py` |
| dense CUDA entry | `csrc/libtorch_stable/quantization/marlin/marlin.cu` |
| kernel body | `csrc/libtorch_stable/quantization/marlin/marlin_template.h` |
| MMA wrappers | `csrc/libtorch_stable/quantization/marlin/marlin_mma.h` |
| dequantization | `csrc/libtorch_stable/quantization/marlin/dequant.h` |
| repack kernels | `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu`, `awq_marlin_repack.cu` |
| MoE experts | `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`, `csrc/libtorch_stable/moe/marlin_moe_wna16/` |
| utils (repack, checks) | `vllm/model_executor/layers/quantization/utils/marlin_utils.py` |

The plan's premise is confirmed at the pin: Turing is an explicit, supported
branch, not an excluded architecture. `marlin.cu` and `marlin_template.h`
compile the kernel only for `__CUDA_ARCH__ >= 750`, and `marlin_mma.h`
carries six `__CUDA_ARCH__ == 750` blocks.

## What the incumbent Turing branch does

- `marlin_mma.h` at `__CUDA_ARCH__ == 750` provides FP16 tensor-core MMA
  (`mma.sync.aligned.m16n8k8` and `m16n8k16`, both `f16.f16.f16.f16` and
  `f32.f16.f16.f32` accumulate) and INT8 MMA
  (`mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite`).
- `marlin_template.h` (the activation gate): on Turing the kernel returns
  early unless the activation type is FP16 or INT8. The source comment says
  Turing TensorCore only supports those two.
- `marlin_template.h` (accumulator): Turing is the one architecture where
  `use_fp16_accum` may be true (FP16 activations only), with NVFP4 and
  4-bit-without-grouping excluded. On sm_80 and newer the flag is false.
- `dequant.h`: the fast LUT dequantization path (`lop3`, `prmt`) is compiled
  for `__CUDA_ARCH__ >= 750`, so the incumbent Turing build uses the same
  bit-manipulation dequant style as Ampere.
- The pipeline in `marlin_template.h` is asynchronous-copy based
  (`cp.async` staging with a `stages` parameter) on all architectures
  including Turing; the two-stage characterization from the supplied research
  is a property of the *configuration chosen* on Turing (the stage count and
  tile table selected for sm_75), not of a separate template. Verify the
  selected configs during the characterization task.

## Dense dispatch surface

`MarlinLinearKernel(MPLinearKernel)` in
`vllm/model_executor/kernels/linear/mixed_precision/marlin.py`:

- declares its minimum capability via `get_min_capability` (value to be
  recorded during integration; the sm_80+ guard lives kernel-side at 750);
- allocates its workspace once per process with `marlin_make_workspace_new`,
  and it reuses that storage on weight reload;
- `apply_weights` invokes the custom op (`ops.gptq_marlin_gemm` family) with
  that preallocated workspace, so the launch path carries no allocation and
  no host synchronization, which is what makes CUDA graph capture legal.

The kernel-selection architecture is `MPLinearKernel` subclasses chosen by
`choose_mp_linear_kernel()`; a Turing backend enters as a sibling of
`MarlinLinearKernel`, or as the Turing-capable body behind the same class.
The exact extension point is an integration decision recorded before the
dense integration task starts.

## Weight repack (completed)

The repack kernels (`gptq_marlin_repack.cu`, `awq_marlin_repack.cu`) are
themselves cp.async pipelined transpose kernels. The target tile geometry is
defined in `marlin.cuh`: `tile_k_size` (the architecture tile) and
`tile_n_size = 4 * tile_k_size`; for 8-bit activations the effective tile
doubles in K and halves in N (`is_a_8bit` flips both constants). The source
layout is GPTQ/AWQ canonical: 4-bit values packed `pack_factor = 32/num_bits`
per 32-bit word, K-major. The repack reorders values into the k-tile-major
Marlin layout the kernel's B fragment walks consume directly, optionally
applying the activation-order `perm` during the transpose (`has_perm`), with
its own multi-stage shared-memory pipeline (`repack_stages`, double
buffering). A Turing-specific layout, if measurements ever demand one,
enters exactly here: a sibling repack kernel plus a kernel-side fragment
walk; every upstream producer stays untouched.

## Kernel selection and configuration table (completed)

`marlin.cu` carries two priority-ordered thread-configuration tables:

```
small_batch (thread_m_blocks == 1):
    {thread_k 128, thread_n 128, 256 threads}
    {thread_k 64,  thread_n 128, 128 threads}
    {thread_k 128, thread_n 64,  128 threads}
large_batch (thread_m_blocks > 1):
    {thread_k 64,  thread_n 256, 256 threads}
    {thread_k 64,  thread_n 128, 128 threads}
    {thread_k 128, thread_n 64,  128 threads}
```

`determine_exec_config` walks the table and returns the **first** config
that passes `is_valid_config` (dimension divisibility, minimum thread
constraints, shared-memory fit against `max_shared_mem - 512`, group-size
and act-order constraints) and has a compiled kernel instantiation. There is
no measurement-derived table for any architecture in this version: selection
is validity-first in a fixed priority order. This is exactly the gap the
generated Turing table fills: same machinery, table contents produced by the
search harness rather than by hand.

## Numerical semantics (completed)

- Weights are unsigned 4-bit (`kU4B8`, GPTQ style) dequantized by the LUT
  path in `dequant.h`: `lop3` masks nibbles into halves that carry the FP16
  exponent bias 0x6400 (1024), so one instruction pair yields `1024 + q`
  per nibble.
- Two scale-application modes exist, selected per weight type
  (`skip_flop`): fold the correction into the scale at load time
  (`w = bit_op(q) * (scale * multiplier)`, zero FP16 sub/fma per value) or
  explicit `__hsub2`/`__hfma2` per value (`SUB = 0x64086408` removes the
  1024 bias and the 8 of `q - 8`). This is the dequant instruction-count
  trade the contention measurements price.
- Accumulation: HMMA fragments accumulate in FP32 (or FP16 on Turing with
  `use_fp16_accum`); cross-thread-block partial sums reduce either in FP16
  with global locks or in FP32 via `use_fp32_reduce` (workspace `C_tmp`),
  and `use_atomic_add` switches the split-K reduction to atomics.
- Group semantics: scales cover `group_size` K-elements
  (`group_blocks = group_size/16` shared tiles, or `-1` for per-tensor),
  applied from `b_s`/`g_s` fragments after the MMA, in the accumulator
  domain, per 16-row fragment row.
- Output: FP16/FP8 (`c_type`), optional bias (`b_bias`), written per
  16×8 fragment tile; `workspace` (zeroed by the caller,
  `marlin_make_workspace_new`) backs the global reduction locks.

## MoE (completed)

`marlin_moe.py` exposes `fused_marlin_moe` and `batched_fused_marlin_moe`,
both funneling into `_fused_marlin_moe` and the `marlin_moe_gemm` custom op
over `MarlinExpertsBase` (a `FusedMoEExpertsModular` subclass whose
`_supports_*` predicates gate device and quant scheme). Semantics: expert
weights in the same Marlin packed format (`w13`/`w2`, num_bits 4 or 8),
grouped/batched GEMM over the routed expert list, top-k weights applied
outside the kernel. A Turing MoE path reuses the dense Turing GEMM behind
the same expert interface; no separate MoE kernel architecture is planned.

## Runtime requirements (completed)

The dense custom op (`marlin_gemm` in `vllm/_custom_ops.py`, wrapping
`marlin_mm`) takes caller-preallocated workspace and reduction buffers, has
no data-dependent allocations, and performs no host synchronization in the
launch path, which is what CUDA graph capture requires. `thread_k_init`/
`thread_n_init` may pin a configuration; `sms` may pin the SM count.

