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
  reusing existing storage on weight reload;
- `apply_weights` invokes the custom op (`ops.gptq_marlin_gemm` family) with
  that preallocated workspace, so the launch path carries no allocation and
  no host synchronization, which is what makes CUDA graph capture legal.

The kernel-selection architecture is `MPLinearKernel` subclasses chosen by
`choose_mp_linear_kernel()`; a Turing backend enters as a sibling of
`MarlinLinearKernel`, or as the Turing-capable body behind the same class.
The exact extension point is an integration decision recorded before the
dense integration task starts.

## Open sections (in progress)

- Repack formats: the exact `qweight` layouts produced by
  `gptq_marlin_repack` / `awq_marlin_repack`, and what a Turing-specific
  layout would have to preserve.
- MoE: `marlin_moe.py` backend selection, expert weight semantics, and the
  grouped GEMM entry in `csrc/libtorch_stable/moe/marlin_moe_wna16/`.
- Configuration table: the full `thread_config_t` space and the selection
  function in `marlin.cu`, as the baseline for the generated Turing table.
- Numerical semantics: accumulation order and scale application points, to be
  pinned down so the reference backend can be checked against them.
