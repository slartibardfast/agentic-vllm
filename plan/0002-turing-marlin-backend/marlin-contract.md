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


## The repack interleave contract, RESOLVED (2026-09-16, plan/0011)

Verified term-by-term: the full contract (index map, fragment
consumption, dequant LUTs, scale permutation) was re-implemented as a
NumPy oracle against a reference grouped GEMM - 16384/16384 (k-row,
k, column) terms covered exactly once, every dequantized value
q-8, every scale on its column. Anchors are the lane worktree's pin
(file:line under csrc/libtorch_stable/quantization/marlin/ and
vllm/model_executor/).

### The packed 32-bit word layout (what a repack must emit)

Output tensor: `{size_k/16, size_n*2}` uint32 words; B consumed as
`const int4*` (16-byte units). 4-bit tile = 16k x 64n = 128 words.
Output word `W[T,U,th,w]` at linear address
`T*(2*size_n) + U*128 + th*4 + w` (T = 16-k tile, U = 64-n tile,
th in [0,32), w in [0,4)), nibble i (bits 4i+3:4i):

| nibble i | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| weight (n,k) | (n0,2t) | (n0,8+2t) | (n0+8,2t) | (n0+8,8+2t) | (n0,2t+1) | (n0,9+2t) | (n0+8,2t+1) | (n0+8,9+2t) |

with `n0 = 64U + 16w + c`, `t = th%4`, `c = th/4`, k relative to the
16-k tile. In byte terms: byte 0 = the (k=2t, k=8+2t) pair of n0;
byte 1 = the same pair of n0+8; bytes 2/3 = the (2t+1, 9+2t) pairs.
The (k,k+1) pairing holds per 16-bit extraction (nibbles {0,4} and
{1,5}); at byte granularity each byte is a (k, k+8) pair of one
column - CORRECTING the earlier loose phrasing. Act-order (has_perm)
changes only the source k (src_k = perm[k_idx]); the output map is
identical. AWQ reaches the same target layout after its nibble
reversal. Source: gptq_marlin_repack.cu:100-209.

### Dequant (kU4B8 GPTQ path, dequant.h:121-142), LUT (a&b)|c = 0xea

- lo = lop3(q, 0x000f000f, 0x64006400) -> nibbles {0,4} as fp16
  1024+q; frag_b[0] = hsub2(lo, 0x64086408) -> {q0-8, q4-8} (one SUB
  removes the exponent bias AND the symmetric zero point).
- hi = lop3(q, 0x00f000f0, 0x64006400) -> nibbles {1,5} at mantissa
  bits [7:4] = fp16 1024+16q; frag_b[1] = hfma2(hi, 0x2c00 (2^-4),
  0xd480 (-72)) -> q-8. CORRECTION: on the GPTQ path the 16x
  asymmetry is absorbed by the x2^-4 in the hfma2, NOT by
  premultiplied scales (scales are plain permuted fp16). The pure
  two-lop3 skip-flop variant (dequant.h:107-119) is the kU4/AWQ mode;
  a new GPTQ kernel emulating it would use w = (1024+q)*s - 1032*s as
  one hfma2 with a precomputed addend, or premultiply the odd-half
  scale by 1/16.

### Fragment wiring (Turing m16n8k8 split, marlin_mma.h:22-35)

The k16 step issues two m16n8k8: {a0,a1}x{b[0]} (lower k8) then
{a2,a3}x{b[1]} (upper k8), same accumulator. b_quant_0 = the thread's
repack word j (j selects the 16-n slab at 16j); b_quant_1 = word >> 8.
dequant(word) -> column n0, k {2t,2t+1}/{8+2t,9+2t}; dequant(word>>8)
-> column n0+8, same k. Consuming lane (t,c) reads word th = 4c+t -
the lane's hardware k-pair index equals the repack t: NO register
shuffling anywhere (marlin_template.h:1234-1245, 1286-1289).

### Scales (fp16, group 128, group_blocks = 8)

marlin_permute_scales (marlin_utils.py:460-481): within each 64
consecutive scales, stored[p] = orig[perm[p]] with perm[8i+j] = i+8j
(an 8x8 transpose), applied before launch. In-kernel the per-thread
scale fragment is one int4 = 8 halves at
s_sh_rd = 8*((tx/32)%tb_n_warps) + (tx%32)/4, and the permuted layout
makes half 2j = scale(column 64U+16j+c) and half 2j+1 = scale of
column+8 exactly (verified for all 2048 (thread,k2,j)).

### Notes

qWeightsDBG does not exist anywhere in the worktree (likely a
host-note paraphrase; nothing depends on it). The fork's own opt
kernels use a DIFFERENT self-defined repack (even/odd nibble planes,
sQE/sQO) - disjoint design space; a checkpoint-compatible kernel must
emit the layout above. The incumbent epilogue (D->C global mapping)
was not traced (a new kernel writes its own; the term-level check
proves everything up to the accumulator).
