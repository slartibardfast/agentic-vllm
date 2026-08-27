# Reference Turing W4A16 backend

The smallest correct backend, per the build sequence: correctness first, the
Marlin weight semantics honored, the sm_75 tensor-core shape, and a
performance baseline to improve on. Code: `turing_lab/reference/` on fork
branch `sm75-marlin`; harness: `turing_lab/reference/oracle.py` (verify:
it exits zero when every case passes).

## Weight contract

- `qweight[n, k/8]`: uint32, low-nibble-first 4-bit values along k.
- `scales[g, n]`: fp16, one scale per group of G k-elements (G in 32, 64,
  128, or per-tensor).
- `zps[g, n]`: optional int32 zero points; absent means the symmetric form
  with zero point 8.
- `w[n, k] = (q - zp) * scales[g(k), n]`

Marlin-format consumption is deliberately deferred to the optimized kernel;
the reference fixes its own simpler layout so the math can be validated in
isolation from layout machinery.

## Kernels

1. `w4a16_naive`: one thread per output element, fp32 CUDA-core math, per-k
   zero-point lookup. Slow by construction; it is the on-GPU ground truth
   that catches oracle bugs before kernel bugs.
2. `w4a16_tiled`: BM 64, BN 64, BK 64 blocks; A staged padded in shared;
   weights dequantized (scale and zero point folded into the staged fp16)
   n-major so the `m16n8k8` B fragment reads contiguous k pairs; disjoint
   16-row warp slices; `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`,
   the widest FP32-accumulate shape sm_75 offers.

A note on the scale placement: folding scales into the staged weights (the
reference's choice) rounds the weight to fp16 after scaling, one rounding
step earlier than Marlin's own accumulator-domain application. The oracle's
tolerance covers it. The optimized backend returns the scale to the
accumulator domain, which is Marlin's semantics and the reason the incumbent
prefers the `skip_flop` dequant.

## Oracle results (all pass)

Seven cases spanning M 1 to 100 (guard coverage), N 64 to 512, K 64 to 768,
groups 32/64/128, symmetric and zero-point forms; both kernels checked
against a float64 reference:

```
M16 N64 K64 G64 sym : naive 0.0038  tiled 0.0048  (max abs err)
M64 N128 K256 G64 sym : naive 0.0078  tiled 0.0109
M1 N64 K128 G32 zp : naive 0.0074  tiled 0.0074
M7 N192 K384 G128 zp : naive 0.0156  tiled 0.0176
M100 N256 K256 G64 sym : naive 0.0078  tiled 0.0127
M64 N512 K512 G128 zp : naive 0.0156  tiled 0.0257
M33 N128 K768 G32 sym : naive 0.0156  tiled 0.0203
```

Bugs the oracle caught on the way to green, recorded because each is a
class the optimized kernel can reintroduce: shared-memory row padding that
misaligned `half2` fragment loads (pad to 8, not 1), the B fragment's
consecutive-k pairing requiring n-major staging, per-chunk scale application
needing fresh per-chunk accumulators, and a store/compute row-base mismatch
that only appeared past the first 16 rows.

## Performance baseline (locked 1455 MHz, median of 20)

| M | tiled kernel |
|---|---|
| 1 | 0.16 TFLOP/s |
| 8 | 1.27 |
| 32 | 4.03 |
| 128 | 12.0 |
| 512 | 15.4 |

The baseline sits far below the 101 TFLOP/s ceiling by design: no
double-buffered pipeline, no split-K, one weight read per block. The kernel
search (kernel-search.md) improves from here, and the M-sweep task measures
the crossover with the optimized kernel rather than with this baseline.
