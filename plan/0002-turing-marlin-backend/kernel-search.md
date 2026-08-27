# Kernel search and the generated configuration table

The search framework (plan deliverable: the automated generation,
compilation, SASS validation, benchmarking, and result storage) and the
first generated Turing configuration table. Code:
`turing_lab/search/` on fork branch `sm75-marlin`; harness:
`turing_lab/search/search.py` (verify: it exits zero with every candidate
oracle-checked).

## Search space

Eight candidates over the tile and warp space, all template instantiations
of one staged kernel (BM, BN, BK 64, WARPS_R = BM/16, WARPS_N = WARPS/WARPS_R):

```
bm64_bn64_w4   bm64_bn128_w4   bm64_bn128_w8   bm128_bn64_w8
bm128_bn128_w8 bm32_bn64_w2    bm32_bn128_w4   bm64_bn64_w8
```

WARPS_R tiles the rows in disjoint 16-row slices; WARPS_N splits BN; each
warp covers `BN/WARPS_N` columns as `m16n8k8` fragments.

## Gates, in order

1. Compile gate: ptxas must accept the instantiation for sm_75.
2. Oracle gate: one case per candidate against the float64 reference; a
   candidate that is wrong is never timed.
3. SASS/resource gate: `cuobjdump -res-usage` per kernel; STACK above zero
   means local-memory spills. Measured spread: 64 to 126 registers, stack
   64 to 126 bytes on the BN 128 variants (the 16-accumulator-set variants
   spill; they are timed anyway and the table records it).
4. Shape applicability: a candidate whose BN does not divide the shape's N
   is recorded shape-restricted and skipped for that shape, the same
   divisibility rule the incumbent's `is_valid_config` enforces
   (N 4160 is the adversarial case: it divides by 64, not by 128).

## Timing and selection

Adversarial set: N 4096 and the awkward N 4160, K 4096, M in {1, 8, 32,
128, 512}; median of 10 after warmup, locked clocks. Selection is minimax
over per-shape RELATIVE TFLOP/s (each candidate against the best seen for
the same shape), so the memory-bound M 1 rows cannot dominate. The
selection metric is the worst case over the shape set, not the mean.

## Generated table (first generation)

| M regime | selected config | worst-case relative |
|---|---|---|
| 1 | bm64_bn64_w8 | 1.000 |
| 8 | bm64_bn64_w8 | 1.000 |
| 32 | bm64_bn64_w8 | 1.000 |
| 128 | bm64_bn64_w8 | 1.000 |
| 512 | bm64_bn128_w8 | 1.000 |

Raw data: `config-table-*.json` beside the harness. Reading:

- `bm64_bn64_w8` (eight warps, two column halves, 256 threads) is the most
  steady candidate everywhere except the largest M, where
  `bm64_bn128_w8` takes the lead.
- The worst-case relative scores span roughly four tenths to four
  fifths of the best per shape, confirming the plan's premise: no single configuration is uniformly dominant, so a
  per-M-regime table is real work, not ceremony.
- Absolute numbers remain reference-grade (tens of TFLOP/s): single-stage
  staging, no split-K, one weight read per block. The table's STRUCTURE is
  the deliverable of this task generation; its contents get regenerated
  when the pipelined, Marlin-format kernel replaces the reference.

## Model cross-check

The occupancy constraint from the performance model (two resident blocks
per SM) rules out nothing in this generation: every candidate's shared
footprint is at or under 36.9 KB, and the two-block rule shows up instead
as the flat factor-2 warp-scaling curve measured in the characterization.
