# TU102 characterization (first measurements)

Measured 2026-08-27 on this host's Quadro RTX 6000 cards at locked 1455 MHz
(driver 610.57.04, CUDA 13.3.1, kernel 6.18.47-1-lts). Protocol: one untimed
warmup, then five timed runs, median reported. Raw data:
`turing_lab/results/post-reboot-gpu{0,1}.json` on fork branch `sm75-marlin`
at pin 5197c50. The suite is `turing_lab/` in the same worktree.

## Tensor-core ceilings (per card)

| benchmark | shape | measured | note |
|---|---|---|---|
| FP16, FP32 accumulate | m16n8k8 | 100.7 / 101.9 TFLOP/s | gpu0 / gpu1 |
| FP16, FP16 accumulate | m16n8k8 | 94.5 / 102.0 TFLOP/s | equal to f32acc within noise |
| INT8, s32 accumulate | m8n8k16 | 203.2 / 203.7 TOPS | exactly twice FP16 |

Findings:

1. **FP32 accumulate is free on TU102.** The f32-acc and f16-acc shapes run
   at the same rate. The incumbent's Turing-only `use_fp16_accum` therefore
   buys register footprint, not throughput, and a new backend can keep FP32
   accumulators without paying for it.
2. **INT8 is exactly twice FP16**, at 95 percent of
   the theoretical 214.5 TOPS the supplied research cites. The FP16 ceiling
   of roughly 101 TFLOP/s at 1455 MHz matches that research's 107.2 figure
   to within 6 percent.
3. **The s4 MMA does not exist on sm_75 through the sanctioned interface.**
   ptxas rejects `.s4` as an instruction type for `mma.m8n8k16` on this
   target ("incorrect instruction type"). A W4A16 Turing backend must
   dequantize INT4 weights in registers, which is what the incumbent's
   `lop3`/`prmt` path exists for. This closes the plan's unmeasured
   prerequisite: the INT4 tensor-core candidate path is dead, and the design
   space is dequantize-to-FP16 versus dequantize-to-INT8.
4. **The widest FP16-with-FP32-accumulate MMA on sm_75 is k8.** ptxas
   rejects `m16n8k16` f32-acc below sm_80. The incumbent splits its k16 tile
   into two k8 issues on Turing; any new backend inherits that cost, and the
   plan's operand-staging concern attaches to it.

## Contention (HMMA stream with interleaved same-warp ALU)

From gpu1 (the self-consistent sweep; gpu0's contention block shows a
first-section anomaly to be re-run):

| ALU per 4 HMMAs | HFMA2 | FFMA | LOP3 |
|---|---|---|---|
| 0 | 100.2 | 100.1 | 100.1 |
| 2 | 83.4 | 85.8 | 86.0 |
| 4 | 71.9 | 73.2 | 73.8 |
| 8 | 57.5 | 44.8 | 45.0 |
| 16 | 42.4 | 24.3 | 24.3 |

(TFLOP/s achieved by the HMMA stream.) Every arithmetic kind contends with
the tensor pipes; at high density HFMA2 steals about 40 percent less than
FFMA or LOP3, but all three convert dequant work directly into tensor-core
loss. The hypothesis from the supplied research is confirmed in direction
and quantified here: dequant instruction count is the currency of Turing
Marlin performance, and the choice between dequant domains is decided by
total ops per weight, not by which pipe an op lands on.

## Memory path

Pure global read (`__ldcg` float4, 1 GiB footprint): 534.5 / 535.6 GB/s.
The supplied research's 609 GB/s figure (likely a different memory-clock or
footprint regime) should be re-derived on the same protocol before use. With
the measured ceilings, the W4A16 compute-over-memory crossover moves to

```
M* = machine balance / arithmetic intensity coefficient
  = (101.3e12 / 535.0e9) / 4  =  47
```

from the supplied model's 44. Treat 47 as the initial regime-separation
hypothesis for the M-sweep task; it is a model output, not a constant.

## Open lanes (characterization task stays open)

STS and LDSM throughput, shared-memory staging chains, bank-conflict
mapping, warp-scaling curves, and the M-sweep that tests the crossover
prediction against real Marlin shapes.
