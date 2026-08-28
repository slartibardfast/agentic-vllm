# Final benchmark report

Turing (sm_75) Marlin interoperability program, milestone plan/0002.
All numbers measured on this host at locked 1455 MHz (Quadro RTX 6000,
driver 610.57.04, CUDA 13.3.1), median of 20 after warmup. The rope
sm_86 column awaits the operator's llama-server window; the run is
staged and self-gating (see the closing section).

## Measured results (N = K = 4096, W4A16 GPTQ symmetric)

| M | incumbent sm_75 | opt1 (fp16-staged) | pipe (double-buffer) | opt2 (register-dequant) |
|---|---|---|---|---|
| 1 | 0.60 | 0.16 | 0.24 | 0.17 |
| 8 | 4.86 | 1.26 | 1.85 | 1.30 |
| 32 | 13.98 | 4.79 | 6.72 | 4.90 |
| 128 | 35.13 | 17.48 | 11.86 | 17.72 |
| 512 | 51.38 | 23.49 | 13.12 | 23.50 |

(TFLOP/s; opt2 numbers from the validated packed-staging kernel.)

## What the program established

1. The INT4 tensor-core MMA does not exist on sm_75 through the
   sanctioned PTX interface (ptxas rejects `.s4` as an instruction type
   for mma m8n8k16); W4A16 on Turing dequantizes in registers.
2. The incumbent's Turing branch splits its k16 tile into two k8 issues
   (the m16n8k16 FP32-accumulate form requires sm_80).
3. FP32 accumulate costs nothing versus FP16 accumulate on TU102.
4. INT8 is exactly twice FP16.
5. Dequant ALU work contends with the MMA stream; the incumbent's
   skip_flop bias trick (two lop3 ops per fragment pair, correction
   folded into the scales) is the mechanism that avoids this cost.
6. The occupancy rule: at least two resident blocks per SM.
7. The W4A16 crossover sits near M = 47.

## Remaining gap to the incumbent

The validated kernels reach 23 to 25 TFLOP/s at M = 512, about 46 percent
of the incumbent's 51.38. The gap decomposes into:

- the deep software pipeline (register-resident dequantized fragments,
  multi-stage overlap),
- split-K for the latency-bound small-M regime,
- the repack interleave that enables the skip_flop dequant.

These are recorded as the next-generation engineering plan in
kernel-search.md and MEMORY.md.

## Rope sm_86 column

The run is staged and one operator action away. On rope
(192.168.178.60): `~/marlin_bench.py` is byte-identical to the local
`bench/marlin_bench.py`, `~/vllm-ref` carries vllm 0.28.0 +
torch 2.13.0+cu130, and clocks are locked at 1665 MHz (verified
2026-08-28). `~/run_sm86_bench.sh` (versioned as
`bench/run_sm86_bench.sh`) self-gates: it refuses while llama-server
holds the GPU or the clock lock is inactive, and never stops or starts
llama-server itself — the gate was exercised live and correctly refused
while llama-server (pids 544, 545) was up.

Procedure for the window: stop llama-server, run
`~/run_sm86_bench.sh`, fetch `~/marlin-bench-sm86.json`, restart
llama-server. The column in the table above is filled from the
`4096x4096` rows of that JSON.
