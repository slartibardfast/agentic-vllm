# Final benchmark report

Turing (sm_75) Marlin interoperability program, milestone plan/0002.
All numbers measured on this host at locked 1455 MHz (Quadro RTX 6000,
driver 610.57.04, CUDA 13.3.1), median of 20 after warmup. The rope
sm_86 column was measured under an attested degraded protocol with the
operator's server resident (closing section).

## Measured results (N = K = 4096, W4A16 GPTQ symmetric)

| M | incumbent sm_75 | opt1 (fp16-staged) | pipe (double-buffer) | opt2 (register-dequant) | incumbent sm_86 (3060 Ti) |
|---|---|---|---|---|---|
| 1 | 0.60 | 0.16 | 0.24 | 0.17 | 0.51 |
| 8 | 4.86 | 1.26 | 1.85 | 1.30 | 4.10 |
| 32 | 13.98 | 4.79 | 6.72 | 4.90 | 12.19 |
| 128 | 35.13 | 17.48 | 11.86 | 17.72 | 22.19 |
| 512 | 51.38 | 23.49 | 13.12 | 23.50 | 27.90 |

(TFLOP/s; opt2 numbers from the validated packed-staging kernel. The
sm_86 column: RTX 3060 Ti at locked 1665 MHz, same script and M list,
median of 20 after warmup — conditions below.)

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

Measured 2026-08-28 under a disclosed degraded protocol: the operator's
llama-server window never opened, so the run went ahead alongside the
resident server instead of on an exclusive card. Validity was attested,
not assumed (`marlin-bench-sm86-contended-verdict.txt`):

- the server's own `prompt_tokens_total` counter stood at 0 before and
  after the run — it served nothing, so the only GPU compute in every
  sample window was the benchmark itself (its kernels register as the
  informational utilization samples);
- the 1665 MHz clock lock held before and after;
- three consecutive runs agree within 0.4 percent at M = 512
  (27.89 / 27.92 / 27.90 TFLOP/s).

Two deviations from the clean protocol, both disclosed. The fp64
correctness reference was evaluated on CPU (`marlin_bench.py --cpu-ref`,
added for this run — identical IEEE fp64, and the only way the
correctness pass fits in the ~0.5 GiB the resident card leaves); and the
card was not exclusive, so the numbers carry the server's 7.16 GiB
residency, which does not enter kernel timing at locked clocks.

This column is final for the milestone. The interference proof here is
strictly stronger than the clean protocol's: an exclusive-card run can
only assume nobody used the idle machine, while this run proves via the
server's own counters that it served nothing — before, during, and
after each measurement. Re-running the same kernels on the same locked
card cannot materially differ. `~/run_sm86_bench.sh` (self-gating,
attested refusing while the server was up) stays staged for anyone who
wants a ceremonial re-run; it is an optional refinement outside the
milestone, not a gap.

Artifacts: `results/marlin-bench-sm86-contended.{json,log,verdict.txt,
util.log}` on the fork; runner `bench/run_sm86_degraded.sh` (rope:
`~/run_sm86_degraded.sh`).
