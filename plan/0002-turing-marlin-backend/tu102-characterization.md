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

## Shared-memory path and staging (second sweep, 2026-08-27)

| lane | measured |
|---|---|
| STS, conflict-free float4 | 9959 GB/s |
| STS, 34-float conflict stride | 5399 GB/s |
| ldmatrix x4 | 6.41e9 loads/s (about 3.3 TB/s shared read) |
| staging chain LDG-STS-barrier-consume | 558 GB/s |
| HMMA at 1 block of 8 warps per SM | 43.8 TFLOP/s |
| HMMA at 2+ blocks per SM | 101.7 TFLOP/s (flat to 16) |

Findings:

5. **Bank conflicts halve STS.** The swizzle search in the kernel-design
   space has a hard ceiling proof: a conflicted mapping pays about half.
6. **The no-cp.async staging chain is bandwidth-bound at this footprint**
   (558 GB/s tracks the global read), so the Turing pipeline pays for
   registers and barriers, not for lost bandwidth; the cost shows up in
   occupancy instead (finding 7).
7. **The tensor pipes need two resident blocks per SM.** One block of eight
   warps reaches 43.8 TFLOP/s; two blocks reach the 101.7 ceiling and it
   stays flat to sixteen blocks. This is a hard occupancy constraint for
   the generated configuration table: any config whose shared-memory and
   register footprint allows only one resident block cannot reach the
   pipes' ceiling.

## Levers and red-lines (added 2026-09-07)

Recorded at the operator's direction. This section carries no repeatable
dry specifications; it is the performance envelope the measurements above
prove. Every number below is measured above in this paper or in a
committed plan record of this host. External claims (upstream kernel
releases, Triton codegen floors, vendor spec sheets) are not accepted
here; they live in research/lacunae until verified or measured.

### Red-lines (measured walls)

1. **The FP16 tensor ceiling at the locked clock is 101.7 TFLOP/s** (the
   flat region beyond two resident blocks). No FP16-tensor kernel on this
   host crosses it. The current opt kernels at 23-25 and the incumbent at
   51-57 sit at roughly a quarter and half of it respectively; the
   remaining gap to the incumbent is pipeline depth, and past the
   incumbent the headroom to the wall is about two times.
2. **INT8 tensor math tops at 203 TOPS, exactly twice FP16.** The s4 MMA
   does not exist on sm_75 (finding 3), so sub-INT8 weights pay register
   dequantization permanently. The dense-math design space is
   dequant-to-FP16 versus dequant-to-INT8.
3. **The k8 wall.** f32-accumulate MMA above k8 is rejected below sm_80
   (finding 4): every k16 tile is two issues, and the operand-staging cost
   is inherited by every backend that will ever be written here.
4. **The occupancy red-line.** One resident block per SM caps the tensor
   pipes at 43.8 TFLOP/s (finding 7); two reach 101.7. Any candidate
   configuration whose shared-memory and register footprint allows only
   one resident block is capped at 43 percent of the ceiling before it
   runs. This is a legality gate, not a tuning knob.
5. **A conflicted shared-memory mapping pays half** (finding 5). Swizzle
   correctness is a correctness-class gate, not an optimization.
6. **Decode is memory-bound.** Pure read achieves 535 GB/s and the W4A16
   crossover sits at M* = 47: below it, weight bytes per parameter is the
   only lever; above it, tensor-pipe efficiency is.
7. **TP2 communication is measured, not spec'd.** The NVLink path moved
   43.7 GB/s in the plan/0004 protocol, and at decode batch sizes the
   allreduce is latency-dominated with the variance owned by the engine,
   not the kernel (plan/0007 bisect record).

### Levers (what can still drive gains inside the walls)

1. **Dequant-instruction economy.** The contention table prices every ALU
   op against tensor throughput, and HFMA2 steals about 40 percent less
   than FFMA or LOP3 at high density. The path from 23-25 toward and past
   51-57 is fewer total ops per weight: the swizzled lop3+hsub2 stream
   (regdeq2), the repack interlace that deletes byte_perm, split-K for
   small M, deeper stages.
2. **The INT8 tensor track is open and unused.** 203 TOPS is measured and
   no current kernel touches it (all are FP16-tensor). A W8A8-INT8 row is
   the only way to double dense math per clock on this silicon; the cost
   is activation quantization (a numerics project), not a kernel
   mechanism.
3. **Weight-byte reduction below four bits.** Below the crossover,
   decode latency tracks weight bytes at fixed 535 GB/s; W2A16 halves
   W4's bytes (plan/0005 AutoRound lane).
4. **Software pipeline depth.** Registers and barriers are the cost
   currency of the no-cp.async chain (finding 6); register-resident
   fragments and deeper staging are precisely the incumbent's measured
   advantage, so the gap decomposition is the work order (plan/0004).
5. **Engine-level multipliers.** Speculative decoding and KV-byte
   reduction multiply or protect memory-bound throughput without touching
   the silicon limits above; both are scoped as research/lacunae items
   with no numbers accepted yet.

These red-lines are gate inputs: a candidate that violates one is culled
before any timing, per the standing oracle-before-timing rules.

## Open lanes (characterization task stays open)

The M-sweep that tests the crossover prediction against real Marlin shapes
belongs to the reference-backend and search tasks, which own a kernel to
sweep. The unexploited INT8 tensor track (lever 2) is now an open lane of
this characterization: an activation-quantization feasibility note is the
missing prerequisite.
