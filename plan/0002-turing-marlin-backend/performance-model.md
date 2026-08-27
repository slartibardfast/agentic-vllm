# Turing Marlin performance model (measured)

Inputs are the characterized ceilings (tu102-characterization.md), measured
at locked 1455 MHz on this host's cards. Everything here is a hypothesis for
the kernel search to validate; nothing is a constant.

## Machine balance

```
FP16 tensor ceiling (either accumulate mode)   101.3 TFLOP/s
INT8 tensor ceiling (2x FP16)                  203.5 TOPS
global read bandwidth (__ldcg)                 535.0 GB/s
machine balance = 101.3e12 / 535.0e9           189 FLOP per byte
```

## W4A16 regime crossover

A weight-stationary W4A16 GEMM moves about `K*N/2` weight bytes and `2*M*N`
output bytes per `2*M*N*K` FLOPs, so its arithmetic intensity is `4*M`
FLOP per byte for M small against large N,K:

```
M* = 189 / 4  =  47
```

The supplied research derived 44 from its own ceilings (107.2 TFLOP/s,
609 GB/s); the measured rig gives 47. Prediction for the M-sweep: a
weight-stationary kernel stops improving past M near 47 and an output-stationary
kernel stops winning below it; the sweep over M in {1, 2, 4, 8, 16, 32, 64,
128, 256, 512} on real Marlin shapes tests whether the crossover is sharp
enough to deserve a dispatch split.

## Dequant cost model

Contention measurements price one interleaved ALU op at roughly a quarter
of an HMMA slot once the mix saturates (16 ALU per 4 HMMA leaves the tensor
stream at 24 to 42 percent depending on kind). Per eight 4-bit weights, the
candidate dequant domains cost:

```
dequant to FP16, scale folded at load   2 lop3            (skip_flop path)
dequant to FP16, explicit               2 lop3 + 2 HFMA2
dequant to INT8                         byte-perm + arithmetic shifts
```

The FP16 skip_flop path is the cheapest per weight and is what the
incumbent prefers; the INT8 path only wins if a k16 s8 MMA's doubled K
per issue outweighs its extra unpack arithmetic. The kernel search measures
both, with the SASS gate confirming the intended instruction mix.

## Occupancy constraint (new, measured)

The HMMA stream needs at least two resident blocks of eight warps per SM to
saturate. Every generated config must satisfy this, or its tile table entry
is rejected by the search before it is ever timed.

## Model error

The contention sweeps show the model's per-mix error is real: adjacent ALU
counts swing the tensor stream by tens of percent. The model filters
candidates; only hardware timing selects them.
