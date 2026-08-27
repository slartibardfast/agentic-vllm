# Turing Marlin backend

Serve an operator who wants quantized Marlin inference to run well on Turing
TU102 hardware. The objective is a high-performance sm_75 backend for the
Marlin ecosystem in vLLM, delivered as a fork-hosted branch plus its
verification machinery.

The invariant is Marlin interoperability: the resulting backend must preserve
the Marlin/vLLM computational, quantization, weight-layout, dispatch, and
runtime contracts wherever practical. The internal CUDA implementation is free
to diverge substantially from the Ampere implementation where TU102 requires
it. The central rule: preserve the Marlin contract; replace only the machinery
TU102 cannot execute efficiently.

## Scope

In scope:

- the vLLM Marlin integration (GPTQ/AWQ-style dense GEMMs, and Marlin MoE
  GEMMs where applicable)
- sm_75 CUDA/PTX/SASS implementation, weight packing and repacking, kernel
  configuration and dispatch
- correctness, CUDA graph compatibility, adversarial performance
  optimization, and comparison against existing Marlin implementations

Out of scope: model-specific inference architecture, DeepSeek-specific
optimization, MLA/attention, expert-cache architecture, host offload, NVLink
system architecture, replacing the vLLM runtime, and any general-purpose
Turing inference engine.

## Rigs (recorded in call/0003)

| rig | GPU | arch | clocks | role |
|---|---|---|---|---|
| this host | 2× Quadro RTX 6000, 24 GB | sm_75, driver 610.57.04, CUDA 13.3.1 | locked 1455 MHz | target |
| rope (192.168.178.60) | RTX 3060 Ti, 8 GB | sm_86, driver 610.57.04, CUDA 13.3.1 | locked 1665 MHz | native Ampere reference |

Both rigs run an identical, aligned package set (linux-lts, nvidia-open-dkms,
nvidia-utils, cuda 13.3.1); the full matrix and the lock procedure live in
[environment.md](environment.md).

Every number in any report names its rig and its locked clock. The incumbent
to characterize and beat is the explicit Turing branch already present in the
Marlin sources at the pin (two-stage pipeline, FP16 or INT8 activations);
verify that description against the pin before building on it.

## Deliverables

Status: the contract, the characterization, the model, the reference
backend, and the search are complete and documented here
([marlin-contract.md](marlin-contract.md),
[tu102-characterization.md](tu102-characterization.md),
[performance-model.md](performance-model.md),
[reference-backend.md](reference-backend.md),
[kernel-search.md](kernel-search.md)); integration, the adversarial
runtime campaign, the MoE path, and the final report are in flight.

1. The Marlin compatibility specification: the full contract (inputs, weights,
   quantization semantics, layouts, dispatch, runtime requirements) for dense
   and MoE paths at the pin.
2. The TU102 microbenchmark suite: tensor-core throughput (FP16 HMMA, INT8
   MMA, and the INT4 investigation), the memory path (LDG, STS, LDSM, staging
   chains), contention mixes, register and shared-memory behavior.
3. A correct reference Turing backend: minimal, complete, independently
   testable.
4. The kernel search framework: automated generation, compilation, SASS
   validation, benchmarking, and result storage.
5. The generated Turing configuration table, produced from measurements.
6. vLLM integration: dense first, then applicable MoE paths.
7. The adversarial validation suite: correctness plus performance plus
   held-out workloads.
8. The final benchmark report: upstream sm75 Marlin, the new backend, and the
   native sm_86 reference on rope, with latency, throughput, worst-case
   regression, numerical error, supported configurations, SASS
   characteristics, and the explanation of the remaining architectural gap.

## Build sequence

### Reconstruct the Marlin contract {#marlin-contract}

Trace the whole Marlin pipeline at the pin, once per dense quantization
format. Trace the MoE path the same way. The dense
quantization formats at the pin are the Marlin variants of GPTQ and of AWQ.
The stages, in the order the code executes them:

```
weight loading -> repack -> kernel selection -> custom op -> kernel -> tile configuration -> output
``` Record
activation types, quant formats, group sizes, zero-point modes, activation
ordering, weight layouts, dimension requirements, accumulation type, output
semantics, workspace, dispatch conditions, configuration-table semantics, and
CUDA graph requirements. The write-up lands in this folder as
`marlin-contract.md` beside the traced paths.

- verify: attested operator
- inputs: software/vllm/main/vllm/model_executor/kernels/linear

### Characterize the TU102 microarchitecture {#tu102-characterization}

Build the microbenchmark suite under `turing_lab/` in the fork worktree, lock
clocks (done: 1455 MHz), and measure the resources Marlin actually exercises.
The INT4 investigation comes first among the unknowns: establish whether the
relevant s4 MMA shape exists on Turing, its throughput, its operand
requirements, and its interaction with HMMA. Measure the memory path and the
contention mixes (HMMA against HFMA2, FFMA, integer unpack, LDSM, LDG, STS);
the hypothesis that FP16 arithmetic contends with HMMA on a shared execution
resource is to be validated, not assumed.

- verify: cd software/vllm/sm75-marlin && python3 turing_lab/run_all.py --smoke
- depends: #marlin-contract

### Model the Turing Marlin regimes {#performance-model}

From the measurements, construct the performance model for the Marlin
workload: the FP16 and INT8 ceilings, the shared-memory datapath ceiling, the
measured DRAM bandwidth, and the W4A16 compute/memory crossover (the supplied
analysis predicts M near 44 at 1455 MHz; treat it as the initial hypothesis
for regime separation, never as a constant). Record model error per contended
mix; the model is a filter, not the final arbiter.

- verify: attested operator
- depends: #tu102-characterization

### Build the reference Turing backend {#reference-backend}

The first production implementation is deliberately simple: correct Marlin
semantics, existing Marlin weight representation where possible, correct
sm_75 instructions, arbitrary supported dimensions, complete numerical test
coverage, functional vLLM dispatch. Establish the reference kernel, its
correctness oracle, and its performance baseline before any automated
optimization.

- verify: attested operator
- depends: #marlin-contract

### Integrate the dense path {#dense-integration}

Fit the backend into the current kernel-selection architecture through the
mixed-precision linear kernel abstraction (verify the exact paths against the
pin before implementing): advertise minimum capability sm_75, reject
unsupported configurations cleanly, participate in normal kernel selection,
provide weight processing, use the appropriate custom op, and leave sm_80+
dispatch untouched. The custom op must satisfy the V1 runtime requirements:
registered fake/meta implementation, no data-dependent shape allocation,
caller-provided workspace, no host synchronization in the launch path, and
CUDA graph compatibility.

- verify: attested operator
- depends: #reference-backend

### Search the kernel design space {#kernel-search}

Run the constrained search once the baseline is correct: weight staging
(global to registers, through shared, or hybrid), pipeline stages, MMA
decomposition against actual Turing MMA shapes, dequantization placement
(scale in FP16 or in the accumulator domain), warp organization, tile
geometry, and swizzling. Every timed kernel passes the static SASS gate before its numbers count: the
gate inspects the compiled SASS for the intended sm_75 instructions, and it
fails the candidate on unintended conversions, register spills, excess
barriers, or compiler fallback paths. The adversary searches small M,
large M, misalignment, awkward aspect ratios, group boundaries, and unusual
valid quantization configurations; selection is by worst-case regression over
the adversarial set, then validated on held-out shapes.

- verify: attested operator
- depends: #reference-backend, #performance-model

### Generate the configuration table {#config-table}

Generate the Turing-specific configuration space (thread_m_blocks, thread_k,
thread_n, num_threads, stages) under register, shared-memory, occupancy, and
operand-shape constraints, from measured results. Never port the Ampere table
by hand.

- verify: attested operator
- depends: #kernel-search

### Validate graphs and runtime {#runtime-validation}

On the integrated dense path, compare eager, graph capture, and graph replay;
verify no allocation in the launch path, no host synchronization, stable
workspace, stable shapes, correct replay, and no graph-specific numerical
differences.

- verify: attested operator
- depends: #dense-integration

### Integrate Marlin MoE {#moe-integration}

Adapt the validated Turing GEMM machinery to the Marlin expert abstraction:
preserve expert weight semantics and grouped/batched GEMM semantics, extend
the existing backend selection rather than bypassing it, and repeat numerical
and performance validation with expert shapes added to the adversary.

- verify: attested operator
- depends: #dense-integration

### Accept and report {#acceptance}

Accept only when all four hold: interop with the existing Marlin ecosystem,
no known numerical regressions, demonstrated improvement on the targeted
regimes with no unacceptable worst-case regression against the incumbent
sm_75 path, and Turing-specific assumptions isolated behind the backend
boundary. Produce the final report. It includes the rope sm_86 reference runs and the
split of the remaining gap into unavoidable hardware limits and recoverable
implementation margin.

- verify: attested operator
- depends: #runtime-validation, #moe-integration, #config-table
