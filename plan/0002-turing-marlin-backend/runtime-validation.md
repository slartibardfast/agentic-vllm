# Runtime and CUDA-graph validation

Runtime validation of the pipelined packed-staging kernel
(`turing_w4a16_pipe.cu` exposed as `w4a16_opt2`), run on TU102 at locked
1455 MHz.

- Eager vs reference: all oracle cases pass (see reference-backend.md).
- CUDA graph capture: the kernel launches with no host-side allocation in
  the launch path (preallocated output), so capture succeeds on the first
  attempt without warmup reallocation.
- The graph replay after an activation mutation reproduces the updated
  reference within fp16 tolerance, so the graph recomputes rather than
  replaying stale results.
- No host synchronization in the launch path; stable shapes; stable
  workspace.

Status: the runtime lane passes. The remaining campaign items (multi-shape
graph re-capture, MoE dispatch path) are recorded in the build sequence.
