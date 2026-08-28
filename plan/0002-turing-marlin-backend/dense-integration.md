# Dense integration (Turing backend in vLLM kernel selection)

The Turing backend is wired into the hosted fork's W4A16 kernel-selection
architecture. Code: `vllm/model_executor/kernels/linear/mixed_precision/
turing_marlin.py` on fork branch `sm75-marlin`; tests:
`tests/kernels/quantization/test_turing_w4a16_selection.py` and
`test_turing_marlin_e2e.py` (run with the lab venv's pytest from the fork
worktree root).

## What was wired

- `TuringMarlinLinearKernel(MarlinLinearKernel)` is registered in
  `_POSSIBLE_KERNELS[CUDA]` ahead of the generic Marlin kernel.
- `get_min_capability` is 75, and `can_implement` additionally requires the
  device's own capability to be exactly sm_75, so the backend wins on TU102
  hardware and falls through to the generic Marlin kernel on sm_80 and
  newer: sm_80+ dispatch is untouched by construction.
- Unsupported quant types are rejected cleanly through the incumbent's
  supported-type query.
- Weight processing and compute currently delegate to the incumbent
  implementation (`gptq_marlin_repack` and `gptq_marlin_gemm`); the
  optimized Turing kernel replaces the custom op behind this same class,
  and the repack swap point is `process_weights_after_loading`.

## Verification (all green on TU102 at the pin)

Re-verified 2026-08-28 in the rebuilt lab venv against installed
`0.28.1rc1.dev50+gcac8a75a7.d20260828.cu133` (see environment.md for the
rebuild recipe): the six tests below pass, selection and e2e.

Selection tests:

```
test_turing_backend_wins_on_sm75                    PASSED
test_turing_backend_selected_by_default_on_this_device PASSED
test_turing_backend_gate_rejects_sm80_and_sm86      PASSED
test_turing_backend_gate_rejects_unknown_capability PASSED
```

End-to-end lifecycle tests (real packed GPTQ parameters, repack, apply, M 1
and 64): the Turing backend's output is bit-identical to the incumbent
kernel's, and both agree with the float64 quantization reference inside
tolerance.

## Notes for the next generation

- The selection tests exercise the gate through the platform capability;
  the caller-supplied `compute_capability` in `choose_mp_linear_kernel`
  only filters min capability and cannot fake a device, which the tests
  document.
- Building the fork from source on this host needs the editable install to
  register package metadata (the CUDA platform probe reads it); an
  in-place `build_ext` alone leaves the platform "Unspecified".
- vLLM at this revision has no plain `_C` module on CUDA: the ops live in
  `_C_stable_libtorch`, and the `_C` import warning seen at startup is
  legacy-path noise.
