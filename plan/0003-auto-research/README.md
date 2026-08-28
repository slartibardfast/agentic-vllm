# Auto-research harness (SToA search, transports, dispatch table)

Plan/0003. Extends the turing_lab search framework with state-of-the-art
auto-research layers and makes the Turing backend consume the result.
Replanned mid-course at the operator's direction: all three transport
modes (single GPU, PCIe, NVLink) are in scope, and the Bayesian and
generative layers were pulled forward from plan/0004 into this milestone.

## What was built

1. **Candidate space** (`search/turing_search.cu`, `variants.def`):
   three strategies x tiles, runtime split-K —
   - `staged`: weights dequantized in shared memory (the only
     zero-point-capable strategy; the reference contract);
   - `regdequant`: packed words staged raw, dequant in registers via
     byte_perm + the 0x6400 bias trick, scales applied per chunk in the
     accumulator domain (the incumbent's skip-flop mechanism);
   - `pipe`: per-thread staging registers, double-buffered shared memory;
   - split-K is a runtime grid.z with fp32 partials + reduce.
   The variant table lives in `variants.def`, generated and rewritten by
   the proposal layer; the dispatch is by strategy token-paste, so row
   order can never desynchronize strategy selection.
2. **Analytic pre-filter** (`space.py`): the documented model, enforced —
   shared-memory budget, the >=2-resident-blocks occupancy rule,
   roofline prediction with wave quantization.
3. **Measurement harness** (`harness.py`): CUDA-event medians, per-shape
   noise floors, `max(5%, 3 sigma)` winner gates, interleaved A/B
   against clock drift, clock-lock attestation per session.
4. **Bayesian multi-fidelity scheduler** (`schedule.py`): Optuna TPE
   sampling over a three-rung ladder (2 M points x 3 reps -> full M
   sweep -> full adversarial shape set), promotion/cull of the top
   third, baseline seeding on the first-generation winners, and bisection
   of regime boundaries to granularity 8. Emits versioned dispatch
   tables (`dispatch-table-*.json`) with attestation and gate evidence.
5. **Generative proposal layer** (`propose.py`): stochastic mutation of
   the variant table around the frontier; every proposal rides the same
   gates (legality model -> oracle -> timing -> winner margin). The
   checked-in table grew 20 -> 22 rows this way. This is the hook a
   future LLM-in-the-loop proposer plugs into — any source of legal rows
   rides the identical gates.
6. **Transports and multi-GPU** (`mgpu.cu`, `transport.py`): all three
   modes measured (results below); N-sharded and K-sharded strategies
   composed from the single-GPU variants and timed end to end.
7. **Dispatch consumption** (`turing_marlin.py`): `DispatchTable`
   loads the emitted artifact and maps runtime M -> (variant, split-K);
   env-gated on `VLLM_TURING_DISPATCH_TABLE`, absent the incumbent
   delegation is untouched. Selection tests cover the mapping, fallback,
   and shape-coverage gates.

## Transport results (this host, two Quadro RTX 6000s)

The pair is NVLink-bridged (`NV2` in the topology, 2 x 25.78 GB/s).
Measured (`transport-results.json`):

| path | bandwidth | small-transfer latency |
|---|---|---|
| P2P over NVLink | 43.7 GB/s | 22.8 us |
| staged through host (PCIe x2) | 11.4 GB/s | 19.1 us |

Multi-GPU strategies at N=K=4096 (staged config, oracle-checked):

- **nshard2** (N split, gather halves): 1.32x at M=64, **1.67x at
  M=512** — oracle-clean; the viable multi-GPU strategy.
- **kshard2** (K split, reduce partials): slower at every M and oracle
  FAIL — fp16 partial precision. Recorded as precision-limited; needs
  fp32 partial outputs before it can be judged.

## Verification

- 22-variant table x split-K {1,2,4,8}: **oracle ALL PASS, twice**
  (determinism check) at N=K=4096 M=64 against the float64 reference.
- Full-fidelity ladder run completes; winner gates held the
  first-generation baselines (correct: no new combination beat them
  beyond noise — consistent with D8's finding that regdequant needs the
  repack interleave before it can win).
- vLLM: 10/10 tests green (4 selection, 2 e2e, 4 dispatch-table).

## Defects found and fixed on the way

- `regdequant` port originally applied one slice-group scale to the
  whole slice; corrected to the incumbent's per-chunk partial
  accumulation.
- `pipe` port originally staged 8 of 16 A halves per segment (4 u32
  vs the required 8 u32 at 2-element stride) — a +2 k-skew; and lacked
  guards for tiles where THREADS > 2*min(BM,BN).
- `variant_launch` mapped strategy by index thresholds, which silently
  misdispatched once the proposal layer reordered the table; replaced by
  strategy token-paste.
- Driver treated a zero-element ZP tensor as present (data_ptr of an
  empty tensor is not guaranteed null) — nondeterministic garbage.
- **"Reference `turing_w4a16_pipe.cu` NaNs at K=4096" — RETRACTED.**
  The NaN was an artifact of the ad-hoc cross-check, which fed a
  1-row scale tensor with G=128 (out-of-bounds group reads). With
  proper (K/G, N) scales the reference is clean at K=4096
  (err 0.0033, re-verified 2026-08-28). No kernel defect.

## Deferred (plan/0004 axes)

- fp32 partial outputs for kshard2; repack interleave for regdequant;
  deep software pipeline. LLM-in-the-loop proposal hooks on propose.py's
  row interface.
