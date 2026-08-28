# Gap closure: repack interleave, deep pipeline, fp32 split-K, LLM proposer

Plan/0004. The goal this milestone exists to hit: **the emitted dispatch
table must statistically beat the first-generation winners in at least
one M regime, with no regime regressing** — measured by the plan/0003
harness's own winner gates (`max(5%, 3 sigma)`, oracle-clean,
clock-attested). Today the gates correctly retain the first-generation
baselines everywhere; the D8 gap decomposition names exactly what is
missing.

## Goal 1 — swizzled staging for regdequant (IMPLEMENTED as `regdeq2`)

The incumbent's weight ordering lets the skip-flop dequant run without
per-fragment byte_perm shuffles. Rather than a host-side repack, the
permutation is hoisted into the kernel's staging loop: four lane-swizzled
words are precomputed per packed word into shared memory
(`sQW[BN][BK/8][4]`), so the mma stream is lop3 + hsub2 only. Cost: 2x
the scale smem of regdequant (25.6 KiB at 64x128x64, still >=2 blocks).

- Implemented as strategy `regdeq2` in turing_search.cu; 8 tile rows;
  oracle-clean across the table.
- Accept when regdeq2 beats the staged baseline beyond the gate margin
  in at least one regime (final ladder run decides).

## Goal 2 — deep software pipeline

Multi-stage overlap per the D8 decomposition: register-resident
dequantized fragments, staging for chunk k+1 overlapped with compute on
chunk k, beyond the current 2-stage pipe (which the reference kernel's
NaN-at-K=4096 defect shows is not yet trustworthy either — root-cause
that defect first, it is the same family).

- RESOLVED before this milestone opened: the "reference pipe NaN at
  K=4096" was a test artifact (1-row scale tensor with G=128); the
  reference is clean with proper scales. Remaining here: add a 3+ stage
  variant to the ladder via `propose.py`'s row interface (strategy
  `pipe3`), gated identically.

## Goal 3 — fp32 split-K partials (unlocks kshard2)

kshard2 failed its oracle on fp16 partial precision. Add an fp32-partial
epilogue variant (the split-K machinery already writes fp32 internally;
the gap is the single-GPU K-shard path returning fp16). Re-measure
cross-GPU K-shard over the NVLink path (43.7 GB/s) — with 2 GPUs the
compute doubles and the partial transfer for M<=64 is tiny, so small-M
latency is where this can win.

## Goal 4 — LLM proposer hook

`propose.py`'s row interface is the contract: any source of legal rows
rides the identical gates. Add an optional proposer backend (env-gated
API key) that prompts for kernel-source mutations, compiles them
through the same ptxas/SASS/oracle/timing gates, and only folds rows
that beat their parents. Failure modes (bad compile, oracle fail) are
data, not errors.

## Standing rules (carried from plan/0003)

- Every kernel change: 2x deterministic oracle passes + SASS gate +
  occupancy legality before any timing counts.
- The emitted dispatch table can never regress below the current one
  (baseline seeding).
- Absolute numbers at small M come from marlin_bench-style measurement,
  not the selection harness (constant per-launch overhead cancels in
  selection but not in absolute latency claims).

## Acceptance

1. Ladder full-fidelity run with the new space; emitted
   `dispatch-table-*.json` beats the first-gen baseline beyond the gate
   margin in >= 1 regime, no regime regressed (gate transcripts
   committed).
2. Reference pipe NaN root-caused and fixed, or its defect boundary
   precisely documented.
3. kshard2 re-judged on fp32 partials with transport evidence.
4. vLLM tests green (selection, e2e, dispatch-table).
5. Artifacts committed; fork tip pinned here as plan/0003 was.
