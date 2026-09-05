# plan/0006 retrospective

Written 2026-09-05, after the plan closed with its correctness gate
green. This is the operator-facing postmortem of the delivery arc: what
was actually built, what the evidence says, the two retracted findings,
the one real defect the gates caught, the numerics trade we knowingly
carry, and the process casualties. Numbers are quoted from the
committed records (vllm-engine-run.md, vllm-correctness-gate.md,
flashinfer-sm75-validation.md, gate-logs/, the quilt checker logs).

## What the arc delivered

The plan opened with the bridge primitives (cu_sm80_on_sm75), a
bridge-native attention forward, and quilts routing flash-attn and
FlashInfer at it. The endgame added the piece that turned out to be
missing from the acceptance wording: the vLLM engine itself had no path
to the bridge. On capability (7,5) the V1 selector auto-selects
TRITON_ATTN — FLASH_ATTN and FLASHINFER are gated >= (8,0), and the
FLASH_ATTN backend imports the vendored vllm_flash_attn, not the
quilt-patched upstream package. The recorded 24.2 tok/s engine run had
been running Triton attention all along; a committed validation doc
even claimed otherwise and had to be corrected. Closing the loop
required real wiring: quilt v3 (bottom-right causal via a kernel q0
parameter, causal zero-padding to 64-row tiles) and a
BridgeAttentionBackend — the plan's own "thin dispatch adapter" rule,
design rule 3 — which auto-selects on sm_75 and calls the real upstream
flash_attn_func per layer. Evidence: the engine log selects
BRIDGE_ATTN, and the committed profiler trace shows
bridge_fwd_kernel<128> firing 224 times in a profiled window — exactly
28 layers x 8 decode steps.

## The honest scoreboard

Single-card and TP=2 A/B on Qwen2.5-1.5B (head_dim 128, GQA), identical
protocol both arms:

| workload | BRIDGE | TRITON | ratio |
|---|---|---|---|
| 1 card, 4x64 tok short | 56.1 tok/s | 124.5 tok/s | 0.45 |
| 1 card, ctx 512 -> 32 tok | 30.1 tok/s | 42.6 tok/s | 0.71 |
| 1 card, ctx 2048 -> 32 tok | 28.0 tok/s | 28.0 tok/s | 1.00 |
| TP=2, 8x64 tok short | 24.2 tok/s | 212.4 tok/s | 0.11 |
| TP=2, ctx 512 -> 32 tok | 19.0 tok/s | 41.4 tok/s | 0.46 |
| TP=2, ctx 2048 -> 32 tok | 14.4 tok/s | 33.1 tok/s | 0.44 |

Correctness gate (numerics before timing; wikitext-2 test PPL via
prompt_logprobs): bridge 9.5012 vs triton 9.5009 on 1.5B (0.003%),
13.4709 vs 13.4700 on 0.5B (0.007%); chunked prefill and prefix
caching bit-identical to the unchunked run (the q0 workout); the 27B
hybrid stack (marlin W4A16 + 48 linear-attention layers + TRITON
full-attention, head_dim 256 by scope) coherent to ctx 32768, PPL
6.7763, stable 512-token decode; TP1 vs TP2 greedy identity 3/3 at ctx
512/2048/8192.

The speed picture is a statement about the backend's architecture, not
the kernel: per-(layer, request) python dispatch plus paged-KV gather
copies outnumbered the real bridge kernels ~8:1 in kernel counts in the
trace (1372 direct-copy + 448 index-put vs 224 bridge kernels), and
decode pays a zero-padded 64-row Q tile. Parity at ctx 2048 single-card
shows the kernel itself is not the short-context bottleneck. The fix
list is known and recorded: a paged/varlen bridge kernel consuming
block tables directly (removes gather and python loop in one stroke),
d=256 tiles (32-row KV + Q-in-registers restructuring; until then the
27B declines to TRITON by design), ldmatrix fragment loads for the
kernel's own 16.3 vs 32.1 TFLOP/s gap (d=128 single-buffered at 10.6).

## Two retracted findings — both were ours

1. "FlashInfer 0.6.18 multi-request decode faults on compute_75"
   (b>=2, illegal address, sanitizer OOB reads) — RETRACTED. Our
   harness passed (b, 2, kvlen, h, d) against the paged-cache contract
   (num_pages, 2, page_size, h, d); at b>=2 the kernel indexed past
   the too-small page dimension. With the contract-correct layout,
   batch decode passes b up to 16, mixed lengths, both kernel variants
   (max_err <= 0.0007). Nothing was filed upstream; the "vLLM must not
   use FlashInfer decode on sm_75" consequence line was also wrong and
   was corrected. The recorded throughput numbers survived (the kernel
   had processed the same KV volume).
2. The earlier "reference turing_w4a16_pipe NaNs at K=4096" retraction
   (a scale-tensor shape artifact) set the precedent: when a fault
   claim rests on an ad-hoc cross-check, audit the cross-check first.

Pattern worth keeping: both retractions were harness bugs that looked
like kernel bugs because the harness passed on the "easy" shape and
failed on the "hard" one. The asymmetry felt like evidence; it was
confounding.

## The one real defect: fp16 saturation in the softmax

The correctness gate's first bridge arm failed immediately — all
logprobs non-finite, degenerate "locklocklock..." output. Root cause
(instrumented-kernel bisection): the online-softmax exponent path
packed the RAW (unscaled) S and row-max m into fp16 BEFORE subtracting.
Real activations produce raw dots far past fp16 range — observed
S_raw = 110,575 against a 65,504 ceiling (the scaled score was a
comfortable 9,774; the engine tensors also showed k absmax 319, q 44.8)
— so the packed values saturated to inf and inf - inf = NaN, poisoning
layer 0 and everything after it.

The fix computes (S - m) * scale * log2e in fp32 and converts to half
only for the vectorized ex2.approx.f16x2; the quantity is <= 0, so
negative overflow flushes to -inf and exp2 gives 0 — safe by
construction.

Why every prior gate was green is the lesson: the C++ oracle used
random +-1 inputs (raw dots ~ tens), the route oracle the same, and the
backend unit tests were self-confirming on top — their fp64 reference
consumed the SAME gather the backend used, so a gather bug (a missing
permute, caught later by a test that compared against raw tensors) and
the magnitude blindness both passed vacuously. Random-unit inputs test
layout and logic, not value distributions. The fix is now regression-
locked three ways: in_scale=100 magnitude cases in fwd_oracle.cu (raw
dots ~1e5), the route-level checks, and the engine gate itself.

Corroborating curve: oracle max_err grows 0.0003 (unit inputs) ->
0.035 (x100 inputs) with score spread, exactly as an fp16-P mechanism
predicts.

## The slightly-worse PPL, named

The bridge sits 0.003-0.007% above TRITON in PPL on both dense models.
Since the arms differ only in attention, the delta is attention
numerics, and it is the accuracy cost of a speed lever we pulled
deliberately: the h2exp2 half2-vectorized softmax (part of the
15.2 -> 16.3 TFLOP/s tuning win) quantizes the exponent argument to
fp16 and uses the approximate hardware exp2, where TRITON computes
softmax in fp32 throughout. Round-to-nearest on the exponent is
monotone, so this is not noise but a slightly warped score vector —
softmax of a deterministic distortion — which is about the most benign
failure shape available: attention output is a convex combination of V
rows, the PV numerator and O/l denominator are built from the same
quantized P (partial cancellation), and downstream layernorms absorb
drift. PPL simply integrates log-prob over every token, so it resolves
differences long before generation quality would. Against the error
budget it is also dominated: int4 weight quantization carries orders
of magnitude more error than this on the model where numbers matter
most.

We sit ~100x inside the 0.5% cross-backend gate. The dial is explicit
if the margin is ever wanted back: fp32 exponent (or fp32-exp-then-
round) at a measured slice of the tuning win. The more interesting
version, parked for the auto-research push: treat it as a searchable
axis — max TFLOP/s subject to cross-backend PPL within epsilon — which
turns the speed/accuracy knee into a measured point instead of a
judgment call. Caveat for that experiment: 16 wikitext windows give a
~0.5% per-window drift band, so the harness's interleaved A/B
discipline is load-bearing.

## Operational findings

- Scheduler boundary wedge: an 8192-token prompt exactly equal to
  max_num_batched_tokens wedged the single-card run (no room for the
  decode step); chunking at 4096 resolved it. Worth an upstream look.
- Engine run outputs should be validated, not just counted: every
  throughput run before the gate used ignore_eos and never looked at
  the text. The gate's coherence/distinct-3gram/logprob-finite checks
  are cheap and caught a kernel bug in the first minutes.
- Measurement variance: the same 27B single-card short-context
  protocol produced 24.2 and 47.9 tok/s on different days (JIT warmup
  amortization suspected). Within-session A/B pairs, not cross-session
  absolute numbers, are the currency.
- Flash-attn's own dormant sub-80 path still faults on tiny non-causal
  shapes (sq=2): irrelevant to our route (it declines those shapes)
  but a reminder that the code we replaced was genuinely broken.

## Process casualties, honestly

- A stray `git checkout -- .` in the fork (verifying pre-existing test
  failures) silently reverted uncommitted lab files — kernel q0 edits,
  oracle, bench. The clone/quilt tree the engine actually uses was
  never affected, but an hour of debugging ran against a stale header
  before the mismatch was noticed. Lesson: materialize work into the
  tree the build consumes (or commit) before destructive git commands;
  the lab-vs-clone split is a footgun the lane layout now makes
  avoidable.
- The backend unit tests initially passed vacuously (reference shared
  the implementation's gather). Fixed by adding a test that compares
  against raw pre-gather tensors.
- Minor: intermittent tool-gate flapping blocked mid-task edits and
  cost several retries; a gh account mismatch (slartibardfast active,
  fork under connollydavid) surfaced only at push time.

## Where this leaves the program

plan/0006's acceptance is met on evidence: quilts compile and pass
oracles; the engine runs through the bridge with committed tokens/s,
context scaling, and profiler proof against the prior engine
attention; upstream tracking is one command, green; every quilt patch
is a primitive substitution. The correctness gate — a deliberate
numerics-before-timing pause — validated the stack end to end and paid
for itself in the first arm. Backlog is recorded, not hidden: the
architectural backend speedup, d=256 support, ldmatrix loads, and the
dual-objective (perf under PPL-epsilon) auto-research axis that would
put the speed/accuracy knee under measurement.
