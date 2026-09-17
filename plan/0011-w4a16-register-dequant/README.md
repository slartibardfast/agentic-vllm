# W4A16 register-dequant + repack: the GEMM gap front

The second front of the operator's two-front directive (2026-09-15):
after the decode-parity front (plan/0010) resolves, this milestone
attacks the largest quantified gap in the program - the fork's W4A16
kernel at 23.50 TFLOP/s (M512 seated baseline) versus the incumbent
Marlin's 51-57 on the same silicon, same protocol. The falsification
ledger is unambiguous about why: the fork's kernels stage dequantized
fp16 tiles through shared memory beside the MMA stream, while the
incumbent keeps B fragments in registers with the skip_flop lop3
dequant whose nibble interleave is baked into its repack layout. The
survivor surgery, scoped in the 72h window: register-dequant plus
repack, days-grade, its own window.

Discipline: the standing gate train (standalone compile, the fp64
oracle, the seated weco acceptance eval at
`turing_lab/results/w4a16-mregime-baseline/`, then a fully-worked
paired committed A/B versus the incumbent numbers under the same
median-of-20 locked-clocks protocol). Gate-1 equivalents: the
kernel-search bench harness at the M sweep. Negative results are
terminal states with mechanism.

## Build sequence

### Resolve the repack interleave contract {#repack-study}

Close the fragment-pairing question the 2026-08-28 probes left: the
m16n8k8 B fragment consumes weights with the k,k+1 nibble pairing
baked into the incumbent's repack (byte_perm interleave). Produce the
exact interleave function from the incumbent's repack source at the
pin, validate it against the fragment probes in
`turing_lab/probes/`, and write the contract into
`plan/0002-turing-marlin-backend/marlin-contract.md` (the layout
table the new kernel's repack must emit).

- depends: (none)
- verify: attested operator
- inputs: software/vllm/sm75-marlin/plan/0002-turing-marlin-backend/marlin-contract.md

### Author the register-dequant kernel {#kernel-author}

The k_opt3-class kernel: consume the repacked layout, dequant in
registers (two lop3 per fragment, scales premultiplied at load),
double-buffered staging, BN128 W8 form. CPU-authored against the
contract; standalone compile clean before any GPU hour.

- depends: #repack-study
- verify: cd software/vllm/sm75-marlin && PATH=/opt/cuda/bin:$PATH nvcc -arch=sm_75 -O3 -c turing_lab/turing_w4a16_regdeq.cu -o /dev/null
- inputs: software/vllm/sm75-marlin/turing_lab/turing_w4a16_regdeq.cu

### Oracle and seat the challenger {#oracle-bench}

fp64 oracle over the seven-case battery, then the kernel-search bench
across the M sweep (median-of-20, locked clocks, N=K=4096 and the
model shapes). The number that matters: M512 against 23.50 (self)
and 51-57 (incumbent). Gate-1 kill line: no M-regime win, no
committed A/B.

- depends: #kernel-author
- verify: attested operator

### The committed paired A/B {#gemm-ab}

The fully-worked paired protocol: challenger versus incumbent
Marlin, both fresh same-day, median-of-20 per M point, locked clocks,
correctness-checked against float64. A WIN at the failing regime
(M>=128) with no regression elsewhere moves the dispatch table
through the ledger; anything less is recorded with mechanism.

- depends: #oracle-bench
- verify: attested operator

### Close out {#close-out}

Ledger and records current, pushes and pins, receipts.

- depends: #gemm-ab
- verify: attested operator

## Execution record (2026-09-16)

- #kernel-author DONE (receipted): 386-line v1, compile rc=0, PTX
  confirms the design (16 mma.sync m16n8k8, 16 lop3, zero
  cp.async/shfl/ldmatrix). Consumes the incumbent repacked layout
  directly (checkpoint-compatible - the engine's existing repack
  feeds it). Design notes: hsub2/hfma2 dequant chosen over the
  precomputed-addend form on dequant.h's own accuracy warning; the
  per-warp n8 partition and direct-half2 A loads inherited from the
  fork's validated opt lineage.
- #oracle-bench DONE (receipted): the battery passed 6/6 at FIRST
  GPU CONTACT - the term-verified contract produced an exact kernel
  with zero numerics debugging. The M sweep measured M512 at
  17.04 TFLOP/s: below the fork's seated 23.50 and the incumbent's
  51-57, so the kill-line holds and no committed A/B runs for v1.
  One falsification on the way: the author's flagged A-staging
  store conflict was treated with a pad change (+8 to +4 halves)
  that cost 26 pct at M512 (the +8 pad's row-start spread was
  buying the read side) - reverted, documented in-source.
  #gemm-ab stays pending a v2 that clears the kill-line; the v2
  lever list is the incumbent's own remaining tricks: multi-stage
  pipeline depth, BN128-class wider tiles, split-K for small M.

## v2 iteration record (2026-09-17)

The v2 levers ran the full train and BOTH falsified at the measured
shapes: the 3-stage ring at BN64 is numerically correct (battery
all-pass) but -49 pct at M512 (8.71 TFLOP/s; 2 CTAs/SM costs more
than prefetch depth buys at this class on sm_75); the BN128 shape
carried a scale-pairing bug (the derived stored-half mapping omits
the plus-8U term for warp blocks past the first 64 columns) AND
runs 1 CTA/SM. The tree defaults now select the measured-best
BN64/S2 (v1-equivalent, 17.04 M512); the falsified shapes remain
compile-time selectable. NEXT NAMED LEVER (v3): thread_m_blocks>1 -
the incumbent's large-M config class amortizes each A/B byte across
multiple 16-row M tiles; that is the structural difference the
falsification ledger has pointed at since August. Lane commit: the
v2 template + falsification record in-source.

## v3 record (2026-09-17): the paired A/B ran — SPLIT verdict

The bounded fix (the permute is per-64-block: absolute stored half
64U + 8c + 2w'; v2 dropped 64U and used w for w&3) unlocked BN128/S2:
oracle 6/6 with errors identical to v1 (exact numerics through the
fix), and the M sweep jumped to 28.63 TFLOP/s at M512 (+68 pct over
v1, clearing the 23.50 kill-line).

The fully-worked paired committed A/B (both arms fresh same-day,
locked clocks, median-of-20, correctness-gated) returned a SPLIT:
- small-M regime FLIPPED TO THE FORK: M1 +30 pct, M8 +29 pct, M32
  +41 pct — and M<=32 is the serving decode shape;
- the incumbent keeps large-M decisively (52.59 vs 28.63 at M512,
  2x — its deep pipeline owns the compute-bound regime);
- the plan's original acceptance (a WIN at M>=128) is NOT met,
  recorded as such.

DISPOSITION: the regime map is the deliverable. The named follow-on:
engine-side dispatch (regdeq for the decode regime M<64, incumbent
Marlin at M>=64) — engine-lane work, its own task; the large-M gap
stays open for a pipeline-depth generation. Lane 8a42950ba5.
