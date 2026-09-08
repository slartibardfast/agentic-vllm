# Research consolidation (2026-09-08)

The single front door to this host's research material. Read this
before the plan READMEs when planning. Organized by topic, not by run:
each topic carries the FINAL truth (post-corrections), where it lives
authoritatively, and what decision it feeds. Chronological sources
(MEMORY, run logs) remain the audit trail; this file is the synthesis.

## Material map

| Artifact | Holds | Status |
|---|---|---|
| research/lacunae/ (20 JSONs + SUMMARY.md) | the future-work corpus: engine, kernels, formats, strategy; every claim source-cited | complete, validated 20/20 |
| research/headshape/ (10 JSONs) | head-dim landscape, Qwen3.5 architecture, kernel coverage, route choice | complete; consumed by plan/0007 |
| plan/0006-cu-sm80-on-sm75/research.md | PTX ISA inventory for sm_75 (mma shapes, ldmatrix, cp.async absence) | complete; empirical, this host |
| plan/0002 tu102-characterization.md | measured silicon envelope + levers/red-lines section | complete; THE silicon authority |
| plan/0002..0007 READMEs | campaign records (kernels, gates, bisects) | authoritative per plan |
| turing_lab/results/restart-variance-20260907/ | probe harness, REPORT (root cause), provenance patches | complete through window close |
| turing_lab/results/w4a16-mregime-baseline/ | weco acceptance eval + seated 23.50 baseline | complete, awaiting session |
| turing_lab/thirdparty/NEXT-WINDOW-RUNBOOK.md | the turnkey window sequence + floor derivation | current |
| MEMORY.md | chronological findings log | audit trail, not authoritative |
| call/0002, call/0004 | decisions (repro waiver; template upgrade) | standing |

## Topic consolidation

**T1 Engine and TP2 serving.** Final truth: the tp2 decode swing
(12-17 tok/s) was recorded on the 1.5B gate model; on the
same-dimension 4B hybrid the TRITON control shows a restart-stable
SIGNAL at n=2 restarts (0.8-1.2 percent bands; the methodology floor
is n>=3, so preliminary, not settled); the bridge arm is unmeasured in-engine (the
window closed on a CUDA_HOME residual; fix applied, driver turnkey).
The d256 engine path had broken post-campaign (staged half-revert in
the vendored FA2 tree + load-bearing kernel API living uncommitted);
root-caused, restored, kernel edits committed (lane f4b708b122),
provenance archived. Upstream facts: v0.23.0 last-good FlashInfer-sm75;
the v0.24.0 drop deliberate (#45375); restore PR #55380 open, tested on
the Qwen3.8-27B class; FlashInfer sm75 prefill broken on a 64 KiB smem
overflow (#3620) - the bridge kernel fills that hole. Rebase verdicts:
dense REBASE, MoE REBASE+absorb, attention DIVERGE (one floor constant,
weicj-proven one generation up). Feeds: the 27B variance verdict and
the both-card committed run (runbook).

**T2 Attention kernels and the bridge.** d=256 delivered and validated
(plan/0007: oracle, PPL identity 6.7761/6.7763, both ranks); k-chunked
route B was the right call (headshape corpus); llama.cpp's Turing
config tables prove (512,512)/(576,512) if d512 ever matters (de-weight:
Gemma 4's problem). Register wall live: d256 sits at 255/254 regs.
Partial RoPE (25 percent) + q/k RMSNorm + swish gate are contract
items no kernel may fold away. Softcap carried through the shim
(FA2-tree commit 9eb4012). GDN layers: stock FLA Triton path is the
only maintained route inside the vLLM/FLA ecosystem (llama.cpp-family
serves GDN on CUDA independently); FlashQLA sm75 fork = +0.61 percent
decode e2e (honest ceiling); FLA 0.4.2 is the pin (post-NaN-fix).

**T3 W4A16 kernels and acceptance.** Final truth (correction ledger):
the incumbent leads at every measured M; the retracted claim that the
fork wins M 1-32 was wrong (it beat first-gen baselines only). Gap:
ours 23.50 (opt2, seated baseline: this protocol, median-of-20,
locked clocks, N=K=4096) vs incumbent 51-57 (kernel-search.md sweep:
same shape, same median-of-20, same locked clocks - protocols
comparable; the seated eval independently reproduced the recorded
opt1 23.49 to within 0.01). Failing regime M>=128 only. Live upstream
contradiction on record: the Marlin-Turing PR author claims f32
accumulate halves throughput on Turing; our paper measured it free at
the 101.7 ceiling - unresolved, flagged uncertain in the corpus. Attack levers: swizzled regdeq2 staging, repack
interlace, split-K small-M, deeper pipeline; legality = TU102 red-lines.
weco session staged (.weco/w4a16-mregime + committed baseline record).

**T4 Formats.** AutoRound W4 sym g128 is the lane (Intel's own pattern;
g64 only at W2/W3); upstream #52890 (Humming, SM75+) added 2/3/5/6/7-bit
2026-09-07 - build-vs-adopt probe pending; both 27B targets ship bf16 so
recast-at-load is a prerequisite (weights outliers expected zero;
activation space is the real, runtime risk). MXFP4 Marlin is closed to
Turing on principle (bf16 activations) - the CUDA nibble-LUT is the only
tensor-core vehicle. NVFP4 floors at 75, post-#34577 Turing status
unverified (gate-0 probe, falsifiable PPL target). INT8 W8A8 is native
and unused (203 TOPS measured; the only unexploited tensor rate).
Landmine: #48905 (W4A8 negative-group-scale corruption).

**T5 Speculative decode.** MTP is family-native (Qwen3.8-27B recipe
mtp K=3, acceptance 0.75-0.90; Flash-Next ships a native MTP head);
only Turing datapoint: weicj 1.39x e2e, decay 87/68/51 percent at
K=1/3/5. Verdict: MTP3-first + ngram, engine-level, 2-3 sessions;
EAGLE-3/P-EAGLE out (no matched drafts). Fork gates: q_len=K+1 decode
attention (bridge kernels are q=1 instantiations) and GDN state
rollback; #53180 canary mandatory for compressed-KV x MTP.

**T6 KV capacity and preemption.** Envelope (27B hybrid geometry):
64 KiB/token fp16 -> 16K tokens/GiB; K8V4 1.1-1.3M tokens on the pair;
GDN prices 144 MiB/seq flat. V1 is recompute-only: zero-preemption
admission control; PREEMPTED = gate failure. TurboQuant-KV x MTP
silently corrupts on this model class (#53180, stock v0.27.1).

**T7 Model family facts.** Qwen3.5/3.6/3.8 share the shape: full
attention d=256, partial RoPE 0.25, GQA (24Q/4KV on 27B; 16Q/4KV on
4B), GDN 128/128, interval 4 (3:1). The 4B is the smallest same-KV-
geometry member (verified from config.json) and is now a standing
fixture (W4A16, g128, fp16, seed 42, fixture-grade).

**T8 Silicon facts.** TU102 paper is THE authority (measured, locked
clocks): 101.7 fp16-tensor ceiling; INT8 exactly 2x at 203 TOPS; no
s4 MMA usable for W4A16 (reconciled nuance: plan/0006 found the tiny
m8n8k32 .s4 accepted by ptxas with unresolved arity - INT activations
only, irrelevant to W4A16); k8 wall; two-resident-block occupancy law;
conflicted STS halves; 535 GB/s; M*=47 crossover (the paper's own label: a model output
and initial hypothesis, not a constant). Red-lines are gates, not
knobs.

**T9 Tooling, methodology, prior art.** weco lane: observe/adopt-only
discipline, eval seated. Org prior art: sinter (PAUSED - reference
baseline; its DRAM-floor tie proves the floor reachable on this
SILICON by a llama.cpp megakernel, not yet by our vLLM path; roadmap items
not dependencies), calx-mill (the performance-model instrument; its
TU102 adapter's next lane is exactly plan/0002#performance-model),
gguf-recast (the recast prior art), calx-telltale (provenance-typed
discipline). Research skills lane: 5 skills, 5 modules, international
endpoint this run; validated corpus discipline (100 percent coverage
gates). Host methodology: engines v0.53.0/0.19.0, verify battery green.

**T10 Deprioritized or blocked (with the reason on record).** MLA
tuples (other families; cheap gate-0 when one enters the mix); d512
(Gemma); KTransformers (hardware-blocked, closed by citation); Triton
3.2 pin (dead end for this family - predates Qwen3-Next; the escape is
triton-turing fork or CUDA-native substitution, probe designed); GDN
CUDA kernel rebuild (deferred - FLA Triton sufficient for now).

## Corrections and retraction ledger (final)

1. Thinking Machines: atomics are NOT the top inference nondeterminism
   driver (batch-size-dependent reduction order is).
2. "Triton never emitted Turing mma": false for 3.2-era (MMAv2 did);
   the loss is #5066 in 3.3.
3. "Fork wins all decode regimes": retracted (incumbent leads at every
   measured M).
4. ik_llama/llama.cpp KV bugs: neither Turing-specific; FlashMLA-2 in
   ik_llama is ikawrakow's own generation.
5. johnscheuer INT8-attention variant: does not exist publicly.
6. KTransformers "3-28x": no such figure (SOSP'25: 4.62-19.74x prefill).
7. The bridge-4B "instantiation gap": RETRACTED - the real cause was
   the staged FA2 half-revert plus uncommitted load-bearing kernel API.
8. Aliyun transistor count (18.6 yi): tenfold transcription error;
   spec-grade numbers anchor to whitepaper/Hot Chips only.

## Uncertainty register (open, named)

- Bridge arm in-engine variance: unmeasured (next window, first item).
- Humming-on-TU102 throughput: unpublished anywhere (build-vs-adopt).
- NVFP4 post-#34577 on real Turing: unverified (gate-0, PPL target).
- The m8n8k32 .s4 arity micro-question (plan/0006; W4A8-IMMA corner).
- FA2 #55380 merge state; weicj v0.2.x CUDA-graph TP2 reproducibility
  on our pair (their numbers are best-case, no restart discipline).
- 120 per-item [uncertain] markers live in the lacunae JSONs
  (verified by script; the 49 first written here and in SUMMARY.md
  was fabricated precision, corrected 2026-09-08).

## Pending decisions and the evidence each needs

| Decision | Blocking evidence |
|---|---|
| 27B both-card committed number | probe + 27B bridge verdict (runbook) |
| W2/W3 execution: build vs adopt (Humming) | Humming-on-TU102 gate-0 bench |
| MXFP4 nibble-LUT build vs skip | NVFP4 gate-0 first (cheaper, same LUT core) |
| MTP3 session start | variance verdict (schedule) + weco login |
| Kernel-lane budget for the 51-57 gap | weco session outcome |
| Rebase timing onto v0.28-era upstream | #55380 merge or local cherry-pick decision |

## Deduplication notes

The variance story exists in five places; authoritative = the probe
REPORT.md + this file. The architecture facts exist in three; the
headshape deep-dive JSON is authoritative, the plan/0007 summary
derivative. MEMORY entries yield to committed reports EXCEPT dated
MEMORY corrections, which override the specific entries they
invalidate (the 2026-08-28 kmap-probe correction versus the committed
turing_lab/probes record is the standing example). Where the lacunae corpus corrected an input, the corpus wins
over the outline descriptions.
