# Lacunae run summary (2026-09-07)

Run: 20/20 items complete, each validated at 100% field coverage by the
lane validator. Endpoint: international. Output: results/*.json (one per
item), this file is the index. Focus weighting per the operator:
Qwen3.8 + Qwen3.8-Next. Framing inputs ratified during the run: the weicj
fork as normalized peer yardstick (MEMORY 164290d), sinter paused =
reference baseline (MEMORY 6ccd867), only verifiable numbers enter
records.

## Corrections the run made to its own inputs

- Thinking Machines nondeterminism: atomics are NOT the top drift driver;
  batch-size-dependent reduction order is. Batch-invariant machinery
  upstream requires CC 8.0+ (sm75 excluded).
- "Triton never emitted Turing mma": false for the modern era. Triton
  3.2's MMAv2 emits native m16n8k8 on cc75; the loss is #5066 in 3.3.
- ik_llama.cpp #1142 and llama.cpp #19036 KV bugs: CPU build flag /
  non-contiguous V dequant, fixed, neither Turing-specific.
- johnscheuer INT8-attention variant: does not exist publicly.
- "Fork wins all decode regimes M 1-32": contradicted by host records
  (incumbent leads at every measured M); retracted, probe prescribed.
- KTransformers "3-28x": no such figure in paper or blog (SOSP'25:
  4.62-19.74x prefill / 1.25-4.09x decode).
- MXFP4 Marlin exclusion from Turing is principled: bf16 activations,
  which Turing lacks. The fork's CUDA nibble-LUT is the only tensor-core
  vehicle for MXFP4 on sm75.

## Core under the focus (Qwen3.8 + 3.8-Next)

1. MTP is family-native and the biggest decode lever: Qwen3.8-27B
   recipe mtp K=3 acceptance 0.754-0.897; Flash-Next ships a native MTP
   head; only public Turing numbers are weicj's (43.6 -> 60.6 tok/s,
   1.39x, dual 2080 Ti). Verdict: MTP3-first + ngram, engine-level,
   2-3 sessions; EAGLE-3/P-EAGLE out (no matched drafts). Fork gates:
   q_len=K+1 attention (our bridge kernels are q=1 instantiations) and
   GDN state rollback on rejection.
2. TP2 restart variance (the named next campaign): profiler-free probe
   first (restarts x NCCL init logging x single-variable A/Bs);
   upstream never root-caused the class (#19576 closed not-planned);
   VLLM_BATCH_INVARIANT needs CC8+; fork pin already carries MRV2-default
   whose Triton input prep rides the sm75 FMA floor - a plausible
   amplifier. Probe RUNNING on the 4B same-dims hybrid (this window).
3. Engine base: the v0.24.0 FlashInfer drop was deliberate (PR #45375,
   "revert once fixed"); restore PR #55380 open, tested on dual 2080 Ti
   with the Qwen3.8-27B class; flashinfer #3526 (d256 smem) merged;
   TRITON_ATTN is today the only sm75 candidate. Rebase verdicts: dense
   REBASE, MoE REBASE+absorb, attention DIVERGE (one floor constant,
   weicj-proven one generation up). Do not strand on v0.23.0.
4. AutoRound execution: W4A16 sym g128 Marlin-servable today (Intel
   27B-int4); W2/W3 CUDA arrived upstream 2026-09-07 (#52890, Humming,
   SM75+, 11 days past our pin) - cherry-pick + measure Humming-on-TU102
   before kernel-ladder budget; Qwen3.8-27B-bpw2.8 exists needing #52890;
   both 27B targets ship bf16 -> recast is a prerequisite (weights
   outliers expected zero; activation space is the real risk, runtime
   not load-time). Landmine: #48905 (W4A8 negative-scale corruption).
5. KV envelope (27B hybrid): 64 KiB/token fp16 -> 16K tokens/GiB; K8V4
   1.1-1.3M tokens on the pair; GDN prices 144 MiB/seq flat. Preemption:
   recompute-only in V1 -> zero-preemption admission control is the
   design rule; PREEMPTED event = gate failure. TurboQuant-KV x MTP
   silently corrupts (#53180, stock v0.27.1, this model class) - real-
   prompt repetition canaries mandatory for compressed-KV arms.
6. GDN lane: FlashQLA sm75 fork proves the port (~2.1x kernel, +0.61%
   decode e2e - honest ceiling); FLA 0.4.2 is the pin (post-#792-NaN
   fix); TileLang carries sm75 MMA as a second substrate.

## De-weighted (other families / blocked)

- MLA tuples: shipping model space now (Mistral Small 4 = first 320/256;
  DeepSeek/GLM 576/512) but not this family; triton_mla anticipates
  fp16-KV sm75 with zero public runs - cheap gate-0 when one enters the
  mix. DSA == dense below 2048 tokens.
- d512: Gemma 4's problem; llama.cpp Turing tables already carry
  (512,512)/(576,512) - structurally proven if ever needed.
- KTransformers: hardware-blocked (kt-kernel floors at CC 8.0; host CPU
  is AVX512-base Skylake without VNNI/AMX) - closed by citation;
  llama.cpp -ncmoe is the only sm75-live CPU-expert route (~48-51 tok/s
  estimated partial offload on 35B - loses to GPU-resident TP2).
- Triton escape matrix: v0.8.5 is the last stock 3.2 carrier (drags the
  stack pre-Qwen3.Next); the healthy small-patch is Chennesxu/
  triton-turing (cp.async-free pipeliner, 12-18x bf16 dot over FMA);
  weicj's production answer is CUDA-native substitution. 1-2-session
  gate-0 probe with three side venvs settles five sibling rows.

## Sibling ecosystem (adoption ledger)

weicj fork: adopt MTP guard rails + NVFP4 fp16-accum fix semantics
(#33972); rebuild GDN prefill in-house; defer TurboQuant (unmerged,
seed-nondeterministic, backend-gated). ik_llama.cpp: healthiest repo,
already runs Qwen3.8-Flash-Next with NextN MTP. flash-attention-sm75:
design reference only. Standing quarterly crosspoll audit row recorded.

## Uncertainty discipline

120 [uncertain] markers across the corpus (verified count), all named per item in the
JSONs. External numbers stay out of host records until verified
(TU102 paper rule); the weicj 100+ tok/s figure is best-case,
unnormalized, peer yardstick only.
