# Format interop: execute newer-format checkpoints on sm_75 at best emulated/recast speed

Plan/0005. The operator's requirement: when a model ships in a format that
upstream vLLM gates to sm_80+ silicon (FP8 today; MXFP4/NVFP4 and friends
tomorrow), this hardware loads and runs it immediately, at the best speed
emulation or recasting allows, with attested numerics — so incoming models
are testable on Turing before llama.cpp backports the format.

The key numerics fact that makes this constructible: **E4M3 is a strict
subset of FP16** — conversion is a sign shift plus exponent rebias (or a
512-byte shared-memory LUT), exact and lossless. Format defines storage;
the math runs in FP16 tensor cores at Turing's full (non-halved) rate.
Block/tile scales reuse the accumulator-domain scale machinery from
plan/0004. The same skeleton covers future formats: a new format is a
checkpoint adapter plus a row in the strategy registry, not a new program.

## Strategies (per format) — FP8 residency is the priority

The point of running FP8 checkpoints on a 24 GB card is keeping weights
resident as FP8: half the footprint of FP16, and 2x the model size that
fits. So the **emulate** path is the headline deliverable and recast is
the fallback for when memory allows or a quick same-day check matters
more than footprint.

1. **Emulate (headline)** — weights stay FP8 in VRAM; dequant-in-flight
   per tile in the staged/regdequant family (byte-wide sibling of the
   nibble path: exponent rebias/LUT instead of nibble extract;
   skip-flop bias tricks where they apply). Speed bound: FP16 tensor
   rate minus dequant ALU (D8's cost table bounds it). At M=1 the
   weight bytes match a native FP16 model's, so bandwidth-bound decode
   latency matches FP16 residency at half the memory.
2. **Recast (fallback)** — convert weights to FP16 at load (lossless
   for E4M3), run the characterized FP16 path. Zero new kernels; 2x
   FP8 memory. This is the "test the model today" path when the
   footprint fits.

Both ride the plan/0003 harness unchanged: float64-oracle (the FP8
dequant reference is exact), ladder, winner gates, dispatch table. The
dispatch table gains a **format axis**: selection gates on "we have an
execution strategy for this checkpoint", replacing the hardware
capability gate.

## Operator target

A Qwen3-class ~27B model on both cards (2x RTX 6000, 48 GB aggregate).
Feasibility per format: FP8 ~27 GB weights -> ~14 GB/card with generous
KV; W4A16 ~14 GB -> single-card possible, both cards for throughput;
W2A16 ~7 GB -> single-card. Both-card serving runs over the NVLink
bridge (vLLM tensor parallel with peer access enabled, or the measured
plan/0004 shard strategies); if the target checkpoint is the MoE
variant, the MoE grouped forward is already oracle-validated
(plan/0002). Acceptance for the milestone: this checkpoint class loads,
generates on both cards, and tokens/s are measured and committed.

## AutoRound formats (Intel) — added to the lacuna

Intel's AutoRound emits W2A16/W3A16/W4A16 checkpoints (asymmetric
signed ints with group scales and zero points, in GPTQ/AWQ-compatible
or compressed-tensors layouts). These fill the low end of the format
ladder on sm_75:

- **W4A16 asymmetric + ZP**: our `staged` strategy is already
  zero-point-capable (the reference contract); the adapter maps
  AutoRound's signed int4 + zp into the staged layout. This is the
  shortest path to serving AutoRound W4 on Turing.
- **W2A16**: 16 weights per u32 — a wider-grain sibling of the nibble
  machinery, cheapest dequant of all (2-bit extract + group scale);
  implement as a strategy row through the standard gates. Precision is
  AutoRound's problem; ours is exact execution of the stored values at
  best speed.

## Stages

1. **Checkpoint adapters** — parse FP8 layouts into a (weights, scales)
   IR: compressed-tensors FP8, fp8 safetensors with per-tensor/channel/
   block scales (`weight_scale_inv`), gguf block variants as stretch.
2. **Recast path** — E4M3 -> FP16 loader + existing fp16 forward;
   acceptance: a real FP8 checkpoint (e.g. an 8B fp8 model) loads and
   generates on TU102, numerics attested against the float64 reference
   of the FP8 values (which the recast preserves exactly).
3. **Emulate path** — byte-dequant strategy rows (E4M3 LUT/rebias) in
   the kernel family; ladder-tuned like every other strategy; acceptance:
   beats recast's memory footprint at no worse than a measured, gated
   compute penalty, or is documented as losing.
4. **AutoRound adapter** — W4A16 asymmetric+ZP through the staged
   strategy (already zero-point-capable); W2A16 as a 2-bit strategy row
   (16 weights per u32); W3A16 as a 3-bit packing row; ladder-gated
   like all rows.
5. **Format axis in selection** — DispatchTable + can_implement gate on
   format coverage; vLLM tests green; incoming-format checklist doc
   ("new model arrives -> which adapter, which strategy, expected
   speed").
6. **FP8 activations (W8A8 emulated)** — per-token scales; activations
   rounded onto the E4M3 grid but stored and computed in FP16 (faithful
   emulation, no hardware FP8 needed); runtime quantization hook at the
   layer boundary.
7. **FP8 KV** — recast to fp16 KV at 2x memory (faithful, simple) or
   int8 KV (measured against fp16-KV quality); the serving config
   records which the checkpoint requires.
8. **MXFP4 adapter** — E2M1 values via a 16-entry nibble LUT, E8M0
   block scales as power-of-two FP16 multiplies (32-element blocks).
9. **NVFP4 adapter** — E2M1 nibble LUT + E4M3 block scales
   (16-element blocks); the scale conversion reuses the FP8 rebias.
10. **BF16 checkpoint adapter** — the most commonly hit gate of all:
    modern releases ship bf16 and sm_75 has no bf16 tensor support
    (upstream answer: pass dtype=float16). Recast bf16->fp16 at load
    (8 mantissa bits fit fp16's 10) with an overflow range check:
    >65504 outliers reported, never silently truncated.
11. **Attention/engine strategy** — the largest blocked surface is not
    a format: upstream vLLM's V1 engine requires FlashAttention or
    FlashInfer (sm_80+) and the V0/xformers path Turing relied on has
    been removed, so current upstream has no working sm_75 engine at
    all. This fork retains the Turing-capable engine (CUDA-graph
    validated); the strategy row is keeping that engine healthy against
    upstream drift, plus a Triton-attention option (Triton runs on
    sm_75) as the portable fallback.
12. **MLA models (DeepSeek-class)** — FlashMLA is sm_90/Blackwell-only;
    the emulated path is Triton-MLA, unmeasured on Turing. Registry row
    + measurement; expected slow, but the interop contract is
    execution with attested numerics, and slow-but-correct is how every
    backport starts.
13. **Both-card target run** — the operator's Qwen3-class ~27B
    checkpoint across both GPUs, tokens/s committed with the serving
    configuration (TP over NVLink or the measured shard strategies).

The operator NAKed deferral: the lacuna is not only quant formats.
Every surface vLLM conventionally gates to sm_80+ — quant formats,
checkpoint dtypes, attention backends, the engine itself — is in this
milestone, sequenced above. A surface is done when its adapter or
backend, kernel rows, oracle transcript, and selection entry are
committed and the coverage table below has no uncovered row. The
engine row (11) is the load-bearing one: without a Turing engine,
every other row is unreachable on current upstream.

## Hardware-fact sheet (why Turing is good at this)

- E4M3 in FP16: lossless (subset); rebias = +8 to the exponent field.
- Turing FP16 tensor cores run at FULL rate with FP32 accumulate
  (1024 FLOP/clk/SM) — the mode emulation uses. Ampere consumer halves
  in this mode; the plan/0002 per-SM advantage (1.75x) carries over.
- Weight memory for emulated FP8 = 1 byte/weight: half of FP16 (so M=1
  decode matches a native FP16 model's bandwidth-bound latency), 2x our
  4-bit path.

## Acceptance

1. An upstream-shipped FP8 checkpoint loads and generates on TU102 via
   the recast path, output attested against the float64 reference of the
   FP8 weights (lossless recast => attests the loader).
2. The emulated path exists, is oracle-clean, and is ladder-tuned; its
   measured speed vs recast is recorded with gate transcripts.
3. The format axis is wired into selection with tests; incumbent
   delegation unchanged when no adapter matches.
4. A written "incoming format" checklist so the next format is an
   adapter + a row.
