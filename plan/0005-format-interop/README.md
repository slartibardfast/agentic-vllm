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
4. **Format axis in selection** — DispatchTable + can_implement gate on
   format coverage; vLLM tests green; incoming-format checklist doc
   ("new model arrives -> which adapter, which strategy, expected speed").
5. Later, recorded but deliberately not opened yet: FP8 activations
   with per-token scales, emulated as FP16 activations; FP8 KV either
   recast to fp16 KV at 2x memory or quantized to int8 KV; MXFP4 and
   NVFP4 adapters.

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
