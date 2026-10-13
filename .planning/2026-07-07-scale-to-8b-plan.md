# Scaling the ternary-expert accuracy test to 8B (2026-07-07)

## Hypothesis
At 2B, ternary experts improve PPL but add ZERO task accuracy (GSM8K 47.4%==47.4% for a full-scope,
task-matched, +16%-PPL expert; HumanEval regressed 20->10%). Question: is the PPL->accuracy gap a
fundamental property of ternary low-rank experts, or a 2B representational-capacity limit that CLOSES
at larger scale? Test at ~8B.

## Model
`HF1BitLLM/Llama3-8B-1.58-100B-tokens` — flagship community 1.58-bit model, llama arch, transformers
BitNet quant (same `transformers.integrations.bitnet.WeightQuant` our wrapper uses). Alternatives:
`tiiuae/Falcon3-7B-Instruct-1.58bit`, `tiiuae/Falcon3-10B-Instruct-1.58bit` (both llama+bitnet).

## Hardware
Strix ONLY (128 GB unified; 8B bf16 master + training doesn't fit the 3060's 12 GB). BitNet loads are
reliable on Strix. The 3060 is out for 8B.

## THE BLOCKER — generation speed (Phase 0, must fix first)
HF BitNet `AutoBitLinear`/`BitLinear` re-ternarizes ALL weights every forward -> ~1 tok/s at 2B,
would be ~0.3 tok/s at 8B -> accuracy evals (which REQUIRE generation) become days. At inference the
ternary weight is CONSTANT (frozen master), so it can be computed ONCE and cached.
- **Fix:** patch BitLinear.forward to cache the quantized weight when an inference flag is set (compute
  once, reuse). Also cache the ternary LoRA A/B in the wrapper at inference. Training path unchanged
  (master changes each step -> must re-ternarize).
- **Validate on 2B first** (fast to test): expect ~1 tok/s -> ~15-25 tok/s. This retroactively makes
  the 2B evals fast too and is the enabler for 8B.

## Harness changes needed
1. Parameterize `BASE_MODEL_ID` -> `--base-model` in `pdiff_ternary_lora.py`,
   `tlora_capability_eval.py`, `tlora_capability_hard.py` (currently hardcoded microsoft/bitnet-2B).
2. Phase-0 ternary-weight inference cache (patch + wrapper).
3. Verify `_wrap_mlp_layers` attaches to llama-arch `mlp.{gate,up,down}_proj` (BitLinear) on the 8B.

## Plan
- **P0 Generation-speed fix** — implement + validate ternary-weight inference caching on the 2B
  (fast->fast). Gate: 2B GSM8K item time drops from ~6 min to <1 min.
- **P1 8B setup on Strix** — download Llama3-8B-1.58; parameterize base-model; verify load + wrapper
  attach + a 5-token generation works; measure tok/s.
- **P2 Decisive experiment at 8B** — (a) base GSM8K + HumanEval accuracy (fast now, n=40+);
  (b) train full-scope r64 expert on GSM8K-train (~2000 ex, lambda-ramp); (c) eval expert accuracy.
  KEY: does expert beat base on GSM8K/HumanEval accuracy at 8B?
- **P3 Interpret** — gap closes at 8B -> mini-MoTE at 8B, then Full MoTE is justified. Gap persists ->
  the PPL-accuracy gap is scale-invariant for ternary low-rank experts (a clean, publishable negative).

## Success metric
The single number: 8B base GSM8K accuracy vs 8B full-scope-GSM8K-expert GSM8K accuracy (matched n).
2B result was +0.0. If 8B shows a clear positive delta beyond noise, scale closes the gap.

## Risks
- 8B model's BitLinear may differ from microsoft/bitnet's AutoBitLinear (verify wrapper + WeightQuant
  import path; adapt if needed).
- Even with caching, 8B generation may be modest on gfx1151; keep eval n bounded.
- The 8B 1.58-bit model may already be weak at GSM8K (low base) -> pick a benchmark with headroom.
