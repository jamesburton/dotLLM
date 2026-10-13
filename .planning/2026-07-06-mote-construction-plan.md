# MoTE construction plan — rapid ternary experts → Mixture-of-Ternary-Experts (2026-07-06)

## Premise (validated primitives)
- **No-merge ternary LoRA + QAT (λ-ramp)** produces a ternary-survivable capability increment on a
  frozen BitNet base; ~25 min/expert on Strix; captures ~71% of the bf16 ceiling at matched rank;
  scales with rank and scope. This is the **rapid ternary-expert training primitive**.
- **Precision phasing must be a continuous anneal, not train-then-convert.** Post-hoc ternarizing a
  bf16-trained LoRA (11.356) < QAT-in-ternary (11.199). The λ-ramp (bf16→ternary within one run) is
  the right form. A **progressive-precision curriculum** (bf16→int8→int4→ternary) is a testable
  refinement — smaller jumps may capture more than the one-shot ramp.
- **Ternary↔higher-precision back-and-forth is just the BitNet QAT regime** (bf16 master + STE
  ternarize every forward). No new mechanism needed; round-tripping through a full bf16 *training*
  phase is what loses grid alignment.

## Decision gate
Proceed to Full MoTE **only if** (a) the mini-MoTE shows domain experts specialize + a router routes,
AND (b) the capability eval shows the ternary gain is real on task-accuracy (not just PPL). Both
running now.

## Expert set (8 = 1 shared + 7 routed) — DeepSeek-MoE shared-expert pattern
Shared/"original" expert = base (general text routes here). 7 domain specialists:

| # | domain | HF dataset | schema (verified 2026-07-06) | status |
|---|--------|-----------|------------------------------|--------|
| shared | general/chat | HuggingFaceH4/no_robots | messages | cached (E:) ✓ |
| 1 | code | iamtarun/python_code_instructions_18k_alpaca | alpaca | cached (E:) ✓ |
| 2 | math | TIGER-Lab/MathInstruct | alpaca/instruction | cached (E:) ✓ |
| 3 | medical | medalpaca/medical_meadow_medical_flashcards | instruction/input/output (alpaca) | verified, drop-in; 7MB/34K |
| 4 | finance | gbharti/finance-alpaca | instruction/input/output/text (alpaca) | verified, drop-in; 23MB/69K |
| 5 | science | allenai/sciq | question/correct_answer/support (QA) | verified; **needs text-map** (q+support+answer); 4MB/11.7K |
| 6 | creative | euclaise/writingprompts | prompt/story | verified; **needs text-map** (prompt+story); 544MB train (use a slice) |
| 7 | legal | TBD (nguha/legalbench slice or lex_glue) | — | not yet verified |

Notes: pick maximally-DISTINCT domains for sharp routing (code/math are close; medical/legal/creative
are far). `_extract_text` handles messages/conversations/alpaca/text; **sciq & writingprompts need a
small schema handler added** (or substitute alpaca-format equivalents, e.g. camel-ai/* for science).
Downloads deferred to proceed-time (disk = 26 GB free); each expert only needs `train[500:2500]`.
Each expert config: no-merge ternary LoRA, full scope or last-12, r64+ (rank scales), λ-ramp, ~1500
steps. Train on Strix (128 GB, fast).

## Router
- **Mini (proof, now):** per-sequence. Mean-pooled base last-hidden → linear classifier → expert id.
  Cheap, decisive for "do experts specialize + can we route."
- **Full:** per-token top-k router (true MoE), trained with a **load-balancing auxiliary loss**, experts
  frozen. Integrate with dotLLM's existing MoE router + quantized-expert CUDA/Vulkan kernels (Track-M
  MoTE) — the inference side already exists; training produces ternary experts + router weights, then
  export to dotLLM's MoTE format. Optional final **joint fine-tune** (router + experts) end-to-end with
  LM loss + load-balance.

## Additional scripts / harness options to add (beyond the mini-MoTE scripts already built)
1. `pdiff_ternary_lora.py --precision-schedule bf16,int8,int4,ternary` — generalize the λ-ramp to a
   multi-precision curriculum (progressive quantization). New quantizer branch + schedule parser.
2. `mote_router_train.py` per-token mode + load-balancing loss (extend the per-sequence proof router).
3. `mote_export.py` — convert trained ternary experts + router → dotLLM native/GGUF MoTE format for the
   C#/CUDA/Vulkan inference engine.
4. Expert-parallel orchestration — train 7 experts across Strix (primary) + 3060 (last-12 configs);
   dataset staging on each machine (download the 5 uncached domains to the training box's HF cache).
5. `mote_eval.py` capability + per-domain matrix at 8-expert scale (extend the mini scripts).
6. (Optional) diffusion warm-start per expert — modest but free; only if a same-domain adapter
   population is cheaply available.

## Phased plan
- **P0 (now):** mini-MoTE proof (code/math/general experts + per-seq router) + capability eval.
- **P1:** if gated pass — stage 5 remaining domain datasets on Strix; train 7 domain experts (QAT
  λ-ramp) + shared; specialization matrix.
- **P2:** per-token router + load-balance; mini→full router upgrade; joint fine-tune optional.
- **P3:** progressive-precision curriculum ablation (does bf16→int8→int4→ternary beat bf16→ternary?).
- **P4:** export to dotLLM MoTE format; validate inference on CUDA + Vulkan; measure end-to-end.

## Open questions to resolve with data
- Per-token vs per-sequence routing quality gap at this model scale (2B).
- Does the progressive-precision curriculum measurably beat the one-shot λ-ramp?
- Optimal expert scope/rank vs count trade-off under a fixed ternary memory budget.
- Load-balancing coefficient / top-k for 8 experts.
