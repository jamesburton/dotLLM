# Task-LoRA — U5 results: train → serve → stack on BitNet b1.58 (ternary 2B)

**Date:** 2026-06-26 · RTX 3060 · base `microsoft/bitnet-b1.58-2B-4T-bf16` (trained) → served on the
**I2_S ternary GGUF** (`ggml-model-i2_s.gguf`) via dotLLM `--lora`. Decoding: greedy + `--repeat-penalty 1.3 --repeat-last-n 256`
(a 2B ternary base degenerates into repetition under pure greedy — the penalty is required for coherent output).

Three task LoRAs trained with `scripts/lora/train_task_lora.py` (**rank 16 / α 32 / 7 projections**, 2000 ex × 400 steps,
`--no-4bit` bf16 base, `--grad-checkpoint --max-seq-len` to fit the 12 GB card). The **tool-use** adapter was trained
with `--chat-template scripts/lora/templates/bitnet_tooluse.jinja` (#100), so it sees a real `<tools>` block — not the
degenerate "no tool context" of BitNet's stock template.

## Adapters
| Adapter | Final loss | Notes |
|---|---:|---|
| instruction | ~1.5 | no_robots |
| coding | 0.30 | python_code_instructions (max-seq-len 1024 + grad checkpointing) |
| tool-use | ~0.5 | glaive_func_calling, rendered via the BitNet tool template (#100) |

All valid PEFT (r=16/α=32/7 modules).

## What works (the pipeline + #100)
1. **End-to-end pipeline works on a ternary base.** train (bf16 BitNet, peft) → serve on the **I2_S GGUF** → stack —
   exactly as on Qwen3-4B (U2), now on BitNet.
2. **#100 tool template is live at serve time.** With `--tools`, dotLLM renders the new BitNet tool-aware template and
   the model **references the correct `get_weather` function name** — vs the previous no-op where tool definitions never
   reached the model at all. The plumbing (template → model → Hermes parser) is exercised end-to-end.
3. **Stacking applies** — `--lora instruction --lora tooluse --lora coding` composes the 3 rank-16 adapters (rank-48) and
   runs through the unchanged single-adapter path on the BitNet I2_S GPU path.
4. **Adapters shift behavior** in the expected direction: the **coding** adapter drops the base's markdown fence/docstring
   and emits direct code (`def is_prime(n):\n    if n < 2:\n        return False …`); the **instruction** adapter gives a
   cleaner, more direct numbered list.

## Honest limitation: the 2B ternary base is capability-bound
BitNet b1.58 2B is a deliberately tiny, extreme-quantisation base. Even with adapters + repetition penalty:
- Coherent on **short** instruction output; **derails on longer** generation (code past the first few lines, repeated `<|eot_id|>`).
- For **tool-use**, it references the right function name but **cannot reliably emit valid `<tool_call>` JSON** — the
  structured-output demand exceeds the base's capacity.

This is a **base-model limitation, not a dotLLM/adapter/#100 defect** — the same harness produces clean, swappable,
stackable results on Qwen3-4B (see `EVAL.md`). U5's value is validating the **pipeline + the BitNet tool template +
stacking on the ternary base**, not matching Qwen quality.

## Clear follow-up
**Constrained decoding for tool-use.** dotLLM already has JSON-schema constrained decoding + `ToolCallSchemaBuilder`.
Forcing the tool-call grammar would let even this weak base emit *valid* `<tool_call>` JSON (correctness guaranteed by the
decoder, not the model) — the natural next demo and the right answer for small/quantised bases. (`--tool-choice required`
wiring into `run` is the gap.)

## Reproduce
Train (per adapter): `train_task_lora.py --task <t> --no-4bit --grad-checkpoint --max-seq-len <N> [--chat-template
scripts/lora/templates/bitnet_tooluse.jinja for tooluse] --base microsoft/bitnet-b1.58-2B-4T-bf16 …`.
Serve/stack: `dotnet run --project src/DotLLM.Cli -c Release -- run <i2_s.gguf> --device gpu --repeat-penalty 1.3
--repeat-last-n 256 [--prompt <rendered> | --tools @tools.json] [--lora <dir> …]`.
