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

## Clear follow-up — ✅ DONE: constrained decoding rescues tool-calling
The `--tool-choice required` gap named here was closed (#106 wiring + #104 strict order-independent
constrained decoding), and a residual constraint bug found during this eval was fixed (#112): the tracker
unconditionally allowed `\` inside a trie-constrained key/enum string, so the weak base could *escape-flood*
(`{"name":"calculate_distance", "a\"\"\"…`) and break out of the constraint. With `\` routed through the
trie like any other char, the constraint now strictly forces the tool-call skeleton **and** the argument keys.

---

# Phase C — Constrained-decoding rescue (quantitative)

**Date:** 2026-06-28 · BitNet I2_S GGUF, GPU, greedy `--repeat-penalty 1.3 --repeat-last-n 256`. Held-out
`glaive_func_calling` rows (n=10, disjoint from training), served via `run --tools @<row tools> [--tool-choice
required] [--lora <tooluse>]` (`scripts/lora/eval_tooluse_bitnet.py`). Metrics: **name** = correct function name
emitted (forced skeleton or parsed); **complete** = a fully-parseable tool-call JSON; **args** = complete AND
arguments equal the gold call.

| config | name | complete | args |
|---|---:|---:|---:|
| base, unconstrained | **0%** | 0% | 0% |
| base, `--tool-choice required` | **70%** | 0% | 0% |
| +tooluse adapter, unconstrained | 0% | 0% | 0% |
| +tooluse adapter, `--tool-choice required` | **80%** | 0% | 0% |

### Findings
1. **The rescue is real.** Unconstrained, BitNet b1.58 2B emits a usable tool name **0%** of the time (it produces
   free-text garbage). Under `--tool-choice required` the decoder *forces* the `{"name":"<tool>", "arguments":{…}}`
   skeleton + the argument keys, yielding the **correct function name 70% (base) / 80% (+adapter)** of the time —
   correctness supplied by the decoder, not the model.
2. **The adapter improves tool *selection*** under the constraint (80% vs 70% name): the constraint guarantees a
   *valid* tool name; the adapter nudges the model toward the *right* one (the <100% comes from multi-tool rows where
   the model must still pick correctly).
3. **`complete` = 0% (the honest remaining limit).** The constraint forces structure + keys, but the argument
   *values* are free-form strings — and this capability-bound 2B base cannot reliably **terminate** a string value
   (it rambles to the token cap). This is a base-capacity limit, **not** a constraint defect (cf. `EVAL.md`: the same
   constraint yields fully-valid, terminating calls on Qwen3-4B). **Next constraint enhancement:** propagate JSON-Schema
   `maxLength` (and enum/format) into `SchemaTracker` so string values are length-bounded — that would let even this base
   emit a fully-terminating valid tool call.

### Verdict
Constrained decoding (#104/#106 + the #112 escape-flood fix) **rescues tool-call structure and function-name accuracy
on a base that fails entirely unconstrained** — exactly the small/quantised-base payoff the constrained-decoding work was
built for. Full argument-value correctness still needs either a stronger base or value-length bounding.

## Reproduce
Train (per adapter): `train_task_lora.py --task <t> --no-4bit --grad-checkpoint --max-seq-len <N> [--chat-template
scripts/lora/templates/bitnet_tooluse.jinja for tooluse] --base microsoft/bitnet-b1.58-2B-4T-bf16 …`.
Serve/stack: `dotnet run --project src/DotLLM.Cli -c Release -- run <i2_s.gguf> --device gpu --repeat-penalty 1.3
--repeat-last-n 256 [--prompt <rendered> | --tools @tools.json] [--lora <dir> …]`.
Phase C eval: `dotnet build src/DotLLM.Cli -c Release` once, then `PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface python
scripts/lora/eval_tooluse_bitnet.py --gguf <i2_s.gguf> --lora <bitnet tooluse dir> --n 10`.
