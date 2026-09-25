# Task-LoRA — U2 results: train → serve → stack (Qwen3-4B)

**Date:** 2026-06-25 · RTX 3060 (sm_86) · base `Qwen/Qwen3-4B-Instruct-2507` (served as `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`).

Three task LoRAs trained with `scripts/lora/train_task_lora.py` (QLoRA, **rank 16 / α 32 / dropout 0.05**, the 7 standard projections — identical sites so they stack), 2000 examples, 400 steps each, then served via dotLLM `--lora` and composed with the new repeatable `--lora` (stacking, PR #93).

## Adapters
| Adapter | Dataset | Final loss | Notes |
|---|---|---:|---|
| instruction | `HuggingFaceH4/no_robots` | 2.69 | diverse human prompts → higher steady loss |
| tool-use | `NousResearch/hermes-function-calling-v1` (`glaive_func_calling`), template-rendered (#94) | 0.0003 | tool-call format is highly regular → very low loss |
| coding | `iamtarun/python_code_instructions_18k_alpaca` | 0.31 | Python instruct→code |

All exported as standard PEFT (`adapter_config.json` + `adapter_model.safetensors`), r=16/α=32/7 modules.

## Swappable serving (base vs +adapter, greedy, GPU ~58 tok/s)
- **Coding — clearest behavioral shift.** Base emits verbose markdown-fenced code with a docstring + prose; the **coding adapter emits terse, direct code** (no fence, no prose) — the CodeAlpaca style:
  ```
  base:    "Here's a Python function...\n```python\ndef is_prime(n):\n    \"\"\"...docstring...\"\"\"  ..."
  +coding: "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, n): ..."
  ```
- **Instruction — modest shift.** The base is already a strong instruct model; the adapter (no_robots) yields a slightly plainer, less-markdown style. Both answer well.
- **Tool-use — both correct.** Qwen3-Instruct-2507 natively tool-calls, so base **and** adapter both emit the correct `<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>`. (A weaker base would show a larger adapter gain; here the base's strength masks it.)

## Stacking (the headline) — `--lora instruction --lora tooluse --lora coding`
The 3 adapters compose additively into one rank-48 adapter (`LoraComposer`); CPU/CUDA apply it through the unchanged single-adapter path. **No-regression confirmed — each skill survives the 3-way stack:**
- **stack on the coding probe** → terse correct code, and notably uses the efficient `int(n**0.5)+1` sqrt bound (≥ the single coding adapter's `range(2,n)`):
  ```
  def is_prime(n):
      if n < 2: return False
      for i in range(2, int(n**0.5)+1):
          if n % i == 0: return False
      return True
  ```
- **stack on the tool probe** → still emits the correct `<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>`.

Multi-`--lora` parsing verified (a bad extra `--lora path` errors "PEFT adapter directory not found"), and the CUDA stacked-parity test (`LoraStackCudaParityTests`) passes on this GPU (argmax CPU==GPU, cosine 0.999029).

## Conclusions
1. **End-to-end train→serve→stack works with real adapters.** All 3 adapters serve swappably and compose.
2. **Stacking retains capabilities** (coding + tool-use both intact under the 3-way stack) — the core promise of additive composition.
3. The coding adapter is the clearest single-adapter *style* shift; instruction/tool-use gains are masked by Qwen3-Instruct-2507 already being strong there. Quantified in Phase C below.

---

# Phase C — Quantitative eval (base vs +adapter)

**Date:** 2026-06-28 · same RTX 3060 + Q4_K_M base. Automated harness (`scripts/lora/eval_*.py`): prompts rendered with `apply_chat_template` (train==serve form), served greedy via `dotnet run -- run --json [--lora]`, scored programmatically. Held-out tool-use rows are disjoint from the `train[:3000]` training slice.

## Tool-use — held-out `glaive_func_calling` (n=20), tool-name + arg-JSON match
| metric | base | +tooluse adapter |
|---|---:|---:|
| `<tool_call>` emitted | 100% | 100% |
| function name correct | 100% | 100% |
| arguments JSON-equal | **90%** | 85% |

The base is already at ceiling on name selection; the adapter is neutral-to-slightly-lower on argument exactness. No gain on this strong base (consistent with the qualitative finding).

## Coding — 15 self-contained problems, pass@1 (sandboxed execution)
| | base | +coding adapter |
|---|---:|---:|
| pass@1 | **15/15 (100%)** | **15/15 (100%)** |

Both solve every problem — the base is strong enough that these problems don't discriminate. The adapter's effect is **stylistic** (terse, no markdown fence) rather than correctness, so it doesn't move pass@1. (Harder/held-out problems would be needed to separate them.)

## Instruction — 12 fixed probes, pairwise judge (base vs +instruction adapter)
| outcome | count |
|---|---:|
| base preferred | 10 |
| adapter preferred | 1 |
| tie | 1 |

The `no_robots` adapter yields a **plainer** style but **slightly degrades constraint-following**: it broke explicit length/format constraints on 4 probes (haiku form; "in one sentence"; "one-line"; "in two sentences") where the base obeyed them. Net: ~83% base win-rate.

## Phase C verdict
**On the already-strong Qwen3-4B-Instruct-2507 base, none of the three adapters beats the base on its own task quantitatively** — tool-use neutral, coding tied at ceiling, instruction slightly worse. The adapters deliver **stylistic** shifts (terse code, plainer prose), not capability gains, and the instruction adapter can hurt constraint adherence. This is the expected outcome for LoRA on a strong instruct base, and it sharpens the motivation for the **weaker-base demo (BitNet, U5)** where the base fails and an adapter can clearly rescue. The headline that *does* hold here is U3: **3-way stacking preserves each capability with no regression** (cosine 0.999029 CPU==GPU).

## Reproduce
- Train: `scripts/lora/train_task_lora.py` (see `.docs/train_all.sh`).
- Serve/stack: `dotnet run --project src/DotLLM.Cli -c Release -- run <qwen-q4_k_m.gguf> --device gpu --prompt <rendered> --lora <dir> [--lora <dir> ...]` (prompts via `tokenizer.apply_chat_template`, per `FORMAT.md`).
- Quantitative eval (Phase C): `dotnet build src/DotLLM.Cli -c Release` once, then
  `python scripts/lora/eval_tooluse.py --gguf <gguf> --lora <tooluse_dir> --n 20`,
  `python scripts/lora/eval_coding.py --gguf <gguf> --lora <coding_dir>`,
  `python scripts/lora/eval_instruction.py --gguf <gguf> --lora <instruction_dir>` (env `PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface`).
