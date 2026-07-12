# B1 integration — PyTorch-side tool-use + instruction evals into `bitdistill.py`

These are the exact hooks to wire `tooluse_exec_eval.py` and `instruction_exec_eval.py` into
`bitdistill.py`. They mirror precisely how `eval_csharp_exec` / `--eval-csharp` (build item B2) is
already wired. **Nothing else in `bitdistill.py` changes.** `bitdistill.py` has **two** training/curve
paths — the standard loop (`~line 693–802`) and the offline-teacher-cache loop (`~line 996–1068`);
apply the load-once, milestone, and final hooks in **both**, exactly as the C# eval already is.

New curve/checkpoint field names introduced: **`tooluse_acc`** and **`instruction_acc`** (floats,
`nan` when the eval is disabled) — siblings of the existing `csharp_pass1`.

---

## 1. Imports (top of file, next to the existing eval import — `~line 87`)

```python
import csharp_exec_eval as cse  # noqa: E402   (existing)
import tooluse_exec_eval as tue  # noqa: E402
import instruction_exec_eval as iue  # noqa: E402
```

## 2. CLI args (in the arg-parser, next to `--eval-csharp` — `~line 879–886`)

```python
p.add_argument("--eval-tooluse", action="store_true", dest="eval_tooluse",
               help="Run in-process tool-use name+args accuracy eval at each milestone "
                    "(held-out glaive prompts -> parse <tool_call> -> name+args match).")
p.add_argument("--eval-n-tooluse", type=int, default=20, dest="eval_n_tooluse",
               help="Number of held-out glaive tool-use examples for the accuracy eval.")
p.add_argument("--eval-instruction", action="store_true", dest="eval_instruction",
               help="Run in-process auto-checkable instruction-following eval at each milestone "
                    "(deterministic length/format/keyword constraint probes; no judge).")
p.add_argument("--eval-n-instruction", type=int, default=None, dest="eval_n_instruction",
               help="Number of instruction constraint probes (default: the full built-in set).")
```

## 3. Load-once (mirror the `csharp_tasks = cse.load_csharp_tasks(...)` line)

Standard loop `~line 710–711`, and offline-cache loop `~line 997–998`. Add after the `csharp_tasks`
load in **each**:

```python
tooluse_ex = tue.load_tooluse_eval(tokenizer, n=args.eval_n_tooluse) \
    if args.eval_tooluse else []
instruction_ex = iue.load_instruction_eval(tokenizer, n=args.eval_n_instruction) \
    if args.eval_instruction else []
```

> `load_tooluse_eval` needs the `tokenizer` (it chat-templates + renders the `<tools>` block at load
> time for train==serve parity). `load_instruction_eval` accepts `tokenizer` for signature parity but
> ignores it. Both return `[]`/non-empty and never raise — same disabled-contract as
> `load_csharp_tasks` / `load_gsm8k`.

## 4. Milestone eval calls (mirror the `csharp_p1 = ...` milestone block)

Standard loop `~line 773–782`, offline-cache loop `~line 1045–1054`. In **each** milestone block:

Compute (after the existing `csharp_p1 = ...`):
```python
tooluse_a = tue.eval_tooluse_inproc(student, tokenizer, tooluse_ex, device) if tooluse_ex else float("nan")
instruction_a = iue.eval_instruction_inproc(student, tokenizer, instruction_ex, device) if instruction_ex else float("nan")
```

Add the fields to the checkpoint `extra=` dict:
```python
extra={"ppl": ppl, "gsm8k_acc": acc, "csharp_pass1": csharp_p1,
       "tooluse_acc": tooluse_a, "instruction_acc": instruction_a,
       "milestone": mtok}
```

Add the fields to the `curve.append({...})`:
```python
curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
              "csharp_pass1": csharp_p1,
              "tooluse_acc": tooluse_a, "instruction_acc": instruction_a})
```

Extend the milestone print:
```python
print(f"[milestone {mtok:.2e}] ppl={ppl:.3f}  gsm8k_acc={acc}  csharp_pass1={csharp_p1}  "
      f"tooluse_acc={tooluse_a}  instruction_acc={instruction_a}  -> {ck}", flush=True)
```

## 5. Final eval + curve (mirror the final `csharp_p1 = ...` block)

Standard loop `~line 788–801`, offline-cache loop `~line 1059–1068`. In **each** final block:

```python
tooluse_a = tue.eval_tooluse_inproc(student, tokenizer, tooluse_ex, device) if tooluse_ex else float("nan")
instruction_a = iue.eval_instruction_inproc(student, tokenizer, instruction_ex, device) if instruction_ex else float("nan")
curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
              "csharp_pass1": csharp_p1,
              "tooluse_acc": tooluse_a, "instruction_acc": instruction_a, "final": True})
```

Add to the final `save_checkpoint(... extra={...})`:
```python
extra={"ppl": ppl, "gsm8k_acc": acc, "csharp_pass1": csharp_p1,
       "tooluse_acc": tooluse_a, "instruction_acc": instruction_a}
```

Extend the final print:
```python
print(f"[bitdistill] done: {step} steps / {tokens_seen} tokens. "
      f"final ppl={ppl:.3f} gsm8k_acc={acc} csharp_pass1={csharp_p1} "
      f"tooluse_acc={tooluse_a} instruction_acc={instruction_a}. curve -> {args.out}/curve.json")
```

## 6. Example invocation

```bash
TORCHDYNAMO_DISABLE=1 HF_HOME=E:/.cache/huggingface PYTHONIOENCODING=utf-8 \
python scripts/lora/bitdistill.py --base Qwen/Qwen3-0.6B ... \
    --eval-gsm8k --eval-n-gsm8k 100 \
    --eval-tooluse --eval-n-tooluse 20 \
    --eval-instruction \
    --milestones ... --save-ckpt
```

## Function signatures (reference)

```python
# tooluse_exec_eval.py
load_tooluse_eval(tokenizer, n=20, start=3000,
                  dataset_id="NousResearch/hermes-function-calling-v1",
                  dataset_config="glaive_func_calling") -> list[dict]   # {"prompt","gold_name","gold_args"}
eval_tooluse_inproc(model, tokenizer, examples, device,
                    max_new_tokens=160, chunk=8) -> float   # name+args accuracy, nan if empty

# instruction_exec_eval.py
load_instruction_eval(tokenizer=None, n=None) -> list[dict]   # {"instruction","verify"}
eval_instruction_inproc(model, tokenizer, examples, device,
                        max_new_tokens=64, chunk=8) -> float   # fraction of constraints satisfied, nan if empty
```

Both `eval_*_inproc` follow the `eval_gsm8k` / `eval_csharp_exec` contract exactly: force
`use_cache=True` for decode, restore prior train/eval mode + `use_cache`, batched left-padded greedy
`model.generate`, and return `nan` for an empty example list.
