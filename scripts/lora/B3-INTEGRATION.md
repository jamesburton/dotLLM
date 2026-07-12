# B3 — C# instruction(+CoT) training stream

`scripts/lora/csharp_train_data.py` produces the **C# capability** slice of the
capability-distillation corpus: chat-templated, completion-masked C# instruction(+CoT)
token windows, shape/dtype-identical to `bitdistill_data.cpt_token_stream`.

## Source chosen + why

| Candidate | Verdict | Evidence |
|---|---|---|
| `nvidia/OpenCodeReasoning` | ❌ unusable | 735K samples but **100 % Python** competitive programming; no C#. |
| `nvidia/OpenCodeInstruct` (C#-filtered) | ⚠️ near-zero | 5M rows, no language tag. Measured C#-content hit-rate **0/3000 = 0.00 %** streamed. Kept only as a documented configurable fallback. |
| `Safurai/Safurai-Csharp-Instruct`, `ise-uiuc/MSCoT` | ❌ gone | Repos no longer resolve on the Hub (removed/renamed). |
| **`layoric/tiny-codes-alpaca-csharp`** | ✅ **DEFAULT** | 125.5K rows, `programming_language == "C#"` for every row (**≈100 % hit-rate**), Alpaca `instruction`/`output`. Each `output` carries a **natural-language control-flow walkthrough** ("Here's some sample code… This code uses `try`/`catch`… The recommended technique depends on…") = the instruction+rationale (CoT-style) signal we want. 126 MB parquet ⇒ cheap to **stream**. |

**Decision:** default `source="tiny-codes-csharp"`. It is the only dense, streamable,
instruction+rationale C# source that actually exists on the Hub today. Dedicated C#
*chain-of-thought* corpora are effectively absent, so the loader is
**source-configurable** (`_SOURCES` registry) with two documented fallbacks:
`source="opencodeinstruct"` (content-filtered, near-zero yield — for completeness), and
synth-via-`OllamaBenchmarks/scripts/coding_tasks/task_runner.py` compile-verified
generation (not implemented here; wire a strong local model → `task_runner` build/test →
keep only compiling C#).

## Public API (signatures)

```python
csharp_train_stream(
    tokenizer, seq_len, split="train", source="tiny-codes-csharp",
    return_labels=False, seed=0, loop=None, max_source_rows=None,
    family_filter=None, tasks_dir=..., stats=None,
) -> Iterator[Tensor | tuple[Tensor, Tensor]]

load_csharp_holdout(
    tokenizer, seq_len, split="val", n_seqs=64, source="tiny-codes-csharp",
    return_labels=False, seed=12345, max_source_rows=None,
    family_filter=None, tasks_dir=..., stats=None,
) -> list
```

- `return_labels=False` → yields `[seq_len]` int64 **input_ids** tensors — a drop-in for
  `cpt_token_stream` (unmasked CPT/PPL use).
- `return_labels=True` → yields `(input_ids, labels)` tuples of `[seq_len]` int64 tensors
  with **prompt positions masked to `-100`** (via `masking.build_labels`) — completion-only
  CE so the student learns to *produce* the C# answer+rationale, not model the prompt.
- Examples are chat-templated with `apply_chat_template(..., add_generation_prompt=True)`
  for **train==serve parity**, then GPT-packed into `seq_len` windows; the completion mask
  travels with the tokens through packing.

## How the trainer consumes it

Mirror the existing `cpt_token_stream` wiring in `bitdistill.py`:

- **Unmasked / CPT + PPL mode** (`return_labels=False`): identical shape to
  `cpt_token_stream`, so it can feed the CPT warm-up or a held-out C# PPL probe with no
  trainer change (`load_csharp_holdout(split="val", return_labels=False)` for PPL).
- **Capability SFT/KD mode** (`return_labels=True`): the trainer computes CE only on
  `labels != -100` (standard HF `ignore_index=-100`), and applies the offline top-K KD term
  on the same completion positions. This is the mode P1 uses for the C# capability bucket.

## How it feeds the future B4 mixer

B4 (multi-corpus weighted mixing + curriculum in `bitdistill_data`) round-robins batches
across capability buckets + general anchor at the locked weights (**tool-use 35 / C# 35 /
instruction 20 / general 10**). `csharp_train_stream(..., return_labels=True)` is the **C#
bucket** — an infinite (`loop=True` for `split="train"`) stream of `[seq_len]` windows the
mixer pulls from at its C# weight, alongside `capability_data` (tooluse/instruction) and
`cpt_token_stream` (general). All four share the `[seq_len]` int64 contract, so the mixer
just interleaves. `family_filter` allows per-family reweighting inside the C# bucket if a
family stalls.

## Holdout & contamination

- **Family-stratified 80/10/10 by problem.** Split is a deterministic `blake2b` hash of the
  normalised instruction (`_split_of`) — stream-safe (no materialisation), stable across
  runs, family-independent so every family splits ≈80/10/10, and whole problems are held out
  (never split mid-problem). **Measured over 2000 rows: train 79.8 % / val 9.5 % / test
  10.8 %.**
- **Never emit a benchmark problem.** The 50 held-out
  `OllamaBenchmarks/scripts/coding_tasks/tasks/NN_*.yaml` execution-eval problems are excluded
  by `_load_bench_signatures` / `_is_contaminated`: drop if a benchmark prompt fingerprint is
  a substring of the example, or the example shares ≥3 distinctive CamelCase identifiers with a
  single benchmark task. (0 drops on tiny-codes — generic C# vs modern-.NET benchmarks barely
  overlap — but the guard is source-agnostic for richer future sources.)
- Report **val** each checkpoint (selection); touch **test** once at the end.

## Family taxonomy

Reuses `csharp_data._FAMILIES`
(aspnet/async/blazor/efcore/linq/masstransit/patterns/vertical/xunit) via a text/keyword
detector (`_detect_family_text`). tiny-codes is generic Console-style C#, so most examples
map to `other` with a light sprinkle of `linq`/`async`/`patterns`; the taxonomy carries real
signal for modern-.NET sources (and for the synth fallback keyed off the benchmark families).

## Sample validation (CPU, no GPU)

```
python scripts/lora/csharp_train_data.py --source tiny-codes-csharp \
    --seq-len 512 --n-windows 16 --max-source-rows 500
```

Observed:
- 16 windows, `input_ids (512,) int64` + `labels (512,) int64`, shapes/dtype OK.
- window[0]: 446/512 positions supervised (rest prompt-masked to `-100`); decoded completion
  span is a real C# snippet + rationale (```csharp ... `try`/`catch` ...`).
- C#-filter hit-rate **100 %** (tiny-codes needs no filter); benchmark-dedup drops **0**.
- Holdout split scan: val kept 8, test kept 9 (families all `other` at this sample size).

## C#-filter hit-rate summary

| Source | Needs content filter | Measured C# yield |
|---|---|---|
| `tiny-codes-csharp` (default) | no (tagged) | 100 % |
| `opencodeinstruct` | yes (content scan) | **0/3000 = 0.00 %** — not viable in practice |

## Open decisions

1. **CoT depth.** tiny-codes rationales are *explanatory* (control-flow walkthroughs), not
   formal step-by-step reasoning traces. If P1 shows weak C# reasoning transfer, add
   synth-via-`task_runner` compile-verified CoT (teacher-generated `<think>` traces over the
   held-out families' *style* — using disjoint prompts) as a second bucket.
2. **General C# breadth.** tiny-codes skews to small Console programs; modern-.NET surface
   (aspnet/efcore/blazor/masstransit) is thin. Consider blending `csharp_data`'s 2,852 `.cs`
   files as raw-CPT C# (unmasked, `return_labels=False`) to broaden idiom coverage without
   contaminating the instruction holdout.
3. **Holdout granularity.** Split is by normalised-instruction hash; tiny-codes instructions
   are highly templated, so a few near-duplicate instructions could co-locate. If exactness
   matters, key `_split_of` on `main_topic+subtopic` instead. Deferred — current ratios are
   clean.
