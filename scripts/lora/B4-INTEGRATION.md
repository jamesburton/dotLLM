# B4 — multi-corpus weighted mixing: integration into `bitdistill.py`

New file `corpus_mix.py` provides the mixed token stream. This doc lists the
**exact** hooks to wire it into `bitdistill.py::train` (and, optionally, the
teacher-cache generator). `corpus_mix.py` does **not** import `bitdistill.py`;
the wiring below is the only change needed and lives entirely in `bitdistill.py`.

Everything the mixer needs already matches the existing contract: each corpus
yields fixed-length `[seq_len]` int64 tensors, and `train()` pulls the stream
with `next(cpt_stream)` (line ~736: `torch.stack([next(cpt_stream) ...])`).

`--save-ckpt` (via `--no-save-ckpt`) and `--milestones` **already exist** — do
not re-add them.

## Public API of `corpus_mix.py`

```python
mixed_token_stream(named_streams: dict, weights: dict, *,
                   schedule="stride", seed=0, strict=True) -> Iterator
standard_capability_mix(factories: dict, weights: dict|None=None, *,
                        schedule="stride", seed=0) -> Iterator
curriculum_stream(factories: dict, *, seq_len, total_tokens,
                  phase_a_fraction=0.15, phase_a_weights=None,
                  phase_b_weights=None, schedule="stride", seed=0) -> Iterator
STANDARD_WEIGHTS = {"tooluse":35,"csharp":35,"instruction":20,"general":10}
```

* `named_streams` / `factories` values may be a **zero-arg factory**, a
  **re-iterable** (a `list` of tensors — cycles automatically), or a **bare
  iterator** (single-pass, dropped on exhaustion). `general` (the streaming cpt
  iterator) is infinite → never cycles; the capability corpora are lists →
  cycle transparently.
* Weights need not sum to 100; they are normalized over the corpora actually
  present with a positive weight (missing corpus ⇒ renormalize).

## 1. New CLI arg (`parse_args`, in the `# data` block ~line 865)

```python
    p.add_argument("--mix-weights", default=None, dest="mix_weights",
                   help="Enable the B4 multi-corpus capability mix. Comma list of "
                        "name=weight (e.g. 'tooluse=35,csharp=35,instruction=20,general=10'). "
                        "Names: tooluse,csharp,instruction,general. Omitted names default to "
                        "corpus_mix.STANDARD_WEIGHTS; a name with no available corpus is "
                        "dropped and the rest renormalize. When unset, uses the single cpt stream.")
    p.add_argument("--mix-schedule", default="stride", choices=["stride","random"],
                   dest="mix_schedule",
                   help="Interleave schedule: 'stride' (deterministic smooth round-robin, "
                        "exact ratios) or 'random' (seeded weighted choice).")
    p.add_argument("--mix-seed", type=int, default=0, dest="mix_seed",
                   help="Seed for --mix-schedule random (stride ignores it).")
    # Optional 2-phase curriculum (design §3 secondary lever):
    p.add_argument("--curriculum-phase-a", type=float, default=0.0,
                   dest="curriculum_phase_a",
                   help="If >0, run a Phase-A (general+instruction) warm-up for this "
                        "FRACTION of --tokens before the capability-heavy Phase-B mix.")
```

## 2. Parse the weights (top of `train`, or a tiny helper near the imports)

```python
def _parse_mix_weights(spec: str) -> dict:
    """'tooluse=35,csharp=35,instruction=20' -> {'tooluse':35.0,...}. Empty -> {}."""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition("=")
        out[name.strip()] = float(val) if val.strip() else 1.0
    return out
```

## 3. Build `named_streams` and swap the stream in `train`

Add `import corpus_mix as cmix` next to the existing `import bitdistill_data as bdata`
(line ~86).

Replace the **else** branch that builds `cpt_stream` (lines ~694–711) so that,
when `--mix-weights` is set, `cpt_stream` becomes the mixed stream. Keep the
existing single-stream path as the default (mix off). `ppl_slice` / `gsm8k` /
`csharp_tasks` are unchanged — PPL still tracks the general anchor.

```python
    else:
        # --- general (cpt) stream: unchanged, still the anchor corpus ---
        try:
            general_stream = bdata.cpt_token_stream(
                tokenizer, seq_len, dataset_name=args.cpt_dataset,
                dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet)
        except Exception as e:
            print(f"[bitdistill] cpt_config {args.cpt_config!r} failed ({e}); retry config=None")
            general_stream = bdata.cpt_token_stream(
                tokenizer, seq_len, dataset_name=args.cpt_dataset, dataset_config=None,
                local_parquet=args.cpt_local_parquet)

        if args.mix_weights:
            import capability_data as capdata
            # C# training stream: prefer the sibling B3 module if present, else
            # fall back to the raw-source csharp_data loader (PPL corpus).
            def _csharp_seqs():
                try:
                    import csharp_train_data as cstd          # B3 (sibling agent)
                    return cstd.load_csharp_train_sequences(tokenizer, seq_len=seq_len)
                except Exception:
                    import csharp_data as csd                  # fallback: raw .cs
                    seqs, _labels = csd.load_csharp_sequences(tokenizer, seq_len=seq_len)
                    return seqs

            # Each value is a list (cycles) or the infinite general iterator.
            # Only build a corpus if its weight is > 0, to avoid needless loads.
            weights = cmix.STANDARD_WEIGHTS | _parse_mix_weights(args.mix_weights)
            factories = {}
            if weights.get("general", 0) > 0:
                factories["general"] = general_stream
            if weights.get("tooluse", 0) > 0:
                factories["tooluse"] = capdata.load_capability_sequences(
                    "tooluse", tokenizer, n_seqs=args.mix_n_seqs, seq_len=seq_len)
            if weights.get("instruction", 0) > 0:
                factories["instruction"] = capdata.load_capability_sequences(
                    "instruction", tokenizer, n_seqs=args.mix_n_seqs, seq_len=seq_len)
            if weights.get("csharp", 0) > 0:
                try:
                    factories["csharp"] = _csharp_seqs()
                except Exception as e:
                    print(f"[bitdistill] C# corpus unavailable ({e}); dropping from mix")

            if args.curriculum_phase_a > 0:
                cpt_stream = cmix.curriculum_stream(
                    factories, seq_len=seq_len, total_tokens=args.tokens,
                    phase_a_fraction=args.curriculum_phase_a,
                    phase_b_weights=weights,
                    schedule=args.mix_schedule, seed=args.mix_seed)
            else:
                cpt_stream = cmix.standard_capability_mix(
                    factories, weights,
                    schedule=args.mix_schedule, seed=args.mix_seed)
            print(f"[bitdistill] MIX enabled ({args.mix_schedule}): "
                  f"{ {k: weights[k] for k in factories} }")
        else:
            cpt_stream = general_stream

        ppl_slice = bdata.load_ppl_slice(...)   # unchanged
        gsm8k = bdata.load_gsm8k(...)           # unchanged
        csharp_tasks = cse.load_csharp_tasks(...)  # unchanged
```

The training loop (line ~736) is **unchanged** — `next(cpt_stream)` transparently
draws from the mix. Token accounting (`tokens_seen += batch.numel()`) and the
milestone/checkpoint logic are untouched.

Add one supporting arg for the per-corpus list size (near `--mix-weights`):

```python
    p.add_argument("--mix-n-seqs", type=int, default=4096, dest="mix_n_seqs",
                   help="Sequences to materialise per finite capability corpus for "
                        "the mix (they cycle, so this is the pool size, not a cap on "
                        "tokens consumed).")
```

## 4. (Optional) teacher-cache generator

`generate_teacher_cache` (line ~901) builds its own `stream` at line ~919 the
same way. To cache the teacher over the **mixed** corpus (so the offline top-K
KD targets match the training distribution), apply the identical swap there:
build `factories` as above and set
`stream = cmix.standard_capability_mix(factories, weights, ...)` when
`args.mix_weights` is set. Reuse the same `_parse_mix_weights` / factory block —
factor it into a small `build_mixed_stream(args, tokenizer, seq_len)` helper and
call it from both `train` and `generate_teacher_cache`.

## Determinism / reproducibility notes

* `schedule="stride"` is deterministic with **no** RNG; over any N draws each
  corpus gets `round(N*w)` (±1). Validated: at N=1000 the standard blend drew
  exactly 350/350/200/100.
* `schedule="random"` is deterministic **given `--mix-seed`** (uses
  `random.Random(seed)`, no wall-clock).
* The mix order is independent of each corpus's own internal shuffling
  (`cpt_token_stream(seed=...)`, `capability_data` order), so a run is
  reproducible end-to-end when those seeds are fixed.

## Open decisions

1. **C# training corpus source** — the fallback (`csharp_data.load_csharp_sequences`)
   is **raw `.cs` source**, not instruction/CoT pairs. Once the sibling
   `csharp_train_data.py` (B3) lands, the `_csharp_seqs()` helper picks it up
   automatically; until then the C# arm trains on raw source (still better than
   0% C# representation, but not CoT).
2. **Completion masking** — the mix currently feeds plain packed sequences; the
   design §5 completion-masked CE (learn to *produce* tool-calls/CoT) is a
   separate hook not covered by B4. If added, the mixer must forward a
   `(tokens, loss_mask)` pair instead of a bare tensor — change the element
   contract in one place (`corpus_mix` forwards whatever the corpus yields, so
   no mixer change is needed, only the corpus factories and the train loop).
3. **`--mix-n-seqs` pool size** — 4096 seqs × 512 tok ≈ 2M tokens per capability
   pool before cycling repeats; fine for a 5M-token run but tune if repetition
   within an epoch matters.
