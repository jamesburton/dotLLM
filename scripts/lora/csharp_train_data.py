"""csharp_train_data.py — C# instruction(+CoT) training stream for bitdistill (build item B3).

Purpose
-------
Yield **chat-templated, completion-masked** C# instruction(+CoT) token sequences as
fixed-length ``[seq_len]`` ``torch.long`` tensors — the *training* counterpart to the
raw-source C# PPL loader in ``csharp_data.py`` and a drop-in sibling of
``bitdistill_data.cpt_token_stream`` (same shape/dtype so it feeds the trainer and the
future B4 multi-corpus mixer unchanged).

Source research (2026-07-12, HF Hub)
------------------------------------
The plan's first choices for C# instruction+CoT do **not** hold up:

* ``nvidia/OpenCodeReasoning`` — 735K reasoning samples but **100 % Python**
  (competitive programming). No C#. Not usable.
* ``nvidia/OpenCodeInstruct`` — 5M samples, **no language tag**, Python-dominant.
  C# exists only as a sparse content-filtered subset (low hit-rate). Kept as a
  *configurable* source (``source="opencodeinstruct"``) to demonstrate the
  "filter a big multilingual set to C#" fallback path.
* ``Safurai/Safurai-Csharp-Instruct`` and ``ise-uiuc/MSCoT`` — **repos no longer
  resolve** on the Hub (removed / renamed). Dead ends.

Best *available* dense C# instruction+CoT source, chosen as the **default**:

* ``layoric/tiny-codes-alpaca-csharp`` — 125.5K rows, ``programming_language == "C#"``
  for every row (≈100 % hit-rate, no language filter needed), Alpaca
  ``instruction`` / ``output`` schema. Crucially each ``output`` contains a
  **natural-language walkthrough of the control flow** ("Here's some sample code…
  This code uses `try`/`catch`… The recommended technique depends on…") — i.e. an
  instruction paired with a step-by-step rationale, the CoT-style signal we want.
  126 MB parquet ⇒ cheap to *stream* (never fully materialised).

The loader is **source-configurable** (``_SOURCES`` registry). Documented additional
fallback: synthesize compile-verified C# instruction+CoT via a strong local model,
verified by ``OllamaBenchmarks/scripts/coding_tasks/task_runner.py`` (not implemented
here; see B3-INTEGRATION.md).

Holdout & contamination rules
-----------------------------
* **Family-stratified 80/10/10 by problem**: each example is assigned to
  train/val/test by a deterministic hash of its normalised instruction (stable
  across runs and streaming-friendly — no need to materialise the corpus). The hash
  is family-independent, so every task family (see ``csharp_data._FAMILIES``) is
  split ≈80/10/10. Whole problems are held out, never split mid-problem.
* **Never emit a benchmark problem**: the 50 held-out
  ``OllamaBenchmarks/.../tasks/NN_*.yaml`` execution-eval problems are excluded by
  a signature/prompt dedup guard (``_load_bench_signatures`` / ``_is_contaminated``).

Public API
----------
* ``csharp_train_stream(tokenizer, seq_len, split="train", source=..., return_labels=False, ...)``
  Iterator of ``[seq_len]`` int64 input-id tensors (``return_labels=False`` — matches
  ``cpt_token_stream``) or ``(input_ids, labels)`` tuples of ``[seq_len]`` int64 tensors
  with prompt positions masked to ``-100`` (``return_labels=True`` — completion-masked
  CE, matches ``masking.build_labels``). Windows are GPT-packed across examples;
  the completion mask is preserved through packing.
* ``load_csharp_holdout(tokenizer, seq_len, split="val", n_seqs=..., ...)``
  Materialised list of ``n_seqs`` windows for the val/test holdout (checkpoint
  selection / final test).

CPU-only: no model load, no GPU op, no PPL. Safe to import and smoke-test on CPU.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import os
import re
import sys
from typing import Iterator, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from masking import build_labels  # noqa: E402  completion-only label helper (reused)

# Reuse the canonical .NET task-family taxonomy from csharp_data. Import lazily so
# this module stays lean (csharp_data pulls in transformers at import time); fall
# back to a local mirror if unavailable.
try:  # pragma: no cover - trivial import guard
    from csharp_data import _FAMILIES as _CSHARP_FAMILIES
except Exception:  # pragma: no cover
    _CSHARP_FAMILIES = [
        "aspnet", "async", "blazor", "efcore", "linq",
        "masstransit", "patterns", "vertical", "xunit",
    ]

# ---------------------------------------------------------------------------
# Held-out benchmark task specs (execution eval — NEVER emit these as training).
# ---------------------------------------------------------------------------
_DEFAULT_BENCH_TASKS_DIR = r"E:/Development/OllamaBenchmarks/scripts/coding_tasks/tasks"


# ---------------------------------------------------------------------------
# Source registry (source-configurable, per B3 design).
# ---------------------------------------------------------------------------
_SOURCES: dict[str, dict] = {
    # DEFAULT: dense C# (≈100% hit-rate), instruction + NL rationale (CoT-style).
    "tiny-codes-csharp": {
        "hf_id": "layoric/tiny-codes-alpaca-csharp",
        "config": None,
        "split": "train",
        "instruction_field": "instruction",
        "response_field": "output",
        "lang_field": "programming_language",  # already "C#" for every row
        "needs_csharp_filter": False,
    },
    # FALLBACK: big multilingual instruction set, C# content-filtered (low hit-rate).
    "opencodeinstruct": {
        "hf_id": "nvidia/OpenCodeInstruct",
        "config": None,
        "split": "train",
        "instruction_field": "input",
        "response_field": "output",
        "lang_field": None,
        "needs_csharp_filter": True,
    },
}

_DEFAULT_SOURCE = "tiny-codes-csharp"


# ---------------------------------------------------------------------------
# C# content detection (only used when a source lacks a language tag).
# ---------------------------------------------------------------------------
_CSHARP_FENCE = re.compile(r"```\s*(?:csharp|c#|cs)\b", re.IGNORECASE)
_CSHARP_MARKERS = (
    "using System",
    "namespace ",
    "public class",
    "Console.WriteLine",
    "public async Task",
    "static void Main",
    "public record",
    "IActionResult",
    "[Fact]",
    "async Task",
)


def _looks_csharp(text: str) -> bool:
    """Heuristic C# detector for language-untagged sources (fence or ≥2 markers)."""
    if _CSHARP_FENCE.search(text):
        return True
    hits = sum(1 for m in _CSHARP_MARKERS if m in text)
    return hits >= 2


# ---------------------------------------------------------------------------
# Family detection (text-based; mirrors csharp_data's basename taxonomy).
# ---------------------------------------------------------------------------
_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "aspnet": ("asp.net", "[apicontroller]", "iactionresult", "controllerbase",
               "app.mapget", "app.mappost", "minimal api", "webapplication", "httpclient"),
    "efcore": ("dbcontext", "entity framework", "efcore", "ef core", "onmodelcreating",
               "tolistasync", ".include(", "executeupdate", "dbset<"),
    "blazor": ("blazor", "@code", "rendermode", "renderfragment", "[parameter]",
               ".razor", "interactiveserver", "cascadingparameter"),
    "masstransit": ("masstransit", "iconsumer", "consumecontext", "ibus",
                    "ipublishendpoint", "saga", "statemachine"),
    "xunit": ("xunit", "[fact]", "[theory]", "[memberdata]", "nsubstitute",
              "substitute.for", "iclassfixture", "assert."),
    "linq": ("linq", ".groupby(", ".selectmany(", ".orderbydescending(", ".aggregate(",
             ".todictionary(", ".tolookup(", "from x in", "select new"),
    "async": ("cancellationtoken", "valuetask", "semaphoreslim", "channel<",
              "task.whenall", "await task", "configureawait"),
    "patterns": ("record ", "switch expression", "pattern match", "timeprovider",
                 "with { ", "is not null", "positional record"),
    "vertical": ("cqrs", "vertical slice", "irequesthandler", "mediator", " isender",
                 "command handler", "query handler"),
}
# Only families present in the reused taxonomy are eligible.
_FAMILY_KEYWORDS = {k: v for k, v in _FAMILY_KEYWORDS.items() if k in set(_CSHARP_FAMILIES)}


def _detect_family_text(text: str) -> str:
    """Map an example (instruction+response) to a task family, or 'other'.

    Keyword-scored so that the family-stratified split works for any source, not
    just modern-.NET corpora. Generic C# (e.g. tiny-codes) mostly lands in 'other'.
    """
    low = text.lower()
    best_fam = "other"
    best_hits = 0
    for fam, kws in _FAMILY_KEYWORDS.items():
        hits = sum(1 for kw in kws if kw in low)
        if hits > best_hits:
            best_hits = hits
            best_fam = fam
    return best_fam if best_hits > 0 else "other"


# ---------------------------------------------------------------------------
# Deterministic, streaming-friendly 80/10/10 split assignment.
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Lowercase + whitespace-collapse for stable hashing / substring matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _split_of(instruction: str) -> str:
    """Assign train/val/test by a stable hash of the normalised instruction.

    Family-independent and deterministic ⇒ every family is split ≈80/10/10 and the
    same problem always lands in the same split (streaming-safe, no materialisation).
    """
    h = int(hashlib.blake2b(_norm(instruction).encode("utf-8"), digest_size=8).hexdigest(), 16) % 10
    if h < 8:
        return "train"
    if h == 8:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# Benchmark contamination guard (never emit the 50 held-out eval problems).
# ---------------------------------------------------------------------------
# Common .NET identifiers that must NOT count as distinctive benchmark signatures.
_IDENT_STOPLIST = frozenset({
    "Console", "System", "String", "Program", "DateTime", "Exception", "Microsoft",
    "AspNetCore", "Substitute", "Assert", "Threading", "Generic", "Collections",
    "IActionResult", "Task", "CancellationToken", "IEnumerable", "AwesomeAssertions",
    "NSubstitute", "Serialization", "ReadLine", "WriteLine", "FormatException",
    "InvalidOperationException", "ArgumentNullException", "Enumerable",
})
_IDENT_RE = re.compile(r"[A-Z][A-Za-z0-9]{7,}")
_PROMPT_BLOCK_RE = re.compile(r"\nprompt:\s*\|\s*\n(.*?)\ntest_code:", re.DOTALL)


def _load_bench_signatures(tasks_dir: str) -> dict:
    """Build a dedup signature set from the 50 held-out benchmark task YAMLs.

    Returns ``{"per_task": [set[str], ...], "prompts": [str, ...]}`` where each
    per-task set holds distinctive CamelCase identifiers (class/record/method names)
    and each prompt string is the normalised task prompt fingerprint.
    """
    per_task: list[set] = []
    prompts: list[str] = []
    if not tasks_dir or not os.path.isdir(tasks_dir):
        return {"per_task": per_task, "prompts": prompts}

    for path in sorted(glob.glob(os.path.join(tasks_dir, "*.yaml"))):
        base = os.path.basename(path)
        if base.startswith("_"):  # skip _smoke_test.yaml
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        idents = {i for i in _IDENT_RE.findall(raw) if i not in _IDENT_STOPLIST}
        per_task.append(idents)
        m = _PROMPT_BLOCK_RE.search(raw)
        if m:
            prompts.append(_norm(m.group(1)))
    return {"per_task": per_task, "prompts": prompts}


def _is_contaminated(instruction: str, full_text: str, sigs: dict) -> bool:
    """True if an example reproduces a held-out benchmark problem.

    Two independent signals (either trips it):
      1. A benchmark prompt fingerprint (first ~120 normalised chars) is a substring
         of the example's normalised instruction.
      2. The example's text shares ≥3 distinctive CamelCase identifiers with a single
         benchmark task (strong signal it's the same problem/solution).
    """
    instr_norm = _norm(instruction)
    for pnorm in sigs.get("prompts", ()):
        if len(pnorm) >= 40 and pnorm[:120] in instr_norm:
            return True
    for idents in sigs.get("per_task", ()):
        if not idents:
            continue
        shared = sum(1 for i in idents if i in full_text)
        if shared >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Chat-template render + completion mask (train==serve parity).
# ---------------------------------------------------------------------------
def _render_example(tokenizer, instruction: str, response: str) -> tuple[list[int], list[int]]:
    """Render one instruction/response pair to ``(input_ids, labels)``.

    The prompt is built with ``apply_chat_template(..., add_generation_prompt=True)``
    so it is bit-identical to dotLLM serve-time rendering. The response (+EOS) is the
    supervised completion; prompt positions are masked to ``-100`` (``build_labels``).
    """
    msgs = [{"role": "user", "content": instruction}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
    except Exception:
        # Tokenizer without a chat template — deterministic fallback format.
        prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"

    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        c_ids = c_ids + [tokenizer.eos_token_id]
    return build_labels(p_ids, c_ids)


# ---------------------------------------------------------------------------
# Per-example generator (one pass over a streamed source, split-filtered).
# ---------------------------------------------------------------------------
def _iter_examples(
    tokenizer,
    source: str,
    split: str,
    sigs: dict,
    seed: int,
    max_source_rows: Optional[int],
    family_filter: Optional[set],
    stats: dict,
) -> Iterator[tuple]:
    """Yield ``(input_ids, labels, family)`` for kept examples of ``split``.

    Updates ``stats`` in place (seen / lang_reject / contam / split_skip / kept /
    per-family counts) so callers can report the C#-filter hit-rate & holdout sizes.
    """
    import datasets as hf_datasets

    if source not in _SOURCES:
        raise ValueError(
            f"Unknown source {source!r}. Known: {sorted(_SOURCES)}. "
            "Add an entry to _SOURCES to plug in another HF dataset."
        )
    cfg = _SOURCES[source]
    ds = hf_datasets.load_dataset(
        cfg["hf_id"], cfg["config"], split=cfg["split"], streaming=True
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    instr_field = cfg["instruction_field"]
    resp_field = cfg["response_field"]
    needs_filter = cfg["needs_csharp_filter"]

    for row in ds:
        if max_source_rows is not None and stats["seen"] >= max_source_rows:
            break
        stats["seen"] += 1

        instruction = (row.get(instr_field) or "").strip()
        response = (row.get(resp_field) or "").strip()
        if not instruction or not response:
            continue

        if needs_filter and not _looks_csharp(response):
            stats["lang_reject"] += 1
            continue

        if _split_of(instruction) != split:
            stats["split_skip"] += 1
            continue

        full_text = instruction + "\n" + response
        if _is_contaminated(instruction, full_text, sigs):
            stats["contam"] += 1
            continue

        fam = _detect_family_text(full_text)
        if family_filter is not None and fam not in family_filter:
            continue

        input_ids, labels = _render_example(tokenizer, instruction, response)
        stats["kept"] += 1
        stats["family"][fam] = stats["family"].get(fam, 0) + 1
        yield input_ids, labels, fam


def _fresh_stats() -> dict:
    return {"seen": 0, "lang_reject": 0, "contam": 0, "split_skip": 0, "kept": 0, "family": {}}


# ---------------------------------------------------------------------------
# Public: packed, completion-masked training stream.
# ---------------------------------------------------------------------------
def csharp_train_stream(
    tokenizer,
    seq_len: int,
    split: str = "train",
    source: str = _DEFAULT_SOURCE,
    return_labels: bool = False,
    seed: int = 0,
    loop: Optional[bool] = None,
    max_source_rows: Optional[int] = None,
    family_filter: Optional[set] = None,
    tasks_dir: str = _DEFAULT_BENCH_TASKS_DIR,
    stats: Optional[dict] = None,
) -> Iterator:
    """Stream fixed-length, completion-masked C# instruction(+CoT) token windows.

    Each example is chat-templated and completion-masked, then GPT-packed into
    contiguous ``seq_len`` windows (the mask travels with the tokens through packing).

    Parameters
    ----------
    tokenizer:
        HF tokenizer compatible with the BitNet student (chat template used for parity).
    seq_len:
        Tokens per yielded window.
    split:
        ``"train"`` | ``"val"`` | ``"test"`` (family-stratified 80/10/10 by problem).
    source:
        Key in ``_SOURCES`` (default ``"tiny-codes-csharp"``; also ``"opencodeinstruct"``).
    return_labels:
        ``False`` → yield ``[seq_len]`` int64 input-id tensors (matches
        ``cpt_token_stream``). ``True`` → yield ``(input_ids, labels)`` tuples of
        ``[seq_len]`` int64 tensors, prompt positions masked to ``-100``.
    seed:
        Shuffle seed (incremented each epoch when ``loop``).
    loop:
        Re-stream the source when exhausted. Defaults to ``True`` for ``"train"``,
        ``False`` for val/test (finite holdouts).
    max_source_rows:
        Cap raw rows *read* from the source per pass (validation / smoke bound).
    family_filter:
        Optional set of family names to restrict to (e.g. reweighting a mixer bucket).
    tasks_dir:
        Directory of the 50 held-out benchmark YAMLs to dedup against.
    stats:
        Optional dict; cleared and populated with filter/holdout counters in place.

    Yields
    ------
    ``torch.Tensor`` of shape ``[seq_len]`` (int64) — or ``(input_ids, labels)`` when
    ``return_labels``.
    """
    sigs = _load_bench_signatures(tasks_dir)
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(_fresh_stats())

    if loop is None:
        loop = split == "train"

    ibuf: list[int] = []
    lbuf: list[int] = []
    cur_seed = seed

    while True:
        produced_any = False
        for input_ids, labels, _fam in _iter_examples(
            tokenizer, source, split, sigs, cur_seed, max_source_rows, family_filter, stats
        ):
            produced_any = True
            ibuf.extend(input_ids)
            lbuf.extend(labels)
            while len(ibuf) >= seq_len:
                iw = ibuf[:seq_len]
                lw = lbuf[:seq_len]
                ibuf = ibuf[seq_len:]
                lbuf = lbuf[seq_len:]
                it = torch.tensor(iw, dtype=torch.long)
                if return_labels:
                    yield it, torch.tensor(lw, dtype=torch.long)
                else:
                    yield it
        if not loop or not produced_any:
            break
        cur_seed += 1  # different shuffle each epoch


# ---------------------------------------------------------------------------
# Public: materialised val/test holdout accessor.
# ---------------------------------------------------------------------------
def load_csharp_holdout(
    tokenizer,
    seq_len: int,
    split: str = "val",
    n_seqs: int = 64,
    source: str = _DEFAULT_SOURCE,
    return_labels: bool = False,
    seed: int = 12345,
    max_source_rows: Optional[int] = None,
    family_filter: Optional[set] = None,
    tasks_dir: str = _DEFAULT_BENCH_TASKS_DIR,
    stats: Optional[dict] = None,
) -> list:
    """Materialise ``n_seqs`` holdout windows for the val/test split.

    Same shape/dtype contract as ``csharp_train_stream`` (int64 ``[seq_len]``, or
    ``(input_ids, labels)`` when ``return_labels``). ``loop`` is forced off so the
    finite holdout is not repeated.
    """
    if split not in ("val", "test"):
        raise ValueError("load_csharp_holdout expects split in {'val','test'}.")
    stream = csharp_train_stream(
        tokenizer, seq_len, split=split, source=source, return_labels=return_labels,
        seed=seed, loop=False, max_source_rows=max_source_rows,
        family_filter=family_filter, tasks_dir=tasks_dir, stats=stats,
    )
    return list(itertools.islice(stream, n_seqs))


# ---------------------------------------------------------------------------
# Smoke-test entry point (CPU only — no model, no PPL).
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Smoke-test csharp_train_data.py: stream a sample of chat-templated, "
            "completion-masked C# instruction(+CoT) windows and report shape/dtype, "
            "one decoded example, the C#-filter hit-rate, and holdout split counts. "
            "No GPU required."
        )
    )
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="Tokenizer (HF id or local path).")
    ap.add_argument("--source", default=_DEFAULT_SOURCE, choices=sorted(_SOURCES),
                    help="C# instruction source.")
    ap.add_argument("--seq-len", type=int, default=512, help="Tokens per window.")
    ap.add_argument("--n-windows", type=int, default=16, help="Windows to pull for the demo.")
    ap.add_argument("--max-source-rows", type=int, default=400,
                    help="Cap raw rows read per split (keeps the smoke test cheap).")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"[csharp_train_data] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)

    # ---- Streaming train windows (labels on) ----
    print(f"\n[csharp_train_data] === TRAIN stream (source={args.source}) ===")
    train_stats: dict = {}
    stream = csharp_train_stream(
        tok, args.seq_len, split="train", source=args.source, return_labels=True,
        loop=False, max_source_rows=args.max_source_rows, stats=train_stats,
    )
    windows = list(itertools.islice(stream, args.n_windows))

    shapes_ok = all(
        iw.shape == (args.seq_len,) and iw.dtype == torch.long
        and lw.shape == (args.seq_len,) and lw.dtype == torch.long
        for iw, lw in windows
    )
    print(f"[csharp_train_data] pulled {len(windows)} windows; shapes/dtype OK: {shapes_ok}")
    if windows:
        iw0, lw0 = windows[0]
        n_supervised = int((lw0 != -100).sum())
        print(f"[csharp_train_data] window[0]: input_ids {tuple(iw0.shape)} {iw0.dtype}, "
              f"labels {tuple(lw0.shape)} {lw0.dtype}, supervised(non -100)={n_supervised}/{args.seq_len}")
        # Decode the supervised (completion) span of window[0] as a sanity check.
        comp_ids = iw0[lw0 != -100].tolist()
        decoded = tok.decode(comp_ids, skip_special_tokens=True)
        print("[csharp_train_data] decoded completion span (window[0], truncated 400 chars):")
        print("    " + decoded[:400].replace("\n", "\n    "))

    seen = train_stats["seen"]
    kept = train_stats["kept"]
    lang_rej = train_stats["lang_reject"]
    csharp_seen = seen - lang_rej  # rows that passed the C# check (or needed none)
    hit_rate = (csharp_seen / seen * 100.0) if seen else 0.0
    print(f"\n[csharp_train_data] source rows seen (train split pass): {seen}")
    print(f"[csharp_train_data]   C#-filter hit-rate: {csharp_seen}/{seen} = {hit_rate:.1f}% "
          f"(lang_reject={lang_rej}; needs_filter={_SOURCES[args.source]['needs_csharp_filter']})")
    print(f"[csharp_train_data]   contamination drops (benchmark dedup): {train_stats['contam']}")
    print(f"[csharp_train_data]   kept examples (this pass): {kept}")
    print(f"[csharp_train_data]   per-family kept: "
          f"{dict(sorted(train_stats['family'].items(), key=lambda kv: -kv[1]))}")

    # ---- Holdout split-size sanity (val vs test, from the same capped scan) ----
    print(f"\n[csharp_train_data] === HOLDOUT split counts (max_source_rows={args.max_source_rows}) ===")
    for hsplit in ("val", "test"):
        hstats: dict = {}
        _ = load_csharp_holdout(
            tok, args.seq_len, split=hsplit, n_seqs=8, source=args.source,
            return_labels=False, max_source_rows=args.max_source_rows, stats=hstats,
        )
        print(f"[csharp_train_data]   {hsplit:4s}: kept {hstats['kept']} examples "
              f"from {hstats['seen']} rows scanned; families="
              f"{dict(sorted(hstats['family'].items(), key=lambda kv: -kv[1]))}")

    print("\n[csharp_train_data] smoke-test PASSED" if shapes_ok
          else "\n[csharp_train_data] smoke-test FAILED (shape/dtype)")


if __name__ == "__main__":
    main()
