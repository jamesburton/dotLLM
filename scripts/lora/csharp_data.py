"""C# / modern-.NET data loader for MoTE OOD headroom experiments.

Dataset
-------
Primary: ``E:/Development/OllamaBenchmarks/results/coding-generated/*/*.cs``
  ~2,852 generated C# files spanning modern .NET 10 task families:
  aspnet · async · blazor · efcore · linq · masstransit · patterns · vertical · xunit

Supplementary: ``E:/Development/OllamaBenchmarks/scripts/coding_tasks/references/*.md``
  8 modern-.NET API reference docs (blazor_net10, efcore_10, masstransit_v8, xunit_v3, etc.).

Public API
----------
* ``load_csharp_sequences(tokenizer, n_seqs, seq_len, corpus_dir, refs_dir)``
  Returns ``(seqs, family_labels)`` where ``seqs`` is a list of ``n_seqs``
  fixed-length ``torch.Tensor`` objects of shape ``[seq_len]`` (int64) and
  ``family_labels`` is a parallel list of family-name strings.

Headroom measurement (run as __main__)
---------------------------------------
  python scripts/lora/csharp_data.py \\
      [--base microsoft/bitnet-b1.58-2B-4T-bf16] \\
      [--device cuda] \\
      [--n-seqs 98] \\
      [--seq-len 512] \\
      [--corpus-dir E:/Development/OllamaBenchmarks/results/coding-generated] \\
      [--refs-dir E:/Development/OllamaBenchmarks/scripts/coding_tasks/references] \\
      [--report path/to/report.md]

Computes dense BitNet val-PPL on:
  (a) csharp — sampled from the local corpus above (per-family breakdown printed)
  (b) chat   — no_robots test split (training-distribution reference)

Issue reference: (#117)
"""

# Windows: suppress dynamo/Triton compilation errors (no cl.exe needed).
try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import gc
import glob
import math
import os
import random
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Corpus paths (defaults; override via CLI args or function parameters)
# ---------------------------------------------------------------------------

_DEFAULT_CORPUS_DIR = r"E:/Development/OllamaBenchmarks/results/coding-generated"
_DEFAULT_REFS_DIR = r"E:/Development/OllamaBenchmarks/scripts/coding_tasks/references"

# Canonical task-family prefixes (derived from the .cs file name conventions).
_FAMILIES = [
    "aspnet",
    "async",
    "blazor",
    "efcore",
    "linq",
    "masstransit",
    "patterns",
    "vertical",
    "xunit",
]

_SEED = 42


# ---------------------------------------------------------------------------
# Internal: family detection
# ---------------------------------------------------------------------------


def _detect_family(path: str) -> str:
    """Return the task family for a .cs file based on its basename prefix."""
    base = os.path.splitext(os.path.basename(path))[0]
    for fam in _FAMILIES:
        if base == fam or base.startswith(fam + "_"):
            return fam
    return "other"


# ---------------------------------------------------------------------------
# Internal: token collection helpers
# ---------------------------------------------------------------------------


def _collect_family_tokens(
    paths: list,
    tokenizer,
    needed: int,
    label: str,
) -> list:
    """Read and tokenize .cs files until ``needed`` token IDs are collected."""
    all_ids: list = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if len(all_ids) >= needed:
            break
    print(f"[csharp_data:{label}] collected {len(all_ids)} tokens from {len(paths)} files")
    return all_ids


def _ids_to_seqs(all_ids: list, n: int, seq_len: int, label: str) -> list:
    """Slice a flat token-ID list into ``n`` fixed-length torch tensors."""
    if len(all_ids) < seq_len:
        print(
            f"[csharp_data:{label}] WARNING: only {len(all_ids)} tokens, "
            f"need {seq_len}. Skipping family."
        )
        return []
    seqs = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n:
            break
    if len(seqs) < n:
        print(
            f"[csharp_data:{label}] only {len(seqs)} sequences available "
            f"(wanted {n}); using all."
        )
    return seqs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_csharp_sequences(
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
    corpus_dir: str = _DEFAULT_CORPUS_DIR,
    refs_dir: str = _DEFAULT_REFS_DIR,
) -> tuple:
    """Load C#/modern-.NET text and tokenize into fixed-length sequences.

    Samples representatively from all task families in the local corpus.

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer compatible with the BitNet model.
    n_seqs:
        Total number of fixed-length sequences to return (split across families).
    seq_len:
        Token sequence length (tokens per sequence).
    corpus_dir:
        Root directory containing per-model subdirectories of ``.cs`` files.
    refs_dir:
        Directory containing reference ``.md`` API docs.

    Returns
    -------
    seqs : list[torch.Tensor]
        Each tensor has shape ``[seq_len]`` and dtype int64.
    family_labels : list[str]
        Parallel list of task-family names (one per sequence in ``seqs``).
    """
    rng = random.Random(_SEED)

    # ---- 1. Discover and group .cs files by family ----
    cs_pattern = os.path.join(corpus_dir, "*", "*.cs")
    all_cs_paths = glob.glob(cs_pattern)
    if not all_cs_paths:
        raise RuntimeError(
            f"[csharp_data] No .cs files found under {corpus_dir!r}. "
            "Check the corpus_dir path."
        )
    print(f"[csharp_data] found {len(all_cs_paths)} .cs files in {corpus_dir}")

    family_paths: dict = {fam: [] for fam in _FAMILIES + ["other"]}
    for p in all_cs_paths:
        fam = _detect_family(p)
        family_paths[fam].append(p)

    # Shuffle per family for diversity (deterministic seed).
    for fam in family_paths:
        rng.shuffle(family_paths[fam])

    # ---- 2. Discover reference .md docs ----
    ref_paths: list = []
    if refs_dir and os.path.isdir(refs_dir):
        ref_paths = glob.glob(os.path.join(refs_dir, "*.md"))
        rng.shuffle(ref_paths)
        print(f"[csharp_data] found {len(ref_paths)} reference docs in {refs_dir}")
    else:
        print(f"[csharp_data] refs_dir not found ({refs_dir!r}); skipping ref docs")

    # ---- 3. Allocate n_seqs proportionally across families + ref_docs ----
    active_families = [fam for fam in _FAMILIES if family_paths.get(fam)]
    if family_paths.get("other"):
        active_families.append("other")
    all_buckets = active_families + (["ref_docs"] if ref_paths else [])
    n_buckets = len(all_buckets)

    base_per = n_seqs // n_buckets
    extras = n_seqs % n_buckets

    alloc: dict = {}
    for idx, bucket in enumerate(all_buckets):
        alloc[bucket] = base_per + (1 if idx < extras else 0)

    print(f"[csharp_data] sequence allocation: {alloc}")

    # ---- 4. Tokenize each bucket and slice into sequences ----
    all_seqs: list = []
    all_labels: list = []

    for bucket in all_buckets:
        n_target = alloc[bucket]
        needed = n_target * seq_len + seq_len

        if bucket == "ref_docs":
            ids = _collect_family_tokens(ref_paths, tokenizer, needed, "ref_docs")
        else:
            ids = _collect_family_tokens(family_paths.get(bucket, []), tokenizer, needed, bucket)

        seqs = _ids_to_seqs(ids, n_target, seq_len, bucket)
        all_seqs.extend(seqs)
        all_labels.extend([bucket] * len(seqs))

    # Shuffle together (preserving label alignment) for a mixed eval pass.
    paired = list(zip(all_seqs, all_labels))
    rng.shuffle(paired)
    all_seqs, all_labels = zip(*paired) if paired else ([], [])
    all_seqs = list(all_seqs)
    all_labels = list(all_labels)

    print(
        f"[csharp_data] total: {len(all_seqs)} C# sequences × {seq_len} tokens "
        f"across {len(set(all_labels))} buckets"
    )
    return all_seqs, all_labels


# ---------------------------------------------------------------------------
# PPL helpers (mirrors code_data.py / domain_data.py)
# ---------------------------------------------------------------------------


def _ppl_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_nll / total_tokens, 100.0))


def _eval_ppl_with_families(
    model: torch.nn.Module,
    seqs: list,
    family_labels: list,
    device: torch.device,
    sample_n: int = 3,
) -> tuple:
    """Compute aggregate PPL and per-family PPL in a single pass.

    Returns
    -------
    agg_ppl : float
    family_ppls : dict[str, float]
    per_seq_ppls : list[float]
    """
    total_nll = 0.0
    total_tokens = 0
    per_seq_ppls: list = []
    actual_vocab: Optional[int] = None

    # Per-family accumulators.
    fam_nll: dict = {}
    fam_tokens: dict = {}

    with torch.no_grad():
        for idx, (seq, fam) in enumerate(zip(seqs, family_labels)):
            seq_t = seq.unsqueeze(0).to(device)
            n_pred = seq_t.size(1) - 1

            logits = model(input_ids=seq_t).logits  # [1, T, V]
            if actual_vocab is None:
                actual_vocab = logits.size(-1)

            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, actual_vocab),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()

            seq_ppl = math.exp(min(nll / n_pred, 100.0))
            per_seq_ppls.append(seq_ppl)

            if idx < sample_n:
                print(f"  [csharp:{fam}] seq[{idx}] PPL = {seq_ppl:.2f}")

            total_nll += nll
            total_tokens += n_pred

            fam_nll[fam] = fam_nll.get(fam, 0.0) + nll
            fam_tokens[fam] = fam_tokens.get(fam, 0) + n_pred

    agg_ppl = _ppl_from_nll(total_nll, total_tokens)
    print(f"[csharp] aggregate PPL over {len(seqs)} seqs = {agg_ppl:.3f}")

    family_ppls: dict = {}
    for fam in sorted(fam_nll.keys()):
        fp = _ppl_from_nll(fam_nll[fam], fam_tokens[fam])
        family_ppls[fam] = fp
        n_fam = sum(1 for l in family_labels if l == fam)
        print(f"  [csharp:{fam:14s}] PPL = {fp:8.3f}  ({n_fam} seqs)")

    return agg_ppl, family_ppls, per_seq_ppls


def _eval_ppl_simple(
    model: torch.nn.Module,
    seqs: list,
    device: torch.device,
    label: str,
    sample_n: int = 0,
) -> float:
    """Compute aggregate PPL for a list of sequences (no per-family tracking)."""
    total_nll = 0.0
    total_tokens = 0
    actual_vocab: Optional[int] = None

    with torch.no_grad():
        for idx, seq in enumerate(seqs):
            seq_t = seq.unsqueeze(0).to(device)
            n_pred = seq_t.size(1) - 1

            logits = model(input_ids=seq_t).logits
            if actual_vocab is None:
                actual_vocab = logits.size(-1)

            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, actual_vocab),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()

            if idx < sample_n:
                print(f"  [{label}] seq[{idx}] PPL = {math.exp(min(nll / n_pred, 100.0)):.2f}")

            total_nll += nll
            total_tokens += n_pred

    agg = _ppl_from_nll(total_nll, total_tokens)
    print(f"[{label}] aggregate PPL over {len(seqs)} seqs = {agg:.3f}")
    return agg


def _load_no_robots_sequences(
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
) -> list:
    """Load no_robots test-split sequences (chat distribution baseline)."""
    try:
        from mote_train import _build_corpus

        seqs = _build_corpus(
            tokenizer=tokenizer,
            dataset_name="HuggingFaceH4/no_robots",
            dataset_config=None,
            dataset_split="test",
            max_seq_len=seq_len,
            max_sequences=n_seqs,
            tiny_random=False,
            vocab_size=tokenizer.vocab_size,
        )
        print(f"[csharp_data] no_robots: {len(seqs)} seqs × {seq_len} tokens (mote_train)")
        return seqs
    except Exception as exc:
        print(f"[csharp_data] mote_train unavailable ({exc}); using inline loader")

    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("HuggingFaceH4/no_robots", split="test")
    all_ids: list = []
    for row in ds:
        if "messages" in row:
            text = " ".join(m["content"] for m in row["messages"] if m.get("content"))
        elif "text" in row:
            text = row["text"]
        else:
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= n_seqs * seq_len:
            break

    seqs = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break
    print(f"[csharp_data] no_robots: {len(seqs)} seqs × {seq_len} tokens (inline)")
    return seqs


# ---------------------------------------------------------------------------
# Main — headroom measurement
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OOD headroom check: dense BitNet PPL on C#/modern-.NET vs chat (no_robots). "
            "Issue #117."
        )
    )
    ap.add_argument(
        "--base",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HF model id for the dense BitNet base.",
    )
    ap.add_argument("--device", default="cuda", help="Inference device: cuda or cpu.")
    ap.add_argument(
        "--n-seqs",
        type=int,
        default=98,
        help="Number of 512-token sequences for each domain (default: 98).",
    )
    ap.add_argument("--seq-len", type=int, default=512, help="Sequence length in tokens.")
    ap.add_argument(
        "--corpus-dir",
        default=_DEFAULT_CORPUS_DIR,
        help="Root of coding-generated .cs corpus.",
    )
    ap.add_argument(
        "--refs-dir",
        default=_DEFAULT_REFS_DIR,
        help="Directory with reference .md API docs.",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Path to a markdown file to append findings to.",
    )
    ap.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip no_robots baseline; use stored reference PPL 13.44.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print(f"[csharp_data] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)

    # ------------------------------------------------------------------
    # 2. Build C# corpus (all on CPU)
    # ------------------------------------------------------------------
    print("[csharp_data] building C# corpus ...")
    csharp_seqs, family_labels = load_csharp_sequences(
        tok,
        n_seqs=args.n_seqs,
        seq_len=args.seq_len,
        corpus_dir=args.corpus_dir,
        refs_dir=args.refs_dir,
    )

    # ------------------------------------------------------------------
    # 3. Build chat baseline
    # ------------------------------------------------------------------
    if not args.skip_chat:
        print("[csharp_data] loading no_robots sequences ...")
        chat_seqs = _load_no_robots_sequences(tok, n_seqs=args.n_seqs, seq_len=args.seq_len)
        chat_seqs_available = True
    else:
        chat_seqs = []
        chat_seqs_available = False

    # ------------------------------------------------------------------
    # 4. Load dense model (bf16)
    # ------------------------------------------------------------------
    print(f"[csharp_data] loading dense BitNet on {device} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        print(
            f"[csharp_data] VRAM after model load: "
            f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )

    # ------------------------------------------------------------------
    # 5. Eval C# PPL (with per-family breakdown, sample 3 per-seq)
    # ------------------------------------------------------------------
    print("\n=== C# / MODERN .NET domain (sample per-seq PPLs) ===")
    ppl_csharp, family_ppls, per_seq_ppls = _eval_ppl_with_families(
        model, csharp_seqs, family_labels, device, sample_n=3
    )

    # ------------------------------------------------------------------
    # 6. Eval chat PPL
    # ------------------------------------------------------------------
    if chat_seqs_available:
        print("\n=== CHAT domain (no_robots test) ===")
        ppl_chat = _eval_ppl_simple(model, chat_seqs, device, "chat", sample_n=0)
    else:
        ppl_chat = 13.44
        print(f"[csharp_data] using stored chat reference PPL: {ppl_chat:.2f}")

    # ------------------------------------------------------------------
    # 7. Peak VRAM + free model
    # ------------------------------------------------------------------
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_vram_gb = 0.0

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    gap = ppl_csharp / ppl_chat if ppl_chat > 0 else float("inf")
    headroom = ppl_csharp > ppl_chat * 1.5  # >50% gap = meaningful headroom

    print()
    print("=" * 64)
    print("[csharp_data] === C# HEADROOM CHECK RESULTS ===")
    print(f"  Dense PPL (C# / coding-generated corpus): {ppl_csharp:.3f}")
    print(f"  Dense PPL (chat / no_robots test):        {ppl_chat:.3f}")
    print(f"  Ratio C#/chat:                            {gap:.2f}×")
    print(f"  Peak VRAM:                                {peak_vram_gb:.2f} GB")
    print()
    print("  Per-family PPL breakdown:")
    for fam, fppl in sorted(family_ppls.items(), key=lambda kv: -kv[1]):
        ratio_f = fppl / ppl_chat if ppl_chat > 0 else float("inf")
        tag = "HIGH" if fppl > ppl_chat * 2.0 else ("MID" if fppl > ppl_chat * 1.5 else "LOW")
        print(f"    {fam:14s}: {fppl:8.3f}  ({ratio_f:.2f}× chat)  [{tag}]")

    if headroom:
        verdict = (
            f"HEADROOM CONFIRMED — C# PPL is {gap:.1f}× chat; "
            "modern-.NET MoTE experiment is justified."
        )
    elif gap < 0.8:
        verdict = (
            f"REVERSED — base is STRONGER on C# than chat (ratio {gap:.2f}×); "
            "corpus too easy for the base model."
        )
    else:
        verdict = f"NO HEADROOM — C#/chat gap is {gap:.2f}× (target >1.5×)."

    # Check for high-PPL subset even if global shows no headroom.
    high_ppl_fams = [
        (fam, fppl)
        for fam, fppl in family_ppls.items()
        if fppl > ppl_chat * 2.0
    ]
    if high_ppl_fams and not headroom:
        high_ppl_fams_str = ", ".join(
            f"{fam}={fppl:.1f}" for fam, fppl in sorted(high_ppl_fams, key=lambda kv: -kv[1])
        )
        verdict += (
            f" HIGH-PPL SUBSET exists: [{high_ppl_fams_str}] — "
            "these families are viable MoTE targets even if global gap is marginal."
        )

    print()
    print(f"  VERDICT: {verdict}")
    print(
        f"  ({len(csharp_seqs)} C# seqs, {len(chat_seqs)} chat seqs, "
        f"seq_len={args.seq_len})"
    )
    print("=" * 64)

    # ------------------------------------------------------------------
    # 9. Append to report file
    # ------------------------------------------------------------------
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        family_rows_md = "\n".join(
            f"| {fam:14s} | **{fppl:.2f}** | {fppl / ppl_chat:.2f}× |"
            for fam, fppl in sorted(family_ppls.items(), key=lambda kv: -kv[1])
        )
        sample_ppls_str = "\n".join(
            f"  - seq[{i}] ({family_labels[i]}): {per_seq_ppls[i]:.2f}"
            for i in range(min(3, len(per_seq_ppls)))
        )

        entry = f"""
## C# (modern .NET) headroom check

**Date:** 2026-07-01
**Issue:** #117
**Model:** `{args.base}` (dense, bf16, {args.device})
**Corpus:** `{args.corpus_dir}` + `{args.refs_dir}`
**Sequences:** {len(csharp_seqs)} C# × {args.seq_len} tokens; {len(chat_seqs)} chat × {args.seq_len} tokens

| Domain | Val-PPL | Ratio vs chat |
|--------|---------|---------------|
| C# (modern .NET corpus) | **{ppl_csharp:.2f}** | {gap:.2f}× |
| Chat (no_robots test)   | **{ppl_chat:.2f}** | 1.00× |

**Peak VRAM:** {peak_vram_gb:.2f} GB

### Per-family PPL breakdown

| Family | Val-PPL | Ratio vs chat |
|--------|---------|---------------|
{family_rows_md}

### Sample per-sequence PPLs (sanity check)
{sample_ppls_str}

### Verdict
{verdict}
"""
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
        print(f"[csharp_data] appended findings to {report_path}")


if __name__ == "__main__":
    main()
