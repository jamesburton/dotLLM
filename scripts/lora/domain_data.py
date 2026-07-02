"""Hard-domain data loaders for MoTE OOD headroom experiments.

Provides fixed-length token-sequence loaders for demanding domains
where 2B-parameter models are expected to show high perplexity relative to
their chat-distribution training:

  * **math**    — formal math reasoning with step-by-step solutions
                  Primary: ``TIGER-Lab/MathInstruct`` (instruction+output, cached).
                  Falls back: ``open-r1/OpenR1-Math-220k`` (problem+solution, cached).
  * **pg19**    — 19th-century British library books (dense, archaic literary prose;
                  sharply OOD vs modern chat in both vocabulary and style)
                  Primary: ``deepmind/pg19`` (text field, cached).
                  Falls back: ``emozilla/pg19-test`` (streaming).
  * **wiki_de** — German Wikipedia (non-English; base trained on English-dominant
                  chat corpus, so non-English is sharply OOD)
                  Primary: ``wikimedia/wikipedia`` 20231101.de (streaming, cached).
  * **wiki_ja** — Japanese Wikipedia (non-Latin CJK; expected sharply higher PPL
                  due to script distance and sparse training representation)
                  Primary: ``wikimedia/wikipedia`` 20231101.ja (streaming, cached).

Public API
----------
* ``load_domain_sequences(domain, tokenizer, n_seqs, seq_len)``
  Returns a list of ``n_seqs`` fixed-length int64 token-ID tensors of shape
  ``[seq_len]``, suitable for LM PPL evaluation.

  ``domain`` is one of: ``"math"``, ``"pg19"``, ``"wiki_de"``, ``"wiki_ja"``.

Headroom measurement (run as __main__)
---------------------------------------
  python scripts/lora/domain_data.py \\
      [--base microsoft/bitnet-b1.58-2B-4T-bf16] \\
      [--device cuda] \\
      [--n-seqs 98] \\
      [--seq-len 512] \\
      [--report path/to/report.md]

Measures dense BitNet val-PPL on all three hard domains plus the
``no_robots`` chat baseline (training-distribution reference, PPL ≈ 13.44)
and prints per-domain results with headroom verdict.

Issue reference: (#117)
"""

# Windows: suppress dynamo/Triton compilation errors.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import gc
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Internal helpers: text extraction per domain
# ---------------------------------------------------------------------------


def _extract_text_math(row: dict) -> str:
    """Combine instruction + output fields for math reasoning text."""
    parts = []
    for field in ("instruction", "problem", "question"):
        v = row.get(field, "")
        if v and isinstance(v, str):
            parts.append(v.strip())
            break
    for field in ("output", "solution", "answer"):
        v = row.get(field, "")
        if v and isinstance(v, str):
            parts.append(v.strip())
            break
    # OpenR1-style: 'messages' list with role/content dicts
    if not parts and "messages" in row:
        msgs = row["messages"]
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    c = m.get("content", "")
                    if c:
                        parts.append(str(c).strip())
    return "\n\n".join(parts)


def _extract_text_physics(row: dict) -> str:
    """Extract dense physics QA text (camel-ai style or generic)."""
    # camel-ai/physics: message_1 (problem) + message_2 (detailed answer)
    for fields in [("message_1", "message_2"), ("question", "answer"), ("input", "output")]:
        texts = []
        for f in fields:
            v = row.get(f, "")
            if v and isinstance(v, str):
                texts.append(v.strip())
        if len(texts) == len(fields):
            return "\n\n".join(texts)
    # Generic: concatenate all non-empty string fields
    return " ".join(str(v) for v in row.values() if isinstance(v, str) and v.strip())


def _extract_text_wiki(row: dict) -> str:
    """Extract Wikipedia article text."""
    return row.get("text", row.get("content", "")) or ""


# ---------------------------------------------------------------------------
# Token-ID collection helpers
# ---------------------------------------------------------------------------


def _collect_ids_from_dataset(
    ds_iter,
    extract_fn,
    tokenizer,
    needed: int,
    label: str,
) -> list[int]:
    """Tokenize rows from an iterable dataset until ``needed`` token IDs collected."""
    all_ids: list[int] = []
    for row in ds_iter:
        text = extract_fn(row)
        if not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(enc)
        if len(all_ids) >= needed:
            break
    print(f"[domain_data:{label}] collected {len(all_ids)} tokens")
    return all_ids


def _ids_to_sequences(
    all_ids: list[int],
    n_seqs: int,
    seq_len: int,
    label: str,
) -> list:
    """Slice a flat token-ID list into fixed-length torch tensors."""
    if len(all_ids) < seq_len:
        raise RuntimeError(
            f"[domain_data:{label}] not enough tokens: "
            f"got {len(all_ids)}, need >= {seq_len}."
        )
    seqs: list = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break
    print(f"[domain_data:{label}] produced {len(seqs)} sequences x {seq_len} tokens")
    return seqs


# ---------------------------------------------------------------------------
# Mixed-language corpus loader (JA + EN) for cross-lingual MoTE probes
# ---------------------------------------------------------------------------


def load_mixed_ja_en_sequences(
    tokenizer,
    n_seqs: int = 4000,
    seq_len: int = 512,
    ja_frac: float = 0.6,
) -> tuple:
    """Build a mixed JA (wiki_ja) + EN (no_robots train) training corpus.

    Parameters
    ----------
    tokenizer:
        HuggingFace tokenizer.
    n_seqs:
        Total number of fixed-length sequences to produce.
    seq_len:
        Token sequence length.
    ja_frac:
        Fraction of sequences that should be Japanese (default 0.6 = 60%).

    Returns
    -------
    ``(seqs, labels)`` where ``seqs`` is a list of :class:`torch.Tensor`
    of shape ``[seq_len]`` and ``labels`` is a parallel list of ``"ja"`` or
    ``"en"`` strings identifying each sequence's source language.
    """
    n_ja = int(n_seqs * ja_frac)
    n_en = n_seqs - n_ja
    needed_ja = n_ja * seq_len + seq_len
    needed_en = n_en * seq_len + seq_len

    # -- JA: wikimedia/wikipedia 20231101.ja (streaming) --
    print(f"[domain_data:mix] loading ~{n_ja} JA sequences from wiki_ja ...")
    ja_ids: list[int] = []
    try:
        ds_ja = load_dataset(
            "wikimedia/wikipedia", "20231101.ja", split="train", streaming=True
        )
        for row in ds_ja:
            text = _extract_text_wiki(row)
            if len(text.strip()) < 200:
                continue
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            ja_ids.extend(enc)
            if len(ja_ids) >= needed_ja:
                break
        print(f"[domain_data:mix] wiki_ja collected {len(ja_ids)} tokens")
    except Exception as exc:
        print(f"[domain_data:mix] wiki_ja failed ({exc}); JA corpus will be truncated")

    # -- EN: HuggingFaceH4/no_robots train split (cached) --
    print(f"[domain_data:mix] loading ~{n_en} EN sequences from no_robots train ...")
    en_ids: list[int] = []
    try:
        ds_en = load_dataset("HuggingFaceH4/no_robots", split="train")
        for row in ds_en:
            if "messages" in row:
                text = " ".join(
                    m["content"] for m in row["messages"] if m.get("content")
                )
            elif "text" in row:
                text = row["text"]
            else:
                text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            en_ids.extend(enc)
            if len(en_ids) >= needed_en:
                break
        print(f"[domain_data:mix] no_robots collected {len(en_ids)} tokens")
    except Exception as exc:
        print(f"[domain_data:mix] no_robots failed ({exc}); EN corpus will be truncated")

    ja_seqs = _ids_to_sequences(ja_ids, n_ja, seq_len, "ja_train")
    en_seqs = _ids_to_sequences(en_ids, n_en, seq_len, "en_train")

    # Interleave at ~3:2 JA:EN ratio so the mix is roughly uniform across the epoch
    seqs: list = []
    labels: list[str] = []
    ja_i = en_i = 0
    while ja_i < len(ja_seqs) or en_i < len(en_seqs):
        for _ in range(3):
            if ja_i < len(ja_seqs):
                seqs.append(ja_seqs[ja_i])
                labels.append("ja")
                ja_i += 1
        for _ in range(2):
            if en_i < len(en_seqs):
                seqs.append(en_seqs[en_i])
                labels.append("en")
                en_i += 1

    n_ja_actual = labels.count("ja")
    n_en_actual = labels.count("en")
    print(
        f"[domain_data:mix] mixed corpus: {len(seqs)} seqs total "
        f"({n_ja_actual} JA = {100*n_ja_actual//max(len(seqs),1)}%, "
        f"{n_en_actual} EN = {100*n_en_actual//max(len(seqs),1)}%)"
    )
    return seqs, labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Tracks which HF dataset ID was actually used, per domain.
_DOMAIN_DATASET_USED: dict[str, str] = {}


def load_domain_sequences(
    domain: str,
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
) -> list:
    """Load hard-domain text and tokenize into fixed-length sequences.

    Parameters
    ----------
    domain:
        One of ``"math"``, ``"pg19"``, ``"wiki_de"``.
    tokenizer:
        A HuggingFace tokenizer compatible with the BitNet model.
    n_seqs:
        Number of fixed-length sequences to return.
    seq_len:
        Token sequence length.

    Returns
    -------
    A list of exactly ``n_seqs`` :class:`torch.Tensor` objects of shape
    ``[seq_len]`` (dtype int64).
    """
    needed = n_seqs * seq_len + seq_len

    if domain == "math":
        return _load_math_sequences(tokenizer, n_seqs, seq_len, needed)
    elif domain == "pg19":
        return _load_pg19_sequences(tokenizer, n_seqs, seq_len, needed)
    elif domain == "wiki_de":
        return _load_wiki_de_sequences(tokenizer, n_seqs, seq_len, needed)
    elif domain == "wiki_ja":
        return _load_wiki_ja_sequences(tokenizer, n_seqs, seq_len, needed)
    else:
        raise ValueError(f"Unknown domain {domain!r}. Choose: math, pg19, wiki_de, wiki_ja")


def _load_math_sequences(tokenizer, n_seqs, seq_len, needed) -> list:
    """Load formal math reasoning text.

    Priority order:
    1. ``open-r1/OpenR1-Math-220k`` — long chain-of-thought reasoning with LaTeX;
       preferred because solutions are genuinely multi-step and symbolically dense
       (not simple MCQ template completions like MathInstruct).
    2. ``TIGER-Lab/MathInstruct`` — broader coverage fallback; note that many
       entries are short MCQ answers that are *easy* for the model (PPL reversed).
    3. ``lighteval/MATH`` — competition-level math, streaming fallback.
    """
    label = "math"
    all_ids: list[int] = []

    # Strategy 1: open-r1/OpenR1-Math-220k — long CoT with genuine LaTeX math
    # Use streaming=True to avoid slow parquet→arrow generation step.
    try:
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)
        _DOMAIN_DATASET_USED["math"] = "open-r1/OpenR1-Math-220k (long CoT + LaTeX, streaming)"
        all_ids = _collect_ids_from_dataset(ds, _extract_text_math, tokenizer, needed, label)
    except Exception as exc:
        print(f"[domain_data:math] OpenR1 unavailable ({exc}); trying MathInstruct ...")
        all_ids = []

    # Strategy 2: TIGER-Lab/MathInstruct (simpler MCQ text — PPL may be reversed)
    if len(all_ids) < needed:
        try:
            ds2 = load_dataset("TIGER-Lab/MathInstruct", split="train")
            _DOMAIN_DATASET_USED["math"] = "TIGER-Lab/MathInstruct (instruction+output, cached)"
            extra = _collect_ids_from_dataset(ds2, _extract_text_math, tokenizer, needed - len(all_ids), label)
            all_ids.extend(extra)
        except Exception as exc2:
            print(f"[domain_data:math] MathInstruct unavailable ({exc2})")

    # Strategy 3: lighteval/MATH streaming (competition math, small download)
    if len(all_ids) < needed:
        try:
            ds3 = load_dataset("lighteval/MATH", split="train", streaming=True)
            _DOMAIN_DATASET_USED["math"] = "lighteval/MATH (competition math, streaming)"
            extra3 = _collect_ids_from_dataset(ds3, _extract_text_math, tokenizer, needed - len(all_ids), label)
            all_ids.extend(extra3)
        except Exception as exc3:
            print(f"[domain_data:math] lighteval/MATH unavailable ({exc3})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _extract_text_pg19(row: dict) -> str:
    """Extract full book text from PG19 rows."""
    return row.get("text", row.get("content", "")) or ""


def _load_pg19_sequences(tokenizer, n_seqs, seq_len, needed) -> list:
    """Load 19th-century book text from deepmind/pg19 (cached).

    PG19 is a dataset of 19th-century British library books from Project Gutenberg.
    It uses archaic vocabulary, complex sentence structures, and literary prose
    that is sharply OOD relative to modern instruction-following chat corpora.
    """
    label = "pg19"
    all_ids: list[int] = []

    # Strategy 1: deepmind/pg19 train split (cached)
    try:
        ds = load_dataset("deepmind/pg19", split="train", streaming=True)
        _DOMAIN_DATASET_USED["pg19"] = "deepmind/pg19 (19th-century books, train, cached)"
        all_ids = _collect_ids_from_dataset(ds, _extract_text_pg19, tokenizer, needed, label)
    except Exception as exc:
        print(f"[domain_data:pg19] deepmind/pg19 unavailable ({exc}); trying emozilla/pg19-test ...")
        all_ids = []

    # Strategy 2: emozilla/pg19-test (smaller, also cached)
    if len(all_ids) < needed:
        try:
            ds2 = load_dataset("emozilla/pg19-test", split="test", streaming=True)
            _DOMAIN_DATASET_USED["pg19"] = "emozilla/pg19-test (19th-century books, test, cached)"
            extra = _collect_ids_from_dataset(
                ds2, _extract_text_pg19, tokenizer, needed - len(all_ids), label
            )
            all_ids.extend(extra)
        except Exception as exc2:
            print(f"[domain_data:pg19] emozilla/pg19-test also unavailable ({exc2})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _load_wiki_de_sequences(tokenizer, n_seqs, seq_len, needed) -> list:
    label = "wiki_de"
    all_ids: list[int] = []

    # wikimedia/wikipedia 20231101.de — cached, use streaming to avoid memory overhead
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.de",
            split="train",
            streaming=True,
        )
        _DOMAIN_DATASET_USED["wiki_de"] = "wikimedia/wikipedia 20231101.de (streaming, cached)"
        all_ids = _collect_ids_from_dataset(ds, _extract_text_wiki, tokenizer, needed, label)
    except Exception as exc:
        print(f"[domain_data:wiki_de] wikimedia/wikipedia de unavailable ({exc})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


def _load_wiki_ja_sequences(tokenizer, n_seqs, seq_len, needed) -> list:
    """Load Japanese Wikipedia text (20231101.ja, streaming).

    Japanese is a non-Latin, CJK script language. The BitNet tokenizer
    (derived from a Llama/Mistral-family BPE vocabulary) is heavily
    English-centric, so CJK characters are expected to be split into many
    byte-fallback tokens — artificially increasing token count per character.
    This makes PPL potentially inflated vs. a native CJK tokenizer, which is
    noted in the cross-lingual verdict.
    """
    label = "wiki_ja"
    all_ids: list[int] = []

    # Filter out stub/very short articles (common in JA Wikipedia)
    def _is_substantive(row: dict) -> bool:
        text = _extract_text_wiki(row)
        return len(text.strip()) >= 200

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.ja",
            split="train",
            streaming=True,
        )
        _DOMAIN_DATASET_USED["wiki_ja"] = "wikimedia/wikipedia 20231101.ja (streaming, cached)"
        for row in ds:
            if not _is_substantive(row):
                continue
            text = _extract_text_wiki(row)
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            all_ids.extend(enc)
            if len(all_ids) >= needed:
                break
        print(f"[domain_data:wiki_ja] collected {len(all_ids)} tokens")
    except Exception as exc:
        print(f"[domain_data:wiki_ja] wikimedia/wikipedia ja unavailable ({exc})")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, label)


# ---------------------------------------------------------------------------
# PPL helpers (mirrors code_data.py)
# ---------------------------------------------------------------------------


def _ppl_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_nll / total_tokens, 100.0))


def _eval_ppl(
    model: torch.nn.Module,
    seqs: list,
    device: torch.device,
    label: str,
    sample_n: int = 3,
) -> tuple[float, list[float]]:
    """Compute aggregate PPL; also print per-seq PPL for the first ``sample_n`` seqs."""
    total_nll = 0.0
    total_tokens = 0
    per_seq: list[float] = []
    actual_vocab: Optional[int] = None

    with torch.no_grad():
        for idx, seq in enumerate(seqs):
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
            per_seq.append(seq_ppl)

            if idx < sample_n:
                print(f"  [{label}] seq[{idx}] PPL = {seq_ppl:.2f}")

            total_nll += nll
            total_tokens += n_pred

    agg = _ppl_from_nll(total_nll, total_tokens)
    print(f"[{label}] aggregate PPL over {len(seqs)} seqs = {agg:.3f}")
    return agg, per_seq


def _load_no_robots_sequences(tokenizer, n_seqs: int, seq_len: int) -> list:
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
        print(f"[domain_data] no_robots: {len(seqs)} seqs x {seq_len} tokens (mote_train)")
        return seqs
    except Exception as exc:
        print(f"[domain_data] mote_train unavailable ({exc}); using inline loader")

    import datasets as hf_datasets
    ds = hf_datasets.load_dataset("HuggingFaceH4/no_robots", split="test")
    all_ids: list[int] = []
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
    print(f"[domain_data] no_robots: {len(seqs)} seqs x {seq_len} tokens (inline)")
    return seqs


# ---------------------------------------------------------------------------
# Main — headroom scan
# ---------------------------------------------------------------------------

CHAT_REFERENCE_PPL = 13.44  # known dense BitNet PPL on no_robots test (prior scan)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Hard-domain OOD headroom scan: dense BitNet PPL on math / physics / wiki_de "
            "vs chat (no_robots). Issue #117."
        )
    )
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-seqs", type=int, default=98)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument(
        "--domains",
        default="math,pg19,wiki_de",
        help="Comma-separated list of domains to evaluate (default: math,pg19,wiki_de). Also supports: wiki_ja",
    )
    ap.add_argument("--report", default=None)
    ap.add_argument(
        "--skip-chat",
        action="store_true",
        help="Skip the no_robots baseline (use stored reference PPL 13.44).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    # ------------------------------------------------------------------
    # 1. Tokenizer
    # ------------------------------------------------------------------
    print(f"[domain_data] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)

    # ------------------------------------------------------------------
    # 1b. Tokenizer CJK analysis (informational — for Japanese headroom)
    # ------------------------------------------------------------------
    _ja_sample = "日本語のウィキペディアは、さまざまなトピックについて詳細な記事を提供しています。"
    _ja_chars = len(_ja_sample)
    _ja_tok_ids = tok(_ja_sample, add_special_tokens=False)["input_ids"]
    _ja_tok_count = len(_ja_tok_ids)
    _ja_tpc = _ja_tok_count / _ja_chars
    print(
        f"[domain_data] JA tokenizer probe: {_ja_chars} chars → {_ja_tok_count} tokens "
        f"({_ja_tpc:.2f} tok/char); "
        f"{'HIGH fragmentation — BPE byte-fallback' if _ja_tpc > 1.5 else 'moderate fragmentation'}"
    )

    # ------------------------------------------------------------------
    # 2. Build all corpora on CPU (token IDs only, negligible memory)
    # ------------------------------------------------------------------
    domain_seqs: dict[str, list] = {}
    for dom in domains:
        print(f"\n[domain_data] loading domain: {dom!r} ...")
        domain_seqs[dom] = load_domain_sequences(dom, tok, n_seqs=args.n_seqs, seq_len=args.seq_len)

    if not args.skip_chat:
        print("\n[domain_data] loading chat baseline (no_robots test) ...")
        chat_seqs = _load_no_robots_sequences(tok, n_seqs=args.n_seqs, seq_len=args.seq_len)
    else:
        chat_seqs = []

    # ------------------------------------------------------------------
    # 3. Load dense model (bf16, single-pass, then free)
    # ------------------------------------------------------------------
    print(f"\n[domain_data] loading dense BitNet on {device} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        print(
            f"[domain_data] VRAM after model load: "
            f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB"
        )

    # ------------------------------------------------------------------
    # 4. Evaluate each domain
    # ------------------------------------------------------------------
    ppl_results: dict[str, float] = {}
    per_seq_results: dict[str, list[float]] = {}

    for dom in domains:
        print(f"\n=== DOMAIN: {dom.upper()} (sample per-seq PPLs) ===")
        ppl, per_seq = _eval_ppl(model, domain_seqs[dom], device, dom, sample_n=3)
        ppl_results[dom] = ppl
        per_seq_results[dom] = per_seq

    if chat_seqs:
        print("\n=== CHAT (no_robots test) ===")
        ppl_chat, _ = _eval_ppl(model, chat_seqs, device, "chat", sample_n=0)
    else:
        ppl_chat = CHAT_REFERENCE_PPL
        print(f"[domain_data] using stored chat reference PPL: {ppl_chat:.2f}")

    # ------------------------------------------------------------------
    # 5. Peak VRAM + free model
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
    # 6. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("[domain_data] === HARD-DOMAIN HEADROOM SCAN RESULTS ===")
    print(f"  Chat baseline (no_robots test):  {ppl_chat:.3f}")
    print()

    winner_dom: Optional[str] = None
    winner_ppl: float = 0.0

    for dom in domains:
        ppl = ppl_results[dom]
        ratio = ppl / ppl_chat if ppl_chat > 0 else float("inf")
        tag = "HEADROOM" if ppl > ppl_chat * 1.5 else ("MILD" if ppl > ppl_chat else "REVERSED")
        ds_label = _DOMAIN_DATASET_USED.get(dom, "unknown")
        print(f"  [{dom:8s}] PPL = {ppl:8.3f}  ratio = {ratio:.2f}x  [{tag}]  ({ds_label})")
        if ppl > winner_ppl:
            winner_ppl = ppl
            winner_dom = dom

    print()
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  n_seqs={args.n_seqs}, seq_len={args.seq_len}")

    if winner_dom and winner_ppl > ppl_chat * 1.5:
        print(
            f"\n  WINNER: '{winner_dom}' with PPL={winner_ppl:.2f} "
            f"({winner_ppl/ppl_chat:.1f}x chat).  "
            f"Dataset: {_DOMAIN_DATASET_USED.get(winner_dom, 'unknown')}"
        )
        print(f"  => MoTE broadening experiment on '{winner_dom}' is JUSTIFIED.")
    elif winner_dom:
        print(
            f"\n  BEST: '{winner_dom}' PPL={winner_ppl:.2f} "
            f"({winner_ppl/ppl_chat:.1f}x chat) — gap MARGINAL (target >1.5x)."
        )
        print("  => Base may be broadly strong; consider harder slices or longer contexts.")
    else:
        print("\n  No domain exceeded chat baseline. Base appears broadly strong.")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 7. Append to report file
    # ------------------------------------------------------------------
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        rows_md = "\n".join(
            f"| {dom:8s} | **{ppl_results[dom]:.2f}** | {ppl_results[dom]/ppl_chat:.2f}x |"
            for dom in domains
        )
        ds_notes = "\n".join(
            f"- `{dom}`: `{_DOMAIN_DATASET_USED.get(dom, 'unknown')}`"
            for dom in domains
        )
        winner_verdict = (
            f"**WINNER: `{winner_dom}`** — PPL={winner_ppl:.2f} ({winner_ppl/ppl_chat:.1f}x chat). "
            f"Dataset: `{_DOMAIN_DATASET_USED.get(winner_dom or '', 'N/A')}`. "
            "MoTE broadening experiment JUSTIFIED."
            if winner_dom and winner_ppl > ppl_chat * 1.5
            else f"Best domain `{winner_dom}` PPL={winner_ppl:.2f} ({winner_ppl/ppl_chat:.1f}x chat) — "
            "gap marginal. Base broadly strong across tested domains."
        )
        sample_ppls = "\n".join(
            "- `{}` seq[0..2]: {}".format(
                dom,
                ", ".join(f"{p:.1f}" for p in per_seq_results[dom][:3]),
            )
            for dom in domains
        )

        # Cross-lingual verdict logic (for wiki_de + wiki_ja comparison)
        crosslingual_note = ""
        if "wiki_de" in ppl_results and "wiki_ja" in ppl_results:
            ppl_de = ppl_results["wiki_de"]
            ppl_ja = ppl_results["wiki_ja"]
            if ppl_ja > ppl_de * 1.5:
                xverdict = "PROMISING — PPL scales with language distance from English (JA >> DE). Cross-lingual MoTE experiment recommended."
            elif ppl_ja > ppl_de * 1.1:
                xverdict = "MILD — JA slightly above DE but not dramatically so. Tokenizer fragmentation may inflate JA PPL."
            else:
                xverdict = "FLAT — JA ≈ DE, no clear language-distance scaling. Tokenizer artifact likely. Recommend moving to downstream task-accuracy instead."
            crosslingual_note = (
                f"\n### Cross-lingual scaling verdict\n"
                f"- DE PPL: {ppl_de:.2f} ({ppl_de/ppl_chat:.2f}x chat)\n"
                f"- JA PPL: {ppl_ja:.2f} ({ppl_ja/ppl_chat:.2f}x chat)\n"
                f"- JA/DE ratio: {ppl_ja/ppl_de:.2f}x\n"
                f"- JA tokenizer: {_ja_tok_count} tokens / {_ja_chars} chars = {_ja_tpc:.2f} tok/char "
                f"({'HIGH fragmentation — BPE byte-fallback' if _ja_tpc > 1.5 else 'moderate'})\n"
                f"- **Verdict: {xverdict}**\n"
            )

        entry = f"""
## cross-lingual spike (de + ja)

**Date:** 2026-07-01
**Issue:** #117
**Model:** `{args.base}` (dense, bf16, {args.device})
**n_seqs:** {args.n_seqs} × {args.seq_len} tokens per domain

### Datasets used
{ds_notes}

### Results

| Domain   | Val-PPL | Ratio vs chat |
|----------|---------|---------------|
| chat (no_robots test) | **{ppl_chat:.2f}** | 1.00x |
{rows_md}

**Peak VRAM:** {peak_vram_gb:.2f} GB

### Sample per-sequence PPLs (sanity check)
{sample_ppls}
{crosslingual_note}
### Verdict
{winner_verdict}
"""
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
        print(f"\n[domain_data] appended findings to {report_path}")


if __name__ == "__main__":
    main()
