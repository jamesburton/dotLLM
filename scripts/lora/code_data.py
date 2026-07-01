"""Code-domain data loader for MoTE OOD headroom experiments.

Dataset
-------
Primary: ``iamtarun/python_code_instructions_18k_alpaca``, ``split="train"``.
Extracts the ``output`` field (Python code solutions) — no network required
when already cached at ``HF_HOME``.  Fallback: ``codeparrot/github-code-clean``
streamed (requires network, no disk write).

Public API
----------
* ``load_code_sequences(tokenizer, n_seqs, seq_len)`` — returns a list of
  fixed-length ``[seq_len]`` int64 token-ID tensors suitable for LM PPL eval.

Headroom measurement (run as __main__)
---------------------------------------
  python scripts/lora/code_data.py \\
      [--base microsoft/bitnet-b1.58-2B-4T-bf16] \\
      [--device cuda] \\
      [--n-seqs 98] \\
      [--seq-len 512] \\
      [--report path/to/report.md]

Computes dense BitNet val-PPL on:
  (a) code   — Python code from the dataset above
  (b) chat   — no_robots test split (training-distribution reference)

Prints individual PPLs for 3 sample code sequences as a sanity check, then
the aggregate code PPL, the chat PPL, and a headroom verdict.

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
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Allow sibling imports (mote_train._build_corpus for the no_robots baseline).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_CODE_DATASET_USED: str = ""  # set by load_code_sequences for reporting


def load_code_sequences(
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
) -> list:
    """Load Python code and tokenize into fixed-length sequences.

    Returns a list of exactly ``n_seqs`` ``torch.Tensor`` objects of shape
    ``[seq_len]`` (dtype int64).  Each tensor is a contiguous window of tokens
    from concatenated Python code text.

    Strategy
    --------
    1. Try ``iamtarun/python_code_instructions_18k_alpaca`` (cached on disk —
       no network needed); extract ``output`` field (Python code solutions).
    2. Fall back to streaming ``codeparrot/github-code-clean`` (requires
       network; no local cache written).

    Parameters
    ----------
    tokenizer:
        A HuggingFace tokenizer compatible with the BitNet model.
    n_seqs:
        Number of fixed-length sequences to return.
    seq_len:
        Token sequence length (must be ≤ model ``max_position_embeddings``).
    """
    global _CODE_DATASET_USED
    needed = n_seqs * seq_len + seq_len  # extra window for safety

    all_ids: list = []
    dataset_label = ""

    # ---- Strategy 1: cached Python-code-instructions dataset ----
    try:
        ds = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        )
        dataset_label = (
            "iamtarun/python_code_instructions_18k_alpaca (output field, cached)"
        )
        for row in ds:
            code: str = row.get("output", "")
            if not code.strip():
                continue
            enc = tokenizer(code, add_special_tokens=False)["input_ids"]
            all_ids.extend(enc)
            if len(all_ids) >= needed:
                break
        print(
            f"[code_data] cached dataset: {len(all_ids)} tokens collected "
            f"from {dataset_label}"
        )
    except Exception as exc:
        print(f"[code_data] cached dataset unavailable ({exc}); trying streaming ...")
        all_ids = []

    # ---- Strategy 2: streamed github-code-clean (Python only) ----
    if len(all_ids) < needed:
        ds_stream = load_dataset(
            "codeparrot/github-code-clean",
            streaming=True,
            split="train",
        )
        dataset_label = "codeparrot/github-code-clean (Python, streaming)"
        for row in ds_stream:
            if row.get("language", "") != "Python":
                continue
            content: str = row.get("code", "")
            if not content.strip():
                continue
            enc = tokenizer(content, add_special_tokens=False)["input_ids"]
            all_ids.extend(enc)
            if len(all_ids) >= needed:
                break
        print(
            f"[code_data] streaming dataset: {len(all_ids)} tokens collected "
            f"from {dataset_label}"
        )

    if len(all_ids) < seq_len:
        raise RuntimeError(
            f"[code_data] Not enough tokens collected: "
            f"got {len(all_ids)}, need ≥ {seq_len}.  "
            "Check HF access."
        )

    seqs: list = []
    for i in range(0, len(all_ids) - seq_len, seq_len):
        seqs.append(torch.tensor(all_ids[i : i + seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break

    _CODE_DATASET_USED = dataset_label
    print(
        f"[code_data] loaded {len(seqs)} code sequences × {seq_len} tokens "
        f"({dataset_label})"
    )
    return seqs


# ---------------------------------------------------------------------------
# Internal helpers (PPL computation, no_robots loader)
# ---------------------------------------------------------------------------


def _ppl_from_nll(total_nll: float, total_tokens: int) -> float:
    if total_tokens == 0:
        return float("inf")
    avg = total_nll / total_tokens
    return math.exp(min(avg, 100.0))


def _load_no_robots_sequences(
    tokenizer,
    n_seqs: int = 98,
    seq_len: int = 512,
) -> list:
    """Load n_seqs sequences from HuggingFaceH4/no_robots (test split)."""
    # Use mote_train._build_corpus if available; otherwise inline equivalent.
    try:
        from mote_train import _build_corpus  # noqa: F401 (local import)
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
        print(
            f"[code_data] no_robots: {len(seqs)} sequences × {seq_len} tokens "
            f"(via mote_train._build_corpus, split=test)"
        )
        return seqs
    except Exception as exc:
        print(f"[code_data] mote_train import failed ({exc}); using inline loader")

    # Inline fallback (same logic as _build_corpus but self-contained).
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset(
        "HuggingFaceH4/no_robots", split="test"
    )
    all_ids: list = []
    for row in ds:
        if "messages" in row:
            text = " ".join(
                m["content"] for m in row["messages"] if m.get("content")
            )
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

    print(
        f"[code_data] no_robots: {len(seqs)} sequences × {seq_len} tokens "
        f"(inline loader, split=test)"
    )
    return seqs


def _eval_ppl(
    model: torch.nn.Module,
    seqs: list,
    device: torch.device,
    vocab_size: int,
    label: str,
    sample_n: int = 3,
) -> tuple[float, list]:
    """Compute PPL over seqs; also print per-sequence PPL for the first sample_n."""
    total_nll = 0.0
    total_tokens = 0
    per_seq_ppl: list = []
    # Use actual logit vocab dim (may differ from tokenizer.vocab_size by padding).
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
            per_seq_ppl.append(seq_ppl)

            if idx < sample_n:
                print(f"  [{label}] seq[{idx}] PPL = {seq_ppl:.2f}")

            total_nll += nll
            total_tokens += n_pred

    agg_ppl = _ppl_from_nll(total_nll, total_tokens)
    print(f"[{label}] aggregate PPL over {len(seqs)} seqs = {agg_ppl:.3f}")
    return agg_ppl, per_seq_ppl


# ---------------------------------------------------------------------------
# Main — headroom measurement
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OOD headroom check: dense BitNet PPL on code vs chat (no_robots). "
            "Issue #117."
        )
    )
    ap.add_argument(
        "--base",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HF model id for the dense BitNet base.",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="Inference device: cuda or cpu (default: cuda).",
    )
    ap.add_argument(
        "--n-seqs",
        type=int,
        default=98,
        help="Number of 512-token sequences for each domain (default: 98).",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length in tokens (default: 512).",
    )
    ap.add_argument(
        "--report",
        default=None,
        help=(
            "Path to a markdown file to append findings to. "
            "If not set, only prints to stdout."
        ),
    )
    args = ap.parse_args()

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Tokenizer (shared for both domains)
    # ------------------------------------------------------------------
    print(f"[code_data] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)
    vocab_size: int = tok.vocab_size

    # ------------------------------------------------------------------
    # 2. Build corpora (CPU token IDs — trivial memory)
    # ------------------------------------------------------------------
    print("[code_data] streaming code sequences ...")
    code_seqs = load_code_sequences(tok, n_seqs=args.n_seqs, seq_len=args.seq_len)

    print("[code_data] loading no_robots sequences ...")
    chat_seqs = _load_no_robots_sequences(tok, n_seqs=args.n_seqs, seq_len=args.seq_len)

    n_code = len(code_seqs)
    n_chat = len(chat_seqs)

    # ------------------------------------------------------------------
    # 3. Load dense model (bf16) — single pass, then free
    # ------------------------------------------------------------------
    print(f"[code_data] loading dense BitNet model on {device} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    peak_before = (
        torch.cuda.memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    print(f"[code_data] model loaded; VRAM allocated = {peak_before:.2f} GB")

    # ------------------------------------------------------------------
    # 4. Eval PPL on code (sample 3 per-seq PPLs as sanity check)
    # ------------------------------------------------------------------
    print("\n=== CODE domain (sample per-seq PPLs) ===")
    ppl_code, code_per_seq = _eval_ppl(
        model, code_seqs, device, vocab_size, "code", sample_n=3
    )

    # ------------------------------------------------------------------
    # 5. Eval PPL on chat/no_robots
    # ------------------------------------------------------------------
    print("\n=== CHAT domain (no_robots test) ===")
    ppl_chat, _ = _eval_ppl(
        model, chat_seqs, device, vocab_size, "chat", sample_n=0
    )

    # ------------------------------------------------------------------
    # 6. Peak VRAM
    # ------------------------------------------------------------------
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    else:
        peak_vram_gb = 0.0

    # ------------------------------------------------------------------
    # 7. Free model
    # ------------------------------------------------------------------
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    gap = ppl_code / ppl_chat if ppl_chat > 0 else float("inf")
    headroom = ppl_code > ppl_chat * 1.5  # >50% gap = meaningful headroom

    print()
    print("=" * 60)
    print("[code_data] === HEADROOM CHECK RESULTS ===")
    print(f"  Dense PPL (code  / {_CODE_DATASET_USED[:40]}...): {ppl_code:.3f}")
    print(f"  Dense PPL (chat  / no_robots test):              {ppl_chat:.3f}")
    print(f"  Ratio code/chat:                                 {gap:.2f}×")
    print(f"  Peak VRAM:                                       {peak_vram_gb:.2f} GB")
    if headroom:
        verdict = "YES — code PPL > chat PPL; proceed with code MoTE"
    elif gap < 0.5:
        verdict = "REVERSED — base is STRONGER on code than chat (code PPL < chat PPL); dataset too easy"
    else:
        verdict = "NO — gap is too small; domain not sufficiently OOD"
    print(f"  HEADROOM:  {verdict}")
    print(f"  ({n_code} code seqs, {n_chat} chat seqs, seq_len={args.seq_len})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 9. Append to report file (if requested)
    # ------------------------------------------------------------------
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        entry = f"""
## code-domain headroom check

**Date:** 2026-07-01
**Issue:** #117
**Model:** `{args.base}` (dense, bf16, cuda)
**Dataset (code):** `{_CODE_DATASET_USED}`
**Dataset (chat):** `HuggingFaceH4/no_robots` `split=test`
**Sequences:** {n_code} code × {args.seq_len} tokens; {n_chat} chat × {args.seq_len} tokens

| Domain | Val-PPL |
|--------|---------|
| Code (the-stack-smol Python) | **{ppl_code:.2f}** |
| Chat (no_robots test)        | **{ppl_chat:.2f}** |

**Ratio:** {gap:.2f}× | **Peak VRAM:** {peak_vram_gb:.2f} GB

**Sample code-sequence PPLs (sanity check):**
{chr(10).join(f"  - seq[{i}]: {code_per_seq[i]:.2f}" for i in range(min(3, len(code_per_seq))))}

**Verdict:** {"HEADROOM CONFIRMED — code PPL is " + f"{gap:.1f}× chat PPL; MoTE code experiment is justified." if headroom else ("REVERSED — base is STRONGER on code than chat (ratio " + f"{gap:.2f}×); dataset too easy / base already strong at Python. Try harder domain (math, scientific)." if gap < 0.5 else "NO HEADROOM — code/chat gap is " + f"{gap:.2f}×; domain not sufficiently OOD.")}
"""
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
        print(f"[code_data] appended findings to {report_path}")


if __name__ == "__main__":
    main()
