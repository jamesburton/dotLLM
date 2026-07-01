"""Cross-lingual PPL spike: German + Japanese Wikipedia vs. chat baseline.

Fast-fail experiment to determine whether cross-lingual headroom scales
with language distance from English before committing to a MoTE cross-lingual
experiment. Issue (#117).

Usage:
    PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface \\
    python scripts/lora/crosslingual_spike.py \\
        [--base microsoft/bitnet-b1.58-2B-4T-bf16] \\
        [--device cuda] [--n-seqs 98] [--seq-len 512] \\
        [--report E:/Development/dotLLM/.superpowers/sdd/kaggle-harness-report.md]
"""

import argparse
import gc
import math
import os
import sys

# Windows: suppress dynamo/Triton compilation errors.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make domain_data importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_data import (
    _load_wiki_de_sequences,
    _load_wiki_ja_sequences,
    _DOMAIN_DATASET_USED,
    _ppl_from_nll,
)

CHAT_REFERENCE_PPL = 13.44  # known dense BitNet PPL on no_robots test


def _eval_ppl(model, seqs, device, label, sample_n=3):
    total_nll = 0.0
    total_tokens = 0
    per_seq = []
    vocab = None

    with torch.no_grad():
        for idx, seq in enumerate(seqs):
            seq_t = seq.unsqueeze(0).to(device)
            n_pred = seq_t.size(1) - 1
            logits = model(input_ids=seq_t).logits
            if vocab is None:
                vocab = logits.size(-1)
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab),
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


def main():
    ap = argparse.ArgumentParser(description="Cross-lingual spike: German + Japanese Wikipedia PPL.")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-seqs", type=int, default=98)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    needed = args.n_seqs * args.seq_len + args.seq_len

    # -----------------------------------------------------------------------
    # 1. Tokenizer + CJK probe
    # -----------------------------------------------------------------------
    print(f"[spike] loading tokenizer from {args.base!r} ...")
    tok = AutoTokenizer.from_pretrained(args.base)

    ja_sample = "日本語のウィキペディアは、さまざまなトピックについて詳細な記事を提供しています。"
    ja_chars = len(ja_sample)
    ja_ids = tok(ja_sample, add_special_tokens=False)["input_ids"]
    ja_tok_count = len(ja_ids)
    ja_tpc = ja_tok_count / ja_chars
    print(
        f"[spike] JA tokenizer probe: {ja_chars} chars → {ja_tok_count} tokens "
        f"({ja_tpc:.2f} tok/char); "
        f"{'HIGH fragmentation — BPE byte-fallback dominant' if ja_tpc > 1.5 else 'moderate'}"
    )

    # -----------------------------------------------------------------------
    # 2. Load corpora (CPU only, streaming)
    # -----------------------------------------------------------------------
    print("\n[spike] loading German Wikipedia corpus ...")
    de_seqs = _load_wiki_de_sequences(tok, args.n_seqs, args.seq_len, needed)

    print("\n[spike] loading Japanese Wikipedia corpus ...")
    ja_seqs = _load_wiki_ja_sequences(tok, args.n_seqs, args.seq_len, needed)

    # -----------------------------------------------------------------------
    # 3. Load dense model
    # -----------------------------------------------------------------------
    print(f"\n[spike] loading {args.base} (bf16) on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = False
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        print(
            f"[spike] VRAM after model load: "
            f"{torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB"
        )

    # -----------------------------------------------------------------------
    # 4. Eval DE + JA
    # -----------------------------------------------------------------------
    print("\n=== DOMAIN: WIKI_DE ===")
    ppl_de, per_seq_de = _eval_ppl(model, de_seqs, device, "wiki_de", sample_n=3)

    print("\n=== DOMAIN: WIKI_JA ===")
    ppl_ja, per_seq_ja = _eval_ppl(model, ja_seqs, device, "wiki_ja", sample_n=3)

    # -----------------------------------------------------------------------
    # 5. Peak VRAM + cleanup
    # -----------------------------------------------------------------------
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    else:
        peak_vram_gb = 0.0

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # 6. Summary + verdict
    # -----------------------------------------------------------------------
    ppl_chat = CHAT_REFERENCE_PPL
    print()
    print("=" * 64)
    print("[spike] === CROSS-LINGUAL SPIKE RESULTS ===")
    print(f"  Chat baseline (reference):  {ppl_chat:.2f}")
    print(f"  German (wiki_de):           {ppl_de:.3f}  ({ppl_de / ppl_chat:.2f}x chat)")
    print(f"  Japanese (wiki_ja):         {ppl_ja:.3f}  ({ppl_ja / ppl_chat:.2f}x chat)")
    print(f"  JA / DE ratio:              {ppl_ja / ppl_de:.2f}x")
    print(f"  JA tok/char:                {ja_tpc:.2f}  ({'HIGH frag.' if ja_tpc > 1.5 else 'moderate'})")
    print(f"  Peak VRAM:                  {peak_vram_gb:.2f} GB")
    print()

    if ppl_ja > ppl_de * 1.5:
        verdict = (
            "PROMISING — PPL scales with language distance (JA >> DE). "
            "Recommend cross-lingual MoTE experiment (train on EN+JA, test closing JA gap)."
        )
    elif ppl_ja > ppl_de * 1.1:
        verdict = (
            "MILD — JA marginally above DE. "
            f"{'Tokenizer fragmentation (%.2f tok/char) may be inflating JA PPL.' % ja_tpc} "
            "Consider downstream task-accuracy instead."
        )
    else:
        verdict = (
            "FLAT — JA ≈ DE, no clear language-distance scaling. "
            "Tokenizer artifact likely dominant. Recommend moving to task-accuracy."
        )

    if ja_tpc > 1.5:
        tok_caveat = (
            f"CAVEAT: {ja_tpc:.2f} tok/char for JA vs ~0.3 for English — BPE byte-fallback. "
            "PPL may be inflated vs. a CJK-aware tokenizer; headroom may reflect tokenizer poverty, "
            "not semantic gap. A true cross-lingual adapter must also address this."
        )
    else:
        tok_caveat = f"Tokenizer: {ja_tpc:.2f} tok/char for JA — moderate fragmentation."

    print(f"  VERDICT: {verdict}")
    print(f"  TOK NOTE: {tok_caveat}")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # 7. Append to report
    # -----------------------------------------------------------------------
    if args.report:
        report_path = os.path.abspath(args.report)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        entry = f"""
## cross-lingual spike (de + ja)

**Date:** 2026-07-01
**Issue:** #117
**Model:** `{args.base}` (dense, bf16, {args.device})
**n_seqs:** {args.n_seqs} × {args.seq_len} tokens per language

### Datasets
- `wiki_de`: `{_DOMAIN_DATASET_USED.get("wiki_de", "wikimedia/wikipedia 20231101.de")}`
- `wiki_ja`: `{_DOMAIN_DATASET_USED.get("wiki_ja", "wikimedia/wikipedia 20231101.ja")}`

### Results

| Language | Val-PPL | Ratio vs chat (13.44) |
|----------|---------|----------------------|
| chat (no_robots, reference) | **13.44** | 1.00x |
| German (wiki_de) | **{ppl_de:.2f}** | {ppl_de / ppl_chat:.2f}x |
| Japanese (wiki_ja) | **{ppl_ja:.2f}** | {ppl_ja / ppl_chat:.2f}x |

**JA / DE PPL ratio:** {ppl_ja / ppl_de:.2f}x
**Peak VRAM:** {peak_vram_gb:.2f} GB

### Tokenizer note (JA)
- Sample: `日本語のウィキペディア…` ({ja_chars} chars) → {ja_tok_count} tokens = **{ja_tpc:.2f} tok/char**
- {'HIGH fragmentation: BPE byte-fallback dominant. Each CJK character becomes multiple byte tokens.' if ja_tpc > 1.5 else 'Moderate fragmentation.'}
- This means JA PPL is measured over fewer semantic units per sequence than DE/EN — PPL may be inflated or uninformative as a pure semantic gap measure.

### Sample per-sequence PPLs
- `wiki_de` seq[0..2]: {", ".join(f"{p:.1f}" for p in per_seq_de[:3])}
- `wiki_ja` seq[0..2]: {", ".join(f"{p:.1f}" for p in per_seq_ja[:3])}

### Cross-lingual scaling verdict
**{verdict}**

**Tokenizer caveat:** {tok_caveat}
"""
        with open(report_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
        print(f"\n[spike] appended findings to {report_path}")


if __name__ == "__main__":
    main()
