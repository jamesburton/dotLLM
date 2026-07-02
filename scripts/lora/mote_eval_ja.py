"""Per-language PPL + routing analysis for the JA MoTE quick probe.

Loads a trained MoTE adapter (from mote_train.py --mix ja_en) and evaluates:

  (a) PPL on held-out JA (wiki_ja, streamed) and EN (no_robots test, cached)
      versus the frozen dense BitNet baseline.
  (b) Expert routing histogram SEPARATELY for JA and EN sequences.
      Checks whether JA tokens concentrate on a different dominant expert
      than EN tokens (language specialization emerged from emergent routing).

Note on JA held-out disjointness: wiki_ja is streamed fresh each eval run.
With --ja-skip-tokens 0 (default) the eval shares the first N articles with
training, which is acceptable for a routing-analysis first-indication probe
(specialization signal is unaffected by mild overlap).  Set --ja-skip-tokens
to the training token count (~1.2e6) to skip to a disjoint slice; this
requires streaming through that many tokens and may add a few minutes.

Usage
-----
  python scripts/lora/mote_eval_ja.py \\
      --adapter .docs/mote/ja_probe \\
      --device cuda \\
      [--eval-seqs 50]

Writes:
  <adapter>/eval_ja.json  — PPL + routing results

Issue reference: (#117)
"""

# Suppress Windows Triton/dynamo compilation errors.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import gc
import json
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mote_upcycle import build_mote  # noqa: E402
from mote_train import MoTEShim, _wrap_mote_shims, _build_corpus  # noqa: E402
from domain_data import _ids_to_sequences  # noqa: E402


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _load_ja_eval_seqs(
    tokenizer,
    n_seqs: int,
    seq_len: int,
    skip_tokens: int = 0,
) -> list:
    """Stream wiki_ja and return n_seqs fixed-length sequences.

    Parameters
    ----------
    skip_tokens:
        Number of tokens to discard from the start of the stream before
        collecting.  Set to the training corpus JA token count to ensure
        held-out disjointness (at the cost of extra streaming time).
    """
    needed = n_seqs * seq_len + seq_len
    all_ids: list[int] = []
    skipped = 0

    try:
        ds = load_dataset(
            "wikimedia/wikipedia", "20231101.ja", split="train", streaming=True
        )
        for row in ds:
            text = row.get("text", "") or ""
            if len(text.strip()) < 200:
                continue
            enc = tokenizer(text, add_special_tokens=False)["input_ids"]
            # Skip phase: discard tokens until skip_tokens consumed
            if skipped < skip_tokens:
                to_skip = min(len(enc), skip_tokens - skipped)
                skipped += to_skip
                enc = enc[to_skip:]
                if not enc:
                    continue
            all_ids.extend(enc)
            if len(all_ids) >= needed:
                break
        print(f"[eval_ja] wiki_ja: collected {len(all_ids)} tokens (skip={skip_tokens})")
    except Exception as exc:
        print(f"[eval_ja] wiki_ja stream error: {exc}")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, "wiki_ja_eval")


def _load_en_eval_seqs(tokenizer, n_seqs: int, seq_len: int, vocab_size: int) -> list:
    """Load held-out EN sequences from no_robots test split (disjoint from train)."""
    seqs = _build_corpus(
        tokenizer=tokenizer,
        dataset_name="HuggingFaceH4/no_robots",
        dataset_config=None,
        dataset_split="test",
        max_seq_len=seq_len,
        max_sequences=n_seqs,
        tiny_random=False,
        vocab_size=vocab_size,
    )
    print(f"[eval_ja] no_robots test: {len(seqs)} seqs x {seq_len} tokens")
    return seqs


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _eval_ppl_and_routing(
    model: torch.nn.Module,
    seqs: list,
    vocab_size: int,
    device: torch.device,
    label: str,
    n_experts: int,
) -> tuple[float, list, list]:
    """Compute PPL + expert dispatch counts + argmax fractions for a sequence list.

    Returns (ppl, counts_list, argmax_frac_list).
    """
    # Register router-argmax hooks on all MoTEShim layers
    argmax_buckets: list[list] = []
    hooks: list = []
    for layer in model.model.layers:
        shim = layer.mlp
        if isinstance(shim, MoTEShim):
            bucket: list = []
            argmax_buckets.append(bucket)

            def _mk_hook(acc: list):
                def _hook(
                    module: torch.nn.Module,
                    _inp: tuple,
                    output: torch.Tensor,
                ) -> None:
                    # output: [n_tokens, n_experts] router logits
                    acc.append(torch.argmax(output, dim=-1).cpu())
                return _hook

            hooks.append(shim.router.register_forward_hook(_mk_hook(bucket)))

    total_nll = 0.0
    total_tok = 0
    cum_counts: Optional[torch.Tensor] = None

    with torch.no_grad():
        for seq in seqs:
            seq_t = seq.unsqueeze(0).to(device)
            n_pred = seq_t.size(1) - 1
            logits = model(input_ids=seq_t).logits  # [1, T, V]
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab_size),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()
            total_nll += nll
            total_tok += n_pred

            # Accumulate dispatch counts from all MoTEShim layers
            for layer in model.model.layers:
                shim = layer.mlp
                if isinstance(shim, MoTEShim) and shim.last_counts is not None:
                    c = shim.last_counts.cpu()
                    cum_counts = c if cum_counts is None else cum_counts + c

    for h in hooks:
        h.remove()

    ppl = math.exp(min(total_nll / max(total_tok, 1), 100.0))

    # Expert dispatch counts
    counts_list: list[int] = (
        [int(x) for x in cum_counts.tolist()] if cum_counts is not None else []
    )

    # Argmax distribution over tokens (from hooks)
    all_argmax: list = []
    for bucket in argmax_buckets:
        all_argmax.extend(bucket)
    if all_argmax:
        combined = torch.cat(all_argmax, dim=0)  # [n_total_tokens]
        total_t = combined.numel()
        argmax_frac = [
            round(int((combined == i).sum().item()) / max(total_t, 1), 4)
            for i in range(n_experts)
        ]
    else:
        argmax_frac = []

    print(f"[{label}] PPL={ppl:.3f}  counts={counts_list}  argmax_frac={argmax_frac}")
    return ppl, counts_list, argmax_frac


def _eval_dense_ppl(
    model: torch.nn.Module,
    seqs: list,
    vocab_size: int,
    device: torch.device,
    label: str,
) -> float:
    """Compute dense baseline PPL for a sequence list."""
    total_nll = 0.0
    total_tok = 0
    with torch.no_grad():
        for seq in seqs:
            seq_t = seq.unsqueeze(0).to(device)
            logits = model(input_ids=seq_t).logits
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab_size),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()
            total_nll += nll
            total_tok += seq_t.size(1) - 1
    ppl = math.exp(min(total_nll / max(total_tok, 1), 100.0))
    print(f"[{label}] PPL={ppl:.3f}")
    return ppl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Per-language PPL + routing analysis for the JA MoTE quick probe (#117)."
        )
    )
    ap.add_argument("--adapter", required=True, help="Adapter directory (contains mote_config.json + adapter_weights.pt)")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16", help="Base model (overridden by mote_config.json)")
    ap.add_argument("--device", default="cuda", help="Eval device: cpu or cuda")
    ap.add_argument(
        "--eval-seqs", type=int, default=50,
        help="Number of held-out sequences per language (default 50 × 512 = 25.6K tokens each)",
    )
    ap.add_argument(
        "--ja-skip-tokens", type=int, default=0,
        help=(
            "Skip this many JA tokens at the start of wiki_ja to avoid training overlap "
            "(default 0 — acceptable for a routing-analysis probe; costs extra stream time)."
        ),
    )
    args = ap.parse_args()

    adapter_dir = os.path.abspath(args.adapter)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load MoTE config
    # ------------------------------------------------------------------
    cfg_path = os.path.join(adapter_dir, "mote_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"mote_config.json not found in {adapter_dir!r}")
    with open(cfg_path, encoding="utf-8") as fh:
        mcfg = json.load(fh)

    n_experts: int = mcfg["n_experts"]
    top_k: int = mcfg["top_k"]
    shared: str = mcfg["shared"]
    layer_indices: list = mcfg["layers"]
    seq_len: int = mcfg.get("max_seq_len", 512)
    base: str = mcfg.get("base", args.base)

    print(
        f"[eval_ja] adapter={adapter_dir!r}\n"
        f"[eval_ja] n_experts={n_experts}  top_k={top_k}  shared={shared!r}  "
        f"layers={layer_indices}  device={device}"
    )

    # ------------------------------------------------------------------
    # 2. Tokenizer + vocab_size
    # ------------------------------------------------------------------
    print(f"[eval_ja] loading tokenizer from {base!r} ...")
    tok = AutoTokenizer.from_pretrained(base)
    vocab_size: int = AutoConfig.from_pretrained(base).vocab_size

    # ------------------------------------------------------------------
    # 3. Build held-out corpora (CPU, token IDs only)
    # ------------------------------------------------------------------
    print(f"[eval_ja] building eval corpora ({args.eval_seqs} seqs each) ...")
    ja_seqs = _load_ja_eval_seqs(tok, args.eval_seqs, seq_len, skip_tokens=args.ja_skip_tokens)
    en_seqs = _load_en_eval_seqs(tok, args.eval_seqs, seq_len, vocab_size)

    if not ja_seqs:
        raise RuntimeError("[eval_ja] JA eval corpus is empty. Check network / HF Hub access.")
    if not en_seqs:
        raise RuntimeError("[eval_ja] EN eval corpus is empty.")

    # ------------------------------------------------------------------
    # 4. Load MoTE student, eval on JA + EN, free
    # ------------------------------------------------------------------
    print("[eval_ja] loading MoTE student ...")
    student = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    student.config.use_cache = False
    student = build_mote(
        student, layers=layer_indices, n_experts=n_experts, top_k=top_k, shared=shared
    )
    student = _wrap_mote_shims(student)

    adapter_path = os.path.join(adapter_dir, "adapter_weights.pt")
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"adapter_weights.pt not found in {adapter_dir!r}")
    try:
        adapter_state = torch.load(adapter_path, map_location=device, weights_only=True)
    except TypeError:
        adapter_state = torch.load(adapter_path, map_location=device)  # type: ignore[call-arg]

    missing, unexpected = student.load_state_dict(adapter_state, strict=False)
    assert not unexpected, f"[eval_ja] unexpected adapter keys: {unexpected}"
    print(f"[eval_ja] adapter loaded: {len(adapter_state)} tensors (missing={len(missing)})")

    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    student.to(device)

    if device.type == "cuda":
        peak_after_student = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[eval_ja] peak VRAM after student load: {peak_after_student:.2f} GB")

    print("[eval_ja] evaluating MoTE on JA ...")
    mote_ja_ppl, ja_counts, ja_argmax = _eval_ppl_and_routing(
        student, ja_seqs, vocab_size, device, "JA-MoTE", n_experts
    )
    print("[eval_ja] evaluating MoTE on EN ...")
    mote_en_ppl, en_counts, en_argmax = _eval_ppl_and_routing(
        student, en_seqs, vocab_size, device, "EN-MoTE", n_experts
    )

    del student
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[eval_ja] student freed")

    # ------------------------------------------------------------------
    # 5. Load dense baseline, eval on JA + EN, free
    # ------------------------------------------------------------------
    print("[eval_ja] loading dense baseline ...")
    dense = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    dense.config.use_cache = False
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        peak_after_dense = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[eval_ja] peak VRAM after dense load: {peak_after_dense:.2f} GB")

    print("[eval_ja] evaluating dense on JA ...")
    dense_ja_ppl = _eval_dense_ppl(dense, ja_seqs, vocab_size, device, "JA-dense")
    print("[eval_ja] evaluating dense on EN ...")
    dense_en_ppl = _eval_dense_ppl(dense, en_seqs, vocab_size, device, "EN-dense")

    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        peak_vram_gb = 0.0

    del dense
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # 6. Routing specialization analysis
    # ------------------------------------------------------------------
    ja_dominant = ja_argmax.index(max(ja_argmax)) if ja_argmax else -1
    en_dominant = en_argmax.index(max(en_argmax)) if en_argmax else -1
    specialization = (ja_dominant != en_dominant) and n_experts > 1

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("[eval_ja] === JA MoTE QUICK PROBE — FINAL RESULTS ===")
    print()
    print(f"  JA: dense={dense_ja_ppl:.2f}  MoTE={mote_ja_ppl:.2f}  "
          f"delta={mote_ja_ppl - dense_ja_ppl:+.2f}")
    print(f"  EN: dense={dense_en_ppl:.2f}  MoTE={mote_en_ppl:.2f}  "
          f"delta={mote_en_ppl - dense_en_ppl:+.2f}")
    print()
    print(f"  JA expert dispatch counts:  {ja_counts}")
    print(f"  EN expert dispatch counts:  {en_counts}")
    print(f"  JA argmax expert fraction:  {ja_argmax}")
    print(f"  EN argmax expert fraction:  {en_argmax}")
    print()
    print(f"  JA dominant expert:  {ja_dominant}")
    print(f"  EN dominant expert:  {en_dominant}")
    print(f"  Specialization emerged (JA != EN dominant): {specialization}")
    print()
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  eval_seqs: {args.eval_seqs} × {seq_len} tokens per language")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 8. Write eval_ja.json
    # ------------------------------------------------------------------
    results = {
        "dense_ja_ppl": dense_ja_ppl,
        "mote_ja_ppl": mote_ja_ppl,
        "ja_ppl_delta": mote_ja_ppl - dense_ja_ppl,
        "dense_en_ppl": dense_en_ppl,
        "mote_en_ppl": mote_en_ppl,
        "en_ppl_delta": mote_en_ppl - dense_en_ppl,
        "ja_expert_counts": ja_counts,
        "en_expert_counts": en_counts,
        "ja_argmax_frac": ja_argmax,
        "en_argmax_frac": en_argmax,
        "ja_dominant_expert": ja_dominant,
        "en_dominant_expert": en_dominant,
        "specialization_emerged": specialization,
        "peak_vram_gb": peak_vram_gb,
        "n_experts": n_experts,
        "top_k": top_k,
        "shared": shared,
        "layers": layer_indices,
        "eval_seqs_per_lang": args.eval_seqs,
        "ja_skip_tokens": args.ja_skip_tokens,
    }
    out_path = os.path.join(adapter_dir, "eval_ja.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval_ja] results -> {out_path}")
    print("[eval_ja] DONE")


if __name__ == "__main__":
    main()
