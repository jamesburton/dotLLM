"""Per-token-class PPL + expert-routing analysis for a tool-use MoTE adapter.

Loads a trained MoTE adapter (from mote_train.py with a tool-use corpus) and
evaluates:

  (a) PPL on held-out TOOL-CALL text (glaive_func_calling train[:3000]) and
      PROSE text (no_robots test split) versus the frozen dense BitNet baseline.
  (b) Expert routing histogram SEPARATELY for tool-call and prose sequences.
      Checks whether tool-call tokens concentrate on a different dominant expert
      than prose tokens (capability specialization emerged from routing).
  (c) Specialization metric: L1 distance + cosine similarity between the
      toolcall vs prose argmax_frac vectors.  Interpretation:
        - large L1 + different dominant experts  → routing specialized on format
        - uniform / identical distributions      → no specialization emerged
          (same null result as prior mid-FFN domain experiments)

Data slices (no training overlap)
----------------------------------
  toolcall: NousResearch/hermes-function-calling-v1 glaive_func_calling
            train[:3000] — disjoint from MoTE training which uses train[3000:]
            (see capability_data.py probe-train CLI comment and _load_tooluse_sequences).
  prose:    HuggingFaceH4/no_robots test split — disjoint from no_robots train
            (instruction task-LoRA used train; mote_eval_ja.py EN eval also uses
            test, so results are comparable across evals).

Usage
-----
  python scripts/lora/mote_eval_tooluse.py \\
      --adapter .docs/mote/tooluse_probe2 \\
      --device cuda \\
      [--eval-seqs 50]

Writes:
  <adapter>/eval_tooluse.json  — PPL + routing + specialization results

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
import itertools
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
from mote_train import MoTEShim, _wrap_mote_shims  # noqa: E402
from capability_data import (  # noqa: E402
    _collect_ids,
    _ids_to_sequences,
    _extract_text_tooluse,
    _extract_text_instruction,
)


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _load_toolcall_eval_seqs(tokenizer, n_seqs: int, seq_len: int) -> list:
    """Load held-out tool-call sequences from glaive_func_calling (first 3000 rows).

    Row limit: first 3000 rows of the train split, disjoint from MoTE training
    which uses train[3000:] (see capability_data.py probe-train CLI comment).

    Note: HuggingFace streaming does not support [:N] slice syntax for this
    dataset config (only [N:] skip syntax is supported in streaming mode).
    We therefore load split="train" with streaming and cap via itertools.islice
    so we consume exactly the first 3000 rows — identical to train[:3000].
    """
    needed = n_seqs * seq_len + seq_len
    all_ids: list[int] = []

    try:
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            "glaive_func_calling",
            split="train",
            streaming=True,
        )
        # Cap at first 3000 rows — equivalent to train[:3000], disjoint from
        # MoTE training slice train[3000:].
        capped = itertools.islice(ds, 3000)
        all_ids = _collect_ids(capped, _extract_text_tooluse, tokenizer, needed, "toolcall_eval")
    except Exception as exc:
        print(f"[eval_tooluse] glaive_func_calling stream error: {exc}")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, "toolcall_eval")


def _load_prose_eval_seqs(tokenizer, n_seqs: int, seq_len: int) -> list:
    """Load held-out prose sequences from no_robots test split.

    The test split is disjoint from the train split used for the instruction
    task-LoRA (U2 Phase A).  It is also the same corpus used by mote_eval_ja.py
    for its EN baseline, so PPL figures are directly comparable across evals.
    """
    needed = n_seqs * seq_len + seq_len
    all_ids: list[int] = []

    try:
        ds = load_dataset(
            "HuggingFaceH4/no_robots",
            split="test",
            streaming=True,
        )
        all_ids = _collect_ids(ds, _extract_text_instruction, tokenizer, needed, "prose_eval")
    except Exception as exc:
        print(f"[eval_tooluse] no_robots test stream error: {exc}")

    return _ids_to_sequences(all_ids, n_seqs, seq_len, "prose_eval")


# ---------------------------------------------------------------------------
# Eval helpers (identical to mote_eval_ja.py — only label strings differ)
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
# Specialization metric helpers
# ---------------------------------------------------------------------------


def _l1_distance(a: list[float], b: list[float]) -> float:
    """L1 distance between two equal-length probability vectors."""
    return round(sum(abs(x - y) for x, y in zip(a, b)), 4)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    return round(dot / denom if denom > 0.0 else 0.0, 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Per-token-class PPL + routing analysis for a tool-use MoTE adapter (#117)."
        )
    )
    ap.add_argument("--adapter", required=True, help="Adapter directory (contains mote_config.json + adapter_weights.pt)")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16", help="Base model (overridden by mote_config.json)")
    ap.add_argument("--device", default="cuda", help="Eval device: cpu or cuda")
    ap.add_argument(
        "--eval-seqs", type=int, default=50,
        help="Number of held-out sequences per class (default 50 × 512 = 25.6K tokens each)",
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
        f"[eval_tooluse] adapter={adapter_dir!r}\n"
        f"[eval_tooluse] n_experts={n_experts}  top_k={top_k}  shared={shared!r}  "
        f"layers={layer_indices}  device={device}"
    )

    # ------------------------------------------------------------------
    # 2. Tokenizer + vocab_size
    # ------------------------------------------------------------------
    print(f"[eval_tooluse] loading tokenizer from {base!r} ...")
    tok = AutoTokenizer.from_pretrained(base)
    vocab_size: int = AutoConfig.from_pretrained(base).vocab_size

    # ------------------------------------------------------------------
    # 3. Build held-out corpora (CPU, token IDs only)
    # ------------------------------------------------------------------
    print(f"[eval_tooluse] building eval corpora ({args.eval_seqs} seqs each) ...")
    toolcall_seqs = _load_toolcall_eval_seqs(tok, args.eval_seqs, seq_len)
    prose_seqs = _load_prose_eval_seqs(tok, args.eval_seqs, seq_len)

    if not toolcall_seqs:
        raise RuntimeError("[eval_tooluse] tool-call eval corpus is empty. Check network / HF Hub access.")
    if not prose_seqs:
        raise RuntimeError("[eval_tooluse] prose eval corpus is empty.")

    # ------------------------------------------------------------------
    # 4. Load MoTE student, eval on toolcall + prose, free
    # ------------------------------------------------------------------
    print("[eval_tooluse] loading MoTE student ...")
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
    assert not unexpected, f"[eval_tooluse] unexpected adapter keys: {unexpected}"
    print(f"[eval_tooluse] adapter loaded: {len(adapter_state)} tensors (missing={len(missing)})")

    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    student.to(device)

    if device.type == "cuda":
        peak_after_student = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[eval_tooluse] peak VRAM after student load: {peak_after_student:.2f} GB")

    print("[eval_tooluse] evaluating MoTE on tool-call sequences ...")
    mote_tc_ppl, tc_counts, tc_argmax = _eval_ppl_and_routing(
        student, toolcall_seqs, vocab_size, device, "TOOLCALL-MoTE", n_experts
    )
    print("[eval_tooluse] evaluating MoTE on prose sequences ...")
    mote_pr_ppl, pr_counts, pr_argmax = _eval_ppl_and_routing(
        student, prose_seqs, vocab_size, device, "PROSE-MoTE", n_experts
    )

    del student
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[eval_tooluse] student freed")

    # ------------------------------------------------------------------
    # 5. Load dense baseline, eval on toolcall + prose, free
    # ------------------------------------------------------------------
    print("[eval_tooluse] loading dense baseline ...")
    dense = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    dense.config.use_cache = False
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)

    if device.type == "cuda":
        peak_after_dense = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[eval_tooluse] peak VRAM after dense load: {peak_after_dense:.2f} GB")

    print("[eval_tooluse] evaluating dense on tool-call sequences ...")
    dense_tc_ppl = _eval_dense_ppl(dense, toolcall_seqs, vocab_size, device, "TOOLCALL-dense")
    print("[eval_tooluse] evaluating dense on prose sequences ...")
    dense_pr_ppl = _eval_dense_ppl(dense, prose_seqs, vocab_size, device, "PROSE-dense")

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
    tc_dominant = tc_argmax.index(max(tc_argmax)) if tc_argmax else -1
    pr_dominant = pr_argmax.index(max(pr_argmax)) if pr_argmax else -1
    specialization = (tc_dominant != pr_dominant) and n_experts > 1

    l1_dist = _l1_distance(tc_argmax, pr_argmax) if tc_argmax and pr_argmax else 0.0
    cos_sim = _cosine_similarity(tc_argmax, pr_argmax) if tc_argmax and pr_argmax else 0.0

    # Interpretation guideline:
    #   l1_dist > 0.3 AND tc_dominant != pr_dominant → strong format specialization
    #   l1_dist < 0.1                                → no routing specialization (null result)
    interp = (
        "SPECIALIZED: routing differs significantly by token class"
        if (specialization and l1_dist > 0.3)
        else "PARTIAL: dominant experts differ but distributions overlap"
        if specialization
        else "NULL: no routing specialization (distributions are similar)"
    )

    # ------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("[eval_tooluse] === TOOL-USE MoTE ROUTING PROBE — FINAL RESULTS ===")
    print()
    print(f"  TOOLCALL: dense={dense_tc_ppl:.2f}  MoTE={mote_tc_ppl:.2f}  "
          f"delta={mote_tc_ppl - dense_tc_ppl:+.2f}")
    print(f"  PROSE:    dense={dense_pr_ppl:.2f}  MoTE={mote_pr_ppl:.2f}  "
          f"delta={mote_pr_ppl - dense_pr_ppl:+.2f}")
    print()
    print(f"  TOOLCALL expert dispatch counts:  {tc_counts}")
    print(f"  PROSE    expert dispatch counts:  {pr_counts}")
    print(f"  TOOLCALL argmax expert fraction:  {tc_argmax}")
    print(f"  PROSE    argmax expert fraction:  {pr_argmax}")
    print()
    print(f"  TOOLCALL dominant expert:  {tc_dominant}")
    print(f"  PROSE    dominant expert:  {pr_dominant}")
    print(f"  Specialization emerged (TOOLCALL != PROSE dominant): {specialization}")
    print()
    print(f"  L1 distance (toolcall vs prose argmax_frac):  {l1_dist:.4f}")
    print(f"  Cosine similarity (toolcall vs prose):        {cos_sim:.4f}")
    print(f"  Interpretation: {interp}")
    print()
    print(f"  Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"  eval_seqs: {args.eval_seqs} × {seq_len} tokens per class")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 8. Write eval_tooluse.json
    # ------------------------------------------------------------------
    results = {
        "dense_toolcall_ppl": dense_tc_ppl,
        "mote_toolcall_ppl": mote_tc_ppl,
        "toolcall_ppl_delta": mote_tc_ppl - dense_tc_ppl,
        "dense_prose_ppl": dense_pr_ppl,
        "mote_prose_ppl": mote_pr_ppl,
        "prose_ppl_delta": mote_pr_ppl - dense_pr_ppl,
        "toolcall_expert_counts": tc_counts,
        "prose_expert_counts": pr_counts,
        "toolcall_argmax_frac": tc_argmax,
        "prose_argmax_frac": pr_argmax,
        "toolcall_dominant_expert": tc_dominant,
        "prose_dominant_expert": pr_dominant,
        "specialization_emerged": specialization,
        "l1_distance": l1_dist,
        "cosine_similarity": cos_sim,
        "interpretation": interp,
        "peak_vram_gb": peak_vram_gb,
        "n_experts": n_experts,
        "top_k": top_k,
        "shared": shared,
        "layers": layer_indices,
        "eval_seqs_per_class": args.eval_seqs,
        "toolcall_data": "NousResearch/hermes-function-calling-v1 glaive_func_calling train first 3000 rows (equiv train[:3000])",
        "prose_data": "HuggingFaceH4/no_robots test",
    }
    out_path = os.path.join(adapter_dir, "eval_tooluse.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[eval_tooluse] results -> {out_path}")
    print("[eval_tooluse] DONE")


if __name__ == "__main__":
    main()
