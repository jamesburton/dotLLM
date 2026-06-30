"""MoTE init-PPL probe: diagnose training vs architecture regression (#117).

Builds the UNTRAINED upcycle matching c1_local (layers 26-29, 4 experts, top-1, FP shared)
and computes val-PPL on the no_robots test split vs the dense baseline — without loading
any trained adapter weights.

Hypothesis under test
---------------------
With ``shared="fp"`` (full frozen dense FFN) plus non-normalised routed experts, the
MoTE block output at init is:

    mote_out = gate_i * expert_i(x)  +  shared(x)
             ≈ (1/n_experts) * dense_ffn(x)  +  dense_ffn(x)
             ≈ (1 + 1/n_experts) * dense_ffn(x)
             ≈ 1.25 * dense_ffn(x)   (for n_experts=4, top-k=1)

If this over-count is real the model starts training from a perturbed (not dense)
initialisation, so the LM-only heal is fighting miscalibration rather than genuine
specialisation.

Interpretation
--------------
* PPL_untrained ≈ PPL_dense  (within ≈10 %):
    Architecture is clean at init.  Blame TRAINING (lr, gate calibration, need KD).
* PPL_untrained ≫ PPL_dense  (>20 % worse):
    FP-shared + non-normalised combine over-counts the FFN at init.
    Fix: normalise routed gates, or replace shared rather than add it, or scale.

Memory layout
-------------
Sequential: one 2B model on GPU at a time.
  1. Load untrained MoTE → block-norm probe + PPL eval → del + empty_cache.
  2. Load dense → PPL eval → del + empty_cache.

Usage
-----
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface \\
  C:/Python311/python.exe scripts/lora/mote_init_probe.py \\
      --base microsoft/bitnet-b1.58-2B-4T-bf16 \\
      --device cuda
"""

# Windows: torch.compile/Triton/Inductor needs cl.exe; suppress dynamo errors.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Allow sibling imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mote_upcycle import build_mote  # noqa: E402
from mote_train import MoTEShim, _wrap_mote_shims, _build_corpus  # noqa: E402


_BASE_DEFAULT = "microsoft/bitnet-b1.58-2B-4T-bf16"
_DATASET = "HuggingFaceH4/no_robots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_ppl(total_nll: float, total_tokens: int) -> float:
    """Convert summed token NLL to perplexity (capped at exp(100))."""
    if total_tokens == 0:
        return float("inf")
    avg = total_nll / total_tokens
    if avg > 100.0:
        import warnings
        warnings.warn(f"[probe] avg NLL {avg:.2f} > 100; capping PPL at exp(100).", UserWarning)
    return math.exp(min(avg, 100.0))


def _measure_overcounting(
    model: torch.nn.Module,
    probe_seqs: list,
    device: torch.device,
) -> Optional[float]:
    """Quantify the init over-count factor for the first MoTE layer with a shared expert.

    Runs ``len(probe_seqs)`` forward passes, captures the input **x** and output
    **hidden** of the first MoTEShim, then re-evaluates the shared expert alone on
    the same **x** to get ``shared_out``.

    Returns
    -------
    Mean per-token ratio ``||hidden|| / ||shared_out||`` across all probe tokens,
    or ``None`` if no shared expert is present (``shared="none"``).

    Expected values at init
    -----------------------
    * ``shared="fp"``, ``n_experts=4``, ``top_k=1``:
        ratio ≈ 1.25 (FP shared + 0.25× ternary routed clone)
    * ``shared="none"``:
        returns ``None`` — no over-counting from shared path
    """
    # Locate first MoTEShim that has a shared expert.
    target_shim: Optional[MoTEShim] = None
    for layer in model.model.layers:
        if isinstance(layer.mlp, MoTEShim) and layer.mlp.mote.shared is not None:
            target_shim = layer.mlp
            break
    if target_shim is None:
        print("[probe] no shared expert found — over-count factor: N/A")
        return None

    captured_x: list = []
    captured_out: list = []

    def _hook(module: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
        # inp[0]: x [B, T, H] (input to MoTEShim.forward)
        # out:    hidden [B, T, H] (returned to residual stream)
        captured_x.append(inp[0].detach().cpu())
        captured_out.append(out.detach().cpu())

    h = target_shim.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for seq in probe_seqs:
                seq_t = seq.unsqueeze(0).to(device)
                model(input_ids=seq_t)
    finally:
        h.remove()

    if not captured_x:
        return None

    H = captured_x[0].size(-1)
    # [1, T1, H], [1, T2, H], ... → [N_tokens, H]
    x_flat_cpu = torch.cat(captured_x, dim=1).view(-1, H)
    out_flat_cpu = torch.cat(captured_out, dim=1).view(-1, H)
    del captured_x, captured_out

    # Re-run shared expert on device with flattened x.
    x_flat_dev = x_flat_cpu.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        shared_out_cpu = target_shim.mote.shared(x_flat_dev).cpu()
    del x_flat_dev

    # Per-token L2 norm ratio: ||mote_out|| / ||shared_only||
    mote_norms = out_flat_cpu.float().norm(dim=-1)       # [N]
    shared_norms = shared_out_cpu.float().norm(dim=-1)   # [N]
    valid = shared_norms > 1e-8
    if not valid.any():
        return None
    ratio: float = (mote_norms[valid] / shared_norms[valid]).mean().item()
    return ratio


def _run_ppl_pass(
    model: torch.nn.Module,
    cached_seqs: list,
    device: torch.device,
    vocab_size: int,
) -> float:
    """Sequential per-sequence PPL evaluation; returns perplexity."""
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for seq in cached_seqs:
            seq_t = seq.unsqueeze(0).to(device)
            logits = model(input_ids=seq_t).logits  # [1, T, V]
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, vocab_size),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()
            total_nll += nll
            total_tokens += seq_t.size(1) - 1
    return _compute_ppl(total_nll, total_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MoTE init-PPL probe: training vs architecture regression (#117)"
    )
    ap.add_argument("--base", default=_BASE_DEFAULT, help="HF model id or local path")
    ap.add_argument("--layers", default="26-29", help="Layer range, e.g. 26-29 or 26,27,28,29")
    ap.add_argument("--n-experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--shared", default="fp", choices=["fp", "ternary", "none"])
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    ap.add_argument(
        "--n-seqs", type=int, default=98,
        help="Held-out sequences (matches mote_eval default for --eval-tokens 5e4 / seq_len 512)",
    )
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument(
        "--probe-seqs", type=int, default=5,
        help="Number of sequences used for the block over-count norm probe",
    )
    ap.add_argument("--out", default=None, help="Optional path for probe_results.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Resolve layer indices.
    if "-" in args.layers and "," not in args.layers:
        s, e = args.layers.split("-")
        layer_indices = list(range(int(s), int(e) + 1))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    print(
        f"[probe] config: layers={layer_indices}  n_experts={args.n_experts}  "
        f"top_k={args.top_k}  shared={args.shared!r}  device={device}"
    )

    # ------------------------------------------------------------------
    # 1. Build held-out corpus (CPU; token IDs only)
    # ------------------------------------------------------------------
    print("[probe] loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_cfg = AutoConfig.from_pretrained(args.base)
    vocab_size: int = base_cfg.vocab_size

    print(
        f"[probe] building held-out corpus "
        f"({args.n_seqs} seqs × {args.max_seq_len} tok, split='test') ..."
    )
    corpus_raw = _build_corpus(
        tokenizer=tokenizer,
        dataset_name=_DATASET,
        dataset_config=None,
        dataset_split="test",
        max_seq_len=args.max_seq_len,
        max_sequences=args.n_seqs,
        tiny_random=False,
        vocab_size=vocab_size,
    )
    if not corpus_raw:
        raise RuntimeError(
            f"Corpus is empty — check dataset {_DATASET!r} availability."
        )
    cached_seqs: list = [s.cpu() for s in corpus_raw]
    del corpus_raw
    print(f"[probe] {len(cached_seqs)} sequences cached on CPU")

    # ------------------------------------------------------------------
    # 2. Untrained MoTE pass
    # ------------------------------------------------------------------
    print("[probe] loading base model for UNTRAINED MoTE ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.config.use_cache = False

    print("[probe] applying build_mote (NO adapter loaded, NO training) ...")
    model = build_mote(
        model,
        layers=layer_indices,
        n_experts=args.n_experts,
        top_k=args.top_k,
        shared=args.shared,
    )
    model = _wrap_mote_shims(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    if device.type == "cuda":
        vram_after_mote = torch.cuda.memory_allocated() / 1e9
        print(f"[probe] VRAM after untrained MoTE load: {vram_after_mote:.2f} GB")

    # -- Block norm probe --
    print(
        f"[probe] measuring block over-count factor "
        f"({args.probe_seqs} probe sequences) ..."
    )
    probe_seqs = cached_seqs[: args.probe_seqs]
    over_count = _measure_overcounting(model, probe_seqs, device)
    if over_count is not None:
        expected = 1.0 + 1.0 / args.n_experts
        print(
            f"[probe] block over-count factor = {over_count:.4f}  "
            f"(theoretical at perfect-uniform routing = {expected:.4f})"
        )
    else:
        print("[probe] shared=none — no FP-path over-count")

    # -- PPL eval --
    print(f"[probe] running UNTRAINED MoTE PPL eval over {len(cached_seqs)} sequences ...")
    ppl_mote = _run_ppl_pass(model, cached_seqs, device, vocab_size)
    print(f"[probe] PPL_untrained_mote = {ppl_mote:.3f}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[probe] untrained MoTE freed; GPU cache cleared")

    # ------------------------------------------------------------------
    # 3. Dense baseline pass
    # ------------------------------------------------------------------
    print("[probe] loading DENSE baseline ...")
    dense = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    dense.config.use_cache = False
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)
    dense.to(device)

    if device.type == "cuda":
        vram_after_dense = torch.cuda.memory_allocated() / 1e9
        print(f"[probe] VRAM after dense load: {vram_after_dense:.2f} GB")

    print(f"[probe] running dense baseline PPL eval over {len(cached_seqs)} sequences ...")
    ppl_dense = _run_ppl_pass(dense, cached_seqs, device, vocab_size)
    print(f"[probe] PPL_dense = {ppl_dense:.3f}")

    del dense
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("[probe] dense freed; GPU cache cleared")

    # ------------------------------------------------------------------
    # 4. Results + verdict
    # ------------------------------------------------------------------
    delta = ppl_mote - ppl_dense
    # Threshold: >25 % worse than dense => architecture problem
    arch_threshold = ppl_dense * 1.25
    is_arch_problem = ppl_mote > arch_threshold
    verdict = (
        "ARCHITECTURE/calibration problem (FP-shared + non-normalised-routed over-counts FFN at init)"
        if is_arch_problem
        else "TRAINING problem (architecture clean at init; LM-only heal degraded quality)"
    )

    print()
    print("[probe] ============================================================")
    print("[probe]  INIT-PPL DIAGNOSTIC RESULTS")
    print("[probe] ============================================================")
    print(f"  PPL_untrained_mote      = {ppl_mote:.3f}")
    print(f"  PPL_dense               = {ppl_dense:.3f}")
    print(f"  delta (mote - dense)    = {delta:+.3f}")
    if over_count is not None:
        theoretical = 1.0 + 1.0 / args.n_experts
        print(
            f"  block over-count factor = {over_count:.4f}x  "
            f"(theoretical = {theoretical:.4f}x at uniform routing)"
        )
    print(f"  VERDICT                 = {verdict}")
    print("[probe] ============================================================")
    print()
    if is_arch_problem:
        print(
            "  The untrained MoTE block over-counts the FFN at init because the FP\n"
            "  shared expert (= full dense FFN) is ADDED to the routed output, not\n"
            "  substituted for it.  With top-k=1 and near-uniform routing, the init\n"
            "  block output ≈ (1 + gate) × dense_ffn(x) ≈ 1.25 × dense_ffn(x),\n"
            "  perturbing every converted layer before any training begins.\n\n"
            "  Recommended fixes:\n"
            "    (a) Normalise routed gate by (1 - gate_shared) so total ≈ 1×.\n"
            "    (b) Have shared path REPLACE, not ADD: remove the residual add.\n"
            "    (c) Scale the combined output by 1 / (1 + mean_gate)."
        )
    else:
        print(
            "  Architecture is clean at init — the upcycle starts at ≈ dense PPL.\n"
            "  The 15M-token LM-only heal run (c1_local) degraded it.\n\n"
            "  Probable causes:\n"
            "    (a) LR too high: router + expert weights diverge from dense init.\n"
            "    (b) No KD during heal: without teacher signal, the LM loss alone\n"
            "        pushes the model towards mode collapse / expert specialisation\n"
            "        that is not calibrated by a dense reference.\n"
            "    (c) Too few tokens (15M): gates not settled; experts half-trained.\n\n"
            "  Next step: run c1 on Kaggle with kd_weight=0.5, 400M tokens."
        )

    # Optional JSON output.
    results: dict = {
        "ppl_untrained_mote": ppl_mote,
        "ppl_dense": ppl_dense,
        "ppl_delta": delta,
        "over_count_factor": over_count,
        "verdict": verdict,
        "is_arch_problem": is_arch_problem,
        "layers": layer_indices,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "shared": args.shared,
        "n_seqs": len(cached_seqs),
        "max_seq_len": args.max_seq_len,
    }
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"[probe] results written -> {args.out}")

    print("[probe] DONE")


if __name__ == "__main__":
    main()
