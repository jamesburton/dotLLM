"""MoTE eval module: expert entropy / val-PPL / router input-dependence.

Loads a trained MoTE adapter (from ``mote_train.py``) and computes three
diagnostic metrics on a held-out token slice that is *disjoint* from the
heal-training corpus:

(a) **Expert-utilization entropy** H over the routed-expert dispatch histogram.
    Also reports ``H / log(n_experts)`` (0 = dead router, 1 = perfectly uniform).
    Flags collapse when H < 0.5 · log(n_experts).

(b) **Val-PPL** of the MoTE model vs a frozen dense BitNet baseline on the
    *same* held-out tokens.  The dense baseline is loaded separately and never
    passes through ``build_mote``.

(c) **Router input-dependence** = fraction of distinct argmax-experts seen
    across all tokens (constant-argmax => dead router, value = 1/n_experts;
    all experts used as argmax at some point => 1.0).

CLI
---
  python scripts/lora/mote_eval.py \\
      --adapter .docs/mote/smoke \\
      --base microsoft/bitnet-b1.58-2B-4T-bf16 \\
      [--tiny-random]

Smoke test (CPU, seconds — uses the tiny-random adapter from ``mote_train``):
  python scripts/lora/mote_eval.py \\
      --adapter .docs/mote/smoke \\
      --base microsoft/bitnet-b1.58-2B-4T-bf16 \\
      --tiny-random

Reads from <adapter>/:
  mote_config.json   — n_experts, top_k, shared, layers, max_seq_len, tiny_random
  adapter_weights.pt — trainable MoTE weights (router + routed experts)

Writes:
  <adapter>/eval.json — all three metrics + dense reference PPL
"""

# Windows: torch.compile's Triton/Inductor back-end requires cl.exe; suppress
# dynamo errors so the script runs on Windows without a compiler tool-chain.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass  # older torch or dynamo not available

import argparse
import json
import math
import os
import sys
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow importing sibling scripts from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mote_upcycle import MoTEBlock, build_mote  # noqa: E402
from mote_train import MoTEShim, _wrap_mote_shims, _build_corpus  # noqa: E402


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_expert_counts(model: torch.nn.Module) -> Optional[torch.Tensor]:
    """Sum ``last_counts`` across all MoTEShim layers after a student forward pass.

    Returns a 1-D int64 tensor of shape [n_experts] (summed over all MoTE layers
    and all tokens in the batch), or ``None`` if no MoTE layers are present.
    """
    total: Optional[torch.Tensor] = None
    for layer in model.model.layers:
        shim = layer.mlp
        if isinstance(shim, MoTEShim) and shim.last_counts is not None:
            c = shim.last_counts.cpu()
            total = c if total is None else total + c
    return total


def _register_argmax_hooks(
    model: torch.nn.Module,
) -> tuple[list, dict]:
    """Register forward hooks on every MoTEShim router to capture per-token argmax.

    The router ``nn.Linear`` emits raw logits ``[n_tokens, n_experts]``.
    ``argmax(dim=-1)`` over those logits equals ``argmax`` over the post-softmax
    probabilities (softmax is monotonically order-preserving), so hooking the
    linear output is correct and avoids an extra pass.

    Args:
        model: The student MoTE model (with MoTEShim wrappers applied).

    Returns:
        A 2-tuple:
          * ``hooks`` — list of ``RemovableHook`` objects; call ``.remove()`` on
            each when evaluation is complete.
          * ``argmax_per_layer`` — dict mapping layer index to a list of
            ``[n_tokens]`` int64 tensors (one per forward call).
    """
    argmax_per_layer: dict = {}
    hooks: list = []

    for i, layer in enumerate(model.model.layers):
        shim = layer.mlp
        if not isinstance(shim, MoTEShim):
            continue
        lst: list = []
        argmax_per_layer[i] = lst

        def _make_hook(accumulator: list):
            def _hook(
                module: torch.nn.Module,
                _input: tuple,
                output: torch.Tensor,
            ) -> None:
                # output: [n_tokens, n_experts] raw linear logits (pre-softmax)
                accumulator.append(torch.argmax(output, dim=-1).cpu())

            return _hook

        # ``shim.router`` is the property that returns ``self.mote.router``
        # (a plain ``nn.Linear``).  Registering the hook on it directly attaches
        # to the ``nn.Linear`` module object.
        hooks.append(shim.router.register_forward_hook(_make_hook(lst)))

    return hooks, argmax_per_layer


def _compute_entropy(counts: torch.Tensor, n_experts: int) -> tuple[float, float]:
    """Compute Shannon entropy H and normalised H / log(n_experts) from dispatch counts.

    Args:
        counts: [n_experts] int64 dispatch counts across the eval corpus.
        n_experts: Total number of routed experts.

    Returns:
        ``(H, H_norm)`` where H is in nats and H_norm in [0, 1].
    """
    total = counts.sum().item()
    if total == 0:
        return 0.0, 0.0
    p = counts.float() / total
    nonzero = p > 0
    h: float = -(p[nonzero] * torch.log(p[nonzero])).sum().item()
    log_n = math.log(n_experts) if n_experts > 1 else 1.0
    return h, h / log_n


def _compute_router_dependence(argmax_per_layer: dict, n_experts: int) -> float:
    """Fraction of distinct argmax experts observed across all tokens and layers.

    **Metric definition**: Measures **top-1 argmax diversity** (expert specialisation)
    across tokens. A value of 1.0 means every expert appeared as the argmax for at
    least one token; 1/n_experts means only a single expert was ever selected
    (dead router). This is a dead-router detection metric, valid for any top_k setting.

    **For load-balance diagnostics across all top_k dispatched experts**, use the
    **expert-utilization entropy metric** instead (metric a), which captures whether
    all routed experts (not just argmax) receive roughly equal dispatch volume.

    Args:
        argmax_per_layer: Output of :func:`_register_argmax_hooks`.
        n_experts: Total number of routed experts.

    Returns:
        Fraction in [1/n_experts, 1.0], or 0.0 if no tokens were processed.
    """
    all_argmax: list = []
    for tensors in argmax_per_layer.values():
        all_argmax.extend(tensors)
    if not all_argmax:
        return 0.0
    combined = torch.cat(all_argmax, dim=0)
    n_distinct = int(combined.unique().numel())
    return n_distinct / max(n_experts, 1)


def _compute_ppl(total_nll: float, total_tokens: int) -> float:
    """Convert summed NLL (token-averaged) to perplexity.

    Args:
        total_nll: Sum of per-token negative log-likelihoods over the corpus.
        total_tokens: Number of prediction targets (seq_len - 1 per sequence).

    Returns:
        Perplexity as a float (capped at ``exp(100)`` to avoid overflow on a
        freshly-initialised tiny-random model whose loss may be large).
    """
    if total_tokens == 0:
        return float("inf")
    avg = total_nll / total_tokens
    if avg > 100.0:
        warnings.warn(
            f"[mote_eval] PPL avg NLL ({avg:.2f}) exceeds cap (100.0); "
            "capping at exp(100). Adapter may be untrained or catastrophically wrong.",
            UserWarning
        )
    return math.exp(min(avg, 100.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MoTE eval: expert entropy / val-PPL / router input-dependence"
    )
    ap.add_argument(
        "--adapter",
        required=True,
        help=(
            "Adapter directory (must contain mote_config.json and adapter_weights.pt "
            "as written by mote_train.py).  eval.json is written here."
        ),
    )
    ap.add_argument(
        "--base",
        default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="Base model checkpoint (HF hub id or local path); overridden by mote_config.json",
    )
    ap.add_argument(
        "--tiny-random",
        action="store_true",
        help=(
            "Treat the adapter as a tiny-random model (no downloads; synthetic corpus). "
            "This flag is OR-ed with the tiny_random field in mote_config.json."
        ),
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="Inference device: cpu or cuda (default: cpu)",
    )
    ap.add_argument(
        "--eval-tokens",
        type=float,
        default=5e4,
        help=(
            "Approximate held-out token count for evaluation (default: 50k). "
            "Rounded up to a whole number of sequences."
        ),
    )
    ap.add_argument(
        "--dataset",
        default="HuggingFaceH4/no_robots",
        help="HF dataset identifier (default: HuggingFaceH4/no_robots)",
    )
    ap.add_argument(
        "--dataset-split",
        default="test",
        help=(
            "HF dataset split for evaluation (default: 'test').  "
            "Must be disjoint from the training split used in mote_train.py ('train')."
        ),
    )
    args = ap.parse_args()

    adapter_dir = os.path.abspath(args.adapter)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Load MoTE config
    # ------------------------------------------------------------------
    config_path = os.path.join(adapter_dir, "mote_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"mote_config.json not found in {adapter_dir!r}")
    with open(config_path, encoding="utf-8") as fh:
        mote_cfg = json.load(fh)

    n_experts: int = mote_cfg["n_experts"]
    top_k: int = mote_cfg["top_k"]
    shared: str = mote_cfg["shared"]
    layer_indices: list = mote_cfg["layers"]
    max_seq_len: int = mote_cfg.get("max_seq_len", 512)
    # Honour tiny_random from both the CLI flag and the saved config.
    tiny_random: bool = args.tiny_random or bool(mote_cfg.get("tiny_random", False))
    # Base model: prefer saved config, CLI flag is a fallback.
    base_name: str = mote_cfg.get("base", args.base)

    print(
        f"[mote_eval] adapter={adapter_dir!r}\n"
        f"[mote_eval] n_experts={n_experts}  top_k={top_k}  "
        f"shared={shared!r}  layers={layer_indices}\n"
        f"[mote_eval] tiny_random={tiny_random}  device={device}"
    )

    # ------------------------------------------------------------------
    # 2. Tiny-random config (shared by teacher and student when active)
    # ------------------------------------------------------------------
    tiny_cfg = None
    if tiny_random:
        from transformers import AutoConfig  # noqa: F401

        tiny_cfg = AutoConfig.from_pretrained(base_name, local_files_only=True)
        tiny_cfg.hidden_size = 64
        tiny_cfg.intermediate_size = 128
        tiny_cfg.num_hidden_layers = 2
        tiny_cfg.num_attention_heads = 2
        tiny_cfg.num_key_value_heads = 2
        tiny_cfg.max_position_embeddings = 256
        # Retain real vocab_size so token IDs are valid.

    # ------------------------------------------------------------------
    # 3. Dense baseline (frozen, no MoTE) — reference PPL
    # ------------------------------------------------------------------
    if tiny_random:
        from transformers import BitNetForCausalLM

        dense = BitNetForCausalLM(tiny_cfg).to(device=device)
    else:
        dense = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    dense.config.use_cache = False
    dense.eval()
    for p in dense.parameters():
        p.requires_grad_(False)
    dense.to(device)  # defensive: device_map / .to() above should cover it; belt-and-suspenders
    print("[mote_eval] dense baseline loaded and frozen")

    # ------------------------------------------------------------------
    # 4. Student: rebuild MoTE, load adapter weights, wrap shims
    # ------------------------------------------------------------------
    if tiny_random:
        from transformers import BitNetForCausalLM

        student = BitNetForCausalLM(tiny_cfg).to(device=device)
    else:
        student = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    student.config.use_cache = False

    student = build_mote(
        student,
        layers=layer_indices,
        n_experts=n_experts,
        top_k=top_k,
        shared=shared,
    )
    # Wrap MoTEBlocks so the HF decoder layer forward receives a plain tensor.
    student = _wrap_mote_shims(student)

    # Load trainable adapter weights (router + routed experts); strict=False
    # because adapter_weights.pt contains only the trainable subset of the state dict.
    adapter_path = os.path.join(adapter_dir, "adapter_weights.pt")
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"adapter_weights.pt not found in {adapter_dir!r}")
    try:
        adapter_state = torch.load(
            adapter_path, map_location=device, weights_only=True
        )
    except TypeError:
        # weights_only not supported on this torch version — fall back gracefully.
        adapter_state = torch.load(adapter_path, map_location=device)  # type: ignore[call-arg]

    missing_keys, unexpected_keys = student.load_state_dict(
        adapter_state, strict=False
    )
    assert not unexpected_keys, (
        f"[mote_eval] adapter keys did not match the rebuilt model (config drift?): "
        f"{unexpected_keys}"
    )
    print(
        f"[mote_eval] adapter_weights.pt loaded: {len(adapter_state)} tensors  "
        f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
    )

    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    # Move the entire student graph to the eval device.  build_mote() creates new
    # nn.Linear (router) and expert modules that initialise on CPU.  device_map from
    # from_pretrained covers only the base weights loaded by that call; new modules
    # added afterwards start on CPU.  load_state_dict with map_location copies values
    # but does NOT relocate the destination parameter storage (copy_ keeps the target
    # tensor on its original device).  Calling .to(device) here is the same pattern
    # used in mote_train.py and fixes the cuda:0/cpu device-mismatch crash at
    # self.router() seen in Kaggle eval (RuntimeError: mat2 is on cpu, different
    # from other tensors on cuda:0).
    student.to(device)
    for name, p in student.named_parameters():
        assert p.device.type == device.type, (
            f"[mote_eval] parameter {name!r} is on {p.device} but eval device is {device}"
        )
    for name, buf in student.named_buffers():
        assert buf.device.type == device.type, (
            f"[mote_eval] buffer {name!r} is on {buf.device} but eval device is {device}"
        )
    print(f"[mote_eval] student fully on {device}; all params/buffers verified")

    # ------------------------------------------------------------------
    # 5. Register per-token argmax hooks on student routers
    # ------------------------------------------------------------------
    hooks, argmax_per_layer = _register_argmax_hooks(student)
    print(
        f"[mote_eval] argmax hooks registered on {len(hooks)} MoTE layer(s)"
    )

    # ------------------------------------------------------------------
    # 6. Build held-out corpus (disjoint from training)
    # ------------------------------------------------------------------
    vocab_size: int = dense.config.vocab_size
    max_seqs = max(50, int(args.eval_tokens) // max_seq_len + 1)

    tokenizer = (
        None
        if tiny_random
        else AutoTokenizer.from_pretrained(base_name)
    )

    # For real data: use 'test' split (training used 'train') to ensure disjointness.
    # For tiny_random: _build_corpus generates fresh random IDs; split arg is ignored.
    corpus = _build_corpus(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_config=None,
        dataset_split=args.dataset_split,
        max_seq_len=max_seq_len,
        max_sequences=max_seqs,
        tiny_random=tiny_random,
        vocab_size=vocab_size,
    )
    if not corpus:
        raise RuntimeError(
            "Eval corpus is empty.  "
            f"Check --dataset / --dataset-split={args.dataset_split!r}, "
            "or use --tiny-random."
        )
    print(
        f"[mote_eval] held-out corpus: {len(corpus)} sequences × {max_seq_len} tokens "
        f"(split={args.dataset_split!r})"
    )

    # ------------------------------------------------------------------
    # 7. Evaluation loop
    # ------------------------------------------------------------------
    total_student_nll: float = 0.0
    total_dense_nll: float = 0.0
    total_tokens: int = 0
    cumulative_counts: Optional[torch.Tensor] = None

    with torch.no_grad():
        for seq in corpus:
            seq_t = seq.unsqueeze(0).to(device)  # [1, T]
            n_pred = seq_t.size(1) - 1           # causal targets (shifted)

            # Dense baseline forward
            dense_logits = dense(input_ids=seq_t).logits  # [1, T, V]
            dense_nll = F.cross_entropy(
                dense_logits[:, :-1, :].contiguous().view(-1, vocab_size),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()

            # Student (MoTE) forward — argmax hooks fire here
            student_logits = student(input_ids=seq_t).logits  # [1, T, V]
            student_nll = F.cross_entropy(
                student_logits[:, :-1, :].contiguous().view(-1, vocab_size),
                seq_t[:, 1:].contiguous().view(-1),
                reduction="sum",
            ).item()

            # Accumulate expert dispatch counts from all MoTE shims
            counts = _collect_expert_counts(student)
            if counts is not None:
                cumulative_counts = (
                    counts if cumulative_counts is None else cumulative_counts + counts
                )

            total_dense_nll += dense_nll
            total_student_nll += student_nll
            total_tokens += n_pred

    # Clean up hooks now that evaluation is complete.
    for h in hooks:
        h.remove()

    print(
        f"[mote_eval] eval done: {total_tokens} prediction tokens across "
        f"{len(corpus)} sequences"
    )

    # ------------------------------------------------------------------
    # 8. Compute metrics
    # ------------------------------------------------------------------

    # (a) Expert-utilization entropy
    if cumulative_counts is not None:
        h_entropy, h_norm = _compute_entropy(cumulative_counts, n_experts)
        expert_histogram: list = [int(x) for x in cumulative_counts.tolist()]
    else:
        h_entropy, h_norm = 0.0, 0.0
        expert_histogram = []

    log_n = math.log(n_experts) if n_experts > 1 else 1.0
    collapsed: bool = (n_experts > 1) and (h_entropy < 0.5 * log_n)

    # (b) Val-PPL
    ppl_mote = _compute_ppl(total_student_nll, total_tokens)
    ppl_dense = _compute_ppl(total_dense_nll, total_tokens)
    ppl_delta = ppl_mote - ppl_dense

    # (c) Router input-dependence
    router_dependence = _compute_router_dependence(argmax_per_layer, n_experts)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    log_n_str = f"{log_n:.4f}" if n_experts > 1 else "N/A (n_experts=1)"
    print()
    print("[mote_eval] === RESULTS ===")
    print(
        f"  (a) Expert entropy  H          = {h_entropy:.4f} nats"
        f"  (H / log(N) = {h_norm:.4f}, log(N) = {log_n_str})"
    )
    print(f"      Expert histogram           = {expert_histogram}")
    print(f"      Collapsed (H < 0.5*log N)  = {collapsed}")
    print(f"  (b) Val-PPL (MoTE)             = {ppl_mote:.3f}")
    print(f"      Val-PPL (dense baseline)   = {ppl_dense:.3f}")
    print(f"      PPL delta (MoTE - dense)   = {ppl_delta:+.3f}")
    print(
        f"  (c) Router input-dependence    = {router_dependence:.4f} "
        f"({n_experts} experts; fraction with distinct argmax)"
    )
    print()

    # ------------------------------------------------------------------
    # 10. Write eval.json
    # ------------------------------------------------------------------
    results = {
        "expert_entropy_H": h_entropy,
        "expert_entropy_H_norm": h_norm,
        "expert_histogram": expert_histogram,
        "expert_collapsed": collapsed,
        "ppl_mote": ppl_mote,
        "ppl_dense": ppl_dense,
        "ppl_delta": ppl_delta,
        "router_dependence": router_dependence,
        "n_experts": n_experts,
        "top_k": top_k,
        "shared": shared,
        "eval_tokens": total_tokens,
        "eval_sequences": len(corpus),
        "eval_split": args.dataset_split,
    }
    out_path = os.path.join(adapter_dir, "eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"[mote_eval] eval.json -> {out_path}")

    if collapsed:
        print(
            "[mote_eval] WARNING: expert router appears collapsed "
            f"(H={h_entropy:.4f} < 0.5·log(N)={0.5 * log_n:.4f}).  "
            "Check expert histogram and consider longer heal-training."
        )
    print("[mote_eval] DONE")


if __name__ == "__main__":
    main()
