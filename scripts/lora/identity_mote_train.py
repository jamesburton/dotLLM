"""identity_mote_train.py — supervised-routing heal-train for identity-MoTE BitNet.

Campaign: trackM-mote — identity-expert routed-MoTE in NEW depth layers.

Pipeline
--------
  1. Load a BitNet b1.58 base (or a tiny synthetic model with --tiny-random).
  2. Depth-expand it (LLaMA-Pro): insert zero-residual ternary identity blocks
     (``bitnet_depth_expand.expand_model``) → ~2B becomes ~4.5B, model logits
     bit-for-bit unchanged at init.
  3. Convert each inserted block's FFN into an ``IdentityMoTEBlock``
     (``identity_mote.build_identity_mote``): a frozen skip expert (down_proj==0,
     the permanent no-regression fallback) + K trainable capability experts +
     a top-1 router.
  4. Freeze the base; unfreeze routers + capability experts only.
  5. VERIFY identity-at-init (max logit diff ≈ 0 vs base) — the critical invariant.
  6. Heal-train on a task-labeled multi-task corpus with

         loss = lm_loss(routed model) + route_weight * CE(router_logits, task_label)

     The supervised routing CE forces each task's data to its OWN expert
     (math→expert 1, instruction→expert 2, …), fixing the homogeneity failure mode
     of the prior null MoTE campaign.

This is BUILD-ONLY tooling. The CPU ``--tiny-random`` smoke path proves (a) identity
at init and (b) one end-to-end train step; the real GPU run is launched separately.

Differences vs mote_train.py (the FFN-expert-in-existing-layers / learned-routing
trainer) — intentionally a NEW trainer, not a modification:
  * experts live in NEW inserted layers (not existing ones);
  * routing is SUPERVISED from task labels (not a Switch load-balance aux);
  * an identity/skip expert guarantees no regression.

Smoke test (CPU, seconds, no downloads)::

    CUDA_VISIBLE_DEVICES= python scripts/lora/identity_mote_train.py \
        --config smoke --tiny-random \
        --capabilities math,instruction \
        --steps 3 --seq-len 32 --n-seqs-per-cap 4 \
        --out .docs/mote/identity_smoke

Real GPU run (launched separately — do NOT run here)::

    python scripts/lora/identity_mote_train.py \
        --config idm_2b --device cuda --grad-checkpoint \
        --every 4 --n-capability-experts 3 \
        --capabilities math,instruction,tooluse \
        --seq-len 256 --n-seqs-per-cap 4000 --tokens 6e6 \
        --route-weight 1.0 --out .docs/mote/idm_2b
"""

from __future__ import annotations

# Windows: suppress dynamo/inductor errors so training runs without a C toolchain.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# scripts/ dir (for bitnet_depth_expand) is the parent of scripts/lora/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bitnet_depth_expand as bde  # noqa: E402
from identity_mote import (  # noqa: E402
    IdentityMoTEBlock,
    build_identity_mote,
)
from multitask_routed_data import build_routed_corpus  # noqa: E402


# ---------------------------------------------------------------------------
# IdentityMoTEBlock returns a plain tensor and stashes router logits/counts as
# attributes (block.last_logits in-graph, block.last_counts detached), so it is
# directly usable as an HF decoder layer's ``.mlp`` — no shim wrapper needed.
# ---------------------------------------------------------------------------


def _iter_mote_blocks(model: nn.Module):
    for layer in model.model.layers:
        if isinstance(layer.mlp, IdentityMoTEBlock):
            yield layer.mlp


# ---------------------------------------------------------------------------
# Freeze / routing-loss helpers
# ---------------------------------------------------------------------------


def _freeze_for_training(model: nn.Module, train_inserted_attn: bool) -> None:
    """Freeze everything; unfreeze routers + capability experts (not the skip expert).

    Optionally unfreeze the inserted blocks' attention (o_proj was zeroed at init;
    off by default so inserted layers add capacity ONLY through routed FFN experts).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    inserted = set(getattr(model, "_identity_mote_layers", []))
    for li, layer in enumerate(model.model.layers):
        block = layer.mlp
        if isinstance(block, IdentityMoTEBlock):
            for p in block.router.parameters():
                p.requires_grad_(True)
            # experts[0] is the frozen skip expert; unfreeze capability experts only.
            for e in range(1, block.n_experts):
                for p in block.experts[e].parameters():
                    p.requires_grad_(True)
        if train_inserted_attn and li in inserted:
            for p in layer.self_attn.parameters():
                p.requires_grad_(True)


def _supervised_route_loss(
    model: nn.Module, labels, device: torch.device
) -> torch.Tensor:
    """Mean over identity-MoTE layers of per-token CE(router_logits, task_label).

    ``labels`` is a sequence of ``B`` per-sequence routing targets (one per sequence
    in the micro-batch). The router runs per-token, so ``block.last_logits`` is
    ``[B*T, E]`` in **sequence-major** order (row ``n`` is token ``n % T`` of
    sequence ``n // T``). Each sequence's label is therefore broadcast to its ``T``
    tokens via ``repeat_interleave(T)``, so a batch that mixes labels aligns cleanly:
    token ``n`` is supervised toward ``labels[n // T]``. In-graph.

    For ``B == 1`` this reduces to the original behaviour (the single label broadcast
    to all ``T`` tokens).
    """
    labels_t = torch.as_tensor([int(x) for x in labels], dtype=torch.long)
    B = labels_t.numel()
    total: Optional[torch.Tensor] = None
    n = 0
    for block in _iter_mote_blocks(model):
        logits = block.last_logits
        if logits is None:
            continue
        # Recover T from the flattened [B*T, E] router logits (B known from labels).
        T = logits.size(0) // B
        target = labels_t.to(logits.device).repeat_interleave(T)  # [B*T]
        ce = F.cross_entropy(logits.float(), target)
        total = ce if total is None else total + ce
        n += 1
    if total is None:
        return torch.zeros((), device=device)
    return total / n


def _collect_counts(model: nn.Module) -> Optional[torch.Tensor]:
    total: Optional[torch.Tensor] = None
    for block in _iter_mote_blocks(model):
        if block.last_counts is None:
            continue
        total = block.last_counts.clone() if total is None else total + block.last_counts
    return total


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _build_tiny_base():
    """Tiny synthetic BitNet (reuses bitnet_depth_expand.build_tiny_bitnet)."""
    return bde.build_tiny_bitnet()


def _build_real_base(base_id: str, device: torch.device):
    from transformers import AutoModelForCausalLM
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map={"": device}
        )
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Supervised-routing heal-train for identity-MoTE BitNet."
    )
    ap.add_argument("--config", required=True, help="Experiment name (logged/saved).")
    ap.add_argument("--out", required=True, help="Output dir for adapter + metrics.")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="Base BitNet checkpoint (HF id or local path).")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu).")
    ap.add_argument("--tiny-random", action="store_true",
                    help="Tiny synthetic BitNet + synthetic labeled corpus (no downloads).")

    # Depth expansion
    ap.add_argument("--every", type=int, default=None,
                    help="LLaMA-Pro interleave: insert one identity block after every "
                         "k original layers (e.g. --every 4 on 30 layers → ~37 layers).")
    ap.add_argument("--at", type=str, default=None,
                    help="Comma-separated original indices to insert before "
                         "(mutually exclusive with --every).")

    # Identity-MoTE
    ap.add_argument("--n-capability-experts", type=int, default=None,
                    help="K capability experts per inserted block. Default: number of "
                         "loaded capabilities (so one expert per capability).")
    ap.add_argument("--capability-init", choices=["zero", "template"], default="zero",
                    help="zero = bulletproof identity (down_proj zeroed, any router); "
                         "template = warm-start full neighbour FFN (identity via router bias).")
    ap.add_argument("--router-identity-bias", type=float, default=0.0,
                    help="Initial skip-logit bias. 0 for zero-init; set large (e.g. 30) "
                         "for capability-init=template so identity holds at init.")
    ap.add_argument("--train-inserted-attn", action="store_true",
                    help="Also train the inserted blocks' attention (default off).")

    # Corpus
    ap.add_argument("--capabilities", default="math,instruction,tooluse",
                    help="Comma-separated capabilities (label order). Label 0 = skip.")
    ap.add_argument("--n-seqs-per-cap", type=int, default=64,
                    help="Sequences per capability.")
    ap.add_argument("--seq-len", type=int, default=256, help="Tokens per sequence.")

    # Training
    ap.add_argument("--tokens", type=float, default=0.0,
                    help="Approx training-token budget. 0 = one full pass over the corpus.")
    ap.add_argument("--steps", type=int, default=0,
                    help="Hard cap on optimizer steps (0 = unlimited; used for smoke).")
    ap.add_argument("--route-weight", type=float, default=1.0,
                    help="Weight of the supervised routing CE loss.")
    ap.add_argument("--lr", type=float, default=1e-4, help="Expert LR.")
    ap.add_argument("--router-lr", type=float, default=1e-3,
                    help="Router LR (higher: the router is tiny and must move logits fast).")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Micro-batch: sequences stacked per optimizer step ([B, seq_len]). "
                         "Default 1 preserves the original per-sequence path. The LM loss is the "
                         "batch mean; the supervised routing CE stays per-sequence (each sequence's "
                         "label is broadcast to its own tokens, so a batch may mix labels).")
    ap.add_argument("--optim", choices=["adamw", "adamw-fused", "adafactor", "adamw8bit"],
                    default="adamw",
                    help="Optimizer. 'adamw' (default) AUTO-USES fused AdamW on CUDA — the fast "
                         "path on memory-rich GPUs (Strix); 'adamw-fused' forces it (both fall "
                         "back to foreach/default if fused is unavailable). 'adafactor' is the "
                         "memory-light path (12GB 3060). 'adamw8bit' tries bitsandbytes w/ "
                         "adafactor fallback.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Gradient checkpointing (activation recompute) — for the 12GB 3060. "
                         "OMIT on memory-rich GPUs (Strix) for ~25-35%% speedup (it is opt-in).")
    ap.add_argument("--verify-identity", action="store_true", default=True,
                    help="Verify identity-at-init vs base before training (default on).")
    ap.add_argument("--no-verify-identity", dest="verify_identity", action="store_false")
    ap.add_argument("--identity-tol", type=float, default=1e-3,
                    help="Max-abs logit tolerance for the identity-at-init check.")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]

    print(f"[idm] config={args.config!r} device={device} tiny_random={args.tiny_random}")
    print(f"[idm] capabilities(requested)={caps}")

    # ------------------------------------------------------------------
    # 1. Tokenizer (real only) + base model
    # ------------------------------------------------------------------
    tokenizer = None
    if not args.tiny_random:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base)

    if args.tiny_random:
        base_model = _build_tiny_base().to(device)
    else:
        base_model = _build_real_base(args.base, device)
    base_model.eval()
    vocab_size = base_model.config.vocab_size

    # ------------------------------------------------------------------
    # 2. Routed multi-task corpus (built first: determines K = #capabilities loaded)
    # ------------------------------------------------------------------
    sequences, labels, label_map = build_routed_corpus(
        tokenizer=tokenizer,
        capabilities=caps,
        n_seqs_per_cap=args.n_seqs_per_cap,
        seq_len=args.seq_len,
        tiny_random=args.tiny_random,
        vocab_size=vocab_size,
    )
    n_caps = len(label_map)
    K = args.n_capability_experts if args.n_capability_experts is not None else n_caps
    if K < n_caps:
        print(f"[idm] WARNING: K={K} < #capabilities={n_caps}; labels > K will be unroutable.")
    print(f"[idm] label_map={label_map}  K(capability experts)={K}  "
          f"n_experts={K + 1} (incl. skip)")
    print(f"[idm] corpus: {len(sequences)} sequences x {args.seq_len} tokens")

    # ------------------------------------------------------------------
    # 3. Snapshot base logits for the identity check (before in-place expansion)
    # ------------------------------------------------------------------
    id_check_ids = None
    base_logits_snapshot = None
    if args.verify_identity:
        torch.manual_seed(0)
        _t = min(16, args.seq_len)
        id_check_ids = torch.randint(0, vocab_size, (1, _t), device=device)
        with torch.no_grad():
            base_logits_snapshot = base_model(id_check_ids, use_cache=False).logits.float().cpu()

    # ------------------------------------------------------------------
    # 4. Depth-expand (insert identity blocks)
    # ------------------------------------------------------------------
    at = [int(x) for x in args.at.split(",")] if args.at else None
    every = args.every
    if every is None and at is None:
        every = 1  # default: interleave one block after every original layer
    n_layers_before = base_model.config.num_hidden_layers
    positions = bde.plan_insertions(n_layers_before, every=every, at=at)
    model, info = bde.expand_model(base_model, positions)
    inserted_indices = info["inserted_indices"]
    print(f"[idm] depth-expand: {info['original_layers']} -> {info['final_layers']} "
          f"layers (+{info['inserted']} identity at {inserted_indices})")

    # ------------------------------------------------------------------
    # 5. Convert inserted blocks -> IdentityMoTE + shims
    # ------------------------------------------------------------------
    model = build_identity_mote(
        model,
        inserted_indices=inserted_indices,
        n_capability_experts=K,
        capability_init=args.capability_init,
        router_identity_bias=args.router_identity_bias,
        top_k=1,
    )
    model.to(device)
    model.config.use_cache = False

    # ------------------------------------------------------------------
    # 6. Identity-at-init verification (THE critical invariant)
    # ------------------------------------------------------------------
    if args.verify_identity:
        model.eval()
        with torch.no_grad():
            mote_logits = model(id_check_ids, use_cache=False).logits.float().cpu()
        max_abs = (mote_logits - base_logits_snapshot).abs().max().item()
        argmax_ok = bool(
            (mote_logits.argmax(-1) == base_logits_snapshot.argmax(-1)).all().item()
        )
        print(f"[idm] IDENTITY@init: max_logit_diff={max_abs:.3e} "
              f"argmax_match={argmax_ok} (tol={args.identity_tol:.1e})")
        if max_abs > args.identity_tol or not argmax_ok:
            raise RuntimeError(
                f"IDENTITY INVARIANT BROKEN at init (max_abs={max_abs:.3e}). "
                "Refusing to train — the whole no-regression guarantee rests on this."
            )
        # Bulletproof check for zero-init: identity must survive a RANDOM router.
        if args.capability_init == "zero":
            saved = {}
            with torch.no_grad():
                for i, block in enumerate(_iter_mote_blocks(model)):
                    saved[i] = (block.router.weight.clone(), block.router.bias.clone())
                    block.router.weight.normal_(0.0, 5.0)
                    block.router.bias.normal_(0.0, 5.0)
                rnd_logits = model(id_check_ids, use_cache=False).logits.float().cpu()
                for i, block in enumerate(_iter_mote_blocks(model)):
                    block.router.weight.copy_(saved[i][0])
                    block.router.bias.copy_(saved[i][1])
            rnd_max = (rnd_logits - base_logits_snapshot).abs().max().item()
            print(f"[idm] IDENTITY@init (RANDOM router, zero-init): max_logit_diff={rnd_max:.3e}")
            if rnd_max > args.identity_tol:
                raise RuntimeError(
                    f"zero-init identity failed under random router (max_abs={rnd_max:.3e})."
                )

    # ------------------------------------------------------------------
    # 7. Freeze + optimizer
    # ------------------------------------------------------------------
    _freeze_for_training(model, train_inserted_attn=args.train_inserted_attn)

    router_params, expert_params = [], []
    for block in _iter_mote_blocks(model):
        router_params += [p for p in block.router.parameters() if p.requires_grad]
        for e in range(1, block.n_experts):
            expert_params += [p for p in block.experts[e].parameters() if p.requires_grad]
    # inserted-attn params (if enabled) train at the expert LR.
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in {id(x) for x in router_params + expert_params}
    ]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[idm] trainable={trainable:,} / total={total:,} "
          f"({100 * trainable / max(total, 1):.3f}%)  "
          f"[router={sum(p.numel() for p in router_params):,} "
          f"experts={sum(p.numel() for p in expert_params):,}]")

    param_groups = [
        {"params": router_params, "lr": args.router_lr},
        {"params": expert_params + other_params, "lr": args.lr},
    ]
    opt = _make_optimizer(args.optim, param_groups, args.lr, device)

    if args.grad_checkpoint:
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[idm] gradient checkpointing ENABLED")

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    model.train()
    B = max(1, int(args.batch_size))
    max_tokens = int(args.tokens) if args.tokens and args.tokens > 0 else None
    max_steps = args.steps if args.steps and args.steps > 0 else None
    tokens_seen = 0
    step = 0
    final_lm = final_route = float("nan")
    recent_counts: list = []
    order = list(range(len(sequences)))
    _t0 = time.perf_counter()

    def _budget_left() -> bool:
        if max_steps is not None and step >= max_steps:
            return False
        if max_tokens is not None and tokens_seen >= max_tokens:
            return False
        if max_steps is None and max_tokens is None:
            return step * B < len(sequences)  # one pass (B sequences per step)
        return True

    while _budget_left():
        # Micro-batch of B sequences (wraps around the corpus); each carries its own
        # per-sequence routing label, so the batch may mix labels.
        batch_idx = [order[(step * B + j) % len(order)] for j in range(B)]
        seqs = torch.stack([sequences[i] for i in batch_idx]).to(device)  # [B, T]
        batch_labels = [labels[i] for i in batch_idx]

        logits = model(input_ids=seqs, use_cache=False).logits  # [B, T, V]
        lm_loss = F.cross_entropy(  # mean over all next-token positions == batch mean
            logits[:, :-1, :].contiguous().view(-1, vocab_size),
            seqs[:, 1:].contiguous().view(-1),
        )
        route_loss = _supervised_route_loss(model, batch_labels, device)
        loss = lm_loss + args.route_weight * route_loss

        lm_val = lm_loss.item()
        route_val = route_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        opt.zero_grad()

        # Per-step dispatch histogram: last_counts already sums over ALL B*T tokens
        # in every mote block, so it is meaningful for a batch as-is.
        counts = _collect_counts(model)
        if counts is not None:
            recent_counts.append(counts.cpu())
            if len(recent_counts) > 100:
                recent_counts.pop(0)

        tokens_seen += seqs.numel()  # B * T
        step += 1
        final_lm, final_route = lm_val, route_val
        if step <= 5 or step % 20 == 0:
            # Compact per-label multiplicity for the batch (sensible when B > 1).
            lbl_counts = {l: batch_labels.count(l) for l in sorted(set(batch_labels))}
            lbl_str = str(batch_labels[0]) if B == 1 else str(lbl_counts)
            print(f"  step {step:5d}  tok {tokens_seen:.2e}  labels {lbl_str}  "
                  f"lm {lm_val:.4f}  route {route_val:.4f}", flush=True)

    elapsed = time.perf_counter() - _t0
    print(f"[idm] training done: {step} steps / {tokens_seen} tokens in {elapsed:.1f}s")
    print(f"[idm] final: lm={final_lm:.4f}  route={final_route:.4f}")

    # ------------------------------------------------------------------
    # 9. Save adapter (trainable tensors only) + config + metrics
    # ------------------------------------------------------------------
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    adapter = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
    torch.save(adapter, os.path.join(args.out, "adapter_weights.pt"))

    histo = torch.stack(recent_counts).sum(0).tolist() if recent_counts else []
    experts_used = sum(1 for c in histo if c > 0)

    cfg = {
        "config": args.config,
        "base": args.base,
        "capabilities": caps,
        "label_map": label_map,
        "n_capability_experts": K,
        "n_experts": K + 1,
        "capability_init": args.capability_init,
        "inserted_indices": inserted_indices,
        "layers_before": n_layers_before,
        "layers_after": info["final_layers"],
        "seq_len": args.seq_len,
        "batch_size": B,
        "optim": args.optim,
        "steps": step,
        "tokens": tokens_seen,
        "route_weight": args.route_weight,
        "tiny_random": args.tiny_random,
    }
    with open(os.path.join(args.out, "identity_mote_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)

    metrics = {
        "lm_loss": final_lm,
        "route_loss": final_route,
        "expert_counts": histo,
        "experts_used": experts_used,
        "steps": step,
        "tokens": tokens_seen,
    }
    if device.type == "cuda":
        metrics["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[idm] adapter+config+metrics -> {args.out}")
    print(f"[idm] expert dispatch histogram (recent): {histo}  experts_used={experts_used}")

    # ------------------------------------------------------------------
    # 10. Smoke gate
    # ------------------------------------------------------------------
    lm_ok = final_lm == final_lm  # not NaN
    route_ok = final_route == final_route
    print(f"\n[idm] GATE: lm_finite={lm_ok}  route_finite={route_ok}  steps={step}")
    if not (lm_ok and route_ok and step > 0):
        raise RuntimeError("Smoke gate FAILED — see GATE line above.")
    print("[idm] GATE PASSED")


def _build_adamw(param_groups, lr: float, device: torch.device, prefer_fused: bool):
    """torch.optim.AdamW with the fused CUDA kernel when available.

    Fused AdamW (``fused=True``) is a single-kernel foreach optimizer that is the
    fast path on memory-rich GPUs (e.g. Strix). It is CUDA-only; on CPU or when the
    build lacks it we fall back to ``foreach=True`` and then to the default impl.
    """
    kwargs = dict(betas=(0.9, 0.95), weight_decay=0.0)
    if prefer_fused and device.type == "cuda":
        try:
            opt = torch.optim.AdamW(param_groups, lr=lr, fused=True, **kwargs)
            print("[idm] optimizer: AdamW (fused CUDA kernel — fast path on memory-rich GPUs)")
            return opt
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"[idm] fused AdamW unavailable ({exc}); falling back to foreach/default")
    try:
        opt = torch.optim.AdamW(param_groups, lr=lr, foreach=True, **kwargs)
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(param_groups, lr=lr, **kwargs)
    print(f"[idm] optimizer: AdamW ({'foreach' if device.type != 'cuda' else 'non-fused'})")
    return opt


def _make_optimizer(choice: str, param_groups, lr: float, device: torch.device):
    if choice == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(param_groups, lr=lr, relative_step=False,
                         scale_parameter=False, warmup_init=False)
    if choice == "adamw8bit":
        try:
            import bitsandbytes as bnb
            try:
                opt = bnb.optim.PagedAdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
            except AttributeError:
                opt = bnb.optim.AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
            _vp = torch.zeros(8, device=device, requires_grad=True)
            _vo = type(opt)([_vp], lr=lr)
            _vp.sum().backward(); _vo.step(); _vo.zero_grad()
            print("[idm] optimizer: bitsandbytes AdamW8bit (verified)")
            return opt
        except Exception as exc:  # noqa: BLE001
            print(f"[idm] bnb unavailable ({exc}); falling back to AdamW")
    # 'adamw' auto-uses fused on CUDA; 'adamw-fused' forces the same preference
    # (both degrade to foreach/default off-CUDA or when fused is unavailable).
    prefer_fused = choice in ("adamw", "adamw-fused")
    return _build_adamw(param_groups, lr, device, prefer_fused=prefer_fused)


if __name__ == "__main__":
    main()
