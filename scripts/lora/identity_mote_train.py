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

Expert-parallel (Branch-Train-MiX) modes
-----------------------------------------
Because a MoTE's experts are DISJOINT parameter blocks over a FROZEN shared base,
each capability expert can be trained INDEPENDENTLY (as if on a different machine)
and then losslessly slotted into one MoTE; only the router needs joint retraining.
Three modes, all reproducible from a shared ``base_spec``:

  * ``--train-only-expert CAP``  (BRANCH) — freeze base + all other experts/routers,
    force every token through CAP's slot, and train ONLY that expert on ITS data.
    ``--capabilities`` must still list the FULL canonical roster (fixes slot order).
    Emits ``expert_<CAP>.pt`` (that slot's weights + base_spec).
  * ``mote_merge_experts.py``    (MIX)    — merge N per-expert checkpoints into one
    MoTE (lossless slot concatenation; asserts slot-compatibility). Separate tool.
  * ``--train-only-router --load-experts <merged>``  (COMBINE) — slot the experts in,
    freeze ALL experts, train ONLY the routers (supervised routing CE). Emits
    ``mote_combined.pt`` (all slots + routers + base_spec) for eval.

End-to-end CPU PoC: ``scripts/test_expert_parallel.py`` (proves branch/merge/combine,
losslessness, correct routing, and no-regression on the tiny synthetic model).

    # BRANCH (run each on its own machine; --capabilities is the shared roster):
    python scripts/lora/identity_mote_train.py --config mA --tiny-random --tiny-distinct \
        --capabilities math,instruction --train-only-expert math --steps 40 --out mA
    python scripts/lora/identity_mote_train.py --config mB --tiny-random --tiny-distinct \
        --capabilities math,instruction --train-only-expert instruction --steps 40 --out mB
    # MIX:
    python scripts/lora/mote_merge_experts.py --experts mA/expert_math.pt \
        mB/expert_instruction.pt --out merged/merged_experts.pt
    # COMBINE:
    python scripts/lora/identity_mote_train.py --config combine --tiny-random --tiny-distinct \
        --train-only-router --load-experts merged/merged_experts.pt --steps 120 --out combine

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

Real GPU run (launched separately — do NOT run here). ``general`` maps to label 0
(the frozen skip expert), so general-web text trains the router to leave base
behaviour untouched; ``--balance`` equalizes the label mix so no capability starves::

    python scripts/lora/identity_mote_train.py \
        --config idm_2b --device cuda --grad-checkpoint --batch-size 4 \
        --every 4 \
        --capabilities general,math,instruction,coding --balance \
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
from multitask_routed_data import (  # noqa: E402
    build_routed_corpus,
    _assign_labels as _assign_routing_labels,
)


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
    model: nn.Module, labels, device: torch.device, label_weights: Optional[dict] = None
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

    ``label_weights`` (optional) maps ``label -> multiplier``: tokens whose label is
    listed get that weight in a per-token weighted CE. Used by ``--route-force-weight``
    to push extra routing pressure onto a STARVED expert so training drives its tokens
    to the intended label. ``None`` (default) is the unweighted mean.
    """
    labels_t = torch.as_tensor([int(x) for x in labels], dtype=torch.long)
    B = labels_t.numel()
    if label_weights:
        w_seq = torch.tensor(
            [float(label_weights.get(int(x), 1.0)) for x in labels], dtype=torch.float32
        )
    else:
        w_seq = None
    total: Optional[torch.Tensor] = None
    n = 0
    for block in _iter_mote_blocks(model):
        logits = block.last_logits
        if logits is None:
            continue
        # Recover T from the flattened [B*T, E] router logits (B known from labels).
        T = logits.size(0) // B
        target = labels_t.to(logits.device).repeat_interleave(T)  # [B*T]
        if w_seq is not None:
            w_tok = w_seq.to(logits.device).repeat_interleave(T)  # [B*T]
            ce_tok = F.cross_entropy(logits.float(), target, reduction="none")
            ce = (ce_tok * w_tok).sum() / w_tok.sum().clamp_min(1e-8)
        else:
            ce = F.cross_entropy(logits.float(), target)
        total = ce if total is None else total + ce
        n += 1
    if total is None:
        return torch.zeros((), device=device)
    return total / n


# ---------------------------------------------------------------------------
# Routing-imbalance monitoring
# ---------------------------------------------------------------------------


def _data_label_fracs(labels, n_experts: int) -> torch.Tensor:
    """Expected (target) dispatch fractions = per-label share of the training data."""
    hist = torch.zeros(n_experts, dtype=torch.float64)
    for l in labels:
        li = int(l)
        if 0 <= li < n_experts:
            hist[li] += 1.0
    s = hist.sum()
    return (hist / s) if s > 0 else hist


def _dispatch_fracs(recent_counts: list, n_experts: int) -> torch.Tensor:
    """Observed dispatch fractions over the rolling window of per-step counts."""
    if not recent_counts:
        return torch.zeros(n_experts, dtype=torch.float64)
    tot = torch.stack(recent_counts).sum(0).to(torch.float64)
    s = tot.sum()
    return (tot / s) if s > 0 else tot


def _route_monitor(disp: torch.Tensor, target: torch.Tensor, threshold: float):
    """Compare observed vs target dispatch fractions.

    Returns ``(warnings, starved, overused)`` where ``starved``/``overused`` are sets
    of labels whose dispatch fraction is below ``1/threshold`` / above ``threshold``
    times their data fraction (default threshold 2.0 => the 0.5x / 2x band).
    """
    warnings: list = []
    starved: set = set()
    overused: set = set()
    for e in range(target.numel()):
        df = float(target[e])
        if df <= 0.0:
            continue
        pf = float(disp[e]) if e < disp.numel() else 0.0
        ratio = pf / df if df > 0 else 0.0
        if ratio < (1.0 / threshold):
            starved.add(e)
            warnings.append(
                f"label {e} STARVED: dispatch {pf:.3f} << data {df:.3f} (ratio {ratio:.2f}x)"
            )
        elif ratio > threshold:
            overused.add(e)
            warnings.append(
                f"label {e} OVERUSED: dispatch {pf:.3f} >> data {df:.3f} (ratio {ratio:.2f}x)"
            )
    return warnings, starved, overused


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
# Expert-parallel (Branch-Train-MiX style) construction / freeze / slot I/O
# ---------------------------------------------------------------------------
#
# The design premise: a MoE's experts are DISJOINT parameter blocks. A depth-
# expanded BitNet with a FROZEN shared base and ZERO-init experts (capability_init
# ="zero") is fully reproducible: two independent processes that build it from the
# same ``base_spec`` get bit-identical initial experts and routers (no RNG is
# consumed after the deterministic base build). So each capability expert can be
# trained ALONE against the identical base ("on a different machine"), and merging
# is a lossless slot-wise concatenation — never weight averaging. Only the router
# needs joint (re)training, and supervised routing makes that cheap.


def _resolve_positions(n_layers: int, every, at):
    """Depth-expansion insertion plan from --every / --at (str or list), with the
    trainer's default (every=1) when neither is given. Returned list is the set of
    ORIGINAL layer indices after which an identity block is inserted."""
    at_list = None
    if isinstance(at, str) and at.strip():
        at_list = [int(x) for x in at.split(",") if x.strip() != ""]
    elif isinstance(at, (list, tuple)):
        at_list = [int(x) for x in at]
    ev = every
    if ev is None and at_list is None:
        ev = 1
    return bde.plan_insertions(n_layers, every=ev, at=at_list)


def build_base_only(base: str, tiny_random: bool, device: torch.device):
    """Build just the (pre-expansion) base model — used for the identity reference."""
    m = _build_tiny_base().to(device) if tiny_random else _build_real_base(base, device)
    m.eval()
    return m


def construct_mote(
    *,
    base: str,
    tiny_random: bool,
    device: torch.device,
    every,
    at,
    n_capability_experts: int,
    capability_init: str,
    router_identity_bias: float,
    top_k: int = 1,
):
    """Deterministically build base -> depth-expand -> IdentityMoTE.

    Returns ``(model, info, n_layers_before)``. ``model`` IS the expanded base
    (``expand_model`` mutates in place). No RNG is consumed after the base build,
    so two calls with identical args produce bit-identical experts/routers — the
    property that makes independent per-expert training slot-compatible.
    """
    base_model = build_base_only(base, tiny_random, device)
    n_layers_before = base_model.config.num_hidden_layers
    positions = _resolve_positions(n_layers_before, every, at)
    model, info = bde.expand_model(base_model, positions)
    model = build_identity_mote(
        model,
        inserted_indices=info["inserted_indices"],
        n_capability_experts=n_capability_experts,
        capability_init=capability_init,
        router_identity_bias=router_identity_bias,
        top_k=top_k,
    )
    model.to(device)
    model.config.use_cache = False
    return model, info, n_layers_before


def make_base_spec(
    *,
    base: str,
    tiny_random: bool,
    every,
    at,
    capabilities: list,
    label_map: dict,
    n_capability_experts: int,
    capability_init: str,
    router_identity_bias: float,
    inserted_indices: list,
    n_layers_before: int,
    n_layers_after: int,
    hidden_size: int,
    vocab_size: int,
    seq_len: int,
    top_k: int = 1,
) -> dict:
    """The reproducible recipe needed to rebuild the IDENTICAL MoTE elsewhere.

    Every field here is a construction input; two builders that agree on this spec
    produce slot-compatible models. Used by the per-expert checkpoint, the merge
    tool, and the router-retrain step.
    """
    return {
        "base": base,
        "tiny_random": bool(tiny_random),
        "every": every,
        "at": at,
        "capabilities": list(capabilities),
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "n_capability_experts": int(n_capability_experts),
        "capability_init": capability_init,
        "router_identity_bias": float(router_identity_bias),
        "inserted_indices": list(inserted_indices),
        "n_layers_before": int(n_layers_before),
        "n_layers_after": int(n_layers_after),
        "hidden_size": int(hidden_size),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "top_k": int(top_k),
    }


def construct_mote_from_spec(base_spec: dict, device: torch.device):
    """Rebuild the exact MoTE described by ``base_spec`` (used by merge + PoC checks)."""
    model, info, n_before = construct_mote(
        base=base_spec["base"],
        tiny_random=base_spec["tiny_random"],
        device=device,
        every=base_spec.get("every"),
        at=base_spec.get("at"),
        n_capability_experts=base_spec["n_capability_experts"],
        capability_init=base_spec["capability_init"],
        router_identity_bias=base_spec["router_identity_bias"],
        top_k=base_spec.get("top_k", 1),
    )
    if info["inserted_indices"] != base_spec["inserted_indices"]:
        raise RuntimeError(
            "rebuild mismatch: inserted_indices "
            f"{info['inserted_indices']} != spec {base_spec['inserted_indices']}"
        )
    return model, info


def specs_compatible(a: dict, b: dict) -> tuple[bool, list]:
    """Two base_specs are slot-compatible iff they agree on every construction field
    that affects the model's parameter layout. Returns (ok, list_of_diffs)."""
    keys = [
        "base", "tiny_random", "every", "at", "n_capability_experts",
        "capability_init", "inserted_indices", "n_layers_before",
        "n_layers_after", "hidden_size", "vocab_size", "top_k",
    ]
    diffs = [k for k in keys if a.get(k) != b.get(k)]
    return (len(diffs) == 0), diffs


def _mote_layer_indices(model: nn.Module) -> list:
    return list(getattr(model, "_identity_mote_layers", []))


def expert_slot_weights(model: nn.Module, slot: int) -> dict:
    """Extract ONLY expert ``slot``'s per-layer weights, namespaced ``"{layer}.{suffix}"``.

    This is the per-expert payload: disjoint from every other slot, so it can be
    slotted back into an identically-built MoTE losslessly.
    """
    out: dict = {}
    for layer_idx in _mote_layer_indices(model):
        block = model.model.layers[layer_idx].mlp
        expert = block.experts[slot]
        for name, p in expert.state_dict().items():
            out[f"{layer_idx}.{name}"] = p.detach().cpu().clone()
    return out


def load_expert_slot_weights(model: nn.Module, slot: int, weights: dict) -> None:
    """Load a per-expert payload (from :func:`expert_slot_weights`) into ``slot``."""
    for layer_idx in _mote_layer_indices(model):
        block = model.model.layers[layer_idx].mlp
        expert = block.experts[slot]
        ref = expert.state_dict()
        new_sd = {}
        for name in ref:
            key = f"{layer_idx}.{name}"
            if key not in weights:
                raise KeyError(f"missing expert weight {key!r} for slot {slot}")
            new_sd[name] = weights[key].to(ref[name].dtype)
        expert.load_state_dict(new_sd)


def router_slot_weights(model: nn.Module) -> dict:
    """Extract every router's weights, namespaced ``"{layer}.{suffix}"``."""
    out: dict = {}
    for layer_idx in _mote_layer_indices(model):
        block = model.model.layers[layer_idx].mlp
        for name, p in block.router.state_dict().items():
            out[f"{layer_idx}.{name}"] = p.detach().cpu().clone()
    return out


def load_router_slot_weights(model: nn.Module, weights: dict) -> None:
    for layer_idx in _mote_layer_indices(model):
        block = model.model.layers[layer_idx].mlp
        ref = block.router.state_dict()
        new_sd = {name: weights[f"{layer_idx}.{name}"].to(ref[name].dtype) for name in ref}
        block.router.load_state_dict(new_sd)


def _freeze_only_expert(model: nn.Module, slot: int, train_router: bool) -> None:
    """Freeze EVERYTHING except capability expert ``slot`` (Branch-Train per-expert).

    The base, the skip expert, ALL other capability experts, and (by default) the
    router are frozen; only slot ``slot`` trains. With ``train_router`` the router is
    also unfrozen (its "slice"), but the router is NOT part of the per-expert payload
    — it is (re)trained jointly in the merge/combine step regardless.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for block in _iter_mote_blocks(model):
        for p in block.experts[slot].parameters():
            p.requires_grad_(True)
        if train_router:
            for p in block.router.parameters():
                p.requires_grad_(True)


def _freeze_only_router(model: nn.Module) -> None:
    """Freeze ALL experts (skip + capability); train ONLY the routers (the combine step)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for block in _iter_mote_blocks(model):
        for p in block.router.parameters():
            p.requires_grad_(True)


CKPT_FORMAT = "idm-expert-parallel/v1"


def read_experts_arg(spec: str):
    """Resolve ``--load-experts`` into ``(slots, base_spec)``.

    ``spec`` is either a merged-experts checkpoint (``kind == "merged"`` from
    mote_merge_experts.py) or a comma-separated list of per-expert checkpoints
    (``kind == "expert"``). Returns ``slots = {slot:int -> {capability,label,weights}}``
    and the shared ``base_spec`` (asserted identical across per-expert files).
    """
    paths = [p.strip() for p in spec.split(",") if p.strip()]
    slots: dict = {}
    base_spec = None
    for path in paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("format") != CKPT_FORMAT:
            raise ValueError(f"{path}: not an {CKPT_FORMAT} checkpoint")
        bs = obj["base_spec"]
        if base_spec is None:
            base_spec = bs
        else:
            ok, diffs = specs_compatible(base_spec, bs)
            if not ok:
                raise ValueError(f"{path}: base_spec incompatible with earlier file(s): {diffs}")
        if obj.get("kind") == "merged":
            for s, payload in obj["slots"].items():
                slots[int(s)] = payload
        elif obj.get("kind") == "expert":
            slots[int(obj["slot"])] = {
                "capability": obj.get("capability"),
                "label": obj.get("label", int(obj["slot"])),
                "weights": obj["weights"],
            }
        else:
            raise ValueError(f"{path}: unknown kind {obj.get('kind')!r}")
    return slots, base_spec


def save_expert_checkpoint(path, *, capability, slot, base_spec, weights, metrics=None):
    """Persist ONE capability expert's per-layer weights (the BRANCH output)."""
    torch.save(
        {
            "format": CKPT_FORMAT,
            "kind": "expert",
            "capability": capability,
            "slot": int(slot),
            "label": int(slot),
            "base_spec": base_spec,
            "weights": weights,
            "metrics": metrics or {},
        },
        path,
    )


def save_combined_checkpoint(path, *, base_spec, slots, router):
    """Persist a complete MoTE spec: all capability-expert slots + trained routers.

    This is the COMBINE output — one artifact an eval harness can rebuild the whole
    merged, router-retrained model from (base_spec drives construction; ``slots``
    fills the experts; ``router`` fills the routers). The frozen skip expert and base
    are omitted (deterministic from base_spec).
    """
    torch.save(
        {
            "format": CKPT_FORMAT,
            "kind": "combined",
            "base_spec": base_spec,
            "slots": slots,   # {slot:int -> {capability,label,weights}}
            "router": router,  # {"{layer}.{suffix}" -> tensor}
        },
        path,
    )


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
    ap.add_argument("--balance", action="store_true",
                    help="Balance the corpus across labels: oversample under-represented "
                         "capabilities (e.g. no_robots @ 544) and cap over-represented ones "
                         "so every expert sees an equal share and no cap exhausts late.")
    ap.add_argument("--balance-target", type=int, default=None,
                    help="Per-capability target count for --balance (default: max loaded).")

    # Training
    ap.add_argument("--tokens", type=float, default=0.0,
                    help="Approx training-token budget. 0 = one full pass over the corpus.")
    ap.add_argument("--steps", type=int, default=0,
                    help="Hard cap on optimizer steps (0 = unlimited; used for smoke).")
    ap.add_argument("--route-weight", type=float, default=1.0,
                    help="Weight of the supervised routing CE loss.")
    ap.add_argument("--route-monitor", action="store_true", default=True,
                    help="Monitor per-label dispatch fractions vs the data mix and WARN on "
                         "imbalance (default ON, log-only).")
    ap.add_argument("--no-route-monitor", dest="route_monitor", action="store_false")
    ap.add_argument("--route-imbalance-threshold", type=float, default=2.0,
                    help="Warn when a label's dispatch fraction is <1/T or >T times its data "
                         "fraction (default 2.0 => the 0.5x/2x band).")
    ap.add_argument("--route-force-weight", type=float, default=0.0,
                    help="Correction knob (opt-in, default 0=off). When >0, tokens of a "
                         "capability whose expert the monitor flags as STARVED get their "
                         "supervised-routing CE multiplied by (1 + this), pushing training to "
                         "drive those tokens to the intended expert.")
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

    # Expert-parallel (Branch-Train-MiX) modes — mutually exclusive with each other.
    ap.add_argument("--train-only-expert", default=None,
                    help="BRANCH step: freeze the base AND all experts/routers except "
                         "the named capability's expert, and train ONLY that expert on "
                         "ITS single-capability data. --capabilities must still list the "
                         "FULL canonical roster (it fixes the slot ordering); only the "
                         "named capability's data is loaded. Output: a per-expert "
                         "checkpoint (that slot's weights only) + base_spec, ready to "
                         "merge. Simulates training on a separate machine.")
    ap.add_argument("--train-expert-router-slice", action="store_true",
                    help="With --train-only-expert, also unfreeze the router while "
                         "training the expert. The router is discarded at merge (it is "
                         "re-trained jointly in the combine step); off by default.")
    ap.add_argument("--train-only-router", action="store_true",
                    help="COMBINE step: freeze ALL experts (skip + capability) and train "
                         "ONLY the routers on the full multi-capability labeled corpus "
                         "(supervised routing CE). Use with --load-experts to slot in the "
                         "independently-trained experts first.")
    ap.add_argument("--load-experts", default=None,
                    help="Path to a merged-experts checkpoint (from mote_merge_experts.py) "
                         "OR a comma-separated list of per-expert checkpoints. Their slots "
                         "are loaded into the freshly-built MoTE before training. The "
                         "base_spec in the file(s) drives construction (guarantees an "
                         "identical, slot-compatible model). Typically paired with "
                         "--train-only-router.")
    ap.add_argument("--tiny-distinct", action="store_true",
                    help="(--tiny-random only) draw each capability's synthetic ids from a "
                         "DISJOINT vocab band so capabilities are content-separable and the "
                         "supervised router can actually learn label-i -> expert-i. Used by "
                         "the expert-parallel PoC to prove correct routing on a tiny model.")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    from collections import Counter

    # ------------------------------------------------------------------
    # 0. Mode resolution (Branch-Train-MiX). If loading experts, the file's
    #    base_spec DRIVES construction so the rebuilt model is slot-compatible.
    # ------------------------------------------------------------------
    if args.train_only_expert and args.train_only_router:
        raise SystemExit("--train-only-expert and --train-only-router are mutually exclusive")
    mode = "expert" if args.train_only_expert else ("router" if args.train_only_router else "joint")

    loaded_slots = None
    loaded_base_spec = None
    if args.load_experts:
        loaded_slots, loaded_base_spec = read_experts_arg(args.load_experts)
        # Override construction-affecting args with the spec (guarantees identical layout).
        args.base = loaded_base_spec["base"]
        args.tiny_random = loaded_base_spec["tiny_random"]
        args.every = loaded_base_spec.get("every")
        args.at = loaded_base_spec.get("at")
        args.capability_init = loaded_base_spec["capability_init"]
        args.router_identity_bias = loaded_base_spec["router_identity_bias"]
        args.capabilities = ",".join(loaded_base_spec["capabilities"])
        print(f"[idm] --load-experts: {sorted(loaded_slots)} slots; base_spec drives construction")

    caps = [c.strip() for c in args.capabilities.split(",") if c.strip()]
    # Canonical roster -> stable slot ordering (general/web -> 0, others -> 1..K).
    full_label_map = _assign_routing_labels(caps)
    roster_K = len({v for v in full_label_map.values() if v >= 1})

    print(f"[idm] config={args.config!r} device={device} tiny_random={args.tiny_random} mode={mode}")
    print(f"[idm] capabilities(roster)={caps}  full_label_map={full_label_map}")

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

    # K (capability experts) = roster K, or spec/arg override. All modes build the
    # SAME K slots so per-expert checkpoints slot back losslessly.
    if loaded_base_spec is not None:
        K = loaded_base_spec["n_capability_experts"]
    elif args.n_capability_experts is not None:
        K = args.n_capability_experts
    else:
        K = roster_K

    # ------------------------------------------------------------------
    # 2. Routed corpus. BRANCH (expert) mode loads ONLY the target capability's
    #    data but keeps its canonical slot (from the full roster); JOINT/COMBINE
    #    load the full multi-capability corpus. label_override keeps labels==slots.
    # ------------------------------------------------------------------
    target_slot = None
    if mode == "expert":
        target_cap = args.train_only_expert
        if target_cap not in full_label_map:
            raise SystemExit(f"--train-only-expert {target_cap!r} not in roster {caps}")
        target_slot = full_label_map[target_cap]
        if target_slot < 1:
            raise SystemExit(
                f"{target_cap!r} maps to label {target_slot} (the FROZEN skip expert); "
                "only capability experts (label>=1) are independently trainable")
        if target_slot > K:
            raise SystemExit(f"target slot {target_slot} exceeds K={K}")
        corpus_caps = [target_cap]
        override = {target_cap: target_slot}
        print(f"[idm] BRANCH: training ONLY expert {target_cap!r} -> slot {target_slot} "
              f"(all else frozen); loading only {target_cap!r} data")
    else:
        corpus_caps = caps
        override = full_label_map

    sequences, labels, label_map = build_routed_corpus(
        tokenizer=tokenizer,
        capabilities=corpus_caps,
        n_seqs_per_cap=args.n_seqs_per_cap,
        seq_len=args.seq_len,
        tiny_random=args.tiny_random,
        vocab_size=vocab_size,
        balance=args.balance,
        balance_target=args.balance_target,
        tiny_distinct=args.tiny_distinct,
        label_override=override,
    )
    n_cap_experts = len({v for v in full_label_map.values() if v >= 1})
    if K < n_cap_experts:
        print(f"[idm] WARNING: K={K} < #capability-experts={n_cap_experts}; "
              f"labels > K will be unroutable.")
    print(f"[idm] label_map={label_map}  K(capability experts)={K}  "
          f"n_experts={K + 1} (incl. skip=label 0)")
    print(f"[idm] corpus: {len(sequences)} sequences x {args.seq_len} tokens  "
          f"per-label counts={dict(sorted(Counter(labels).items()))}")

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
    n_layers_before = base_model.config.num_hidden_layers
    positions = _resolve_positions(n_layers_before, args.every, args.at)
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

    # Reproducible recipe for rebuilding this exact MoTE elsewhere (per-expert ckpt,
    # merge tool, combine step). Slot-compatibility is checked against it.
    base_spec = make_base_spec(
        base=args.base, tiny_random=args.tiny_random, every=args.every, at=args.at,
        capabilities=caps, label_map=full_label_map, n_capability_experts=K,
        capability_init=args.capability_init, router_identity_bias=args.router_identity_bias,
        inserted_indices=inserted_indices, n_layers_before=n_layers_before,
        n_layers_after=info["final_layers"], hidden_size=model.config.hidden_size,
        vocab_size=vocab_size, seq_len=args.seq_len, top_k=1,
    )
    if loaded_base_spec is not None:
        ok, diffs = specs_compatible(base_spec, loaded_base_spec)
        if not ok:
            raise RuntimeError(f"rebuilt model is NOT slot-compatible with --load-experts: {diffs}")

    # ------------------------------------------------------------------
    # 5b. Slot in independently-trained experts (COMBINE / --load-experts).
    # ------------------------------------------------------------------
    if loaded_slots:
        for slot, payload in sorted(loaded_slots.items()):
            load_expert_slot_weights(model, slot, payload["weights"])
            # Lossless check: the just-loaded slot must equal the source payload.
            back = expert_slot_weights(model, slot)
            for k in payload["weights"]:
                if not torch.equal(back[k], payload["weights"][k]):
                    raise RuntimeError(f"slot {slot} load not lossless at {k}")
        print(f"[idm] slotted {len(loaded_slots)} independently-trained experts "
              f"(lossless verified): slots {sorted(loaded_slots)}")

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
        # Only valid at TRUE init (all experts output 0). Once independently-trained
        # experts are slotted in (--load-experts / COMBINE), capability experts have
        # non-zero down_proj, so routing to them legitimately changes the output — the
        # random-router check no longer applies. The plain check above (default router
        # routes every token to the skip expert) still validates no-regression at start.
        if args.capability_init == "zero" and not loaded_slots:
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
    # 7. Freeze + optimizer (mode-specific)
    # ------------------------------------------------------------------
    if mode == "expert":
        _freeze_only_expert(model, target_slot, train_router=args.train_expert_router_slice)
        # Force ALL tokens through the target slot so the expert trains as a dense FFN
        # over its own data (the untrained router would otherwise gate them to skip).
        for block in _iter_mote_blocks(model):
            block.force_slot = target_slot
        print(f"[idm] FREEZE: base + all experts + {'router kept' if args.train_expert_router_slice else 'router'} "
              f"frozen; training ONLY capability expert slot {target_slot} (force_slot={target_slot})")
    elif mode == "router":
        _freeze_only_router(model)
        print("[idm] FREEZE: ALL experts (skip + capability) frozen; training ONLY routers")
    else:
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
    n_experts_total = K + 1
    target_fracs = _data_label_fracs(labels, n_experts_total)
    starved_labels: set = set()  # updated by the monitor; drives --route-force-weight
    last_disp = target_fracs.clone()
    if args.route_monitor:
        print(f"[idm] route-monitor ON (threshold {args.route_imbalance_threshold}x); "
              f"target dispatch fracs={[round(x, 3) for x in target_fracs.tolist()]}")
    if args.route_force_weight > 0:
        print(f"[idm] route-force ON: starved-label CE x{1 + args.route_force_weight:.2f}")
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
        label_weights = None
        if args.route_force_weight > 0 and starved_labels:
            mult = 1.0 + args.route_force_weight
            label_weights = {l: mult for l in starved_labels}
        route_loss = _supervised_route_loss(model, batch_labels, device, label_weights)
        # BRANCH (expert) mode trains the expert on its data with pure LM loss — the
        # router is frozen and (re)trained jointly in the COMBINE step, so the routing
        # objective is excluded here to keep the expert slot-clean. route_loss is still
        # computed for logging.
        route_coeff = 0.0 if mode == "expert" else args.route_weight
        loss = lm_loss + route_coeff * route_loss

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

        # Routing-imbalance monitor (rolling window) — log-only by default; also
        # refreshes the STARVED set that --route-force-weight acts on.
        if args.route_monitor and recent_counts and (step <= 5 or step % 20 == 0):
            last_disp = _dispatch_fracs(recent_counts, n_experts_total)
            warns, starved_labels, _overused = _route_monitor(
                last_disp, target_fracs, args.route_imbalance_threshold
            )
            print(f"  [monitor] dispatch={[round(x, 3) for x in last_disp.tolist()]}  "
                  f"target={[round(x, 3) for x in target_fracs.tolist()]}", flush=True)
            for w in warns:
                print(f"  [monitor] WARNING route imbalance: {w}", flush=True)
            if args.route_force_weight > 0 and starved_labels:
                print(f"  [monitor] forcing CE x{1 + args.route_force_weight:.2f} for "
                      f"starved labels {sorted(starved_labels)}", flush=True)

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
    # 9. Save adapter (trainable tensors only) + config + metrics.
    #    Plus a mode-specific expert-parallel artifact:
    #      expert  -> per-expert checkpoint (that slot's weights + base_spec)
    #      router  -> combined checkpoint (all slots + trained routers + base_spec)
    # ------------------------------------------------------------------
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    adapter = {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names}
    torch.save(adapter, os.path.join(args.out, "adapter_weights.pt"))

    expert_ckpt_path = None
    combined_ckpt_path = None
    if mode == "expert":
        expert_ckpt_path = os.path.join(args.out, f"expert_{args.train_only_expert}.pt")
        save_expert_checkpoint(
            expert_ckpt_path,
            capability=args.train_only_expert,
            slot=target_slot,
            base_spec=base_spec,
            weights=expert_slot_weights(model, target_slot),
            metrics={"lm_loss": final_lm, "route_loss": final_route, "steps": step},
        )
        print(f"[idm] BRANCH output -> {expert_ckpt_path} (slot {target_slot} weights + base_spec)")
    elif mode == "router":
        # Gather every capability slot (1..K) so the combined file rebuilds the whole
        # merged, router-retrained MoTE. Slots carry the loaded experts' identity.
        combined_slots = {}
        for s in range(1, K + 1):
            payload = loaded_slots.get(s) if loaded_slots else None
            combined_slots[s] = {
                "capability": (payload or {}).get("capability"),
                "label": s,
                "weights": expert_slot_weights(model, s),
            }
        combined_ckpt_path = os.path.join(args.out, "mote_combined.pt")
        save_combined_checkpoint(
            combined_ckpt_path,
            base_spec=base_spec,
            slots=combined_slots,
            router=router_slot_weights(model),
        )
        print(f"[idm] COMBINE output -> {combined_ckpt_path} (all {K} expert slots + routers + base_spec)")

    histo = torch.stack(recent_counts).sum(0).tolist() if recent_counts else []
    experts_used = sum(1 for c in histo if c > 0)

    cfg = {
        "config": args.config,
        "mode": mode,
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
        "balance": args.balance,
        "balance_target": args.balance_target,
        "route_force_weight": args.route_force_weight,
        "tiny_random": args.tiny_random,
    }
    with open(os.path.join(args.out, "identity_mote_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(args.out, "base_spec.json"), "w", encoding="utf-8") as fh:
        json.dump(base_spec, fh, indent=2)

    metrics = {
        "lm_loss": final_lm,
        "route_loss": final_route,
        "expert_counts": histo,
        "experts_used": experts_used,
        "steps": step,
        "tokens": tokens_seen,
        "dispatch_fracs": [round(x, 4) for x in last_disp.tolist()],
        "target_fracs": [round(x, 4) for x in target_fracs.tolist()],
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
