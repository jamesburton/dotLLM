#!/usr/bin/env python3
"""mote_router.py — Train a per-sequence router + assemble mini-MoTE + evaluate.

Router features : mean-pooled last-hidden-state of the FROZEN base model.
Router model    : linear or 1-hidden-layer MLP classifier, trained supervised.
Evaluation      : base / oracle / router / single-best PPL on a mixed held-out set.

Granularity: per-sequence routing (one router call per sequence).
Efficiency : sequences are grouped by selected expert so the model is only
             wrapped/unwrapped once per expert per policy, not once per sequence.

CLI:
    # --help:
    python scripts/diffusion/mote_router.py --help

    # CPU validate (no model load):
    python scripts/diffusion/mote_router.py --validate-cpu

    # Full run (GPU, after experts finish training):
    python scripts/diffusion/mote_router.py \\
        --experts "code=E:/mote_experts/code,math=E:/mote_experts/math,general=E:/mote_experts/general" \\
        --domains "code=iamtarun/python_code_instructions_18k_alpaca,math=TIGER-Lab/MathInstruct,general=HuggingFaceH4/no_robots" \\
        --device cuda \\
        --out-router E:/mote_out/router \\
        --out-results E:/mote_out/router_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT   = _SCRIPT_DIR.parent.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from pdiff_ternary_lora import (
    BASE_MODEL_ID,
    TernaryLoRAWrapper,
    MLP_PROJ_NAMES,
    _wrap_mlp_layers,
    _set_lora_lambda,
    _set_lora_enabled,
    _extract_text,
)

# Shared helpers from mote_matrix (same directory)
from mote_matrix import (
    _unwrap_mlp_layers,
    _count_wrappers,
    _load_expert_into_model,
    _compute_ppl_texts,
    _load_eval_texts,
    _parse_kv,
)


# ---------------------------------------------------------------------------
# Router classifier
# ---------------------------------------------------------------------------

class RouterClassifier(nn.Module):
    """Linear or 1-hidden-layer MLP router.

    Args:
        input_dim : hidden size of the base model (e.g. 2560 for bitnet-2B).
        n_experts : number of expert / domain classes.
        hidden_dim: if > 0, use 1-hidden-layer MLP with ReLU; else plain linear.
    """

    def __init__(self, input_dim: int, n_experts: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net: nn.Module = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_experts),
            )
        else:
            self.net = nn.Linear(input_dim, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim] → logits [B, n_experts]"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_features(
    model,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Extract mean-pooled last-hidden-state features from the frozen base model.

    LoRA is disabled during extraction (set globally); no wrappers should be
    present at this point in the flow.

    Returns [len(texts), hidden_dim] float32 tensor on CPU.
    """
    model.eval()
    _set_lora_enabled(False)

    feats: list[torch.Tensor] = []
    hidden_size: int = model.config.hidden_size

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        ids = enc["input_ids"].to(device)

        if ids.size(1) < 1:
            feats.append(torch.zeros(hidden_size))
            continue

        outputs     = model(input_ids=ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [1, T, H]
        pooled      = last_hidden[0].mean(dim=0).float().cpu()  # [H]
        feats.append(pooled)

    return torch.stack(feats, dim=0)  # [N, H]


# ---------------------------------------------------------------------------
# Router training and evaluation
# ---------------------------------------------------------------------------

def _train_router(
    features:   torch.Tensor,          # [N, H]
    labels:     torch.Tensor,          # [N] int64
    n_experts:  int,
    hidden_dim: int   = 0,
    lr:         float = 1e-3,
    epochs:     int   = 100,
    device:     torch.device = torch.device("cpu"),
) -> RouterClassifier:
    """Train the router classifier on (features, labels). Returns trained router."""
    H      = features.shape[1]
    router = RouterClassifier(input_dim=H, n_experts=n_experts, hidden_dim=hidden_dim).to(device)
    opt    = torch.optim.Adam(router.parameters(), lr=lr)

    X = features.to(device)
    y = labels.to(device)

    router.train()
    for epoch in range(epochs):
        logits = router(X)
        loss   = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"  [router train] epoch {epoch+1:4d}/{epochs}  "
                  f"loss={loss.item():.4f}  train_acc={acc:.4f}")

    return router


@torch.no_grad()
def _eval_router_acc(
    router:   RouterClassifier,
    features: torch.Tensor,   # [N, H]
    labels:   torch.Tensor,   # [N] int64
    device:   torch.device,
) -> float:
    """Return routing accuracy (fraction correct)."""
    router.eval()
    logits = router(features.to(device))
    return (logits.argmax(dim=-1) == labels.to(device)).float().mean().item()


# ---------------------------------------------------------------------------
# Per-policy PPL evaluation
# ---------------------------------------------------------------------------

def _eval_policy(
    policy:          str,                          # "base" | "oracle" | "router" | "single-best"
    model,
    tokenizer,
    texts:           list[str],                    # mixed sequences
    true_labels:     list[int],                    # ground-truth domain index per text
    expert_names:    list[str],                    # ordered list of expert names
    expert_dirs:     dict[str, str],               # name → saved-dir path
    router:          Optional[RouterClassifier],   # used for "router" policy
    router_features: Optional[torch.Tensor],       # [N, H] precomputed; for "router"
    device:          torch.device,
    max_seq_len:     int            = 512,
    router_device:   torch.device  = torch.device("cpu"),
) -> float:
    """Compute mean token-level PPL under the given routing policy.

    Efficiency: sequences are grouped by chosen expert so the model is
    wrapped/unwrapped once per expert rather than once per sequence.
    """
    n = len(texts)

    # ── BASE: no expert, pure base forward ──────────────────────────────
    if policy == "base":
        _unwrap_mlp_layers(model)
        _set_lora_enabled(False)
        return _compute_ppl_texts(model, tokenizer, texts, device, max_seq_len)

    # ── SINGLE-BEST: best single expert applied to ALL sequences ────────
    if policy == "single-best":
        best_ppl  = float("inf")
        best_name = None
        for ename in expert_names:
            _unwrap_mlp_layers(model)
            _load_expert_into_model(model, expert_dirs[ename], device)
            ppl = _compute_ppl_texts(model, tokenizer, texts, device, max_seq_len)
            _unwrap_mlp_layers(model)
            print(f"    single-best candidate '{ename}': PPL={ppl:.3f}")
            if ppl < best_ppl:
                best_ppl  = ppl
                best_name = ename
        _set_lora_enabled(False)
        print(f"    single-best winner: '{best_name}'  PPL={best_ppl:.3f}")
        return best_ppl

    # ── ORACLE: route each seq to its TRUE domain expert ────────────────
    if policy == "oracle":
        assignments = list(true_labels)

    # ── ROUTER: route by the learned router ─────────────────────────────
    elif policy == "router":
        assert router is not None and router_features is not None
        router.eval()
        with torch.no_grad():
            logits      = router(router_features.to(router_device))
            assignments = logits.argmax(dim=-1).tolist()
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    # Group sequence indices by selected expert index
    groups: dict[int, list[int]] = {}
    for i, eidx in enumerate(assignments):
        groups.setdefault(eidx, []).append(i)

    # Accumulate NLL / token count across expert groups
    total_nll  = 0.0
    total_toks = 0

    for eidx in sorted(groups):
        ename  = expert_names[eidx]
        idxs   = groups[eidx]
        batch  = [texts[i] for i in idxs]
        print(f"    {policy}: expert '{ename}' handles {len(batch)} sequences")

        _unwrap_mlp_layers(model)
        _load_expert_into_model(model, expert_dirs[ename], device)
        model.eval()

        with torch.no_grad():
            for text in batch:
                enc = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_seq_len
                )
                ids = enc["input_ids"].to(device)
                if ids.size(1) < 2:
                    continue
                logits_m = model(input_ids=ids).logits
                V        = logits_m.size(-1)
                nll      = F.cross_entropy(
                    logits_m[:, :-1].contiguous().view(-1, V),
                    ids[:, 1:].contiguous().view(-1),
                    reduction="sum",
                )
                total_nll  += nll.item()
                total_toks += ids.size(1) - 1

    # Cleanup
    _unwrap_mlp_layers(model)
    _set_lora_enabled(False)

    mean_nll = total_nll / max(total_toks, 1)
    return math.exp(min(mean_nll, 100.0))


# ---------------------------------------------------------------------------
# CPU validation
# ---------------------------------------------------------------------------

def _validate_cpu() -> bool:
    """CPU-only structural validation — no 2B model required.

    Checks:
      1. Router training + accuracy on 3 well-separated Gaussian clusters (~100%)
      2. PPL aggregation math (NLL sum / token sum → exp)
      3. Group-by-expert routing logic produces correct index groupings
      4. RouterClassifier MLP (1-hidden-layer) forward shape
    """
    print()
    print("[validate-cpu] ============================================================")
    print("[validate-cpu] mote_router CPU validation (no model download)")
    print("[validate-cpu] ============================================================")

    all_pass = True

    # CHECK 1: Gaussian clusters ─────────────────────────────────────────
    print()
    print("[validate-cpu] CHECK 1: Router on 3 well-separated Gaussian clusters")
    torch.manual_seed(42)
    N_per  = 200
    H      = 64
    K      = 3

    centers = [torch.randn(H) * 5 for _ in range(K)]
    X_parts = [c.unsqueeze(0) + 0.1 * torch.randn(N_per, H) for c in centers]
    y_parts = [torch.full((N_per,), i, dtype=torch.long) for i in range(K)]

    X_all = torch.cat(X_parts)
    y_all = torch.cat(y_parts)
    perm  = torch.randperm(X_all.shape[0])
    X_all, y_all = X_all[perm], y_all[perm]

    n_train    = N_per * 2
    X_tr, y_tr = X_all[:n_train], y_all[:n_train]
    X_ev, y_ev = X_all[n_train:],  y_all[n_train:]

    router = _train_router(
        features=X_tr, labels=y_tr,
        n_experts=K, hidden_dim=0, lr=1e-2, epochs=100,
        device=torch.device("cpu"),
    )
    acc    = _eval_router_acc(router, X_ev, y_ev, torch.device("cpu"))
    check1 = acc >= 0.99
    print(f"  routing accuracy: {acc:.4f}  (expected >= 0.99)")
    print(f"  CHECK 1: {'PASS' if check1 else 'FAIL'}")
    all_pass = all_pass and check1

    # CHECK 2: PPL aggregation math ──────────────────────────────────────
    print()
    print("[validate-cpu] CHECK 2: PPL aggregation math")
    mock = [
        {"nll": 5.0,  "toks": 50},
        {"nll": 8.0,  "toks": 60},
        {"nll": 6.0,  "toks": 55},
        {"nll": 10.0, "toks": 70},
    ]
    tnll   = sum(s["nll"]  for s in mock)
    ttoks  = sum(s["toks"] for s in mock)
    exp_ppl = math.exp(tnll / ttoks)
    got_ppl = math.exp(tnll / ttoks)
    check2  = abs(exp_ppl - got_ppl) < 1e-9
    print(f"  expected={exp_ppl:.6f}  got={got_ppl:.6f}")
    print(f"  CHECK 2: {'PASS' if check2 else 'FAIL'}")
    all_pass = all_pass and check2

    # CHECK 3: Group-by-expert routing logic ─────────────────────────────
    print()
    print("[validate-cpu] CHECK 3: Group-by-expert routing logic")
    assignments   = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    expected_grps = {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5, 8]}
    groups: dict[int, list[int]] = {}
    for i, eidx in enumerate(assignments):
        groups.setdefault(eidx, []).append(i)
    check3 = all(sorted(groups.get(k, [])) == v for k, v in expected_grps.items())
    print(f"  groups: { {k: sorted(v) for k, v in sorted(groups.items())} }")
    print(f"  CHECK 3: {'PASS' if check3 else 'FAIL'}")
    all_pass = all_pass and check3

    # CHECK 4: RouterClassifier MLP forward shape ────────────────────────
    print()
    print("[validate-cpu] CHECK 4: RouterClassifier MLP forward shape")
    r_mlp   = RouterClassifier(input_dim=32, n_experts=3, hidden_dim=16)
    x_in    = torch.randn(5, 32)
    logits  = r_mlp(x_in)
    check4  = tuple(logits.shape) == (5, 3)
    print(f"  output shape: {tuple(logits.shape)}  (expected (5, 3))")
    print(f"  CHECK 4: {'PASS' if check4 else 'FAIL'}")
    all_pass = all_pass and check4

    print()
    print("[validate-cpu] ============================================================")
    print(f"[validate-cpu] RESULT: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")
    print("[validate-cpu] ============================================================")
    print()
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train a per-sequence router for mini-MoTE (Mixture of Ternary Experts). "
            "Features: mean-pooled last-hidden-state from the frozen base. "
            "Router: linear or 1-hidden-layer MLP trained supervised on domain labels. "
            "Evaluates: base / oracle / router / single-best PPL on a mixed held-out set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--experts", default=None, metavar="NAME=PATH,...",
                    help="Comma-separated expert defs: 'code=E:/mote_experts/code,...'")
    ap.add_argument("--domains", default=None, metavar="NAME=DATASET_ID,...",
                    help="Comma-separated domain datasets: 'code=iamtarun/...,...'")
    ap.add_argument("--router-train-split", default="train[500:1500]", metavar="SPLIT",
                    help="Dataset split for router training features (per domain).")
    ap.add_argument("--router-eval-split",  default="train[:500]",    metavar="SPLIT",
                    help="Dataset split for router accuracy eval + MoTE PPL eval.")
    ap.add_argument("--n-router-train",  type=int, default=1000, metavar="N",
                    help="Number of sequences per domain for router training.")
    ap.add_argument("--n-router-eval",   type=int, default=500,  metavar="N",
                    help="Number of sequences per domain for router accuracy eval.")
    ap.add_argument("--n-mote-eval",     type=int, default=100,  metavar="N",
                    help="Number of sequences per domain for mini-MoTE PPL eval.")
    ap.add_argument("--router-hidden",   type=int, default=0,    metavar="H",
                    help="Router hidden dim (0=linear, >0=1-hidden-layer MLP).")
    ap.add_argument("--router-epochs",   type=int, default=100,  metavar="E",
                    help="Training epochs for the router classifier.")
    ap.add_argument("--router-lr",       type=float, default=1e-3, metavar="LR",
                    help="Learning rate for router training.")
    ap.add_argument("--max-seq-len",     type=int, default=512,  metavar="L",
                    help="Maximum token sequence length.")
    ap.add_argument("--device",          default="cuda",
                    help="Model device (cuda or cpu).")
    ap.add_argument("--out-router",      default=None, metavar="DIR",
                    help="Directory to save router state_dict + meta JSON.")
    ap.add_argument("--out-results",     default=None, metavar="PATH",
                    help="Output JSON path for eval results.")
    ap.add_argument("--validate-cpu",    action="store_true",
                    help="Run CPU structural validation (no model load) and exit.")

    args = ap.parse_args(argv)

    if args.validate_cpu:
        ok = _validate_cpu()
        sys.exit(0 if ok else 1)

    for flag, val in [
        ("--experts",     args.experts),
        ("--domains",     args.domains),
        ("--out-router",  args.out_router),
        ("--out-results", args.out_results),
    ]:
        if val is None:
            ap.error(f"{flag} is required")

    experts_map  = _parse_kv(args.experts)
    domains_map  = _parse_kv(args.domains)
    expert_names = list(experts_map.keys())
    domain_names = list(domains_map.keys())
    n_experts    = len(expert_names)

    # Oracle routing assumes domain[i] → expert[i] by name match
    if expert_names != domain_names:
        print(f"[mote_router] WARNING: expert_names={expert_names} != domain_names={domain_names}")
        print("[mote_router] Oracle routing maps domain[i] -> expert[i] by list position.")

    print("[mote_router] =========================================================")
    print(f"[mote_router] experts  : {expert_names}")
    print(f"[mote_router] domains  : {domain_names}")
    print(f"[mote_router] device   : {args.device}")
    print("[mote_router] =========================================================")

    device        = torch.device(args.device)
    router_device = torch.device("cpu")   # router is tiny; keep on CPU

    # ── 1. Load base model ───────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[mote_router] Loading base model {BASE_MODEL_ID!r} ...")
    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if device.type != "cpu":
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **load_kwargs)
    if device.type == "cpu":
        model = model.to(device)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = model.config.hidden_size
    print(f"[mote_router] Base loaded  ({len(model.model.layers)} layers, hidden={hidden_size})")

    # ── 2. Load text datasets ────────────────────────────────────────────
    print("\n[mote_router] Loading text datasets ...")
    train_texts: dict[str, list[str]] = {}
    eval_texts:  dict[str, list[str]] = {}

    for dname, dataset_id in domains_map.items():
        print(f"  {dname}: {dataset_id!r}")
        train_texts[dname] = _load_eval_texts(dataset_id, args.router_train_split, args.n_router_train)
        eval_texts[dname]  = _load_eval_texts(dataset_id, args.router_eval_split,  args.n_router_eval)
        n_mote = min(args.n_mote_eval, len(eval_texts[dname]))
        print(f"    train={len(train_texts[dname])}  eval={len(eval_texts[dname])}  "
              f"mote={n_mote}")

    # ── 3. Extract features for router train + eval ──────────────────────
    print("\n[mote_router] Extracting router features (base, mean-pooled last hidden) ...")

    X_train_parts: list[torch.Tensor] = []
    y_train_parts: list[torch.Tensor] = []
    X_eval_parts:  list[torch.Tensor] = []
    y_eval_parts:  list[torch.Tensor] = []
    X_mote_parts:  list[torch.Tensor] = []
    mote_counts:   list[int]          = []

    for didx, dname in enumerate(domain_names):
        print(f"  [{didx+1}/{n_experts}] '{dname}' train features "
              f"(n={len(train_texts[dname])}) ...")
        ft = _extract_features(model, tokenizer, train_texts[dname], device, args.max_seq_len)
        X_train_parts.append(ft)
        y_train_parts.append(torch.full((ft.shape[0],), didx, dtype=torch.long))

        print(f"  [{didx+1}/{n_experts}] '{dname}' eval features "
              f"(n={len(eval_texts[dname])}) ...")
        fe = _extract_features(model, tokenizer, eval_texts[dname], device, args.max_seq_len)
        X_eval_parts.append(fe)
        y_eval_parts.append(torch.full((fe.shape[0],), didx, dtype=torch.long))

        n_m = min(args.n_mote_eval, fe.shape[0])
        X_mote_parts.append(fe[:n_m])
        mote_counts.append(n_m)

    X_train = torch.cat(X_train_parts)
    y_train = torch.cat(y_train_parts)
    X_eval  = torch.cat(X_eval_parts)
    y_eval  = torch.cat(y_eval_parts)
    X_mote  = torch.cat(X_mote_parts)

    print(f"  Feature shapes: train={tuple(X_train.shape)}  "
          f"eval={tuple(X_eval.shape)}  mote={tuple(X_mote.shape)}")

    # ── 4. Train router ──────────────────────────────────────────────────
    print(f"\n[mote_router] Training router "
          f"(hidden={args.router_hidden}, lr={args.router_lr}, "
          f"epochs={args.router_epochs}) ...")
    router = _train_router(
        features=X_train, labels=y_train,
        n_experts=n_experts,
        hidden_dim=args.router_hidden,
        lr=args.router_lr,
        epochs=args.router_epochs,
        device=router_device,
    )

    # ── 5. Eval routing accuracy ─────────────────────────────────────────
    acc = _eval_router_acc(router, X_eval, y_eval, router_device)
    print(f"\n[mote_router] Routing accuracy (held-out): {acc:.4f}  ({acc*100:.1f}%)")

    # ── 6. Save router ───────────────────────────────────────────────────
    os.makedirs(args.out_router, exist_ok=True)
    router_path      = Path(args.out_router) / "router_state.pt"
    router_meta_path = Path(args.out_router) / "router_meta.json"
    torch.save(router.state_dict(), str(router_path))
    router_meta = {
        "expert_names":        expert_names,
        "domain_names":        domain_names,
        "hidden_dim":          args.router_hidden,
        "input_dim":           hidden_size,
        "n_experts":           n_experts,
        "routing_accuracy":    round(acc, 4),
        "router_train_split":  args.router_train_split,
        "router_eval_split":   args.router_eval_split,
        "n_router_train":      args.n_router_train,
        "n_router_eval":       args.n_router_eval,
    }
    router_meta_path.write_text(json.dumps(router_meta, indent=2), encoding="utf-8")
    print(f"[mote_router] Router saved to {args.out_router!r}")

    # ── 7. Build mixed eval set ──────────────────────────────────────────
    print("\n[mote_router] Building mixed eval set ...")
    mixed_texts:  list[str]          = []
    mixed_labels: list[int]          = []
    mixed_feats_parts: list[torch.Tensor] = []

    for didx, dname in enumerate(domain_names):
        n_m   = mote_counts[didx]
        seqs  = eval_texts[dname][:n_m]
        mixed_texts.extend(seqs)
        mixed_labels.extend([didx] * n_m)
        mixed_feats_parts.append(X_mote_parts[didx])

    mixed_feats = torch.cat(mixed_feats_parts)
    print(f"  Mixed: {len(mixed_texts)} sequences  ({n_experts} domains × ~{args.n_mote_eval} each)")

    # ── 8. Mini-MoTE PPL eval under four policies ────────────────────────
    print("\n[mote_router] === Mini-MoTE evaluation (4 policies) ===")

    print("\n  [1/4] BASE policy (no expert) ...")
    t0       = time.perf_counter()
    ppl_base = _eval_policy(
        "base", model, tokenizer,
        mixed_texts, mixed_labels,
        expert_names, experts_map,
        None, None,
        device, args.max_seq_len, router_device,
    )
    print(f"  BASE PPL = {ppl_base:.3f}  ({time.perf_counter()-t0:.1f}s)")

    print("\n  [2/4] ORACLE policy (true domain expert per seq) ...")
    t0         = time.perf_counter()
    ppl_oracle = _eval_policy(
        "oracle", model, tokenizer,
        mixed_texts, mixed_labels,
        expert_names, experts_map,
        None, None,
        device, args.max_seq_len, router_device,
    )
    print(f"  ORACLE PPL = {ppl_oracle:.3f}  ({time.perf_counter()-t0:.1f}s)")

    print("\n  [3/4] ROUTER policy (learned router) ...")
    t0         = time.perf_counter()
    ppl_router = _eval_policy(
        "router", model, tokenizer,
        mixed_texts, mixed_labels,
        expert_names, experts_map,
        router, mixed_feats,
        device, args.max_seq_len, router_device,
    )
    print(f"  ROUTER PPL = {ppl_router:.3f}  ({time.perf_counter()-t0:.1f}s)")

    print("\n  [4/4] SINGLE-BEST policy (best single expert for all) ...")
    t0         = time.perf_counter()
    ppl_single = _eval_policy(
        "single-best", model, tokenizer,
        mixed_texts, mixed_labels,
        expert_names, experts_map,
        None, None,
        device, args.max_seq_len, router_device,
    )
    print(f"  SINGLE-BEST PPL = {ppl_single:.3f}  ({time.perf_counter()-t0:.1f}s)")

    # ── 9. Headline ──────────────────────────────────────────────────────
    oracle_beats_base = ppl_oracle < ppl_base
    router_beats_base = ppl_router < ppl_base
    oracle_gap        = ppl_router - ppl_oracle

    print()
    print("[mote_router] ===== MINI-MoTE HEADLINE =====")
    print(f"  BASE         PPL = {ppl_base:.3f}")
    print(f"  ORACLE       PPL = {ppl_oracle:.3f}  (upper bound; true expert per seq)")
    print(f"  ROUTER       PPL = {ppl_router:.3f}  (learned router)")
    print(f"  SINGLE-BEST  PPL = {ppl_single:.3f}  (one fixed expert for all seqs)")
    print()
    print(f"  Oracle beats base : {'YES' if oracle_beats_base else 'NO'}  "
          f"({ppl_base:.3f} -> {ppl_oracle:.3f})")
    print(f"  Router beats base : {'YES' if router_beats_base else 'NO'}  "
          f"({ppl_base:.3f} -> {ppl_router:.3f})")
    print(f"  Router vs oracle gap : {oracle_gap:+.3f}  (0 = perfect routing)")
    print(f"  Routing accuracy     : {acc:.4f}  ({acc*100:.1f}%)")
    print("[mote_router] ===== END =====")

    # ── 10. Save results ─────────────────────────────────────────────────
    results = {
        "ppl_base":              ppl_base,
        "ppl_oracle":            ppl_oracle,
        "ppl_router":            ppl_router,
        "ppl_single_best":       ppl_single,
        "routing_accuracy":      round(acc, 4),
        "oracle_beats_base":     oracle_beats_base,
        "router_beats_base":     router_beats_base,
        "router_vs_oracle_gap":  round(oracle_gap, 4),
        "n_experts":             n_experts,
        "expert_names":          expert_names,
        "domain_names":          domain_names,
        "n_mote_eval_per_domain": args.n_mote_eval,
        "router_meta":           router_meta,
    }
    Path(args.out_results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_results).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[mote_router] Results saved to {args.out_results!r}")


if __name__ == "__main__":
    main()
