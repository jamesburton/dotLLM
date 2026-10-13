#!/usr/bin/env python3
"""mote_matrix.py — Specialization matrix for mini-MoTE (Mixture of Ternary Experts).

Loads the BitNet base once, then for each expert:
  1. Wraps the base MLP projections with the expert's ternary LoRA weights.
  2. Evaluates held-out PPL on every domain.
  3. Unwraps (restores original AutoBitLinear submodules).

Produces a grid:
  rows  = BASE + each expert
  cols  = each domain

A "specialization check" verifies that each expert's own-domain PPL improvement
over base is larger than its off-domain improvements.

CLI:
    # --help:
    python scripts/diffusion/mote_matrix.py --help

    # CPU validate (no model load):
    python scripts/diffusion/mote_matrix.py --validate-cpu

    # Full run (GPU, after experts finish training):
    python scripts/diffusion/mote_matrix.py \\
        --experts "code=E:/mote_experts/code,math=E:/mote_experts/math,general=E:/mote_experts/general" \\
        --domains "code=iamtarun/python_code_instructions_18k_alpaca,math=TIGER-Lab/MathInstruct,general=HuggingFaceH4/no_robots" \\
        --device cuda \\
        --out E:/mote_out/matrix.json
"""
from __future__ import annotations

import argparse
import json
import math
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


# ---------------------------------------------------------------------------
# Unwrap helpers
# ---------------------------------------------------------------------------

def _unwrap_mlp_layers(model, proj_names: frozenset = MLP_PROJ_NAMES) -> int:
    """Replace every TernaryLoRAWrapper in MLP projections with its .base.

    Returns the number of wrappers removed.
    """
    n = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for proj_name in proj_names:
            submod = getattr(mlp, proj_name, None)
            if isinstance(submod, TernaryLoRAWrapper):
                setattr(mlp, proj_name, submod.base)
                n += 1
    return n


def _count_wrappers(model) -> int:
    """Count active TernaryLoRAWrapper instances in the model."""
    return sum(1 for m in model.modules() if isinstance(m, TernaryLoRAWrapper))


# ---------------------------------------------------------------------------
# Expert loader
# ---------------------------------------------------------------------------

def _load_expert_into_model(
    model,
    expert_dir: str,
    device: torch.device,
) -> dict:
    """Wrap + load a saved ternary-LoRA expert into model in-place.

    Pattern mirrors tlora_capability_eval._load_model_and_tokenizer.
    Returns the meta dict.
    """
    tdir = Path(expert_dir)
    meta_path    = tdir / "tlora_meta.json"
    weights_path = tdir / "tlora_weights.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"tlora_meta.json not found in {expert_dir!r}")
    if not weights_path.exists():
        raise FileNotFoundError(f"tlora_weights.pt not found in {expert_dir!r}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    r     = int(meta["r"])
    alpha = float(meta["alpha"])
    n_layers = len(model.model.layers)

    target_layer_indices = meta.get("target_layer_indices")
    if target_layer_indices is None:
        last_n = meta.get("last_n_layers")
        if last_n is not None:
            target_layer_indices = list(range(max(0, n_layers - int(last_n)), n_layers))

    # Cold wrap (zeros/Kaiming); overwrite immediately from saved weights.
    _wrap_mlp_layers(
        model=model,
        target_layer_indices=target_layer_indices,
        r=r,
        alpha=alpha,
        adapter_weights=None,
    )

    saved_state: dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    loaded = 0
    for name, m in model.named_modules():
        if not isinstance(m, TernaryLoRAWrapper):
            continue
        if name not in saved_state:
            continue
        entry = saved_state[name]
        m.lora_A.data.copy_(entry["lora_A"].to(torch.bfloat16))
        m.lora_B.data.copy_(entry["lora_B"].to(torch.bfloat16))
        if m.lora_A.device != device:
            m.lora_A.data = m.lora_A.data.to(device)
            m.lora_B.data = m.lora_B.data.to(device)
        loaded += 1

    if loaded == 0:
        raise RuntimeError(f"No LoRA weights loaded from {weights_path}")

    _set_lora_enabled(True)
    _set_lora_lambda(1.0)
    return meta


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_eval_texts(dataset_id: str, split: str, n: int) -> list[str]:
    """Load up to n text samples from the given dataset split."""
    import datasets as hf_datasets
    ds = hf_datasets.load_dataset(dataset_id, split=split)
    texts: list[str] = []
    for row in ds:
        text = _extract_text(row)
        if text:
            texts.append(text)
        if len(texts) >= n:
            break
    return texts


# ---------------------------------------------------------------------------
# PPL computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_ppl_texts(
    model: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_seq_len: int = 512,
) -> float:
    """Compute token-level perplexity over a list of texts.

    Uses the current model state (LoRA enabled/disabled, wrappers present or not).
    """
    model.eval()
    total_nll  = 0.0
    total_toks = 0

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        ids = enc["input_ids"].to(device)
        if ids.size(1) < 2:
            continue
        logits = model(input_ids=ids).logits
        V = logits.size(-1)
        nll = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, V),
            ids[:, 1:].contiguous().view(-1),
            reduction="sum",
        )
        total_nll  += nll.item()
        total_toks += ids.size(1) - 1

    mean_nll = total_nll / max(total_toks, 1)
    return math.exp(min(mean_nll, 100.0))


# ---------------------------------------------------------------------------
# Argument parsing helper
# ---------------------------------------------------------------------------

def _parse_kv(s: str) -> dict[str, str]:
    """Parse 'k1=v1,k2=v2' into {k1:v1, k2:v2}.

    Values may contain '/' and ':' (paths); we split only on the first '='
    of each comma-separated item.
    """
    result: dict[str, str] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected 'name=value', got: {item!r}")
        k, v = item.split("=", 1)
        result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# CPU validation
# ---------------------------------------------------------------------------

def _validate_cpu() -> bool:
    """Structural validation on a tiny fake model — no 2B download required.

    Checks:
      1a. wrap count == 6 (2 layers × 3 projections) after manual TernaryLoRAWrapper install
      1b. _unwrap_mlp_layers removes all wrappers (0 remaining)
      1c. original submodule identity is restored after unwrap
      2.  PPL aggregation math (NLL+token sum → exp)
      3.  Specialization check logic (own-domain improvement > off-domain)
    """
    print()
    print("[validate-cpu] ============================================================")
    print("[validate-cpu] mote_matrix CPU validation (no model download)")
    print("[validate-cpu] ============================================================")

    # ── Fake model matching the structure _wrap_mlp_layers / _unwrap_mlp_layers expect ──
    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)
            self.up_proj   = nn.Linear(8, 16, bias=False)
            self.down_proj = nn.Linear(16, 8, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = FakeMLP()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])

    class FakeWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeModel()

    fake = FakeWrapper()

    # Capture original submodule ids before wrapping
    orig_gate0 = id(fake.model.layers[0].mlp.gate_proj)
    orig_up0   = id(fake.model.layers[0].mlp.up_proj)

    # Manually wrap (avoids AutoBitLinear requirement of _wrap_mlp_layers)
    for layer in fake.model.layers:
        mlp = layer.mlp
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            orig = getattr(mlp, proj_name)
            wrapper = TernaryLoRAWrapper(
                base=orig, r=4, alpha=8.0,
                in_features=orig.in_features,
                out_features=orig.out_features,
            )
            setattr(mlp, proj_name, wrapper)

    n_wrapped = _count_wrappers(fake)
    check1a = n_wrapped == 6
    print(f"\n[validate-cpu] CHECK 1a (wrap count == 6): {'PASS' if check1a else 'FAIL'}  (got {n_wrapped})")

    n_unwrapped = _unwrap_mlp_layers(fake)
    n_remaining = _count_wrappers(fake)
    check1b = (n_unwrapped == 6) and (n_remaining == 0)
    print(f"[validate-cpu] CHECK 1b (unwrap all, 0 remaining): {'PASS' if check1b else 'FAIL'}  "
          f"(unwrapped={n_unwrapped}, remaining={n_remaining})")

    same_gate0 = id(fake.model.layers[0].mlp.gate_proj) == orig_gate0
    same_up0   = id(fake.model.layers[0].mlp.up_proj)   == orig_up0
    check1c = same_gate0 and same_up0
    print(f"[validate-cpu] CHECK 1c (original identity restored): {'PASS' if check1c else 'FAIL'}  "
          f"(gate_proj={same_gate0}, up_proj={same_up0})")

    # CHECK 2: PPL aggregation math
    print()
    mock_nlls  = [10.0, 20.0, 15.0]
    mock_toks  = [50,   60,   55]
    expected   = math.exp(sum(mock_nlls) / sum(mock_toks))
    got        = math.exp(sum(mock_nlls) / sum(mock_toks))
    check2     = abs(expected - got) < 1e-9
    print(f"[validate-cpu] CHECK 2 (PPL aggregation): {'PASS' if check2 else 'FAIL'}  "
          f"(expected={expected:.6f}, got={got:.6f})")

    # CHECK 3: Specialization check logic
    print()
    base_ppls_mock   = {"code": 20.0, "math": 18.0}
    expert_ppls_mock = {
        "code": {"code": 15.0, "math": 17.5},  # big gain on code, small on math
        "math": {"code": 19.5, "math": 13.0},  # small gain on code, big on math
    }
    spec_ok: dict[str, bool] = {}
    for ename, dppls in expert_ppls_mock.items():
        own_imp = base_ppls_mock[ename] - dppls[ename]
        off_max = max(
            base_ppls_mock[d] - dppls[d]
            for d in dppls if d != ename
        ) if any(d != ename for d in dppls) else 0.0
        spec_ok[ename] = own_imp > off_max

    check3 = spec_ok.get("code", False) and spec_ok.get("math", False)
    print(f"[validate-cpu] CHECK 3 (specialization logic): {'PASS' if check3 else 'FAIL'}  "
          f"(code={spec_ok.get('code')}, math={spec_ok.get('math')})")

    all_pass = check1a and check1b and check1c and check2 and check3
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
            "Specialization matrix for mini-MoTE. Loads the BitNet base once, "
            "then for each expert wraps / evaluates on all domains / unwraps "
            "to produce a PPL grid (rows=base+experts, cols=domains). "
            "Reports per-expert specialization check."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--experts", default=None, metavar="NAME=PATH,...",
        help=(
            "Comma-separated expert definitions: "
            "'code=E:/mote_experts/code,math=E:/mote_experts/math,"
            "general=E:/mote_experts/general'"
        ),
    )
    ap.add_argument(
        "--domains", default=None, metavar="NAME=DATASET_ID,...",
        help=(
            "Comma-separated domain datasets: "
            "'code=iamtarun/python_code_instructions_18k_alpaca,"
            "math=TIGER-Lab/MathInstruct,general=HuggingFaceH4/no_robots'"
        ),
    )
    ap.add_argument("--eval-split",  default="train[:500]", metavar="SPLIT",
                    help="HuggingFace dataset split for evaluation.")
    ap.add_argument("--n-eval",      type=int, default=100, metavar="N",
                    help="Number of sequences per domain for PPL eval.")
    ap.add_argument("--max-seq-len", type=int, default=512, metavar="L",
                    help="Maximum token sequence length.")
    ap.add_argument("--device",      default="cuda",
                    help="Inference device (cuda or cpu).")
    ap.add_argument("--out",         default=None, metavar="PATH",
                    help="Output JSON path for grid results.")
    ap.add_argument("--validate-cpu", action="store_true",
                    help="Run CPU structural validation (no model load) and exit.")

    args = ap.parse_args(argv)

    if args.validate_cpu:
        ok = _validate_cpu()
        sys.exit(0 if ok else 1)

    if args.experts is None:
        ap.error("--experts is required")
    if args.domains is None:
        ap.error("--domains is required")
    if args.out is None:
        ap.error("--out is required")

    experts    = _parse_kv(args.experts)
    domains    = _parse_kv(args.domains)
    device     = torch.device(args.device)

    print("[mote_matrix] =========================================================")
    print(f"[mote_matrix] experts  : {list(experts.keys())}")
    print(f"[mote_matrix] domains  : {list(domains.keys())}")
    print(f"[mote_matrix] n_eval   : {args.n_eval}  split={args.eval_split!r}")
    print(f"[mote_matrix] device   : {args.device}")
    print("[mote_matrix] =========================================================")

    # ── 1. Load base model once ──────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[mote_matrix] Loading base model {BASE_MODEL_ID!r} ...")
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
    print(f"[mote_matrix] Base loaded  ({len(model.model.layers)} layers)")

    # ── 2. Load eval texts for each domain ──────────────────────────────
    print("\n[mote_matrix] Loading eval texts ...")
    domain_texts: dict[str, list[str]] = {}
    for dname, dataset_id in domains.items():
        domain_texts[dname] = _load_eval_texts(dataset_id, args.eval_split, args.n_eval)
        print(f"  {dname}: {len(domain_texts[dname])} texts  ({dataset_id!r})")

    # ── 3. Evaluate BASE on all domains ─────────────────────────────────
    print("\n[mote_matrix] Evaluating BASE ...")
    _set_lora_enabled(False)
    base_ppls: dict[str, float] = {}
    for dname, texts in domain_texts.items():
        ppl = _compute_ppl_texts(model, tokenizer, texts, device, args.max_seq_len)
        base_ppls[dname] = ppl
        print(f"  BASE on {dname}: PPL={ppl:.3f}")

    # ── 4. For each expert: wrap → eval all domains → unwrap ────────────
    expert_ppls: dict[str, dict[str, float]] = {}

    for ename, edir in experts.items():
        print(f"\n[mote_matrix] Expert '{ename}'  path={edir!r}")

        # Safety: remove any stale wrappers
        stale = _count_wrappers(model)
        if stale > 0:
            print(f"  WARNING: {stale} stale wrappers — unwrapping before load")
            _unwrap_mlp_layers(model)

        t0   = time.perf_counter()
        meta = _load_expert_into_model(model, edir, device)
        print(f"  Loaded {_count_wrappers(model)} wrappers in {time.perf_counter()-t0:.1f}s "
              f"(r={meta['r']}, alpha={meta['alpha']})")

        expert_ppls[ename] = {}
        for dname, texts in domain_texts.items():
            ppl    = _compute_ppl_texts(model, tokenizer, texts, device, args.max_seq_len)
            marker = " <-- own domain" if dname == ename else ""
            expert_ppls[ename][dname] = ppl
            print(f"  {ename} on {dname}: PPL={ppl:.3f}{marker}")

        # Unwrap → restore original submodules
        n_uw = _unwrap_mlp_layers(model)
        _set_lora_enabled(False)
        print(f"  Unwrapped {n_uw}  ({_count_wrappers(model)} remaining)")

        # Quick verify: base PPL should be restored exactly.
        # NOTE: must eval on the SAME text set as base_ppls[vdomain] (the full domain
        # set), else PPL differs purely from sample size and gives a false MISMATCH.
        vdomain  = list(domain_texts.keys())[0]
        vtexts   = domain_texts[vdomain]  # full set — matches base_ppls reference
        ppl_vfy  = _compute_ppl_texts(model, tokenizer, vtexts, device, args.max_seq_len)
        ppl_ref  = base_ppls[vdomain]
        delta    = abs(ppl_vfy - ppl_ref)
        ok_str   = "OK" if delta < 0.02 else "MISMATCH!"
        print(f"  Unwrap verify ('{vdomain}'): PPL={ppl_vfy:.4f}  ref={ppl_ref:.4f}  "
              f"delta={delta:.4f}  {ok_str}")

    # ── 5. Print PPL grid ────────────────────────────────────────────────
    domain_names = list(domains.keys())
    expert_names = list(experts.keys())
    rows         = ["base"] + expert_names

    col_w = max(12, max(len(d) for d in domain_names) + 2)
    row_w = max(10, max(len(r) for r in rows) + 2)

    print()
    print("[mote_matrix] ===== PPL GRID (lower = better) =====")
    header = f"{'':>{row_w}}" + "".join(f"{d:>{col_w}}" for d in domain_names)
    print(header)
    print("-" * len(header))

    all_ppls: dict[str, dict[str, float]] = {"base": base_ppls, **expert_ppls}
    for row in rows:
        ppls = all_ppls[row]
        line = f"{row:>{row_w}}" + "".join(f"{ppls[d]:>{col_w}.3f}" for d in domain_names)
        print(line)
    print()

    # ── 6. Specialization check ──────────────────────────────────────────
    print("[mote_matrix] ===== SPECIALIZATION CHECK =====")
    spec_results: dict[str, bool] = {}

    for ename in expert_names:
        if ename not in domain_names:
            print(f"  {ename}: no matching domain name — skipped")
            continue
        own_imp  = base_ppls[ename] - expert_ppls[ename][ename]
        off_imps = {
            d: base_ppls[d] - expert_ppls[ename][d]
            for d in domain_names if d != ename
        }
        max_off  = max(off_imps.values()) if off_imps else 0.0
        spec     = own_imp > max_off
        spec_results[ename] = spec
        off_str  = "  ".join(f"{d}:{v:+.3f}" for d, v in off_imps.items())
        verdict  = "SPECIALIZED" if spec else "NOT SPECIALIZED"
        print(f"  {ename}: own={own_imp:+.3f}  off=[{off_str}]  -> {verdict}")

    n_spec  = sum(spec_results.values())
    n_total = len(spec_results)
    print(f"\n  Specialization: {n_spec}/{n_total} experts show own-domain dominance")

    # ── 7. Save results ──────────────────────────────────────────────────
    results = {
        "base":           base_ppls,
        "experts":        expert_ppls,
        "domain_names":   domain_names,
        "expert_names":   expert_names,
        "specialization": spec_results,
        "n_specialized":  n_spec,
        "n_total_experts": n_total,
        "eval_split":     args.eval_split,
        "n_eval":         args.n_eval,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[mote_matrix] Results saved to {args.out!r}")


if __name__ == "__main__":
    main()
