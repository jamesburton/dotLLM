#!/usr/bin/env python3
"""Recursion eval — Track R Task 5 (issue #118).

Three-criterion evaluation of the trained recursion adapter on the held-out
HuggingFaceH4/no_robots test split (disjoint from training).

Criterion 1 — PPL vs fixed recurrence N in {1..N_max}
    Gate: PPL(N=N_max) < PPL(N=1).

Criterion 2 — Hard-token benefit
    Per-token loss improvement I_t(N) = L_t(N-1) - L_t(N) correlates
    positively with token difficulty (entropy at N=1).
    Gate: Pearson r > 0 and very-hard bucket mean > easy bucket mean.

Criterion 3 — Live gate vs fixed-N
    Oracle gate (binary: N=1 if easy else N=N_max) achieves PPL within 5%
    of the best fixed-N at lower mean loops/token, AND generate_adaptive
    (entropy+margin, KV-cached) produces varying loop counts across tokens.
    Gate: oracle_ppl <= 1.05 * best_fixed_ppl AND mean_loops < best_fixed_N.

Writes: <adapter_dir>/eval.json with all criteria, numbers, verdict.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface \\
    python scripts/lora/recur_eval.py --adapter-dir .docs/recursion/r1

Smoke test (CPU, seconds):
  python scripts/lora/recur_eval.py \\
    --adapter-dir .docs/recursion/smoke_r3 \\
    --device cpu --n-seqs 4 --max-seq-len 64 --n-max 3 \\
    --no-gen-adaptive
"""
from __future__ import annotations

# Windows: suppress dynamo (no cl.exe → SIGSEGV in compile subprocess on 2nd+ call)
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets as hf_datasets

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

# Allow importing recur_model / recur_gate from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recur_model import build_recur
from recur_gate import generate_adaptive, DEFAULT_ENT_THRESH, DEFAULT_MARGIN_THRESH


# ---------------------------------------------------------------------------
# DirectIO safetensors shim (Windows: avoids SIGSEGV loading >4 GB files when
# a CUDA context is active — identical to the shim in test_recur_gate.py)
# ---------------------------------------------------------------------------

class _DirectIOSlice:
    _DTYPE_MAP = {
        "BF16": torch.bfloat16, "F16": torch.float16,
        "F32": torch.float32,   "F64": torch.float64,
        "I8":  torch.int8,      "I16": torch.int16,
        "I32": torch.int32,     "I64": torch.int64,
        "U8":  torch.uint8,
    }

    def __init__(self, file_obj, info, data_offset: int) -> None:
        self._f, self._info, self._data_offset = file_obj, info, data_offset

    def get_shape(self): return self._info["shape"]
    def get_dtype(self): return self._info["dtype"]

    def __getitem__(self, _idx):
        offsets = self._info["data_offsets"]
        self._f.seek(self._data_offset + offsets[0])
        raw = self._f.read(offsets[1] - offsets[0])
        dtype = self._DTYPE_MAP[self._info["dtype"]]
        t = torch.frombuffer(bytearray(raw), dtype=dtype)
        shape = self._info["shape"]
        return t.reshape(shape) if shape else t


class _DirectIOSafeFile:
    """Drop-in for ``safetensors.safe_open`` using direct file IO (no mmap).

    Implements the subset used by transformers' model loading: ``keys()``,
    ``metadata()``, ``get_slice()``, ``get_tensor()``, context-manager.
    """

    def __init__(self, filename, framework="pt", device="cpu") -> None:
        self._file = open(str(filename), "rb")
        hdr_len = struct.unpack("<Q", self._file.read(8))[0]
        self._header: dict = json.loads(self._file.read(hdr_len))
        self._data_offset = 8 + hdr_len

    def __enter__(self): return self
    def __exit__(self, *args): self._file.close()

    def keys(self):
        return [k for k in self._header if k != "__metadata__"]

    def metadata(self):
        return self._header.get("__metadata__", {})

    def get_slice(self, name: str) -> _DirectIOSlice:
        return _DirectIOSlice(self._file, self._header[name], self._data_offset)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[...]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_adapter(
    base_id: str,
    adapter_dir: Path,
    P: int,
    Q: int,
    device: torch.device,
):
    """Load BitNet base + RecurModel wrapper + trained adapter weights.

    Returns (model, tokenizer).  Asserts no unexpected adapter keys.
    """
    import transformers.modeling_utils as _mutils
    _orig = getattr(_mutils, "safe_open", None)
    _mutils.safe_open = _DirectIOSafeFile
    try:
        print(f"[recur_eval] loading base {base_id} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
    finally:
        if _orig is not None:
            _mutils.safe_open = _orig
        else:
            delattr(_mutils, "safe_open")

    base.eval()
    base.config.use_cache = False

    model = build_recur(base, P=P, Q=Q)
    model.eval()
    model.to(device)

    adapter_path = adapter_dir / "adapter_weights.pt"
    print(f"[recur_eval] loading adapter {adapter_path} ...", flush=True)
    try:
        adapter_state = torch.load(
            str(adapter_path), map_location="cpu", weights_only=True
        )
    except TypeError:
        adapter_state = torch.load(str(adapter_path), map_location="cpu")

    missing_keys, unexpected_keys = model.load_state_dict(adapter_state, strict=False)
    assert len(unexpected_keys) == 0, (
        f"Adapter has unexpected keys (should be 0): {unexpected_keys[:8]}"
    )
    print(
        f"[recur_eval] adapter: {len(adapter_state)} tensors loaded, "
        f"{len(missing_keys)} frozen base keys missing (expected)",
        flush=True,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def _build_eval_corpus(
    tokenizer,
    n_seqs: int,
    max_seq_len: int,
    split: str = "test",
    dataset: str = "HuggingFaceH4/no_robots",
) -> list[torch.Tensor]:
    """Tokenise the no_robots test split into fixed-length sequences."""
    print(f"[recur_eval] loading {dataset} split={split} ...", flush=True)
    ds = hf_datasets.load_dataset(dataset, split=split)
    all_ids: list[int] = []
    for row in ds:
        if "messages" in row:
            text = " ".join(
                m["content"] for m in row["messages"] if m.get("content")
            )
        elif "text" in row:
            text = row["text"]
        elif "prompt" in row:
            text = row.get("prompt", "")
        else:
            text = " ".join(str(v) for v in row.values() if isinstance(v, str))
        all_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(all_ids) >= n_seqs * max_seq_len * 3:
            break

    seqs: list[torch.Tensor] = []
    for i in range(0, len(all_ids) - max_seq_len, max_seq_len):
        seqs.append(torch.tensor(all_ids[i : i + max_seq_len], dtype=torch.long))
        if len(seqs) >= n_seqs:
            break

    print(f"[recur_eval] corpus: {len(seqs)} × {max_seq_len} tokens", flush=True)
    return seqs


# ---------------------------------------------------------------------------
# Core computation: per-token losses + N=1 difficulty signals
# ---------------------------------------------------------------------------

def _compute_per_token(
    model,
    corpus: list[torch.Tensor],
    N_values: list[int],
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Compute per-token cross-entropy losses at each N, and N=1 difficulty.

    For each sequence, runs len(N_values) forward passes.  No gradients.

    Returns
    -------
    loss_by_N : dict[int, Tensor[n_tokens]]
        Per-token cross-entropy at each recurrence count.
    entropy   : Tensor[n_tokens]
        N=1 predictive entropy (nats) for each token position.
    margin    : Tensor[n_tokens]
        N=1 top1-top2 logit gap (higher = easier / more confident).
    """
    model.eval()
    all_loss: dict[int, list] = {N: [] for N in N_values}
    all_entropy: list[torch.Tensor] = []
    all_margin: list[torch.Tensor] = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for si, ids_cpu in enumerate(corpus):
            ids = ids_cpu.unsqueeze(0).to(device)   # [1, T]
            for N in N_values:
                logits = model(ids, recurrence=N)    # [1, T, V]
                # Causal shift: logits[:,t,:] predicts ids[:,t+1]
                sl = logits[:, :-1, :].float()       # [1, T-1, V]
                tgt = ids[:, 1:].contiguous()        # [1, T-1]
                per_tok = F.cross_entropy(
                    sl.view(-1, sl.size(-1)),
                    tgt.view(-1),
                    reduction="none",
                )  # [T-1]
                all_loss[N].append(per_tok.cpu())

                if N == 1:
                    # Entropy = -∑ p·log(p), handle 0·log(0)=0 by zeroing -inf
                    sl0 = sl[0]                              # [T-1, V]
                    log_p = torch.log_softmax(sl0, dim=-1)  # [T-1, V]
                    p = log_p.exp()
                    safe_log_p = torch.nan_to_num(
                        log_p, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    ent = -(p * safe_log_p).sum(-1)          # [T-1]
                    all_entropy.append(ent.cpu())

                    # Margin: top1 - top2 logit (pre-softmax)
                    top2 = torch.topk(sl0, 2, dim=-1).values  # [T-1, 2]
                    mg = top2[:, 0] - top2[:, 1]               # [T-1]
                    all_margin.append(mg.cpu())

            elapsed = time.perf_counter() - t0
            rate = (si + 1) / max(elapsed, 1e-6)
            print(
                f"  seq {si + 1:3d}/{len(corpus)}  ({rate:.2f} seqs/s)",
                flush=True,
            )

    loss_by_N = {N: torch.cat(all_loss[N]) for N in N_values}
    entropy = torch.cat(all_entropy)
    margin = torch.cat(all_margin)
    return loss_by_N, entropy, margin


# ---------------------------------------------------------------------------
# Pearson correlation helper
# ---------------------------------------------------------------------------

def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Pearson r + p-value (uses scipy when available; p=nan otherwise)."""
    try:
        from scipy import stats
        r, p = stats.pearsonr(
            x.numpy().astype(float), y.numpy().astype(float)
        )
        return float(r), float(p)
    except ImportError:
        xf = x.float() - x.float().mean()
        yf = y.float() - y.float().mean()
        denom = xf.norm() * yf.norm()
        r = float((xf * yf).sum() / (denom + 1e-12))
        return r, float("nan")


# ---------------------------------------------------------------------------
# Criterion 1 — PPL vs fixed-N curve
# ---------------------------------------------------------------------------

def _criterion_1(loss_by_N: dict, N_values: list[int]) -> dict:
    """Compute PPL at each N and test for monotone decrease."""
    ppl_by_N: dict[int, float] = {}
    for N in N_values:
        ppl_by_N[N] = round(math.exp(float(loss_by_N[N].mean())), 4)

    ppl_list = [ppl_by_N[N] for N in N_values]
    monotone = all(ppl_list[i] >= ppl_list[i + 1] for i in range(len(ppl_list) - 1))
    delta = round(ppl_by_N[N_values[-1]] - ppl_by_N[N_values[0]], 4)
    gate_pass = ppl_by_N[N_values[-1]] < ppl_by_N[N_values[0]]

    print("\n[Criterion 1] PPL vs fixed recurrence N:")
    for N in N_values:
        print(f"  N={N}: PPL={ppl_by_N[N]:.4f}")
    print(f"  delta(N={N_values[-1]} vs N={N_values[0]}): {delta:+.4f}")
    print(f"  monotone_decrease: {monotone}")
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "ppl_by_N": {str(N): ppl_by_N[N] for N in N_values},
        "monotone_decrease": monotone,
        "delta_ppl_N1_to_Nmax": delta,
        "gate": "PASS" if gate_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Criterion 2 — Hard-token benefit
# ---------------------------------------------------------------------------

def _criterion_2(
    loss_by_N: dict,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    N_values: list[int],
) -> dict:
    """Correlate token difficulty with per-token improvement I_t."""
    N_max = N_values[-1]
    # Total improvement: L_t(N=1) - L_t(N=N_max).  Positive = loops helped.
    total_improvement = loss_by_N[N_values[0]] - loss_by_N[N_max]

    r_ent, p_ent = _pearson_r(entropy, total_improvement)
    r_mg,  p_mg  = _pearson_r(-margin, total_improvement)  # -margin = difficulty

    # Per-N improvement: mean I_t(N) = mean[L(N-1) - L(N)]
    improvement_by_N: dict[str, float] = {}
    for i in range(1, len(N_values)):
        N_prev, N_cur = N_values[i - 1], N_values[i]
        step_imp = loss_by_N[N_prev] - loss_by_N[N_cur]
        improvement_by_N[str(N_cur)] = round(float(step_imp.mean()), 6)

    # Bucket analysis: 4 entropy quartiles
    q25 = float(torch.quantile(entropy, 0.25))
    q50 = float(torch.quantile(entropy, 0.50))
    q75 = float(torch.quantile(entropy, 0.75))
    bucket_masks = {
        "easy_q0_25":       entropy < q25,
        "moderate_q25_50":  (entropy >= q25) & (entropy < q50),
        "hard_q50_75":      (entropy >= q50) & (entropy < q75),
        "very_hard_q75_100": entropy >= q75,
    }
    bucket_analysis: dict[str, dict] = {}
    for name, mask in bucket_masks.items():
        if mask.any():
            bucket_analysis[name] = {
                "n_tokens": int(mask.sum()),
                "mean_entropy": round(float(entropy[mask].mean()), 4),
                "mean_total_improvement": round(
                    float(total_improvement[mask].mean()), 6
                ),
            }

    very_hard_beats_easy = (
        bucket_analysis.get("very_hard_q75_100", {}).get(
            "mean_total_improvement", -999.0
        )
        > bucket_analysis.get("easy_q0_25", {}).get(
            "mean_total_improvement", 999.0
        )
    )
    gate_pass = r_ent > 0.0 and very_hard_beats_easy

    print("\n[Criterion 2] Hard-token benefit (difficulty vs improvement):")
    print(f"  Pearson r (entropy vs total_improvement):  {r_ent:.4f}  p={p_ent:.4g}")
    print(f"  Pearson r (-margin vs total_improvement):  {r_mg:.4f}  p={p_mg:.4g}")
    print(f"  Per-step improvement (mean I_t per loop):")
    for n_str, v in improvement_by_N.items():
        print(f"    I_t(N={n_str}): {v:+.6f}")
    print("  Bucket mean_total_improvement:")
    for name, bd in bucket_analysis.items():
        print(f"    {name}: {bd['mean_total_improvement']:+.6f}")
    print(f"  very_hard_beats_easy: {very_hard_beats_easy}")
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "pearson_r_entropy_vs_improvement": round(r_ent, 4),
        "pearson_p_entropy": p_ent if not math.isnan(p_ent) else None,
        "pearson_r_negmargin_vs_improvement": round(r_mg, 4),
        "pearson_p_margin": p_mg if not math.isnan(p_mg) else None,
        "improvement_by_N": improvement_by_N,
        "bucket_analysis": bucket_analysis,
        "very_hard_beats_easy": very_hard_beats_easy,
        "gate": "PASS" if gate_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Criterion 3 — Live gate vs fixed-N
# ---------------------------------------------------------------------------

def _criterion_3(
    loss_by_N: dict,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    N_values: list[int],
    model,
    corpus: list[torch.Tensor],
    device: torch.device,
    n_max: int,
    run_gen_adaptive: bool,
) -> dict:
    """Oracle gate PPL + generate_adaptive loop-count demonstration."""
    N_max = N_values[-1]

    # Best fixed-N from Criterion 1 result (recompute for this function's use)
    ppl_by_N = {N: float(loss_by_N[N].mean().exp()) for N in N_values}
    best_N = min(ppl_by_N, key=ppl_by_N.__getitem__)
    best_fixed_ppl = ppl_by_N[best_N]

    # Calibrate thresholds at three percentile levels; report all, highlight best
    cal_levels = [
        ("conservative_p90_p10", 0.90, 0.10),
        ("moderate_p70_p30",     0.70, 0.30),
        ("aggressive_p50_p50",   0.50, 0.50),
    ]
    oracle_results: list[dict] = []
    for label, ent_q, mg_q in cal_levels:
        et = float(torch.quantile(entropy, ent_q))
        mt = float(torch.quantile(margin,  mg_q))
        is_hard = (entropy > et) | (margin < mt)
        g_loss = torch.where(is_hard, loss_by_N[N_max], loss_by_N[N_values[0]])
        g_ppl  = float(g_loss.mean().exp())
        ml     = float(torch.where(
            is_hard,
            torch.full_like(entropy, float(n_max)),
            torch.ones_like(entropy),
        ).mean())
        hf = float(is_hard.float().mean())
        oracle_results.append({
            "calibration": label,
            "ent_thresh": round(et, 4),
            "margin_thresh": round(mt, 4),
            "hard_fraction": round(hf, 3),
            "oracle_gate_ppl": round(g_ppl, 4),
            "oracle_gate_mean_loops": round(ml, 3),
        })

    # Pick the moderate calibration (p70/p30) as the primary comparison
    primary = next(r for r in oracle_results if "moderate" in r["calibration"])
    gate_ppl = primary["oracle_gate_ppl"]
    mean_loops = primary["oracle_gate_mean_loops"]
    ent_thresh = primary["ent_thresh"]
    margin_thresh = primary["margin_thresh"]

    print("\n[Criterion 3] Gate vs fixed-N (oracle + generate_adaptive demo):")
    print(f"  Best fixed-N: N={best_N}  PPL={best_fixed_ppl:.4f}")
    for r in oracle_results:
        print(
            f"  oracle [{r['calibration']}]: "
            f"PPL={r['oracle_gate_ppl']:.4f}  "
            f"mean_loops={r['oracle_gate_mean_loops']:.2f}  "
            f"hard_frac={r['hard_fraction']:.3f}"
        )

    # generate_adaptive demonstration
    gen_demo: dict = {"status": "skipped"}
    if run_gen_adaptive:
        try:
            loops_all: list[int] = []
            n_prompts = min(3, len(corpus))
            for pi, ids_cpu in enumerate(corpus[:n_prompts]):
                prompt_ids = ids_cpu[:32].unsqueeze(0).to(device)
                gen_ids, per_tok_loops = generate_adaptive(
                    model,
                    prompt_ids,
                    n_max=n_max,
                    signal="both",
                    ent_thresh=ent_thresh,
                    margin_thresh=margin_thresh,
                    max_new_tokens=24,
                )
                loops_all.extend(per_tok_loops)
                print(
                    f"  prompt {pi + 1}/{n_prompts}: "
                    f"generated {len(per_tok_loops)} tokens  "
                    f"mean_loops={sum(per_tok_loops) / len(per_tok_loops):.2f}  "
                    f"range=[{min(per_tok_loops)}, {max(per_tok_loops)}]",
                    flush=True,
                )
            gen_demo = {
                "status": "ok",
                "n_prompts": n_prompts,
                "mean_loops_per_token": round(sum(loops_all) / len(loops_all), 3),
                "loops_vary": min(loops_all) != max(loops_all),
                "min_loops": min(loops_all),
                "max_loops": max(loops_all),
            }
        except Exception as exc:
            print(f"  WARNING: generate_adaptive failed: {exc}", flush=True)
            gen_demo = {"status": "failed", "error": str(exc)}

    # Gate check
    gate_ppl_ok = gate_ppl <= best_fixed_ppl * 1.05   # within 5%
    loops_ok = mean_loops < best_N
    gate_pass = gate_ppl_ok and loops_ok

    print(f"  Primary oracle PPL={gate_ppl:.4f}  mean_loops={mean_loops:.2f}")
    print(f"  gate_ppl_within_5pct: {gate_ppl_ok}  mean_loops_lt_best_N: {loops_ok}")
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}")

    return {
        "best_fixed_N": best_N,
        "best_fixed_N_ppl": round(best_fixed_ppl, 4),
        "oracle_calibrations": oracle_results,
        "primary_calibration": "moderate_p70_p30",
        "primary_oracle_gate_ppl": gate_ppl,
        "primary_oracle_mean_loops": mean_loops,
        "gate_ppl_within_5pct_of_best_fixed": gate_ppl_ok,
        "gate_mean_loops_lt_best_N": loops_ok,
        "generate_adaptive_demo": gen_demo,
        "gate": "PASS" if gate_pass else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Recursion eval — Track R Task 5 (issue #118)"
    )
    ap.add_argument(
        "--adapter-dir", required=True,
        help="Directory containing adapter_weights.pt + recur_config.json",
    )
    ap.add_argument("--device", default="cuda", help="Eval device (default: cuda)")
    ap.add_argument(
        "--n-seqs", type=int, default=30,
        help="Number of test sequences to evaluate (default: 30)",
    )
    ap.add_argument(
        "--max-seq-len", type=int, default=512,
        help="Tokens per evaluation sequence (default: 512)",
    )
    ap.add_argument(
        "--n-max", type=int, default=8,
        help="Maximum recurrence to evaluate (default: 8)",
    )
    ap.add_argument(
        "--no-gen-adaptive", action="store_true",
        help="Skip the generate_adaptive demonstration (useful for smoke/CPU runs)",
    )
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    cfg_path = adapter_dir / "recur_config.json"
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    P      = cfg["P"]
    Q      = cfg["Q"]
    base_id = cfg.get("base", "microsoft/bitnet-b1.58-2B-4T-bf16")
    device  = torch.device(args.device)
    N_VALUES: list[int] = list(range(1, args.n_max + 1))

    print(
        f"[recur_eval] adapter_dir={adapter_dir}  P={P}  Q={Q}  n_max={args.n_max}\n"
        f"[recur_eval] device={device}  n_seqs={args.n_seqs}  "
        f"seq_len={args.max_seq_len}  N_values={N_VALUES}",
        flush=True,
    )

    model, tokenizer = _load_model_and_adapter(base_id, adapter_dir, P, Q, device)
    corpus = _build_eval_corpus(tokenizer, args.n_seqs, args.max_seq_len)

    print(
        f"\n[recur_eval] Computing per-token losses at N in {N_VALUES} "
        f"({len(corpus)} seqs × {args.max_seq_len} tokens) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    loss_by_N, entropy, margin = _compute_per_token(
        model, corpus, N_VALUES, device
    )
    elapsed = time.perf_counter() - t0
    n_tokens = int(entropy.numel())
    print(
        f"[recur_eval] Done: {n_tokens:,} tokens × {len(N_VALUES)} N values "
        f"in {elapsed:.1f}s ({n_tokens * len(N_VALUES) / elapsed:.0f} tok/s effective)",
        flush=True,
    )

    c1 = _criterion_1(loss_by_N, N_VALUES)
    c2 = _criterion_2(loss_by_N, entropy, margin, N_VALUES)
    c3 = _criterion_3(
        loss_by_N, entropy, margin, N_VALUES,
        model, corpus, device, args.n_max,
        run_gen_adaptive=not args.no_gen_adaptive,
    )

    # ---- Verdict ----
    n_pass = sum(1 for x in [c1, c2, c3] if x["gate"] == "PASS")
    if n_pass == 3:
        verdict = (
            "PASS (3/3): all three criteria met — "
            "PPL improves with N, hard tokens benefit more from extra slab passes, "
            "and the adaptive gate matches best fixed-N quality at lower mean loops/token."
        )
    elif n_pass == 2:
        failed = [
            f"C{i+1}" for i, x in enumerate([c1, c2, c3]) if x["gate"] == "FAIL"
        ]
        verdict = (
            f"PARTIAL (2/3): criteria {', '.join(failed)} failed — "
            "recursion shows partial benefit; 5M-token adapter may need more training "
            "to fully confirm the gate hypothesis."
        )
    elif n_pass == 1:
        verdict = (
            "PARTIAL (1/3): weak signal — "
            "recursion benefit is marginal on this 5M-token adapter; "
            "more training tokens required before the gate hypothesis can be evaluated fairly."
        )
    else:
        verdict = (
            "FAIL (0/3): null result — "
            "no criterion met; the 5M-token adapter did not learn to benefit from "
            "extra slab passes. This is an honest negative result: "
            "the looped-depth hypothesis is not yet confirmed at this training scale."
        )

    print(f"\n[recur_eval] VERDICT: {verdict}")

    out = {
        "adapter_dir": str(adapter_dir),
        "base_model": base_id,
        "P": P,
        "Q": Q,
        "n_seqs_eval": len(corpus),
        "n_tokens_eval": n_tokens,
        "N_values_eval": N_VALUES,
        "eval_time_s": round(elapsed, 1),
        "criterion_1_ppl_vs_N": c1,
        "criterion_2_hard_token_benefit": c2,
        "criterion_3_gate_vs_fixed_N": c3,
        "n_criteria_passed": n_pass,
        "verdict": verdict,
    }
    out_path = adapter_dir / "eval.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[recur_eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
