"""Experiment 0 harness: weight-space diffusion representation gate (CPU-only)."""
import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from wsd_ddpm import make_schedule, DenoiseMLP, train_ddpm, sample_ddpm, nearest_target
from wsd_targets import extract_targets, denormalize

_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"


def _synthetic_targets(k: int, dim: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(k, dim, generator=g)
    meta = {"mean": 0.0, "std": 1.0, "dim": dim, "synthetic": True}
    return raw, meta


def run_gate(k: int, epochs: int, n_samples: int, seed: int, base_id: str, synthetic: bool) -> dict:
    if synthetic:
        norm_targets, meta = _synthetic_targets(k, dim=64, seed=seed)
    else:
        norm_targets, meta = extract_targets(base_id, k=k, seed=seed)
    dim = meta["dim"]
    real_targets = denormalize(norm_targets, meta)      # compare in original weight space

    sched = make_schedule()
    model = DenoiseMLP(dim=dim, hidden=256)
    final_loss = train_ddpm(model, norm_targets, sched, epochs=epochs, lr=1e-3, seed=seed)
    samples_norm = sample_ddpm(model, sched, n=n_samples, dim=dim, seed=seed)
    samples = denormalize(samples_norm, meta)

    per_sample, hit = [], set()
    for i in range(n_samples):
        idx, cos, rl2 = nearest_target(samples[i], real_targets)
        per_sample.append({"idx": idx, "cosine": cos, "rel_l2": rl2}); hit.add(idx)
    coss = [p["cosine"] for p in per_sample]
    mean_cos = sum(coss) / len(coss); min_cos = min(coss)
    mean_rl2 = sum(p["rel_l2"] for p in per_sample) / len(per_sample)
    coverage = len(hit)
    passed = (min_cos >= 0.99) if k == 1 else (mean_cos >= 0.98 and coverage == k)
    return {"k": k, "n_samples": n_samples, "final_loss": final_loss,
            "mean_nearest_cosine": mean_cos, "min_nearest_cosine": min_cos,
            "mean_rel_l2": mean_rl2, "coverage": coverage,
            "per_sample": per_sample, "passed": bool(passed)}


def main():
    ap = argparse.ArgumentParser(description="Weight-space diffusion Exp 0 gate (CPU).")
    ap.add_argument("--gate", choices=["0a", "0b"], default="0a")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default=_BASE)
    ap.add_argument("--out", default=os.path.join(".docs", "diffusion", "wsd_exp0", "report.json"))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.gate == "0a":
        k = a.k or 1; n = a.n_samples or 10
    else:
        k = a.k or 5; n = a.n_samples or 50

    rep = run_gate(k=k, epochs=a.epochs, n_samples=n, seed=a.seed,
                   base_id=a.base, synthetic=a.self_test)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    verdict = "PASS" if rep["passed"] else "FAIL"
    print(f"[wsd_exp0] gate={a.gate} k={k} n={n}  {verdict}")
    print(f"  mean_nearest_cosine={rep['mean_nearest_cosine']:.4f}  "
          f"min={rep['min_nearest_cosine']:.4f}  coverage={rep['coverage']}/{k}  "
          f"mean_rel_l2={rep['mean_rel_l2']:.4f}  final_loss={rep['final_loss']:.5f}")
    print(f"[wsd_exp0] report -> {a.out}")


if __name__ == "__main__":
    main()
