# Weight-Space Diffusion — Experiment 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-only ε-prediction DDPM that memorizes and reconstructs real BitNet FFN weight sub-matrices, gating whether weight-space diffusion can represent points in FFN weight space (Experiment 0).

**Architecture:** Three focused Python modules under `scripts/diffusion/` — a pure testable DDPM core (schedule + ε-MLP + train/sample + metrics), a target extractor that memory-maps real weight blocks out of the base safetensors, and a CLI harness that runs gates 0a/0b and emits a pass/fail JSON report. A plain-assert test file validates each in isolation, with a seconds-fast synthetic end-to-end smoke standing in for the full 41 K-D run.

**Tech Stack:** Python 3.11 (`C:\Python311\python.exe`), PyTorch (CPU), `safetensors`, `huggingface_hub` cache. No pytest — plain-assert test scripts (repo convention). No new dependencies.

## Global Constraints

- CPU-only. Never move tensors to CUDA — the RTX 3060 is running a training job; Experiment 0 must not touch it. (`device='cpu'` everywhere.)
- No new third-party dependencies beyond what the repo already uses (torch, numpy, safetensors, huggingface_hub).
- Base model id is exactly `microsoft/bitnet-b1.58-2B-4T-bf16` (matches `scripts/lora/mote_train.py --base`); load weights from the HF cache with `local_files_only=True`; never download.
- All randomness seeded and deterministic: every train/sample function takes a `seed` and uses a local `torch.Generator`; identical seed → identical output.
- Tests are plain-python assert scripts run with `C:\Python311\python.exe scripts\diffusion\test_wsd.py` (no pytest). New tests append to that one file.
- DDPM constants (verbatim): `T=100`, `beta_start=1e-4`, `beta_end=0.02`, linear schedule; ε-prediction; MSE loss; Adam.
- Target vector dimensionality for the base-weight gates: `dim = 16*2560 = 40960` (a `[16, 2560]` weight block flattened).
- Do NOT `git commit` or `git add` — leave all changes uncommitted. This session commits only on explicit user request. (Omit the commit step the TDD template usually ends on.)

---

### Task 1: DDPM core + metrics (`wsd_ddpm.py`) with synthetic end-to-end smoke

**Files:**
- Create: `scripts/diffusion/wsd_ddpm.py`
- Test: `scripts/diffusion/test_wsd.py`

**Interfaces:**
- Produces:
  - `make_schedule(T=100, beta_start=1e-4, beta_end=0.02) -> Schedule` where `Schedule` is a dataclass with float32 tensors `betas, alphas, alpha_bars` each shape `[T]`.
  - `q_sample(x0: Tensor[B,dim], t: Tensor[B] long, noise: Tensor[B,dim], sched: Schedule) -> Tensor[B,dim]`.
  - `DenoiseMLP(dim: int, hidden: int = 256)` — `nn.Module`, `forward(x_t: Tensor[B,dim], t: Tensor[B] long) -> Tensor[B,dim]` (predicts x0). **NOTE (bring-up correction): the core uses x0-prediction, not ε-prediction, and `make_schedule` defaults `beta_end=0.10`, and `sample_ddpm` is deterministic DDIM — see the implemented `wsd_ddpm.py`; the ε-prediction code originally shown in Task 1 was superseded during debugging.**
  - `train_ddpm(model, targets: Tensor[K,dim], sched, epochs: int, lr: float, seed: int) -> float` (returns final loss; trains in place).
  - `sample_ddpm(model, sched, n: int, dim: int, seed: int) -> Tensor[n,dim]`.
  - `cosine(a: Tensor[dim], b: Tensor[dim]) -> float`; `rel_l2(a, b) -> float`; `nearest_target(sample: Tensor[dim], targets: Tensor[K,dim]) -> tuple[int, float, float]` (idx, cosine, rel_l2).

- [ ] **Step 1: Write the failing tests**

Append to `scripts/diffusion/test_wsd.py` (create the file with this content):

```python
"""Plain-assert tests for weight-space diffusion Exp 0. Run:
    C:\\Python311\\python.exe scripts\\diffusion\\test_wsd.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from wsd_ddpm import (
    make_schedule, q_sample, EpsMLP, train_ddpm, sample_ddpm,
    cosine, rel_l2, nearest_target,
)

def test_schedule():
    s = make_schedule(T=100)
    assert s.betas.shape == (100,) and s.alpha_bars.shape == (100,)
    assert torch.all(s.alpha_bars[1:] < s.alpha_bars[:-1])          # strictly decreasing
    assert float(s.alpha_bars.min()) > 0.0 and float(s.alpha_bars[0]) <= 1.0
    print("[pass] test_schedule")

def test_q_sample_identity_at_t0():
    s = make_schedule(T=100)
    x0 = torch.randn(2, 8)
    noise = torch.randn(2, 8)
    t0 = torch.zeros(2, dtype=torch.long)
    xt = q_sample(x0, t0, noise, s)
    # at t=0 alpha_bar≈alphas[0]=1-1e-4, so xt ≈ x0 (not exact); assert dominated by x0
    assert cosine(xt[0], x0[0]) > 0.999
    print("[pass] test_q_sample_identity_at_t0")

def test_metrics():
    v = torch.randn(40)
    assert abs(cosine(v, v) - 1.0) < 1e-5
    assert rel_l2(v, v) < 1e-6
    tgts = torch.stack([v, torch.randn(40), torch.randn(40)])
    idx, cos, _ = nearest_target(v, tgts)
    assert idx == 0 and cos > 0.999
    print("[pass] test_metrics")

def test_sample_shape_and_determinism():
    s = make_schedule(T=20)
    m = EpsMLP(dim=8, hidden=32)
    a = sample_ddpm(m, s, n=3, dim=8, seed=1)
    b = sample_ddpm(m, s, n=3, dim=8, seed=1)
    c = sample_ddpm(m, s, n=3, dim=8, seed=2)
    assert a.shape == (3, 8)
    assert torch.allclose(a, b)               # same seed → identical
    assert not torch.allclose(a, c)           # different seed → different
    print("[pass] test_sample_shape_and_determinism")

def test_synthetic_end_to_end_smoke():
    # Gate-0a proxy at tiny dim: memorize ONE random vector, recover it.
    torch.manual_seed(0)
    s = make_schedule(T=100)
    dim = 64
    target = torch.randn(1, dim)
    m = EpsMLP(dim=dim, hidden=256)
    loss = train_ddpm(m, target, s, epochs=3000, lr=1e-3, seed=0)
    samples = sample_ddpm(m, s, n=5, dim=dim, seed=0)
    cosines = [cosine(samples[i], target[0]) for i in range(5)]
    assert min(cosines) > 0.99, f"min cosine {min(cosines):.4f}, loss {loss:.5f}"
    print(f"[pass] test_synthetic_end_to_end_smoke (min cos {min(cosines):.4f})")

if __name__ == "__main__":
    test_schedule()
    test_q_sample_identity_at_t0()
    test_metrics()
    test_sample_shape_and_determinism()
    test_synthetic_end_to_end_smoke()
    print("ALL wsd_ddpm TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'wsd_ddpm'` (module not created yet).

- [ ] **Step 3: Implement `wsd_ddpm.py`**

Create `scripts/diffusion/wsd_ddpm.py`:

```python
"""Weight-space diffusion Exp 0 — pure DDPM core + reconstruction metrics (CPU-only)."""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> Schedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return Schedule(betas, alphas, alpha_bars)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: Schedule) -> torch.Tensor:
    ab = sched.alpha_bars[t].unsqueeze(-1)               # [B,1]
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    # sinusoidal embedding, t: [B] long -> [B, dim]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class EpsMLP(nn.Module):
    """3-layer ε-predictor with additive sinusoidal timestep conditioning."""
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(dim, hidden)
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(_timestep_embedding(t, self.hidden))
        h = F.silu(self.l1(x_t) + temb)
        h = F.silu(self.l2(h))
        return self.l3(h)


def train_ddpm(model: nn.Module, targets: torch.Tensor, sched: Schedule,
               epochs: int, lr: float, seed: int) -> float:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    T = sched.betas.shape[0]
    K = targets.shape[0]
    loss_val = float("nan")
    for _ in range(epochs):
        t = torch.randint(0, T, (K,), generator=g)
        noise = torch.randn(targets.shape, generator=g)
        x_t = q_sample(targets, t, noise, sched)
        eps = model(x_t, t)
        loss = F.mse_loss(eps, noise)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_val = float(loss.detach())
    return loss_val


@torch.no_grad()
def sample_ddpm(model: nn.Module, sched: Schedule, n: int, dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    model.eval()
    T = sched.betas.shape[0]
    x = torch.randn(n, dim, generator=g)
    for t in range(T - 1, -1, -1):
        tb = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tb)
        alpha = sched.alphas[t]; ab = sched.alpha_bars[t]; beta = sched.betas[t]
        mean = (1.0 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1.0 - ab)) * eps)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn(n, dim, generator=g)
        else:
            x = mean
    return x


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12))


def nearest_target(sample: torch.Tensor, targets: torch.Tensor):
    cs = [cosine(sample, targets[i]) for i in range(targets.shape[0])]
    idx = int(max(range(len(cs)), key=lambda i: cs[i]))
    return idx, cs[idx], rel_l2(sample, targets[idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: PASS — prints `[pass] ...` for all five and `ALL wsd_ddpm TESTS PASSED`. If `test_synthetic_end_to_end_smoke` fails on min-cosine, increase `epochs` (try 5000) — but do NOT relax the 0.99 threshold; a failure to memorize one 64-D vector is itself the Gate-0a signal and must be reported, not hidden.

---

### Task 2: Target extraction from base weights (`wsd_targets.py`)

**Files:**
- Create: `scripts/diffusion/wsd_targets.py`
- Test: `scripts/diffusion/test_wsd.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `extract_targets(base_id: str, k: int, seed: int) -> tuple[Tensor[k,40960], dict]`.
    Returns normalized float32 targets and `meta = {"mean": float, "std": float, "dim": 40960, "block_shape": [16,2560], "locations": list, "base_id": str}`.
  - `denormalize(x: Tensor, meta: dict) -> Tensor` (`x * meta["std"] + meta["mean"]`).
  - `normalize(x: Tensor, meta: dict) -> Tensor` (inverse; used by tests).

**Notes for the implementer:**
- Find the base weights in the HF cache without downloading. Use
  `from huggingface_hub import hf_hub_download` with `local_files_only=True` to locate the
  `model.safetensors` (or the sharded index) for `microsoft/bitnet-b1.58-2B-4T-bf16`; open with
  `safetensors.safe_open(path, framework="pt", device="cpu")`. If sharded, read the
  `model.safetensors.index.json` weight_map to find which shard holds a given tensor key.
- FFN projection tensor keys follow HF Llama-style naming:
  `model.layers.{i}.mlp.gate_proj.weight` (shape `[6912, 2560]`). Verify the exact key by listing
  keys from the safetensors and matching `mlp.gate_proj.weight`; do not hardcode a layer count.
- Each target = a `[16, 2560]` block: rows `[row_off : row_off+16]` of a chosen layer's
  `gate_proj.weight`, cast to float32, flattened to `[40960]`. Pick `k` distinct `(layer, row_off)`
  locations deterministically from `seed` (e.g. seeded choice of layer indices and row offsets),
  ensuring no two targets share the same `(layer, row_off)`.
- Normalization: stack the `k` raw target vectors, compute scalar `mean` and `std` over all
  elements, store in `meta`, return `(stacked - mean) / std`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/diffusion/test_wsd.py` (add these functions and add their calls to `__main__`):

```python
from wsd_targets import extract_targets, denormalize, normalize

_BASE = "microsoft/bitnet-b1.58-2B-4T-bf16"

def test_extract_targets_shapes():
    tgts, meta = extract_targets(_BASE, k=5, seed=0)
    assert tgts.shape == (5, 40960) and tgts.dtype == torch.float32
    assert meta["dim"] == 40960 and meta["block_shape"] == [16, 2560]
    assert len(meta["locations"]) == 5 and len(set(map(tuple, meta["locations"]))) == 5  # distinct
    print("[pass] test_extract_targets_shapes")

def test_normalize_roundtrip():
    tgts, meta = extract_targets(_BASE, k=2, seed=1)
    raw = denormalize(tgts, meta)
    back = normalize(raw, meta)
    assert torch.allclose(tgts, back, atol=1e-4)
    # normalized targets should be roughly unit-scale
    assert 0.3 < float(tgts.std()) < 3.0
    print("[pass] test_normalize_roundtrip")
```

Add to `__main__` before the final print:
```python
    test_extract_targets_shapes()
    test_normalize_roundtrip()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'wsd_targets'`.

- [ ] **Step 3: Implement `wsd_targets.py`**

Create `scripts/diffusion/wsd_targets.py`:

```python
"""Extract real FFN weight blocks from the BitNet base as diffusion targets (CPU, mmap)."""
import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

_HIDDEN = 2560
_ROWS = 16
_DIM = _ROWS * _HIDDEN  # 40960


def _resolve_shard(base_id: str, key: str) -> str:
    """Return local path to the safetensors shard containing `key` (cache-only)."""
    try:
        idx_path = hf_hub_download(base_id, "model.safetensors.index.json", local_files_only=True)
        weight_map = json.load(open(idx_path))["weight_map"]
        shard = weight_map[key]
    except Exception:
        shard = "model.safetensors"  # single-file model
    return hf_hub_download(base_id, shard, local_files_only=True)


def _gate_proj_key(layer: int) -> str:
    return f"model.layers.{layer}.mlp.gate_proj.weight"


def extract_targets(base_id: str, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    # discover layer count by probing keys of the first shard
    first = _resolve_shard(base_id, _gate_proj_key(0))
    with safe_open(first, framework="pt", device="cpu") as f:
        keys = [kk for kk in f.keys() if kk.endswith("mlp.gate_proj.weight")]
    n_layers = len(keys)
    assert n_layers > 0, "no mlp.gate_proj.weight tensors found in base"

    # choose k distinct (layer, row_off) locations deterministically
    locations = []
    seen = set()
    while len(locations) < k:
        layer = int(torch.randint(0, n_layers, (1,), generator=g))
        # gate_proj rows = 6912; leave room for a 16-row block
        row_off = int(torch.randint(0, 6912 - _ROWS, (1,), generator=g))
        if (layer, row_off) in seen:
            continue
        seen.add((layer, row_off)); locations.append([layer, row_off])

    vecs = []
    for layer, row_off in locations:
        key = _gate_proj_key(layer)
        shard = _resolve_shard(base_id, key)
        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(key)                       # [6912, 2560]
        block = w[row_off:row_off + _ROWS, :].to(torch.float32).reshape(-1)  # [40960]
        vecs.append(block)
    raw = torch.stack(vecs)                              # [k, 40960]

    mean = float(raw.mean()); std = float(raw.std()) + 1e-12
    meta = {"mean": mean, "std": std, "dim": _DIM, "block_shape": [_ROWS, _HIDDEN],
            "locations": locations, "base_id": base_id}
    return (raw - mean) / std, meta


def denormalize(x: torch.Tensor, meta: dict) -> torch.Tensor:
    return x * meta["std"] + meta["mean"]


def normalize(x: torch.Tensor, meta: dict) -> torch.Tensor:
    return (x - meta["mean"]) / meta["std"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: PASS — the two new tests print `[pass]`. If `hf_hub_download(..., local_files_only=True)` raises (base not in cache), STOP and report: the base must already be cached (it is used by `mote_train.py`); do not download.

---

### Task 3: Exp-0 harness (`wsd_exp0.py`) — gates 0a/0b + report

**Files:**
- Create: `scripts/diffusion/wsd_exp0.py`
- Test: `scripts/diffusion/test_wsd.py` (append)

**Interfaces:**
- Consumes: `wsd_ddpm` (all) and `wsd_targets.extract_targets/denormalize`.
- Produces:
  - `run_gate(k: int, epochs: int, n_samples: int, seed: int, base_id: str, synthetic: bool) -> dict`
    — returns a report dict: `{"k", "n_samples", "mean_nearest_cosine", "min_nearest_cosine",
    "mean_rel_l2", "coverage", "per_sample": [...], "passed": bool}`.
  - `main()` — argparse CLI: `--gate {0a,0b}`, `--k`, `--epochs` (default 4000), `--n-samples`,
    `--seed` (default 0), `--base` (default `microsoft/bitnet-b1.58-2B-4T-bf16`),
    `--out` (default `.docs/diffusion/wsd_exp0/report.json`), `--self-test`.
- Pass rule inside `run_gate`: `passed = (min_nearest_cosine >= 0.99 if k == 1
  else mean_nearest_cosine >= 0.98 and coverage == k)`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/diffusion/test_wsd.py` (add function + call in `__main__`):

```python
from wsd_exp0 import run_gate

def test_run_gate_synthetic():
    # synthetic mode: no base model; validates harness wiring + report shape + pass logic
    rep = run_gate(k=1, epochs=3000, n_samples=5, seed=0, base_id="", synthetic=True)
    assert set(rep) >= {"mean_nearest_cosine", "min_nearest_cosine", "coverage", "passed", "per_sample"}
    assert len(rep["per_sample"]) == 5
    assert rep["passed"] is True and rep["min_nearest_cosine"] > 0.99
    print("[pass] test_run_gate_synthetic")
```

Add `test_run_gate_synthetic()` to `__main__` before the final print.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'wsd_exp0'`.

- [ ] **Step 3: Implement `wsd_exp0.py`**

Create `scripts/diffusion/wsd_exp0.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Python311\python.exe scripts\diffusion\test_wsd.py`
Expected: PASS — all tests including `test_run_gate_synthetic` print `[pass]`, then `ALL wsd_ddpm TESTS PASSED`.

- [ ] **Step 5: Run the real Gate 0a end-to-end (the actual experiment)**

Run: `C:\Python311\python.exe scripts\diffusion\wsd_exp0.py --gate 0a --epochs 6000`
Expected: prints `PASS` or `FAIL` with metrics and writes `.docs/diffusion/wsd_exp0/report.json`.
This is the experiment result — report the verdict and metrics; do not tune to force a pass. A genuine FAIL (min cosine < 0.99, or worse < 0.9) is the kill signal for the whole weight-space direction and must be surfaced as-is.

- [ ] **Step 6: Run Gate 0b (multi-point / mode-coverage)**

Run: `C:\Python311\python.exe scripts\diffusion\wsd_exp0.py --gate 0b --epochs 8000`
Expected: prints PASS/FAIL; `coverage` should be `5/5` on a pass. Report coverage + mean cosine; low coverage with high cosine = mode collapse (a soft-fail to surface, per spec).

---

## Self-Review

**Spec coverage:** wsd_ddpm (schedule/model/train/sample/metrics) → Task 1; wsd_targets (extraction/normalization) → Task 2; wsd_exp0 harness + gates 0a/0b + report + --self-test → Task 3; test_wsd.py → all tasks; reconstruction-fidelity metric (cosine/rel_l2, not PPL) → Tasks 1&3; kill criteria surfaced in Steps 5/6. CPU-only + base id + determinism → Global Constraints. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step contains complete runnable code.

**Type consistency:** `Schedule` fields, `make_schedule/q_sample/EpsMLP/train_ddpm/sample_ddpm/cosine/rel_l2/nearest_target` signatures identical across Task 1 defs and Task 3 imports; `extract_targets/denormalize/normalize` and the `meta` keys (`mean/std/dim/block_shape/locations`) consistent between Task 2 def and Task 3 use; `run_gate` return keys match the Task-3 test assertions; `dim=40960` consistent with `[16,2560]` block.
