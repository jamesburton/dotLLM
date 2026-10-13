"""pdiff_gauge_validate.py — Cheap 2-adapter gauge-fix validation.

Trains exactly 2 adapters with IDENTICAL LoRA init (--lora-init-seed) but
DIFFERENT data windows, extracts lora_A for layer 0 gate_proj, and reports
the pairwise cosine similarity.

Expected results:
  OLD (random init, per-adapter PYTHONHASHSEED): pairwise cos ~0.12
  FIXED (shared init, data-window variation only): pairwise cos >> 0.12

Run (GPU, ~8-9 min per adapter):
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface TORCHDYNAMO_DISABLE=1 \\
    C:/Python311/python.exe scripts/diffusion/pdiff_gauge_validate.py \\
    --base microsoft/bitnet-b1.58-2B-4T-bf16 \\
    --dataset HuggingFaceH4/no_robots \\
    --out .docs/diffusion/pdiff_boot_fixedinit \\
    --device cuda

Arguments (all have sensible defaults — match the full-16 bootstrap):
  --base, --dataset, --out, --steps, --examples-per, --lora-init-seed, --seed-base,
  --layer, --proj, --rank, --device, --no-4bit
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR   = Path(__file__).parent
_REPO_ROOT    = _SCRIPT_DIR.parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "lora" / "train_task_lora.py"
_BOOTSTRAP    = _REPO_ROOT / "scripts" / "diffusion" / "pdiff_bootstrap.py"

# re-use extraction utilities from the bootstrap
sys.path.insert(0, str(_SCRIPT_DIR))
from pdiff_bootstrap import extract_lora_vec


def train_adapter(idx: int, out_dir: Path, args: argparse.Namespace) -> Path:
    """Train one adapter using the fixed-init bootstrap (calls train_task_lora.py)."""
    adapter_dir = out_dir / f"adapter_{idx:03d}"
    if adapter_dir.exists() and (adapter_dir / "adapter_model.safetensors").exists():
        print(f"[adapter {idx}] Already trained — skipping.")
        return adapter_dir

    start = idx * args.examples_per
    end   = start + args.examples_per
    split = f"train[{start}:{end}]"

    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--task",           "instruction",
        "--base",           args.base,
        "--dataset",        args.dataset,
        "--split",          split,
        "--steps",          str(args.steps),
        "--lora-init-seed", str(args.lora_init_seed),
        "--no-4bit",        # required flag — no positional args
        "--max-seq-len",    "512",
        "--grad-checkpoint",
        "--out",            str(adapter_dir),
    ]
    # --no-4bit is a store_true flag; include it unconditionally (matches full bootstrap)
    # Already added above.

    env = {**os.environ, "PYTHONHASHSEED": str(args.seed_base), "TORCHDYNAMO_DISABLE": "1"}
    print(f"\n[adapter {idx}] Training | split={split} | lora_init_seed={args.lora_init_seed}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for adapter {idx} (exit {result.returncode})")
    return adapter_dir


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def baseline_cosines(baseline_dir: Path) -> list[float]:
    """Compute all pairwise cosines from the OLD random-init population."""
    vecs = []
    for p in sorted(baseline_dir.glob("adapter_*_vec.npy")):
        vecs.append(np.load(str(p)).astype(np.float64))
    if len(vecs) < 2:
        return []
    cosines = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cosines.append(cosine(vecs[i], vecs[j]))
    return cosines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="2-adapter gauge-fix validation for weight-space diffusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base",    default="microsoft/bitnet-b1.58-2B-4T-bf16")
    ap.add_argument("--dataset", default="HuggingFaceH4/no_robots")
    ap.add_argument("--out",     default=str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit"))
    ap.add_argument("--steps",            type=int, default=200)
    ap.add_argument("--examples-per",     type=int, default=200, dest="examples_per")
    ap.add_argument("--lora-init-seed",   type=int, default=0,   dest="lora_init_seed")
    ap.add_argument("--seed-base",        type=int, default=42,  dest="seed_base")
    ap.add_argument("--layer",            type=int, default=0)
    ap.add_argument("--proj",             default="gate_proj")
    ap.add_argument("--rank",             type=int, default=16)
    ap.add_argument("--device",           default="cuda")
    ap.add_argument(
        "--baseline-dir",
        default=str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot"),
        dest="baseline_dir",
        help="Dir containing OLD random-init adapter_*_vec.npy files for comparison.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("pdiff_gauge_validate: 2-adapter fixed-init gauge check")
    print("=" * 60)
    print(f"  base={args.base}  dataset={args.dataset}")
    print(f"  lora_init_seed={args.lora_init_seed}  seed_base={args.seed_base}")
    print(f"  steps={args.steps}  examples_per={args.examples_per}")
    print(f"  layer={args.layer}  proj={args.proj}  rank={args.rank}")
    print(f"  out={out_dir}")

    # --- Train 2 adapters sequentially ---
    dirs = []
    for i in range(2):
        d = train_adapter(i, out_dir, args)
        dirs.append(d)

    # --- Extract lora_A vectors ---
    vecs = []
    for i, d in enumerate(dirs):
        print(f"\n[adapter {i}] Extracting lora_A from {d} ...")
        v = extract_lora_vec(d, args.layer, args.proj, args.rank)
        print(f"  shape={v.shape}  l2_norm={float(np.linalg.norm(v)):.4f}")
        np_path = out_dir / f"adapter_{i:03d}_vec.npy"
        np.save(str(np_path), v.astype(np.float32))
        vecs.append(v)

    # --- Compute pairwise cosine ---
    cos_fixed = cosine(vecs[0], vecs[1])
    print(f"\n{'='*60}")
    print(f"FIXED-INIT pairwise cosine (adapter 0 vs 1): {cos_fixed:.4f}")

    # --- Baseline comparison ---
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        bl = baseline_cosines(baseline_dir)
        if bl:
            bl_mean = float(np.mean(bl))
            bl_min  = float(np.min(bl))
            bl_max  = float(np.max(bl))
            print(f"OLD random-init baseline: mean={bl_mean:.4f}  min={bl_min:.4f}  max={bl_max:.4f}  (N={len(bl)} pairs)")
        else:
            print("(no baseline vectors found for comparison)")
    else:
        print(f"(baseline dir not found: {baseline_dir})")
        bl_mean = 0.12  # documented baseline

    verdict = "FIXED+VALIDATED" if cos_fixed > 0.5 else "FIXED+FAILED"
    print(f"\nVerdict: {verdict}  (threshold >0.5 for clear gauge alignment)")
    print(f"{'='*60}\n")

    # --- Write report ---
    report_path = Path(r"C:\Users\james\AppData\Local\Temp\claude\E--Development-dotLLM\f5d62daa-0be0-4e86-81e4-cf8f16601e91\scratchpad\gauge-fix-report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bl_summary = ""
    if baseline_dir.exists():
        bl = baseline_cosines(baseline_dir)
        if bl:
            bl_summary = f"mean={np.mean(bl):.4f}, min={np.min(bl):.4f}, max={np.max(bl):.4f} ({len(bl)} pairs)"
        else:
            bl_summary = "no vec files found"
    else:
        bl_summary = "baseline dir missing (documented: ~0.12)"

    full_run_cmd = (
        f"PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface TORCHDYNAMO_DISABLE=1 "
        f"C:/Python311/python.exe scripts/diffusion/pdiff_bootstrap.py "
        f"--n 16 --task instruction --layer {args.layer} --proj {args.proj} --rank {args.rank} "
        f"--base {args.base} "
        f"--dataset {args.dataset} "
        f"--steps {args.steps} --examples-per {args.examples_per} "
        f"--lora-init-seed {args.lora_init_seed} "
        f"--out .docs/diffusion/pdiff_boot_fixedinit "
        f"--seed-base {args.seed_base} --device cuda --no-4bit"
    )

    report = f"""# Gauge Fix Validation Report

## Code changes

### `scripts/lora/train_task_lora.py`
- Added `--lora-init-seed` argument (default=0).
- Added `torch.manual_seed(args.lora_init_seed)` immediately before `model = get_peft_model(model, lcfg)`.
- Comment documents gauge-freedom root cause (A→RA, B→BR⁻¹).

### `scripts/diffusion/pdiff_bootstrap.py`
- Added `--lora-init-seed` argument (default=0), forwarded identically to every training call.
- PYTHONHASHSEED now **shared** (seed_base for all adapters, not seed_base+i).
- `run_train()` passes `--lora-init-seed` to train_task_lora.py.
- meta.json now records `lora_init_seed` and `gauge_fixed: true`.

## Results

| Metric | Value |
|--------|-------|
| Fixed-init pairwise cosine (adapter 0 vs 1) | **{cos_fixed:.4f}** |
| OLD random-init baseline | {bl_summary} |
| Verdict | **{verdict}** |

## Full 16-adapter re-run command

```
{full_run_cmd}
```
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
