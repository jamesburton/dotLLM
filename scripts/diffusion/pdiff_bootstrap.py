"""pdiff_boot: LoRA-population bootstrap for weight-space diffusion (Gate P1).

Runs N same-task LoRA fine-tunes of a base model by calling the existing
scripts/lora/train_task_lora.py in a subprocess loop, extracting lora_A
weights for a chosen layer/projection, and saving a population array.

All adapters solve the same task, varied only by data window and PYTHONHASHSEED,
so the LoRA vectors lie on a low-dimensional manifold (intrinsic dim ~10-50).
This gives Gate P1 the genuine manifold structure needed to train a generalizing
linear autoencoder — unlike the base-weight population which is effectively
isotropic in 40960-D (pairwise cosine ~0.005, Gate P1 held cos=0.038).

Checkpoint/resume:
    Each adapter's vector is saved as adapter_{i:03d}_vec.npy immediately after
    extraction.  On re-run, any adapter whose vec file already exists is skipped
    (both the training subprocess and the extraction step are bypassed).
    The combined population.npy is (re)built at the end from all vec files.

Queued GPU command (run when GPU is free):
    PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface \\
        C:/Python311/python.exe scripts/diffusion/pdiff_bootstrap.py \\
        --n 16 --task instruction --layer 0 --proj gate_proj --rank 16 \\
        --base microsoft/bitnet-b1.58-2B-4T-bf16 \\
        --dataset HuggingFaceH4/no_robots \\
        --steps 200 --examples-per 200 \\
        --out .docs/diffusion/pdiff_boot \\
        --seed-base 42 --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = Path(__file__).parent
_REPO_ROOT    = _SCRIPT_DIR.parent.parent
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "lora" / "train_task_lora.py"

POPULATION_NPY = "population.npy"
META_JSON      = "meta.json"


# ---------------------------------------------------------------------------
# LoRA weight extraction
# ---------------------------------------------------------------------------

def find_lora_key(sf_or_dict, layer: int, proj: str, part: str) -> str:
    """Find the key for lora_{part}.weight at layer/proj in a safetensors handle or state dict.

    Searches for any key containing 'layers.{layer}.mlp.{proj}.lora_{part}' and
    ending in '.weight', to be robust to PEFT version differences in key prefixes
    (e.g. 'base_model.model.model...' vs 'model.model...').
    """
    fragment = f"layers.{layer}.mlp.{proj}.lora_{part}"
    if hasattr(sf_or_dict, "keys"):
        all_keys = list(sf_or_dict.keys())
    else:
        all_keys = list(sf_or_dict)
    candidates = [k for k in all_keys if fragment in k and k.endswith(".weight")]
    if not candidates:
        raise KeyError(
            f"No lora_{part}.weight key found for layer={layer}, proj={proj}. "
            f"Checked {len(all_keys)} keys. First 8: {all_keys[:8]}"
        )
    # Prefer shortest key (fewest wrapper prefixes)
    candidates.sort(key=len)
    return candidates[0]


def extract_lora_vec(adapter_dir: Path, layer: int, proj: str, rank: int) -> np.ndarray:
    """Extract lora_A.weight for a given layer/proj from a saved PEFT adapter directory.

    Tries adapter_model.safetensors first, then adapter_model.bin.

    Args:
        adapter_dir: Path to the saved PEFT adapter directory.
        layer:       Transformer layer index.
        proj:        Projection name (e.g. 'gate_proj').
        rank:        LoRA rank (for validation only; actual shape is read from file).

    Returns:
        1-D float32 numpy array of shape [rank * in_features] (= lora_A flattened).
        For gate_proj with rank=16, hidden_size=2560: shape is [40960] = BLOCK_DIM.
    """
    st_path  = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if st_path.exists():
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors not installed. Run: pip install safetensors"
            ) from exc
        with safe_open(str(st_path), framework="pt", device="cpu") as f:
            key = find_lora_key(f, layer, proj, "A")
            w   = f.get_tensor(key)          # [rank, in_features] or transposed
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        key   = find_lora_key(state, layer, proj, "A")
        w     = state[key]
    else:
        raise FileNotFoundError(
            f"No adapter file found in {adapter_dir}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )

    w_f32 = w.to(torch.float32).reshape(-1).contiguous()
    vec   = w_f32.numpy().copy()
    return vec   # shape [rank * in_features], dtype float32


# ---------------------------------------------------------------------------
# Population management (save / resume)
# ---------------------------------------------------------------------------

def vec_path(out_dir: Path, idx: int) -> Path:
    """Path to the per-adapter float32 vector file."""
    return out_dir / f"adapter_{idx:03d}_vec.npy"


def is_done(out_dir: Path, idx: int) -> bool:
    """Return True if the vector for adapter idx has already been extracted."""
    return vec_path(out_dir, idx).exists()


def save_vec(out_dir: Path, idx: int, vec: np.ndarray) -> None:
    """Save per-adapter vector file atomically (write temp -> rename).

    The temp file is named *_tmp.npy so that np.save does not append an extra
    .npy suffix (np.save adds .npy when the path does not already end in .npy).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"adapter_{idx:03d}_vec_tmp.npy"
    np.save(str(tmp), vec.astype(np.float32))
    dest = vec_path(out_dir, idx)
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)


def load_vec(out_dir: Path, idx: int) -> np.ndarray:
    """Load a previously-saved per-adapter vector."""
    return np.load(str(vec_path(out_dir, idx))).astype(np.float32)


def build_population(out_dir: Path, done_idxs: list) -> np.ndarray:
    """Stack per-adapter vec files into a [N, D] float32 array.

    Args:
        out_dir:    Directory containing adapter_{i:03d}_vec.npy files.
        done_idxs:  Ordered list of adapter indices that have vec files.

    Raises:
        FileNotFoundError: if any index in done_idxs is missing its vec file.
    """
    if not done_idxs:
        raise ValueError("done_idxs is empty; no vectors to stack")
    vecs = []
    for i in done_idxs:
        p = vec_path(out_dir, i)
        if not p.exists():
            raise FileNotFoundError(f"Vector file missing for adapter {i}: {p}")
        vecs.append(np.load(str(p)).astype(np.float32))
    return np.stack(vecs)   # [N, D]


def save_population(out_dir: Path, population: np.ndarray, meta: dict) -> None:
    """Write population.npy and meta.json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / POPULATION_NPY), population)
    with open(out_dir / META_JSON, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved population {population.shape} -> {out_dir / POPULATION_NPY}")
    print(f"Saved meta -> {out_dir / META_JSON}")


# ---------------------------------------------------------------------------
# Training subprocess
# ---------------------------------------------------------------------------

def run_train(
    idx: int,
    adapter_dir: Path,
    args: argparse.Namespace,
    env: dict,
) -> None:
    """Invoke scripts/lora/train_task_lora.py as a subprocess for one adapter.

    Data window: adapter i uses train[i*examples_per : (i+1)*examples_per],
    ensuring each adapter sees a distinct slice of the dataset (same task).

    Gauge fix: every adapter uses the SAME --lora-init-seed (torch RNG seed at
    LoRA construction time) and the SAME PYTHONHASHSEED.  Previously PYTHONHASHSEED
    was incremented per-adapter (seed_base+i) on the assumption it would introduce
    useful diversity; in practice each subprocess also drew a fresh OS-entropy torch
    RNG state, making raw lora_A matrices mutually orthogonal (pairwise cos ≈ 0.12)
    even for functionally similar adapters — breaking weight-space diffusion (Gate P1
    held-out cos 0.12, near-random).  The only intentional variation is the data window.
    """
    if not _TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"train_task_lora.py not found: {_TRAIN_SCRIPT}")

    start = idx * args.examples_per
    end   = start + args.examples_per
    split = f"train[{start}:{end}]"

    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--task",           args.task,
        "--base",           args.base,
        "--dataset",        args.dataset,
        "--split",          split,
        "--steps",          str(args.steps),
        "--lora-init-seed", str(args.lora_init_seed),   # identical for ALL adapters
        "--out",            str(adapter_dir),
    ]
    if args.no_4bit:
        cmd.append("--no-4bit")
    # Full-bf16 BitNet at the default 4096 seq-len thrashes the 12GB card (~1 step/min).
    # Cap seq-len and enable grad checkpointing to keep VRAM in budget and training fast.
    cmd += ["--max-seq-len", "512", "--grad-checkpoint"]

    seed_str = env.get("PYTHONHASHSEED", "unset")
    print(
        f"\n[adapter {idx:03d}] Training | split={split} | "
        f"PYTHONHASHSEED={seed_str} (shared) | lora_init_seed={args.lora_init_seed} (shared)"
    )
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"train_task_lora.py failed for adapter {idx} (exit code {result.returncode})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Bootstrap a LoRA adapter population for weight-space diffusion Gate P1. "
            "Calls scripts/lora/train_task_lora.py in a loop, extracts lora_A vectors, "
            "and saves population.npy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--n", type=int, default=16,
        help="Number of adapters to train.",
    )
    ap.add_argument(
        "--task", default="instruction",
        choices=["instruction", "tooluse", "coding"],
        help="Task type passed to train_task_lora.py. Same task for all adapters.",
    )
    ap.add_argument(
        "--layer", type=int, default=0,
        help="Transformer layer index to extract lora_A from.",
    )
    ap.add_argument(
        "--proj", default="gate_proj",
        help="MLP projection name to extract (gate_proj, up_proj, down_proj).",
    )
    ap.add_argument(
        "--rank", type=int, default=16,
        help="LoRA rank. train_task_lora.py always uses rank=16; keep in sync.",
    )
    ap.add_argument(
        "--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HuggingFace base model ID.",
    )
    ap.add_argument(
        "--dataset", default="HuggingFaceH4/no_robots",
        help="HuggingFace dataset ID.",
    )
    ap.add_argument(
        "--steps", type=int, default=200,
        help="Training steps per adapter (passed to train_task_lora.py).",
    )
    ap.add_argument(
        "--examples-per", type=int, default=200, dest="examples_per",
        help="Examples per data window. Adapter i uses train[i*N:(i+1)*N].",
    )
    ap.add_argument(
        "--out",
        default=str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot"),
        help="Output directory for adapter dirs, vec files, population.npy, meta.json.",
    )
    ap.add_argument(
        "--seed-base", type=int, default=42, dest="seed_base",
        help="Base seed for PYTHONHASHSEED (shared by all adapters; per-adapter "
             "variation was the root cause of gauge misalignment and has been removed). "
             "Kept for backward-compat but no longer incremented per-adapter.",
    )
    ap.add_argument(
        "--lora-init-seed", type=int, default=0, dest="lora_init_seed",
        help="Fixed torch seed forwarded to --lora-init-seed in train_task_lora.py. "
             "MUST be identical for every adapter in the population so all lora_A "
             "matrices share the same gauge (basis). Data variation comes from the "
             "sliding data window (--examples-per), NOT from this seed.",
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Device hint (cuda for real run). train_task_lora.py targets GPU 0.",
    )
    ap.add_argument(
        "--no-4bit", action="store_true",
        help="Pass --no-4bit to train_task_lora.py (bf16 full precision; ~4× more VRAM).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("pdiff_bootstrap: LoRA population builder")
    print("=" * 60)
    print(f"  n={args.n}  task={args.task}  base={args.base}")
    print(f"  dataset={args.dataset}  layer={args.layer}  proj={args.proj}  rank={args.rank}")
    print(f"  steps={args.steps}  examples_per={args.examples_per}")
    print(f"  seed_base={args.seed_base} (PYTHONHASHSEED, shared)  lora_init_seed={args.lora_init_seed} (shared — gauge fix)")
    print(f"  out={out_dir}")
    print(f"  train_script={_TRAIN_SCRIPT}")

    base_env = os.environ.copy()
    trained_new = 0
    skipped     = 0

    # Gauge fix: PYTHONHASHSEED is now SHARED (seed_base, not seed_base+i).
    # Previously incrementing per-adapter was intended to vary training but had no
    # effect on torch RNG (which draws from OS entropy), so it was pure noise that
    # could only hurt reproducibility.  The data window (split) provides all intended
    # variation between adapters.
    shared_env = {**base_env, "PYTHONHASHSEED": str(args.seed_base)}

    for i in range(args.n):
        # --- Skip if already extracted ---
        if is_done(out_dir, i):
            print(f"[adapter {i:03d}] SKIP — vec file already exists.")
            skipped += 1
            continue

        adapter_dir = out_dir / f"adapter_{i:03d}"
        env = shared_env

        # --- Train ---
        run_train(i, adapter_dir, args, env)

        # --- Extract lora_A vector ---
        print(f"[adapter {i:03d}] Extracting lora_A from {adapter_dir} ...")
        vec = extract_lora_vec(adapter_dir, args.layer, args.proj, args.rank)
        print(f"  shape={vec.shape}  dtype={vec.dtype}  l2_norm={float(np.linalg.norm(vec)):.4f}")

        # --- Save (enables resume on next run) ---
        save_vec(out_dir, i, vec)
        print(f"  Saved -> {vec_path(out_dir, i)}")
        trained_new += 1

    # --- Summary ---
    done_idxs = [i for i in range(args.n) if is_done(out_dir, i)]
    print(f"\nDone: {trained_new} trained, {skipped} skipped. "
          f"Complete vecs: {len(done_idxs)}/{args.n}")

    if not done_idxs:
        print("No vectors collected. Nothing to save.")
        return

    # --- Build population.npy ---
    population = build_population(out_dir, done_idxs)   # [N, D]
    D = population.shape[1]

    meta = {
        "task":            args.task,
        "base":            args.base,
        "dataset":         args.dataset,
        "layer":           args.layer,
        "proj":            args.proj,
        "lora_part":       "A",
        "rank":            args.rank,
        "N":               len(done_idxs),
        "D":               D,
        "done_idxs":       done_idxs,
        "seed_base":       args.seed_base,
        "lora_init_seed":  args.lora_init_seed,
        "steps":           args.steps,
        "examples_per":    args.examples_per,
        "gauge_fixed":     True,
        "note": (
            f"lora_A of {args.proj} at layer {args.layer}; "
            f"shape [{args.rank}, {D // args.rank}] -> flattened to {D}-D. "
            f"D={D} == BLOCK_DIM={16*2560} for rank=16, hidden=2560. "
            f"Gauge-fixed: all adapters share lora_init_seed={args.lora_init_seed} "
            f"and PYTHONHASHSEED={args.seed_base}; variation is data window only."
        ),
    }
    save_population(out_dir, population, meta)
    print(f"\nPopulation shape: {population.shape}  (D={D})")
    print(f"Next step: run pdiff_gate_p1_adapters.py --population {out_dir / POPULATION_NPY}")


if __name__ == "__main__":
    main()
