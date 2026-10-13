"""pdiff_func_extract.py — Re-extract MLP LoRA module populations from fixed-init adapters.

For each combination of (layer, proj, mat), loads the weight tensor from all
N adapter dirs and stacks them into a [N, D] float32 population array, saved
under <out_dir>/<module_id>.npy.  Also writes a manifest JSON recording the
shape, full PEFT key fragment, and file path for every module.

Module ID format: l{layer:02d}_{proj}_{mat}  (e.g. l00_gate_proj_lora_A)

This re-uses find_lora_key logic from pdiff_bootstrap so the key search is
robust to PEFT version differences in key prefixes.

CPU-only.  Never retrains adapters.

Usage (small subset for validation):
    python pdiff_func_extract.py \\
        --adapter-dir .docs/diffusion/pdiff_boot_fixedinit \\
        --out-dir .docs/diffusion/pdiff_boot_fixedinit/modpops \\
        --layers 0,15 --projs gate_proj --mats lora_A,lora_B

Full run (all 180 MLP modules):
    python pdiff_func_extract.py \\
        --adapter-dir .docs/diffusion/pdiff_boot_fixedinit \\
        --out-dir .docs/diffusion/pdiff_boot_fixedinit/modpops
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from safetensors import safe_open
except ImportError as exc:
    raise ImportError("safetensors not installed.  Run: pip install safetensors") from exc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent

_DEFAULT_ADAPTER_DIR = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit")
_DEFAULT_OUT_DIR     = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit" / "modpops")

DEFAULT_LAYERS = list(range(30))
DEFAULT_PROJS  = ["gate_proj", "up_proj", "down_proj"]
DEFAULT_MATS   = ["lora_A", "lora_B"]

MANIFEST_NAME  = "manifest.json"

# ---------------------------------------------------------------------------
# Key lookup
# ---------------------------------------------------------------------------

def find_module_key(sf_handle, layer: int, proj: str, mat: str) -> str:
    """Find the safetensors key for a given (layer, proj, mat) combination.

    Searches for any key containing 'layers.{layer}.mlp.{proj}.{mat}' and
    ending with '.weight', robust to PEFT prefix differences.

    Args:
        sf_handle: an open safetensors.safe_open handle.
        layer:     Transformer layer index.
        proj:      MLP projection (gate_proj, up_proj, down_proj).
        mat:       Weight matrix (lora_A, lora_B).

    Returns:
        The shortest matching key string.

    Raises:
        KeyError: if no matching key exists.
    """
    fragment = f"layers.{layer}.mlp.{proj}.{mat}"
    all_keys = list(sf_handle.keys())
    candidates = [k for k in all_keys if fragment in k and k.endswith(".weight")]
    if not candidates:
        raise KeyError(
            f"No key found for layer={layer} proj={proj} mat={mat}.  "
            f"Checked {len(all_keys)} keys.  Fragment: {fragment!r}  "
            f"First 8 keys: {all_keys[:8]}"
        )
    candidates.sort(key=len)  # prefer shortest (fewest wrapper prefixes)
    return candidates[0]


def extract_module_vec(adapter_dir: Path, layer: int, proj: str, mat: str) -> tuple:
    """Load one LoRA module tensor from an adapter dir, return (flat_vec, shape).

    Args:
        adapter_dir: Path to the PEFT adapter directory.
        layer:       Transformer layer index.
        proj:        MLP projection name.
        mat:         'lora_A' or 'lora_B'.

    Returns:
        (vec, shape) where vec is float32 ndarray [D] and shape is the original
        tensor shape as a list.
    """
    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        key = find_module_key(f, layer, proj, mat)
        t   = f.get_tensor(key).to(torch.float32)

    shape = list(t.shape)
    vec   = t.reshape(-1).numpy().copy()
    return vec, shape


# ---------------------------------------------------------------------------
# Adapter discovery
# ---------------------------------------------------------------------------

def discover_adapters(adapter_dir: Path) -> list:
    """Return sorted list of adapter sub-directories (adapter_000, adapter_001, …)."""
    dirs = sorted(
        p for p in adapter_dir.iterdir()
        if p.is_dir() and p.name.startswith("adapter_")
        and (p / "adapter_model.safetensors").exists()
    )
    return dirs


# ---------------------------------------------------------------------------
# Module ID
# ---------------------------------------------------------------------------

def module_id(layer: int, proj: str, mat: str) -> str:
    """Canonical short identifier for a (layer, proj, mat) combination.

    Format: l{layer:02d}_{proj}_{mat}  →  e.g. l00_gate_proj_lora_A
    """
    return f"l{layer:02d}_{proj}_{mat}"


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def run_extract(
    adapter_dir: Path,
    out_dir: Path,
    layers: list,
    projs: list,
    mats: list,
) -> dict:
    """Extract per-module populations from all adapters in adapter_dir.

    For every (layer, proj, mat) triple, stacks vectors from all N adapters
    into a [N, D] float32 array and saves it to out_dir/<module_id>.npy.

    Returns the manifest dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_dirs = discover_adapters(adapter_dir)
    N = len(adapter_dirs)
    if N == 0:
        raise RuntimeError(f"No adapter dirs found under {adapter_dir}")
    print(f"Found {N} adapters: {[p.name for p in adapter_dirs]}")

    manifest = {
        "adapter_dir":  str(adapter_dir),
        "n_adapters":   N,
        "adapter_names": [p.name for p in adapter_dirs],
        "layers":       layers,
        "projs":        projs,
        "mats":         mats,
        "modules":      {},
    }

    total = len(layers) * len(projs) * len(mats)
    done  = 0

    for layer in layers:
        for proj in projs:
            for mat in mats:
                mid       = module_id(layer, proj, mat)
                npy_file  = f"{mid}.npy"
                out_path  = out_dir / npy_file

                print(f"[{done+1}/{total}]  {mid} ...", end="  ", flush=True)

                vecs   = []
                shapes = []
                for ad in adapter_dirs:
                    vec, shape = extract_module_vec(ad, layer, proj, mat)
                    vecs.append(vec)
                    shapes.append(shape)

                # Sanity: all adapters should give the same shape
                assert all(s == shapes[0] for s in shapes), (
                    f"Shape mismatch across adapters for {mid}: {shapes}"
                )

                pop = np.stack(vecs).astype(np.float32)   # [N, D]
                assert pop.shape == (N, vecs[0].shape[0])

                np.save(str(out_path), pop)
                D = pop.shape[1]
                print(f"shape={pop.shape}  D={D}  -> {npy_file}")

                manifest["modules"][mid] = {
                    "module_id":          mid,
                    "layer":              layer,
                    "proj":               proj,
                    "mat":                mat,
                    "peft_key_fragment":  f"layers.{layer}.mlp.{proj}.{mat}",
                    "tensor_shape":       shapes[0],
                    "D":                  D,
                    "npy_file":           npy_file,
                }
                done += 1

    # Write manifest
    manifest_path = out_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print()
    print(f"Extracted {done} modules from {N} adapters.")
    print(f"Manifest -> {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Re-extract MLP LoRA module populations from fixed-init adapter dirs. "
            "Produces [N, D] .npy population files and a manifest JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--adapter-dir", default=_DEFAULT_ADAPTER_DIR,
        help="Root directory containing adapter_000/ … adapter_015/ subdirs.",
    )
    ap.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help="Output directory for .npy population files and manifest.json.",
    )
    ap.add_argument(
        "--layers",
        default=",".join(str(l) for l in DEFAULT_LAYERS),
        help="Comma-separated list of layer indices to extract (e.g. '0,15' or '0,1,...,29').",
    )
    ap.add_argument(
        "--projs",
        default=",".join(DEFAULT_PROJS),
        help="Comma-separated list of MLP projections (gate_proj, up_proj, down_proj).",
    )
    ap.add_argument(
        "--mats",
        default=",".join(DEFAULT_MATS),
        help="Comma-separated list of LoRA matrices (lora_A, lora_B).",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    projs  = [x.strip() for x in args.projs.split(",")  if x.strip()]
    mats   = [x.strip() for x in args.mats.split(",")   if x.strip()]

    adapter_dir = Path(args.adapter_dir)
    out_dir     = Path(args.out_dir)

    if not adapter_dir.exists():
        print(f"ERROR: adapter-dir not found: {adapter_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("pdiff_func_extract: MLP LoRA module population extractor")
    print("=" * 60)
    print(f"  adapter-dir : {adapter_dir}")
    print(f"  out-dir     : {out_dir}")
    print(f"  layers      : {layers}")
    print(f"  projs       : {projs}")
    print(f"  mats        : {mats}")
    print(f"  modules     : {len(layers) * len(projs) * len(mats)} total")
    print()

    run_extract(adapter_dir, out_dir, layers, projs, mats)


if __name__ == "__main__":
    main()
