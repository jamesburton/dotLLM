"""pdiff_func_assemble.py — Assemble PEFT adapter dirs from generated/reconstructed modules.

Takes a real adapter as scaffold and produces two new PEFT adapter directories:

  gen_adapter/   — generated module tensors swapped in (diffusion path)
  recon_adapter/ — AE-reconstructed module tensors swapped in (AE path)

Every key NOT in the generated/recon module set is copied verbatim from the
real adapter so both outputs are valid PEFT adapters loadable by
PeftModel.from_pretrained.

Non-safetensors files (adapter_config.json, tokenizer files, etc.) are also
copied from the real adapter directory.

CPU-only.  Does not run any model inference.

Usage:
    python pdiff_func_assemble.py \\
        --real-adapter   .docs/diffusion/pdiff_boot_fixedinit/adapter_015 \\
        --gen-dir        .docs/diffusion/pdiff_func_out/gen \\
        --recon-dir      .docs/diffusion/pdiff_func_out/recon \\
        --manifest       .docs/diffusion/pdiff_boot_fixedinit/modpops/manifest.json \\
        --out-dir        .docs/diffusion/pdiff_func_out
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as st_save_file
except ImportError as exc:
    raise ImportError("safetensors not installed.  Run: pip install safetensors") from exc

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT  = _SCRIPT_DIR.parent.parent

_DEFAULT_REAL_ADAPTER = str(
    _REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit" / "adapter_015"
)
_DEFAULT_GEN_DIR  = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_func_out" / "gen")
_DEFAULT_RECON_DIR = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_func_out" / "recon")
_DEFAULT_MANIFEST = str(
    _REPO_ROOT / ".docs" / "diffusion" / "pdiff_boot_fixedinit" / "modpops" / "manifest.json"
)
_DEFAULT_OUT_DIR  = str(_REPO_ROOT / ".docs" / "diffusion" / "pdiff_func_out")


# ---------------------------------------------------------------------------
# Key / module helpers
# ---------------------------------------------------------------------------

def peft_key_for_module(module_info: dict) -> str:
    """Reconstruct the full PEFT safetensors key for a module (adapter A or B weight).

    The key fragment stored in the manifest (e.g. 'layers.0.mlp.gate_proj.lora_A')
    is searched against the real adapter keys to find the exact match, which may
    carry a prefix like 'base_model.model.model...'.
    """
    return module_info["peft_key_fragment"]   # returned for use in find_peft_key below


def find_peft_key(all_keys: list, module_info: dict) -> str:
    """Find the full PEFT key from the key list using the module's key fragment.

    Searches for a key containing the fragment (e.g. 'layers.0.mlp.gate_proj.lora_A')
    and ending with '.weight'.  Returns the shortest match.

    Raises:
        KeyError: if no matching key is found.
    """
    fragment = module_info["peft_key_fragment"]   # e.g. 'layers.0.mlp.gate_proj.lora_A'
    candidates = [k for k in all_keys if fragment in k and k.endswith(".weight")]
    if not candidates:
        raise KeyError(
            f"No PEFT key found for fragment {fragment!r} among {len(all_keys)} keys."
        )
    candidates.sort(key=len)   # prefer shortest (fewest wrapper prefixes)
    return candidates[0]


# ---------------------------------------------------------------------------
# Load real adapter weights
# ---------------------------------------------------------------------------

def load_real_adapter(adapter_dir: Path) -> dict:
    """Load all tensors from a PEFT adapter safetensors file as a dict.

    Returns {key: torch.Tensor (float32 CPU)}.
    """
    st_path = adapter_dir / "adapter_model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")

    tensors = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(torch.float32)
    return tensors


# ---------------------------------------------------------------------------
# Build replacement maps
# ---------------------------------------------------------------------------

def build_replacement_map(
    real_keys:   list,
    modules:     dict,
    npy_dir:     Path,
) -> dict:
    """Build a {peft_key: tensor} replacement map from .npy files in npy_dir.

    For each module entry in the manifest, if a matching .npy file exists under
    npy_dir, the corresponding PEFT key gets a replacement tensor reshaped to the
    original tensor_shape.

    Args:
        real_keys:   List of PEFT keys from the real adapter.
        modules:     manifest['modules'] dict.
        npy_dir:     Directory containing <module_id>.npy files (gen/ or recon/).

    Returns:
        {peft_key: torch.Tensor (float32)} for every module that has a .npy file.
    """
    replacements = {}
    for mid, info in modules.items():
        npy_path = npy_dir / f"{mid}.npy"
        if not npy_path.exists():
            continue   # module not generated (e.g. subset run) — keep original

        # Find the matching full PEFT key in the real adapter
        try:
            peft_key = find_peft_key(real_keys, info)
        except KeyError as exc:
            print(f"  WARNING: {exc}  (skipping {mid})")
            continue

        # Load flat vec and reshape to original tensor shape
        flat_vec = np.load(str(npy_path)).astype(np.float32)
        shape    = info["tensor_shape"]                              # e.g. [16, 2560]
        tensor   = torch.from_numpy(flat_vec).reshape(shape)
        replacements[peft_key] = tensor

    return replacements


# ---------------------------------------------------------------------------
# Write PEFT adapter dir
# ---------------------------------------------------------------------------

def write_peft_adapter(
    out_dir:      Path,
    real_tensors: dict,
    replacements: dict,
    real_adapter: Path,
    label:        str,
) -> dict:
    """Write a new PEFT adapter directory.

    Args:
        out_dir:      Destination directory (created if absent).
        real_tensors: All tensors from the real adapter {key: tensor}.
        replacements: Tensors to swap in {key: tensor}.
        real_adapter: Real adapter dir to copy non-safetensors files from.
        label:        Short label for logging (e.g. 'gen' or 'recon').

    Returns:
        Summary dict with key/shape statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build merged tensor dict
    merged = {}
    swapped = 0
    for key, tensor in real_tensors.items():
        if key in replacements:
            merged[key] = replacements[key]
            swapped += 1
        else:
            merged[key] = tensor

    # Save safetensors
    st_path = out_dir / "adapter_model.safetensors"
    st_save_file(merged, str(st_path))

    # Copy non-safetensors files (adapter_config.json, tokenizer files, etc.)
    copied_files = []
    for src in real_adapter.iterdir():
        if src.name == "adapter_model.safetensors":
            continue
        if src.is_file():
            dst = out_dir / src.name
            shutil.copy2(str(src), str(dst))
            copied_files.append(src.name)

    print(
        f"  [{label}] {st_path.name}: {len(merged)} keys  "
        f"({swapped} swapped, {len(merged)-swapped} original)  "
        f"+ {len(copied_files)} side files"
    )

    return {
        "out_dir":       str(out_dir),
        "n_keys":        len(merged),
        "n_swapped":     swapped,
        "n_original":    len(merged) - swapped,
        "side_files":    copied_files,
        "safetensors":   str(st_path),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_adapter(out_dir: Path, real_tensors: dict) -> dict:
    """Verify the written adapter has the same keys and shapes as the real adapter.

    Returns a validation summary dict.
    """
    st_path = out_dir / "adapter_model.safetensors"
    if not st_path.exists():
        return {"status": "FAIL", "reason": "safetensors file missing"}

    written = {}
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            written[key] = tuple(t.shape)

    real_shapes = {k: tuple(v.shape) for k, v in real_tensors.items()}

    missing   = set(real_shapes) - set(written)
    extra     = set(written)     - set(real_shapes)
    misshapen = {k for k in (set(written) & set(real_shapes)) if written[k] != real_shapes[k]}

    ok = not missing and not extra and not misshapen
    return {
        "status":    "PASS" if ok else "FAIL",
        "n_keys":    len(written),
        "missing":   sorted(missing),
        "extra":     sorted(extra),
        "misshapen": sorted(misshapen),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_assemble(
    real_adapter: Path,
    gen_dir:      Path,
    recon_dir:    Path,
    manifest_path: Path,
    out_dir:      Path,
    held_out_idx: int,
) -> dict:
    """Assemble gen_adapter/ and recon_adapter/ under out_dir."""

    # Load manifest
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    modules = manifest["modules"]

    # Load real adapter tensors
    print(f"Loading real adapter: {real_adapter}")
    real_tensors = load_real_adapter(real_adapter)
    real_keys    = list(real_tensors.keys())
    print(f"  {len(real_keys)} keys loaded")

    # Build replacement maps
    print(f"\nBuilding gen replacements from {gen_dir} ...")
    gen_map   = build_replacement_map(real_keys, modules, gen_dir)
    print(f"  {len(gen_map)} keys will be replaced")

    print(f"Building recon replacements from {recon_dir} ...")
    recon_map = build_replacement_map(real_keys, modules, recon_dir)
    print(f"  {len(recon_map)} keys will be replaced")

    # Write gen_adapter/
    gen_out   = out_dir / "gen_adapter"
    recon_out = out_dir / "recon_adapter"
    print()

    gen_summary   = write_peft_adapter(gen_out,   real_tensors, gen_map,   real_adapter, "gen")
    recon_summary = write_peft_adapter(recon_out, real_tensors, recon_map, real_adapter, "recon")

    # Validate
    print()
    gen_val   = validate_adapter(gen_out,   real_tensors)
    recon_val = validate_adapter(recon_out, real_tensors)
    print(f"  Validation gen   : {gen_val['status']}  ({gen_val['n_keys']} keys)")
    print(f"  Validation recon : {recon_val['status']}  ({recon_val['n_keys']} keys)")

    if gen_val["status"] != "PASS":
        print(f"    gen missing:   {gen_val['missing'][:5]}")
        print(f"    gen extra:     {gen_val['extra'][:5]}")
        print(f"    gen misshapen: {gen_val['misshapen'][:5]}")
    if recon_val["status"] != "PASS":
        print(f"    recon missing:   {recon_val['missing'][:5]}")
        print(f"    recon extra:     {recon_val['extra'][:5]}")
        print(f"    recon misshapen: {recon_val['misshapen'][:5]}")

    result = {
        "gen_adapter":   gen_summary,
        "recon_adapter": recon_summary,
        "gen_validation":   gen_val,
        "recon_validation": recon_val,
        "held_out_idx":     held_out_idx,
        "real_adapter":     str(real_adapter),
    }

    # Save assembly summary
    summary_path = out_dir / "assemble_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSummary -> {summary_path}")
    print(f"gen_adapter   -> {gen_out}")
    print(f"recon_adapter -> {recon_out}")

    return result


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "Assemble PEFT adapter dirs from diffusion-generated and AE-reconstructed "
            "module tensors.  Produces gen_adapter/ and recon_adapter/ under --out-dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--real-adapter", default=_DEFAULT_REAL_ADAPTER,
        help="Path to the held-out real PEFT adapter directory (scaffold).",
    )
    ap.add_argument(
        "--gen-dir", default=_DEFAULT_GEN_DIR,
        help="Directory containing gen/<module_id>.npy files (from pdiff_func_generate.py).",
    )
    ap.add_argument(
        "--recon-dir", default=_DEFAULT_RECON_DIR,
        help="Directory containing recon/<module_id>.npy files (from pdiff_func_generate.py).",
    )
    ap.add_argument(
        "--manifest", default=_DEFAULT_MANIFEST,
        help="Path to manifest.json (from pdiff_func_extract.py).",
    )
    ap.add_argument(
        "--out-dir", default=_DEFAULT_OUT_DIR,
        help="Output directory; gen_adapter/ and recon_adapter/ created inside.",
    )
    ap.add_argument(
        "--held-out-idx", type=int, default=15,
        help="Held-out adapter index (informational only for the summary).",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    real_adapter  = Path(args.real_adapter)
    gen_dir       = Path(args.gen_dir)
    recon_dir     = Path(args.recon_dir)
    manifest_path = Path(args.manifest)
    out_dir       = Path(args.out_dir)

    for p, label in [
        (real_adapter,  "--real-adapter"),
        (manifest_path, "--manifest"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    if not gen_dir.exists():
        print(f"ERROR: --gen-dir not found: {gen_dir}", file=sys.stderr)
        print("Run pdiff_func_generate.py first.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("pdiff_func_assemble: PEFT adapter assembler")
    print("=" * 60)
    print(f"  real-adapter  : {real_adapter}")
    print(f"  gen-dir       : {gen_dir}")
    print(f"  recon-dir     : {recon_dir}")
    print(f"  manifest      : {manifest_path}")
    print(f"  out-dir       : {out_dir}")
    print(f"  held-out-idx  : {args.held_out_idx}")
    print()

    result = run_assemble(
        real_adapter=real_adapter,
        gen_dir=gen_dir,
        recon_dir=recon_dir,
        manifest_path=manifest_path,
        out_dir=out_dir,
        held_out_idx=args.held_out_idx,
    )

    overall = (
        result["gen_validation"]["status"] == "PASS"
        and result["recon_validation"]["status"] == "PASS"
    )
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
