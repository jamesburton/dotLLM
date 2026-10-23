"""
Pair-wise compare two dotLLM tensor dump directories.

Usage:
    python compare_dumps.py <ref_dir> <buggy_dir>

Lists files in name (post-index-stripped) order and prints which tensor first
diverges (and by how much).
"""
import sys
import struct
from pathlib import Path
import numpy as np


def read_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = []
        if rank >= 1: shape.append(d0)
        if rank >= 2: shape.append(d1)
        if rank >= 3: shape.append(d2)
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data, tuple(shape)


def name_of(p: Path) -> str:
    # strip leading "NNNNN_" index prefix
    return p.stem.split("_", 1)[1] if "_" in p.stem else p.stem


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_dumps.py <ref_dir> <buggy_dir>")
        sys.exit(1)
    ref_dir, buggy_dir = Path(sys.argv[1]), Path(sys.argv[2])

    ref_files = sorted(ref_dir.glob("*.bin"))
    buggy_files = sorted(buggy_dir.glob("*.bin"))

    # Pair by stripped tensor-name (the leading 5-digit index may shift if
    # the dump order changed between runs).
    ref_by_name = {name_of(p): p for p in ref_files}
    buggy_by_name = {name_of(p): p for p in buggy_files}

    common = sorted(set(ref_by_name) & set(buggy_by_name), key=lambda n: ref_files[
        [name_of(p) for p in ref_files].index(n)
    ])

    only_ref = sorted(set(ref_by_name) - set(buggy_by_name))
    only_buggy = sorted(set(buggy_by_name) - set(ref_by_name))
    if only_ref:
        print(f"Only in REF: {len(only_ref)} tensors, e.g. {only_ref[:5]}")
    if only_buggy:
        print(f"Only in BUGGY: {len(only_buggy)} tensors, e.g. {only_buggy[:5]}")

    print(f"=== {len(common)} common tensors")
    first_divergence = None
    for n in common:
        ref_data, ref_shape = read_bin(ref_by_name[n])
        bug_data, bug_shape = read_bin(buggy_by_name[n])
        if ref_shape != bug_shape:
            print(f"  [SHAPE] {n}: ref={ref_shape} bug={bug_shape}")
            if first_divergence is None: first_divergence = n
            continue
        if ref_data.size == 0:
            continue
        diff = np.abs(ref_data - bug_data)
        max_abs = float(diff.max())
        # Use threshold based on quant rounding error band
        ref_abs = np.abs(ref_data) + 1e-9
        max_rel = float((diff / ref_abs).max())
        rms = float(np.sqrt((diff.astype(np.float64) ** 2).mean()))
        ref_rms = float(np.sqrt((ref_data.astype(np.float64) ** 2).mean()))
        rel_rms = rms / (ref_rms + 1e-12)
        # Heuristic: numerical-noise band is ~1e-3 abs / 1% rel_rms for Q6_K.
        status = "PASS" if (max_abs < 1e-2 and rel_rms < 5e-3) else "FAIL"
        print(f"  [{status}] {n} shape={ref_shape}: max_abs={max_abs:.4e} max_rel={max_rel:.4e} rms={rms:.4e} ref_rms={ref_rms:.4e} rel_rms={rel_rms:.4e}")
        if status == "FAIL" and first_divergence is None:
            first_divergence = n

    print()
    if first_divergence is not None:
        print(f"FIRST DIVERGENCE: {first_divergence}")
    else:
        print("All tensors match within tolerance.")


if __name__ == "__main__":
    main()
