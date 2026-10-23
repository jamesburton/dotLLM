"""Element-by-element diff between llama.cpp and dotLLM binary tensor dumps.

Maps llama.cpp tensor names (e.g. attn_norm-0) to dotLLM names (e.g. blk.0.attn_norm)
and compares raw float32 contents.

Usage:
    python bin_diff.py <llama_bin_dir> <dotllm_bin_dir>
"""
import sys, struct, re
from pathlib import Path
import numpy as np


def read_bin(path):
    with open(path, "rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data, tuple(shape)


# llama.cpp name -> dotLLM name pattern (with {L} for layer index)
NAME_MAP = [
    (r"^attn_norm-(?P<L>\d+)$", "blk.{L}.attn_norm"),
    (r"^linear_attn_qkv_mixed-(?P<L>\d+)$", "blk.{L}.linear_attn_qkv_mixed"),
    (r"^z-(?P<L>\d+)$", "blk.{L}.z"),
    (r"^alpha-(?P<L>\d+)$", "blk.{L}.alpha_proj"),
    (r"^conv_output_silu-(?P<L>\d+)$", "blk.{L}.conv_output_silu"),
    (r"^q_conv-(?P<L>\d+)$", "blk.{L}.q_conv"),
    (r"^k_conv-(?P<L>\d+)$", "blk.{L}.k_conv"),
    (r"^v_conv-(?P<L>\d+)$", "blk.{L}.v_conv"),
    (r"^q_conv_predelta-(?P<L>\d+)$", "blk.{L}.q_conv_predelta"),
    (r"^k_conv_predelta-(?P<L>\d+)$", "blk.{L}.k_conv_predelta"),
    (r"^attn_output-(?P<L>\d+)$", "blk.{L}.attn_output"),
    (r"^final_output-(?P<L>\d+)$", "blk.{L}.final_output"),
    (r"^linear_attn_out-(?P<L>\d+)$", "blk.{L}.linear_attn_out"),
    (r"^attn_post_norm-(?P<L>\d+)$", "blk.{L}.attn_post_norm"),
    (r"^l_out-(?P<L>\d+)$", "blk.{L}.l_out"),
    (r"^ffn_out-(?P<L>\d+)$", "blk.{L}.ffn_out"),
    (r"^Qcur_full-(?P<L>\d+)$", "blk.{L}.fa_qg"),
    (r"^Qcur-(?P<L>\d+)$", "blk.{L}.fa_q_postrope"),
    (r"^Kcur-(?P<L>\d+)$", "blk.{L}.fa_k_postrope"),
    (r"^Vcur-(?P<L>\d+)$", "blk.{L}.fa_v"),
    (r"^attn_pregate-(?P<L>\d+)$", "blk.{L}.fa_attnout_pregate"),
    (r"^attn_gated-(?P<L>\d+)$", "blk.{L}.fa_attnout_postgate"),
    (r"^result_norm$", "result_norm"),
    (r"^result_output$", "result_output"),
]


def strip_index(filename):
    """'00012_linear_attn_qkv_mixed-0.bin' -> 'linear_attn_qkv_mixed-0'"""
    stem = Path(filename).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def map_to_dotllm(llama_name):
    for pat, tpl in NAME_MAP:
        m = re.match(pat, llama_name)
        if m:
            return tpl.format(**m.groupdict())
    return None


def find_dotllm_dump(dump_dir, dotllm_name):
    matches = sorted(dump_dir.glob(f"*_{dotllm_name}.bin"))
    return matches[0] if matches else None


def compare_arrays(a, b):
    """Return (max_abs, max_rel, rel_rms, snr_db, cosine)."""
    if a.size != b.size:
        return None
    a = a.astype(np.float64); b = b.astype(np.float64)
    err = a - b
    max_abs = float(np.abs(err).max())
    denom_abs = np.abs(a) + 1e-9
    max_rel = float((np.abs(err) / denom_abs).max())
    a_pow = float((a ** 2).sum()); err_pow = float((err ** 2).sum())
    snr = float('inf') if err_pow == 0 else 10.0 * np.log10(a_pow / err_pow)
    rel_rms = np.sqrt(err_pow / max(a_pow, 1e-30))
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb + 1e-30))
    return max_abs, max_rel, float(rel_rms), snr, cos


def main():
    llama_dir = Path(sys.argv[1])
    dotllm_dir = Path(sys.argv[2])

    llama_files = sorted(llama_dir.glob("*.bin"))
    print(f"Loaded {len(llama_files)} llama.cpp tensors")

    pairs = []
    first_divergence = None
    matched = 0
    diverged = 0
    skipped = 0

    for lf in llama_files:
        lname = strip_index(lf.name)
        dotllm_name = map_to_dotllm(lname)
        if dotllm_name is None:
            skipped += 1
            continue
        df = find_dotllm_dump(dotllm_dir, dotllm_name)
        if df is None:
            continue
        la, lshape = read_bin(lf)
        da, dshape = read_bin(df)
        if la.size != da.size:
            print(f"  [SHAPE MISMATCH] {lname} llama={lshape} dotLLM={dshape}")
            continue
        m = compare_arrays(la, da)
        max_abs, max_rel, rel_rms, snr, cos = m
        # SNR > 50 dB means rel_rms < ~0.003 — that's well below quant noise floor.
        ok = (snr > 50 and cos > 0.9999) or (max_abs < 1e-5)
        if ok:
            matched += 1
            # print(f"  [OK] {lname} -> {dotllm_name} SNR={snr:.1f} cos={cos:.6f}")
        else:
            diverged += 1
            print(f"  [DIFF] {lname} -> {dotllm_name} (shape={lshape})")
            print(f"         max_abs={max_abs:.4e} max_rel={max_rel:.4e} rel_rms={rel_rms:.4e} SNR={snr:.1f} dB cos={cos:.6f}")
            if first_divergence is None:
                first_divergence = lname
            if diverged >= 8:
                print(f"  ... stopping at 8 divergences")
                break

    print()
    print("=== SUMMARY ===")
    print(f"  Matched: {matched}")
    print(f"  Diverged: {diverged}")
    print(f"  Skipped (unmapped): {skipped}")
    if first_divergence:
        print(f"  FIRST DIVERGENCE: {first_divergence}")


if __name__ == "__main__":
    main()
