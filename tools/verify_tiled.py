"""Verify llama.cpp uses TILED (modulo) head broadcasting."""
import struct
import numpy as np
from pathlib import Path


def read_bin(p):
    with open(p, 'rb') as f:
        h = f.read(16)
        r, d0, d1, d2 = struct.unpack('<iiii', h)
        s = [d for d in [d0, d1, d2][:r]]
        n = int(np.prod(s)) if s else 0
        return np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(s)


q_predelta = read_bin('C:/dotllm_qwen35_hi/00012_blk.0.q_conv_predelta.bin')  # [1, 16, 128]
k_predelta = read_bin('C:/dotllm_qwen35_hi/00013_blk.0.k_conv_predelta.bin')  # [1, 16, 128]
v_conv = read_bin('C:/dotllm_qwen35_hi/00011_blk.0.v_conv.bin')  # [1, 32, 128]
beta_sig = read_bin('C:/dotllm_qwen35_hi/00007_blk.0.beta_sigmoid.bin').flatten()  # [32]
llama_attn_out = read_bin('C:/llama_hi_bin/00046_attn_output-0.bin')  # [32, 128]

NVHead = 32
NKHead = 16
DState = 128

t = 0
print("Compare TILED (vh%NKHead) vs INTERLEAVED (vh//(NVHead/NKHead)) head mapping vs llama.cpp:")
print()
print(f"{'vh':>3} {'kh_tiled':>9} {'kh_interleaved':>15} {'llama_rms':>11} {'tiled_rms':>11} {'inter_rms':>11} {'tiled_match':>12} {'inter_match':>12}")
for vh in range(32):
    v = v_conv[t, vh].astype(np.float64)
    beta = float(beta_sig[vh])
    actual_llama = llama_attn_out[vh].astype(np.float64)

    # TILED: kh = vh % NKHead
    kh_tiled = vh % NKHead
    qk_tiled = float(np.dot(q_predelta[t, kh_tiled].astype(np.float64), k_predelta[t, kh_tiled].astype(np.float64)))
    expected_tiled = beta * v * qk_tiled / np.sqrt(DState)

    # INTERLEAVED: kh = vh // (NVHead/NKHead) = vh // 2
    kh_inter = vh // (NVHead // NKHead)
    qk_inter = float(np.dot(q_predelta[t, kh_inter].astype(np.float64), k_predelta[t, kh_inter].astype(np.float64)))
    expected_inter = beta * v * qk_inter / np.sqrt(DState)

    llama_rms = float(np.sqrt((actual_llama ** 2).mean()))
    tiled_rms = float(np.sqrt((expected_tiled ** 2).mean()))
    inter_rms = float(np.sqrt((expected_inter ** 2).mean()))

    # Check which matches
    err_tiled = float(np.sqrt(((expected_tiled - actual_llama) ** 2).mean()))
    err_inter = float(np.sqrt(((expected_inter - actual_llama) ** 2).mean()))
    tiled_match = "MATCH" if err_tiled < 1e-5 else "no"
    inter_match = "MATCH" if err_inter < 1e-5 else "no"

    print(f"{vh:3d} {kh_tiled:9d} {kh_inter:15d} {llama_rms:11.6f} {tiled_rms:11.6f} {inter_rms:11.6f} {tiled_match:>12} {inter_match:>12}")
