"""Independently compute the expected GDN scan output for vh=1 from raw inputs."""
import struct
import numpy as np
from pathlib import Path


def read_bin(p):
    with open(p, 'rb') as f:
        h = f.read(16)
        r, d0, d1, d2 = struct.unpack('<iiii', h)
        s = [d for d in [d0, d1, d2][:r]]
        n = int(np.prod(s)) if s else 0
        return np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(s) if s else np.frombuffer(f.read(n * 4), dtype=np.float32)


# Load inputs
q_predelta = read_bin('C:/dotllm_qwen35_hi/00012_blk.0.q_conv_predelta.bin')  # [1, 16, 128]
k_predelta = read_bin('C:/dotllm_qwen35_hi/00013_blk.0.k_conv_predelta.bin')  # [1, 16, 128]
v_conv = read_bin('C:/dotllm_qwen35_hi/00011_blk.0.v_conv.bin')  # [1, 32, 128]
beta_sig = read_bin('C:/dotllm_qwen35_hi/00007_blk.0.beta_sigmoid.bin').flatten()  # [32]
g = read_bin('C:/dotllm_qwen35_hi/00006_blk.0.g.bin').flatten()  # [32]
# Compare against expected output
llama_attn_out = read_bin('C:/llama_hi_bin/00046_attn_output-0.bin')  # [32, 128]
dotllm_attn_out = read_bin('C:/dotllm_qwen35_hi/00014_blk.0.attn_output.bin')  # [1, 32, 128]

NVHead = 32
NKHead = 16
DState = 128
vHeadsPerKHead = NVHead // NKHead  # 2

t = 0
print(f"Independent computation for token {t}:")
for vh in [0, 1, 2, 3, 5, 16, 19]:
    kh = vh // vHeadsPerKHead
    q = q_predelta[t, kh].astype(np.float64)  # [128]
    k = k_predelta[t, kh].astype(np.float64)
    v = v_conv[t, vh].astype(np.float64)
    beta = float(beta_sig[vh])
    g_vh = float(g[vh])

    qk = float(np.dot(q, k))
    # First token with zero state:
    # state = outer(k, beta*v) ; out[j] = sum_i state[i,j] * q[i] = beta * v[j] * (q.k)
    # then scale by 1/sqrt(d)
    expected = beta * v * qk / np.sqrt(DState)

    actual_llama = llama_attn_out[vh].astype(np.float64)
    actual_dotllm = dotllm_attn_out[0, vh].astype(np.float64)

    print(f"  vh={vh:2d} kh={kh}: beta={beta:.4f} g={g_vh:.4f} qk={qk:.6f}")
    print(f"    expected first3: {[f'{expected[i]:.6f}' for i in range(3)]}")
    print(f"    llama    first3: {[f'{actual_llama[i]:.6f}' for i in range(3)]}")
    print(f"    dotllm   first3: {[f'{actual_dotllm[i]:.6f}' for i in range(3)]}")
    # ratios
    expected_rms = float(np.sqrt((expected ** 2).mean()))
    llama_rms = float(np.sqrt((actual_llama ** 2).mean()))
    dotllm_rms = float(np.sqrt((actual_dotllm ** 2).mean()))
    print(f"    expected_rms={expected_rms:.6f} llama_rms={llama_rms:.6f} dotllm_rms={dotllm_rms:.6f}")
    print(f"    expected/llama={expected_rms/llama_rms:.4f} expected/dotllm={expected_rms/dotllm_rms:.4f}")
