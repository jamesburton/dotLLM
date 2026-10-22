"""Check linear_attn_qkv_mixed RMS at position 0 only."""
import sys, struct
from pathlib import Path
import numpy as np
from gguf import GGUFReader, dequantize


def read_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape)


def dequant(reader, name):
    t = next((t for t in reader.tensors if t.name == name), None)
    if t is None: return None
    arr = dequantize(t.data, t.tensor_type)
    return arr.reshape(tuple(reversed(t.shape))).astype(np.float32)


gguf = Path(sys.argv[1]); dump = Path(sys.argv[2])
reader = GGUFReader(str(gguf))

dot_embd = read_bin(dump / "00000_token_embd.bin")
seqLen = dot_embd.shape[0]
print(f"seqLen={seqLen}")

# Layer 0 attn_norm
anw = dequant(reader, "blk.0.attn_norm.weight").flatten()
eps = 1e-6
ref_norm = np.empty_like(dot_embd)
for t in range(seqLen):
    x = dot_embd[t].astype(np.float64)
    rms = np.sqrt((x ** 2).mean() + eps)
    ref_norm[t] = (x / rms * anw).astype(np.float32)
dot_norm = read_bin(dump / "00001_blk.0.attn_norm.bin")

# Per-position RMS comparison
print("=== attn_norm per-position RMS ===")
for t in range(seqLen):
    r = np.sqrt((ref_norm[t].astype(np.float64) ** 2).mean())
    d = np.sqrt((dot_norm[t].astype(np.float64) ** 2).mean())
    print(f"  pos {t}: ref_rms={r:.4f} dot_rms={d:.4f} diff_pct={100*(d-r)/r:.3f}%")

# Layer 0 qkv matmul per position
qkv_w = dequant(reader, "blk.0.attn_qkv.weight")
ref_qkv = (ref_norm.astype(np.float64) @ qkv_w.T.astype(np.float64)).astype(np.float32)
dot_qkv = read_bin(dump / "00002_blk.0.linear_attn_qkv_mixed.bin")

print()
print("=== linear_attn_qkv_mixed per-position RMS ===")
for t in range(seqLen):
    r = np.sqrt((ref_qkv[t].astype(np.float64) ** 2).mean())
    d = np.sqrt((dot_qkv[t].astype(np.float64) ** 2).mean())
    diff = (dot_qkv[t] - ref_qkv[t]).astype(np.float64)
    err_rms = np.sqrt((diff ** 2).mean())
    rel_err = err_rms / r
    print(f"  pos {t}: ref_rms={r:.4f} dot_rms={d:.4f} err_rms={err_rms:.4e} rel_err={100*rel_err:.4f}%")
