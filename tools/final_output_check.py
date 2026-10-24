"""Verify result_output = output.weight @ result_norm at each position via gguf-py."""
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


def snr(ref, dot):
    ref = ref.astype(np.float64).flatten()
    dot = dot.astype(np.float64).flatten()
    err = dot - ref
    rp = (ref ** 2).sum()
    ep = (err ** 2).sum()
    if ep == 0: return float('inf'), 1.0
    s = 10.0 * np.log10(rp / ep)
    cos = float(np.dot(ref, dot) / (np.linalg.norm(ref) * np.linalg.norm(dot) + 1e-30))
    return s, cos


gguf = Path(sys.argv[1]); dump = Path(sys.argv[2])
reader = GGUFReader(str(gguf))
dot_norm = read_bin(dump / "00721_result_norm.bin")  # last layer
print(f"result_norm shape: {dot_norm.shape}")
# Find the result_norm and result_output files
norm_files = sorted(dump.glob("*_result_norm.bin"))
out_files = sorted(dump.glob("*_result_output.bin"))
print(f"result_norm files: {[p.name for p in norm_files]}")
print(f"result_output files: {[p.name for p in out_files]}")
dot_norm = read_bin(norm_files[-1])
dot_out = read_bin(out_files[-1])
print(f"result_norm: {dot_norm.shape}, result_output: {dot_out.shape}")

output_w = dequant(reader, "output.weight")
# Or tied embeddings? Let me check token_embd
print(f"output.weight shape: {output_w.shape}")  # should be (vocab, hidden)

for t in range(dot_norm.shape[0]):
    x = dot_norm[t]
    ref_logits = (output_w @ x).astype(np.float32)
    s, c = snr(ref_logits, dot_out[t])
    # Top-K diff
    ref_top = np.argsort(-ref_logits)[:5]
    dot_top = np.argsort(-dot_out[t])[:5]
    print(f"  pos {t}: SNR={s:5.1f} dB cos={c:.6f}")
    print(f"    ref top-5: {ref_top.tolist()} logits {[f'{ref_logits[i]:+.2f}' for i in ref_top]}")
    print(f"    dot top-5: {dot_top.tolist()} logits {[f'{dot_out[t][i]:+.2f}' for i in dot_top]}")
