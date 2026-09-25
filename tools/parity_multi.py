"""
Multi-token parity check: verify dotLLM's final logits at the LAST position
of a multi-token prompt match what Python computes from the (verified) last-layer
hidden state. This isolates the "result_norm + lm_head" tail from the multi-token
prefill internals — if this passes, every layer's contribution at the last
position cumulatively matches the reference.

Usage:
    python parity_multi.py <model.gguf> <dotllm_dump_dir>
"""
import sys
import struct
import numpy as np
from pathlib import Path
from gguf import GGUFReader, dequantize


def read_dotllm_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = []
        if rank >= 1: shape.append(d0)
        if rank >= 2: shape.append(d1)
        if rank >= 3: shape.append(d2)
        count = int(np.prod(shape))
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape) if rank > 0 else data, shape


def dequant_tensor(reader, name):
    t = next((t for t in reader.tensors if t.name == name), None)
    if t is None:
        raise KeyError(f"tensor '{name}' not in GGUF")
    arr = dequantize(t.data, t.tensor_type)
    arr = arr.reshape(tuple(reversed(t.shape)))
    return arr.astype(np.float32), tuple(reversed(t.shape))


def stat(name, a):
    a = a.flatten()
    return (f"{name}: n={a.size} min={a.min():+.6f} max={a.max():+.6f} "
            f"mean={a.mean():+.6f} rms={np.sqrt((a.astype(np.float64)**2).mean()):.6f}")


def compare(name, ref, dot, tol_abs=5e-3, tol_rel=1e-2):
    ref_f = ref.astype(np.float32).flatten()
    dot_f = dot.astype(np.float32).flatten()
    n = min(ref_f.size, dot_f.size)
    diff = np.abs(ref_f[:n] - dot_f[:n])
    max_diff = diff.max()
    matches = (diff <= tol_abs + tol_rel * np.abs(ref_f[:n])).all()
    status = "PASS" if matches else "FAIL"
    print(f"  [{status}] {name}: max_abs_diff={max_diff:.4e}")
    return matches


def main():
    gguf_path = Path(sys.argv[1])
    dump_dir = Path(sys.argv[2])

    print(f"=== Loading GGUF: {gguf_path}")
    reader = GGUFReader(str(gguf_path))

    # Read dotLLM dumps.
    dot_embd, _ = read_dotllm_bin(dump_dir / "00000_token_embd.bin")
    seq_len, hidden = dot_embd.shape if dot_embd.ndim == 2 else (1, dot_embd.size)
    print(f"=== Multi-token dump: seqLen={seq_len}, hidden={hidden}")

    # Identify each token by nearest-row search in token_embd.
    embd_full, _ = dequant_tensor(reader, "token_embd.weight")
    for t in range(seq_len):
        row = dot_embd[t] if dot_embd.ndim == 2 else dot_embd
        d = np.linalg.norm(embd_full - row[None, :], axis=1)
        idx = int(np.argmin(d))
        print(f"  token[{t}]: id={idx} l2_dist={float(d[idx]):.3e}")

    # Final layer's l_out (last layer = blk.39).
    print()
    print(f"=== Verifying result_output (logits) at LAST position ===")
    final_l_out, _ = read_dotllm_bin(dump_dir / "00610_blk.39.l_out.bin")
    print(f"  blk.39.l_out shape: {final_l_out.shape}")
    last_l_out = final_l_out[-1] if final_l_out.ndim == 2 else final_l_out

    # Apply output_norm + lm_head.
    output_norm_w, _ = dequant_tensor(reader, "output_norm.weight")
    output_norm_w = output_norm_w.flatten()
    eps = 1e-6
    rms = np.sqrt((last_l_out.astype(np.float64) ** 2).mean() + eps)
    ref_result_norm = (last_l_out / rms * output_norm_w).astype(np.float32)

    dot_result_norm, _ = read_dotllm_bin(dump_dir / "00611_result_norm.bin")
    if dot_result_norm.ndim == 2:
        dot_result_norm_last = dot_result_norm[-1]
    else:
        dot_result_norm_last = dot_result_norm
    print(f"  ref:    {stat('result_norm (computed for last pos)', ref_result_norm)}")
    print(f"  dotLLM: {stat('result_norm (dumped last row)', dot_result_norm_last)}")
    compare("result_norm @ pos", ref_result_norm, dot_result_norm_last, 1e-4, 1e-4)

    # lm_head.
    try:
        lm_head_w, _ = dequant_tensor(reader, "output.weight")
    except KeyError:
        lm_head_w, _ = dequant_tensor(reader, "token_embd.weight")
    ref_logits = (lm_head_w @ ref_result_norm).astype(np.float32)
    dot_logits, _ = read_dotllm_bin(dump_dir / "00612_result_output.bin")
    if dot_logits.ndim == 2:
        dot_logits_last = dot_logits[-1]
    else:
        dot_logits_last = dot_logits
    print(f"  ref:    {stat('logits (last pos, py)', ref_logits)}")
    print(f"  dotLLM: {stat('logits (last pos, dotllm)', dot_logits_last)}")
    compare("logits @ last pos", ref_logits, dot_logits_last, 5e-2, 1e-2)

    # Top-5 token argmax — the critical sanity check.
    ref_top = np.argsort(ref_logits)[::-1][:5]
    dot_top = np.argsort(dot_logits_last)[::-1][:5]
    print(f"  ref top-5: {ref_top.tolist()}")
    print(f"  dot top-5: {dot_top.tolist()}")
    if ref_top[0] == dot_top[0]:
        print(f"  [PASS] Top-1 token MATCH ({int(ref_top[0])})")
    else:
        print(f"  [FAIL] Top-1 MISMATCH: ref={int(ref_top[0])} dot={int(dot_top[0])}")


if __name__ == "__main__":
    main()
