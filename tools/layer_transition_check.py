"""Verify layer-to-layer transition integrity.

For each layer L > 0:
- Take dotLLM's `l_out[L-1]` (post-MoE residual stream from prior layer)
- Compute: ref_attn_norm[L] = RMSnorm(l_out[L-1], attn_norm_weight[L])
- Compare against dotLLM's blk.L.attn_norm

Then independently:
- Compute ref_x_after_attn[L] = l_out[L-1] + linear_attn_out[L]
- Compute ref_attn_post_norm[L] = RMSnorm(ref_x_after_attn[L], post_attn_norm_weight[L])
- Compare against dotLLM's blk.L.attn_post_norm
"""
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
    rp = (ref ** 2).sum(); ep = (err ** 2).sum()
    if ep == 0: return float('inf'), 1.0
    return 10.0 * np.log10(rp / ep), float(np.dot(ref, dot) / (np.linalg.norm(ref) * np.linalg.norm(dot) + 1e-30))


def find(dump, layer, name):
    files = sorted(dump.glob(f"*_blk.{layer}.{name}.bin"))
    return read_bin(files[0]) if files else None


def main():
    gguf = Path(sys.argv[1]); dump = Path(sys.argv[2])
    reader = GGUFReader(str(gguf))
    eps = 1e-6

    # The embedding for layer 0 input is "00000_token_embd"
    embd_files = sorted(dump.glob("00000_token_embd.bin"))
    dot_embd = read_bin(embd_files[0])
    seqLen = dot_embd.shape[0]

    # Step 1: verify blk.0.attn_norm = RMSnorm(token_embd, attn_norm[0]) — already
    # confirmed in parity_check.py, skip.

    # Step 2 onwards: for each layer L > 0, check attn_norm.
    print(f"=== Inter-layer transition check, seqLen={seqLen}, pos=0 ===")
    pos = 0

    for L in [1, 2, 5, 10, 20, 30, 38, 39]:
        # blk.L.attn_norm should equal RMSnorm(l_out[L-1], attn_norm_weight[L])
        l_out_prev = find(dump, L - 1, "l_out")
        if l_out_prev is None: continue
        anw_arr = dequant(reader, f"blk.{L}.attn_norm.weight")
        if anw_arr is None: continue
        anw = anw_arr.flatten()
        x = l_out_prev[pos].astype(np.float64)
        rms_inv = 1.0 / np.sqrt((x ** 2).mean() + eps)
        ref_attn_norm = (x * rms_inv * anw).astype(np.float32)
        dot_attn_norm = find(dump, L, "attn_norm")[pos]
        s, c = snr(ref_attn_norm, dot_attn_norm)
        flag = "PASS" if s > 60 else ("WEAK" if s > 30 else "FAIL")
        print(f"  [{flag}] blk.{L}.attn_norm: SNR={s:5.1f} dB cos={c:.6f}  ref_rms={np.sqrt((ref_attn_norm.astype(np.float64)**2).mean()):.4f} dot_rms={np.sqrt((dot_attn_norm.astype(np.float64)**2).mean()):.4f}")

    print()
    print(f"=== Inter-layer attn_post_norm transition ===")
    # For each layer L:
    # ref_x_after_attn = l_out[L-1] (or token_embd for L=0) + linear_attn_out[L] (GDN) or fa_attnout_postgate->o_proj (full-attn)
    # ref_attn_post_norm = RMSnorm(ref_x_after_attn, post_attn_norm_w[L])
    for L in [0, 1, 2, 5, 10, 20, 30, 38]:
        # For GDN layers only — linear_attn_out is the GDN output AFTER ssm_out projection
        # x_after_attn = (l_out[L-1] or token_embd) + linear_attn_out[L]
        if L == 0:
            prev = dot_embd
        else:
            prev = find(dump, L - 1, "l_out")
            if prev is None: continue
        attn_out_dotllm = find(dump, L, "linear_attn_out")
        if attn_out_dotllm is None:
            print(f"  blk.{L}: no linear_attn_out (full-attn layer, skip)")
            continue
        x_after = (prev[pos] + attn_out_dotllm[pos]).astype(np.float64)
        post_w_arr = dequant(reader, f"blk.{L}.post_attention_norm.weight")
        post_w = post_w_arr.flatten()
        rms_inv = 1.0 / np.sqrt((x_after ** 2).mean() + eps)
        ref_post = (x_after * rms_inv * post_w).astype(np.float32)
        dot_post = find(dump, L, "attn_post_norm")[pos]
        s, c = snr(ref_post, dot_post)
        flag = "PASS" if s > 60 else ("WEAK" if s > 30 else "FAIL")
        print(f"  [{flag}] blk.{L}.attn_post_norm: SNR={s:5.1f} dB cos={c:.6f}  ref_rms={np.sqrt((ref_post.astype(np.float64)**2).mean()):.4f} dot_rms={np.sqrt((dot_post.astype(np.float64)**2).mean()):.4f}")


if __name__ == "__main__":
    main()
