"""Verify layer 0 forward at ALL token positions."""
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
    ref_pow = (ref ** 2).sum()
    err_pow = (err ** 2).sum()
    if err_pow == 0: return float('inf'), 1.0
    snr_db = 10.0 * np.log10(ref_pow / err_pow)
    cos = float(np.dot(ref, dot) / (np.linalg.norm(ref) * np.linalg.norm(dot) + 1e-30))
    return snr_db, cos


def main():
    gguf = Path(sys.argv[1]); dump = Path(sys.argv[2])
    reader = GGUFReader(str(gguf))
    dot_embd = read_bin(dump / "00000_token_embd.bin")
    seqLen, hidden = dot_embd.shape

    # Layer 0 attn_norm
    attn_norm_w = dequant(reader, "blk.0.attn_norm.weight").flatten()
    eps = 1e-6
    # Compute attn_norm ref
    ref_norm = np.empty_like(dot_embd)
    for t in range(seqLen):
        x = dot_embd[t].astype(np.float64)
        rms = np.sqrt((x ** 2).mean() + eps)
        ref_norm[t] = (x / rms * attn_norm_w).astype(np.float32)
    dot_norm = read_bin(dump / "00001_blk.0.attn_norm.bin")
    print(f"=== blk.0.attn_norm (per pos) ===")
    for t in range(seqLen):
        s, c = snr(ref_norm[t], dot_norm[t])
        print(f"  pos {t}: SNR={s:5.1f} dB cos={c:.6f}")

    # blk.0.linear_attn_qkv_mixed (Q8_0 matmul)
    qkv_w = dequant(reader, "blk.0.attn_qkv.weight")
    ref_qkv = (ref_norm @ qkv_w.T).astype(np.float32)
    dot_qkv = read_bin(dump / "00002_blk.0.linear_attn_qkv_mixed.bin")
    print(f"=== blk.0.linear_attn_qkv_mixed (per pos) ===")
    for t in range(seqLen):
        s, c = snr(ref_qkv[t], dot_qkv[t])
        print(f"  pos {t}: SNR={s:5.1f} dB cos={c:.6f}")

    # Skip to MoE input (post_attention_norm output is sum of GDN out + embedding)
    # Just compare directly using dotLLM's attn_post_norm dump as the input.
    dot_post = read_bin(dump / "00017_blk.0.attn_post_norm.bin")
    dot_ffn = read_bin(dump / "00018_blk.0.ffn_out.bin")
    print(f"=== blk.0.ffn_out (per pos via gguf-py MoE) ===")
    # Router
    router_w = dequant(reader, "blk.0.ffn_gate_inp.weight")
    # Dequant experts ONCE
    print("  dequantizing experts (slow)...")
    gate_exps_full = dequant(reader, "blk.0.ffn_gate_exps.weight")
    up_exps_full = dequant(reader, "blk.0.ffn_up_exps.weight")
    down_exps_full = dequant(reader, "blk.0.ffn_down_exps.weight")
    sgate_w = dequant(reader, "blk.0.ffn_gate_shexp.weight")
    sup_w = dequant(reader, "blk.0.ffn_up_shexp.weight")
    sdown_w = dequant(reader, "blk.0.ffn_down_shexp.weight")
    shexp_gate_w = dequant(reader, "blk.0.ffn_gate_inp_shexp.weight").flatten()

    K = 8
    for t in range(seqLen):
        x = dot_post[t]
        gate_logits = (router_w @ x).astype(np.float64)
        probs = np.exp(gate_logits - gate_logits.max())
        probs /= probs.sum()
        # Top-K stable on ties: argsort -prob, kind='stable'
        topk_idx = np.argsort(-probs, kind='stable')[:K]
        topk_prob = probs[topk_idx]
        topk_prob /= topk_prob.sum()
        moe_out = np.zeros(hidden, dtype=np.float64)
        for e_idx, w in zip(topk_idx, topk_prob):
            W1 = gate_exps_full[e_idx].astype(np.float64)
            W3 = up_exps_full[e_idx].astype(np.float64)
            W2 = down_exps_full[e_idx].astype(np.float64)
            gate = W1 @ x.astype(np.float64)
            up = W3 @ x.astype(np.float64)
            inter = gate * (1.0 / (1.0 + np.exp(-gate))) * up
            out = W2 @ inter
            moe_out += w * out
        sg = (sgate_w @ x).astype(np.float64)
        su = (sup_w @ x).astype(np.float64)
        sinter = sg * (1.0 / (1.0 + np.exp(-sg))) * su
        sout = (sdown_w @ sinter).astype(np.float64)
        sscalar = 1.0 / (1.0 + np.exp(-(shexp_gate_w @ x)))
        ref_ffn = (moe_out + float(sscalar) * sout).astype(np.float32)
        s, c = snr(ref_ffn, dot_ffn[t])
        print(f"  pos {t}: SNR={s:5.1f} dB cos={c:.6f}  py_topk={topk_idx.tolist()[:4]}...")


if __name__ == "__main__":
    main()
