"""
Layer-by-layer SNR (signal-to-noise) parity check.

For each per-layer dotLLM-dumped tensor, compute the gguf-py reference
analytically (where possible) and compare via cosine similarity + relative
RMS error. SNR is a much more meaningful signal than max_abs_diff for
Q8_0/Q6_K matmuls, where small absolute errors aggregated across K=2048
produce non-trivial individual elements but the overall direction is
preserved.

Usage:
    python parity_check2.py <model.gguf> <dotllm_dump_dir>
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
    return data.reshape(shape), tuple(shape)


def dequant_tensor(reader, name):
    t = next((t for t in reader.tensors if t.name == name), None)
    if t is None:
        return None, None, None
    arr = dequantize(t.data, t.tensor_type)
    arr = arr.reshape(tuple(reversed(t.shape))).astype(np.float32)
    return arr, tuple(reversed(t.shape)), t.tensor_type.name


def snr(ref, dot):
    """Return signal-to-noise ratio in dB and other metrics."""
    ref = ref.astype(np.float64).flatten()
    dot = dot.astype(np.float64).flatten()
    err = dot - ref
    ref_pow = (ref ** 2).sum()
    err_pow = (err ** 2).sum()
    if err_pow == 0: return float('inf'), 1.0, 0.0
    snr_db = 10.0 * np.log10(ref_pow / err_pow)
    cos = float(np.dot(ref, dot) / (np.linalg.norm(ref) * np.linalg.norm(dot) + 1e-30))
    rel_rms = float(np.sqrt(err_pow / ref_pow))
    return snr_db, cos, rel_rms


def stat_line(name, ref, dot):
    snr_db, cos, rel_rms = snr(ref, dot)
    flag = "PASS" if snr_db > 40 and cos > 0.9999 else \
           ("WEAK" if snr_db > 25 and cos > 0.99 else "FAIL")
    print(f"  [{flag}] {name}: SNR={snr_db:5.1f} dB  cos={cos:.6f}  rel_rms={rel_rms:.3e}")
    return flag


def main():
    gguf_path = Path(sys.argv[1])
    dump_dir = Path(sys.argv[2])
    print(f"Loading {gguf_path}")
    reader = GGUFReader(str(gguf_path))

    # Token-embed: dotLLM dump = (seqLen, hidden)
    dot_embd, _ = read_bin(dump_dir / "00000_token_embd.bin")
    seqLen, hidden = dot_embd.shape
    print(f"seqLen={seqLen} hidden={hidden}")

    embd_full, _, _ = dequant_tensor(reader, "token_embd.weight")
    # For position 0
    pos = 0
    x0 = dot_embd[pos]
    nearest = int(np.argmin(np.linalg.norm(embd_full - x0[None, :], axis=1)))
    print(f"pos {pos} -> nearest token id {nearest}")

    # Step 1: blk.0.attn_norm via F32 ref
    attn_norm_w, _, _ = dequant_tensor(reader, "blk.0.attn_norm.weight")
    attn_norm_w = attn_norm_w.flatten()
    eps = 1e-6
    print()
    print("=== Layer 0 ===")
    # Compute attn_norm for all positions
    dot_attn_norm, _ = read_bin(dump_dir / "00001_blk.0.attn_norm.bin")
    ref_attn_norm = np.empty_like(dot_attn_norm)
    for t in range(seqLen):
        x = dot_embd[t].astype(np.float64)
        rms = np.sqrt((x ** 2).mean() + eps)
        ref_attn_norm[t] = (x / rms * attn_norm_w).astype(np.float32)
    stat_line("blk.0.attn_norm", ref_attn_norm, dot_attn_norm)

    # Step 2: blk.0.attn_qkv
    qkv_w, _, qkv_qt = dequant_tensor(reader, "blk.0.attn_qkv.weight")
    print(f"  (attn_qkv quant: {qkv_qt})")
    dot_qkv, _ = read_bin(dump_dir / "00002_blk.0.linear_attn_qkv_mixed.bin")
    out_dim = qkv_w.shape[0]  # 8192
    ref_qkv = (ref_attn_norm @ qkv_w.T).astype(np.float32)  # [seqLen, 8192]
    stat_line("blk.0.linear_attn_qkv_mixed", ref_qkv, dot_qkv)

    # Step 3: attn_gate
    g_w, _, g_qt = dequant_tensor(reader, "blk.0.attn_gate.weight")
    dot_z, _ = read_bin(dump_dir / "00003_blk.0.z.bin")
    ref_z = (ref_attn_norm @ g_w.T).astype(np.float32)
    stat_line(f"blk.0.attn_gate (z) [{g_qt}]", ref_z, dot_z)

    # Steps 4: ssm_alpha / ssm_beta (F32)
    a_w, _, a_qt = dequant_tensor(reader, "blk.0.ssm_alpha.weight")
    dot_alpha, _ = read_bin(dump_dir / "00004_blk.0.alpha_proj.bin")
    ref_alpha = (ref_attn_norm @ a_w.T).astype(np.float32)
    stat_line(f"blk.0.ssm_alpha [{a_qt}]", ref_alpha, dot_alpha)

    b_w, _, b_qt = dequant_tensor(reader, "blk.0.ssm_beta.weight")
    dot_beta, _ = read_bin(dump_dir / "00005_blk.0.beta_proj.bin")
    ref_beta = (ref_attn_norm @ b_w.T).astype(np.float32)
    stat_line(f"blk.0.ssm_beta [{b_qt}]", ref_beta, dot_beta)

    # Skip the GDN scan steps (complicated) — jump to MoE input
    print()
    print("=== Layer 0 MoE ===")
    # Compute ffn_out independently — requires the post_attention_norm input from dotLLM.
    dot_post_norm, _ = read_bin(dump_dir / "00017_blk.0.attn_post_norm.bin")
    # Router
    router_w, _, r_qt = dequant_tensor(reader, "blk.0.ffn_gate_inp.weight")
    print(f"  (router quant: {r_qt})")
    # For position 0 only (MoE check is expensive)
    x_post = dot_post_norm[0]  # [hidden]
    gate_logits = (router_w @ x_post).astype(np.float64)
    probs = np.exp(gate_logits - gate_logits.max())
    probs /= probs.sum()
    K = 8
    # Argsort with stable tie-break (lower index wins)
    sorted_idx = np.argsort(-probs, kind='stable')
    topk_idx = sorted_idx[:K]
    topk_prob = probs[topk_idx]
    topk_prob /= topk_prob.sum()
    print(f"  py topk_idx: {topk_idx.tolist()}")
    print(f"  py topk_prob: {[f'{p:.4f}' for p in topk_prob]}")

    # Compute MoE ffn_out for position 0
    print(f"  dequantizing expert tensors...")
    gate_exps_full, gshape, gqt = dequant_tensor(reader, "blk.0.ffn_gate_exps.weight")
    up_exps_full, _, uqt = dequant_tensor(reader, "blk.0.ffn_up_exps.weight")
    down_exps_full, _, dqt = dequant_tensor(reader, "blk.0.ffn_down_exps.weight")
    print(f"  gate_exps quant {gqt} shape={gate_exps_full.shape}")
    print(f"  up_exps quant {uqt}")
    print(f"  down_exps quant {dqt}")

    moe_out = np.zeros(hidden, dtype=np.float64)
    for e_idx, w in zip(topk_idx, topk_prob):
        W1 = gate_exps_full[e_idx].astype(np.float64)  # [intermediate, hidden]
        W3 = up_exps_full[e_idx].astype(np.float64)
        W2 = down_exps_full[e_idx].astype(np.float64)  # [hidden, intermediate]
        gate = W1 @ x_post.astype(np.float64)
        up = W3 @ x_post.astype(np.float64)
        inter = gate * (1.0 / (1.0 + np.exp(-gate))) * up
        out = W2 @ inter
        moe_out += w * out
    print(f"  routed-only RMS: {np.sqrt((moe_out**2).mean()):.4f}")

    # Shared expert
    sgate_w, _, _ = dequant_tensor(reader, "blk.0.ffn_gate_shexp.weight")
    sup_w, _, _ = dequant_tensor(reader, "blk.0.ffn_up_shexp.weight")
    sdown_w, _, _ = dequant_tensor(reader, "blk.0.ffn_down_shexp.weight")
    shexp_gate_w, _, _ = dequant_tensor(reader, "blk.0.ffn_gate_inp_shexp.weight")
    sg = (sgate_w @ x_post).astype(np.float64)
    su = (sup_w @ x_post).astype(np.float64)
    sinter = sg * (1.0 / (1.0 + np.exp(-sg))) * su
    sout = (sdown_w @ sinter).astype(np.float64)
    sscalar = 1.0 / (1.0 + np.exp(-(shexp_gate_w.flatten() @ x_post)))
    print(f"  shared scalar: {sscalar:.4f}")
    ref_ffn_pos0 = (moe_out + float(sscalar) * sout).astype(np.float32)

    # Compare against dotLLM ffn_out at position 0
    dot_ffn, _ = read_bin(dump_dir / "00018_blk.0.ffn_out.bin")
    print(f"  ref_ffn pos0 RMS: {np.sqrt((ref_ffn_pos0**2).mean()):.4f}")
    print(f"  dot_ffn pos0 RMS: {np.sqrt((dot_ffn[0]**2).mean()):.4f}")
    stat_line("blk.0.ffn_out (pos0)", ref_ffn_pos0, dot_ffn[0])


if __name__ == "__main__":
    main()
