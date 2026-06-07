"""
Layer-by-layer parity check between dotLLM tensor dumps and direct dequant
of the source GGUF using gguf-py + numpy.

Usage:
    python parity_check.py <model.gguf> <dotllm_dump_dir>

This is the REFERENCE side of the parity verification. We dequantize tensors
directly from the GGUF (using the same algorithms llama.cpp uses, via gguf-py)
and compare specific values against what dotLLM dumped.
"""
import sys
import struct
import numpy as np
from pathlib import Path
from gguf import GGUFReader, dequantize, GGMLQuantizationType


def read_dotllm_bin(path: Path):
    """Read a dotLLM tensor dump: [int32 rank, int32 d0, int32 d1, int32 d2, float32...]."""
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
    """Dequantize a tensor by name to F32 numpy array. Returns (array, shape)."""
    t = next((t for t in reader.tensors if t.name == name), None)
    if t is None:
        raise KeyError(f"tensor '{name}' not in GGUF")
    # gguf-py's dequantize takes raw bytes + quant type + shape.
    arr = dequantize(t.data, t.tensor_type)
    arr = arr.reshape(tuple(reversed(t.shape)))  # GGUF innermost-first -> numpy outermost-first
    return arr.astype(np.float32), tuple(reversed(t.shape))


def stat(name, a):
    a = a.flatten()
    return (f"{name}: shape={a.shape if a.ndim == 1 else tuple(a.shape)} "
            f"min={a.min():+.6f} max={a.max():+.6f} "
            f"mean={a.mean():+.6f} rms={np.sqrt((a.astype(np.float64)**2).mean()):.6f} "
            f"abs_max={np.abs(a).max():.6f}")


def compare(name, ref, dot, tol_abs=1e-3, tol_rel=1e-3):
    ref_f = ref.astype(np.float32).flatten()
    dot_f = dot.astype(np.float32).flatten()
    n = min(ref_f.size, dot_f.size)
    diff = np.abs(ref_f[:n] - dot_f[:n])
    rel = diff / (np.abs(ref_f[:n]) + 1e-9)
    max_diff = diff.max()
    max_rel = rel.max()
    matches = (diff <= tol_abs + tol_rel * np.abs(ref_f[:n])).all()
    status = "PASS" if matches else "FAIL"
    print(f"  [{status}] {name}: max_abs_diff={max_diff:.4e} max_rel_diff={max_rel:.4e}")
    if not matches:
        # Show first 3 worst offenders.
        worst = np.argsort(diff)[-3:][::-1]
        for idx in worst:
            print(f"     idx={idx}: ref={ref_f[idx]:+.6f} dot={dot_f[idx]:+.6f} diff={diff[idx]:.4e}")
    return matches


def main():
    if len(sys.argv) != 3:
        print("Usage: parity_check.py <model.gguf> <dotllm_dump_dir>")
        sys.exit(1)

    gguf_path = Path(sys.argv[1])
    dump_dir = Path(sys.argv[2])

    print(f"=== Loading GGUF: {gguf_path}")
    reader = GGUFReader(str(gguf_path))
    print(f"=== Total tensors: {len(reader.tensors)}")

    # Pick a single token to verify (matches dotLLM's "Hi" prompt -> 1 token).
    # Hi -> token id 13048 (from llama-tokenize earlier). But the dotLLM dump
    # used "Hi" not "The capital of France is". So token id for "Hi" alone.
    # Verify by re-tokenizing.

    # ── Step 1: Token embedding for the single token of "Hi" ──
    # dotLLM dumped 1 token's embedding (shape [1, 2048]).
    print()
    print(f"=== Step 1: token embedding ===")
    embd_name = "token_embd.weight"
    embd_full, embd_shape = dequant_tensor(reader, embd_name)
    print(f"  GGUF {embd_name}: numpy shape={embd_full.shape} -> expect [vocab=248320, hidden=2048]")

    dot_embd, dot_shape = read_dotllm_bin(dump_dir / "00000_token_embd.bin")
    print(f"  dotLLM dump: shape={dot_shape}")
    # Multi-token support: dump may be [seqLen, hidden]. Use position 0 for embedding-row
    # identification, but keep ALL rows around for downstream sub-step checks.
    if dot_embd.ndim == 2 and dot_embd.shape[0] > 1:
        print(f"  Multi-token dump: seqLen={dot_embd.shape[0]}; using POSITION 0 for embedding match.")
        dot_embd_mat = dot_embd  # [seqLen, hidden]
        dot_row = dot_embd[0]
    else:
        dot_embd_mat = dot_embd.reshape(1, -1)
        dot_row = dot_embd.flatten()
    print(f"  dotLLM {stat('token_embd[0]', dot_row)}")
    # Compute L2 distance to each vocab row — vectorised over the whole table.
    diffs = np.linalg.norm(embd_full - dot_row[None, :], axis=1)
    nearest = int(np.argmin(diffs))
    nearest_diff = float(diffs[nearest])
    print(f"  -> nearest vocab row: id={nearest} l2_dist={nearest_diff:.6e}")
    if nearest_diff < 1e-4:
        print(f"  [PASS] Token embedding row {nearest} matches dotLLM's dump (l2 < 1e-4)")
    else:
        print(f"  [FAIL] Closest embedding row differs by l2={nearest_diff:.4e} from dotLLM dump")
        print(f"         Possible causes: wrong token id, wrong embedding dequant.")
        # Show row 13048 (expected "Hi" tokenization) for comparison.
        for try_id in (13048, 13046, 14990, 12365):
            if try_id < embd_full.shape[0]:
                d = np.linalg.norm(embd_full[try_id] - dot_row)
                print(f"           ref[{try_id}]: l2_dist={d:.4e}")

    # ── Step 2: blk.0.attn_norm.weight loaded value ──
    print()
    print(f"=== Step 2: blk.0.attn_norm.weight ===")
    nm_name = "blk.0.attn_norm.weight"
    nm_arr, nm_shape = dequant_tensor(reader, nm_name)
    print(f"  GGUF {nm_name}: shape={nm_arr.shape}")
    print(f"  GGUF {stat('attn_norm.weight', nm_arr)}")

    # Compute expected attn_norm output given dotLLM's embedding dump and the
    # weight, then compare to dotLLM's attn_norm dump.
    eps = 1e-6
    x = dot_embd[0]  # [hidden]
    x_rms = np.sqrt((x.astype(np.float64) ** 2).mean() + eps)
    ref_attn_norm = (x / x_rms * nm_arr.flatten()).astype(np.float32)
    dot_attn_norm, _ = read_dotllm_bin(dump_dir / "00001_blk.0.attn_norm.bin")
    print(f"  ref:    {stat('attn_norm (computed)', ref_attn_norm)}")
    print(f"  dotLLM: {stat('attn_norm (dumped)', dot_attn_norm)}")
    compare("attn_norm", ref_attn_norm, dot_attn_norm.flatten(), tol_abs=1e-3, tol_rel=1e-3)

    # ── Step 3: blk.0.attn_qkv.weight @ attn_norm  (the first quantized matmul) ──
    print()
    print(f"=== Step 3: blk.0.attn_qkv.weight @ attn_norm ===")
    qkv_w, qkv_shape = dequant_tensor(reader, "blk.0.attn_qkv.weight")
    print(f"  GGUF blk.0.attn_qkv.weight: shape={qkv_w.shape}")
    print(f"  GGUF {stat('attn_qkv.weight', qkv_w)}")
    # Compute the linear_attn_qkv_mixed = attn_norm @ W^T.
    # GGUF stores weight as [n_embd, conv_dim] with n_embd innermost.
    # After dequant.reshape(reversed(shape)) = numpy shape (conv_dim, n_embd) row-major.
    # The matmul output is then attn_norm[1, 2048] @ qkv_w.T = [1, conv_dim].
    ref_qkv = ref_attn_norm @ qkv_w.T  # [conv_dim]
    dot_qkv, _ = read_dotllm_bin(dump_dir / "00002_blk.0.linear_attn_qkv_mixed.bin")
    print(f"  ref:    {stat('linear_attn_qkv_mixed (computed)', ref_qkv)}")
    print(f"  dotLLM: {stat('linear_attn_qkv_mixed (dumped)', dot_qkv)}")
    compare("linear_attn_qkv_mixed", ref_qkv, dot_qkv.flatten(), tol_abs=5e-3, tol_rel=5e-3)

    # ── Step 4-6: alpha, beta, z projections (all smaller, all Q8_0-ish) ──
    for stage, fname in [
        ("attn_gate", "00003_blk.0.z.bin"),
        ("ssm_alpha", "00004_blk.0.alpha_proj.bin"),
        ("ssm_beta",  "00005_blk.0.beta_proj.bin"),
    ]:
        print()
        print(f"=== Step: blk.0.{stage}.weight @ attn_norm ===")
        w, _ = dequant_tensor(reader, f"blk.0.{stage}.weight")
        print(f"  GGUF blk.0.{stage}.weight: shape={w.shape}")
        ref = ref_attn_norm @ w.T
        dot, _ = read_dotllm_bin(dump_dir / fname)
        print(f"  ref:    {stat(stage, ref)}")
        print(f"  dotLLM: {stat(stage, dot)}")
        compare(stage, ref, dot.flatten(), tol_abs=5e-3, tol_rel=5e-3)

    # ── Step 7: blk.0.ssm_a values ──
    print()
    print(f"=== Step 7: blk.0.ssm_a (raw values, must be all <=0) ===")
    a_arr, _ = dequant_tensor(reader, "blk.0.ssm_a")
    print(f"  ref ssm_a: shape={a_arr.shape}")
    print(f"  ref {stat('ssm_a', a_arr)}")
    if (a_arr <= 0).all():
        print(f"  [PASS] all ssm_a <= 0 (formula assumption holds)")
    else:
        print(f"  [FAIL] ssm_a has positive values - formula needs negation")

    # ── Step 8: blk.0.ssm_dt.bias ──
    print()
    print(f"=== Step 8: blk.0.ssm_dt.bias ===")
    dt_arr, _ = dequant_tensor(reader, "blk.0.ssm_dt.bias")
    print(f"  ref {stat('ssm_dt.bias', dt_arr)}")

    # ── Step 9: blk.0.ssm_conv1d.weight + manual conv1d ──
    # We'd need the conv state (zeros for first token) and the qkv_mixed value.
    # Validate output element-wise.
    print()
    print(f"=== Step 9: blk.0.conv_output_silu (manual conv1d + SiLU on first token) ===")
    conv_w, _ = dequant_tensor(reader, "blk.0.ssm_conv1d.weight")
    print(f"  GGUF blk.0.ssm_conv1d.weight: shape={conv_w.shape}")
    # GGUF shape [d_conv=4, conv_dim=8192] -> dequant reshape to (8192, 4) row-major
    # (channel-major). Per channel c, taps are at conv_w[c, 0..3].
    # For the first token with zero conv state:
    #   y[t=0, c] = bias[c] + sum_k qkv[t-k+d_conv-1, c] * w[c, k]
    # All k>=1 use the zero state, k=0 uses the current qkv (the only non-zero row).
    # With d_conv=4, t=0 output:
    #   y[0, c] = 0 + qkv[0,c]*w[c,d_conv-1]  (only last tap is non-zero state)
    # Wait — depends on indexing. Let me check: input is [state(d_conv-1) | qkv(T)],
    # so input[d_conv-1 + t, c] for t in [0, T). For t=0 the kernel sees:
    #   input[(0+k) for k in 0..d_conv-1] = input[0..3]
    # input[0..2] = state (zeros), input[3] = qkv[0].
    # So y[0, c] = bias[c] + 0 + 0 + 0 + qkv[0,c] * w[c, d_conv-1=3].
    # i.e. y[0, c] = qkv[0, c] * w[c, 3]  (bias is 0 per LoadGdnLayer)
    dot_qkv_full = dot_qkv.flatten()  # = qkv_mixed for first token
    expected_conv = dot_qkv_full * conv_w[:, 3]
    # Then SiLU.
    expected_conv_silu = expected_conv * (1.0 / (1.0 + np.exp(-expected_conv)))
    dot_conv_silu, _ = read_dotllm_bin(dump_dir / "00008_blk.0.conv_output_silu.bin")
    print(f"  ref:    {stat('conv_output_silu (computed)', expected_conv_silu)}")
    print(f"  dotLLM: {stat('conv_output_silu (dumped)', dot_conv_silu)}")
    compare("conv_output_silu", expected_conv_silu, dot_conv_silu.flatten(), tol_abs=5e-3, tol_rel=5e-3)

    # ── Step 10: Q/K/V de-interleave from conv_output_silu ──
    # Layout per token (1 token here): [Q (kDim) | K (kDim) | V (vDim)]
    # NVHead=32, NKHead=16, DState=128 -> kDim=16*128=2048, vDim=32*128=4096
    print()
    print(f"=== Step 10: Q/K/V de-interleave ===")
    NVHead, NKHead, DState = 32, 16, 128
    kDim = NKHead * DState
    vDim = NVHead * DState
    expected_q = expected_conv_silu[:kDim].reshape(NKHead, DState)
    expected_k = expected_conv_silu[kDim:2*kDim].reshape(NKHead, DState)
    expected_v = expected_conv_silu[2*kDim:].reshape(NVHead, DState)
    dot_q, _ = read_dotllm_bin(dump_dir / "00009_blk.0.q_conv.bin")
    dot_k, _ = read_dotllm_bin(dump_dir / "00010_blk.0.k_conv.bin")
    dot_v, _ = read_dotllm_bin(dump_dir / "00011_blk.0.v_conv.bin")
    compare("q_conv (pre-L2)", expected_q.flatten(), dot_q.flatten(), 1e-6, 1e-6)
    compare("k_conv (pre-L2)", expected_k.flatten(), dot_k.flatten(), 1e-6, 1e-6)
    compare("v_conv         ", expected_v.flatten(), dot_v.flatten(), 1e-6, 1e-6)

    # ── Step 11: L2 normalize Q and K per head ──
    print()
    print(f"=== Step 11: L2-normalize Q and K (per head, eps=1e-6) ===")
    eps_l2 = 1e-6
    def l2_per_head(x):
        # x shape [n_heads, d_state]
        norms = np.sqrt((x.astype(np.float64) ** 2).sum(axis=1)) + eps_l2
        return (x / norms[:, None]).astype(np.float32)
    expected_q_l2 = l2_per_head(expected_q)
    expected_k_l2 = l2_per_head(expected_k)
    dot_q_l2, _ = read_dotllm_bin(dump_dir / "00012_blk.0.q_conv_predelta.bin")
    dot_k_l2, _ = read_dotllm_bin(dump_dir / "00013_blk.0.k_conv_predelta.bin")
    compare("q_conv_predelta", expected_q_l2.flatten(), dot_q_l2.flatten(), 1e-5, 1e-5)
    compare("k_conv_predelta", expected_k_l2.flatten(), dot_k_l2.flatten(), 1e-5, 1e-5)

    # ── Step 12: Compute g (decay) and beta (write-gate) ──
    print()
    print(f"=== Step 12: g = exp(softplus(alpha+dt)*A), beta = sigmoid(beta_proj) ===")
    alpha_proj, _ = read_dotllm_bin(dump_dir / "00004_blk.0.alpha_proj.bin")
    beta_proj, _ = read_dotllm_bin(dump_dir / "00005_blk.0.beta_proj.bin")
    ap = alpha_proj.flatten()
    bp = beta_proj.flatten()
    softplus_val = np.log1p(np.exp(ap + dt_arr))
    expected_g = np.exp(softplus_val * a_arr).astype(np.float32)
    expected_beta_sig = (1.0 / (1.0 + np.exp(-bp))).astype(np.float32)
    dot_g, _ = read_dotllm_bin(dump_dir / "00006_blk.0.g.bin")
    dot_beta_sig, _ = read_dotllm_bin(dump_dir / "00007_blk.0.beta_sigmoid.bin")
    compare("g (decay)        ", expected_g, dot_g.flatten(), 1e-5, 1e-5)
    compare("beta_sigmoid     ", expected_beta_sig, dot_beta_sig.flatten(), 1e-5, 1e-5)

    # ── Step 13: GDN scan output for the single first token ──
    # First-token scan with zero state:
    #   For each value head vh:
    #     kh = vh // VHeadsPerKHead   (=vh//2 for NVHead=32, NKHead=16)
    #     state_vh starts at zero. After step:
    #       retrieve = 0 (zero state)
    #       delta = beta * (v[vh] - 0) = beta * v[vh]   (length DState)
    #       state_vh = k[kh] outer delta = outer_product(k[kh], delta)
    #     Then output:
    #       out[vh] = state_vh.T @ q[kh] / sqrt(DState)
    #              = sum_row outer_product(k,delta)[row,col]*q[row] / sqrt(DState)
    #              = delta[col] * sum_row k[row]*q[row] / sqrt(DState)
    #              = (q·k) * delta / sqrt(DState)
    # Note: decay g doesn't matter on first step (initial state = 0).
    print()
    print(f"=== Step 13: GDN scan output (attn_output) - first-token analytic ===")
    qk_dot = (expected_q_l2 * expected_k_l2).sum(axis=1)  # [NKHead]
    expected_attn = np.zeros((NVHead, DState), dtype=np.float32)
    for vh in range(NVHead):
        kh = vh // (NVHead // NKHead)
        delta = expected_beta_sig[vh] * expected_v[vh]  # [DState]
        expected_attn[vh] = qk_dot[kh] * delta / np.sqrt(DState)
    dot_attn, _ = read_dotllm_bin(dump_dir / "00014_blk.0.attn_output.bin")
    compare("attn_output (scan)", expected_attn.flatten(), dot_attn.flatten(), 1e-4, 1e-3)

    # ── Step 14: Per-head RMSNorm with ssm_norm.weight + silu(z) gate ──
    # final_output[vh, i] = (attn_output[vh, i] / rms_per_head_vh) * ssm_norm[i] * silu(z[vh, i])
    print()
    print(f"=== Step 14: final_output = RmsNorm(scan, ssm_norm) * silu(z) ===")
    ssm_norm_w, _ = dequant_tensor(reader, "blk.0.ssm_norm.weight")
    ssm_norm_w = ssm_norm_w.flatten()  # [DState=128]
    z_proj, _ = read_dotllm_bin(dump_dir / "00003_blk.0.z.bin")
    z_proj = z_proj.flatten().reshape(NVHead, DState)
    eps_rms = 1e-6
    expected_final = np.empty((NVHead, DState), dtype=np.float32)
    attn = dot_attn.reshape(NVHead, DState)
    for vh in range(NVHead):
        head = attn[vh].astype(np.float64)
        rms_h = np.sqrt((head ** 2).mean() + eps_rms)
        normed = (head / rms_h).astype(np.float32) * ssm_norm_w  # [DState]
        z_head = z_proj[vh]
        silu_z = z_head * (1.0 / (1.0 + np.exp(-z_head)))
        expected_final[vh] = (normed * silu_z).astype(np.float32)
    dot_final, _ = read_dotllm_bin(dump_dir / "00015_blk.0.final_output.bin")
    compare("final_output", expected_final.flatten(), dot_final.flatten(), 1e-4, 1e-3)

    # ── Step 15: ssm_out projection ──
    print()
    print(f"=== Step 15: linear_attn_out = ssm_out @ final_output ===")
    ssm_out_w, _ = dequant_tensor(reader, "blk.0.ssm_out.weight")
    print(f"  GGUF blk.0.ssm_out.weight: shape={ssm_out_w.shape}")
    # GGUF [value_dim=4096, hidden=2048] -> numpy (hidden, value_dim)
    # output[h] = sum_v ssm_out[h, v] * final_output_flat[v]
    final_flat = dot_final.flatten()  # use dotLLM dump as input to ssm_out
    ref_linear_out = (final_flat @ ssm_out_w.T).astype(np.float32)
    dot_linear_out, _ = read_dotllm_bin(dump_dir / "00016_blk.0.linear_attn_out.bin")
    compare("linear_attn_out", ref_linear_out, dot_linear_out.flatten(), 5e-3, 5e-3)

    # ── Step 16: attn_post_norm = RmsNorm(embedding + linear_attn_out, post_attention_norm) ──
    print()
    print(f"=== Step 16: attn_post_norm ===")
    post_norm_w, _ = dequant_tensor(reader, "blk.0.post_attention_norm.weight")
    post_norm_w = post_norm_w.flatten()
    x_after_gdn = (dot_embd.flatten() + dot_linear_out.flatten()).astype(np.float32)
    rms_h = np.sqrt((x_after_gdn.astype(np.float64) ** 2).mean() + eps)
    ref_post = (x_after_gdn / rms_h * post_norm_w).astype(np.float32)
    dot_post, _ = read_dotllm_bin(dump_dir / "00017_blk.0.attn_post_norm.bin")
    compare("attn_post_norm ", ref_post, dot_post.flatten(), 1e-4, 1e-4)

    # ── Step 17: MoE forward (full computation) ──
    # ffn_out = sum_{e in topk}(renorm_w[e] * SwiGLU(W1_e, W2_e, W3_e, x))
    #        + sigmoid(gate_inp_shexp . x) * SwiGLU(sW1, sW2, sW3, x)
    print()
    print(f"=== Step 17: ffn_out (full MoE forward) ===")
    # 17a) routing
    router_w, _ = dequant_tensor(reader, "blk.0.ffn_gate_inp.weight")
    print(f"  router: shape={router_w.shape}")
    x = dot_post.flatten()
    gate_logits = (router_w @ x).astype(np.float32)
    probs = np.exp(gate_logits - gate_logits.max())
    probs /= probs.sum()
    # Top-8 selection (stable: lower index wins on tie via argsort kind='stable')
    K = 8
    topk_idx_full = np.argsort(probs)[::-1][:K]
    # Reorder stable on ties: argmax-walk
    topk_prob = probs[topk_idx_full]
    # Renorm
    topk_prob_renorm = topk_prob / topk_prob.sum()
    print(f"  topk_idx (py): {topk_idx_full.tolist()}")
    print(f"  topk_prob (py): {[f'{p:.4f}' for p in topk_prob_renorm]}")
    # dotLLM logged topk_idx=[112,238,127,57,66,106,200,120] — compare?

    # 17b) routed-expert SwiGLU sum. Dequant 3 fused expert tensors (slow!).
    print(f"  dequantizing 3 fused expert tensors (3GB, slow~30s)...")
    gate_exps_full, gshape = dequant_tensor(reader, "blk.0.ffn_gate_exps.weight")
    up_exps_full, _ = dequant_tensor(reader, "blk.0.ffn_up_exps.weight")
    down_exps_full, _ = dequant_tensor(reader, "blk.0.ffn_down_exps.weight")
    # GGUF shape [hid, intermediate, num_experts] -> numpy (num_experts, intermediate, hid).
    print(f"  gate_exps_full.shape={gate_exps_full.shape}")
    # Per expert: [intermediate, hidden] row-major for W1/W3, [hidden, intermediate] for W2.
    moe_out = np.zeros_like(x, dtype=np.float64)
    intermediate = 512
    for slot, (e_idx, w) in enumerate(zip(topk_idx_full, topk_prob_renorm)):
        W1 = gate_exps_full[e_idx]  # [intermediate, hidden]
        W3 = up_exps_full[e_idx]   # [intermediate, hidden]
        W2 = down_exps_full[e_idx]  # [hidden, intermediate]
        gate = W1 @ x  # [intermediate]
        up = W3 @ x
        inter = gate * (1.0 / (1.0 + np.exp(-gate))) * up  # silu(gate) * up
        out = W2 @ inter  # [hidden]
        moe_out += w * out
    print(f"  routed-only contribution: {stat('routed_sum', moe_out.astype(np.float32))}")

    # 17c) shared expert
    sgate_w, _ = dequant_tensor(reader, "blk.0.ffn_gate_shexp.weight")
    sup_w, _ = dequant_tensor(reader, "blk.0.ffn_up_shexp.weight")
    sdown_w, _ = dequant_tensor(reader, "blk.0.ffn_down_shexp.weight")
    shexp_gate, _ = dequant_tensor(reader, "blk.0.ffn_gate_inp_shexp.weight")
    s_gate = sgate_w @ x
    s_up = sup_w @ x
    s_inter = s_gate * (1.0 / (1.0 + np.exp(-s_gate))) * s_up
    s_out = sdown_w @ s_inter
    s_scalar = 1.0 / (1.0 + np.exp(-(shexp_gate.flatten() @ x)))
    print(f"  shared scalar: {s_scalar}")
    print(f"  shared expert raw: {stat('shared', s_out.astype(np.float32))}")
    moe_out_full = (moe_out + float(s_scalar) * s_out).astype(np.float32)
    print(f"  ref ffn_out: {stat('ffn_out (py)', moe_out_full)}")

    dot_ffn, _ = read_dotllm_bin(dump_dir / "00018_blk.0.ffn_out.bin")
    print(f"  dotLLM ffn_out: {stat('ffn_out (dotllm)', dot_ffn)}")
    compare("ffn_out", moe_out_full, dot_ffn.flatten(), 1e-2, 1e-2)

    # ── Step 18: Final result_norm and result_output ──
    # result_norm = RmsNorm(blk.39.l_out, output_norm.weight)
    # result_output = output.weight @ result_norm  (= logits)
    print()
    print(f"=== Step 18: final result_norm + lm_head ===")
    final_l_out, _ = read_dotllm_bin(dump_dir / "00610_blk.39.l_out.bin")
    output_norm_w, _ = dequant_tensor(reader, "output_norm.weight")
    output_norm_w = output_norm_w.flatten()
    flo = final_l_out.flatten()
    rms_h = np.sqrt((flo.astype(np.float64) ** 2).mean() + eps)
    ref_result_norm = (flo / rms_h * output_norm_w).astype(np.float32)
    dot_result_norm, _ = read_dotllm_bin(dump_dir / "00611_result_norm.bin")
    compare("result_norm", ref_result_norm, dot_result_norm.flatten(), 1e-4, 1e-4)

    # LM head — note GGUF stores as "output.weight" OR uses tied "token_embd.weight".
    # Inspect found both an output.weight (Q8_0 × 294 sample) and a separate token_embd.
    try:
        lm_head_w, _ = dequant_tensor(reader, "output.weight")
    except KeyError:
        lm_head_w, _ = dequant_tensor(reader, "token_embd.weight")  # tied
    print(f"  lm_head.shape={lm_head_w.shape}")
    ref_logits = (lm_head_w @ ref_result_norm).astype(np.float32)
    dot_logits, _ = read_dotllm_bin(dump_dir / "00612_result_output.bin")
    compare("result_output (logits)", ref_logits, dot_logits.flatten(), 5e-3, 1e-2)

    # Argmax = predicted token
    ref_top = np.argsort(ref_logits)[::-1][:5]
    dot_top = np.argsort(dot_logits.flatten())[::-1][:5]
    print(f"  ref top-5 token IDs: {ref_top.tolist()}")
    print(f"  ref top-5 logits:    {[float(ref_logits[i]) for i in ref_top]}")
    print(f"  dot top-5 token IDs: {dot_top.tolist()}")
    print(f"  dot top-5 logits:    {[float(dot_logits.flatten()[i]) for i in dot_top]}")
    # For multi-token prompts, dotLLM emits prefill logits per row. Just check the
    # LAST row (position seqLen-1) since that's what the sampler reads.
    if dot_logits.ndim == 2 and dot_logits.shape[0] > 1:
        print(f"  Note: dotLLM dumped {dot_logits.shape[0]} rows; using LAST row for argmax.")
        last_row = dot_logits[-1]
        dot_top_last = np.argsort(last_row)[::-1][:5]
        print(f"  dot LAST-row top-5: {dot_top_last.tolist()}")


if __name__ == "__main__":
    main()
