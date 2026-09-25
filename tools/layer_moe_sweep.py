"""For each MoE layer, recompute ffn_out via gguf-py using dotLLM's attn_post_norm and compare."""
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


def main():
    gguf = Path(sys.argv[1]); dump = Path(sys.argv[2])
    layers_to_check = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 5, 10, 20, 30, 38]
    pos = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # which token position
    reader = GGUFReader(str(gguf))

    # Find dump files by layer
    def find_dump(layer, name):
        files = sorted(dump.glob(f"*_blk.{layer}.{name}.bin"))
        if not files:
            return None
        return read_bin(files[0])

    for L in layers_to_check:
        # We need: blk.L.attn_post_norm (input to MoE), blk.L.ffn_out (output)
        post = find_dump(L, "attn_post_norm")
        ffn = find_dump(L, "ffn_out")
        if post is None or ffn is None:
            print(f"  Layer {L}: missing dumps")
            continue
        if post.shape[0] != ffn.shape[0]:
            print(f"  Layer {L}: shape mismatch post={post.shape} ffn={ffn.shape}")
            continue
        x = post[pos]
        # Router
        router_w = dequant(reader, f"blk.{L}.ffn_gate_inp.weight")
        if router_w is None:
            print(f"  Layer {L}: no ffn_gate_inp, skipping")
            continue
        gate_logits = (router_w @ x).astype(np.float64)
        probs = np.exp(gate_logits - gate_logits.max())
        probs /= probs.sum()
        K = 8
        topk_idx = np.argsort(-probs, kind='stable')[:K]
        topk_prob = probs[topk_idx]
        topk_prob /= topk_prob.sum()
        print(f"Layer {L} pos {pos}: pre-MoE x[:5]={x[:5]} topk_prob_sum=1.0", flush=True)

        # Experts (slow: dequant once per layer)
        print(f"  dequant L{L} experts (slow ~30s)...", flush=True)
        gate_exps = dequant(reader, f"blk.{L}.ffn_gate_exps.weight")
        up_exps = dequant(reader, f"blk.{L}.ffn_up_exps.weight")
        down_exps = dequant(reader, f"blk.{L}.ffn_down_exps.weight")
        moe_out = np.zeros(x.shape[0], dtype=np.float64)
        for e_idx, w in zip(topk_idx, topk_prob):
            W1 = gate_exps[e_idx].astype(np.float64)
            W3 = up_exps[e_idx].astype(np.float64)
            W2 = down_exps[e_idx].astype(np.float64)
            g = W1 @ x.astype(np.float64)
            u = W3 @ x.astype(np.float64)
            inter = g * (1.0 / (1.0 + np.exp(-g))) * u
            out = W2 @ inter
            moe_out += w * out
        # Shared expert
        sgate_w = dequant(reader, f"blk.{L}.ffn_gate_shexp.weight")
        sup_w = dequant(reader, f"blk.{L}.ffn_up_shexp.weight")
        sdown_w = dequant(reader, f"blk.{L}.ffn_down_shexp.weight")
        shexp_gate_w = dequant(reader, f"blk.{L}.ffn_gate_inp_shexp.weight")
        sg = (sgate_w @ x).astype(np.float64)
        su = (sup_w @ x).astype(np.float64)
        sinter = sg * (1.0 / (1.0 + np.exp(-sg))) * su
        sout = (sdown_w @ sinter).astype(np.float64)
        sscalar = 1.0 / (1.0 + np.exp(-(shexp_gate_w.flatten() @ x)))
        ref = (moe_out + float(sscalar) * sout).astype(np.float32)
        s, c = snr(ref, ffn[pos])
        print(f"  layer {L} pos {pos}: SNR={s:5.1f} dB cos={c:.6f}  ref_rms={np.sqrt((ref**2).mean()):.4f} dot_rms={np.sqrt((ffn[pos]**2).mean()):.4f}  sscalar={sscalar:.3f}", flush=True)


if __name__ == "__main__":
    main()
