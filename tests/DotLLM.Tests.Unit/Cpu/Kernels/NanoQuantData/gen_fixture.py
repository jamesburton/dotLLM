"""Offline generator for NanoQuantRealFixture (#866). Run: python -I gen_fixture.py <community.gguf> <out_dir>

Reads real tensors of arelath/Qwen3-0.6B-nanoquant-GGUF with the `gguf` package and writes
  nanoquant_crop_fixture.json  : a CROP (real bits/scales, d_out=64, r=80, d_in=300) of blk.0.ffn_gate with both salient
                                 columns kept, plus a float64 numpy reference y for a fixed x  (<50 KB, committed)
  nanoquant_full_expect.json   : float64 reference y (first 32 + sum + sum of squares) of the FULL layers blk.0.ffn_gate and
                                 blk.0.attn_qkv for x[i] = 1.5*sin(0.37*i+0.1)  (tiny, committed; the real-file test
                                 compares the C# kernel against it when the GGUF is present)
Reference maths (float64): y = post*(U(mid*(V(pre*x)))) + Wsal @ x[idx].
"""
import sys, json, gguf, numpy as np

path, out = sys.argv[1], sys.argv[2]
R = gguf.GGUFReader(path)
T = {t.name: t for t in R.tensors}

def bf16(a):
    return (np.asarray(a).view(np.uint16).astype(np.uint32) << 16).view(np.float32)

def scale(name):
    t = T[name]
    return bf16(t.data) if t.tensor_type.name == "BF16" else np.asarray(t.data, dtype=np.float32)

def signs(words, n):  # words [rows, w] int32 -> [rows, n] +-1 float64
    u = np.asarray(words).view(np.uint32)
    bits = ((u[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(u.shape[0], -1)[:, :n]
    return 1.0 - 2.0 * bits.astype(np.float64)

def pack(sg):  # [rows, n] +-1 -> int32 words, tail clear
    rows, n = sg.shape
    w = (n + 31) // 32
    bits = np.zeros((rows, w * 32), dtype=np.uint64)
    bits[:, :n] = (sg < 0)
    u = (bits.reshape(rows, w, 32) << np.arange(32, dtype=np.uint64)).sum(axis=2).astype(np.uint32)
    return u.view(np.int32)

def layer(base):
    u, v = T[base + ".nq_u"], T[base + ".nq_v"]
    pre, mid, post = scale(base + ".nq_scale_pre"), scale(base + ".nq_scale_mid"), scale(base + ".nq_scale_post")
    idx = np.asarray(T[base + ".nq_salient_idx"].data).astype(np.int64)
    sw = np.asarray(T[base + ".nq_salient_weight"].data).astype(np.float32)  # [d_out, k] (ggml [k, d_out])
    return np.asarray(u.data), np.asarray(v.data), pre, mid, post, idx, sw

def ref(U, V, pre, mid, post, idx, sw, x, dout, din, r):
    Us, Vs = signs(U, r), signs(V, din)
    lat = mid.astype(np.float64) * (Vs @ (pre.astype(np.float64) * x.astype(np.float64)))
    return post.astype(np.float64) * (Us @ lat) + sw.astype(np.float64) @ x.astype(np.float64)[idx]

def xvec(n):
    return (1.5 * np.sin(0.37 * np.arange(n) + 0.1)).astype(np.float32)

# ---- crop fixture
U, V, pre, mid, post, idx, sw = layer("blk.0.ffn_gate")
dout, r, din = 64, 80, 300
r_full, din_full = V.shape[0], 1024
Us, Vs = signs(U, r_full)[:dout, :r], signs(V, din_full)[:r, :din]
assert all(i < din for i in idx), idx
x = xvec(din)
cpre, cmid, cpost = pre[:din], mid[:r], post[:dout]
assert (cpre[idx] == 0).all()
Uw, Vw = pack(Us), pack(Vs)
csw = sw[:dout, :]
y = ref(Uw, Vw, cpre, cmid, cpost, idx, csw, x, dout, din, r)
fx = dict(d_out=dout, d_in=din, r=r, U=Uw.tolist(), V=Vw.tolist(), scale_pre=cpre.tolist(), scale_mid=cmid.tolist(),
          scale_post=cpost.tolist(), salient_idx=idx.tolist(), salient_weight=csw.tolist(), x=x.tolist(), y=y.tolist(),
          source="arelath/Qwen3-0.6B-nanoquant-GGUF blk.0.ffn_gate cropped rows[:64], rank[:80], cols[:300]")
open(out + "/nanoquant_crop_fixture.json", "w").write(json.dumps(fx, separators=(",", ":")))

# ---- full-layer expectations
exp = {}
for base in ("blk.0.ffn_gate", "blk.0.attn_qkv", "blk.27.ffn_down"):
    U, V, pre, mid, post, idx, sw = layer(base)
    do, r_, di = U.shape[0], V.shape[0], pre.shape[0]
    y = ref(U, V, pre, mid, post, idx, sw, xvec(di), do, di, r_)
    exp[base] = dict(d_out=do, d_in=di, r=r_, y_first=y[:32].tolist(), y_sum=float(y.sum()), y_sumsq=float((y * y).sum()), k=len(idx))
open(out + "/nanoquant_full_expect.json", "w").write(json.dumps(exp, indent=1))
print({k: (v["d_out"], v["d_in"], v["r"], v["k"]) for k, v in exp.items()})
