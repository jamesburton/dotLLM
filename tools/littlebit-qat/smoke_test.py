"""CPU smoke test: packing round-trip, SmoothSign grad, toy 2-layer Qwen3 convert -> train steps -> export ->
reload -> bit-exact bf16 logits, plus fixture generation for dotLLM's unit test.
  python smoke_test.py [--write-fixture PATH]"""
import argparse, copy, json, os, tempfile

import torch

from littlebit import (LittleBitConfig, LittleBitLinear, SmoothSign, convert_model, export_checkpoint,
                       load_checkpoint, pack_signs, quantised_modules, rank_for_bpw, unpack_signs,
                       bits_spec, bits_stored, fp16_range_report)
import train as T


def test_ranks():
    assert rank_for_bpw(4096, 4096, 0.1, True) == 80                      # spec worked example
    assert rank_for_bpw(1024, 3072, 0.55, True) == 192
    assert rank_for_bpw(1024, 2048, 0.55, True) == 168
    assert rank_for_bpw(1024, 1024, 0.55, True) == 120
    assert rank_for_bpw(64, 64, 0.1, True) == 8                           # degenerate -> min_split_dim
    b = bits_spec(1024, 3072, 192, True)
    assert b == 1710080, b                                                # 0.5436 bpw (spike's PaperBits would say 3283712)


def test_pack():
    g = torch.Generator().manual_seed(0)
    for rows, cols in [(3, 1), (3, 31), (5, 32), (4, 33), (2, 100), (7, 64)]:
        x = torch.randn(rows, cols, generator=g)
        x[0, 0] = 0.0                                                      # sign(0) = +1
        p = pack_signs(x)
        assert p.dtype == torch.int32 and p.shape == (rows, (cols + 31) // 32)
        u = unpack_signs(p, cols, torch.float32)
        assert torch.equal(u, torch.where(x >= 0, 1.0, -1.0))
    # bit layout: column c -> word c//32, bit c%32, 1 == -1, LSB first; bit 31 -> negative int32
    x = torch.ones(1, 40); x[0, 0] = -1; x[0, 31] = -1; x[0, 33] = -1
    p = pack_signs(x)
    assert p[0, 0].item() == (1 | (1 << 31)) - (1 << 32), p
    assert p[0, 1].item() == 2


def test_smoothsign_grad():
    x = torch.tensor([-0.02, 0.0, 0.003, 1.0], requires_grad=True)
    y = SmoothSign.apply(x); y.sum().backward()
    exp = 100 * (1 - torch.tanh(100 * x.detach()) ** 2)
    assert torch.allclose(x.grad, exp) and y.tolist() == [-1, 1, 1, 1]


def toy():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    c = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=16, max_position_embeddings=128, tie_word_embeddings=True)
    return Qwen3ForCausalLM._from_config(c, dtype=torch.bfloat16).eval()


def test_toy_roundtrip_and_training():
    teacher = toy()
    cfg = LittleBitConfig(eff_bit=2.0, residual=True, use_itq=True, itq_n_iter=10)
    student, ranks = T.build_student(teacher, cfg, "cpu", log=None)
    assert all(r >= 8 for r in ranks.values()), ranks
    # --- a few real optimiser steps through the distillation loss (checkpointed KL + hidden MSE)
    student.train()
    cs, ct = T.HiddenCapture(student), T.HiddenCapture(teacher)
    opt = T.MasterAdamW(list(student.parameters()), 1e-3)
    ids = torch.randint(0, 256, (2, 48))
    losses = []
    for _ in range(25):
        opt.zero_grad()
        loss, kl, mse = T.distill_loss(student, teacher, cs, ct, ids, kl_chunk=16)
        loss.backward(); opt.step(); losses.append(loss.item())
    print("toy distill loss", losses[0], "->", losses[-1])
    assert losses[-1] < losses[0] * 0.9
    student.eval()
    with torch.no_grad():
        ref = student(input_ids=ids).logits
    with tempfile.TemporaryDirectory() as d:
        rep = export_checkpoint(student, d, cfg)
        print("fp16 range report", rep)
        re = load_checkpoint(d)
        with torch.no_grad():
            got = re(input_ids=ids).logits
        assert torch.equal(ref, got), (ref - got).abs().max()
        # packed words round-trip exactly
        for n, m in quantised_modules(student):
            f = dict(re.named_modules())[n]
            for s in m.suffixes:
                assert torch.equal(pack_signs(getattr(m, "U" + s).detach()), getattr(f, f"U{s}_packed"))
                assert torch.equal(pack_signs(getattr(m, "V" + s).detach()), getattr(f, f"V{s}_packed"))
        from safetensors import safe_open
        with safe_open(os.path.join(d, "model.safetensors"), "pt") as f:
            keys = set(f.keys())
        assert "model.layers.0.self_attn.q_proj.U_R_packed" in keys and "model.layers.0.mlp.down_proj.v1" in keys
        assert "lm_head.weight" not in keys and not any(k.endswith(".weight") and "q_proj" in k for k in keys)
    print("bits", bits_spec(64, 64, 24, True) / 4096, bits_stored(64, 64, 24, True) / 4096)


def bf16r(x):
    return x.to(torch.bfloat16).to(torch.float64)


def make_fixture(path):
    """Spec-shaped single layer: d_out=40, d_in=100, r=40 (neither a multiple of 32), two paths."""
    g = torch.Generator().manual_seed(1234)
    d_out, d_in, r = 40, 100, 40
    lay = LittleBitLinear(d_in, d_out, r, True, None, LittleBitConfig())
    paths, y64 = [], torch.zeros(d_out, dtype=torch.float64)
    x = torch.randn(d_in, generator=g).to(torch.bfloat16)
    for sfx in ("", "_R"):
        U, V = torch.randn(d_out, r, generator=g), torch.randn(r, d_in, generator=g)
        sc = {k: (torch.rand(n, generator=g) * 1.45 + 0.05).to(torch.bfloat16)
              for k, n in (("u1", d_out), ("u2", r), ("v1", r), ("v2", d_in))}
        setattr(lay, "U" + sfx, torch.nn.Parameter(U)); setattr(lay, "V" + sfx, torch.nn.Parameter(V))
        for k, v in sc.items():
            setattr(lay, k + sfx, torch.nn.Parameter(v.float()))
        l = (sc["v1"] * sc["u2"])                                          # bf16 product (spec 2.1)
        Us, Vs = torch.where(U >= 0, 1.0, -1.0).double(), torch.where(V >= 0, 1.0, -1.0).double()
        t = l.double() * (Vs @ (sc["v2"].double() * x.double()))
        y64 += sc["u1"].double() * (Us @ t)
        paths.append(dict(U_packed=pack_signs(U).tolist(), V_packed=pack_signs(V).tolist(),
                          u1=sc["u1"].float().tolist(), u2=sc["u2"].float().tolist(), v1=sc["v1"].float().tolist(),
                          v2=sc["v2"].float().tolist(), l=l.float().tolist()))
    with torch.no_grad():
        ybf = lay(x[None]).float()[0]
    fx = dict(d_out=d_out, d_in=d_in, r=r, paths=paths, x=x.float().tolist(), y_ref64=y64.tolist(), y_bf16=ybf.tolist(),
              note="U_packed (d_out,2) / V_packed (r,4) int32 words per the clean-room spec: LSB-first, bit 1 = -1, "
                   "right-padded with +1; l = bf16(v1*u2). y_ref64 = float64 from unpacked signs; y_bf16 = bf16 torch forward.")
    rel = (ybf.double() - y64).abs().max() / y64.abs().max()
    print("fixture bf16-vs-f64 max rel", float(rel))
    assert rel < 0.05
    with open(path, "w") as f:
        json.dump(fx, f, separators=(",", ":"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write-fixture")
    a = ap.parse_args()
    test_ranks(); test_pack(); test_smoothsign_grad(); test_toy_roundtrip_and_training()
    if a.write_fixture:
        make_fixture(a.write_fixture)
    print("SMOKE OK")
