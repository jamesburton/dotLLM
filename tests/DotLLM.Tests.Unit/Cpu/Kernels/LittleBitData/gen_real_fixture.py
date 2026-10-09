"""Generates littlebit_real_fixture.json for LittleBitRealCheckpointTests (issue #864).

Usage: python gen_real_fixture.py <littlebit export dir> <path to tools/littlebit-qat> <out.json>

For two real layers of the Qwen3-0.6B / 0.55 bpw checkpoint (a gate_proj and a k_proj, different shapes) it records a
fixed input vector x (bf16-exact), the float64 reference output of the factorized layer computed from the SAME packed
tensors (signs unpacked, scales widened from bf16, l = bf16(v1*u2) as the trainer's forward does), and the output of
the trainer's own bf16 LittleBitLinear forward (every stage rounded to bf16). Weights are NOT stored.
"""
import base64, json, sys
import numpy as np
import torch
from safetensors.torch import load_file

ckpt, qat, out = sys.argv[1:4]
sys.path.insert(0, qat)
import littlebit as lb

t = load_file(f"{ckpt}/model.safetensors")
cfg = lb.LittleBitConfig(**json.load(open(f"{ckpt}/littlebit_config.json")))
LAYERS = ["model.layers.3.mlp.gate_proj.", "model.layers.7.self_attn.k_proj."]
g = torch.Generator().manual_seed(864)


def b64(a):
    return base64.b64encode(a.tobytes()).decode()


def ref64(t, p, sfx):
    us = lb.unpack_signs(t[f"{p}U{sfx}_packed"], int(t[f"{p}U{sfx}_shape"][1]), torch.float64)
    vs = lb.unpack_signs(t[f"{p}V{sfx}_packed"], int(t[f"{p}V{sfx}_shape"][1]), torch.float64)
    f = lambda k: t[f"{p}{k}{sfx}"].to(torch.bfloat16).flatten()
    l = (f("v1") * f("u2")).double()          # bf16 product, as the trainer's forward
    return us, vs, f("u1").double(), f("v2").double(), l


cases = []
for p in LAYERS:
    residual = (p + "U_R_packed") in t
    m = lb.LittleBitLinear.from_tensors(t, p, cfg, residual=residual)
    x = torch.randn(1, m.in_f, generator=g).to(torch.bfloat16)
    y16 = m(x).float()[0]
    y64 = torch.zeros(m.out_f, dtype=torch.float64)
    for sfx in m.suffixes:
        us, vs, u1, v2, l = ref64(t, p, sfx)
        y64 += ((((x.double()[0] * v2) @ vs.t()) * l) @ us.t()) * u1
    cases.append(dict(prefix=p, d_out=m.out_f, d_in=m.in_f, r=m.r, paths=len(m.suffixes),
                      x_bf16=b64(x.view(torch.int16).numpy()[0]),
                      y_f64_as_f32=b64(y64.float().numpy()),
                      y_bf16_as_bf16=b64(y16.to(torch.bfloat16).view(torch.int16).numpy())))
json.dump(dict(source="littlebit-qwen3-0.6b-055", seed=864, cases=cases), open(out, "w"))
print([(c["prefix"], c["d_out"], c["d_in"], c["r"], c["paths"]) for c in cases])
