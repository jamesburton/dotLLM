"""HF reference for the Qwen4-Exp gated residual (hyper-connection) ops, issue #816 sub-PR 1.

Instantiates `Qwen4ExpTextGatedResidual` directly (its __init__ reads only hc_count / hidden_size / hc_lowrank /
rms_norm_eps). Gammas are exported GGUF-convention (HF `1 + weight`, as llama.cpp's converter folds them).
Dimensions are deliberately awkward (H=48, R=20, T=5, S=4) with per-stream magnitudes differing ~10x and
per-token / per-stream varying values so sum-vs-mean, wrong-axis and broadcast bugs cannot pass.
"""
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from qwen4exp_ref_common import FixtureWriter  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual  # noqa: E402

S, H, R, T = 4, 48, 20, 5
EPS = 1e-6


def build(use_combine, seed):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(hc_count=S, hidden_size=H, hc_lowrank=R, rms_norm_eps=EPS)
    m = Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine)
    with torch.no_grad():
        m.hc_norm.weight.copy_(torch.randn(S * H) * 0.3)
        m.input_mix_weight_down.weight.copy_(torch.randn(R, S * H) * 0.4)
        m.input_mix_weight_up.weight.copy_(torch.randn(S * H, R) * 0.6)
        if use_combine:
            m.block_inject_weight.weight.copy_(torch.randn(S, S * H) * 0.8)
    return m


def main(out):
    w = FixtureWriter(hc_count=S, hidden_size=H, lowrank=R, seq_len=T, eps=EPS)
    scales = torch.tensor([0.3, 1.0, 3.0, 9.0]).view(1, S, 1)  # per-stream magnitude spread (~30x)
    for tag, use_combine, seed in (("gr", True, 11), ("head", False, 12)):
        m = build(use_combine, seed)
        torch.manual_seed(seed + 100)
        r = (torch.randn(1, T, S, H) * scales).flatten(-2)  # [1, T, S*H]
        with torch.no_grad():
            normed = m.hc_norm(r)
            down_act = torch.nn.functional.silu(m.input_mix_weight_down(normed) / S)
            if use_combine:
                mixed, hyper_input, inj = m(r)
            else:
                mixed = m(r)
        w.add(f"{tag}.residual", r[0])
        w.add(f"{tag}.gamma", 1.0 + m.hc_norm.weight)
        w.add(f"{tag}.down", m.input_mix_weight_down.weight)   # [R, S*H]
        w.add(f"{tag}.up", m.input_mix_weight_up.weight)       # [S*H, R]
        w.add(f"{tag}.normed", normed[0])
        w.add(f"{tag}.down_act", down_act[0])                  # silu(down(xn)/S), [T, R]
        w.add(f"{tag}.mixed", mixed[0])                        # block input [T, H]
        if use_combine:
            w.add(f"{tag}.inject", m.block_inject_weight.weight)  # [S, S*H]
            w.add(f"{tag}.gains", inj[0])                         # [T, S]
            # write step exactly as the decoder layer does
            torch.manual_seed(seed + 200)
            y = torch.randn(1, T, H)
            injection = y.unsqueeze(-2) * inj.unsqueeze(-1)
            new_r = hyper_input + injection.flatten(-2)
            w.add(f"{tag}.block_out", y[0])
            w.add(f"{tag}.new_residual", new_r[0])
    w.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
