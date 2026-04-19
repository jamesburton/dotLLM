"""Capture Mamba-3 reference intermediate tensors for cross-reference testing.

Generates a single fixture.json that dotLLM C# tests can load and compare
against their kernel outputs. Uses VikramKarLex/mamba3-minimal as the trusted
reference (Albert Gu-endorsed pure-PyTorch impl).

Run:
    .mamba3-python-venv\\Scripts\\python.exe tests\\DotLLM.Tests.Integration\\Fixtures\\Mamba3\\capture_fixtures.py

Output: fixture.json next to this script.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # dotLLM-mamba3 repo root
sys.path.insert(0, str(ROOT / ".mamba3-reference" / "mamba3-minimal"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from mamba3 import Mamba3, Mamba3Config  # noqa: E402


def tensor_to_fixture(t: torch.Tensor):
    """Serialise a tensor as {shape, data}. data is a flat row-major F32 list."""
    t_f32 = t.detach().to(torch.float32).contiguous()
    return {"shape": list(t_f32.shape), "data": t_f32.flatten().tolist()}


def main():
    torch.manual_seed(42)

    # Tiny SISO config for fast fixture generation + easy eye-balling.
    cfg = Mamba3Config(
        d_model=8,
        n_layer=1,
        vocab_size=32,
        expand=1,          # d_inner = expand * d_model = 8
        headdim=4,         # nheads = d_inner / headdim = 2
        d_state=4,
        chunk_size=2,
        use_mimo=False,
        mimo_rank=1,
    )

    layer = Mamba3(cfg, device="cpu").eval()

    batch, seqlen = 1, 4
    u = torch.randn(batch, seqlen, cfg.d_model, dtype=torch.float32)

    with torch.no_grad():
        # ── Manually replicate the SISO forward path to capture intermediates. ──
        # Mirrors mamba3.py:313-490 forward() SISO branch.

        A = -torch.exp(layer.A_log)                       # (nheads,)
        proj = layer.in_proj(u)                            # (b, l, d_in_proj)
        bc_dim = cfg.d_state                               # SISO
        z, x, B_raw, C_raw, dt_raw, lam_raw, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, bc_dim, bc_dim, cfg.nheads, cfg.nheads, cfg.d_state // 2],
            dim=-1,
        )

        dt = F.softplus(dt_raw + layer.dt_bias)            # (b, l, nheads)
        lam = torch.sigmoid(lam_raw)                       # (b, l, nheads)

        # Discretization
        dA = dt * rearrange(A, "h -> 1 1 h")               # (b, l, nheads)
        alpha = torch.exp(dA)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt

        # QkNorm + BC-bias + RoPE
        B_qkn = layer.B_norm(B_raw)
        C_qkn = layer.C_norm(C_raw)
        B_biased = rearrange(B_qkn, "b l n -> b l 1 n") + layer.B_bias   # (b, l, h, n)
        C_biased = rearrange(C_qkn, "b l n -> b l 1 n") + layer.C_bias

        raw_angles = dt.unsqueeze(-1) * rearrange(theta, "b l n -> b l 1 n")
        cum_angles = -torch.cumsum(raw_angles, dim=1)      # (b, l, h, d_state/2)

        from mamba3 import apply_rope
        B_roped = apply_rope(B_biased, cum_angles)
        C_roped = apply_rope(C_biased, cum_angles)

        x_heads = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)

        # Two-SSD trapezoidal scan
        from mamba3 import ssd
        y_gamma, state_gamma = ssd(
            x_heads * rearrange(gamma, "b l h -> b l h 1"),
            dA, B_roped, C_roped, cfg.chunk_size,
        )
        B_prev = F.pad(B_roped[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = F.pad(x_heads[:, :-1], (0, 0, 0, 0, 1, 0))
        y_beta, state_beta = ssd(
            x_prev * rearrange(beta, "b l h -> b l h 1"),
            dA, B_prev, C_roped, cfg.chunk_size,
        )
        y_scan = y_gamma + y_beta                          # (b, l, h, p)
        ssm_state = state_gamma + state_beta               # (b, h, p, n)

        last_Bx = torch.einsum("bhn,bhp->bhpn", B_roped[:, -1], x_heads[:, -1])

        # Post-D skip + gate + out_proj → full block output
        y_with_d = y_scan + x_heads * layer.D.view(1, 1, -1, 1)
        y_gated = rearrange(y_with_d, "b l h p -> b l (h p)") * F.silu(z)
        y_final = layer.out_proj(y_gated)  # (b, l, d_model)

    # ── Fixture payload ──
    fixture = {
        "config": {
            "d_model": cfg.d_model, "n_layer": cfg.n_layer, "vocab_size": cfg.vocab_size,
            "nheads": cfg.nheads, "headdim": cfg.headdim, "d_state": cfg.d_state,
            "d_inner": cfg.d_inner, "chunk_size": cfg.chunk_size,
            "use_mimo": cfg.use_mimo, "mimo_rank": cfg.mimo_rank,
            "batch": batch, "seqlen": seqlen,
        },
        # Inputs we'll feed into our kernels
        "inputs": {
            "u": tensor_to_fixture(u),                     # (b, l, d_model) — block input
            "in_proj_weight": tensor_to_fixture(layer.in_proj.weight),   # (d_in_proj, d_model)
            "out_proj_weight": tensor_to_fixture(layer.out_proj.weight), # (d_model, d_inner)
            "A": tensor_to_fixture(A),                     # (nheads,)
            "dt_raw": tensor_to_fixture(dt_raw),            # (b, l, nheads)
            "dt_bias": tensor_to_fixture(layer.dt_bias),    # (nheads,)
            "lam_raw": tensor_to_fixture(lam_raw),          # (b, l, nheads)
            "B_raw": tensor_to_fixture(B_raw),              # (b, l, d_state)
            "C_raw": tensor_to_fixture(C_raw),              # (b, l, d_state)
            "B_norm_weight": tensor_to_fixture(layer.B_norm.weight),    # (d_state,)
            "C_norm_weight": tensor_to_fixture(layer.C_norm.weight),
            "B_bias": tensor_to_fixture(layer.B_bias),      # (nheads, d_state)
            "C_bias": tensor_to_fixture(layer.C_bias),
            "theta": tensor_to_fixture(theta),              # (b, l, d_state/2)
            "x": tensor_to_fixture(x_heads),                # (b, l, h, p) scan input
            "D": tensor_to_fixture(layer.D),
            "z": tensor_to_fixture(z),
        },
        # Post-activation inputs to our Discretize kernel
        "activated": {
            "dt": tensor_to_fixture(dt),                    # softplus applied
            "lam": tensor_to_fixture(lam),                  # sigmoid applied
        },
        # Expected outputs from each kernel
        "expected": {
            "alpha": tensor_to_fixture(alpha),
            "beta": tensor_to_fixture(beta),
            "gamma": tensor_to_fixture(gamma),
            "B_qkn": tensor_to_fixture(B_qkn),
            "C_qkn": tensor_to_fixture(C_qkn),
            "B_biased": tensor_to_fixture(B_biased),
            "C_biased": tensor_to_fixture(C_biased),
            "cum_angles": tensor_to_fixture(cum_angles),
            "B_roped": tensor_to_fixture(B_roped),
            "C_roped": tensor_to_fixture(C_roped),
            "y_scan": tensor_to_fixture(y_scan),
            "ssm_state": tensor_to_fixture(ssm_state),
            "last_Bx": tensor_to_fixture(last_Bx),
            "y_gated_pre_outproj": tensor_to_fixture(y_gated),
            "y_final": tensor_to_fixture(y_final),
        },
    }

    out_path = Path(__file__).parent / "fixture.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=None, separators=(",", ":"))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  alpha shape: {fixture['expected']['alpha']['shape']}")
    print(f"  B_roped shape: {fixture['expected']['B_roped']['shape']}")
    print(f"  y_scan shape: {fixture['expected']['y_scan']['shape']}")


if __name__ == "__main__":
    main()
