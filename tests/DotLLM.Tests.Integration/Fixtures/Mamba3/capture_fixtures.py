"""Capture Mamba-3 reference intermediate tensors for cross-reference testing.

Generates two fixtures:
  • fixture.json       — SISO (use_mimo=False)
  • fixture_mimo.json  — MIMO (use_mimo=True, R=2)

dotLLM C# tests load these and compare against their kernel outputs. Uses
VikramKarLex/mamba3-minimal as the trusted reference (Albert Gu-endorsed
pure-PyTorch impl).

Run:
    .mamba3-python-venv\\Scripts\\python.exe tests\\DotLLM.Tests.Integration\\Fixtures\\Mamba3\\capture_fixtures.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # dotLLM-mamba3 repo root
sys.path.insert(0, str(ROOT / ".mamba3-reference" / "mamba3-minimal"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402
from mamba3 import Mamba3, Mamba3Config, apply_rope, ssd, ssd_mimo  # noqa: E402


def tensor_to_fixture(t: torch.Tensor):
    """Serialise a tensor as {shape, data}. data is a flat row-major F32 list."""
    t_f32 = t.detach().to(torch.float32).contiguous()
    return {"shape": list(t_f32.shape), "data": t_f32.flatten().tolist()}


def zero_init_empty_params(layer):
    """The reference `Mamba3.__init__` uses `torch.empty(...)` for A_log, D, dt_bias
    (no explicit initializer), so their values are uninitialised memory. Lock them
    to deterministic zeros to keep fixture capture reproducible — this matches the
    original SISO fixture (which happened to get zeroed memory on first capture).
    """
    with torch.no_grad():
        layer.A_log.zero_()   # -> A = -exp(0) = -1.0
        layer.D.zero_()
        layer.dt_bias.zero_()


def capture_siso():
    """Capture SISO reference intermediates → fixture.json."""
    torch.manual_seed(42)

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
    zero_init_empty_params(layer)

    batch, seqlen = 1, 4
    u = torch.randn(batch, seqlen, cfg.d_model, dtype=torch.float32)

    with torch.no_grad():
        # Manually replicate the SISO forward path to capture intermediates.
        # Mirrors mamba3.py:313-491 forward() SISO branch.

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

        B_roped = apply_rope(B_biased, cum_angles)
        C_roped = apply_rope(C_biased, cum_angles)

        x_heads = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)

        # Two-SSD trapezoidal scan
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

    # Fixture payload
    fixture = {
        "config": {
            "d_model": cfg.d_model, "n_layer": cfg.n_layer, "vocab_size": cfg.vocab_size,
            "nheads": cfg.nheads, "headdim": cfg.headdim, "d_state": cfg.d_state,
            "d_inner": cfg.d_inner, "chunk_size": cfg.chunk_size,
            "use_mimo": cfg.use_mimo, "mimo_rank": cfg.mimo_rank,
            "batch": batch, "seqlen": seqlen,
        },
        "inputs": {
            "u": tensor_to_fixture(u),
            "in_proj_weight": tensor_to_fixture(layer.in_proj.weight),
            "out_proj_weight": tensor_to_fixture(layer.out_proj.weight),
            "A": tensor_to_fixture(A),
            "dt_raw": tensor_to_fixture(dt_raw),
            "dt_bias": tensor_to_fixture(layer.dt_bias),
            "lam_raw": tensor_to_fixture(lam_raw),
            "B_raw": tensor_to_fixture(B_raw),
            "C_raw": tensor_to_fixture(C_raw),
            "B_norm_weight": tensor_to_fixture(layer.B_norm.weight),
            "C_norm_weight": tensor_to_fixture(layer.C_norm.weight),
            "B_bias": tensor_to_fixture(layer.B_bias),
            "C_bias": tensor_to_fixture(layer.C_bias),
            "theta": tensor_to_fixture(theta),
            "x": tensor_to_fixture(x_heads),
            "D": tensor_to_fixture(layer.D),
            "z": tensor_to_fixture(z),
        },
        "activated": {
            "dt": tensor_to_fixture(dt),
            "lam": tensor_to_fixture(lam),
        },
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


def capture_mimo():
    """Capture MIMO reference intermediates → fixture_mimo.json.

    Uses use_mimo=True, mimo_rank=2. Seeded identically to SISO so tensor
    shapes match where compatible; model weights differ because of the extra
    rank axes on bias and the extra mimo_{x,z,down}_proj params.
    """
    torch.manual_seed(42)

    cfg = Mamba3Config(
        d_model=8,
        n_layer=1,
        vocab_size=32,
        expand=1,
        headdim=4,
        d_state=4,
        chunk_size=2,
        use_mimo=True,
        mimo_rank=2,
    )

    layer = Mamba3(cfg, device="cpu").eval()
    zero_init_empty_params(layer)

    batch, seqlen = 1, 4
    u = torch.randn(batch, seqlen, cfg.d_model, dtype=torch.float32)

    R = cfg.mimo_rank
    with torch.no_grad():
        # Mirrors mamba3.py:313-491 forward() MIMO branch.
        A = -torch.exp(layer.A_log)                        # (nheads,)
        proj = layer.in_proj(u)                             # (b, l, d_in_proj)
        bc_dim = cfg.d_state * R                            # MIMO

        z, x, B_raw, C_raw, dt_raw, lam_raw, theta = torch.split(
            proj,
            [cfg.d_inner, cfg.d_inner, bc_dim, bc_dim, cfg.nheads, cfg.nheads, cfg.d_state // 2],
            dim=-1,
        )

        dt = F.softplus(dt_raw + layer.dt_bias)             # (b, l, nheads)
        lam = torch.sigmoid(lam_raw)                        # (b, l, nheads)

        # Discretization
        dA = dt * rearrange(A, "h -> 1 1 h")                # (b, l, nheads)
        alpha = torch.exp(dA)
        beta = (1 - lam) * dt * alpha
        gamma = lam * dt

        # QkNorm over bc_dim = d_state*R
        B_qkn = layer.B_norm(B_raw)                         # (b, l, bc_dim)
        C_qkn = layer.C_norm(C_raw)

        # Reshape (b, l, d_state*R) -> (b, l, d_state, R)
        B_reshaped = rearrange(B_qkn, "b l (n r) -> b l n r", r=R)
        C_reshaped = rearrange(C_qkn, "b l (n r) -> b l n r", r=R)

        # Broadcast to heads + add bias: (b, l, 1, n, r) + (h, n, r) -> (b, l, h, n, r)
        B_biased = rearrange(B_reshaped, "b l n r -> b l 1 n r") + layer.B_bias
        C_biased = rearrange(C_reshaped, "b l n r -> b l 1 n r") + layer.C_bias

        # RoPE
        raw_angles = dt.unsqueeze(-1) * rearrange(theta, "b l n -> b l 1 n")
        cum_angles = -torch.cumsum(raw_angles, dim=1)       # (b, l, h, d_state/2)

        # Apply RoPE on d_state dim for each rank (per reference): move d_state last
        B_rn = rearrange(B_biased, "b l h n r -> b l h r n")  # (b,l,h,R,n)
        C_rn = rearrange(C_biased, "b l h n r -> b l h r n")
        B_rn = apply_rope(B_rn, cum_angles.unsqueeze(3))      # broadcast over rank dim
        C_rn = apply_rope(C_rn, cum_angles.unsqueeze(3))
        B_roped = rearrange(B_rn, "b l h r n -> b l h n r")   # back to (b,l,h,n,R)
        C_roped = rearrange(C_rn, "b l h r n -> b l h n r")

        # Reshape x for SSD
        x_heads = rearrange(x, "b l (h p) -> b l h p", p=cfg.headdim)  # (b,l,h,p)

        # Expand x to rank R: x_mimo[b,l,h,p,r] = x[b,l,h,p] * mimo_x_proj[h,p,r]
        x_mimo = x_heads.unsqueeze(-1) * layer.mimo_x_proj   # (b,l,h,p,R)

        # Two-SSD MIMO trapezoidal scan
        y_gamma, state_gamma = ssd_mimo(
            x_mimo * rearrange(gamma, "b l h -> b l h 1 1"),
            dA, B_roped, C_roped, cfg.chunk_size,
        )
        B_prev = F.pad(B_roped[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))      # (b,l,h,n,R)
        x_mimo_prev = F.pad(x_mimo[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))  # (b,l,h,p,R)
        y_beta, state_beta = ssd_mimo(
            x_mimo_prev * rearrange(beta, "b l h -> b l h 1 1"),
            dA, B_prev, C_roped, cfg.chunk_size,
        )
        y_scan = y_gamma + y_beta                            # (b, l, h, p, R)
        ssm_state = state_gamma + state_beta                 # (b, h, p, n)  — rank contracted

        # last_Bx: contract rank, stored per head/channel/state
        last_Bx = torch.einsum("bhnr, bhpr -> bhpn", B_roped[:, -1], x_mimo[:, -1])

        # D skip in rank-R space — D is per-head, broadcast over channel and rank
        y_with_d = y_scan + (x_heads * layer.D.unsqueeze(-1)).unsqueeze(-1)  # (b,l,h,p,R)

        # Gate in rank-R space: z_mimo[b,l,h,p,r] = z[b,l,h,p] * mimo_z_proj[h,p,r]
        z_heads = rearrange(z, "b l (h p) -> b l h p", p=cfg.headdim)
        z_mimo = z_heads.unsqueeze(-1) * layer.mimo_z_proj   # (b,l,h,p,R)
        y_gated_rank = y_with_d * F.silu(z_mimo)             # (b,l,h,p,R)

        # Contract rank → (b,l,h,p) via weighted sum with mimo_down
        y_contracted = (y_gated_rank * layer.mimo_down).sum(dim=-1)   # (b,l,h,p)
        y_flat = rearrange(y_contracted, "b l h p -> b l (h p)")
        y_final = layer.out_proj(y_flat)                     # (b, l, d_model)

    fixture = {
        "config": {
            "d_model": cfg.d_model, "n_layer": cfg.n_layer, "vocab_size": cfg.vocab_size,
            "nheads": cfg.nheads, "headdim": cfg.headdim, "d_state": cfg.d_state,
            "d_inner": cfg.d_inner, "chunk_size": cfg.chunk_size,
            "use_mimo": cfg.use_mimo, "mimo_rank": cfg.mimo_rank, "bc_dim": bc_dim,
            "batch": batch, "seqlen": seqlen,
        },
        "inputs": {
            "u": tensor_to_fixture(u),
            "in_proj_weight": tensor_to_fixture(layer.in_proj.weight),
            "out_proj_weight": tensor_to_fixture(layer.out_proj.weight),
            "A": tensor_to_fixture(A),
            "dt_raw": tensor_to_fixture(dt_raw),
            "dt_bias": tensor_to_fixture(layer.dt_bias),
            "lam_raw": tensor_to_fixture(lam_raw),
            # B_raw/C_raw are length bc_dim = d_state*R now
            "B_raw": tensor_to_fixture(B_raw),                       # (b, l, bc_dim)
            "C_raw": tensor_to_fixture(C_raw),
            "B_norm_weight": tensor_to_fixture(layer.B_norm.weight), # (bc_dim,)
            "C_norm_weight": tensor_to_fixture(layer.C_norm.weight),
            "B_bias": tensor_to_fixture(layer.B_bias),               # (nheads, d_state, R)
            "C_bias": tensor_to_fixture(layer.C_bias),
            "theta": tensor_to_fixture(theta),
            "x": tensor_to_fixture(x_heads),                         # (b, l, h, p) scan input
            "D": tensor_to_fixture(layer.D),
            "z": tensor_to_fixture(z),                               # (b, l, d_inner)
            "mimo_x_proj": tensor_to_fixture(layer.mimo_x_proj),     # (nheads, headdim, R)
            "mimo_z_proj": tensor_to_fixture(layer.mimo_z_proj),
            "mimo_down": tensor_to_fixture(layer.mimo_down),
        },
        "activated": {
            "dt": tensor_to_fixture(dt),
            "lam": tensor_to_fixture(lam),
        },
        "expected": {
            "alpha": tensor_to_fixture(alpha),
            "beta": tensor_to_fixture(beta),
            "gamma": tensor_to_fixture(gamma),
            "B_qkn": tensor_to_fixture(B_qkn),                       # (b, l, bc_dim)
            "C_qkn": tensor_to_fixture(C_qkn),
            "B_biased": tensor_to_fixture(B_biased),                 # (b, l, h, d_state, R)
            "C_biased": tensor_to_fixture(C_biased),
            "cum_angles": tensor_to_fixture(cum_angles),             # (b, l, h, d_state/2)
            "B_roped": tensor_to_fixture(B_roped),                   # (b, l, h, d_state, R)
            "C_roped": tensor_to_fixture(C_roped),
            "x_mimo": tensor_to_fixture(x_mimo),                     # (b, l, h, p, R)
            "y_scan": tensor_to_fixture(y_scan),                     # (b, l, h, p, R)
            "ssm_state": tensor_to_fixture(ssm_state),               # (b, h, p, d_state)
            "last_Bx": tensor_to_fixture(last_Bx),                   # (b, h, p, d_state)
            "y_with_d": tensor_to_fixture(y_with_d),                 # (b, l, h, p, R)
            "z_mimo": tensor_to_fixture(z_mimo),                     # (b, l, h, p, R)
            "y_gated_rank": tensor_to_fixture(y_gated_rank),         # (b, l, h, p, R)
            "y_contracted": tensor_to_fixture(y_contracted),         # (b, l, h, p)
            "y_final": tensor_to_fixture(y_final),                   # (b, l, d_model)
        },
    }

    out_path = Path(__file__).parent / "fixture_mimo.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=None, separators=(",", ":"))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    print(f"  B_roped shape: {fixture['expected']['B_roped']['shape']}")
    print(f"  y_scan shape:  {fixture['expected']['y_scan']['shape']}")
    print(f"  y_final shape: {fixture['expected']['y_final']['shape']}")


def main():
    capture_siso()
    capture_mimo()


if __name__ == "__main__":
    main()
