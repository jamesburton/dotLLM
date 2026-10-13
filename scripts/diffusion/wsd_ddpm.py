"""Weight-space diffusion Exp 0 — pure DDPM core + reconstruction metrics (CPU-only).

Uses x0-prediction (the network predicts the clean vector, not the noise) and deterministic
DDIM sampling. Rationale (found empirically during Exp-0 bring-up): a tiny MLP cannot fit the
epsilon-prediction target for small t (the eps target blows up as 1/sqrt(1-alpha_bar_t)),
leaving a loss floor that prevents single-point reconstruction. x0-prediction has a bounded
target and memorizes/represents weight vectors cleanly. beta_end is 0.10 so that alpha_bar_T ~ 0
over T=100 steps (with the default 0.02 the forward process only reaches alpha_bar_T ~ 0.36,
which mismatches sampling from pure N(0,1)).
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Schedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.10) -> Schedule:
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return Schedule(betas, alphas, alpha_bars)


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sched: Schedule) -> torch.Tensor:
    ab = sched.alpha_bars[t].unsqueeze(-1)               # [B,1]
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    # sinusoidal embedding, t: [B] long -> [B, dim]
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DenoiseMLP(nn.Module):
    """3-layer x0-predictor with additive sinusoidal timestep conditioning.

    forward(x_t, t) -> x0_hat  (an estimate of the clean vector).
    """
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(dim, hidden)
        self.time = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(_timestep_embedding(t, self.hidden))
        h = F.silu(self.l1(x_t) + temb)
        h = F.silu(self.l2(h))
        return self.l3(h)


def train_ddpm(model: nn.Module, targets: torch.Tensor, sched: Schedule,
               epochs: int, lr: float, seed: int) -> float:
    """x0-prediction training: predict the clean vector from its noised version. Returns final loss."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    T = sched.betas.shape[0]
    K = targets.shape[0]
    loss_val = float("nan")
    for _ in range(epochs):
        t = torch.randint(0, T, (K,), generator=g)
        noise = torch.randn(targets.shape, generator=g)
        x_t = q_sample(targets, t, noise, sched)
        x0_hat = model(x_t, t)
        loss = F.mse_loss(x0_hat, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_val = float(loss.detach())
    return loss_val


@torch.no_grad()
def sample_ddpm(model: nn.Module, sched: Schedule, n: int, dim: int, seed: int) -> torch.Tensor:
    """Deterministic DDIM sampling from an x0-prediction model.

    Deterministic given `seed` (only x_T is random); identical seed -> identical output.
    """
    g = torch.Generator().manual_seed(seed)
    model.eval()
    T = sched.betas.shape[0]
    x = torch.randn(n, dim, generator=g)
    # Data is normalized to ~unit scale; clamp keeps the eps-reconstruction (which divides by
    # sqrt(1-alpha_bar_t) ~ 0.01 at low t) from amplifying x0 errors into inf/NaN over the chain.
    _C = 10.0
    for t in range(T - 1, -1, -1):
        tb = torch.full((n,), t, dtype=torch.long)
        x0_hat = model(x, tb).clamp(-_C, _C)
        ab_t = sched.alpha_bars[t]
        ab_prev = sched.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
        eps = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt(1.0 - ab_t)
        x = (torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps).clamp(-_C, _C)
    return x


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12))


def nearest_target(sample: torch.Tensor, targets: torch.Tensor):
    cs = [cosine(sample, targets[i]) for i in range(targets.shape[0])]
    idx = int(max(range(len(cs)), key=lambda i: cs[i]))
    return idx, cs[idx], rel_l2(sample, targets[idx])
