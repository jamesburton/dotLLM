"""LittleBit-style sub-1-bit factorized linear: init, module, packing, export/load.

Clean-room implementation written from the functional spec (.docs/LITTLEBIT_CLEANROOM_SPEC.md, not
committed) and the dotLLM spike (src/DotLLM.Cpu/Kernels/Experimental/LittleBit.cs). No third-party code.

Per path (primary, plus optional residual):  W_hat = diag(u1) Us diag(u2*v1) Vs diag(v2)
  Us in {+-1}^{out x r}, Vs in {+-1}^{r x in};  u1 (1,out)  u2 (1,r)  v1 (1,r)  v2 (1,in)
  y = sum_paths  u1 * ( ( ((x*v2) @ Vs^T) * (v1*u2) ) @ Us^T )   (+ bias)
Packed sign format: bit 1 == -1, bit 0 == +1 (sign(0) = +1), LSB-first, int32 words along the LAST
axis, right-padded with +1 (bit 0) to a multiple of 32; true sizes live in the `*_shape` tensors.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

ALPHA = 100.0  # SmoothSign surrogate sharpness (spec 2.2)


# ----------------------------------------------------------------------------- config / ranks
@dataclass
class LittleBitConfig:
    quant_func: str = "SmoothSign"   # or "STEBinary"
    eff_bit: float = 0.55
    split_dim: int = 1024            # fallback only when eff_bit is None
    residual: bool = True
    kv_factor: float = 1.0
    min_split_dim: int = 8
    quant_mod: str = "LittleBitLinear"
    use_itq: bool = True
    itq_n_iter: int = 50


def rank_for_bpw(a: int, b: int, t: float | None, residual: bool, is_kv: bool = False,
                 kv_factor: float = 1.0, min_split: int = 8, split_dim: int = 1024) -> int:
    """Spec 1.6. a = in_features, b = out_features, t = target bits/weight."""
    if t is None:
        rf = float(split_dim)
    elif residual:
        rf = (a * b * t - 32 * (a + b)) / (2 * (a + b + 16))
    else:
        rf = (a * b * t - 16 * (a + b)) / (a + b + 16)
    if is_kv:
        rf *= kv_factor
    r = int(rf)            # truncate toward zero
    r = (r // 8) * 8       # floor to multiple of 8
    if r == 0:
        r = min_split
    return max(r, min_split)


def bits_spec(a: int, b: int, r: int, residual: bool) -> float:
    """Authors' accounting (one 16-bit latent vector per path; ignores padding and *_shape)."""
    paths = 2 if residual else 1
    return paths * (r * (a + b) + 16 * (a + b) + 16 * r)


def bits_stored(a: int, b: int, r: int, residual: bool) -> float:
    """What the checkpoint really holds: v1 AND u2 are separate bf16 vectors (+16r per path)."""
    paths = 2 if residual else 1
    return bits_spec(a, b, r, residual) + paths * 16 * r


# ----------------------------------------------------------------------------- sign / packing
class SmoothSign(torch.autograd.Function):
    """forward: sign with sign(0)=+1.  backward: g * alpha*(1-tanh^2(alpha*x))."""

    @staticmethod
    def forward(ctx, x, alpha=ALPHA):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        one = torch.ones((), dtype=x.dtype, device=x.device)
        return torch.where(x >= 0, one, -one)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        return g * (a * (1 - torch.tanh(a * x) ** 2)), None


class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        one = torch.ones((), dtype=x.dtype, device=x.device)
        return torch.where(x >= 0, one, -one)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * ((x > -1) & (x < 1)).to(g.dtype)


def sign_pm1(x: torch.Tensor) -> torch.Tensor:
    one = torch.ones((), dtype=x.dtype, device=x.device)
    return torch.where(x >= 0, one, -one)


def pack_signs(s: torch.Tensor) -> torch.Tensor:
    """s: (rows, cols), any numeric; value < 0 -> bit 1 (-1), else bit 0 (+1). Returns int32 (rows, ceil(cols/32))."""
    rows, cols = s.shape
    words = (cols + 31) // 32
    bits = torch.zeros(rows, words * 32, dtype=torch.int64, device=s.device)
    bits[:, :cols] = (s < 0).to(torch.int64)
    bits = bits.view(rows, words, 32)
    w = (bits << torch.arange(32, device=s.device, dtype=torch.int64)).sum(-1)  # [0, 2^32)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)


def unpack_signs(p: torch.Tensor, cols: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Inverse of pack_signs: int32 (rows, W) -> (rows, cols) of +-1 in `dtype`."""
    rows, words = p.shape
    w = p.to(torch.int64) & 0xFFFFFFFF
    bits = (w.unsqueeze(-1) >> torch.arange(32, device=p.device, dtype=torch.int64)) & 1
    bits = bits.reshape(rows, words * 32)[:, :cols]
    return (1 - 2 * bits).to(dtype)


# ----------------------------------------------------------------------------- init (Dual-SVID + Joint-ITQ)
SVD_NITER = 2        # spec: torch.svd_lowrank default power iterations
RANK1_NITER = 4      # documented choice (spec leaves it open); nonneg matrix, converges fast


def _rank1(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-1 randomised SVD of a nonnegative matrix -> (sqrt(s)*left, sqrt(s)*right), made non-negative."""
    u, s, v = torch.svd_lowrank(a, q=1, niter=RANK1_NITER)
    l, rgt = u[:, 0], v[:, 0]
    if l.sum() < 0:
        l, rgt = -l, -rgt
    sq = s[0].sqrt()
    return sq * l, sq * rgt


def _joint_itq(ul: torch.Tensor, vl: torch.Tensor, n_iter: int, gen: torch.Generator):
    r = ul.shape[1]
    z = torch.cat([ul, vl.t()], 0)  # (out+in, r)
    R = torch.linalg.qr(torch.randn(r, r, generator=gen, device="cpu", dtype=torch.float32))[0].to(z.device)
    for _ in range(n_iter):
        b = torch.sign(z @ R)                 # raw sign (0 stays 0), per spec
        m = b.t() @ z                         # (r, r)
        p, _s, qh = torch.linalg.svd(m)
        R = qh.t() @ p.t()                    # Procrustes: Q P^T
    return ul @ R, R.t() @ vl


def _dual_svid_once(w: torch.Tensor, r: int, itq: bool, itq_iters: int, gen: torch.Generator) -> dict:
    uk, s, vk = torch.svd_lowrank(w, q=r, niter=SVD_NITER)   # w ~ uk diag(s) vk^T ; vk (in, r)
    sq = s.sqrt()
    ul = uk * sq                    # (out, r)
    vl = (vk * sq).t().contiguous()  # (r, in)
    if itq:
        ul, vl = _joint_itq(ul, vl, itq_iters, gen)
    v1, v2 = _rank1(vl.abs())       # (r), (in)
    u1, u2 = _rank1(ul.abs())       # (out), (r)
    return dict(U=ul, V=vl, u1=u1[None], u2=u2[None], v1=v1[None], v2=v2[None])


def _path_dense(d: dict) -> torch.Tensor:
    us, vs = sign_pm1(d["U"]), sign_pm1(d["V"])
    return (d["u1"].t() * us) @ ((d["u2"] * d["v1"]).t() * vs) * d["v2"]


def dual_svid_init(w: torch.Tensor, r: int, residual: bool, itq: bool, itq_iters: int, seed: int = 42) -> dict:
    """Spec section 3. w: (out, in) float. Returns fp32 tensors keyed U,V,u1,u2,v1,v2 (+ *_R)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w32 = w.detach().float()
    main = _dual_svid_once(w32, r, itq, itq_iters, gen)
    out = dict(main)
    if residual:
        res = _dual_svid_once(w32 - _path_dense(main), r, itq, itq_iters, gen)
        out.update({k + "_R": v for k, v in res.items()})
    return out


# ----------------------------------------------------------------------------- module
_PATH_KEYS = ("U", "V", "u1", "u2", "v1", "v2")


def _path_forward(x, Us, Vs, u1, u2, v1, v2):
    dt = x.dtype
    z = (x * v2.to(dt)) @ Vs.t()
    z = z * (v1.to(dt) * u2.to(dt))     # product rounded to the activation dtype first (spec 2.1/2.3)
    y = z @ Us.t()
    return y * u1.to(dt)


class LittleBitLinear(nn.Module):
    """Training form (latent float U,V -> SmoothSign) or fixed form (packed int32 buffers)."""

    def __init__(self, in_f: int, out_f: int, r: int, residual: bool, bias: torch.Tensor | None,
                 cfg: LittleBitConfig, fixed: bool = False):
        super().__init__()
        self.in_f, self.out_f, self.r, self.residual, self.fixed = in_f, out_f, r, residual, fixed
        self.cfg = cfg
        self.compute_dtype = torch.bfloat16
        self._sign_cache: dict = {}
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=not fixed)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("_eff_bit_target", torch.tensor(-1.0 if cfg.eff_bit is None else float(cfg.eff_bit)))
        self.register_buffer("_split_dim_final", torch.tensor(r, dtype=torch.int64))
        self.register_buffer("_eff_bit_actual", torch.tensor(bits_spec(in_f, out_f, r, residual) / (in_f * out_f)))

    @property
    def suffixes(self):
        return ("", "_R") if self.residual else ("",)

    @classmethod
    def from_linear(cls, lin: nn.Linear, r: int, cfg: LittleBitConfig, seed: int = 42):
        m = cls(lin.in_features, lin.out_features, r, cfg.residual, lin.bias, cfg)
        d = dual_svid_init(lin.weight, r, cfg.residual, cfg.use_itq, cfg.itq_n_iter, seed)
        for k, v in d.items():
            setattr(m, k, nn.Parameter(v.contiguous()))
        return m

    @classmethod
    def from_tensors(cls, t: dict, prefix: str, cfg: LittleBitConfig, residual: bool):
        us = t[prefix + "U_shape"].tolist()
        vs = t[prefix + "V_shape"].tolist()
        out_f, r = us
        _, in_f = vs
        bias = t.get(prefix + "bias")
        m = cls(in_f, out_f, r, residual, bias, cfg, fixed=True)
        for sfx in m.suffixes:
            for k in ("U", "V"):
                m.register_buffer(f"{k}{sfx}_packed", t[f"{prefix}{k}{sfx}_packed"].to(torch.int32))
                m.register_buffer(f"{k}{sfx}_shape", t[f"{prefix}{k}{sfx}_shape"].to(torch.int64))
            for k in ("u1", "u2", "v1", "v2"):
                m.register_buffer(f"{k}{sfx}", t[f"{prefix}{k}{sfx}"].to(torch.bfloat16))
        return m

    def _signs(self, sfx: str, dt):
        if self.fixed:
            key = (sfx, dt, self.U_packed.device)
            if key not in self._sign_cache:
                us = unpack_signs(getattr(self, f"U{sfx}_packed"), self.r, dt)
                vs = unpack_signs(getattr(self, f"V{sfx}_packed"), self.in_f, dt)
                self._sign_cache[key] = (us, vs)
            return self._sign_cache[key]
        fn = SmoothSign.apply if self.cfg.quant_func == "SmoothSign" else STEBinary.apply
        return fn(getattr(self, "U" + sfx)).to(dt), fn(getattr(self, "V" + sfx)).to(dt)

    def forward(self, x):
        dt = x.dtype
        y = None
        for sfx in self.suffixes:
            us, vs = self._signs(sfx, dt)
            p = _path_forward(x, us, vs, *[getattr(self, k + sfx) for k in ("u1", "u2", "v1", "v2")])
            y = p if y is None else y + p
        if self.bias is not None:
            y = y + self.bias.to(dt)
        return y

    def export_tensors(self, prefix: str) -> dict:
        t = {}
        for sfx in self.suffixes:
            u, v = getattr(self, "U" + sfx), getattr(self, "V" + sfx)
            t[f"{prefix}U{sfx}_packed"] = pack_signs(u.detach())
            t[f"{prefix}U{sfx}_shape"] = torch.tensor(list(u.shape), dtype=torch.int64)
            t[f"{prefix}V{sfx}_packed"] = pack_signs(v.detach())
            t[f"{prefix}V{sfx}_shape"] = torch.tensor(list(v.shape), dtype=torch.int64)
            for k in ("u1", "u2", "v1", "v2"):
                t[f"{prefix}{k}{sfx}"] = getattr(self, k + sfx).detach().to(torch.bfloat16)
        if self.bias is not None:
            t[prefix + "bias"] = self.bias.detach().to(torch.bfloat16)
        t[prefix + "_eff_bit_target"] = self._eff_bit_target.detach().float().cpu()
        t[prefix + "_split_dim_final"] = self._split_dim_final.detach().cpu()
        t[prefix + "_eff_bit_actual"] = self._eff_bit_actual.detach().float().cpu()
        return t


# ----------------------------------------------------------------------------- model conversion / export / load
def _set_module(root: nn.Module, name: str, new: nn.Module):
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def convert_model(model: nn.Module, cfg: LittleBitConfig, log=print) -> dict:
    """Replace every nn.Linear except names containing 'lm_head' by LittleBitLinear (Dual-SVID init)."""
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and "lm_head" not in n]
    ranks = {}
    for i, (n, m) in enumerate(targets):
        is_kv = n.endswith(".k_proj") or n.endswith(".v_proj")
        r = rank_for_bpw(m.in_features, m.out_features, cfg.eff_bit, cfg.residual, is_kv, cfg.kv_factor,
                         cfg.min_split_dim, cfg.split_dim)
        dev = m.weight.device
        lb = LittleBitLinear.from_linear(m, r, cfg, seed=42 + i)
        _set_module(model, n, lb)
        ranks[n] = r
        if log and (i % 20 == 0 or i == len(targets) - 1):
            log(f"  init {i + 1}/{len(targets)} {n} {m.in_features}x{m.out_features} r={r}")
        del m
    return ranks


def quantised_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LittleBitLinear)]


def fp16_range_report(tensors: dict) -> dict:
    """bf16 -> fp16 is exact only for normal fp16 magnitudes (6.1e-5 .. 65504); dotLLM's spike holds fp16 scales."""
    bad, tot = 0, 0
    for k, v in tensors.items():
        if v.dtype == torch.bfloat16:
            a = v.float().abs()
            nz = a > 0
            tot += int(nz.sum())
            bad += int(((a < 6.103515625e-05) & nz).sum() + (a > 65504).sum())
    return dict(scale_values=tot, outside_fp16_normal_range=bad)


def export_checkpoint(model: nn.Module, out_dir: str, cfg: LittleBitConfig, tokenizer=None) -> dict:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    qprefixes = []
    for n, m in quantised_modules(model):
        tensors.update({k: v.contiguous().cpu() for k, v in m.export_tensors(n + ".").items()})
        qprefixes.append(n + ".")
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    for k, v in model.state_dict().items():
        if any(k.startswith(p) for p in qprefixes):
            continue
        if tied and k == "lm_head.weight":
            continue
        tensors[k] = v.detach().to(torch.bfloat16).contiguous().cpu()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    model.config.save_pretrained(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "littlebit_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return fp16_range_report(tensors)


def load_checkpoint(path: str, dtype=torch.bfloat16, device="cpu") -> nn.Module:
    """Rebuild an HF model from `path` with LittleBitLinear (fixed, packed) layers. Needs only plain transformers."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    with open(os.path.join(path, "littlebit_config.json")) as f:
        cfg = LittleBitConfig(**json.load(f))
    t = load_file(os.path.join(path, "model.safetensors"))
    hf = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_config(hf, dtype=dtype)
    prefixes = sorted(k[: -len("U_packed")] for k in t if k.endswith(".U_packed"))
    for p in prefixes:
        _set_module(model, p[:-1], LittleBitLinear.from_tensors(t, p, cfg, residual=(p + "U_R_packed") in t))
    qp = tuple(prefixes)
    rest = {k: v for k, v in t.items() if not k.startswith(qp)}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    missing = [k for k in missing if not (k == "lm_head.weight" and hf.tie_word_embeddings)
               and not k.startswith(qp)]
    if missing or unexpected:
        raise RuntimeError(f"load mismatch missing={missing} unexpected={unexpected}")
    model.tie_weights()
    return model.to(device).eval()
