#!/usr/bin/env python3
"""bitdistill.py — BitNet-Distillation conversion harness (arXiv 2510.13998).

Ternarize a full-precision **dense** LLM (default ``Qwen/Qwen3-0.6B``) to 1.58-bit
{-1,0,+1} weights via the SOTA "BitNet Distillation" recipe, distilling from a
frozen FP teacher (the same base model).

Recipe implemented
-------------------
1. **Modeling refinement (SubLN).** Insert an RMSNorm right before the *output
   projection* of attention (``o_proj``) and before the *output projection* of
   the FFN (``down_proj``) in every layer — the exact BitDistill / BitNet SubLN
   placement. Then wrap every attention + FFN Linear (q/k/v/o, gate/up/down) as a
   **BitLinear**: absmean per-tensor ternary weights with a straight-through
   estimator (STE) on the backward, and 8-bit per-token activation quantization
   on the forward. BitLinears are **initialised from the teacher weights**, not
   from scratch. Embeddings, ``lm_head``, and all norms stay in full precision.
   Qwen3 specifics preserved: SwiGLU (silu) FFN, GQA, RMSNorm, QK-norm, RoPE.

2. **Distillation loss.**
       L = L_CE  +  λ · L_LD  +  γ · L_AD
   * ``L_CE``  — causal LM cross-entropy on the student.
   * ``L_LD``  — logit distillation: KL(teacher‖student) at temperature τ=5,
                 scaled by τ² (Hinton).
   * ``L_AD``  — MiniLM-style multi-head self-attention *relation* distillation on
                 a SINGLE layer: KL between teacher and student relation matrices
                 R = softmax(A Aᵀ / √d_r) for A ∈ {Q, K, V}.
   Paper regimes (both CLI flags): classification λ=10, γ=1e5 (see caveat: the
   attention-relation weight is tiny in practice — this harness DEFAULTS γ=1e-5,
   which is what the requesting recipe note specified; flip with ``--gamma``).

3. **Continual-pretrain (CPT) warm-up + progressive precision.** Optional λ-ramp
   and a bf16→ternary anneal (``--precision-warmup-steps``) blend FP and ternary
   forward for early-step stability. CPT corpus is a streaming general-web slice
   (FineWeb-Edu / FALCON family); task slice (GSM8K) is used for eval.

4. **Budget curve.** ``--milestones`` checkpoints + evals at token milestones
   {0.25B, 0.5B, 1B, 2B, 5B}: records task-accuracy + PPL vs tokens — the go/no-go
   signal (accuracy must RISE toward FP, not just PPL).

5. **Export.** ``bitdistill_export.py`` writes the trained ternary student to a
   dotLLM-loadable bitnet-style checkpoint (I2_S at load). See that file's header
   for the exact tensor mapping and the dotLLM-side gaps (silu-SwiGLU expert path
   + QK-norm in the BitNet attention kernel).

Self-test (CPU, seconds, no download)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 python scripts/lora/bitdistill.py --self-test

Tiny real-model smoke (a few CPU steps on the actual Qwen3-0.6B)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 python scripts/lora/bitdistill.py \
        --base Qwen/Qwen3-0.6B --tiny-random-corpus --max-steps 3 --device cpu

GPU budget-curve (launch when a GPU frees — see the deliverable notes)::

    CUDA_VISIBLE_DEVICES=0 TORCHDYNAMO_DISABLE=1 python scripts/lora/bitdistill.py \
        --base Qwen/Qwen3-0.6B --device cuda --tokens 5e9 \
        --milestones 0.25e9,0.5e9,1e9,2e9,5e9 --lambda-kd 10 --gamma 1e-5 --tau 5 \
        --batch-size 32 --max-seq-len 512 --lr 1e-4 \
        --out .docs/bitdistill/qwen3_0p6b_curve
"""

from __future__ import annotations

# Windows: torch.compile's Triton/Inductor back-end needs cl.exe; suppress dynamo.
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill_data as bdata  # noqa: E402
import csharp_exec_eval as cse  # noqa: E402


# ===========================================================================
# Quantizers (match transformers WeightQuant + BitNet absmean semantics)
# ===========================================================================
def weight_quant_ternary(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternary quantization -> values in {-s, 0, +s}.

    scale = 1 / mean(|w|).clamp(min=1e-5);  w_q = round(w*scale).clamp(-1,1) / scale.
    For an all-zero w, mean(|w|)=0 clamps to 1e-5 -> scale=1e5 -> w_q=0 exactly
    (no NaN). Identical to transformers.integrations.bitnet.WeightQuant.
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale


def activation_quant_8bit(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax symmetric 8-bit activation quantization (int8 range)."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    return (x * scale).round().clamp_(-128, 127) / scale


def _ste(real: torch.Tensor, quant: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward = quant, backward = identity to real."""
    return real + (quant - real).detach()


# ===========================================================================
# BitLinear — ternary-weight, 8-bit-activation linear with optional SubLN
# ===========================================================================
class BitLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` (bias-free) with:

    * FP *master* weight (trained; ternary applied only in forward via STE),
    * absmean ternary weight quantization,
    * 8-bit per-token activation quantization,
    * an OPTIONAL internal ``sub_norm`` (RMSNorm) applied to the input before
      quantization — this is the BitDistill/BitNet SubLN placed right before the
      output projection (used for ``o_proj`` and ``down_proj``).

    A module-level float ``quant_alpha`` in [0,1] anneals FP→ternary for the
    progressive-precision warm-up: forward blends ``(1-α)·fp + α·quant`` for both
    weight and activation. α=1 is fully ternary/8-bit (steady state).
    """

    def __init__(self, in_features: int, out_features: int, sub_norm: bool,
                 rms_eps: float = 1e-6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # SubLN: RMSNorm over the input feature dim (kept in full precision).
        self.sub_norm = _RMSNorm(in_features, eps=rms_eps) if sub_norm else None
        # Annealing scalar (buffer so it moves with .to(device); not a parameter).
        self.register_buffer("quant_alpha", torch.tensor(1.0), persistent=False)

    @classmethod
    def from_linear(cls, lin: nn.Linear, sub_norm: bool, rms_eps: float = 1e-6) -> "BitLinear":
        m = cls(lin.in_features, lin.out_features, sub_norm=sub_norm, rms_eps=rms_eps)
        with torch.no_grad():
            m.weight.copy_(lin.weight)          # init from teacher weights
        assert lin.bias is None, "BitLinear assumes bias-free Linear (Qwen3/BitNet)"
        # Adopt the source Linear's dtype so a bf16 model yields bf16 BitLinear master
        # weights (and bf16 SubLN) — otherwise the fresh float32 params mismatch the
        # bf16 activations in F.linear (the GPU train() path has no autocast to hide it).
        m.to(lin.weight.dtype)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sub_norm is not None:
            x = self.sub_norm(x)
        alpha = float(self.quant_alpha)

        # Activation: 8-bit with STE, annealed.
        xq = _ste(x, activation_quant_8bit(x))
        x_eff = x if alpha <= 0.0 else (xq if alpha >= 1.0 else (1 - alpha) * x + alpha * xq)

        # Weight: ternary with STE, annealed.
        w = self.weight
        wq = _ste(w, weight_quant_ternary(w))
        w_eff = w if alpha <= 0.0 else (wq if alpha >= 1.0 else (1 - alpha) * w + alpha * wq)

        return F.linear(x_eff, w_eff)


class _RMSNorm(nn.Module):
    """Minimal RMSNorm (weight init to ones) used for the inserted SubLN."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


# ===========================================================================
# Student construction: SubLN + BitLinear conversion of a Qwen3 model
# ===========================================================================
# The attention/FFN Linears to ternarize. o_proj / down_proj carry the SubLN.
_ATTN_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")
_SUBNORM_LINEARS = ("o_proj", "down_proj")


def convert_to_bitnet_student(model, rms_eps: float = 1e-6) -> dict:
    """In-place transform a Qwen3 (or Qwen3-like) HF model into a BitDistill student.

    For every decoder layer: replace attention + FFN Linears with BitLinear
    (init from the existing weights); ``o_proj`` and ``down_proj`` gain an internal
    SubLN RMSNorm. Embeddings / lm_head / layernorms / q_norm / k_norm untouched.

    Returns a small info dict (counts) for logging / assertions.
    """
    n_bitlinear = 0
    n_subnorm = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in _ATTN_LINEARS:
            lin = getattr(attn, name)
            bl = BitLinear.from_linear(lin, sub_norm=(name in _SUBNORM_LINEARS), rms_eps=rms_eps)
            setattr(attn, name, bl)
            n_bitlinear += 1
            n_subnorm += 1 if bl.sub_norm is not None else 0
        mlp = layer.mlp
        for name in _MLP_LINEARS:
            lin = getattr(mlp, name)
            bl = BitLinear.from_linear(lin, sub_norm=(name in _SUBNORM_LINEARS), rms_eps=rms_eps)
            setattr(mlp, name, bl)
            n_bitlinear += 1
            n_subnorm += 1 if bl.sub_norm is not None else 0
    return {"bitlinears": n_bitlinear, "subnorms": n_subnorm,
            "layers": len(model.model.layers)}


def set_quant_alpha(model, alpha: float) -> None:
    """Set the FP→ternary anneal scalar on every BitLinear (0=FP, 1=ternary)."""
    a = float(max(0.0, min(1.0, alpha)))
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.quant_alpha.fill_(a)


class _SiluReluSquaredBlend(nn.Module):
    """FFN activation that anneals silu -> relu2: (1-a)*silu(x) + a*relu2(x).

    ``a`` is a settable buffer (0 = pure silu, 1 = pure relu2). Ramping a 0->1 over
    training eases the source-target activation gap when distilling a silu teacher
    into a relu2 student, instead of taking the full silu->relu2 jump at step 0.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = float(self.alpha)
        s = F.silu(x)
        if a <= 0.0:
            return s
        r = F.relu(x).pow(2)
        return r if a >= 1.0 else (1.0 - a) * s + a * r


def set_ffn_anneal(model, alpha: float) -> None:
    """Set the silu->relu2 anneal scalar on every blended FFN activation."""
    a = float(max(0.0, min(1.0, alpha)))
    for m in model.modules():
        if isinstance(m, _SiluReluSquaredBlend):
            m.alpha.fill_(a)


def set_student_ffn_activation(model, activation: str, anneal_steps: int = 0) -> int:
    """Override each decoder MLP's activation (default silu-SwiGLU -> relu2 ablation).

    Squared-ReLU-GLU (relu2) is the canonical BitNet-2B FFN activation. Distilling a
    silu teacher into a relu2 student tests whether relu2 yields a better ternary
    student (the student must absorb the activation change on top of ternarization;
    the TEACHER keeps silu, so the KD targets are unchanged). With ``anneal_steps>0``
    the student's FFN activation is a silu->relu2 blend (ramp a 0->1 externally via
    :func:`set_ffn_anneal`) to ease the source-target gap. Updates config.hidden_act
    so the exported dotLLM checkpoint round-trips the (final) activation.
    """
    if activation == "silu":
        return 0
    if activation != "relu2":
        raise ValueError(f"unsupported ffn activation {activation!r}")

    if anneal_steps > 0:
        make_act = _SiluReluSquaredBlend  # starts at silu (a=0), annealed to relu2
    else:
        try:
            from transformers.activations import ACT2FN
            hard = ACT2FN["relu2"]
        except (ImportError, KeyError):
            hard = lambda x: F.relu(x).pow(2)  # noqa: E731
        make_act = lambda: hard  # noqa: E731

    n = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "act_fn"):
            mlp.act_fn = make_act()
            n += 1
    model.config.hidden_act = "relu2"
    return n


# ===========================================================================
# Distillation losses
# ===========================================================================
def logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  tau: float) -> torch.Tensor:
    """KL(teacher ‖ student) at temperature ``tau``, scaled by tau² (Hinton).

    Penalises the student for missing teacher mass — the standard soft-target
    distillation direction. Teacher logits are detached.
    """
    V = student_logits.size(-1)
    t = teacher_logits.to(student_logits.device).float().reshape(-1, V).detach()
    s = student_logits.float().reshape(-1, V)
    s_logp = F.log_softmax(s / tau, dim=-1)
    t_p = F.softmax(t / tau, dim=-1)
    # F.kl_div(log_p_student, p_teacher) = sum p_teacher * (log p_teacher - log p_student)
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (tau * tau)


def sparse_logit_kd_loss(student_logits: torch.Tensor, teacher_topk_logits: torch.Tensor,
                         teacher_topk_idx: torch.Tensor, teacher_lse: torch.Tensor,
                         tau: float) -> torch.Tensor:
    """Top-K sparse variant of :func:`logit_kd_loss` for an offline-cached teacher.

    Uses the teacher's TRUE top-K probabilities (no renormalisation): with the cached
    per-token log-sum-exp of the full teacher logits at ``tau``, the true probability of
    top-K token k is ``exp(logit_k/tau - lse)``. The KL then simply drops the (small)
    tail mass outside top-K rather than re-inflating the kept mass to 1 — so it never
    sharpens the teacher and converges to :func:`logit_kd_loss` as K → vocab.

    ``student_logits`` [N, V]; ``teacher_topk_logits`` / ``teacher_topk_idx`` [N, K];
    ``teacher_lse`` [N] = ``logsumexp(teacher_logits / tau, dim=-1)``.
    """
    t_logp_k = teacher_topk_logits.float() / tau - teacher_lse.float().unsqueeze(-1)  # [N, K] true log-probs
    t_p_k = t_logp_k.exp()                                                            # [N, K] true (partial) probs
    s_logp = F.log_softmax(student_logits.float() / tau, dim=-1)                      # [N, V]
    s_logp_k = torch.gather(s_logp, -1, teacher_topk_idx)                             # [N, K]
    # KL(t‖s) restricted to the top-K support (tail mass dropped), mean over N rows.
    kl = (t_p_k * (t_logp_k - s_logp_k)).sum(-1).mean()
    return kl * (tau * tau)


def _relation(a: torch.Tensor, n_rel_heads: int) -> torch.Tensor:
    """MiniLM relation matrix for a projection output ``a`` of shape [B, T, C].

    Reshape C into ``n_rel_heads`` relation heads of size d_r = C / R, then
    R = softmax(a aᵀ / √d_r) over the last (key) dim -> [B, R, T, T].
    """
    B, T, C = a.shape
    dr = C // n_rel_heads
    a = a.view(B, T, n_rel_heads, dr).permute(0, 2, 1, 3)      # [B, R, T, dr]
    scores = torch.matmul(a, a.transpose(-1, -2)) / math.sqrt(dr)  # [B, R, T, T]
    return F.softmax(scores, dim=-1)


def attn_relation_loss(student_qkv: dict, teacher_qkv: dict,
                       n_rel_heads: int) -> torch.Tensor:
    """MiniLM-style Q-Q / K-K / V-V relation KL on one captured layer.

    ``*_qkv`` dicts hold {"q","k","v"} = the captured projection OUTPUTS
    [B, T, C]. Q width differs from K/V width under GQA — each relation is
    computed within its own width. Returns the mean KL over the three relations.
    """
    total = None
    for key in ("q", "k", "v"):
        s = student_qkv[key].float()
        t = teacher_qkv[key].float().detach()
        # relation-head count must divide this projection's width
        C = s.size(-1)
        R = n_rel_heads
        while C % R != 0 and R > 1:
            R -= 1
        s_rel = _relation(s, R)                       # [B, R, T, T]
        t_rel = _relation(t, R)
        T = s_rel.size(-1)
        # Per-distribution mean KL: reshape to [B*R*T, T] so batchmean divides by
        # the number of relation distributions (seq-len-independent -> gamma stable).
        kl = F.kl_div(s_rel.clamp_min(1e-9).log().reshape(-1, T),
                      t_rel.reshape(-1, T), reduction="batchmean")
        total = kl if total is None else total + kl
    return total / 3.0


# ---- QKV capture hooks -----------------------------------------------------
class QKVCapture:
    """Register forward hooks on one layer's q/k/v projections and stash outputs.

    Works for both the FP teacher (nn.Linear) and the BitLinear student — the
    hook reads the module OUTPUT tensor either way.
    """

    def __init__(self, model, layer_idx: int):
        self.buf: dict = {}
        self.handles = []
        attn = model.model.layers[layer_idx].self_attn
        for key, mod in (("q", attn.q_proj), ("k", attn.k_proj), ("v", attn.v_proj)):
            self.handles.append(mod.register_forward_hook(self._make(key)))

    def _make(self, key):
        def hook(_m, _inp, out):
            self.buf[key] = out
        return hook

    def get(self) -> dict:
        return dict(self.buf)

    def remove(self):
        for h in self.handles:
            h.remove()


# ===========================================================================
# Eval: PPL + GSM8K accuracy
# ===========================================================================
def compute_ppl(model, seqs: list, device) -> float:
    if not seqs:
        return float("nan")
    was_training = model.training
    model.eval()
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for seq in seqs:
            s = seq.unsqueeze(0).to(device)
            logits = model(input_ids=s, use_cache=False).logits
            V = logits.size(-1)
            nll = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, V),
                s[:, 1:].contiguous().view(-1), reduction="sum",
            ).item()
            tot_nll += nll
            tot_tok += s.size(1) - 1
    if was_training:
        model.train()
    return math.exp(min(tot_nll / max(tot_tok, 1), 100.0))


def _extract_number(text: str) -> str:
    import re
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else ""


def eval_gsm8k(model, tokenizer, examples: list, device,
               max_new_tokens: int = 256, chunk: int = 16) -> float:
    """Greedy-decode GSM8K prompts (batched, left-padded); exact-match accuracy of the
    final generated number. Generation temporarily forces ``use_cache=True`` (the student
    trains with it off, which makes autoregressive decode O(n²)); with few-shot CoT prompts
    the answer is the last number before the model rolls into the next ``Question:``."""
    if not examples:
        return float("nan")
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    correct = 0
    with torch.no_grad():
        for i in range(0, len(examples), chunk):
            batch = examples[i:i + chunk]
            maxlen = max(e["prompt_ids"].size(0) for e in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for j, e in enumerate(batch):  # left-pad so all sequences end aligned
                L = e["prompt_ids"].size(0)
                input_ids[j, maxlen - L:] = e["prompt_ids"]
                attn[j, maxlen - L:] = 1
            out = model.generate(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                                 max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            for j, e in enumerate(batch):
                gen = tokenizer.decode(out[j, maxlen:], skip_special_tokens=True).split("Question:")[0]
                if _extract_number(gen) == e["answer"] and e["answer"] != "":
                    correct += 1
    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    return correct / len(examples)


# ===========================================================================
# Checkpoint save (student BitLinear master weights + sub_norms)
# ===========================================================================
def save_checkpoint(model, out_dir: str, step: int, tokens: int, extra: Optional[dict] = None):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(sd, os.path.join(out_dir, "student_state.pt"))
    state = {"step": step, "tokens": tokens}
    if extra:
        state.update(extra)
    with open(os.path.join(out_dir, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ===========================================================================
# Self-test — tiny synthetic Qwen3, no download
# ===========================================================================
def build_tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.eval()
    return m


def run_self_test(tau: float = 5.0, lambda_kd: float = 10.0, gamma: float = 1e-5) -> bool:
    import copy
    print("[self-test] building tiny synthetic Qwen3 (hidden=32, 2 layers, GQA 4/2)...")
    teacher = build_tiny_qwen3()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    student = copy.deepcopy(teacher)
    info = convert_to_bitnet_student(student)
    print(f"[self-test] converted: {info['bitlinears']} BitLinears, "
          f"{info['subnorms']} SubLNs over {info['layers']} layers")
    assert info["subnorms"] == 2 * info["layers"], "expected 2 SubLNs per layer"
    student.train()

    # --- (a) BitLinear actually ternarizes: 3 distinct weight-values * scale ---
    bl = None
    for m in student.modules():
        if isinstance(m, BitLinear):
            bl = m
            break
    with torch.no_grad():
        # perturb so the weight is non-degenerate
        bl.weight.normal_(0.0, 0.5)
        wq = weight_quant_ternary(bl.weight)
        uniq = torch.unique(wq)
    print(f"[self-test] BitLinear ternary distinct values = {uniq.numel()} "
          f"(expect <=3): {sorted(round(v, 6) for v in uniq.tolist())}")
    assert uniq.numel() <= 3, "ternary weight must take at most 3 distinct values"

    # --- (b) STE passes gradients to the FP master weight ---
    x = torch.randn(2, 5, bl.in_features, requires_grad=False)
    y = bl(x)
    y.sum().backward()
    assert bl.weight.grad is not None and torch.isfinite(bl.weight.grad).all(), \
        "STE must pass finite gradients to the master weight"
    print("[self-test] STE gradient to master weight: OK (finite)")
    student.zero_grad()

    # --- (c) full distill loss vs teacher + one optimizer step, all finite ---
    torch.manual_seed(1)
    input_ids = torch.randint(0, teacher.config.vocab_size, (2, 12))

    s_cap = QKVCapture(student, layer_idx=teacher.config.num_hidden_layers - 1)
    t_cap = QKVCapture(teacher, layer_idx=teacher.config.num_hidden_layers - 1)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)

    with torch.no_grad():
        t_out = teacher(input_ids=input_ids, use_cache=False)
    s_out = student(input_ids=input_ids, use_cache=False)

    ce = F.cross_entropy(
        s_out.logits[:, :-1, :].contiguous().view(-1, teacher.config.vocab_size),
        input_ids[:, 1:].contiguous().view(-1),
    )
    ld = logit_kd_loss(s_out.logits[:, :-1, :], t_out.logits[:, :-1, :], tau=tau)
    ad = attn_relation_loss(s_cap.get(), t_cap.get(),
                            n_rel_heads=teacher.config.num_key_value_heads)
    loss = ce + lambda_kd * ld + gamma * ad
    print(f"[self-test] losses: CE={ce.item():.4f}  LD={ld.item():.4f}  "
          f"AD={ad.item():.4e}  total={loss.item():.4f}")
    assert torch.isfinite(loss).all(), "loss must be finite"

    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()
    print(f"[self-test] one optimizer step done; grad_norm={float(gnorm):.4f}")
    assert torch.isfinite(torch.tensor(float(gnorm))), "grad norm must be finite"

    # --- (d) precision anneal endpoints behave (alpha=0 ~ FP forward) ---
    set_quant_alpha(student, 0.0)
    with torch.no_grad():
        _ = student(input_ids=input_ids, use_cache=False).logits
    set_quant_alpha(student, 1.0)
    with torch.no_grad():
        _ = student(input_ids=input_ids, use_cache=False).logits
    print("[self-test] precision anneal alpha in {0,1} forwards: OK")

    # --- (e) bf16 model + convert => consistent-dtype forward (GPU bf16 path guard) ---
    # Regression: convert() must not leave BitLinear master weights in float32 while
    # the model is bf16 (the GPU path had no autocast -> "BFloat16 != float" matmul).
    bf16_student = copy.deepcopy(teacher).to(torch.bfloat16)
    convert_to_bitnet_student(bf16_student)
    bad = [n for n, p in bf16_student.named_parameters() if p.dtype != torch.bfloat16]
    assert not bad, f"convert left non-bf16 params on a bf16 model: {bad[:4]} ..."
    bf16_ids = torch.randint(0, teacher.config.vocab_size, (1, 8))
    with torch.no_grad():
        _ = bf16_student(input_ids=bf16_ids, use_cache=False).logits
    print("[self-test] bf16 convert+forward: OK (no dtype mismatch)")

    s_cap.remove()
    t_cap.remove()
    print("[self-test] PASS")
    return True


# ===========================================================================
# Training loop / budget curve
# ===========================================================================
def train(args) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # --- tokenizer (skipped in tiny-random-corpus mode with synthetic model) ---
    tokenizer = None
    if not args.tiny_model:
        tokenizer = AutoTokenizer.from_pretrained(args.base)

    # --- teacher (frozen FP) + student (converted) ---
    if args.tiny_model:
        teacher = build_tiny_qwen3().to(device)
        import copy
        student = copy.deepcopy(teacher).to(device)
    else:
        print(f"[bitdistill] loading base {args.base!r} (fp) ...")
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        # Load the safetensors file ONCE and move to device, then deepcopy the on-device model for
        # the second instance. Reading the same file twice segfaults torch/safetensors on the
        # ROCm-Windows build (0xC0000005 in the 2nd from_pretrained); and deepcopying on CPU before
        # moving trips a HIP allocator failure. Load→to(device)→deepcopy matches the tiny_model path.
        import copy
        teacher = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
        student = copy.deepcopy(teacher)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False
    student.config.use_cache = False

    info = convert_to_bitnet_student(student)
    if args.ffn_activation != "silu":
        nover = set_student_ffn_activation(student, args.ffn_activation, args.ffn_anneal_steps)
        anneal_note = (f", silu->relu2 anneal over {args.ffn_anneal_steps} steps"
                       if args.ffn_anneal_steps > 0 else "")
        print(f"[bitdistill] student FFN activation -> {args.ffn_activation} "
              f"({nover} MLPs{anneal_note}; teacher keeps silu for KD targets)")
    student.to(device)
    student.train()
    if args.grad_checkpoint:
        # Activation recompute — trades compute for VRAM so the full-vocab logit-KD
        # workload fits a 12 GB card (identical math, just recomputed on backward).
        try:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        print("[bitdistill] gradient checkpointing ENABLED (student)")
    print(f"[bitdistill] student: {info['bitlinears']} BitLinears + "
          f"{info['subnorms']} SubLNs over {info['layers']} layers")

    attn_layer = args.attn_distill_layer
    if attn_layer < 0:
        attn_layer = student.config.num_hidden_layers + attn_layer
    s_cap = QKVCapture(student, attn_layer)
    t_cap = QKVCapture(teacher, attn_layer)
    rel_heads = args.relation_heads or student.config.num_key_value_heads
    print(f"[bitdistill] attn-relation distill on layer {attn_layer}, "
          f"relation_heads={rel_heads}")

    trainable = [p for p in student.parameters() if p.requires_grad]
    if args.optimizer == "adafactor":
        # Memory-light optimizer (no per-param 2nd-moment state like AdamW) — lets batch>2 fit the
        # 3060's 12 GB with a teacher+student in memory. Fixed-lr mode to match the AdamW A/B.
        from transformers.optimization import Adafactor
        opt = Adafactor(trainable, lr=args.lr, relative_step=False, scale_parameter=False,
                        warmup_init=False, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    print(f"[bitdistill] optimizer={args.optimizer}")

    # --- corpus + eval slices ---
    seq_len = args.max_seq_len
    if args.tiny_model or args.tiny_random_corpus:
        vocab = student.config.vocab_size
        cpt_stream = bdata.tiny_random_stream(vocab, seq_len)
        ppl_slice = bdata.tiny_random_list(vocab, seq_len, n=4)
        gsm8k = []
    else:
        try:
            cpt_stream = bdata.cpt_token_stream(
                tokenizer, seq_len, dataset_name=args.cpt_dataset,
                dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet)
        except Exception as e:
            print(f"[bitdistill] cpt_config {args.cpt_config!r} failed ({e}); retry config=None")
            cpt_stream = bdata.cpt_token_stream(
                tokenizer, seq_len, dataset_name=args.cpt_dataset, dataset_config=None,
                local_parquet=args.cpt_local_parquet)
        ppl_slice = bdata.load_ppl_slice(tokenizer, n=20, seq_len=seq_len,
                                         dataset_name=args.cpt_dataset,
                                         dataset_config=args.cpt_config,
                                         local_parquet=args.cpt_local_parquet)
        gsm8k = bdata.load_gsm8k(tokenizer, n=args.eval_n_gsm8k, seq_len=seq_len) \
            if args.eval_gsm8k else []
        csharp_tasks = cse.load_csharp_tasks(n=args.eval_n_csharp, bench_dir=args.csharp_bench_dir) \
            if args.eval_csharp else []

    milestones = sorted(int(float(x)) for x in args.milestones.split(",")) if args.milestones else []
    curve = []
    next_milestone = 0

    # --- loop ---
    max_tokens = int(args.tokens)
    warmup = args.precision_warmup_steps
    lam_warmup = args.lambda_warmup_steps
    bs = args.batch_size
    tokens_seen, step = 0, 0
    t0 = time.perf_counter()
    print(f"[bitdistill] training to {max_tokens:.2e} tokens  bs={bs}  seq_len={seq_len}  "
          f"lr={args.lr}  lambda={args.lambda_kd}  gamma={args.gamma}  tau={args.tau}")

    while tokens_seen < max_tokens and (args.max_steps == 0 or step < args.max_steps):
        # progressive precision anneal (bf16 -> ternary)
        alpha = 1.0 if warmup <= 0 else min(1.0, step / warmup)
        set_quant_alpha(student, alpha)
        # FFN activation anneal (silu -> relu2), if requested
        if args.ffn_anneal_steps > 0:
            set_ffn_anneal(student, min(1.0, step / args.ffn_anneal_steps))
        lam = args.lambda_kd if lam_warmup <= 0 else args.lambda_kd * min(1.0, step / lam_warmup)

        batch = torch.stack([next(cpt_stream) for _ in range(bs)]).to(device)  # [B, T]

        with torch.no_grad():
            t_logits = teacher(input_ids=batch, use_cache=False).logits
        t_qkv = t_cap.get()

        s_logits = student(input_ids=batch, use_cache=False).logits
        s_qkv = s_cap.get()

        ce = F.cross_entropy(
            s_logits[:, :-1, :].contiguous().view(-1, student.config.vocab_size),
            batch[:, 1:].contiguous().view(-1),
        )
        ld = logit_kd_loss(s_logits[:, :-1, :], t_logits[:, :-1, :], tau=args.tau)
        ad = attn_relation_loss(s_qkv, t_qkv, n_rel_heads=rel_heads)
        loss = ce + lam * ld + args.gamma * ad

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        tokens_seen += batch.numel()
        step += 1

        if step <= 5 or step % args.log_every == 0:
            sps = step / max(time.perf_counter() - t0, 1e-9)
            print(f"  step {step:6d}  tok {tokens_seen:.3e}  a={alpha:.2f}  "
                  f"CE {ce.item():.4f}  LD {ld.item():.4f}  AD {ad.item():.3e}  "
                  f"loss {loss.item():.4f}  {sps:.2f} it/s", flush=True)

        # budget-curve milestone: checkpoint + eval
        while next_milestone < len(milestones) and tokens_seen >= milestones[next_milestone]:
            mtok = milestones[next_milestone]
            set_quant_alpha(student, 1.0)  # eval at full ternary
            ppl = compute_ppl(student, ppl_slice, device)
            acc = eval_gsm8k(student, tokenizer, gsm8k, device) if gsm8k else float("nan")
            csharp_p1 = cse.eval_csharp_exec(student, tokenizer, csharp_tasks, device,
                                             bench_dir=args.csharp_bench_dir) if csharp_tasks else float("nan")
            ck = os.path.join(args.out, f"ckpt_{mtok}")
            if args.save_ckpt:
                save_checkpoint(student, ck, step, tokens_seen,
                                extra={"ppl": ppl, "gsm8k_acc": acc, "csharp_pass1": csharp_p1,
                                       "milestone": mtok})
            curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
                          "csharp_pass1": csharp_p1})
            print(f"[milestone {mtok:.2e}] ppl={ppl:.3f}  gsm8k_acc={acc}  csharp_pass1={csharp_p1}  -> {ck}", flush=True)
            next_milestone += 1

    # final save + curve
    set_quant_alpha(student, 1.0)
    ppl = compute_ppl(student, ppl_slice, device)
    acc = eval_gsm8k(student, tokenizer, gsm8k, device) if gsm8k else float("nan")
    csharp_p1 = cse.eval_csharp_exec(student, tokenizer, csharp_tasks, device,
                                     bench_dir=args.csharp_bench_dir) if csharp_tasks else float("nan")
    curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
                  "csharp_pass1": csharp_p1, "final": True})
    if args.save_ckpt:
        save_checkpoint(student, os.path.join(args.out, "final"), step, tokens_seen,
                        extra={"ppl": ppl, "gsm8k_acc": acc, "csharp_pass1": csharp_p1})
    with open(os.path.join(args.out, "curve.json"), "w", encoding="utf-8") as f:
        json.dump({"base": args.base, "curve": curve,
                   "args": {k: v for k, v in vars(args).items()}}, f, indent=2)
    s_cap.remove(); t_cap.remove()
    print(f"[bitdistill] done: {step} steps / {tokens_seen} tokens. "
          f"final ppl={ppl:.3f} gsm8k_acc={acc} csharp_pass1={csharp_p1}. curve -> {args.out}/curve.json")
    return 0


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="bitdistill.py",
        description="BitNet-Distillation: ternarize a dense LLM to 1.58-bit via "
                    "SubLN + BitLinear + logit/attention distillation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--self-test", action="store_true",
                   help="Tiny synthetic Qwen3 build+forward+loss+step validation (no download).")
    p.add_argument("--base", default="Qwen/Qwen3-0.6B", help="Base FP model (HF id or path).")
    p.add_argument("--out", default=".docs/bitdistill/run", help="Output dir for ckpts + curve.")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 ...")
    # loss weights / temperature (paper: classification λ=10 γ=1e5 [caveat: γ default 1e-5]; τ=5)
    p.add_argument("--lambda-kd", type=float, default=10.0, dest="lambda_kd",
                   help="Weight λ on logit-KD loss L_LD (paper classification: 10).")
    p.add_argument("--gamma", type=float, default=1e-5,
                   help="Weight γ on attention-relation loss L_AD. Paper text reads 1e5 for "
                        "classification; this DEFAULTS to 1e-5 per the recipe note (see caveat).")
    p.add_argument("--tau", type=float, default=5.0, help="Logit-KD temperature (paper: 5).")
    p.add_argument("--attn-distill-layer", type=int, default=-1, dest="attn_distill_layer",
                   help="Single layer for MiniLM attn-relation distill (paper |Y|=1). -1=last.")
    p.add_argument("--relation-heads", type=int, default=0, dest="relation_heads",
                   help="MiniLM relation heads (0 = num_key_value_heads).")
    # optimization
    p.add_argument("--tokens", type=float, default=1e5, help="Total training tokens (budget).")
    p.add_argument("--max-steps", type=int, default=0, dest="max_steps",
                   help="Hard cap on optimizer steps (0 = unlimited; smoke uses e.g. 3).")
    p.add_argument("--batch-size", type=int, default=8, dest="batch_size",
                   help="Sequences per step (paper: 32).")
    p.add_argument("--max-seq-len", type=int, default=512, dest="max_seq_len",
                   help="Tokens per sequence (paper: 512).")
    p.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    p.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adamw",
                   help="adamw (default) or adafactor (memory-light; enables batch>2 on 12GB).")
    p.add_argument("--precision-warmup-steps", type=int, default=0, dest="precision_warmup_steps",
                   help="Anneal FP->ternary over this many steps (0 = ternary from step 0).")
    p.add_argument("--lambda-warmup-steps", type=int, default=0, dest="lambda_warmup_steps",
                   help="Ramp λ from 0 to --lambda-kd over this many steps (0 = full from step 0).")
    # data
    p.add_argument("--cpt-dataset", default="HuggingFaceFW/fineweb-edu", dest="cpt_dataset",
                   help="Streaming CPT corpus (general web; FALCON-family alt: tiiuae/falcon-refinedweb).")
    p.add_argument("--cpt-config", default="sample-10BT", dest="cpt_config",
                   help="CPT dataset config (use '' / None if the dataset has no config).")
    p.add_argument("--no-save-ckpt", action="store_false", dest="save_ckpt",
                   help="Skip writing model checkpoints (milestone + final); still records the PPL "
                        "curve. Use on hosts whose CPU torch allocator fails on the state_dict copy "
                        "(the ablation only needs curve.json, not the weights).")
    p.add_argument("--grad-checkpoint", action="store_true", dest="grad_checkpoint",
                   help="Enable gradient checkpointing (activation recompute) so the run fits a "
                        "12 GB GPU. Identical math to the non-checkpointed path; only slower.")
    p.add_argument("--ffn-activation", choices=["silu", "relu2"], default="silu", dest="ffn_activation",
                   help="Student FFN activation: silu (SwiGLU, matches Qwen3 teacher; default) or "
                        "relu2 (squared-ReLU-GLU, canonical BitNet). Teacher stays silu for KD targets.")
    p.add_argument("--ffn-anneal-steps", type=int, default=0, dest="ffn_anneal_steps",
                   help="With --ffn-activation relu2: anneal the student FFN activation silu->relu2 "
                        "over this many steps (0 = hard relu2 from step 0). Eases the source-target "
                        "gap so the relu2 student can realize its quant-friendliness advantage.")
    p.add_argument("--cpt-local-parquet", default=None, dest="cpt_local_parquet",
                   help="Path/glob to a LOCAL parquet shard to stream from (no network). Robust "
                        "path for unattended runs — pre-download one fineweb-edu shard and pass it.")
    p.add_argument("--tiny-random-corpus", action="store_true", dest="tiny_random_corpus",
                   help="Use a synthetic random token stream (real model, no dataset download).")
    p.add_argument("--tiny-model", action="store_true", dest="tiny_model",
                   help="Use the tiny synthetic Qwen3 for both teacher+student (fastest smoke).")
    # budget curve + eval
    p.add_argument("--milestones", default="0.25e9,0.5e9,1e9,2e9,5e9",
                   help="Comma token milestones to checkpoint+eval at.")
    p.add_argument("--eval-gsm8k", action="store_true", dest="eval_gsm8k",
                   help="Run GSM8K accuracy eval at each milestone (the go/no-go signal).")
    p.add_argument("--eval-n-gsm8k", type=int, default=100, dest="eval_n_gsm8k",
                   help="Number of GSM8K examples for the accuracy eval.")
    p.add_argument("--eval-csharp", action="store_true", dest="eval_csharp",
                   help="Run C# compile+xUnit pass@1 eval at each milestone (in-process generate "
                        "-> dotnet build+test via the coding_tasks harness).")
    p.add_argument("--eval-n-csharp", type=int, default=12, dest="eval_n_csharp",
                   help="Number of held-out C# execution tasks for the pass@1 eval (expensive: "
                        "each task generates + runs dotnet build/test).")
    p.add_argument("--csharp-bench-dir", default=cse.DEFAULT_BENCH_DIR, dest="csharp_bench_dir",
                   help="Path to the coding_tasks harness dir (tasks/ + templates/ + task_runner).")
    p.add_argument("--log-every", type=int, default=20, dest="log_every")
    # offline KD (two-phase): cache a frozen teacher's top-K logits, then train student from cache.
    p.add_argument("--make-teacher-cache", default=None, dest="make_teacher_cache",
                   help="Phase 1: run the teacher over the CPT stream and write its top-K logit cache "
                        "to this dir (only the teacher is resident — avoids teacher+student co-residence).")
    p.add_argument("--from-teacher-cache", default=None, dest="from_teacher_cache",
                   help="Phase 2: train the ternary student against a cached teacher (this dir). Only "
                        "the student is resident. Loss = CE + lambda * sparse top-K KD (no AD term).")
    p.add_argument("--top-k", type=int, default=128, dest="top_k",
                   help="Top-K teacher logits to cache per token (tau softens the teacher, so larger K "
                        "keeps more mass; the cache reports mean top-K mass coverage).")
    return p.parse_args(argv)


def generate_teacher_cache(args) -> int:
    """Phase 1 of offline KD: run the frozen FP teacher once over the CPT stream and cache its
    top-K logits (+ per-token log-sum-exp for faithful sparse KD, + the input_ids) per batch.
    Only ONE model is ever resident, so this sidesteps the teacher+student co-residence that
    crashes at >=1.7B on the ROCm-Windows build. The cache is reusable across student runs."""
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"[cache] loading teacher {args.base!r} ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False

    seq_len, bs, K, tau = args.max_seq_len, args.batch_size, args.top_k, args.tau
    stream = bdata.cpt_token_stream(
        tokenizer, seq_len, dataset_name=args.cpt_dataset, dataset_config=args.cpt_config,
        local_parquet=args.cpt_local_parquet)
    max_tokens = int(args.tokens)
    n_batches = (max_tokens + bs * seq_len - 1) // (bs * seq_len)
    os.makedirs(args.make_teacher_cache, exist_ok=True)
    print(f"[cache] {n_batches} batches (bs={bs} seq={seq_len} top_k={K} tau={tau}) -> {args.make_teacher_cache}", flush=True)

    mass_sum, mass_cnt = 0.0, 0
    for i in range(n_batches):
        batch = torch.stack([next(stream) for _ in range(bs)]).to(device)      # [B, T]
        with torch.no_grad():
            logits = teacher(input_ids=batch, use_cache=False).logits.float()  # [B, T, V]
            lse = torch.logsumexp(logits / tau, dim=-1)                        # [B, T]
            topv, topi = logits.topk(K, dim=-1)                               # [B, T, K]
            mass = (topv / tau - lse.unsqueeze(-1)).exp().sum(-1)             # [B, T] top-K softmax mass
        mass_sum += float(mass.mean()); mass_cnt += 1
        torch.save({"input_ids": batch.to(torch.int32).cpu(),
                    "topk_logits": topv.to(torch.float16).cpu(),
                    "topk_idx": topi.to(torch.int32).cpu(),
                    "lse": lse.to(torch.float32).cpu()},
                   os.path.join(args.make_teacher_cache, f"batch_{i:06d}.pt"))
        if i % 50 == 0 or i == n_batches - 1:
            print(f"  cached {i + 1}/{n_batches}  mean top-{K} mass={mass_sum / mass_cnt:.4f}", flush=True)

    meta = {"n_batches": n_batches, "batch_size": bs, "seq_len": seq_len, "top_k": K,
            "tau": tau, "base": args.base, "tokens": max_tokens,
            "mean_topk_mass": mass_sum / max(mass_cnt, 1)}
    with open(os.path.join(args.make_teacher_cache, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] DONE {n_batches} batches; mean top-{K} mass={meta['mean_topk_mass']:.4f} "
          f"(higher = sparse KD closer to dense) -> {args.make_teacher_cache}", flush=True)
    return 0


def train_from_cache(args) -> int:
    """Phase 2 of offline KD: train the ternary student alone against a cached teacher (Phase 1).
    Student is the only resident model. Loss = CE + lambda * sparse_top-K KD (no attention-relation
    term — it needs teacher hidden states and was shown not to help)."""
    import json, glob, time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device(args.device)
    cache_dir = args.from_teacher_cache
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    seq_len, tau = meta["seq_len"], meta["tau"]
    base = args.base or meta["base"]
    print(f"[from-cache] base={base} seq_len={seq_len} tau={tau} top_k={meta['top_k']} "
          f"mean_topk_mass={meta.get('mean_topk_mass', float('nan')):.4f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    student = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    student.config.use_cache = False
    info = convert_to_bitnet_student(student)
    if args.ffn_activation != "silu":
        set_student_ffn_activation(student, args.ffn_activation, args.ffn_anneal_steps)
    student.to(device); student.train()
    if args.grad_checkpoint:
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    print(f"[from-cache] student: {info['bitlinears']} BitLinears + {info['subnorms']} SubLNs", flush=True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    if args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(trainable, lr=args.lr, relative_step=False, scale_parameter=False,
                        warmup_init=False, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    ppl_slice = bdata.load_ppl_slice(tokenizer, n=20, seq_len=seq_len, dataset_name=args.cpt_dataset,
                                     dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet)
    gsm8k = bdata.load_gsm8k(tokenizer, n=args.eval_n_gsm8k, seq_len=seq_len) if args.eval_gsm8k else []
    csharp_tasks = cse.load_csharp_tasks(n=args.eval_n_csharp, bench_dir=args.csharp_bench_dir) \
        if args.eval_csharp else []
    milestones = sorted(int(float(x)) for x in args.milestones.split(",")) if args.milestones else []
    curve, next_ms = [], 0

    cache_files = sorted(glob.glob(os.path.join(cache_dir, "batch_*.pt")))
    if not cache_files:
        raise SystemExit(f"no cache batches in {cache_dir}")
    V = student.config.vocab_size
    max_tokens, warmup = int(args.tokens), args.precision_warmup_steps
    tokens_seen, step, ci = 0, 0, 0
    t0 = time.perf_counter()
    print(f"[from-cache] training to {max_tokens:.2e} tokens over {len(cache_files)} cached batches", flush=True)

    while tokens_seen < max_tokens and (args.max_steps == 0 or step < args.max_steps):
        alpha = 1.0 if warmup <= 0 else min(1.0, step / warmup)
        set_quant_alpha(student, alpha)
        lam = args.lambda_kd
        c = torch.load(cache_files[ci % len(cache_files)]); ci += 1
        batch = c["input_ids"].to(device).long()                          # [B, T]
        topv = c["topk_logits"].to(device)                                # [B, T, K]
        topi = c["topk_idx"].to(device).long()
        lse = c["lse"].to(device)                                         # [B, T]

        s_logits = student(input_ids=batch, use_cache=False).logits
        ce = F.cross_entropy(s_logits[:, :-1, :].contiguous().view(-1, V),
                             batch[:, 1:].contiguous().view(-1))
        Kd = topv.size(-1)
        ld = sparse_logit_kd_loss(s_logits[:, :-1, :].contiguous().view(-1, V),
                                  topv[:, :-1, :].contiguous().view(-1, Kd),
                                  topi[:, :-1, :].contiguous().view(-1, Kd),
                                  lse[:, :-1].contiguous().view(-1), tau)
        loss = ce + lam * ld
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        tokens_seen += batch.numel(); step += 1

        if step <= 5 or step % args.log_every == 0:
            sps = step / max(time.perf_counter() - t0, 1e-9)
            print(f"  step {step:6d}  tok {tokens_seen:.3e}  a={alpha:.2f}  "
                  f"CE {ce.item():.4f}  LD {ld.item():.4f}  loss {loss.item():.4f}  {sps:.2f} it/s", flush=True)

        while next_ms < len(milestones) and tokens_seen >= milestones[next_ms]:
            mtok = milestones[next_ms]
            set_quant_alpha(student, 1.0)
            ppl = compute_ppl(student, ppl_slice, device)
            acc = eval_gsm8k(student, tokenizer, gsm8k, device) if gsm8k else float("nan")
            csharp_p1 = cse.eval_csharp_exec(student, tokenizer, csharp_tasks, device,
                                             bench_dir=args.csharp_bench_dir) if csharp_tasks else float("nan")
            ck = os.path.join(args.out, f"ckpt_{mtok}")
            if args.save_ckpt:
                save_checkpoint(student, ck, step, tokens_seen,
                                extra={"ppl": ppl, "gsm8k_acc": acc, "csharp_pass1": csharp_p1,
                                       "milestone": mtok})
            curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
                          "csharp_pass1": csharp_p1})
            print(f"[milestone {mtok:.2e}] ppl={ppl:.3f}  gsm8k_acc={acc}  csharp_pass1={csharp_p1}  -> {ck}", flush=True)
            next_ms += 1

    set_quant_alpha(student, 1.0)
    ppl = compute_ppl(student, ppl_slice, device)
    acc = eval_gsm8k(student, tokenizer, gsm8k, device) if gsm8k else float("nan")
    csharp_p1 = cse.eval_csharp_exec(student, tokenizer, csharp_tasks, device,
                                     bench_dir=args.csharp_bench_dir) if csharp_tasks else float("nan")
    curve.append({"tokens": tokens_seen, "step": step, "ppl": ppl, "gsm8k_acc": acc,
                  "csharp_pass1": csharp_p1, "final": True})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "curve.json"), "w") as f:
        json.dump({"base": base, "from_cache": cache_dir, "curve": curve, "meta": meta}, f, indent=2)
    print(f"[from-cache] done: {step} steps / {tokens_seen} tokens. final ppl={ppl:.3f} "
          f"gsm8k_acc={acc} csharp_pass1={csharp_p1}. curve -> {args.out}/curve.json", flush=True)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cpt_config in ("", "none", "None"):
        args.cpt_config = None
    if args.self_test:
        return 0 if run_self_test(tau=args.tau, lambda_kd=args.lambda_kd, gamma=args.gamma) else 1
    if args.make_teacher_cache:
        return generate_teacher_cache(args)
    if args.from_teacher_cache:
        return train_from_cache(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
