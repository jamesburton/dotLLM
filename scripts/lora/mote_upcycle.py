"""MoTE (Mixture-of-Ternary-Experts) upcycle module.

Replaces dense BitNetMLP layers in a BitNetForCausalLM with MoTE blocks:
  - N routed experts (identical deep copies of the original BitNetMLP)
  - A BF16 router nn.Linear(hidden_size, n_experts), init N(0, 0.02)
  - An optional shared expert (frozen bf16 plain copy, frozen ternary copy, or none)

Each cloned routed expert retains its AutoBitLinear layers, so ternary weight-quant
(absmean, online) and int8 activation-quant are applied automatically in the
expert's own forward pass.  No additional quant instrumentation is required.

Router combine weights are NOT renormalized after top-k selection.  The output is
the raw gate-weighted sum:

    out = Σ_{i∈top-k} gate_i · expert_i(x)

where gate_i are the raw softmax probabilities.  This matches the Komatsuzaki et al.
finding that normalizing the combine weights hurts upcycled LMs, and it means the
init-identity property no longer holds trivially — the test asserts the correct
structural ratio form instead.

Usage
-----
    mote_model = build_mote(
        base_model,
        layers=list(range(15, 30)),
        n_experts=4,
        top_k=1,
        shared="fp",
    )

Forward signature (MoTEBlock)
-----------------------------
    hidden, aux_loss, expert_counts = mote_block(x)

where
    hidden        : torch.Tensor  [batch, seq, hidden_size]
    aux_loss      : torch.Tensor  scalar — Switch load-balance loss
    expert_counts : torch.Tensor  [n_experts] int64 — dispatch count per expert
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helper: plain bf16 shared expert
# ---------------------------------------------------------------------------


def _make_fp_shared_expert(dense_mlp: nn.Module) -> nn.Module:
    """Return a frozen bf16 (plain) copy of the dense FFN.

    Each ``AutoBitLinear`` sublayer is replaced with a plain ``nn.Linear``
    using the same bf16 master weights.  ``online_quant`` ternary weight-quant
    (``WeightQuant``) and int8 activation-quant (``ActQuant``) are both omitted,
    giving a true bf16 forward.  Non-quantized submodules (e.g. ``BitNetRMSNorm``,
    ``act_fn``) are preserved as-is.  All parameters are frozen.

    This works for ``online_quant=True`` AutoBitLinear instances (the standard
    mode for the BitNet b1.58 model) where ``self.weight`` holds the bf16 master
    weights directly.

    Args:
        dense_mlp: The original ``BitNetMLP`` module to copy.

    Returns:
        A frozen ``nn.Module`` whose forward is a plain bf16 matmul.
    """
    fp_expert = copy.deepcopy(dense_mlp)
    # Replace every AutoBitLinear with a plain nn.Linear (same bf16 weights, no quant).
    # We match by class name to avoid importing AutoBitLinear at module level.
    for attr_path, module in list(fp_expert.named_modules()):
        if type(module).__name__ != "AutoBitLinear":
            continue
        lin = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
        )
        lin.weight = nn.Parameter(module.weight.data.clone(), requires_grad=False)
        if module.bias is not None:
            lin.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        # Navigate to the parent module and replace the child attribute.
        *parent_parts, attr_name = attr_path.split(".")
        parent: nn.Module = fp_expert
        for part in parent_parts:
            parent = getattr(parent, part)
        setattr(parent, attr_name, lin)
    # Freeze all remaining parameters (e.g. BitNetRMSNorm scale weights).
    for p in fp_expert.parameters():
        p.requires_grad_(False)
    return fp_expert


# ---------------------------------------------------------------------------
# MoTEBlock
# ---------------------------------------------------------------------------


class MoTEBlock(nn.Module):
    """One MoTE layer: BF16 router + N ternary routed experts + optional shared expert.

    Args:
        experts: ModuleList of N cloned BitNetMLP modules.
        router: nn.Linear(hidden_size, n_experts) in BF16, bias=False.
        top_k: Number of experts selected per token.
        shared: Optional shared expert applied to every token:
            - ``"fp"``      — frozen bf16 (plain) forward; plain ``nn.Linear``
                              sublayers, no ternary weight-quant, no act-quant.
            - ``"ternary"`` — trainable ternary (AutoBitLinear) forward; identical
                              deep copy of the dense FFN with bf16 shadow weights,
                              updated during heal-training (QAT).
            - ``None``      — no shared expert.
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        router: nn.Linear,
        top_k: int,
        shared: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.router = router
        self.top_k = top_k
        self.shared = shared
        self.n_experts: int = len(experts)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MoTE forward pass.

        Args:
            x: Input activations, shape [batch, seq, hidden_size].

        Returns:
            A 3-tuple:
                hidden      : [batch, seq, hidden_size] — MoTE output
                aux_loss    : scalar — Switch load-balance loss
                              E * mean(frac_tokens_i * mean_prob_i)
                expert_counts: [n_experts] int64 — total dispatch count per expert
        """
        B, T, H = x.shape
        n_tokens = B * T
        x_flat = x.view(n_tokens, H)  # [n_tokens, H]

        # --- Router ---
        # Cast input to BF16 for the BF16 router, then compute softmax probabilities.
        g = torch.softmax(
            self.router(x_flat.to(self.router.weight.dtype)), dim=-1
        )  # [n_tokens, n_experts]

        # --- Top-k selection (raw gates, NOT renormalized) ---
        # Output = Σ_{selected} gate_i · expert_i(x) with raw softmax gates.
        # Renormalizing the selected gates was removed (Komatsuzaki et al. show it
        # hurts upcycled LMs).
        top_g, top_idx = torch.topk(g, self.top_k, dim=-1)  # [n_tokens, top_k]

        # --- Dispatch: accumulate gate-weighted expert outputs ---
        out = torch.zeros(n_tokens, H, dtype=x.dtype, device=x.device)
        for k_slot in range(self.top_k):
            for e in range(self.n_experts):
                mask = top_idx[:, k_slot] == e  # [n_tokens] bool
                if not mask.any():
                    continue
                expert_out = self.experts[e](x_flat[mask])           # [m, H]
                gate = top_g[mask, k_slot : k_slot + 1]              # [m, 1]
                # Accumulate: out[mask] += gate * expert_out
                # Use index assignment to stay autograd-friendly.
                out[mask] = out[mask] + gate * expert_out

        # --- Optional shared expert (all tokens, always applied) ---
        if self.shared is not None:
            out = out + self.shared(x_flat)

        # --- Switch load-balance aux loss ---
        # frac[e] = fraction of token-expert slots assigned to expert e, normalized
        #           by top_k so that frac.sum() ≈ 1 (Switch-consistent for top_k > 1).
        # mean_prob[e] = mean router probability assigned to expert e.
        # aux = E * mean(frac * mean_prob)   (γ coefficient applied by the caller)
        one_hot = F.one_hot(top_idx, self.n_experts)           # [n_tokens, top_k, n_experts]
        frac = one_hot.sum(dim=1).float().mean(dim=0) / self.top_k  # [n_experts], sum≈1
        mean_prob = g.mean(dim=0)                              # [n_experts]
        aux: torch.Tensor = self.n_experts * (frac * mean_prob).mean()

        # --- Expert dispatch counts (for logging / eval) ---
        expert_counts: torch.Tensor = one_hot.sum(dim=(0, 1))  # [n_experts] int64

        return out.view(B, T, H), aux, expert_counts


# ---------------------------------------------------------------------------
# build_mote
# ---------------------------------------------------------------------------


def build_mote(
    base_model: nn.Module,
    layers: list[int],
    n_experts: int,
    top_k: int,
    shared: str,
) -> nn.Module:
    """Replace selected FFN layers in a BitNetForCausalLM with MoTE blocks.

    Modifies ``base_model`` in-place and returns it.

    Each target layer's ``mlp`` attribute is replaced with a :class:`MoTEBlock`
    containing:

    * N routed experts — deep copies of the original ``BitNetMLP``.  Each copy
      retains its ``AutoBitLinear`` layers, so ternary weight-quant (absmean,
      online) and int8 activation-quant are applied in the expert's own forward
      without any additional instrumentation.
    * A BF16 ``nn.Linear(hidden_size, n_experts, bias=False)`` router
      initialised with weights ~ N(0, 0.02).
    * An optional shared expert:

      - ``"fp"``      — frozen bf16 (plain) forward: ``AutoBitLinear`` sublayers
                        are replaced with plain ``nn.Linear`` using the bf16 master
                        weights.  No ternary weight-quant, no act-quant.  Frozen.
      - ``"ternary"`` — trainable ternary (AutoBitLinear) forward: identical deep copy
                        of the dense ``BitNetMLP`` with bf16 shadow weights.  Updated
                        during heal-training (QAT).
      - ``"none"``    — no shared expert.

    Args:
        base_model: A ``BitNetForCausalLM`` (or compatible HF causal LM) with
            a ``model.layers`` attribute.
        layers: Indices of transformer layers whose ``mlp`` will be replaced.
        n_experts: Number of routed experts per MoTE block.
        top_k: Number of experts to select per token (1 ≤ top_k ≤ n_experts).
        shared: Shared-expert mode — one of ``"fp"``, ``"ternary"``, ``"none"``.

    Returns:
        The modified ``base_model``.

    Raises:
        ValueError: If ``shared`` is not one of the allowed values or if
            ``top_k`` is outside [1, n_experts].
    """
    if shared not in ("fp", "ternary", "none"):
        raise ValueError(
            f"shared must be 'fp', 'ternary', or 'none'; got {shared!r}"
        )
    if not (1 <= top_k <= n_experts):
        raise ValueError(
            f"top_k={top_k} must satisfy 1 ≤ top_k ≤ n_experts={n_experts}"
        )

    hidden_size: int = base_model.config.hidden_size

    for layer_idx in layers:
        dense_mlp: nn.Module = base_model.model.layers[layer_idx].mlp

        # N identical routed experts — deep copies that preserve AutoBitLinear.
        experts = nn.ModuleList(
            [copy.deepcopy(dense_mlp) for _ in range(n_experts)]
        )

        # BF16 router, init N(0, 0.02) as per spec.
        router = nn.Linear(hidden_size, n_experts, bias=False, dtype=torch.bfloat16)
        nn.init.normal_(router.weight, mean=0.0, std=0.02)

        # Optional shared expert.
        shared_module: Optional[nn.Module] = None
        if shared == "fp":
            # True bf16 forward: replace AutoBitLinear with plain nn.Linear.
            shared_module = _make_fp_shared_expert(dense_mlp)
        elif shared == "ternary":
            # Ternary (AutoBitLinear) forward — trainable during heal-training.
            shared_module = copy.deepcopy(dense_mlp)
        # else "none": shared_module stays None

        mote_block = MoTEBlock(experts, router, top_k, shared_module)
        base_model.model.layers[layer_idx].mlp = mote_block

    return base_model
