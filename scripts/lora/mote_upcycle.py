"""MoTE (Mixture-of-Ternary-Experts) upcycle module.

Replaces dense BitNetMLP layers in a BitNetForCausalLM with MoTE blocks:
  - N routed experts (identical deep copies of the original BitNetMLP)
  - A BF16 router nn.Linear(hidden_size, n_experts), init N(0, 0.02)
  - An optional shared expert (frozen fp copy, trainable ternary copy, or none)

Each cloned expert retains its AutoBitLinear layers, so ternary weight-quant
(absmean, online) and int8 activation-quant are applied automatically in the
expert's own forward pass.  No additional quant instrumentation is required.

Router combine weights are renormalized within the top-k selection so they
sum to 1 per token.  This preserves the init-identity property: with N
identical clones the MoTE output equals the original dense FFN output.

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
# MoTEBlock
# ---------------------------------------------------------------------------


class MoTEBlock(nn.Module):
    """One MoTE layer: BF16 router + N ternary routed experts + optional shared expert.

    Args:
        experts: ModuleList of N cloned BitNetMLP modules.
        router: nn.Linear(hidden_size, n_experts) in BF16, bias=False.
        top_k: Number of experts selected per token.
        shared: Optional shared expert applied to every token (fp-frozen or
            ternary-trainable copy of the dense FFN), or None to omit.
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

        # --- Top-k selection; renormalize selected gates to sum to 1 per token ---
        # Renormalization is essential for the init-identity property:
        # with top_k=1, the single selected gate becomes 1.0, so
        # out = 1.0 * expert(x) == dense(x) when all experts are identical clones.
        top_g, top_idx = torch.topk(g, self.top_k, dim=-1)  # [n_tokens, top_k]
        top_g = top_g / top_g.sum(dim=-1, keepdim=True)     # renorm → sums to 1

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
        # frac[e] = mean fraction of tokens that selected expert e across all k-slots
        # mean_prob[e] = mean router probability assigned to expert e
        # aux = E * mean(frac * mean_prob)   (γ coefficient applied by the caller)
        one_hot = F.one_hot(top_idx, self.n_experts)           # [n_tokens, top_k, n_experts]
        frac = one_hot.sum(dim=1).float().mean(dim=0)          # [n_experts]
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

      - ``"fp"``      — frozen deep copy of the dense ``BitNetMLP`` (bf16 weights,
                        quantized in forward, but no gradient updates).
      - ``"ternary"`` — trainable deep copy (bf16 shadow weights updated during
                        QAT heal-training; ternary quant applied in forward).
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
            shared_module = copy.deepcopy(dense_mlp)
            for p in shared_module.parameters():
                p.requires_grad_(False)
        elif shared == "ternary":
            shared_module = copy.deepcopy(dense_mlp)
        # else "none": shared_module stays None

        mote_block = MoTEBlock(experts, router, top_k, shared_module)
        base_model.model.layers[layer_idx].mlp = mote_block

    return base_model
