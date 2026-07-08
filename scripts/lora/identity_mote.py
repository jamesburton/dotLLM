"""identity_mote.py — turn a depth-inserted BitNet identity block into a MoTE.

Campaign: trackM-mote — "identity-expert routed-MoTE in new depth layers".

Motivation (from .docs/handoff.md)
----------------------------------
The prior full-FFN MoTE campaign was a robust NULL: at 2B, converting *existing*
FFN layers into learned-routing MoTE added no task accuracy (experts stayed
homogeneous, 73% identical predictions). Two diagnosed failure modes:

  1. A converted existing layer can only *match or degrade* the base — it has no
     "do nothing" fallback.
  2. Learned routing left experts homogeneous.

This module implements the fix, combined with LLaMA-Pro depth expansion:

  * `scripts/bitnet_depth_expand.py` already inserts *zero-residual ternary
    identity blocks* into the stack (o_proj + down_proj zeroed → the block is an
    exact identity at init, model logits bit-for-bit unchanged). That gives us
    ~2B → ~4.5B of pure new capacity with **no regression at init**.

  * This module converts each inserted block's FFN into an
    :class:`IdentityMoTEBlock`:

        experts[0]   = IDENTITY / SKIP expert  (down_proj == 0, FROZEN)
        experts[1..K]= K capability experts     (trainable)
        router       = nn.Linear(hidden, K+1)   (trainable, top-1)

    - The skip expert always outputs 0 → routing a token to it = "skip this
      inserted layer for this token" = exact base path. This is the permanent
      no-regression fallback that fixes failure mode (1).

    - Routing is SUPERVISED (see identity_mote_train.py): each task's data is
      routed to its OWN expert (math→expert 1, instruction→expert 2, …) via a
      cross-entropy loss on task labels. This *forces* specialization and fixes
      failure mode (2). Top-1 routing matches the per-sequence task label
      naturally.

Identity-at-init invariant (THE critical property)
--------------------------------------------------
Default ``capability_init="zero"``:
    Every expert (skip AND capability) has ``down_proj`` zeroed at init. A BitNet
    FFN with an all-zero ``down_proj`` outputs exactly 0 (ternary-quant of 0 is 0;
    matmul by an exact-zero matrix is 0 — see bitnet_depth_expand.py). Therefore
    the FFN branch is 0 **for every token regardless of which expert the router
    picks**, so the inserted block stays an exact identity for an ARBITRARY router
    state. Capability experts keep the template's *gate_proj/up_proj* (a useful
    warm start) and grow their ``down_proj`` away from 0 during heal-training —
    the per-expert LLaMA-Pro recipe.

``capability_init="template"`` (experimental warm-start):
    Capability experts restore the template's real ``down_proj`` (full warm-started
    FFN). Identity then holds ONLY because the router is initialised to route every
    token to the skip expert (``router_identity_bias`` large). Verified accordingly.

Forward signature (IdentityMoTEBlock)
-------------------------------------
    hidden = block(x)                 # plain tensor — drop-in for HF ``self.mlp(x)``

    # side-channel (stashed as attributes each forward, for the trainer):
    block.last_logits : [B*T, K+1]    — PRE-softmax router logits, IN-GRAPH (for
                                        the supervised routing CE loss upstream)
    block.last_counts : [K+1] int64   — per-expert top-1 dispatch counts (detached)

Returning a plain tensor (rather than the tuple that ``mote_upcycle.MoTEBlock``
returns) means the block is directly usable as a decoder layer's ``.mlp`` — HF
decoder layers do ``residual + self.mlp(x)`` and require a tensor. No separate
"shim" wrapper is needed; the trainer reads the router logits from
``block.last_logits`` after each forward, before ``loss.backward()``.

CPU-only: this module contains no GPU-specific code and its self-test runs on the
tiny synthetic BitNet in seconds. Import is torch-only (no dataset/model download).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_down_proj(mlp: nn.Module) -> None:
    """Zero the FFN output projection so the expert outputs exactly 0.

    BitNetMLP's output projection is ``down_proj`` (AutoBitLinear or nn.Linear).
    An all-zero ``down_proj.weight`` makes the expert a zero function:
    ``down_proj(anything) == 0`` (ternary-quant of 0 == 0; matmul by zero == 0).
    """
    with torch.no_grad():
        mlp.down_proj.weight.zero_()
        if getattr(mlp.down_proj, "bias", None) is not None:
            mlp.down_proj.bias.zero_()


def _is_zero_down_proj(mlp: nn.Module) -> bool:
    """True if the FFN output projection is exactly zero (for assertions)."""
    return bool(torch.count_nonzero(mlp.down_proj.weight).item() == 0)


# ---------------------------------------------------------------------------
# IdentityMoTEBlock
# ---------------------------------------------------------------------------


class IdentityMoTEBlock(nn.Module):
    """FFN replacement for a depth-inserted identity block: skip + K experts + router.

    Args:
        experts: ModuleList of ``K+1`` BitNetMLP modules. ``experts[0]`` is the
            frozen identity/skip expert (``down_proj == 0``); ``experts[1:]`` are
            the K trainable capability experts.
        router: ``nn.Linear(hidden_size, K+1)`` (BF16). Produces per-token routing
            logits. Trained with a *supervised* CE loss on task labels upstream.
        top_k: Experts selected per token. ``1`` (supervised top-1) is the design
            default and the only value for which the zero-``down_proj`` identity
            invariant is exact under an arbitrary router.
    """

    def __init__(
        self,
        experts: nn.ModuleList,
        router: nn.Linear,
        top_k: int = 1,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.router = router
        self.top_k = int(top_k)
        self.n_experts: int = len(experts)
        if not (1 <= self.top_k <= self.n_experts):
            raise ValueError(
                f"top_k={self.top_k} must satisfy 1 <= top_k <= n_experts={self.n_experts}"
            )
        # Side-channel set on every forward: router logits (in-graph) + counts (detached).
        self.last_logits: Optional[torch.Tensor] = None
        self.last_counts: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        n_tokens = B * T
        x_flat = x.view(n_tokens, H)

        # --- Router (raw logits kept for the supervised routing loss) ---
        route_logits = self.router(x_flat.to(self.router.weight.dtype))  # [N, E]
        g = torch.softmax(route_logits, dim=-1)                          # [N, E]

        # --- Top-k selection with per-token normalised gates (sum to 1) ---
        top_g, top_idx = torch.topk(g, self.top_k, dim=-1)              # [N, k]
        g_norm = top_g / top_g.sum(dim=-1, keepdim=True)                # [N, k]

        # --- Dispatch ---
        out = torch.zeros(n_tokens, H, dtype=x.dtype, device=x.device)
        for k_slot in range(self.top_k):
            for e in range(self.n_experts):
                mask = top_idx[:, k_slot] == e
                if not mask.any():
                    continue
                expert_out = self.experts[e](x_flat[mask])             # [m, H]
                gate = g_norm[mask, k_slot : k_slot + 1]               # [m, 1]
                out[mask] = out[mask] + gate.to(out.dtype) * expert_out

        # --- Dispatch counts (logging) ---
        one_hot = F.one_hot(top_idx, self.n_experts)                   # [N, k, E]
        expert_counts = one_hot.sum(dim=(0, 1))                        # [E] int64

        # Stash for the trainer: PRE-softmax logits (in-graph) + counts (detached).
        self.last_logits = route_logits
        self.last_counts = expert_counts.detach()
        return out.view(B, T, H)

    # Convenience accessors used by the freeze / optimizer helpers.
    @property
    def skip_expert(self) -> nn.Module:
        return self.experts[0]

    @property
    def capability_experts(self) -> list[nn.Module]:
        return [self.experts[i] for i in range(1, self.n_experts)]


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


def _make_router(
    hidden_size: int,
    n_experts: int,
    identity_bias: float,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Linear:
    """Build the top-1 router.

    ``weight`` is initialised to ZERO and ``bias`` to zero except ``bias[0] =
    identity_bias``. With a zero weight the routing logits at init depend only on
    the bias, so the skip expert (index 0) wins for every token when
    ``identity_bias > 0`` — a deterministic, input-independent default-to-skip.
    This keeps the model's *behaviour* biased toward the exact base path before
    the supervised routing loss has taught any specialization, and is REQUIRED for
    the identity invariant in ``capability_init="template"`` mode.

    The supervised routing CE loss provides gradient to ``weight`` from the very
    first step (∂CE/∂logits · x), so a zero-init weight is not a dead start.
    """
    router = nn.Linear(hidden_size, n_experts, bias=True, dtype=dtype)
    with torch.no_grad():
        router.weight.zero_()
        router.bias.zero_()
        if n_experts > 0:
            router.bias[0] = float(identity_bias)
    return router.to(device)


# ---------------------------------------------------------------------------
# build_identity_mote
# ---------------------------------------------------------------------------


def build_identity_mote(
    model: nn.Module,
    inserted_indices: list[int],
    n_capability_experts: int,
    capability_init: str = "zero",
    router_identity_bias: float = 0.0,
    top_k: int = 1,
    freeze_skip: bool = True,
) -> nn.Module:
    """Convert each inserted identity block's FFN into an :class:`IdentityMoTEBlock`.

    Modifies ``model`` in-place and returns it. Call this AFTER
    ``bitnet_depth_expand.expand_model`` (which supplies ``inserted_indices`` in
    its returned ``info`` dict).

    For each inserted layer index ``i``:

      * ``experts[0]`` = the block's existing ``mlp`` (already ``down_proj==0`` from
        depth expansion) → the frozen identity/skip expert.
      * ``experts[1..K]`` = ``K = n_capability_experts`` deep copies of that mlp.
        - ``capability_init="zero"`` (default): each copy keeps the template's real
          ``gate_proj/up_proj`` but has ``down_proj`` zeroed → outputs 0 at init →
          the identity invariant holds for an ARBITRARY router. Grows smoothly.
        - ``capability_init="template"``: each copy keeps the FULL template FFN
          (real ``down_proj``) → warm-started, but identity then relies on the
          router routing to the skip expert (needs ``router_identity_bias`` large).
      * ``router`` = top-1 router over ``K+1`` experts.

    Args:
        model: A depth-expanded BitNetForCausalLM (``model.model.layers`` present).
        inserted_indices: Final layer indices of the inserted identity blocks.
        n_capability_experts: K, the number of *capability* experts per block
            (total experts per block = K+1 including the skip expert).
        capability_init: ``"zero"`` (default, bulletproof identity) or
            ``"template"`` (warm-started, router-dependent identity).
        router_identity_bias: Initial bias on the skip logit (index 0). Ignored for
            the invariant when ``capability_init="zero"``; should be large (e.g.
            30.0) for ``"template"``.
        top_k: Experts per token (1 = supervised top-1, the design default).
        freeze_skip: Freeze the skip expert's parameters (keeps a permanent exact
            identity fallback). Default True.

    Returns:
        The modified ``model``. Also stashes ``model._identity_mote_layers`` — the
        list of converted layer indices — for downstream freeze/optimizer helpers.
    """
    if capability_init not in ("zero", "template"):
        raise ValueError(
            f"capability_init must be 'zero' or 'template'; got {capability_init!r}"
        )
    if n_capability_experts < 1:
        raise ValueError("n_capability_experts must be >= 1")

    hidden_size: int = model.config.hidden_size
    # Match the model's compute dtype for the router (BitNet master weights are bf16).
    ref_dtype = next(model.parameters()).dtype
    router_dtype = torch.bfloat16 if ref_dtype in (torch.bfloat16, torch.float16) else ref_dtype

    inserted_set = set(inserted_indices)
    n_layers = len(model.model.layers)

    def _nearest_original_mlp(idx: int) -> nn.Module:
        """Nearest non-inserted layer's mlp (a real FFN with a non-zero down_proj)."""
        for d in range(1, n_layers):
            for j in (idx - d, idx + d):
                if 0 <= j < n_layers and j not in inserted_set:
                    return model.model.layers[j].mlp
        raise RuntimeError("no original (non-inserted) layer found for template warm-start")

    converted: list[int] = []
    for layer_idx in inserted_indices:
        layer = model.model.layers[layer_idx]
        base_mlp: nn.Module = layer.mlp
        device = base_mlp.down_proj.weight.device

        # Skip expert (index 0): the block's own identity FFN (down_proj already 0).
        _zero_down_proj(base_mlp)  # defensive: ensure exact zero
        if freeze_skip:
            for p in base_mlp.parameters():
                p.requires_grad_(False)

        experts: list[nn.Module] = [base_mlp]

        # Warm-start source for "template" mode: a real neighbouring FFN.
        template_mlp = _nearest_original_mlp(layer_idx) if capability_init == "template" else base_mlp

        # Capability experts (indices 1..K).
        for _ in range(n_capability_experts):
            expert = copy.deepcopy(template_mlp)
            if capability_init == "zero":
                # gate/up warm-started from the inserted block (real template gate/up);
                # down_proj zeroed → outputs 0 at init → identity holds for ANY router.
                _zero_down_proj(expert)
            # capability_init == "template": keep the neighbour's FULL real FFN
            # (real down_proj). Identity then holds only because the router is
            # biased to the skip expert (router_identity_bias large).
            for p in expert.parameters():
                p.requires_grad_(True)
            experts.append(expert)

        router = _make_router(
            hidden_size=hidden_size,
            n_experts=n_capability_experts + 1,
            identity_bias=router_identity_bias,
            dtype=router_dtype,
            device=device,
        )

        layer.mlp = IdentityMoTEBlock(
            nn.ModuleList(experts), router, top_k=top_k
        )
        converted.append(layer_idx)

    model._identity_mote_layers = converted  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Identity verification helper (used by tests and by the trainer smoke gate)
# ---------------------------------------------------------------------------


def assert_identity_at_init(
    base_model: nn.Module,
    mote_model: nn.Module,
    input_ids: torch.Tensor,
    tol: float = 1e-3,
    randomize_router: bool = False,
) -> dict:
    """Assert the identity-MoTE model's logits equal the base model's at init.

    Args:
        base_model: the pre-expansion base (dense) model.
        mote_model: the depth-expanded + identity-MoTE-converted model.
        input_ids: [B, T] token ids.
        tol: max-abs logit tolerance.
        randomize_router: if True, first scramble every router's weight+bias with
            large random values. In ``capability_init="zero"`` mode the identity
            MUST still hold (every expert outputs 0 regardless of routing) — this
            is the bulletproof check.

    Returns a dict with ``max_abs``, ``mean_abs``, ``argmax_match``, ``ok``.
    """
    if randomize_router:
        with torch.no_grad():
            for layer in mote_model.model.layers:
                mlp = layer.mlp
                if isinstance(mlp, IdentityMoTEBlock):
                    mlp.router.weight.normal_(mean=0.0, std=5.0)
                    mlp.router.bias.normal_(mean=0.0, std=5.0)

    base_model.eval()
    mote_model.eval()
    # use_cache=False sidesteps the known KV-cache "N vs N-1" layer-count mismatch
    # on generation with a changed layer count (see handoff).
    with torch.no_grad():
        base_logits = base_model(input_ids, use_cache=False).logits.float()
        mote_logits = mote_model(input_ids, use_cache=False).logits.float()

    max_abs = (mote_logits - base_logits).abs().max().item()
    mean_abs = (mote_logits - base_logits).abs().mean().item()
    argmax_match = bool(
        (mote_logits.argmax(-1) == base_logits.argmax(-1)).all().item()
    )
    ok = (max_abs <= tol) and argmax_match
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "argmax_match": argmax_match,
        "ok": ok,
    }
