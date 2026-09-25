"""(P,R,C) looped-depth model for BitNet ternary re-feed experiments.

Architecture:
  embed → layers[0:P) (prelude) → LOOP×N {
      s = fusion(cat(prelude_out, state))      # fuse prelude + recurrent state
      slab_out = layers[P:Q)(s)               # re-run the middle slab
      state = s + g * (slab_out - s)          # learned residual gate
  } → layers[Q:L) (coda) → norm → lm_head

The fusion adapter (nn.Linear(2H→H)) implements the approach from arXiv:2511.07384.
The learned gate g (init≈0.5) is the damped-residual form validated in Probe 0b
(commit f6c08ee on spike/recursion-stability): damped γ=0.5 held PPL flat-or-
improving through N=8, while fixed rescaling/RMSNorm collapsed quality.

Pass-through init (used by Task 2 N=1==stock sanity check):
  fusion: W[:, :H] = I, W[:, H:] = 0, bias = 0  → s = prelude_out
  gate.g = 1.0                                   → state = slab_out
  ⇒ N=1 forward == stock model exactly.

Manual layer iteration replicates Probe 0's proven forward pattern; no KV cache
(generation/KV is a later task).  Windows: dynamo errors suppressed (no cl.exe).
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import BitNetForCausalLM

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

# Windows: suppress dynamo compilation errors (no cl.exe available in most setups)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except Exception:
    pass

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
DEFAULT_P = 7   # first slab layer (inclusive)  — Probe 0's L//4 for L=30
DEFAULT_Q = 22  # last slab layer (exclusive)   — Probe 0's 3*L//4 for L=30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_causal_mask(seq: int, dtype: torch.dtype,
                       device: torch.device) -> torch.Tensor:
    """Upper-triangular additive causal mask, shape (1, 1, seq, seq)."""
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq, seq), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def _layer_fwd(layer: nn.Module, hidden: torch.Tensor,
               **kwargs: object) -> torch.Tensor:
    """Call a HF decoder layer; unwrap a tuple return if needed."""
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ResidualGate
# ---------------------------------------------------------------------------

class ResidualGate(nn.Module):
    """Learned per-loop residual gate.

    Applies the damped residual update validated in Probe 0b::

        out = h_in + g * (slab_out - h_in)

    where ``g`` is a scalar trainable :class:`~torch.nn.Parameter` initialised
    to ``init`` (default 0.5).  With ``g=1`` the gate becomes a full pass-through
    (``out == slab_out``); with ``g=0`` the slab is skipped entirely.

    The gate is intentionally small (one scalar) so it can later be extended to
    per-channel or per-loop-step variants without interface changes.
    """

    def __init__(self, init: float = 0.5) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.tensor(float(init)))

    def forward(self, h_in: torch.Tensor, slab_out: torch.Tensor) -> torch.Tensor:
        """Return ``h_in + g * (slab_out - h_in)``."""
        return h_in + self.g * (slab_out - h_in)


# ---------------------------------------------------------------------------
# RecurModel
# ---------------------------------------------------------------------------

class RecurModel(nn.Module):
    """BitNet with a ``(P, Q)`` middle slab that can be looped N times.

    Parameters
    ----------
    base:
        Loaded ``BitNetForCausalLM`` (bf16 master).  Weights are *shared*, not
        copied; freeze them explicitly before training if you only want to train
        the adapter + gate.
    P:
        First slab layer index (inclusive).  Layers ``[0:P)`` form the prelude.
    Q:
        Last slab layer index (exclusive).  Layers ``[Q:L)`` form the coda.
    gate_init:
        Initial value for the learned residual gate ``g`` (default 0.5).
    """

    def __init__(
        self,
        base: "BitNetForCausalLM",
        P: int,
        Q: int,
        gate_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.base = base
        self.P = P
        self.Q = Q

        hidden_size: int = base.config.hidden_size

        # Fusion adapter — concatenate(prelude_out, state) → hidden.
        # Random Gaussian init (std=0.02) per arXiv:2511.07384; use
        # make_passthrough() to force the identity path for the N=1 sanity check.
        self.fusion = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        nn.init.normal_(self.fusion.weight, std=0.02)
        nn.init.zeros_(self.fusion.bias)

        # Learned per-loop residual gate
        self.gate = ResidualGate(init=gate_init)

        # Ensure fusion and gate are in the same dtype and device as the base model.
        # The base model may be bf16 while default nn.Linear/Parameter are float32.
        dtype = next(base.parameters()).dtype
        device = next(base.parameters()).device
        self.fusion = self.fusion.to(dtype=dtype, device=device)
        self.gate = self.gate.to(dtype=dtype, device=device)

        # Internal flag: bypass fusion AND gate with exact operations (no matmul/gate
        # arithmetic).  Used by make_passthrough() so N=1 gives stock model logits
        # exactly.  In bypass mode: fused=prelude_out (slice), state=slab_out
        # (direct assignment — avoids float32 `h_in + g*(slab_out-h_in)` rounding).
        # Set to False in normal forward; True only during testing.
        self._bypass_fusion: bool = False

        # Gradient checkpointing flag — set True externally (e.g., by training script)
        # to wrap each slab forward in torch.utils.checkpoint.checkpoint, trading
        # recomputation for lower peak activation memory (key for high N on 12 GB GPU).
        self.use_grad_checkpoint: bool = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        recurrence: int = 1,
    ) -> torch.Tensor:
        """Run the ``(P, R, C)`` looped forward pass.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``(batch, seq)``.
        recurrence:
            Number of times to run the slab.  With the pass-through init
            (see :func:`make_passthrough`) and ``recurrence=1``, output logits
            match the stock model exactly.

        Returns
        -------
        logits:
            Float tensor of shape ``(batch, seq, vocab_size)``.
        """
        core = self.base.model  # LlamaModel (BitNet backbone)
        device = input_ids.device
        seq = input_ids.shape[1]

        # ---- embed ----
        hidden = core.embed_tokens(input_ids)
        dtype = hidden.dtype

        # Rotary embeddings and causal mask — computed once and reused across
        # all loop iterations (same position ids per pass, as in Probe 0).
        position_ids = torch.arange(seq, device=device).unsqueeze(0)
        position_embeddings = core.rotary_emb(hidden, position_ids)
        attn_mask = _build_causal_mask(seq, dtype, device)

        layer_kwargs: dict = dict(
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=torch.arange(seq, device=device),
            position_embeddings=position_embeddings,
        )

        layers = core.layers
        P, Q = self.P, self.Q

        # ---- prelude: layers[0:P) ----
        for layer in layers[:P]:
            hidden = _layer_fwd(layer, hidden, **layer_kwargs)

        prelude_out = hidden  # frozen context injected on every loop iteration
        state = hidden        # initial recurrent state = prelude output

        # Slab function defined once outside the recurrence loop — reused for
        # every iteration and optionally wrapped in grad-checkpoint.
        _slab_layers = list(layers[P:Q])
        _layer_kw = layer_kwargs

        def _make_slab_fn(slabs: list, kwargs: dict):
            def _fn(h: torch.Tensor) -> torch.Tensor:
                for _layer in slabs:
                    h = _layer_fwd(_layer, h, **kwargs)
                return h
            return _fn

        _slab_fn = _make_slab_fn(_slab_layers, _layer_kw)

        # ---- looped slab: layers[P:Q) × recurrence ----
        for _ in range(recurrence):
            if self._bypass_fusion:
                # Bypass mode (sanity-check only): skip fusion AND gate arithmetic
                # entirely so the N=1==stock comparison is exact in float32.
                # state = slab_out (direct assign — avoids h_in+g*(out-h_in) error).
                state = _slab_fn(prelude_out)
            else:
                # Normal forward: fusion adapter + learned gate.
                fused = self.fusion(torch.cat([prelude_out, state], dim=-1))
                # Slab forward — optionally grad-checkpointed to cap peak VRAM
                # when running high recurrence counts (N×full-slab activations).
                if self.use_grad_checkpoint:
                    slab_out = torch.utils.checkpoint.checkpoint(
                        _slab_fn, fused, use_reentrant=False
                    )
                else:
                    slab_out = _slab_fn(fused)
                # Learned residual gate: state = fused + g * (slab_out - fused)
                state = self.gate(fused, slab_out)

        hidden = state

        # ---- coda: layers[Q:L) ----
        for layer in layers[Q:]:
            hidden = _layer_fwd(layer, hidden, **layer_kwargs)

        hidden = core.norm(hidden)
        logits = self.base.lm_head(hidden)
        return logits


# ---------------------------------------------------------------------------
# Pass-through configuration (for N=1 == stock sanity check)
# ---------------------------------------------------------------------------

def make_passthrough(model: RecurModel) -> None:
    """Configure *in-place* so that ``forward(ids, recurrence=1) == stock logits``.

    Sets ``_bypass_fusion = True`` which engages the exact bypass path in
    :meth:`RecurModel.forward`:

    .. code-block:: text

        fused = prelude_out                      (exact slice, zero matmul error)
        slab_out = layers[P:Q)(prelude_out)      (standard slab pass)
        state = slab_out                         (direct assign, zero gate error)
        logits = lm_head(norm(coda(slab_out)))   (identical to stock model)

    This is the ``"RAW looped slab"`` referred to in the Task 2 spec: no adapter,
    no gate arithmetic — just the plain slab re-run, equivalent to probe0's
    ``manual_forward`` with ``n_loops=1``.

    Note: ``_bypass_fusion`` is only for testing.  Normal forward always uses the
    fusion linear and gate.  Restore with ``model._bypass_fusion = False``.
    """
    model._bypass_fusion = True


# ---------------------------------------------------------------------------
# build_recur — public factory
# ---------------------------------------------------------------------------

def build_recur(
    base: "BitNetForCausalLM",
    P: int = DEFAULT_P,
    Q: int = DEFAULT_Q,
    gate_init: float = 0.5,
) -> RecurModel:
    """Wrap a loaded ``BitNetForCausalLM`` in a :class:`RecurModel`.

    The base model weights are *shared* (no copy).  Only the fusion adapter
    and gate parameter are new trainable tensors.

    Parameters
    ----------
    base:
        Already-loaded ``BitNetForCausalLM`` (HF, bf16 master).
    P:
        Start of the looped slab (default 7 for L=30 BitNet-2B-4T).
    Q:
        End of the looped slab (default 22 for L=30 BitNet-2B-4T).
    gate_init:
        Initial value for the residual gate ``g`` (default 0.5).

    Returns
    -------
    :class:`RecurModel` ready for ``forward(input_ids, recurrence=N)``.

    Example
    -------
    >>> model = BitNetForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    >>> recur = build_recur(model)
    >>> logits = recur(input_ids, recurrence=2)
    """
    return RecurModel(base, P=P, Q=Q, gate_init=gate_init)
