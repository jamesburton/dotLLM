"""Inference-time live exit gate for the (P,R,C) looped-depth model — Task R4.

The novel contribution of Track R: instead of running the recurrent slab a
*fixed* number of times per token, decide *per decode step* how many times to
loop, based on a live difficulty signal computed from the token's own logits.
Easy tokens exit after one pass; hard tokens get re-looped (up to ``n_max``).

Difficulty signal (per :func:`generate_adaptive` ``signal`` arg)
----------------------------------------------------------------
A token is "hard" when, on the logits produced after the current loop:

  * ``signal="entropy"``  : ``entropy(softmax(logits)) > ent_thresh``
  * ``signal="margin"``   : ``(top1_logit - top2_logit) < margin_thresh``
  * ``signal="both"``     : entropy OR margin flags hard (the approved default,
                            Global-Constraints line 15: loop while *either*
                            signal says hard).

We always run at least one loop. After each loop the signal is recomputed on
the new logits; we stop as soon as the token is confident (not hard) or we hit
``n_max``. The emitted token is the greedy argmax of the *final* loop's logits
(temperature 0). The number of loops actually spent on each generated token is
recorded in ``per_token_loops``.

KV-cache scheme (avoids the O(seq^2) full-sequence recompute that broke 0b)
---------------------------------------------------------------------------
We keep a single :class:`~transformers.DynamicCache` for all L layers and drive
the layers manually (mirroring :mod:`recur_model`'s forward). The cache is
partitioned by layer index:

  * prelude ``layers[0:P)``  — run **once** per token; its KV is committed and
    never cropped.
  * slab    ``layers[P:Q)``  — run once **per loop iteration**. Before each
    iteration we crop the slab layers back to the *committed* past length so the
    current token's KV from the previous iteration is discarded and recomputed
    from the evolving recurrent state. Only the **final** loop's KV survives
    (it is never cropped after the loop ends) and becomes the committed KV that
    future tokens attend to.
  * coda    ``layers[Q:L)``  — same crop-each-iteration treatment, because we
    must run the coda + lm_head every loop to obtain logits for the signal.

This realises the Ouro convention's *decode* half exactly: a decoding token's
queries attend to each past token's **last loop-step** KV (one committed entry
per past position), and the current token re-derives its own KV on every loop.
Cost per generated token is O(n_loops * (slab+coda layers)) attention over the
committed past — linear in sequence length, never quadratic recompute.

SIMPLIFICATION (documented, intentional — transformers 5.9.0 here, not 4.57.6):
The Ouro *prefill* half keeps per-loop-step KV for prompt tokens so that, during
the looped prefill, token i can attend to token j's loop-step-k KV. We do **not**
retain per-loop-step KV at prefill; instead the prompt is prefilled with a fixed
``prefill_loops`` count (default 1) and only the final prefill loop's KV is
committed — i.e. prefill uses the same "last-step" convention as decode. This is
a *correct, causal* KV scheme (not an O(seq^2) fallback), it just gives prompt
tokens a fixed loop depth rather than the per-step retention Ouro uses to match
training-time looping. The first generated token is sampled from the prefill's
last-position logits and therefore also has the fixed ``prefill_loops`` depth
(its recorded loop count). All *subsequent* tokens are gated adaptively. With
``prefill_loops=1`` and ``n_max=1`` the whole procedure reduces exactly to plain
1-loop greedy decoding (see the test sanity check).

Windows: dynamo compilation errors suppressed (no cl.exe), per recur_model.
"""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

from transformers import DynamicCache

from recur_model import _build_causal_mask, _layer_fwd

if TYPE_CHECKING:
    from recur_model import RecurModel

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

# Windows: suppress dynamo compilation errors (no cl.exe available in most setups)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]
except Exception:
    pass

# Heuristic default thresholds. These SHOULD be calibrated from the held-out
# entropy/margin distribution of a trained adapter (plan Task 4, Step 1); the
# values below are sane starting points for BitNet's ~128k vocab (max entropy
# ln(128k) ~= 11.76 nats). Pass explicit thresholds to override.
DEFAULT_ENT_THRESH: float = 3.0
DEFAULT_MARGIN_THRESH: float = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fuse(model: "RecurModel", prelude_out: torch.Tensor,
          state: torch.Tensor) -> torch.Tensor:
    """Fusion adapter (or exact bypass when ``model._bypass_fusion``)."""
    if model._bypass_fusion:
        return prelude_out
    return model.fusion(torch.cat([prelude_out, state], dim=-1))


def _gate(model: "RecurModel", fused: torch.Tensor,
          slab_out: torch.Tensor) -> torch.Tensor:
    """Learned residual gate (or exact bypass when ``model._bypass_fusion``)."""
    if model._bypass_fusion:
        return slab_out
    return model.gate(fused, slab_out)


def _run_layers(layers, hidden: torch.Tensor, attn_mask, position_ids,
                position_embeddings, cache: DynamicCache,
                cache_position: torch.Tensor) -> torch.Tensor:
    """Run a contiguous span of HF decoder layers, updating ``cache`` in place."""
    for layer in layers:
        hidden = _layer_fwd(
            layer,
            hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    return hidden


def _committed_len(cache: DynamicCache, layer_idx: int) -> int:
    """Committed sequence length of ``layer_idx`` (0 if not yet initialised)."""
    if layer_idx < len(cache.layers):
        return int(cache.get_seq_length(layer_idx))
    return 0


def _crop_span(cache: DynamicCache, lo: int, hi: int, target_len: int) -> None:
    """Crop layers ``[lo, hi)`` back to ``target_len`` (no-op if shorter/absent).

    Uses ``cache.layers[li].crop(...)`` — pinned to transformers' internal
    DynamicCache per-layer structure (version-gated); the public
    ``DynamicCache.crop`` cannot do the per-span slab/coda crop this needs.
    """
    for li in range(lo, hi):
        if li < len(cache.layers):
            cache.layers[li].crop(target_len)


def _is_hard(logits_last: torch.Tensor, signal: str,
             ent_thresh: float, margin_thresh: float) -> bool:
    """Return True if the (B=1) token is "hard" under the chosen signal."""
    logf = logits_last.float()
    logp = torch.log_softmax(logf, dim=-1)
    logp_safe = torch.nan_to_num(logp, nan=0.0, neginf=0.0)  # 0*log(0)->0
    entropy = float((-(logp.exp() * logp_safe).sum(-1)).item())
    top2 = torch.topk(logf, 2, dim=-1).values
    margin = float((top2[..., 0] - top2[..., 1]).item())
    if signal == "entropy":
        return entropy > ent_thresh
    if signal == "margin":
        return margin < margin_thresh
    if signal == "both":
        return (entropy > ent_thresh) or (margin < margin_thresh)
    raise ValueError(f"unknown signal {signal!r} (expected entropy|margin|both)")


# ---------------------------------------------------------------------------
# generate_adaptive
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_adaptive(
    model: "RecurModel",
    input_ids: torch.Tensor,
    n_max: int = 8,
    signal: str = "both",
    ent_thresh: float = DEFAULT_ENT_THRESH,
    margin_thresh: float = DEFAULT_MARGIN_THRESH,
    max_new_tokens: int = 32,
    prefill_loops: int = 1,
    eos_token_id: int | None = None,
):
    """Adaptive looped-depth greedy decode with a live per-token exit gate.

    Parameters
    ----------
    model:
        A :class:`recur_model.RecurModel` (trained or untrained).
    input_ids:
        Prompt token ids, shape ``(1, seq)`` (batch size 1 only).
    n_max:
        Maximum slab loops per decoded token (Global Constraint: 8).
    signal:
        ``"entropy"``, ``"margin"`` or ``"both"`` (default, loop while either
        flags the token hard).
    ent_thresh, margin_thresh:
        Hardness thresholds; see module docstring. Calibrate on held-out data.
    max_new_tokens:
        Total number of tokens to generate (including the first, prefill-derived
        token).
    prefill_loops:
        Fixed slab loop count used for the prompt. With ``prefill_loops=1`` and
        ``n_max=1`` this reduces to plain 1-loop greedy decode.
    eos_token_id:
        Optional early stop.

    Returns
    -------
    (generated_ids, per_token_loops):
        ``generated_ids`` is a ``(1, n_generated)`` long tensor; ``per_token_loops``
        is a list of the loop counts spent on each generated token, same length
        as ``generated_ids.shape[1]``. Every entry lies in ``[1, n_max]``.
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, \
        "generate_adaptive supports batch size 1 only"
    assert n_max >= 1, "n_max must be >= 1"
    assert prefill_loops >= 1, "prefill_loops must be >= 1"

    core = model.base.model           # LlamaModel/BitNet backbone
    layers = core.layers
    P, Q, L = model.P, model.Q, len(layers)
    device = input_ids.device
    embed = core.embed_tokens
    dtype = embed.weight.dtype

    cache = DynamicCache()
    generated: list[int] = []
    per_token_loops: list[int] = []

    # ---------------------------------------------------------------- prefill
    seq0 = input_ids.shape[1]
    hidden = embed(input_ids)
    pos_ids = torch.arange(seq0, device=device).unsqueeze(0)
    pos_emb = core.rotary_emb(hidden, pos_ids)
    cmask = _build_causal_mask(seq0, dtype, device)
    cache_pos = torch.arange(seq0, device=device)

    # prelude — committed once
    prelude_out = _run_layers(layers[:P], hidden, cmask, pos_ids, pos_emb,
                              cache, cache_pos)
    # looped slab (fixed prefill_loops; only final loop's KV committed)
    state = prelude_out
    for _ in range(prefill_loops):
        _crop_span(cache, P, Q, 0)
        fused = _fuse(model, prelude_out, state)
        h = _run_layers(layers[P:Q], fused, cmask, pos_ids, pos_emb,
                        cache, cache_pos)
        state = _gate(model, fused, h)
    # coda + head — committed once
    hc = _run_layers(layers[Q:], state, cmask, pos_ids, pos_emb,
                     cache, cache_pos)
    last_logits = model.base.lm_head(core.norm(hc))[:, -1, :]

    tok = int(last_logits.argmax(-1).item())
    generated.append(tok)
    per_token_loops.append(prefill_loops)
    cur_pos = seq0

    # ----------------------------------------------------------------- decode
    while len(generated) < max_new_tokens:
        if eos_token_id is not None and tok == eos_token_id:
            break

        inp = torch.tensor([[tok]], device=device)
        hemb = embed(inp)
        pos_ids_d = torch.tensor([[cur_pos]], device=device)
        pos_emb_d = core.rotary_emb(hemb, pos_ids_d)
        cache_pos_d = torch.tensor([cur_pos], device=device)

        # prelude — commit current token's KV once (decode q_len=1 -> mask None)
        prelude_out = _run_layers(layers[:P], hemb, None, pos_ids_d, pos_emb_d,
                                  cache, cache_pos_d)

        slab_past = _committed_len(cache, P)   # = cur_pos (past tokens only)
        coda_past = _committed_len(cache, Q)   # = cur_pos

        state = prelude_out
        loops = 0
        last_logits = None
        while loops < n_max:
            # discard this token's KV from the previous loop iteration so it is
            # recomputed from the evolving recurrent state (last-step reuse).
            _crop_span(cache, P, Q, slab_past)
            _crop_span(cache, Q, L, coda_past)

            fused = _fuse(model, prelude_out, state)
            h = _run_layers(layers[P:Q], fused, None, pos_ids_d, pos_emb_d,
                            cache, cache_pos_d)
            state = _gate(model, fused, h)

            hc = _run_layers(layers[Q:], state, None, pos_ids_d, pos_emb_d,
                             cache, cache_pos_d)
            last_logits = model.base.lm_head(core.norm(hc))[:, -1, :]
            loops += 1

            if loops >= n_max or not _is_hard(last_logits, signal,
                                              ent_thresh, margin_thresh):
                break

        # final loop's KV stays committed (not cropped) -> future tokens attend it
        tok = int(last_logits.argmax(-1).item())
        generated.append(tok)
        per_token_loops.append(loops)
        cur_pos += 1

    gen_ids = torch.tensor([generated], device=device, dtype=input_ids.dtype)
    return gen_ids, per_token_loops
