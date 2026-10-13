"""
Masked Diffusion Language Model — model architecture, masking process, loss, sampling.

Implements the MDLM / LLaDA absorbing-mask discrete diffusion objective for a
standalone bidirectional (non-causal) transformer.

References:
  MDLM: Sahoo et al., arXiv 2406.07524 (Simple Loss, ELBO derivation)
  LLaDA: Nie et al., arXiv 2502.09992 (absorbing-mask forward process, iterative unmasking)

Key design choices:
  - Architecture: bidirectional attention (NO causal mask) — required for MDLM objective
  - Pre-norm transformer blocks (more stable training than post-norm)
  - Learned positional embeddings (absolute)
  - Weight tying: lm_head shares weights with token embedding
  - Mask token id = vocab_size - 1 by default (last vocab slot)

Absorbing-mask forward process:
  q(x_t | x_0) = prod_i [ t * delta(x_i^t = MASK) + (1-t) * delta(x_i^t = x_i^0) ]
  At t=0: no masking. At t=1: fully masked.

MDLM simple loss (uniform over t, no 1/t weighting):
  L(theta) = E_{t~U[t_min, t_max]} E_{mask} [ CE(p_theta(x_i | x_mask), x_i) for i in MASK ]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MDLMConfig:
    """
    Hyperparameters for MDLMModel.

    mask_token_id defaults to vocab_size - 1 (set explicitly if using a pre-existing
    tokenizer that already reserves a [MASK] token at a specific id).
    """
    n_layers:      int   = 12
    d_model:       int   = 768
    n_heads:       int   = 12
    ffn_mult:      int   = 4
    vocab_size:    int   = 32000
    max_seq_len:   int   = 512
    mask_token_id: int   = -1    # -1 means resolve to vocab_size - 1
    dropout:       float = 0.1

    def __post_init__(self) -> None:
        if self.mask_token_id < 0:
            self.mask_token_id = self.vocab_size - 1
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )

    def as_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ffn_mult": self.ffn_mult,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "mask_token_id": self.mask_token_id,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MDLMConfig":
        return cls(**d)


# Named presets used by the --config flag in the CLI
CONFIGS: dict[str, MDLMConfig] = {
    # CPU smoke-test model: 2-layer, 128-dim, tiny vocab
    "smoke": MDLMConfig(
        n_layers=2, d_model=128, n_heads=4, ffn_mult=4,
        vocab_size=128, max_seq_len=64, dropout=0.0,
    ),
    "tiny": MDLMConfig(
        n_layers=2, d_model=128, n_heads=4, ffn_mult=4,
        vocab_size=128, max_seq_len=64, dropout=0.0,
    ),
    # Dense 117M de-risk (GPT-2 Small equivalent, bidirectional).
    # vocab_size overridden at runtime from the tokenizer.
    "derisk117m": MDLMConfig(
        n_layers=12, d_model=768, n_heads=12, ffn_mult=4,
        vocab_size=32002,  # placeholder; set --vocab-size from tokenizer at runtime
        max_seq_len=512, dropout=0.1,
    ),
}


# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------

class BidirectionalSelfAttention(nn.Module):
    """
    Full (non-causal) multi-head self-attention.

    CRITICAL: no causal mask is applied. Every position attends to every other
    position. This is what distinguishes a masked-diffusion LM from a causal AR LM.
    """

    def __init__(self, config: MDLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head  = config.d_model // config.n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv      = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model,     bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv(x)                                  # [B, L, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                     # each [B, L, D]

        # Reshape to [B, H, L, Dh]
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # Scaled dot-product attention — NO causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale   # [B, H, L, L]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                         # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → residual + LayerNorm → FFN → residual."""

    def __init__(self, config: MDLMConfig) -> None:
        super().__init__()
        d_ffn = config.d_model * config.ffn_mult

        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn  = BidirectionalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(config.d_model, d_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_ffn, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MDLMModel(nn.Module):
    """
    Bidirectional transformer for masked-diffusion language modelling.

    Input: token ids [B, L], where some positions equal config.mask_token_id.
    Output: logits [B, L, vocab_size] over token predictions at all positions.

    The model uses FULL (non-causal) attention so each position can attend to
    all other positions, including those that are not masked. This is essential
    for the MDLM denoising objective: the model predicts masked tokens from
    the available unmasked context.
    """

    def __init__(self, config: MDLMConfig) -> None:
        super().__init__()
        self.config  = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: token embedding and output projection share the same matrix
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Standard GPT-style weight init: N(0, 0.02) for linear/embed, 1/0 for layernorm."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token ids (integers), may include mask_token_id values.
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        assert L <= self.config.max_seq_len, (
            f"Sequence length {L} exceeds max_seq_len={self.config.max_seq_len}"
        )
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)  # [1, L]
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))   # [B, L, D]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)                                        # [B, L, V]

    @property
    def n_params(self) -> int:
        """Total parameter count (includes tied params once)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total


# ---------------------------------------------------------------------------
# Absorbing-mask forward process
# ---------------------------------------------------------------------------

def apply_mask(
    tokens: torch.Tensor,
    t: float,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the absorbing-state (masking) forward process at noise level t ∈ [0,1].

    Each token position i is independently replaced with mask_token_id with
    probability t. At least one position per sequence is guaranteed to be masked.

    Args:
        tokens:        [B, L] clean integer token ids
        t:             noise level, scalar ∈ [0, 1]
        mask_token_id: absorbing-state token id (the [MASK] token)

    Returns:
        masked_tokens: [B, L] partially masked token ids
        mask:          [B, L] bool, True where token was replaced with MASK
    """
    B, L = tokens.shape
    device = tokens.device

    # Bernoulli masking: each position masked independently with probability t
    mask = torch.bernoulli(
        torch.full((B, L), t, device=device, dtype=torch.float32)
    ).bool()

    # Guarantee at least one masked position per sequence (avoids zero-gradient steps)
    for b in range(B):
        if not mask[b].any():
            idx = torch.randint(0, L, (1,), device=device)
            mask[b, idx] = True

    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token_id
    return masked_tokens, mask


# ---------------------------------------------------------------------------
# MDLM objective
# ---------------------------------------------------------------------------

def mdlm_simple_loss(
    model: MDLMModel,
    tokens: torch.Tensor,
    t: Optional[float] = None,
    t_min: float = 0.05,
    t_max: float = 0.95,
) -> tuple[torch.Tensor, float, float]:
    """
    MDLM simple loss (LLaDA variant): CE at masked positions, uniform over noise level t.

    L_simple(theta) = E_{t~U[t_min,t_max]} E_{mask~Bernoulli(t)}
                        [ CE( p_theta(x_i | x_masked), x_i ) for i in MASK positions ]

    Note: the 1/t ELBO weighting from the original MDLM paper is omitted (the "simple"
    variant). LLaDA showed this works as well or better empirically.

    Args:
        model:  MDLMModel
        tokens: [B, L] clean token ids
        t:      noise level (sampled uniformly if None)
        t_min:  minimum noise level to sample from (clips near-zero mask)
        t_max:  maximum noise level to sample from (clips near-full mask)

    Returns:
        (loss, t_used, mask_fraction)
        loss:          scalar CE loss at masked positions
        t_used:        the noise level used
        mask_fraction: fraction of tokens that were masked
    """
    if t is None:
        t = float(torch.empty(1).uniform_(t_min, t_max))

    masked_tokens, mask = apply_mask(tokens, t, model.config.mask_token_id)
    logits = model(masked_tokens)                  # [B, L, V]

    # Loss only at masked positions
    masked_logits  = logits[mask]                  # [N_masked, V]
    masked_targets = tokens[mask]                  # [N_masked]

    loss = F.cross_entropy(masked_logits, masked_targets)
    mask_frac = mask.float().mean().item()
    return loss, t, mask_frac


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_masked_accuracy(
    model: MDLMModel,
    tokens: torch.Tensor,
    t: float = 0.5,
    n_repeats: int = 10,
    seed: int = 42,
) -> float:
    """
    Estimate masked-token prediction accuracy at noise level t.

    Averages over n_repeats independent random masks drawn with fixed seed.
    Returns fraction of masked positions correctly predicted by argmax.

    Args:
        model:     MDLMModel in eval mode
        tokens:    [B, L] clean token ids
        t:         mask rate (0.5 = half tokens masked)
        n_repeats: number of independent mask samples to average over
        seed:      RNG seed for reproducibility

    Returns:
        accuracy: float in [0, 1]
    """
    model.eval()
    gen = torch.Generator(device=tokens.device)
    gen.manual_seed(seed)

    total_correct = 0
    total_masked  = 0

    for _ in range(n_repeats):
        mask = torch.bernoulli(
            torch.full(tokens.shape, t, device=tokens.device, dtype=torch.float32),
            generator=gen,
        ).bool()
        # Ensure at least one masked per sequence
        for b in range(tokens.size(0)):
            if not mask[b].any():
                mask[b, 0] = True

        masked_tokens = tokens.clone()
        masked_tokens[mask] = model.config.mask_token_id

        logits = model(masked_tokens)               # [B, L, V]
        preds  = logits.argmax(dim=-1)              # [B, L]

        total_correct += (preds[mask] == tokens[mask]).sum().item()
        total_masked  += mask.sum().item()

    return total_correct / max(total_masked, 1)


@torch.no_grad()
def eval_bpd(
    model: MDLMModel,
    tokens: torch.Tensor,
    n_t_samples: int = 50,
) -> float:
    """
    Estimate bits-per-dimension (BPD) via the MDLM simple ELBO.

    BPD ≈ (1/ln2) * E_{t~U[0.05,0.95]}[ CE_t ]

    Averages CE loss over n_t_samples linearly-spaced noise levels.
    Suitable for tracking training progress and comparing to published MDLM results.

    Args:
        model:       MDLMModel in eval mode
        tokens:      [B, L] clean token ids (one or more sequences)
        n_t_samples: number of noise levels to integrate over

    Returns:
        bpd: float (lower = better; MDLM 110M ≈ 1.78 BPD on LM1B test)
    """
    model.eval()
    t_values  = torch.linspace(0.05, 0.95, n_t_samples).tolist()
    total_ce  = 0.0
    n_valid   = 0

    for t_val in t_values:
        masked_tokens, mask = apply_mask(tokens, t_val, model.config.mask_token_id)
        if not mask.any():
            continue
        logits = model(masked_tokens)
        ce = F.cross_entropy(logits[mask], tokens[mask]).item()
        total_ce += ce
        n_valid  += 1

    mean_ce = total_ce / max(n_valid, 1)
    bpd     = mean_ce / math.log(2)
    return bpd


# ---------------------------------------------------------------------------
# Iterative unmasking (generation / inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def iterative_unmask(
    model: MDLMModel,
    seq_len: int,
    n_steps: int = 10,
    device: str = "cpu",
    temperature: float = 1.0,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Generate a sequence of length seq_len via iterative unmasking.

    Algorithm (greedy MDLM):
      1. Start with all tokens = mask_token_id.
      2. At each step:
         a. Forward pass (bidirectional attention over current sequence).
         b. Compute prediction confidence (max softmax probability) at masked positions.
         c. Unmask ceil(n_masked / remaining_steps) highest-confidence positions.
         d. Sample token from the predicted distribution at those positions.
      3. Repeat until all positions unmasked.

    This produces a fully-unmasked sequence in n_steps forward passes, vs O(L) for AR.

    Args:
        model:       MDLMModel in eval mode
        seq_len:     desired sequence length
        n_steps:     number of iterative unmasking steps
        device:      'cpu' or 'cuda'
        temperature: softmax temperature for sampling (1.0 = no scaling)
        greedy:      if True, use argmax instead of multinomial sampling

    Returns:
        tokens: [1, seq_len] fully unmasked token ids (no mask_token_id remaining)
    """
    model.eval()
    mask_id = model.config.mask_token_id

    # Start fully masked
    tokens = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)

    for step in range(n_steps):
        remaining = n_steps - step
        logits = model(tokens)                           # [1, L, V]

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits[0], dim=-1)         # [L, V]

        is_masked = (tokens[0] == mask_id)               # [L] bool
        n_masked  = int(is_masked.sum())
        if n_masked == 0:
            break

        # Number to unmask this step: spread evenly across remaining steps
        n_unmask = max(1, math.ceil(n_masked / remaining))

        # Select positions by prediction confidence (max prob among non-mask tokens)
        confidence = probs.max(dim=-1).values            # [L]
        confidence = confidence.masked_fill(~is_masked, -1e9)
        _, topk_pos = confidence.topk(min(n_unmask, n_masked))

        for pos in topk_pos:
            pos = pos.item()
            p = probs[pos]                               # [V]
            # Suppress the mask token from being predicted
            p[mask_id] = 0.0
            p = p / p.sum().clamp(min=1e-9)

            if greedy:
                tokens[0, pos] = p.argmax()
            else:
                tokens[0, pos] = torch.multinomial(p, num_samples=1)

    # Final force-unmask: any remaining mask tokens get argmax prediction
    still_masked = (tokens[0] == mask_id)
    if still_masked.any():
        logits = model(tokens)
        preds  = logits[0, still_masked].argmax(dim=-1)
        # Suppress mask token in argmax output
        preds  = torch.where(preds == mask_id,
                             torch.zeros_like(preds), preds)
        tokens[0, still_masked] = preds

    return tokens
