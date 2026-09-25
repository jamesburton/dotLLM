#!/usr/bin/env python3
"""
bitnet_depth_expand.py — LLaMA-Pro-style depth expansion of a BitNet b1.58 checkpoint
using *zero-residual ternary identity blocks*.

Background / key insight
------------------------
BitNet b1.58 weights are ternary: every weight value is one of {-1, 0, +1} (scaled
by a per-tensor absmean factor). The value **0 is in that set**, so a transformer
block whose residual contribution is exactly zero is *exactly representable* in
ternary — there is no quantization error to "heal" before training even starts.

A BitNet decoder block has two residual adds (see
`transformers.models.bitnet.modeling_bitnet.BitNetDecoderLayer`):

    residual = h
    h = residual + self_attn(input_layernorm(h))         # attn branch
    residual = h
    h = residual + mlp(post_attention_layernorm(h))      # ffn branch

If we **zero-initialize** the attention output projection (`self_attn.o_proj`) and
the FFN output projection (`mlp.down_proj`), both branch outputs are exactly 0 at
init, so the block computes `h -> h` (an identity). Inserting such blocks anywhere
in the stack leaves the model's output *bit-for-bit unchanged* at init (up to fp
rounding), then heal-training learns useful residuals — this is exactly the
LLaMA-Pro recipe (arXiv:2401.02415), specialized to ternary.

BitNet specifics handled here
-----------------------------
* Master weights: `microsoft/bitnet-b1.58-2B-4T-bf16` stores plain bf16 `nn.Linear`
  weights (HF `BitNetForCausalLM`). Ternary quantization is applied *in the forward
  pass*, not on disk. So depth expansion operates on bf16 `.weight` tensors directly.
* Sub-norms: BitNet inserts an RMSNorm *inside* each BitLinear, right before the
  output projection — `self_attn.attn_sub_norm` (before o_proj) and
  `mlp.ffn_sub_norm` (before down_proj). These are copied from a base layer (weight
  = ones by default) so the inserted block matches the architecture exactly. They do
  not affect the identity property: a zero o_proj/down_proj kills the branch output
  regardless of what the sub-norm produced.
* absmean gamma -> 0 guard: at forward time the weight quantizer computes
  `scale = 1 / mean(|W|).clamp(min=1e-5)` then `W_q = round(W*scale).clamp(-1,1)/scale`
  (see `transformers.integrations.bitnet.WeightQuant`). For an all-zero W,
  `mean(|W|)=0` is clamped to 1e-5, giving `scale=1e5` and `W_q = round(0)/1e5 = 0`
  exactly — **no NaN, no div-by-zero**. So an all-zero proj is a valid ternary tensor.
  We default to exact zeros. `--near-zero-eps E` is offered to instead seed tiny
  values that still quantize to ternary 0 but live in the master weights; this is
  cosmetic for the master weights (STE passes gradients through a zero weight
  regardless) and is provided only for experimentation.

What this script does
---------------------
1. Loads a BitNet checkpoint dir (config + safetensors) with HF transformers.
2. Computes insertion positions (every `--every k` layers, LLaMA-Pro interleave; or
   explicit `--at i,j,k`). Each insertion *duplicates* the structure of an adjacent
   block (so head/dim shapes match), then identity-inits o_proj + down_proj.
3. Renumbers all layers contiguously, sets `config.num_hidden_layers`, and saves the
   expanded checkpoint + config to `--out`.

It also ships a `--self-test` that builds a TINY synthetic BitNet in-process and
verifies the expanded model's logits equal the base model's, with no big download.

Usage
-----
    # Self-test (no download, ~seconds on CPU):
    python scripts/bitnet_depth_expand.py --self-test

    # Real expansion of the bf16 master weights (download them first, see below):
    python scripts/bitnet_depth_expand.py \
        --src /path/to/bitnet-b1.58-2B-4T-bf16 \
        --out /path/to/bitnet-2B-depth40 \
        --every 4

    # Or insert at explicit positions:
    python scripts/bitnet_depth_expand.py --src SRC --out OUT --at 8,16,24

See `--help` for all flags.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Torch / transformers are imported lazily inside main() so that `--help` works
# even in an environment without them installed.


# ----------------------------------------------------------------------------
# Insertion-position planning (LLaMA-Pro interleave)
# ----------------------------------------------------------------------------
def plan_insertions(num_layers: int, every: Optional[int], at: Optional[list[int]]) -> list[int]:
    """Return the list of *base-layer indices after which* a new block is inserted.

    LLaMA-Pro interleaving: starting from a model of L blocks, to add a group of new
    blocks evenly, insert one new block after every `every` original blocks. The
    returned positions are expressed against the ORIGINAL layer numbering: a value
    `p` means "insert one new identity block immediately after original layer p"
    (0-based). p == -1 would mean "at the very front" (not used by --every).

    With explicit `--at`, positions are insertion slots in the original index space:
    `--at 8` inserts before original layer 8 (i.e. after original layer 7).
    """
    if at is not None:
        # --at gives "insert *before* these original indices". Normalize to
        # "after index a-1". Allow 0 (front) and num_layers (end).
        positions = []
        for a in at:
            if a < 0 or a > num_layers:
                raise ValueError(f"--at index {a} out of range [0, {num_layers}]")
            positions.append(a - 1)  # after (a-1)
        return sorted(positions)

    if every is not None:
        if every < 1:
            raise ValueError("--every must be >= 1")
        # Insert after original layers (every-1), (2*every-1), ... i.e. one new block
        # per group of `every` original blocks. This is the LLaMA-Pro pattern.
        positions = list(range(every - 1, num_layers, every))
        # Drop a trailing insertion that would sit after the final layer only if the
        # user clearly wanted interior interleaving; keeping the tail insertion is
        # harmless (it's still an identity), so we keep it for an even spread.
        return positions

    raise ValueError("Specify exactly one of --every or --at")


# ----------------------------------------------------------------------------
# Identity initialization of a single inserted block
# ----------------------------------------------------------------------------
def identity_init_block(block, near_zero_eps: float = 0.0) -> None:
    """Make an inserted BitNetDecoderLayer a zero-residual identity at init.

    Zero (or near-zero) the attention output projection and the FFN output
    projection so both residual branches contribute 0. All other params keep the
    structure copied from the template block (sub-norms = ones, etc.).
    """
    import torch

    o_proj = block.self_attn.o_proj.weight
    down_proj = block.mlp.down_proj.weight

    with torch.no_grad():
        if near_zero_eps and near_zero_eps > 0.0:
            # Seed tiny symmetric values. These still quantize to ternary 0 in the
            # forward pass (|w| << gamma threshold), but exist in master weights.
            o_proj.normal_(mean=0.0, std=near_zero_eps)
            down_proj.normal_(mean=0.0, std=near_zero_eps)
        else:
            o_proj.zero_()
            down_proj.zero_()

        # Defensive: if the proj layers carry a bias (BitNet default is bias=False),
        # zero it too so the branch output is exactly 0.
        if getattr(block.self_attn.o_proj, "bias", None) is not None:
            block.self_attn.o_proj.bias.zero_()
        if getattr(block.mlp.down_proj, "bias", None) is not None:
            block.mlp.down_proj.bias.zero_()


# ----------------------------------------------------------------------------
# Core expansion on a live BitNetForCausalLM
# ----------------------------------------------------------------------------
def expand_model(model, positions: list[int], near_zero_eps: float = 0.0):
    """Insert identity blocks into a live BitNetForCausalLM and return (model, info).

    `positions[i]` = original layer index after which to insert one new block.
    New blocks are constructed via the model's own layer class, on the model's
    device/dtype, structurally cloned from the nearest existing block (template),
    then identity-initialized. Layers are renumbered and config updated.
    """
    import copy
    import torch
    from torch import nn

    base_layers = model.model.layers
    orig_n = len(base_layers)
    cfg = model.config

    # Build the new ordered list of layers. We walk original indices 0..orig_n-1 and
    # after each index that appears in `positions` we splice in a fresh identity block.
    # Support multiple insertions at the same position (a multiset).
    from collections import Counter

    pos_counts = Counter(positions)

    new_layers: list[nn.Module] = []

    def make_identity_like(template_layer, layer_idx: int):
        # Construct a brand-new layer of the same class with the right layer_idx, on
        # the same device/dtype, then identity-init. We deepcopy the template to
        # inherit exact submodule structure (handles any arch quirks), then reset the
        # two output projections; sub-norms are reset to ones for a clean identity.
        layer = copy.deepcopy(template_layer)
        # Fix the cached layer_idx used by the attention module for the KV cache.
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = layer_idx
        with torch.no_grad():
            # Reset sub-norms to identity (ones) so the inserted block starts neutral.
            if hasattr(layer.self_attn, "attn_sub_norm"):
                layer.self_attn.attn_sub_norm.weight.fill_(1.0)
            if hasattr(layer.mlp, "ffn_sub_norm"):
                layer.mlp.ffn_sub_norm.weight.fill_(1.0)
        identity_init_block(layer, near_zero_eps=near_zero_eps)
        return layer

    # Handle a front insertion (position == -1) before the loop.
    next_idx = 0
    n_front = pos_counts.get(-1, 0)
    for _ in range(n_front):
        new_layers.append(make_identity_like(base_layers[0], next_idx))
        next_idx += 1

    for orig_i in range(orig_n):
        # Keep the original layer (we'll fix its layer_idx after the full list is built).
        new_layers.append(base_layers[orig_i])
        k = pos_counts.get(orig_i, 0)
        for _ in range(k):
            template = base_layers[min(orig_i, orig_n - 1)]
            new_layers.append(make_identity_like(template, 0))  # idx fixed below

    # Reassign contiguous layer indices everywhere they are cached.
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = i

    model.model.layers = nn.ModuleList(new_layers)
    cfg.num_hidden_layers = len(new_layers)

    info = {
        "original_layers": orig_n,
        "inserted": len(new_layers) - orig_n,
        "final_layers": len(new_layers),
        "insert_after_original_idx": sorted(positions),
    }
    return model, info


# ----------------------------------------------------------------------------
# Self-test: tiny synthetic model, no download
# ----------------------------------------------------------------------------
def build_tiny_bitnet():
    """Build a TINY synthetic BitNetForCausalLM for the self-test (no download)."""
    import torch
    from transformers import BitNetConfig, BitNetForCausalLM

    cfg = BitNetConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
        bos_token_id=None,
        eos_token_id=None,
    )
    torch.manual_seed(0)
    model = BitNetForCausalLM(cfg)
    model.eval()
    # Randomize the RMSNorm weights away from 1.0 so the test is non-trivial
    # (an all-ones-norm model would pass for boring reasons).
    with torch.no_grad():
        for p_name, p in model.named_parameters():
            if p.dim() == 1 and "norm" in p_name:
                p.normal_(mean=1.0, std=0.05)
    return model


def run_self_test(every: int = 1, near_zero_eps: float = 0.0, tol: float = 1e-3) -> bool:
    import copy

    import torch

    print("[self-test] building tiny synthetic BitNet (hidden=64, 2 layers, 4 heads)...")
    base = build_tiny_bitnet()
    cfg = base.config

    torch.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))

    with torch.no_grad():
        base_logits = base(input_ids).logits

    positions = plan_insertions(cfg.num_hidden_layers, every=every, at=None)
    print(f"[self-test] inserting identity blocks after original layers {positions}")

    expanded = copy.deepcopy(base)
    expanded, info = expand_model(expanded, positions, near_zero_eps=near_zero_eps)
    expanded.eval()
    print(f"[self-test] layers: {info['original_layers']} -> {info['final_layers']} "
          f"(+{info['inserted']} identity)")

    with torch.no_grad():
        exp_logits = expanded(input_ids).logits

    assert exp_logits.shape == base_logits.shape, (
        f"shape mismatch {exp_logits.shape} vs {base_logits.shape}")
    max_abs = (exp_logits - base_logits).abs().max().item()
    mean_abs = (exp_logits - base_logits).abs().mean().item()
    print(f"[self-test] logit diff: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} (tol={tol:.1e})")

    # Also verify argmax (next-token prediction) is identical at every position.
    base_arg = base_logits.argmax(-1)
    exp_arg = exp_logits.argmax(-1)
    argmax_match = bool((base_arg == exp_arg).all().item())
    print(f"[self-test] argmax identical at all positions: {argmax_match}")

    ok = (max_abs <= tol) and argmax_match
    if near_zero_eps and near_zero_eps > 0.0:
        # With near-zero eps the master weights are perturbed; identity holds only
        # because forward-time ternary quantization rounds them to 0. Report that.
        print(f"[self-test] (near_zero_eps={near_zero_eps} — relies on forward ternary "
              f"quant rounding tiny weights to 0)")

    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="bitnet_depth_expand.py",
        description="LLaMA-Pro-style depth expansion of a BitNet b1.58 checkpoint with "
                    "zero-residual ternary identity blocks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/bitnet_depth_expand.py --self-test\n"
            "  python scripts/bitnet_depth_expand.py --src DIR --out OUT --every 4\n"
            "  python scripts/bitnet_depth_expand.py --src DIR --out OUT --at 8,16,24\n"
        ),
    )
    p.add_argument("--self-test", action="store_true",
                   help="Run the in-process tiny-model identity validation (no download) and exit.")
    p.add_argument("--src", type=str, default=None,
                   help="Source BitNet checkpoint dir (e.g. a local copy of "
                        "microsoft/bitnet-b1.58-2B-4T-bf16).")
    p.add_argument("--out", type=str, default=None,
                   help="Output dir for the expanded checkpoint + config.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--every", type=int, default=None,
                   help="LLaMA-Pro interleave: insert one identity block after every k "
                        "original layers (e.g. --every 4 turns 30 layers into ~37).")
    g.add_argument("--at", type=str, default=None,
                   help="Comma-separated original layer indices to insert *before* "
                        "(e.g. --at 8,16,24). Use 0 for front, num_layers for end.")
    p.add_argument("--near-zero-eps", type=float, default=0.0,
                   help="If >0, seed the zero projections with N(0, eps) tiny values "
                        "instead of exact zeros. They still quantize to ternary 0 in "
                        "forward. Default 0.0 (exact zeros). For experimentation only.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Torch dtype to load/save the model in. Default bfloat16.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device to load the model on for the splice (cpu recommended; "
                        "the operation is memory-bandwidth bound, not compute).")
    p.add_argument("--verify", action="store_true",
                   help="After expansion, run a forward pass on a random prompt and "
                        "compare expanded vs base logits (proves identity on the REAL "
                        "model). Loads the base a second time; needs the RAM.")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Self-test / --verify max-abs logit tolerance. Default 1e-3.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.self_test:
        ok = run_self_test(every=1 if args.every is None else args.every,
                           near_zero_eps=args.near_zero_eps, tol=args.tol)
        return 0 if ok else 1

    if not args.src or not args.out:
        print("error: --src and --out are required (or use --self-test).", file=sys.stderr)
        return 2

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    at = None
    if args.at is not None:
        at = [int(x) for x in args.at.split(",") if x.strip() != ""]

    print(f"[expand] loading config from {args.src} ...")
    cfg = AutoConfig.from_pretrained(args.src, trust_remote_code=True)
    num_layers = cfg.num_hidden_layers
    positions = plan_insertions(num_layers, every=args.every, at=at)
    print(f"[expand] base num_hidden_layers={num_layers}; "
          f"inserting {len(positions)} block(s) after original layers {positions}")

    print(f"[expand] loading weights ({args.dtype}) on {args.device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.src, dtype=dtype, trust_remote_code=True
    ).to(args.device)
    model.eval()

    base_logits = None
    verify_ids = None
    if args.verify:
        torch.manual_seed(0)
        verify_ids = torch.randint(0, cfg.vocab_size, (1, 16)).to(args.device)
        with torch.no_grad():
            base_logits = model(verify_ids).logits.float().cpu()

    model, info = expand_model(model, positions, near_zero_eps=args.near_zero_eps)
    print(f"[expand] layers {info['original_layers']} -> {info['final_layers']} "
          f"(+{info['inserted']} identity); config.num_hidden_layers={model.config.num_hidden_layers}")

    if args.verify:
        with torch.no_grad():
            exp_logits = model(verify_ids).logits.float().cpu()
        max_abs = (exp_logits - base_logits).abs().max().item()
        argmax_ok = bool((exp_logits.argmax(-1) == base_logits.argmax(-1)).all().item())
        print(f"[expand][verify] max_abs logit diff={max_abs:.3e} (tol={args.tol:.1e}), "
              f"argmax identical={argmax_ok}")
        if max_abs > args.tol or not argmax_ok:
            print("[expand][verify] WARNING: identity not within tolerance!", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    print(f"[expand] saving expanded checkpoint to {args.out} ...")
    model.save_pretrained(args.out, safe_serialization=True)
    # Copy tokenizer if present in the source so the output is self-contained.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
        tok.save_pretrained(args.out)
        print("[expand] tokenizer copied.")
    except Exception as e:  # noqa: BLE001
        print(f"[expand] (no tokenizer copied: {e})")

    print("[expand] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
