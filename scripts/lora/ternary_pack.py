#!/usr/bin/env python3
"""ternary_pack.py — self-contained ternary weight packing for frozen BitLinear layers.

A frozen BitLinear's ternary forward value (bitdistill.weight_quant_ternary) is a
deterministic, per-tensor-absmean function of its (unchanging) weight — so it can be
precomputed once and stored packed (~0.25 bytes/param) instead of a full bf16 master
(2 bytes/param), an ~8x reduction. This is what makes a "hot window" staged trainer
(bitdistill_staged.py) fit models that don't fit under full-joint training.

No dependency on bitdistill.py (mirrors gptq_init.py's precedent of a local copy of
the quant formula) so this module stays import-light for fast, no-GPU unit testing.

Packed format: simple sequential 2-bit packing, 4 codes per byte, ONE per-tensor f32
scale trailing (not stored inline here — callers keep it as a separate float/buffer).
Deliberately NOT dotLLM's I2_S byte layout (see docs/QUANTIZATION.md / MatMul.I2S.cs)
-- that interleaved block layout exists purely for CPU AVX2 SIMD-gather decode
efficiency on the production inference side. This packed buffer only ever lives in
PyTorch/CUDA memory for the duration of training and is always fully unpacked back to
ordinary BitLinear weights before bitdistill_export.py runs, so byte-for-byte I2_S
compatibility buys nothing here and the interleave would only add complexity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant_ternary_codes(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Same per-tensor absmean scale/rounding as bitdistill.weight_quant_ternary,
    but returns the discrete {0,1,2} codes (dequant = (code-1)/scale) instead of
    the dequantized tensor -- codes are what gets packed."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    trits = (w * scale).round().clamp(-1, 1)          # {-1, 0, +1}
    codes = (trits + 1).to(torch.uint8)                # {0, 1, 2}
    return codes, float(scale)


def pack_ternary(codes: torch.Tensor) -> torch.Tensor:
    """codes: uint8 tensor, values in {0,1,2}, any shape -> flat packed uint8,
    4 codes/byte, sequential: byte_k = c[4k] | c[4k+1]<<2 | c[4k+2]<<4 | c[4k+3]<<6.
    Padded to a multiple of 4 with code=1 (dequants to 0, inert under a sum/matmul)."""
    flat = codes.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = F.pad(flat, (0, pad), value=1)
    flat = flat.view(-1, 4)
    return (flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)).to(torch.uint8)


def unpack_ternary(packed: torch.Tensor, numel: int, shape, scale: float,
                   dtype: torch.dtype = torch.bfloat16, device=None) -> torch.Tensor:
    """Inverse of pack_ternary + the scale fold-in: reconstructs the SAME dense
    tensor bitdistill.weight_quant_ternary(w) would produce for the original w."""
    b = packed.to(device) if device is not None else packed
    c0 = b & 0x3
    c1 = (b >> 2) & 0x3
    c2 = (b >> 4) & 0x3
    c3 = (b >> 6) & 0x3
    codes = torch.stack([c0, c1, c2, c3], dim=1).reshape(-1)[:numel]
    return ((codes.to(torch.float32) - 1.0) / scale).to(dtype).reshape(shape)


class PackedTernaryLinear(nn.Module):
    """Frozen replacement for a BitLinear: stores its ternary weight packed (~0.25
    bytes/param) instead of a bf16 master (2 bytes/param). No gradient flows to the
    weight (there's nothing left to update -- the fp master is gone); gradient still
    flows to upstream layers through the input, since activation quantization keeps
    its STE path. Round-trips with BitLinear via from_bitlinear()/to_bitlinear() so a
    layer can move in and out of a training "hot window" over the course of a run."""

    def __init__(self, packed: torch.Tensor, numel: int, shape, scale: float,
                 sub_norm: nn.Module | None, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("packed", packed, persistent=True)
        self.numel = int(numel)
        self.shape = tuple(shape)
        self.register_buffer("scale", torch.tensor(float(scale)), persistent=True)
        self.sub_norm = sub_norm

    @classmethod
    def from_bitlinear(cls, bl) -> "PackedTernaryLinear":
        """bl: a bitdistill.BitLinear. Packs its current master weight (whatever
        value it has right now -- warm-started, trained, or freshly RTN-init'd)."""
        w = bl.weight.detach()
        codes, scale = weight_quant_ternary_codes(w)
        packed = pack_ternary(codes).to(w.device)
        sub_norm = bl.sub_norm
        if sub_norm is not None:
            for p in sub_norm.parameters():
                p.requires_grad_(False)
        return cls(packed, w.numel(), tuple(w.shape), scale, sub_norm,
                   bl.in_features, bl.out_features)

    def to_bitlinear(self, dtype: torch.dtype = torch.bfloat16):
        """Unfreeze: dequantize the packed ternary value into a fresh trainable
        BitLinear master weight (this IS what gptq-style warm-start would also
        produce via one-shot RTN, so no information is invented -- the layer simply
        resumes gradient-based refinement from its current best ternary estimate).

        weight_quant_ternary is NOT idempotent on its own dequantized output: its
        scale = 1/mean(|w|) is recomputed from whichever w it's given, and for a
        tensor with any exact zeros mean(|w|) < mean(|w|) of the pre-zero'd trits,
        so re-applying it (which BitLinear.forward always does) would rescale the
        seed by the tensor's nonzero-fraction. Since weight_quant_ternary(c*w) ==
        c*weight_quant_ternary(w) for any positive c, pre-dividing the seed by that
        same nonzero-fraction exactly cancels the effect -- BitLinear's own forward
        then reproduces this layer's packed value bit-for-bit on the first call."""
        from bitdistill import BitLinear  # local import: avoid a hard bitdistill.py
                                          # dependency for callers that only need
                                          # the packing primitives (e.g. unit tests).
        bl = BitLinear(self.in_features, self.out_features,
                       sub_norm=self.sub_norm is not None)
        # BitLinear.__init__ allocates its weight via nn.Parameter(torch.empty(out, in))
        # with NO dtype/device kwarg -- that's the torch global default (fp32, CPU), so the
        # line above transiently holds a full fp32 CPU copy of this layer's weight. Left as
        # `bl.weight.copy_(...)` + a later `.to(dtype)`, that fp32 tensor survives until the
        # .to() call, doubling (attn) to ~4x (MLP, bf16->fp32) the live footprint right when
        # unfreeze_window() does this for a whole hot window back-to-back. Since no
        # empty_cache() runs between unfreeze_window() and the training loop, the CUDA
        # allocator's reserved high-water mark from this transient spike sticks for the
        # WHOLE window -- measured contributor to real 4B usage (~11.4GB) running ~2x the
        # plan's steady-state estimate (~5.8GB). Fix: replace the Parameter with one already
        # in the target dtype/device before ever writing to it, so the wasteful fp32 tensor
        # is dropped (GC'd) immediately after construction instead of living through the
        # unpack+copy.
        with torch.no_grad():
            w = unpack_ternary(self.packed, self.numel, self.shape,
                               float(self.scale), dtype=dtype, device=self.packed.device)
            nonzero_frac = (w != 0).float().mean().clamp(min=1e-8)
            bl.weight = nn.Parameter((w / nonzero_frac).contiguous())
        if self.sub_norm is not None:
            bl.sub_norm.load_state_dict(self.sub_norm.state_dict())
            bl.sub_norm.to(device=self.packed.device, dtype=dtype)
            for p in bl.sub_norm.parameters():
                p.requires_grad_(True)
        return bl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sub_norm is not None:
            x = self.sub_norm(x)
        from bitdistill import _ste, activation_quant_8bit  # noqa: PLC0415 (see to_bitlinear)
        x_eff = _ste(x, activation_quant_8bit(x))
        # Always unpack fresh -- deliberately NOT caching the unpacked weight across the
        # microbatches/recompute calls of one opt.step(), despite that being a real, measured
        # compute saving (~1.4-3.3ms per BitLinear at Qwen3-4B shapes). A prior version of this
        # method did exactly that (`_step_cache`, cleared once per step by clear_step_caches()
        # below) reasoning it was memory-safe since nothing survives PAST a step -- true in
        # isolation, but it missed that WITHIN one step essentially every frozen layer in the
        # model (~32/36 at hot-window=4) ends up caching simultaneously, since the forward pass
        # touches every layer before backward's checkpoint-recompute touches any of them again.
        # Measured directly (2026-07-19, Qwen3-1.7B/4-layer hot window, micro-batch=1 against
        # cached batch_size=4): the cache raised peak reserved CUDA memory 11.10G -> vs 8.89G
        # without it -- a difference that, on the real 4B run, was consistently enough to tip
        # training from ~9s/step into sustained WDDM host-RAM paging (9-10x slower). Given any
        # model that needs hot-window staged training in the first place is already in the
        # memory-constrained regime by construction, the redundant-unpack compute cost here is
        # far cheaper than even one paging episode.
        return F.linear(x_eff, unpack_ternary(self.packed, self.numel, self.shape,
                                              float(self.scale), dtype=x.dtype, device=x.device))


def clear_step_caches(model) -> None:
    """No-op retained for API compatibility -- PackedTernaryLinear no longer caches an
    unpacked weight across a step (see the comment in forward() for why), so there is
    nothing left to clear. Callers (bitdistill_staged.py's post_step_hook wiring) are
    unchanged; this just does nothing now."""
    return


def _self_test() -> bool:
    """No-GPU round-trip checks. Run: python ternary_pack.py --self-test"""
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
        print(f"[self-test] {name}: {status}", flush=True)

    # 1. pack/unpack round-trip matches weight_quant_ternary's own dequant, for a
    #    variety of shapes including numel not a multiple of 4 (exercises padding).
    for shape in [(7, 5), (16, 32), (128, 128), (3, 3, 3)]:
        w = torch.randn(*shape) * 0.7
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        expected = (w * scale).round().clamp(-1, 1) / scale        # bitdistill.weight_quant_ternary
        codes, scale2 = weight_quant_ternary_codes(w)
        packed = pack_ternary(codes)
        got = unpack_ternary(packed, w.numel(), w.shape, scale2, dtype=torch.float32)
        check(f"roundtrip {shape}", torch.equal(got, expected))

    # 2. all-zero tensor: scale clamps to 1e5, must not NaN/inf.
    w0 = torch.zeros(10, 10)
    codes0, scale0 = weight_quant_ternary_codes(w0)
    got0 = unpack_ternary(pack_ternary(codes0), w0.numel(), w0.shape, scale0, dtype=torch.float32)
    check("all-zero no-NaN", torch.isfinite(got0).all().item() and torch.equal(got0, w0))

    # 3. PackedTernaryLinear.forward matches BitLinear.forward (quant_alpha=1) on
    #    random input -- requires bitdistill.py, so import lazily here only.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bitdistill as bd
    lin = torch.nn.Linear(24, 40, bias=False)
    bl = bd.BitLinear.from_linear(lin, sub_norm=True)
    bd.set_quant_alpha(bl, 1.0)
    ptl = PackedTernaryLinear.from_bitlinear(bl)
    x = torch.randn(3, 7, 24)
    with torch.no_grad():
        y_bl = bl(x)
        y_ptl = ptl(x)
    check("PackedTernaryLinear matches BitLinear forward",
         torch.allclose(y_bl, y_ptl, atol=1e-4, rtol=1e-3))

    # 4. round-trip through to_bitlinear() reproduces the same forward again.
    bl2 = ptl.to_bitlinear(dtype=torch.float32)
    bd.set_quant_alpha(bl2, 1.0)
    with torch.no_grad():
        y_bl2 = bl2(x)
    check("to_bitlinear() round-trip matches", torch.allclose(y_bl, y_bl2, atol=1e-4, rtol=1e-3))

    print(f"[self-test] {'ALL PASSED' if ok else 'FAILED'}", flush=True)
    return ok


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        raise SystemExit(0 if _self_test() else 1)
    print("Usage: python ternary_pack.py --self-test")
