"""pytest — MoTE upcycle module tests.

Tests:
    test_mote_block_init_identity         — block(x) == dense_ffn(x) at init for ALL shared modes
    test_mote_shared_fp_is_true_bf16      — fp shared uses plain nn.Linear (no ternary quant)
    test_mote_shared_ternary_init_matches_dense — ternary shared matches dense at init
    test_mote_config_wiring               — module shapes / param counts / shared modes
"""
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytest
from transformers import AutoModelForCausalLM

BASE_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"
_LAYER_IDX = 20  # layer used for init-identity test (per brief)

# ---------------------------------------------------------------------------
# Module-level fixture — loads the heavy base model once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base():
    """Load the BitNet base model once for the entire test module."""
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
    )


@pytest.fixture()
def fresh_layer(base):
    """Yield base and restore layer _LAYER_IDX mlp after each test."""
    original_mlp = copy.deepcopy(base.model.layers[_LAYER_IDX].mlp)
    yield base
    base.model.layers[_LAYER_IDX].mlp = original_mlp


# ---------------------------------------------------------------------------
# Test 1: Block-level init identity — block(x) ≈ dense_ffn(x) for ALL shared modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shared_mode", ["none", "ternary", "fp"])
def test_mote_block_init_identity(fresh_layer, shared_mode):
    """Full MoTE block at init must be equivalent to dense_ffn(x) for all shared modes.

    With normalized routed gates (g_norm sums to 1) and a convex mix α=0.5:

      shared='none':    out = Σ g_norm_i · expert_i(x) = dense(x)  [exact: same ternary path]
      shared='ternary': 0.5·ternary_shared + 0.5·ternary_routed = dense(x)  [exact: identical clones]
      shared='fp':      out = 0.5·fp_shared(x) + 0.5·ternary_routed(x)

    For 'none' and 'ternary', the block output is EXACTLY dense(x) (all paths use the same
    AutoBitLinear clone, per-token activation quantization gives identical per-token results).

    For 'fp', the shared expert uses plain bf16 while routed experts use AutoBitLinear ternary
    quantization.  For random inputs the two paths differ substantially (act-quant error scales
    with input magnitude), so we verify the CONVEX COMBINE FORMULA directly rather than
    comparing to dense_ffn(x).  The formula test discriminates the old additive form
    (routed_out + shared) because that gives a different output for any non-trivial difference
    between the fp and ternary paths.

    Old 1.25× additive form for shared='fp':
      out_old = gate_i * expert_i(x) + shared(x)  ≈ 0.25·ternary + 1.0·fp
      → differs from 0.5·fp + 0.5·ternary by 0.5·(fp - ternary) → fails formula assert.
    Also checks: expert clones identical, aux≥0, dispatch counts consistent.
    """
    from mote_upcycle import build_mote

    base = fresh_layer
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared=shared_mode)
    mote_block = base.model.layers[_LAYER_IDX].mlp

    B, T = 2, 8
    H = base.config.hidden_size
    x = torch.randn(B, T, H, dtype=torch.bfloat16)
    x_flat = x.view(B * T, H)

    # --- All expert clones must produce identical output (same deepcopy) ---
    with torch.no_grad():
        expert_outs = [mote_block.experts[i](x_flat) for i in range(len(mote_block.experts))]
    for i in range(1, len(expert_outs)):
        assert torch.allclose(expert_outs[0], expert_outs[i], atol=1e-4), (
            f"expert[0] and expert[{i}] differ — should be identical deep copies"
        )

    with torch.no_grad():
        y_dense = dense_ffn(x)              # [B, T, H]
        y_mote, aux, counts = mote_block(x) # [B, T, H]

    if shared_mode in ("none", "ternary"):
        # Exact identity: all paths use the same AutoBitLinear clone → per-token quant
        # gives the same output regardless of routing scatter/gather order.
        # none:    g_norm sums to 1 → routed_out = dense exactly.
        # ternary: 0.5*ternary_shared(x_flat) + 0.5*ternary_routed = dense exactly.
        max_diff = (y_mote - y_dense).abs().max().item()
        assert torch.allclose(y_mote, y_dense, atol=1e-4), (
            f"MoTE block init-identity FAILED for shared={shared_mode!r} — "
            f"max abs diff {max_diff:.6f}  (expected exact match: same AutoBitLinear clone)"
        )
    else:
        # shared='fp': verify the convex combine formula alpha*fp + (1-alpha)*ternary.
        # For per-token act quant all expert clones produce the same per-token output, so:
        #   routed_out[i] = experts[0](x_flat)[i]  for any i (all clones identical)
        # → y_mote.view(-1, H) == alpha * shared(x_flat) + (1-alpha) * experts[0](x_flat) EXACTLY
        # (differences only from bf16 summation order, << 1e-3).
        # Old additive form gives 'routed_out + shared(x_flat)', which differs by
        # (0.5 - 1.0)*fp + (0.5 - gate_i)*ternary — large for any non-trivial fp-ternary diff.
        with torch.no_grad():
            fp_out = mote_block.shared(x_flat)           # [n_tokens, H]
            ternary_out = mote_block.experts[0](x_flat)  # [n_tokens, H]
        alpha = mote_block.mix_alpha  # 0.5
        expected_flat = alpha * fp_out + (1.0 - alpha) * ternary_out  # [n_tokens, H]
        expected = expected_flat.view(B, T, H)

        max_diff = (y_mote - expected).abs().max().item()
        assert torch.allclose(y_mote, expected, atol=1e-3), (
            f"MoTE fp combine formula broken — max diff {max_diff:.6f} "
            f"(expected 0.5*fp_shared + 0.5*ternary_routed; "
            f"old additive form 'routed + shared' would give a different result)"
        )

    # --- Aux loss and dispatch counts ---
    assert aux.item() >= 0.0, "aux loss should be non-negative"
    n_exp = len(mote_block.experts)
    assert counts.shape == (n_exp,), f"expected counts shape ({n_exp},), got {counts.shape}"
    assert counts.sum().item() == B * T, (
        f"expected total dispatch count {B * T}, got {counts.sum().item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: shared fp expert uses true bf16 (no ternary quant)
# ---------------------------------------------------------------------------

def test_mote_shared_fp_is_true_bf16(fresh_layer):
    """shared='fp' must compute a true bf16 forward: no ternary weight quant, no act quant.

    Verification:
    1. All linear submodules in fp_shared are plain nn.Linear (not AutoBitLinear).
    2. fp_shared(x) matches a reference plain-bf16 matmul built from the same weights.
    3. fp_shared is fully frozen (requires_grad=False for all params).

    If the bf16 master weights happen to be numerically near-ternary (e.g. after
    BitNet training), the code-path check (assertion 1) remains the discriminating
    factor regardless of numeric closeness to the ternary forward.
    """
    from mote_upcycle import build_mote
    from transformers.integrations.bitnet import AutoBitLinear

    base = fresh_layer
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    # Capture bf16 master weights BEFORE build_mote (deep-copied into shared expert)
    gate_w = dense_ffn.gate_proj.weight.data.clone()
    up_w = dense_ffn.up_proj.weight.data.clone()
    down_w = dense_ffn.down_proj.weight.data.clone()

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared="fp")
    mote_block = base.model.layers[_LAYER_IDX].mlp

    B, T = 2, 8
    x = torch.randn(B, T, base.config.hidden_size, dtype=torch.bfloat16)
    x_flat = x.view(B * T, base.config.hidden_size)

    with torch.no_grad():
        # fp_shared is called with x_flat (2D) in MoTEBlock.forward
        y_fp = mote_block.shared(x_flat)

        # Reference: plain bf16 matmul — same weights, same ffn_sub_norm, no quant
        gate_ref = F.linear(x_flat, gate_w)
        up_ref = F.linear(x_flat, up_w)
        hidden_ref = dense_ffn.act_fn(gate_ref) * up_ref
        hidden_ref = dense_ffn.ffn_sub_norm(hidden_ref)
        y_ref = F.linear(hidden_ref, down_w)

    # 1. Code-path check: fp shared must use plain nn.Linear, not AutoBitLinear
    for mod_name, mod in mote_block.shared.named_modules():
        assert not isinstance(mod, AutoBitLinear), (
            f"fp shared expert still contains AutoBitLinear at '{mod_name}'; "
            "shared='fp' must use plain nn.Linear (true bf16 forward, no WeightQuant/ActQuant)"
        )

    # 2. Numeric match: fp output == reference plain-bf16 matmul
    max_diff = (y_fp - y_ref).abs().max().item()
    assert torch.allclose(y_fp, y_ref, atol=1e-4), (
        f"fp shared expert output diverges from reference bf16 matmul — max diff {max_diff:.6f}"
    )

    # 3. Frozen: no trainable params
    for p in mote_block.shared.parameters():
        assert not p.requires_grad, "fp shared expert must be frozen (requires_grad=False)"


# ---------------------------------------------------------------------------
# Test 3: shared ternary expert output == dense at init
# ---------------------------------------------------------------------------

def test_mote_shared_ternary_init_matches_dense(fresh_layer):
    """shared='ternary' expert is an identical deep copy at init → output == dense FFN.

    The ternary shared expert runs through AutoBitLinear (WeightQuant + ActQuant),
    identical to the dense forward, so y_shared == y_dense at init.
    After heal-training, they will diverge — this test only covers init state.
    """
    from mote_upcycle import build_mote

    base = fresh_layer
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared="ternary")
    mote_block = base.model.layers[_LAYER_IDX].mlp

    x = torch.randn(2, 8, base.config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        y_dense = dense_ffn(x)
        y_shared = mote_block.shared(x)

    max_diff = (y_dense - y_shared).abs().max().item()
    assert torch.allclose(y_dense, y_shared, atol=1e-4), (
        f"shared ternary expert diverges from dense FFN at init — max diff {max_diff:.6f}"
    )

    # ternary shared expert must have trainable parameters
    has_trainable = any(p.requires_grad for p in mote_block.shared.parameters())
    assert has_trainable, "ternary shared expert must have trainable parameters"


# ---------------------------------------------------------------------------
# Test 4: config wiring — shapes, counts, shared modes
# ---------------------------------------------------------------------------

def test_mote_config_wiring(base):
    """Verify that n_experts / top_k / shared produce the correct module structure
    and parameter counts.  Uses layer 0 to avoid clobbering the _LAYER_IDX used
    by the other tests (and is restored manually inside this test).
    """
    from mote_upcycle import build_mote, MoTEBlock

    LAYER = 0
    original_mlp = copy.deepcopy(base.model.layers[LAYER].mlp)
    hidden = base.config.hidden_size

    try:
        # --- Case A: shared="none", n_experts=4, top_k=2 ---
        build_mote(base, layers=[LAYER], n_experts=4, top_k=2, shared="none")
        block = base.model.layers[LAYER].mlp

        assert isinstance(block, MoTEBlock), "mlp should be replaced with MoTEBlock"
        assert len(block.experts) == 4, "should have 4 routed experts"
        assert block.top_k == 2
        assert block.shared is None, "shared='none' → shared attribute should be None"

        # Router shape: (n_experts, hidden_size) — Linear weight is [out, in]
        assert block.router.weight.shape == (4, hidden), (
            f"router weight shape should be (4, {hidden}), got {block.router.weight.shape}"
        )
        assert block.router.weight.dtype == torch.bfloat16, "router must be bfloat16"

        # Experts are BitNetMLP subclasses with AutoBitLinear layers (ternary in forward)
        from transformers.integrations.bitnet import AutoBitLinear
        for i, exp in enumerate(block.experts):
            for name, mod in exp.named_modules():
                if isinstance(mod, nn.Linear):
                    assert isinstance(mod, AutoBitLinear), (
                        f"expert[{i}].{name} should be AutoBitLinear (ternary), got {type(mod).__name__}"
                    )

        # --- Case B: shared="fp" ---
        base.model.layers[LAYER].mlp = original_mlp
        original_mlp_b = copy.deepcopy(base.model.layers[LAYER].mlp)
        build_mote(base, layers=[LAYER], n_experts=2, top_k=1, shared="fp")
        block_fp = base.model.layers[LAYER].mlp

        assert block_fp.shared is not None, "shared='fp' → shared should be present"
        for p in block_fp.shared.parameters():
            assert not p.requires_grad, "fp shared expert must be frozen"
        # fp shared uses plain nn.Linear (not AutoBitLinear)
        for mod_name, mod in block_fp.shared.named_modules():
            assert not isinstance(mod, AutoBitLinear), (
                f"fp shared expert should use plain nn.Linear, found AutoBitLinear at '{mod_name}'"
            )

        # --- Case C: shared="ternary" ---
        base.model.layers[LAYER].mlp = original_mlp_b
        build_mote(base, layers=[LAYER], n_experts=2, top_k=1, shared="ternary")
        block_tern = base.model.layers[LAYER].mlp

        assert block_tern.shared is not None, "shared='ternary' → shared should be present"
        has_trainable = any(p.requires_grad for p in block_tern.shared.parameters())
        assert has_trainable, "ternary shared expert must have trainable params"

    finally:
        base.model.layers[LAYER].mlp = original_mlp
