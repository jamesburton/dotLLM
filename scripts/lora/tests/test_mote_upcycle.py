"""pytest — MoTE upcycle module tests.

Tests:
    test_mote_init_matches_dense          — structural ratio: y_mote == gate_sum * dense(x)
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
# Test 1: structural ratio — y_mote == gate_sum * dense(x) per token
# ---------------------------------------------------------------------------

def test_mote_init_matches_dense(fresh_layer):
    """Upcycled block with shared='none' and identical expert clones must satisfy
    the structural ratio y_mote ≈ gate_sum * dense(x) per token.

    With N identical clones and NON-normalized gates:
        out[i] = Σ_{k∈selected} gate_k * expert_k(x_i)
                 = gate_sum_i * dense(x_i)    (all clones are identical)

    where gate_sum_i = sum of the selected top-k raw softmax gates for token i.

    This discriminates:
      - Wrong clones / quant errors → expert(x) ≠ dense(x) → fails
      - Bad routing → wrong gate_sum computed → fails
      - Gate renormalization present → gate_sum becomes 1 for top_k=1, but the
        expected value uses raw softmax (≠ 1), so y_mote ≠ expected → fails
      - aux-loss wiring errors → checked via aux ≥ 0 and counts.sum == B*T
    """
    from mote_upcycle import build_mote

    base = fresh_layer
    # Capture reference to original dense mlp BEFORE build_mote replaces it.
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared="none")
    mote_block = base.model.layers[_LAYER_IDX].mlp

    B, T = 2, 8
    x = torch.randn(B, T, base.config.hidden_size, dtype=torch.bfloat16)
    x_flat = x.view(B * T, base.config.hidden_size)

    # --- Assert: all expert clones produce identical output on the same input ---
    with torch.no_grad():
        expert_outs = [mote_block.experts[i](x_flat) for i in range(len(mote_block.experts))]
    for i in range(1, len(expert_outs)):
        assert torch.allclose(expert_outs[0], expert_outs[i], atol=1e-4), (
            f"expert[0] and expert[{i}] produce different outputs — "
            "they should be identical deep copies of the original dense FFN"
        )

    # --- Compute gate_sum (sum of selected top-k raw softmax gates per token) ---
    with torch.no_grad():
        g = torch.softmax(
            mote_block.router(x_flat.to(mote_block.router.weight.dtype)), dim=-1
        )  # [n_tokens, n_experts]
        top_g, _ = torch.topk(g, mote_block.top_k, dim=-1)  # [n_tokens, top_k]
        gate_sum = top_g.sum(dim=-1)  # [n_tokens]

    # --- Run MoTE and dense forward ---
    with torch.no_grad():
        y_dense = dense_ffn(x)                         # [B, T, H]
        y_mote, aux, counts = mote_block(x)            # [B, T, H]

    # --- Structural ratio: y_mote == gate_sum * y_dense ---
    gate_sum_bcast = gate_sum.view(B, T, 1).to(y_dense.dtype)  # [B, T, 1]
    expected = gate_sum_bcast * y_dense                          # [B, T, H]

    max_diff = (y_mote - expected).abs().max().item()
    assert torch.allclose(y_mote, expected, atol=1e-4), (
        f"MoTE structural ratio broken — max abs diff {max_diff:.6f} "
        f"(expected y_mote ≈ gate_sum * dense_out; check renorm was removed)"
    )
    assert aux.item() >= 0.0, "aux loss should be non-negative"
    assert counts.shape == (4,), f"expected counts shape (4,), got {counts.shape}"
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
