"""pytest — MoTE upcycle module init-identity gate.

Tests:
    test_mote_init_matches_dense  — routed MoE with identical clones outputs == dense FFN
    test_mote_shared_fp_equals_dense — shared fp expert output == dense FFN; is frozen
    test_mote_config_wiring       — module shapes / param counts / shared modes correct
"""
import copy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress Torch Inductor compilation errors on Windows (no MSVC cl.exe);
# dynamo falls back to eager execution automatically.
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import pytest
import torch.nn as nn
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
# Helper
# ---------------------------------------------------------------------------

def load_bitnet_bf16():
    """Helper matching the brief's test sketch — returns cached base model."""
    # In pytest context this always hits the module-scoped fixture above, but
    # the function form is kept so test code reads identically to the brief.
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="cpu"
    )


# ---------------------------------------------------------------------------
# Test 1: init-identity — routed-MoE output == dense FFN at init
# ---------------------------------------------------------------------------

def test_mote_init_matches_dense(fresh_layer):
    """Upcycled block with shared='none' and identical expert clones must produce
    exactly the same output as the original dense FFN (within bfloat16 tolerance).

    Why this discriminates: with N identical clones and top-k routing whose
    selected gates are renormalized to sum to 1, each token's output is a
    convex combination of identical expert outputs — which collapses to the
    single dense output regardless of which expert is chosen.
    """
    from mote_upcycle import build_mote

    base = fresh_layer
    # Capture reference to original dense mlp BEFORE build_mote replaces it.
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared="none")

    x = torch.randn(2, 8, base.config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        y_dense = dense_ffn(x)
        y_mote, aux, counts = base.model.layers[_LAYER_IDX].mlp(x)

    max_diff = (y_dense - y_mote).abs().max().item()
    assert torch.allclose(y_dense, y_mote, atol=1e-4), (
        f"MoTE output diverges from dense at init — max abs diff {max_diff:.6f}"
    )
    assert aux.item() >= 0.0, "aux loss should be non-negative"
    assert counts.shape == (4,), f"expected counts shape (4,), got {counts.shape}"
    # top_k=1 → each of the 2*8=16 tokens is dispatched to exactly one expert
    assert counts.sum().item() == 2 * 8, (
        f"expected total dispatch count 16, got {counts.sum().item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: shared fp expert output == dense; shared is frozen
# ---------------------------------------------------------------------------

def test_mote_shared_fp_equals_dense(fresh_layer):
    """With shared='fp', the shared expert is a frozen deep copy of the dense FFN
    and its output must match the dense FFN output within tolerance.
    """
    from mote_upcycle import build_mote

    base = fresh_layer
    dense_ffn = base.model.layers[_LAYER_IDX].mlp

    build_mote(base, layers=[_LAYER_IDX], n_experts=4, top_k=1, shared="fp")
    mote_block = base.model.layers[_LAYER_IDX].mlp

    x = torch.randn(2, 8, base.config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        y_dense = dense_ffn(x)
        y_shared = mote_block.shared(x)

    max_diff = (y_dense - y_shared).abs().max().item()
    assert torch.allclose(y_dense, y_shared, atol=1e-4), (
        f"shared fp expert output differs from dense — max abs diff {max_diff:.6f}"
    )
    for p in mote_block.shared.parameters():
        assert not p.requires_grad, "shared fp expert must be frozen (requires_grad=False)"


# ---------------------------------------------------------------------------
# Test 3: config wiring — shapes, counts, shared modes
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

        # --- Case C: shared="ternary" ---
        base.model.layers[LAYER].mlp = original_mlp_b
        build_mote(base, layers=[LAYER], n_experts=2, top_k=1, shared="ternary")
        block_tern = base.model.layers[LAYER].mlp

        assert block_tern.shared is not None, "shared='ternary' → shared should be present"
        has_trainable = any(p.requires_grad for p in block_tern.shared.parameters())
        assert has_trainable, "ternary shared expert must have trainable params"

    finally:
        base.model.layers[LAYER].mlp = original_mlp
