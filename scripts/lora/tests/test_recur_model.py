"""Sanity tests for recur_model.RecurModel — Task R2.

Verifiable properties (all run on CPU, no GPU, no background jobs):

  (a) Pass-through config (_bypass_fusion=True, bypassing fusion+gate) → recurrence=1
      logits match the stock BitNet model within tight numerical tolerance.

  (b) build_recur(...).forward(ids, recurrence=k) for k in {1, 2} returns
      finite tensors with the correct shape (batch, seq, vocab).

  (c) gate.g and fusion adapter weights/bias are trainable nn.Parameters,
      with g initialised ≈ 0.5.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface pytest scripts/lora/tests/test_recur_model.py -v
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytest

from transformers import AutoTokenizer, BitNetForCausalLM
from recur_model import (
    RecurModel,
    ResidualGate,
    build_recur,
    make_passthrough,
    DEFAULT_P,
    DEFAULT_Q,
)

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROMPT = "The capital of France is Paris, and the capital of Germany is"

# -----------------------------------------------------------------------
# Shared fixture — load model once per test session
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_model():
    """Load BitNetForCausalLM on CPU in float32 (bf16 not reliable on CPU)."""
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = BitNetForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return tok, model


@pytest.fixture(scope="module")
def input_ids(base_model):
    tok, _ = base_model
    return tok(PROMPT, return_tensors="pt").input_ids  # (1, seq)


# -----------------------------------------------------------------------
# (a) Pass-through N=1 == stock model
# -----------------------------------------------------------------------

def test_passthrough_n1_matches_stock(base_model, input_ids):
    """Pass-through config → N=1 forward reproduces stock BitNet logits.

    This is the Probe-0 N=1==stock check. make_passthrough() sets
    _bypass_fusion=True, bypassing both the fusion linear and the gate
    arithmetic: fused=prelude_out (exact slice), state=slab_out (direct
    assign). No identity-weight init or g=1 assignment is needed.
    """
    _, model = base_model
    recur = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q)
    make_passthrough(recur)
    recur.eval()

    with torch.no_grad():
        stock_logits = model(input_ids).logits
        recur_logits = recur(input_ids, recurrence=1)

    max_abs = (stock_logits.float() - recur_logits.float()).abs().max().item()
    argmax_match = bool(
        (stock_logits[0, -1].argmax() == recur_logits[0, -1].argmax()).item()
    )

    # Tight tolerance: identity linear + g=1 gate should give near-zero diff
    assert max_abs < 1e-2, (
        f"Pass-through N=1 max_abs_diff={max_abs:.4e} (expected <1e-2); "
        "check fusion weight init in make_passthrough()"
    )
    assert argmax_match, (
        "Pass-through N=1 argmax mismatch vs stock model — forward logic error"
    )


# -----------------------------------------------------------------------
# (b) Finite correct-shape logits for recurrence in {1, 2}
# -----------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_forward_finite_correct_shape(base_model, input_ids, k):
    """forward(ids, recurrence=k) returns finite logits of shape (B, S, vocab)."""
    _, model = base_model
    recur = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q)
    recur.eval()

    with torch.no_grad():
        logits = recur(input_ids, recurrence=k)

    expected_shape = (input_ids.shape[0], input_ids.shape[1], model.config.vocab_size)
    assert logits.shape == expected_shape, (
        f"recurrence={k}: shape {logits.shape} != expected {expected_shape}"
    )
    assert torch.isfinite(logits).all(), (
        f"recurrence={k}: logits contain non-finite values (nan/inf)"
    )


# -----------------------------------------------------------------------
# (c) Trainable parameters — gate.g and fusion adapter
# -----------------------------------------------------------------------

def test_gate_g_is_trainable_parameter(base_model):
    """gate.g is an nn.Parameter requiring grad, initialised ≈ 0.5."""
    _, model = base_model
    recur = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q, gate_init=0.5)

    assert isinstance(recur.gate.g, nn.Parameter), \
        "gate.g must be nn.Parameter"
    assert recur.gate.g.requires_grad, \
        "gate.g must require grad"
    assert abs(recur.gate.g.item() - 0.5) < 1e-6, \
        f"gate.g init should be ≈0.5, got {recur.gate.g.item()}"


def test_fusion_adapter_is_trainable(base_model):
    """fusion adapter weight and bias are trainable nn.Parameters."""
    _, model = base_model
    recur = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q)

    assert isinstance(recur.fusion, nn.Linear), \
        "fusion must be nn.Linear"
    assert recur.fusion.weight.requires_grad, \
        "fusion.weight must require grad"
    assert recur.fusion.bias is not None and recur.fusion.bias.requires_grad, \
        "fusion.bias must exist and require grad"


def test_gate_and_fusion_in_named_parameters(base_model):
    """gate.g and fusion appear in RecurModel.named_parameters()."""
    _, model = base_model
    recur = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q)

    param_names = {n for n, _ in recur.named_parameters()}
    assert "gate.g" in param_names, \
        f"gate.g missing from named_parameters; found: {sorted(param_names)[:10]}"
    assert "fusion.weight" in param_names, \
        f"fusion.weight missing from named_parameters"
    assert "fusion.bias" in param_names, \
        f"fusion.bias missing from named_parameters"


# -----------------------------------------------------------------------
# ResidualGate unit test (no model load required)
# -----------------------------------------------------------------------

def test_residual_gate_pass_through_at_g1():
    """ResidualGate with g=1.0 returns slab_out unchanged."""
    gate = ResidualGate(init=1.0)
    h_in = torch.randn(1, 4, 8)
    slab_out = torch.randn(1, 4, 8)
    out = gate(h_in, slab_out)
    assert torch.allclose(out, slab_out, atol=1e-6), \
        "g=1.0: gate should return slab_out exactly"


def test_residual_gate_skip_at_g0():
    """ResidualGate with g=0.0 returns h_in unchanged."""
    gate = ResidualGate(init=0.0)
    h_in = torch.randn(1, 4, 8)
    slab_out = torch.randn(1, 4, 8)
    out = gate(h_in, slab_out)
    assert torch.allclose(out, h_in, atol=1e-6), \
        "g=0.0: gate should return h_in exactly"


def test_residual_gate_midpoint_at_g05():
    """ResidualGate with g=0.5 returns the midpoint of h_in and slab_out."""
    gate = ResidualGate(init=0.5)
    h_in = torch.zeros(1, 4, 8)
    slab_out = torch.ones(1, 4, 8) * 2.0
    out = gate(h_in, slab_out)
    expected = torch.ones(1, 4, 8)  # 0 + 0.5*(2-0) = 1
    assert torch.allclose(out, expected, atol=1e-6), \
        "g=0.5: gate should return midpoint"
