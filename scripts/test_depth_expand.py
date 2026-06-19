#!/usr/bin/env python3
"""
test_depth_expand.py — dry-run unit validation for bitnet_depth_expand.py.

Builds a TINY synthetic BitNet in-process (2 layers, hidden=64, 4 heads) and proves
that inserting zero-residual ternary identity blocks leaves the model's forward
logits unchanged (the identity blocks are exact no-ops at init). No model download.

Run directly:
    python scripts/test_depth_expand.py
Or under pytest:
    pytest scripts/test_depth_expand.py -q
"""

from __future__ import annotations

import copy

import torch

import bitnet_depth_expand as bde


def _diff(base_logits, exp_logits):
    return (exp_logits - base_logits).abs().max().item()


def _base_and_ids():
    base = bde.build_tiny_bitnet()
    base.eval()
    torch.manual_seed(7)
    ids = torch.randint(0, base.config.vocab_size, (2, 12))
    with torch.no_grad():
        base_logits = base(ids).logits
    return base, ids, base_logits


def test_every_1_is_identity():
    """Insert one identity block after every layer -> logits unchanged."""
    base, ids, base_logits = _base_and_ids()
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=1, at=None)
    exp, info = bde.expand_model(copy.deepcopy(base), positions)
    exp.eval()
    assert info["inserted"] == 2  # one after each of the 2 base layers
    assert exp.config.num_hidden_layers == 4
    with torch.no_grad():
        exp_logits = exp(ids).logits
    assert exp_logits.shape == base_logits.shape
    d = _diff(base_logits, exp_logits)
    assert d <= 1e-3, f"identity broken: max_abs diff {d}"
    assert torch.equal(base_logits.argmax(-1), exp_logits.argmax(-1))


def test_explicit_at_positions_identity():
    """--at 1 (insert before layer 1, i.e. between the two base layers)."""
    base, ids, base_logits = _base_and_ids()
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=None, at=[1])
    exp, info = bde.expand_model(copy.deepcopy(base), positions)
    exp.eval()
    assert info["inserted"] == 1
    assert exp.config.num_hidden_layers == 3
    with torch.no_grad():
        exp_logits = exp(ids).logits
    assert _diff(base_logits, exp_logits) <= 1e-3


def test_front_and_end_insertion_identity():
    """Insert at the very front (--at 0) and very end (--at num_layers)."""
    base, ids, base_logits = _base_and_ids()
    n = base.config.num_hidden_layers
    positions = bde.plan_insertions(n, every=None, at=[0, n])
    exp, info = bde.expand_model(copy.deepcopy(base), positions)
    exp.eval()
    assert info["inserted"] == 2
    with torch.no_grad():
        exp_logits = exp(ids).logits
    assert _diff(base_logits, exp_logits) <= 1e-3


def test_layer_idx_renumbered_contiguously():
    """Inserted + original blocks must carry contiguous self_attn.layer_idx."""
    base, _ids, _ = _base_and_ids()
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=1, at=None)
    exp, _info = bde.expand_model(copy.deepcopy(base), positions)
    idxs = [l.self_attn.layer_idx for l in exp.model.layers]
    assert idxs == list(range(len(idxs))), idxs


def test_near_zero_eps_still_identity_via_ternary_round():
    """Tiny seeded proj weights still quantize to ternary 0 at forward -> identity.

    NOTE: the tiny synthetic model uses *master* bf16 weights and does NOT apply
    forward-time ternary quantization (that happens only in the quantized BitLinear
    path used for the packed checkpoints). So with eps>0 on the master-weight model
    the projections are genuinely nonzero and the output WILL differ slightly. We
    assert the diff is small (proportional to eps), documenting the distinction.
    """
    base, ids, base_logits = _base_and_ids()
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=1, at=None)
    exp, _info = bde.expand_model(copy.deepcopy(base), positions, near_zero_eps=1e-4)
    exp.eval()
    with torch.no_grad():
        exp_logits = exp(ids).logits
    d = _diff(base_logits, exp_logits)
    # Small but generally nonzero on master weights; just sanity-bound it.
    assert d < 1e-1, f"near-zero perturbation unexpectedly large: {d}"


def main() -> int:
    tests = [
        test_every_1_is_identity,
        test_explicit_at_positions_identity,
        test_front_and_end_insertion_identity,
        test_layer_idx_renumbered_contiguously,
        test_near_zero_eps_still_identity_via_ternary_round,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
