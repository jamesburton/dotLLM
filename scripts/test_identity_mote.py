#!/usr/bin/env python3
"""
test_identity_mote.py — unit validation for scripts/lora/identity_mote.py.

Builds a TINY synthetic BitNet (2 layers, hidden=64), depth-expands it with
zero-residual identity blocks, converts each inserted block into an
IdentityMoTEBlock (skip expert + K capability experts + router), and proves the
CRITICAL invariant: the converted model's logits equal the base model's at init.

Covered:
  * zero-init identity holds under the default (skip-favoring) router
  * zero-init identity holds under a RANDOM router (bulletproof — every expert
    outputs 0 regardless of routing)
  * template-init identity holds when the router is biased to the skip expert
  * expert / router structure (K+1 experts, router shape, skip frozen)
  * forward returns (hidden, route_logits, expert_counts) with correct shapes

Run directly:
    CUDA_VISIBLE_DEVICES= python scripts/test_identity_mote.py
Or under pytest:
    pytest scripts/test_identity_mote.py -q
"""

from __future__ import annotations

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bitnet_depth_expand as bde
from identity_mote import (
    IdentityMoTEBlock,
    build_identity_mote,
    assert_identity_at_init,
)


def _base_and_ids():
    base = bde.build_tiny_bitnet()
    base.eval()
    torch.manual_seed(7)
    ids = torch.randint(0, base.config.vocab_size, (2, 12))
    return base, ids


def _expand_and_convert(base, K, capability_init, router_identity_bias):
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=1, at=None)
    model = copy.deepcopy(base)
    model, info = bde.expand_model(model, positions)
    model = build_identity_mote(
        model,
        inserted_indices=info["inserted_indices"],
        n_capability_experts=K,
        capability_init=capability_init,
        router_identity_bias=router_identity_bias,
    )
    return model, info


def test_inserted_indices_reported():
    base, _ = _base_and_ids()
    positions = bde.plan_insertions(base.config.num_hidden_layers, every=1, at=None)
    _, info = bde.expand_model(copy.deepcopy(base), positions)
    # 2 base layers, every=1 -> inserts after each -> new layers at indices 1 and 3
    assert info["inserted"] == 2
    assert info["inserted_indices"] == [1, 3], info["inserted_indices"]


def test_zero_init_identity_favoring_router():
    base, ids = _base_and_ids()
    model, _ = _expand_and_convert(base, K=3, capability_init="zero", router_identity_bias=0.0)
    r = assert_identity_at_init(base, model, ids, tol=1e-3, randomize_router=False)
    assert r["ok"], r
    assert r["max_abs"] == 0.0, r  # exact zero: all experts output 0


def test_zero_init_identity_random_router():
    """Bulletproof: zero-init identity must survive an arbitrary router."""
    base, ids = _base_and_ids()
    model, _ = _expand_and_convert(base, K=3, capability_init="zero", router_identity_bias=0.0)
    r = assert_identity_at_init(base, model, ids, tol=1e-3, randomize_router=True)
    assert r["ok"], r
    assert r["max_abs"] == 0.0, r


def test_template_init_identity_via_router_bias():
    """template warm-start: identity holds because router routes to the skip expert."""
    base, ids = _base_and_ids()
    model, _ = _expand_and_convert(
        base, K=2, capability_init="template", router_identity_bias=30.0
    )
    r = assert_identity_at_init(base, model, ids, tol=1e-3, randomize_router=False)
    assert r["ok"], r
    assert r["max_abs"] == 0.0, r  # skip expert output is exactly 0


def test_structure_and_forward_shapes():
    base, ids = _base_and_ids()
    K = 3
    model, info = _expand_and_convert(base, K=K, capability_init="zero", router_identity_bias=0.0)
    mote_layers = [l.mlp for l in model.model.layers if isinstance(l.mlp, IdentityMoTEBlock)]
    assert len(mote_layers) == info["inserted"], (len(mote_layers), info["inserted"])
    for block in mote_layers:
        assert block.n_experts == K + 1
        assert block.router.out_features == K + 1
        assert block.top_k == 1
        # skip expert frozen
        assert all(not p.requires_grad for p in block.experts[0].parameters())
        # capability experts trainable
        for e in range(1, K + 1):
            assert any(p.requires_grad for p in block.experts[e].parameters())
    # forward returns a plain tensor; logits/counts stashed as attributes
    x = torch.randn(2, 5, base.config.hidden_size)
    hidden = mote_layers[0](x)
    assert hidden.shape == (2, 5, base.config.hidden_size)
    assert mote_layers[0].last_logits.shape == (2 * 5, K + 1)
    assert mote_layers[0].last_counts.shape == (K + 1,)


def main() -> int:
    tests = [
        test_inserted_indices_reported,
        test_zero_init_identity_favoring_router,
        test_zero_init_identity_random_router,
        test_template_init_identity_via_router_bias,
        test_structure_and_forward_shapes,
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
            import traceback
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
