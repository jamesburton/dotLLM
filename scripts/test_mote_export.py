#!/usr/bin/env python3
"""
test_mote_export.py — unit validation for scripts/lora/mote_export.py.

Proves the identity-MoTE -> dotLLM export is COMPLETE and LOSSLESS on a tiny
synthetic BitNet (no downloads, no GPU):

  * a fresh identical-structure model rebuilt ONLY from the exported safetensors
    reproduces the reference model's logits exactly (round-trip parity);
  * the emitted config.json expresses the depth-expanded topology as a per-layer
    top-1 MoE via dotLLM's existing knobs (num_experts, num_experts_per_tok=1,
    decoder_sparse_step=1, mlp_only_layers=<original layers>);
  * the router->gate name remap (and its inverse) is exact.

Run directly:
    CUDA_VISIBLE_DEVICES= python scripts/test_mote_export.py
Or under pytest:
    pytest scripts/test_mote_export.py -q
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bitnet_depth_expand as bde
import mote_export as me


def _cfg(K=2, every=1):
    return {
        "n_capability_experts": K,
        "capability_init": "zero",
        "every": every,
        "inserted_indices": None,
        "label_map": {"math": 1, "instruction": 2},
        "layers_before": None,
    }


def _build_reference(K=2):
    torch.manual_seed(0)
    cfg = _cfg(K=K)
    base = bde.build_tiny_bitnet()
    cfg["layers_before"] = base.config.num_hidden_layers
    model, inserted = me.reconstruct_model(base, cfg, adapter_state=None)
    cfg["inserted_indices"] = inserted
    me._perturb_trained(model, inserted)
    return model, cfg, inserted


def test_name_remap_roundtrips():
    for hf in ("model.layers.3.mlp.router.weight",
               "model.layers.3.mlp.router.bias"):
        exp = me._export_name(hf)
        assert ".mlp.gate." in exp and ".router." not in exp, exp
        assert me._import_name(exp) == hf, (exp, me._import_name(exp))
    # non-router names pass through unchanged both ways.
    passthru = "model.layers.3.mlp.experts.1.ffn_sub_norm.weight"
    assert me._export_name(passthru) == passthru
    assert me._import_name(passthru) == passthru


def test_export_roundtrip_lossless():
    model, cfg, inserted = _build_reference(K=2)
    torch.manual_seed(11)
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    with tempfile.TemporaryDirectory() as d:
        me.export_to_dotllm(model, cfg, inserted, d)

        def _fresh():
            base = bde.build_tiny_bitnet()
            with torch.no_grad():
                for p in base.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            m, _ = me.reconstruct_model(base, cfg, adapter_state=None)
            return m

        r = me.verify_round_trip(model, d, _fresh, ids, tol=1e-3)
    assert r["ok"], r
    assert r["max_abs"] == 0.0, r  # native-dtype export -> exact reload


def test_config_expresses_per_layer_moe():
    model, cfg, inserted = _build_reference(K=3)
    with tempfile.TemporaryDirectory() as d:
        info = me.export_to_dotllm(model, cfg, inserted, d)
    c = info["config"]
    assert c["model_type"] == "bitnet"
    assert c["num_experts"] == 4          # K + 1 (skip + 3 capability)
    assert c["num_experts_per_tok"] == 1  # supervised top-1
    assert c["decoder_sparse_step"] == 1
    assert c["hidden_act"] == "relu2"
    # Only inserted layers are MoE; all originals are force-dense.
    for l in inserted:
        assert l not in c["mlp_only_layers"]
    for l in range(c["num_hidden_layers"]):
        if l not in inserted:
            assert l in c["mlp_only_layers"], l
    assert c["mote"]["skip_expert_index"] == 0
    assert c["mote"]["router_has_bias"] is True


def test_skip_expert_downproj_is_zero():
    """Expert 0 (skip) must ship an all-zero down_proj so it packs to I2_S zeros
    (outputs exactly 0 -> exact base/identity path). dotLLM needs no special-casing."""
    model, cfg, inserted = _build_reference(K=2)
    with tempfile.TemporaryDirectory() as d:
        from safetensors.torch import load_file
        me.export_to_dotllm(model, cfg, inserted, d)
        t = load_file(os.path.join(d, "model.safetensors"))
    for l in inserted:
        w = t[f"model.layers.{l}.mlp.experts.0.down_proj.weight"]
        assert torch.count_nonzero(w).item() == 0, f"skip expert down_proj nonzero at layer {l}"
        # capability expert down_proj IS non-zero (trained).
        w1 = t[f"model.layers.{l}.mlp.experts.1.down_proj.weight"]
        assert torch.count_nonzero(w1).item() > 0


def main() -> int:
    tests = [
        test_name_remap_roundtrips,
        test_export_roundtrip_lossless,
        test_config_expresses_per_layer_moe,
        test_skip_expert_downproj_is_zero,
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
