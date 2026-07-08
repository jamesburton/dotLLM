"""mote_export.py — export a trained identity-MoTE BitNet to a dotLLM-loadable artifact.

Campaign: trackM-mote — identity-expert routed-MoTE. Bridge to dotLLM (C#/.NET),
the core inference platform, so the trained model can be EVALUATED/SERVED there
instead of PyTorch (mote_accuracy_eval.py etc.).

What this produces (the "dotLLM MoTE format")
----------------------------------------------
A self-contained HuggingFace-style checkpoint directory:

    <out>/config.json          — BitNet config with per-layer MoE fields (see below)
    <out>/model.safetensors    — ALL weights of the depth-expanded + MoTE-converted
                                 model (bf16 master weights; dotLLM ternary-quantizes
                                 at load exactly like transformers' WeightQuant).

Why a FULL checkpoint (not just the adapter):
    The trained ``adapter_weights.pt`` contains ONLY the trainable tensors (routers +
    capability experts). The depth-inserted layers' *non-trainable* parts — the cloned
    attention (o_proj zeroed), the two layernorms, the attn/ffn sub-norms, and the
    frozen skip expert (down_proj zeroed) — live in NEITHER the base checkpoint NOR the
    adapter; they are reconstructed deterministically by
    ``bitnet_depth_expand.expand_model`` + ``identity_mote.build_identity_mote``. So the
    export RECONSTRUCTS the exact training-time module (base -> depth-expand -> convert
    -> load adapter) and serialises the whole thing. The result needs no base checkpoint
    at inference time.

dotLLM-side representation (see .planning/2026-07-08-mote-dotllm-export-design.md)
---------------------------------------------------------------------------------
* Architecture stays ``bitnet`` (dotLLM's BitNet I2_S ternary loader is reused for
  every linear: q/k/v/o, gate/up/down; absmean per-tensor scale matches transformers).
* The inserted layers become **routed top-1 MoE layers** expressed with dotLLM's
  EXISTING per-layer MoE config knobs:
      num_experts        = K + 1           (skip expert 0 + K capability experts)
      num_experts_per_tok = 1              (supervised top-1)
      decoder_sparse_step = 1              (candidate: every layer ...)
      mlp_only_layers     = <all ORIGINAL layer indices>   (... but force-dense the
                                                            originals, so ONLY inserted
                                                            layers are MoE)
  dotLLM's ``MoeConfig.IsMoeLayer(l)`` then returns true exactly for the inserted
  layers — no new config surface required.
* Tensor naming reuses dotLLM's MoE convention, with three BitNet/MoTE-specific
  additions that dotLLM must learn to read (documented as the C#-side build items):
      model.layers.{l}.mlp.gate.weight            (router weight, [E, hidden])   — reused
      model.layers.{l}.mlp.gate.bias              (router bias, [E])             — NEW
      model.layers.{l}.mlp.experts.{e}.gate_proj.weight  \
      model.layers.{l}.mlp.experts.{e}.up_proj.weight     > standard MoE expert   — reused
      model.layers.{l}.mlp.experts.{e}.down_proj.weight  /
      model.layers.{l}.mlp.experts.{e}.ffn_sub_norm.weight (BitNet FFN sub-norm) — NEW
  The skip expert (index 0) is a plain expert whose ``down_proj`` is all-zero; under
  I2_S that packs to all-zero trits with a clamped 1e-5 scale and outputs exactly 0 —
  so the identity/skip semantics need NO special-casing on the dotLLM side. The router
  bias only shifts the top-1 argmax (with norm_topk_prob the top-1 gate weight is 1.0),
  so it must be carried for correct expert SELECTION.

Numerical-parity note
---------------------
Capability experts are BitNet FFNs: ``down_proj( ffn_sub_norm( relu2(gate(x)) * up(x) ) )``
with per-BitLinear activation+weight quant. dotLLM's dense BitNet path already matches
this; the dotLLM MoE expert path currently assumes SwiGLU (silu, no sub-norm), so the
BitNet-MoE forward (relu2 + per-expert ffn_sub_norm + I2_S experts + router bias) is the
one dotLLM-side kernel/loader that must be built for end-to-end logit parity.

Self-test (CPU, seconds, no downloads)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 \
      python scripts/lora/mote_export.py --self-test

Real export (after a GPU training run)::

    python scripts/lora/mote_export.py \
        --base microsoft/bitnet-b1.58-2B-4T-bf16 \
        --adapter .docs/mote/idm_2b_v1 \
        --out    .docs/mote/idm_2b_v1_dotllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bitnet_depth_expand as bde  # noqa: E402
from identity_mote import IdentityMoTEBlock, build_identity_mote  # noqa: E402


# ---------------------------------------------------------------------------
# Reconstruction: base -> depth-expand -> convert -> load adapter
# ---------------------------------------------------------------------------


def reconstruct_model(base_model, cfg: dict, adapter_state: Optional[dict]):
    """Rebuild the exact training-time IdentityMoTE module and load the adapter.

    ``cfg`` is the ``identity_mote_config.json`` dict written by the trainer. Uses
    ``inserted_indices`` when present (authoritative), else re-plans from ``every``/``at``.
    """
    # Re-plan the SAME insertion positions the trainer used.
    every = cfg.get("every")
    at = cfg.get("at")
    if at is not None and isinstance(at, str):
        at = [int(x) for x in at.split(",") if x.strip()]
    if every is None and at is None:
        # Fall back: derive positions from the recorded inserted count if needed.
        every = 1
    n_layers_before = cfg.get("layers_before", base_model.config.num_hidden_layers)
    positions = bde.plan_insertions(n_layers_before, every=every, at=at)
    model, info = bde.expand_model(base_model, positions)

    inserted_indices = cfg.get("inserted_indices") or info["inserted_indices"]
    if inserted_indices != info["inserted_indices"]:
        raise ValueError(
            f"inserted_indices mismatch: config={inserted_indices} vs "
            f"reconstructed={info['inserted_indices']}. Base layer count / every/at "
            "must match the training run."
        )

    K = cfg["n_capability_experts"]
    model = build_identity_mote(
        model,
        inserted_indices=inserted_indices,
        n_capability_experts=K,
        capability_init=cfg.get("capability_init", "zero"),
        router_identity_bias=cfg.get("router_identity_bias", 0.0),
        top_k=1,
    )
    if adapter_state is not None:
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
        # The adapter is trainable-only; missing keys (base/skip/attn) are EXPECTED.
        if unexpected:
            raise ValueError(f"adapter has unexpected keys not in model: {unexpected[:8]}")
    model.eval()
    return model, inserted_indices


# ---------------------------------------------------------------------------
# Name mapping: HF module state_dict name -> dotLLM export name
# ---------------------------------------------------------------------------


def _export_name(hf_name: str) -> str:
    """Map an HF module parameter name to the dotLLM MoTE export tensor name.

    Only the router is renamed (``mlp.router.{weight,bias}`` -> ``mlp.gate.{weight,bias}``)
    to match dotLLM's MoE router convention; every other tensor keeps its HF name
    (which dotLLM's BitNet loader already expects).
    """
    if hf_name.endswith(".mlp.router.weight"):
        return hf_name[: -len(".mlp.router.weight")] + ".mlp.gate.weight"
    if hf_name.endswith(".mlp.router.bias"):
        return hf_name[: -len(".mlp.router.bias")] + ".mlp.gate.bias"
    return hf_name


def _import_name(export_name: str) -> str:
    """Inverse of :func:`_export_name` (dotLLM name -> HF module name), for round-trip."""
    if export_name.endswith(".mlp.gate.weight"):
        return export_name[: -len(".mlp.gate.weight")] + ".mlp.router.weight"
    if export_name.endswith(".mlp.gate.bias"):
        return export_name[: -len(".mlp.gate.bias")] + ".mlp.router.bias"
    return export_name


# ---------------------------------------------------------------------------
# config.json builder
# ---------------------------------------------------------------------------


def build_dotllm_config(model, cfg: dict, inserted_indices: list[int]) -> dict:
    """Build the dotLLM config.json for the exported MoTE checkpoint."""
    mc = model.config
    K = cfg["n_capability_experts"]
    n_experts = K + 1
    final_layers = mc.num_hidden_layers
    inserted_set = set(inserted_indices)
    # Force-dense every ORIGINAL (non-inserted) layer; only inserted layers are MoE.
    mlp_only_layers = [l for l in range(final_layers) if l not in inserted_set]

    head_dim = getattr(mc, "head_dim", None) or (mc.hidden_size // mc.num_attention_heads)
    rope_theta = 10000.0
    rp = getattr(mc, "rope_parameters", None) or getattr(mc, "rope_scaling", None)
    if isinstance(rp, dict) and rp.get("rope_theta"):
        rope_theta = float(rp["rope_theta"])
    rope_theta = float(getattr(mc, "rope_theta", rope_theta))

    return {
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": mc.hidden_size,
        "intermediate_size": mc.intermediate_size,
        "num_hidden_layers": final_layers,
        "num_attention_heads": mc.num_attention_heads,
        "num_key_value_heads": getattr(mc, "num_key_value_heads", mc.num_attention_heads),
        "head_dim": head_dim,
        "vocab_size": mc.vocab_size,
        "max_position_embeddings": getattr(mc, "max_position_embeddings", 4096),
        "rms_norm_eps": getattr(mc, "rms_norm_eps", 1e-5),
        "rope_theta": rope_theta,
        "hidden_act": "relu2",
        "tie_word_embeddings": bool(getattr(mc, "tie_word_embeddings", False)),
        # ── per-layer routed MoE (dotLLM MoeConfig knobs) ──
        "num_experts": n_experts,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": mc.intermediate_size,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": mlp_only_layers,
        # ── MoTE provenance (extension; ignored by generic loaders) ──
        "mote": {
            "inserted_indices": list(inserted_indices),
            "n_capability_experts": K,
            "n_experts": n_experts,
            "skip_expert_index": 0,
            "capability_init": cfg.get("capability_init", "zero"),
            "label_map": cfg.get("label_map", {}),
            "router_has_bias": True,
            "expert_activation": "relu2",
            "expert_has_ffn_sub_norm": True,
        },
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_to_dotllm(model, cfg: dict, inserted_indices: list[int], out_dir: str) -> dict:
    """Serialise the reconstructed model to <out_dir> in the dotLLM MoTE format."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)

    # Collect every parameter under its dotLLM export name. Master weights are kept
    # in their NATIVE dtype (bf16 for the real BitNet checkpoint — so the export is
    # lossless; f32 for the tiny self-test). dotLLM's BitNet loader upcasts to f32
    # then ternary-quantizes regardless, so any of bf16/f16/f32 source is accepted.
    tensors: dict[str, torch.Tensor] = {}
    for name, p in model.state_dict().items():
        export = _export_name(name)
        t = p.detach().contiguous().cpu()
        tensors[export] = t

    save_file(tensors, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt", "producer": "mote_export.py"})

    config = build_dotllm_config(model, cfg, inserted_indices)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    return {"n_tensors": len(tensors), "config": config}


# ---------------------------------------------------------------------------
# Round-trip verification (in-Python parity — proves the export is lossless/complete)
# ---------------------------------------------------------------------------


def verify_round_trip(reference_model, out_dir: str, fresh_model_builder, input_ids,
                      tol: float = 1e-3) -> dict:
    """Reload the exported safetensors into a FRESH identical-structure model and
    assert its logits match the reference model's within ``tol``.

    This exercises the exact tensor set dotLLM will consume: if a FRESH module rebuilt
    only from the export reproduces the reference logits, the export captured every
    load-bearing weight (completeness) and stored them faithfully (losslessness).
    """
    from safetensors.torch import load_file

    with torch.no_grad():
        ref_logits = reference_model(input_ids, use_cache=False).logits.float()

    fresh = fresh_model_builder()
    exported = load_file(os.path.join(out_dir, "model.safetensors"))
    fresh_sd = fresh.state_dict()
    remapped = {}
    for export_name, t in exported.items():
        hf_name = _import_name(export_name)
        if hf_name not in fresh_sd:
            raise KeyError(f"exported tensor {export_name!r} -> {hf_name!r} not in fresh model")
        remapped[hf_name] = t.to(fresh_sd[hf_name].dtype)
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)
    if missing:
        raise KeyError(f"fresh model missing tensors after import: {missing[:8]}")
    fresh.eval()

    with torch.no_grad():
        rt_logits = fresh(input_ids, use_cache=False).logits.float()

    max_abs = (rt_logits - ref_logits).abs().max().item()
    mean_abs = (rt_logits - ref_logits).abs().mean().item()
    argmax_match = bool((rt_logits.argmax(-1) == ref_logits.argmax(-1)).all().item())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "argmax_match": argmax_match,
        "ok": (max_abs <= tol) and argmax_match,
    }


# ---------------------------------------------------------------------------
# Self-test — tiny synthetic identity-MoTE, no downloads, no GPU
# ---------------------------------------------------------------------------


def _perturb_trained(model, inserted_indices, seed: int = 3) -> None:
    """Simulate a trained adapter: give capability experts a REAL (non-zero) down_proj
    and a non-trivial router, so the round-trip is a meaningful (non-identity) parity
    test. The skip expert (index 0) stays down_proj==0 (frozen identity)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for li in inserted_indices:
            block = model.model.layers[li].mlp
            assert isinstance(block, IdentityMoTEBlock)
            for e in range(1, block.n_experts):  # capability experts only
                mlp = block.experts[e]
                mlp.down_proj.weight.copy_(
                    torch.randn(mlp.down_proj.weight.shape, generator=g) * 0.05)
                mlp.gate_proj.weight.copy_(
                    torch.randn(mlp.gate_proj.weight.shape, generator=g) * 0.05)
            block.router.weight.normal_(0.0, 1.0, generator=g)
            block.router.bias.normal_(0.0, 1.0, generator=g)


def run_self_test(tmp_dir: Optional[str] = None, tol: float = 1e-3) -> bool:
    import copy
    import tempfile

    cleanup = False
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="mote_export_selftest_")
        cleanup = True

    print("[self-test] building tiny synthetic BitNet + identity-MoTE (K=2, 2 inserted)...")
    torch.manual_seed(0)

    K = 2
    cfg = {
        "n_capability_experts": K,
        "capability_init": "zero",
        "every": 1,
        "inserted_indices": None,   # filled after reconstruct
        "label_map": {"math": 1, "instruction": 2},
        "layers_before": None,
    }

    # Reference model: reconstruct the training-time module, then "train" (perturb).
    base_ref = bde.build_tiny_bitnet()
    cfg["layers_before"] = base_ref.config.num_hidden_layers
    ref_model, inserted = reconstruct_model(base_ref, cfg, adapter_state=None)
    cfg["inserted_indices"] = inserted
    _perturb_trained(ref_model, inserted)
    print(f"[self-test] inserted_indices={inserted}  n_experts={K + 1}")

    torch.manual_seed(11)
    input_ids = torch.randint(0, ref_model.config.vocab_size, (2, 12))

    # Export.
    info = export_to_dotllm(ref_model, cfg, inserted, tmp_dir)
    print(f"[self-test] wrote {info['n_tensors']} tensors + config.json to {tmp_dir}")
    print(f"[self-test] config MoE: num_experts={info['config']['num_experts']} "
          f"top_k={info['config']['num_experts_per_tok']} "
          f"mlp_only_layers={info['config']['mlp_only_layers']}")

    # Fresh model builder for the round-trip (same structure, DIFFERENT init).
    def _fresh():
        base = bde.build_tiny_bitnet()
        with torch.no_grad():  # scramble so the import must supply every weight
            for p in base.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        m, _ = reconstruct_model(base, cfg, adapter_state=None)
        with torch.no_grad():
            for li in inserted:
                blk = m.model.layers[li].mlp
                blk.router.weight.normal_(0, 1)
                blk.router.bias.normal_(0, 1)
        return m

    r = verify_round_trip(ref_model, tmp_dir, _fresh, input_ids, tol=tol)
    print(f"[self-test] round-trip: max_abs={r['max_abs']:.3e} mean_abs={r['mean_abs']:.3e} "
          f"argmax_match={r['argmax_match']} (tol={tol:.1e})")

    # Structural assertions on the config.
    c = info["config"]
    assert c["num_experts"] == K + 1
    assert c["num_experts_per_tok"] == 1
    assert set(c["mlp_only_layers"]).isdisjoint(set(inserted)), "inserted layers must be MoE"
    assert all(l in c["mlp_only_layers"] for l in range(c["num_hidden_layers"])
               if l not in inserted), "all original layers must be force-dense"
    assert c["hidden_act"] == "relu2"
    assert c["mote"]["router_has_bias"] is True

    ok = r["ok"]
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")

    if cleanup:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a trained identity-MoTE BitNet to "
                                             "a dotLLM-loadable checkpoint.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run the tiny-synthetic round-trip validation and exit (no downloads).")
    ap.add_argument("--base", default="microsoft/bitnet-b1.58-2B-4T-bf16",
                    help="Base BitNet checkpoint (HF id or local path).")
    ap.add_argument("--adapter", default=None,
                    help="Trained adapter dir (contains adapter_weights.pt + "
                         "identity_mote_config.json).")
    ap.add_argument("--out", default=None, help="Output dir for the dotLLM checkpoint.")
    ap.add_argument("--tol", type=float, default=1e-3, help="Round-trip logit tolerance.")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test(tol=args.tol) else 1

    if not args.adapter or not args.out:
        print("error: --adapter and --out are required (or use --self-test).", file=sys.stderr)
        return 2

    with open(os.path.join(args.adapter, "identity_mote_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    adapter_state = torch.load(os.path.join(args.adapter, "adapter_weights.pt"),
                               map_location="cpu")

    print(f"[export] loading base {args.base!r} ...")
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)

    model, inserted = reconstruct_model(base, cfg, adapter_state)
    info = export_to_dotllm(model, cfg, inserted, args.out)
    print(f"[export] wrote {info['n_tensors']} tensors + config.json to {args.out}")
    print(f"[export] MoE: num_experts={info['config']['num_experts']} top_k=1 "
          f"inserted={inserted}")

    # Optional in-Python parity check on a random prompt (proves reconstruction is exact).
    torch.manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (1, 16))

    def _fresh():
        b = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
        m, _ = reconstruct_model(b, cfg, adapter_state=None)
        return m

    r = verify_round_trip(model, args.out, _fresh, ids, tol=args.tol)
    print(f"[export] round-trip parity: max_abs={r['max_abs']:.3e} ok={r['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
