#!/usr/bin/env python3
"""bitdistill_export.py — export a BitDistill ternary student to a dotLLM checkpoint.

Mirrors ``mote_export.py``: writes a self-contained HuggingFace-style directory
with a ``config.json`` (``model_type: bitnet``) and ``model.safetensors`` holding
the **fp master weights**. dotLLM's BitNet I2_S loader ternary-quantizes every
linear at load time (absmean per-tensor scale — identical to this harness's
``weight_quant_ternary``), so exporting master weights is lossless.

Tensor mapping (BitDistill student -> dotLLM bitnet names)
----------------------------------------------------------
* ``model.layers.{l}.self_attn.{q,k,v,o}_proj.weight``  — kept (BitLinear.weight).
* ``model.layers.{l}.mlp.{gate,up,down}_proj.weight``   — kept (BitLinear.weight).
* ``self_attn.o_proj.sub_norm.weight``  ->  ``self_attn.attn_sub_norm.weight``  (SubLN).
* ``mlp.down_proj.sub_norm.weight``     ->  ``mlp.ffn_sub_norm.weight``          (SubLN).
* ``self_attn.q_norm.weight`` / ``k_norm.weight``       — kept (Qwen3 QK-norm).
* embeddings / lm_head / input_layernorm / post_attention_layernorm / final norm — kept.

dotLLM-side work required (NOT parity out-of-the-box — this is the "note exactly how")
-------------------------------------------------------------------------------------
dotLLM's existing BitNet path targets microsoft/bitnet-b1.58, which differs from a
ternarized **Qwen3** in two load-bearing ways:

  1. **FFN activation.** BitNet uses ``relu2`` (squared ReLU); Qwen3 is **SwiGLU
     (silu gate)**. ``config.hidden_act`` is exported as ``silu`` and the dotLLM
     BitNet FFN kernel must honour it (it currently assumes relu2).
  2. **QK-norm.** Qwen3 applies an RMSNorm to Q and K per head (``q_norm``/``k_norm``)
     before RoPE. microsoft-BitNet has none. dotLLM's BitNet attention must apply
     these two norms. Tensor names are exported; the kernel path is the gap.

Both are small, localized additions to dotLLM's BitNet loader/kernels. The SubLN
tensors and I2_S ternary packing already match dotLLM's BitNet conventions (as
proven by ``mote_export.py`` / the shipped BitNet support).

Self-test (CPU, seconds, no download)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 python scripts/lora/bitdistill_export.py --self-test

Real export (after a training run)::

    python scripts/lora/bitdistill_export.py \
        --base Qwen/Qwen3-0.6B \
        --ckpt .docs/bitdistill/qwen3_0p6b_curve/final \
        --out  .docs/bitdistill/qwen3_0p6b_dotllm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill as bd  # noqa: E402


def export_name(hf_name: str) -> str:
    """Map a BitDistill student state_dict name to the dotLLM bitnet tensor name."""
    if hf_name.endswith(".self_attn.o_proj.sub_norm.weight"):
        return hf_name[: -len(".o_proj.sub_norm.weight")] + ".attn_sub_norm.weight"
    if hf_name.endswith(".mlp.down_proj.sub_norm.weight"):
        return hf_name[: -len(".down_proj.sub_norm.weight")] + ".ffn_sub_norm.weight"
    return hf_name


def import_name(dotllm_name: str) -> str:
    """Inverse of :func:`export_name` (dotLLM -> BitDistill student name)."""
    if dotllm_name.endswith(".self_attn.attn_sub_norm.weight"):
        return dotllm_name[: -len(".attn_sub_norm.weight")] + ".o_proj.sub_norm.weight"
    if dotllm_name.endswith(".mlp.ffn_sub_norm.weight"):
        return dotllm_name[: -len(".ffn_sub_norm.weight")] + ".down_proj.sub_norm.weight"
    return dotllm_name


def build_dotllm_config(model) -> dict:
    mc = model.config
    head_dim = getattr(mc, "head_dim", None) or (mc.hidden_size // mc.num_attention_heads)
    return {
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": mc.hidden_size,
        "intermediate_size": mc.intermediate_size,
        "num_hidden_layers": mc.num_hidden_layers,
        "num_attention_heads": mc.num_attention_heads,
        "num_key_value_heads": getattr(mc, "num_key_value_heads", mc.num_attention_heads),
        "head_dim": head_dim,
        "vocab_size": mc.vocab_size,
        "max_position_embeddings": getattr(mc, "max_position_embeddings", 4096),
        "rms_norm_eps": getattr(mc, "rms_norm_eps", 1e-6),
        "rope_theta": float(getattr(mc, "rope_theta", 1000000.0)),
        # Qwen3 is SwiGLU (silu) — NOT relu2. dotLLM BitNet FFN kernel must honour this.
        "hidden_act": getattr(mc, "hidden_act", "silu"),
        "tie_word_embeddings": bool(getattr(mc, "tie_word_embeddings", False)),
        # provenance for the dotLLM loader (extension keys; generic loaders ignore)
        "bitdistill": {
            "source": "bitdistill.py",
            "has_attn_sub_norm": True,
            "has_ffn_sub_norm": True,
            "has_qk_norm": True,            # Qwen3 q_norm / k_norm present
            "ffn_activation": getattr(mc, "hidden_act", "silu"),
            "weight_quant": "absmean-ternary-per-tensor",
            "act_quant": "int8-per-token-absmax",
        },
    }


def export(model, out_dir: str) -> dict:
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for name, p in model.state_dict().items():
        tensors[export_name(name)] = p.detach().contiguous().cpu()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt", "producer": "bitdistill_export.py"})
    cfg = build_dotllm_config(model)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return {"n_tensors": len(tensors), "config": cfg}


def verify_round_trip(reference_model, out_dir: str, fresh_builder, input_ids,
                      tol: float = 1e-4) -> dict:
    """Reload exported safetensors into a fresh converted student; compare logits."""
    from safetensors.torch import load_file
    bd.set_quant_alpha(reference_model, 1.0)
    with torch.no_grad():
        ref = reference_model(input_ids, use_cache=False).logits.float()

    fresh = fresh_builder()
    bd.set_quant_alpha(fresh, 1.0)
    exported = load_file(os.path.join(out_dir, "model.safetensors"))
    fresh_sd = fresh.state_dict()
    remapped = {}
    for en, t in exported.items():
        hn = import_name(en)
        if hn not in fresh_sd:
            raise KeyError(f"exported {en!r} -> {hn!r} not in fresh student")
        remapped[hn] = t.to(fresh_sd[hn].dtype)
    missing, unexpected = fresh.load_state_dict(remapped, strict=False)
    if missing:
        raise KeyError(f"fresh student missing after import: {missing[:8]}")
    fresh.eval()
    with torch.no_grad():
        rt = fresh(input_ids, use_cache=False).logits.float()
    max_abs = (rt - ref).abs().max().item()
    return {"max_abs": max_abs,
            "argmax_match": bool((rt.argmax(-1) == ref.argmax(-1)).all().item()),
            "ok": max_abs <= tol}


def run_self_test(tol: float = 1e-4) -> bool:
    import copy, tempfile, shutil
    print("[self-test] tiny Qwen3 -> BitDistill student -> export -> round-trip ...")
    teacher = bd.build_tiny_qwen3()
    student = copy.deepcopy(teacher)
    bd.convert_to_bitnet_student(student)
    # simulate training: perturb master weights + sub_norms so export is non-trivial
    with torch.no_grad():
        for m in student.modules():
            if isinstance(m, bd.BitLinear):
                m.weight.normal_(0.0, 0.3)
                if m.sub_norm is not None:
                    m.sub_norm.weight.normal_(1.0, 0.05)
    student.eval()

    torch.manual_seed(7)
    ids = torch.randint(0, student.config.vocab_size, (2, 12))

    tmp = tempfile.mkdtemp(prefix="bitdistill_export_")
    try:
        info = export(student, tmp)
        print(f"[self-test] wrote {info['n_tensors']} tensors + config.json")
        print(f"[self-test] config: model_type={info['config']['model_type']} "
              f"hidden_act={info['config']['hidden_act']} "
              f"has_qk_norm={info['config']['bitdistill']['has_qk_norm']}")
        # confirm the sub_norm renames landed
        from safetensors.torch import load_file
        names = list(load_file(os.path.join(tmp, "model.safetensors")).keys())
        assert any(n.endswith(".self_attn.attn_sub_norm.weight") for n in names), \
            "attn_sub_norm rename missing"
        assert any(n.endswith(".mlp.ffn_sub_norm.weight") for n in names), \
            "ffn_sub_norm rename missing"

        def _fresh():
            t2 = bd.build_tiny_qwen3()
            with torch.no_grad():
                for p in t2.parameters():
                    p.add_(torch.randn_like(p) * 0.01)  # scramble; import must supply all
            bd.convert_to_bitnet_student(t2)
            return t2

        r = verify_round_trip(student, tmp, _fresh, ids, tol=tol)
        print(f"[self-test] round-trip: max_abs={r['max_abs']:.3e} "
              f"argmax_match={r['argmax_match']} (tol={tol:.1e})")
        ok = r["ok"] and r["argmax_match"]
        print(f"[self-test] {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a BitDistill student to a dotLLM checkpoint.")
    ap.add_argument("--self-test", action="store_true", help="Tiny round-trip validation, no download.")
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B", help="Base FP model (structure source).")
    ap.add_argument("--ckpt", default=None, help="Trained student ckpt dir (student_state.pt).")
    ap.add_argument("--out", default=None, help="Output dir for the dotLLM checkpoint.")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if run_self_test(tol=args.tol) else 1
    if not args.ckpt or not args.out:
        print("error: --ckpt and --out required (or --self-test).", file=sys.stderr)
        return 2

    from transformers import AutoModelForCausalLM
    print(f"[export] loading base {args.base!r} + converting to student structure ...")
    student = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    bd.convert_to_bitnet_student(student)
    state = torch.load(os.path.join(args.ckpt, "student_state.pt"), map_location="cpu")
    missing, unexpected = student.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys in ckpt: {unexpected[:8]}")
    student.eval()
    info = export(student, args.out)
    print(f"[export] wrote {info['n_tensors']} tensors + config.json to {args.out}")
    print("[export] NOTE: dotLLM BitNet path needs silu-SwiGLU + QK-norm support "
          "(see file header).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
