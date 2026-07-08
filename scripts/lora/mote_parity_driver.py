"""mote_parity_driver.py — memory-frugal export + PyTorch reference for the dotLLM MoTE
cross-framework parity gate.

Loads the base ONCE, reconstructs the exact training-time identity-MoTE structure via
mote_export.reconstruct_model (base -> depth-expand -> build_identity_mote), then attaches
the trained adapter (mmap, so the 2.2 GB file stays file-backed rather than commit-charged),
captures reference logits for a FIXED token sequence, and serialises the dotLLM checkpoint.
No round-trip second model (which would overflow the commit charge on this box).
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mote_export as mx  # noqa: E402

# Fixed 12-token sequence (all < vocab 128256). Must match the C# harness exactly.
TOKENS = [1, 15, 234, 1001, 42, 777, 3333, 88, 512, 9, 128, 64]


def main() -> int:
    base_id, adapter_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(os.path.join(adapter_dir, "identity_mote_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    print(f"[parity] cfg: K={cfg['n_capability_experts']} inserted={cfg['inserted_indices']} "
          f"layers_before={cfg['layers_before']}", flush=True)

    from transformers import AutoModelForCausalLM
    print(f"[parity] loading base {base_id!r} (bf16, low_cpu_mem_usage) ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    print("[parity] base loaded", flush=True)

    # Structure only (adapter_state=None) — this reuses the exporter's exact insertion
    # planning + identity-MoTE build, so topology is guaranteed to match training.
    model, inserted = mx.reconstruct_model(base, cfg, adapter_state=None)
    del base
    gc.collect()
    print(f"[parity] reconstructed structure: layers={model.config.num_hidden_layers} "
          f"inserted={inserted}", flush=True)

    # Attach trained weights (mmap keeps the 2.2 GB file file-backed).
    adapter_state = torch.load(
        os.path.join(adapter_dir, "adapter_weights.pt"),
        map_location="cpu", weights_only=True, mmap=True)
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    if unexpected:
        raise ValueError(f"adapter has unexpected keys: {unexpected[:8]}")
    print(f"[parity] adapter attached: {len(adapter_state)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    del adapter_state
    gc.collect()

    model.config.use_cache = False
    model.eval()

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits.float()[0]  # [T, V]
    logits_np = logits.cpu().numpy()
    print(f"[parity] reference logits: shape={logits_np.shape} "
          f"argmax_last={int(logits_np[-1].argmax())} "
          f"finite={bool(np.isfinite(logits_np).all())}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pytorch_ref_logits.npy"), logits_np)
    with open(os.path.join(out_dir, "tokens.json"), "w", encoding="utf-8") as fh:
        json.dump({"tokens": TOKENS, "vocab": int(model.config.vocab_size)}, fh)

    info = mx.export_to_dotllm(model, cfg, inserted, out_dir)
    print(f"[parity] exported {info['n_tensors']} tensors + config.json to {out_dir}", flush=True)
    print(f"[parity] MoE: num_experts={info['config']['num_experts']} top_k=1 "
          f"mlp_only_layers(len)={len(info['config']['mlp_only_layers'])} "
          f"hidden_act={info['config']['hidden_act']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
