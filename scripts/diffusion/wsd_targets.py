"""Extract real FFN weight blocks from the BitNet base as diffusion targets (CPU, mmap)."""
import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

_HIDDEN = 2560
_ROWS = 16
_DIM = _ROWS * _HIDDEN  # 40960


def _resolve_shard(base_id: str, key: str) -> str:
    """Return local path to the safetensors shard containing `key` (cache-only)."""
    try:
        idx_path = hf_hub_download(base_id, "model.safetensors.index.json", local_files_only=True)
        weight_map = json.load(open(idx_path))["weight_map"]
        shard = weight_map[key]
    except Exception:
        shard = "model.safetensors"  # single-file model
    return hf_hub_download(base_id, shard, local_files_only=True)


def _list_gate_proj_keys(base_id: str) -> list:
    """List all gate_proj weight keys across the entire model (handles sharded and single-file)."""
    try:
        # Sharded: index JSON has the complete weight_map for all shards
        idx_path = hf_hub_download(base_id, "model.safetensors.index.json", local_files_only=True)
        weight_map = json.load(open(idx_path))["weight_map"]
        return sorted([k for k in weight_map if k.endswith("mlp.gate_proj.weight")])
    except Exception:
        # Single-file: open the one file and enumerate its keys
        path = hf_hub_download(base_id, "model.safetensors", local_files_only=True)
        with safe_open(path, framework="pt", device="cpu") as f:
            return sorted([k for k in f.keys() if k.endswith("mlp.gate_proj.weight")])


def _gate_proj_key(layer: int) -> str:
    return f"model.layers.{layer}.mlp.gate_proj.weight"


def extract_targets(base_id: str, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    # Discover layer count by listing ALL gate_proj keys (handles sharded and single-file)
    keys = _list_gate_proj_keys(base_id)
    n_layers = len(keys)
    assert n_layers > 0, "no mlp.gate_proj.weight tensors found in base"

    # choose k distinct (layer, row_off) locations deterministically
    locations = []
    seen = set()
    while len(locations) < k:
        layer = int(torch.randint(0, n_layers, (1,), generator=g))
        # gate_proj rows = 6912; leave room for a 16-row block
        row_off = int(torch.randint(0, 6912 - _ROWS, (1,), generator=g))
        if (layer, row_off) in seen:
            continue
        seen.add((layer, row_off)); locations.append([layer, row_off])

    vecs = []
    for layer, row_off in locations:
        key = _gate_proj_key(layer)
        shard = _resolve_shard(base_id, key)
        with safe_open(shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(key)                       # [6912, 2560]
        block = w[row_off:row_off + _ROWS, :].to(torch.float32).reshape(-1)  # [40960]
        vecs.append(block)
    raw = torch.stack(vecs)                              # [k, 40960]

    mean = float(raw.mean()); std = float(raw.std()) + 1e-12
    meta = {"mean": mean, "std": std, "dim": _DIM, "block_shape": [_ROWS, _HIDDEN],
            "locations": locations, "base_id": base_id}
    return (raw - mean) / std, meta


def denormalize(x: torch.Tensor, meta: dict) -> torch.Tensor:
    return x * meta["std"] + meta["mean"]


def normalize(x: torch.Tensor, meta: dict) -> torch.Tensor:
    return (x - meta["mean"]) / meta["std"]
