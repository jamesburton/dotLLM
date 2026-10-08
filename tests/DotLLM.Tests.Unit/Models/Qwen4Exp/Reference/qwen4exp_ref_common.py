"""Shared helpers for the Qwen4-Exp HF reference-fixture generators (issue #816).

Fixtures are small JSON files: {"meta": {...}, "tensors": {name: {"shape": [...], "dtype": "f32"|"i64", "b64": "..."}}}
(little-endian raw bytes, base64). CI has no torch, so HF values are baked in and the generators live next to them.

Recipe (py 3.12, CPU torch, transformers 5.19 which ships `qwen4_exp`):
    py -3.12 -m venv hfenv
    hfenv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    hfenv/Scripts/python -m pip install transformers numpy safetensors
    hfenv/Scripts/python <generator>.py <out.json>
"""
import base64
import json
import sys

import numpy as np
import torch


class FixtureWriter:
    def __init__(self, **meta):
        self.meta = meta
        self.tensors = {}

    def add(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.ascontiguousarray(value)
        if value.dtype == np.int64:
            dtype = "i64"
        else:
            value = value.astype("<f4")
            dtype = "f32"
        self.tensors[name] = {
            "shape": list(value.shape),
            "dtype": dtype,
            "b64": base64.b64encode(value.tobytes()).decode("ascii"),
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": self.meta, "tensors": self.tensors}, f, separators=(",", ":"))
        print(f"wrote {path}: {len(self.tensors)} tensors", file=sys.stderr)
