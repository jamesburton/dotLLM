#!/usr/bin/env python3
"""bench_infer_cache.py — Measure BitNet 2B generation speed: cache OFF vs ON.

Usage (from repo root):
    HF_HOME=E:/.cache/huggingface HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 PYTHONUTF8=1 \
    C:/Python311/python.exe scripts/diffusion/bench_infer_cache.py --device cuda --n-tokens 64

    # CPU benchmark (slower but usable for correctness check):
    HF_HOME=E:/.cache/huggingface HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 PYTHONUTF8=1 \
    C:/Python311/python.exe scripts/diffusion/bench_infer_cache.py --device cpu --n-tokens 16
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdiff_ternary_lora import BASE_MODEL_ID, enable_inference_cache, disable_inference_cache

PROMPT = "The Eiffel Tower is located in"

def _load(device):
    kw = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if device.type != "cpu":
        kw["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kw)
    if device.type == "cpu":
        model = model.to(device)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, local_files_only=True)
    model.config.use_cache = True
    model.eval()
    return model, tok

@torch.no_grad()
def _bench(model, tok, device, n_tokens, label):
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    # Warm-up: 4 tokens
    _ = model.generate(ids, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=n_tokens, do_sample=False)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    generated = out.shape[1] - ids.shape[1]
    tok_per_s = generated / elapsed
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    print(f"  [{label}]  {generated} tokens in {elapsed:.2f}s = {tok_per_s:.2f} tok/s")
    print(f"  output: {text[:80]!r}")
    return tok_per_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-tokens", type=int, default=64)
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"Loading {BASE_MODEL_ID} on {device} ...")
    model, tok = _load(device)
    n_layers = len(model.model.layers)
    print(f"Loaded: {n_layers} layers")

    print(f"\n--- CACHE OFF ---")
    disable_inference_cache(model)
    tps_off = _bench(model, tok, device, args.n_tokens, "cache-OFF")

    print(f"\n--- CACHE ON ---")
    n_patched = enable_inference_cache(model)
    print(f"  ({n_patched} modules patched)")
    tps_on = _bench(model, tok, device, args.n_tokens, "cache-ON")

    speedup = tps_on / max(tps_off, 1e-9)
    print(f"\n  Speedup: {tps_off:.2f} -> {tps_on:.2f} tok/s  ({speedup:.1f}x)")
    print("  NOTE: cached qweights ~= same size as weights -> ~doubles weight memory during inference.")
    print("        Acceptable on Strix 128GB unified; on CUDA 12GB may be tight for 2B.")

if __name__ == "__main__":
    main()
