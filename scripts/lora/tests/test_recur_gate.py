"""Tests for recur_gate.generate_adaptive — Task R4 (live exit gate).

All run on CPU, no GPU, no background jobs. The model is loaded untrained
(build_recur with a random fusion adapter + gate g=0.5); the gate's *control
logic* is what is under test, not generation quality.

Verifiable properties:

  (a) generate_adaptive returns a (1, N) generated_ids tensor and a
      per_token_loops list of matching length N.
  (b) every recorded loop count lies in [1, n_max].
  (c) loop counts respond to difficulty: permissive thresholds force early exit
      (all loops == 1) while strict thresholds force max looping (n_max present),
      and mean(strict) > mean(permissive). This demonstrates both the loops==1
      (early-exit) and loops==n_max (max-loop) regimes deterministically without
      relying on an untrained model's raw signal varying on its own.
  (d) n_max=1 (prefill_loops=1) reproduces plain 1-loop greedy decode exactly
      (sanity: the adaptive path collapses to the stock looped forward).

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface pytest scripts/lora/tests/test_recur_gate.py -v
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from transformers import AutoTokenizer, BitNetForCausalLM
from recur_model import build_recur, DEFAULT_P, DEFAULT_Q
from recur_gate import generate_adaptive

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROMPT = "The capital of France is"
N_MAX = 4  # small for CPU speed; still exercises 1..n_max range


@pytest.fixture(scope="module")
def base_model():
    # Load bf16: the direct fp32 load triggers a native access violation in this
    # transformers/torch build (BitNet quantizer dequant path), and .float() is
    # blocked on the quantized model. bf16 is fine here — these tests check the
    # gate's control logic (loop counts), not generation quality, and (d) compares
    # the cached path against the full-recompute reference using the *same* bf16
    # model so any bf16 noise is shared.
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = BitNetForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


@pytest.fixture(scope="module")
def recur(base_model):
    _, model = base_model
    r = build_recur(model, P=DEFAULT_P, Q=DEFAULT_Q)
    # base model is bf16; cast the new adapter + gate params to match.
    r.fusion.to(torch.bfloat16)
    r.gate.to(torch.bfloat16)
    r.eval()
    return r


@pytest.fixture(scope="module")
def input_ids(base_model):
    tok, _ = base_model
    return tok(PROMPT, return_tensors="pt").input_ids


def _plain_greedy(model, input_ids, max_new_tokens):
    """Reference: plain 1-loop greedy via full-sequence recompute (recurrence=1)."""
    ids = input_ids
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids, recurrence=1)
            tok = int(logits[:, -1, :].argmax(-1).item())
            out.append(tok)
            ids = torch.cat(
                [ids, torch.tensor([[tok]], device=ids.device, dtype=ids.dtype)],
                dim=1,
            )
    return out


# ----------------------------------------------------------------------- (a)/(b)
def test_shape_and_loop_range(recur, input_ids):
    """generated_ids shape + matching per_token_loops, all loops in [1, n_max]."""
    gen, loops = generate_adaptive(
        recur, input_ids, n_max=N_MAX, signal="both", max_new_tokens=4,
    )
    assert gen.dim() == 2 and gen.shape[0] == 1, f"bad shape {tuple(gen.shape)}"
    assert gen.shape[1] == len(loops), \
        f"generated {gen.shape[1]} tokens but {len(loops)} loop counts"
    assert gen.shape[1] == 4, f"expected 4 generated tokens, got {gen.shape[1]}"
    assert all(1 <= n <= N_MAX for n in loops), \
        f"loop counts out of [1,{N_MAX}]: {loops}"


# --------------------------------------------------------------------------- (c)
def test_loop_counts_respond_to_thresholds(recur, input_ids):
    """Permissive -> all early-exit (1); strict -> max-loop (n_max) present."""
    # Permissive: nothing is ever hard (impossible entropy + zero margin gate).
    _, permissive = generate_adaptive(
        recur, input_ids, n_max=N_MAX, signal="both",
        ent_thresh=1e9, margin_thresh=-1.0, max_new_tokens=4,
    )
    # Strict: everything is hard (entropy always exceeds 0; margin gate huge).
    _, strict = generate_adaptive(
        recur, input_ids, n_max=N_MAX, signal="both",
        ent_thresh=-1.0, margin_thresh=1e9, max_new_tokens=4,
    )

    assert all(n == 1 for n in permissive), \
        f"permissive thresholds should early-exit at 1 loop: {permissive}"
    assert 1 in permissive, "permissive run must demonstrate loops==1"
    assert N_MAX in strict, f"strict thresholds should hit n_max: {strict}"
    mean_p = sum(permissive) / len(permissive)
    mean_s = sum(strict) / len(strict)
    assert mean_s > mean_p, \
        f"strict mean loops ({mean_s}) should exceed permissive ({mean_p})"


# --------------------------------------------------------------------------- (d)
def test_nmax1_equals_plain_greedy(recur, input_ids):
    """n_max=1, prefill_loops=1 reproduces plain 1-loop greedy decode exactly."""
    gen, loops = generate_adaptive(
        recur, input_ids, n_max=1, prefill_loops=1, max_new_tokens=4,
    )
    ref = _plain_greedy(recur, input_ids, max_new_tokens=4)

    assert all(n == 1 for n in loops), f"n_max=1 must give all 1-loop: {loops}"
    assert gen[0].tolist() == ref, (
        f"n_max=1 adaptive {gen[0].tolist()} != plain greedy {ref} "
        "(KV-cached decode must match full-sequence recompute)"
    )
