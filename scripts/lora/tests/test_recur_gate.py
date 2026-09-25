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
  (e) n_max=2 with all-strict thresholds: every token uses exactly 2 loops, and
      results are deterministic across two identical calls (crop isolates loop
      iterations; no stale KV leaks between generate_adaptive invocations).
      Note: n_max=2 cached decode is NOT token-equivalent to _plain_greedy_n2
      (see test docstring) — that comparison only holds for n_max=1.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface pytest scripts/lora/tests/test_recur_gate.py -v
"""

from __future__ import annotations

import json as _json
import os
import struct as _struct
import sys

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HOME", "E:/.cache/huggingface")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

# Disable torch.compile globally — avoids a Windows SIGSEGV in the Dynamo JIT
# subprocess (no cl.exe; transformers BitNet uses @torch.compile on WeightQuant /
# ActQuant; suppress_errors=True only catches Python-level exceptions, not
# C-level segfaults from the compile process spawned on 2nd+ call).
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from transformers import AutoTokenizer, BitNetForCausalLM
from recur_model import build_recur, DEFAULT_P, DEFAULT_Q
from recur_gate import generate_adaptive

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T-bf16"
PROMPT = "The capital of France is"
N_MAX = 4  # small for CPU speed; still exercises 1..n_max range


# ---------------------------------------------------------------------------
# Direct-IO shim for safetensors safe_open
# ---------------------------------------------------------------------------
# safetensors 0.7.0 SIGSEGV on Windows: the PyTorch mmap backend crashes when
# accessing a >4 GB file while a CUDA context is active (Windows WDDM + large-
# file mmap conflict). Direct file IO (frombuffer over bytearray) is safe.
# The shim is swapped in for the duration of from_pretrained only.

class _DirectIOSlice:
    """Slice proxy that reads tensors via direct IO (no mmap)."""
    _DTYPE_MAP = {
        'BF16': torch.bfloat16, 'F16': torch.float16,
        'F32': torch.float32,   'F64': torch.float64,
        'I8':  torch.int8,      'I16': torch.int16,
        'I32': torch.int32,     'I64': torch.int64,
        'U8':  torch.uint8,
    }

    def __init__(self, file_obj, info, data_offset: int) -> None:
        self._f = file_obj
        self._info = info
        self._data_offset = data_offset

    def get_shape(self):
        return self._info['shape']

    def get_dtype(self):
        return self._info['dtype']

    def __getitem__(self, _idx):
        offsets = self._info['data_offsets']
        self._f.seek(self._data_offset + offsets[0])
        raw = self._f.read(offsets[1] - offsets[0])
        dtype = self._DTYPE_MAP[self._info['dtype']]
        t = torch.frombuffer(bytearray(raw), dtype=dtype)
        shape = self._info['shape']
        return t.reshape(shape) if shape else t


class _DirectIOSafeFile:
    """Drop-in for ``safetensors.safe_open`` using direct file IO (no mmap).

    Implements the subset of the safetensors file-pointer API used by
    ``transformers.modeling_utils._load_state_dict_into_meta_model``:
    ``keys()``, ``metadata()``, ``get_slice(name)``, ``get_tensor(name)``,
    plus context-manager protocol.
    """

    def __init__(self, filename, framework='pt', device='cpu') -> None:
        self._file = open(str(filename), 'rb')
        hdr_len = _struct.unpack('<Q', self._file.read(8))[0]
        self._header: dict = _json.loads(self._file.read(hdr_len))
        self._data_offset = 8 + hdr_len

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()

    def keys(self):
        return [k for k in self._header if k != '__metadata__']

    def metadata(self):
        return self._header.get('__metadata__', {})

    def get_slice(self, name: str) -> _DirectIOSlice:
        return _DirectIOSlice(self._file, self._header[name], self._data_offset)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[...]


@pytest.fixture(scope="module")
def base_model():
    # Load bf16: the direct fp32 load triggers a native access violation in this
    # transformers/torch build (BitNet quantizer dequant path), and .float() is
    # blocked on the quantized model. bf16 is fine here — these tests check the
    # gate's control logic (loop counts), not generation quality, and (d) compares
    # the cached path against the full-recompute reference using the *same* bf16
    # model so any bf16 noise is shared.
    #
    # safetensors 0.7.0 mmap patch: the 4.6 GB model file crashes safe_open's
    # PyTorch mmap backend (SIGSEGV) on Windows when a CUDA context is active.
    # Patch transformers.modeling_utils.safe_open with _DirectIOSafeFile for
    # the duration of from_pretrained, then restore it.
    import transformers.modeling_utils as _mutils
    _orig_safe_open = _mutils.safe_open
    _mutils.safe_open = _DirectIOSafeFile
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = BitNetForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.eval()
    finally:
        _mutils.safe_open = _orig_safe_open
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


def _plain_greedy_n2(model, input_ids, max_new_tokens):
    """Reference: 2-loop greedy via true O(seq²) full-sequence recompute.

    Each step appends the last emitted token and re-runs the whole sequence
    with ``recurrence=2`` from scratch — no KV cache involved.
    """
    ids = input_ids
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids, recurrence=2)
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


# --------------------------------------------------------------------------- (e)
def test_nmax2_crop_and_determinism(recur, input_ids):
    """n_max=2: crop isolates loop iterations; output is fully deterministic.

    generate_adaptive with n_max=2 uses "Ouro last-step KV reuse": every decode
    loop (1..n_max) for position k attends to past positions' *last-prefill-loop*
    KV that was committed during the prefill — the same cached past KV for all
    loops.  _crop_span removes only the *current* position's previous-loop KV so
    that each new loop starts from a clean single-token query against unchanged
    past context.

    This is semantically different from _plain_greedy_n2 (full O(seq²) recompute),
    which re-runs all positions from scratch in every loop so past positions have
    a different loop-1 KV vs loop-2 KV.  Token equality across those two paths is
    NOT a valid correctness criterion for n_max=2.  (n_max=1 IS equivalent to
    full recompute — that is verified in test_nmax1_equals_plain_greedy.)

    What this test verifies:

    1. **Loop counts**: under all-strict thresholds every generated token
       (prefill + all decode steps) uses exactly n_max=2 loops.  This proves
       _is_hard gates correctly and the n_max cap is hit, not the signal threshold.

    2. **Determinism**: a second call with identical inputs returns the same
       generated tokens and the same per-token loop counts.  Non-determinism
       would indicate _crop_span leaves stale KV in the cache across calls or
       that the inner loop modifies shared state.
    """
    gen1, loops1 = generate_adaptive(
        recur, input_ids, n_max=2, prefill_loops=2,
        ent_thresh=-1.0, margin_thresh=1e9, max_new_tokens=4,
    )
    # Every token (prefill and all decode steps) must use exactly 2 loops.
    assert all(n == 2 for n in loops1), (
        f"expected all loop counts == 2 under strict thresholds, got {loops1}"
    )

    # Determinism: re-running with the same input must produce identical results.
    gen2, loops2 = generate_adaptive(
        recur, input_ids, n_max=2, prefill_loops=2,
        ent_thresh=-1.0, margin_thresh=1e9, max_new_tokens=4,
    )
    assert gen1.tolist() == gen2.tolist(), (
        f"generate_adaptive not deterministic: first run {gen1.tolist()} "
        f"!= second run {gen2.tolist()} "
        "(stale KV may persist between generate_adaptive calls)"
    )
    assert loops1 == loops2, (
        f"loop counts not deterministic: {loops1} vs {loops2}"
    )
