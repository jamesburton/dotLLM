"""
MDLM unit tests + smoke gate.  No pytest required — plain assert statements.

Run with:
    C:\\Python311\\python.exe scripts\\diffusion_lm\\mdlm_test.py

Tests:
  1. test_masking_process    — mask fraction matches t; unmasked tokens preserved
  2. test_loss_finite        — forward pass and loss are finite; no NaN/Inf
  3. test_loss_decreases     — loss strictly decreases on a tiny overfit batch
  4. test_sampling_shape     — iterative_unmask returns correct shape, no MASK tokens left
  5. test_smoke_gate         — overfit tiny model on 8 synthetic sequences;
                               assert masked-token accuracy > 0.90 at t=0.5

All tests use CPU only (device="cpu").
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# --- import from sibling module -------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mdlm_model import (
    MDLMConfig,
    MDLMModel,
    apply_mask,
    eval_masked_accuracy,
    eval_bpd,
    iterative_unmask,
    mdlm_simple_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(vocab: int = 64, seq: int = 32) -> MDLMConfig:
    """Tiny model for fast unit tests."""
    return MDLMConfig(
        n_layers=2, d_model=64, n_heads=4, ffn_mult=4,
        vocab_size=vocab, max_seq_len=seq, dropout=0.0,
    )


def _smoke_config() -> MDLMConfig:
    """Smoke-gate model: slightly larger than tiny for robust overfitting."""
    return MDLMConfig(
        n_layers=2, d_model=128, n_heads=4, ffn_mult=4,
        vocab_size=128, max_seq_len=64, dropout=0.0,
    )


def _make_tokens(config: MDLMConfig, B: int = 4, L: int = 32) -> torch.Tensor:
    """Random clean token batch [B, L] avoiding the mask token."""
    return torch.randint(0, config.vocab_size - 1, (B, L))


# ---------------------------------------------------------------------------
# Test 1: Masking process
# ---------------------------------------------------------------------------

def test_masking_process() -> None:
    """
    Verify the absorbing-mask forward process:
      - Mask fraction is close to t (within 3 standard deviations of Bernoulli noise)
      - Unmasked token values are identical to the clean input
      - Mask token appears only at masked positions
      - At least one position is always masked per sequence
    """
    print("test_masking_process ... ", end="", flush=True)
    torch.manual_seed(7)

    config = _tiny_config()
    B, L   = 32, 32          # large enough for law-of-large-numbers checks
    tokens = _make_tokens(config, B, L)

    for t in [0.05, 0.20, 0.50, 0.80, 0.95]:
        masked, mask = apply_mask(tokens, t, config.mask_token_id)

        # 1. Mask fraction within 3σ of expected
        expected_frac = t
        actual_frac   = mask.float().mean().item()
        std_bernoulli  = math.sqrt(t * (1 - t) / (B * L))
        assert abs(actual_frac - expected_frac) < 3 * std_bernoulli + 0.01, (
            f"t={t}: mask fraction {actual_frac:.3f} not close to {expected_frac:.3f}"
        )

        # 2. Unmasked positions are unchanged
        unchanged = (masked == tokens)
        assert unchanged[~mask].all(), (
            f"t={t}: unmasked positions were modified"
        )

        # 3. Masked positions all equal mask_token_id
        assert (masked[mask] == config.mask_token_id).all(), (
            f"t={t}: masked positions do not all equal mask_token_id"
        )

        # 4. Guarantee: at least one mask per sequence
        for b in range(B):
            assert mask[b].any(), f"t={t}: sequence {b} has no masked token"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 2: Loss is finite
# ---------------------------------------------------------------------------

def test_loss_finite() -> None:
    """
    Verify that:
      - Model forward pass produces finite logits
      - mdlm_simple_loss returns a finite, non-negative scalar
      - Loss value is plausible (≤ log(vocab_size) * some slack factor)
    """
    print("test_loss_finite ... ", end="", flush=True)
    torch.manual_seed(11)

    config = _tiny_config()
    model  = MDLMModel(config)
    tokens = _make_tokens(config, B=4, L=32)

    for t in [0.1, 0.5, 0.9]:
        loss, t_used, mask_frac = mdlm_simple_loss(model, tokens, t=t)

        assert torch.isfinite(loss), f"t={t}: loss is not finite: {loss.item()}"
        assert loss.item() >= 0,     f"t={t}: loss is negative: {loss.item()}"

        # At random init, CE loss ≈ log(vocab_size) = log(64) ≈ 4.15
        upper = math.log(config.vocab_size) * 2.0
        assert loss.item() < upper, (
            f"t={t}: loss={loss.item():.3f} unexpectedly high (upper={upper:.2f})"
        )

        assert 0.0 < mask_frac <= 1.0, f"t={t}: bad mask_frac={mask_frac}"

    print("PASS")


# ---------------------------------------------------------------------------
# Test 3: Loss decreases on overfit batch
# ---------------------------------------------------------------------------

def test_loss_decreases() -> None:
    """
    Verify that the model can overfit a single small batch:
      - Train for 200 steps on 4 fixed sequences
      - Loss at step 200 < loss at step 0 (monotone trend)
      - Final loss < 0.5 (well below random-init loss of ~4)
    """
    print("test_loss_decreases ... ", end="", flush=True)
    torch.manual_seed(13)

    config    = _tiny_config(vocab=64, seq=16)
    model     = MDLMModel(config)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)

    # Small fixed batch (4 sequences of length 16)
    tokens = _make_tokens(config, B=4, L=16)

    # Warmup loss (average over first 5 steps)
    model.train()
    initial_losses = []
    for _ in range(5):
        loss, _, _ = mdlm_simple_loss(model, tokens, t=0.5)
        initial_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    initial_loss = sum(initial_losses) / len(initial_losses)

    # Train 195 more steps
    for _ in range(195):
        loss, _, _ = mdlm_simple_loss(model, tokens, t=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f} final={final_loss:.4f}"
    )
    assert final_loss < 0.5, (
        f"Final loss {final_loss:.4f} too high (expected < 0.5 after overfitting)"
    )
    print(f"PASS  (loss: {initial_loss:.3f} -> {final_loss:.4f})")


# ---------------------------------------------------------------------------
# Test 4: Sampling shape and validity
# ---------------------------------------------------------------------------

def test_sampling_shape() -> None:
    """
    Verify iterative_unmask:
      - Returns tensor of shape [1, seq_len]
      - Contains no mask_token_id values (fully unmasked)
      - All tokens are valid vocab ids in [0, vocab_size - 1]
    """
    print("test_sampling_shape ... ", end="", flush=True)
    torch.manual_seed(17)

    config   = _tiny_config(vocab=64, seq=32)
    model    = MDLMModel(config)
    seq_len  = 32

    for n_steps in [1, 5, 10]:
        sample = iterative_unmask(
            model, seq_len=seq_len, n_steps=n_steps,
            device="cpu", greedy=True,
        )
        assert sample.shape == (1, seq_len), (
            f"n_steps={n_steps}: bad shape {sample.shape}"
        )
        assert (sample != config.mask_token_id).all(), (
            f"n_steps={n_steps}: MASK tokens remain in sample"
        )
        assert sample.min() >= 0, (
            f"n_steps={n_steps}: sample contains negative tokens"
        )
        assert sample.max() < config.vocab_size, (
            f"n_steps={n_steps}: sample contains OOV tokens ({sample.max()} >= {config.vocab_size})"
        )

    print("PASS")


# ---------------------------------------------------------------------------
# Test 5: Smoke gate — overfit + masked-token accuracy
# ---------------------------------------------------------------------------

def test_smoke_gate() -> None:
    """
    Full CPU smoke gate: overfit a tiny model on 8 synthetic sequences and
    verify that masked-token accuracy exceeds 0.90 at t=0.5.

    This validates the complete train → eval pipeline:
      - Absorbing-mask forward process
      - MDLM simple loss + gradient flow
      - Masked-token accuracy evaluation

    Expected runtime: 30–90 seconds on CPU.
    Kill criterion: accuracy <= 0.90 after 5000 steps → implementation bug.
    """
    print("test_smoke_gate ...")
    torch.manual_seed(42)

    config = _smoke_config()  # 2L 128d 4H vocab=128
    model  = MDLMModel(config)
    print(f"  Model params: {model.n_params:,}  |  vocab={config.vocab_size}  "
          f"|  mask_id={config.mask_token_id}")

    # Fixed synthetic dataset: 8 sequences of length 32
    # Tokens drawn from [0, vocab_size - 2] (never the mask token)
    rng    = torch.Generator()
    rng.manual_seed(0)
    seq_len = 32
    data   = torch.randint(0, config.vocab_size - 1, (8, seq_len), generator=rng)
    print(f"  Dataset: {data.shape[0]} sequences × {data.shape[1]} tokens = "
          f"{data.numel()} tokens to memorize")

    optimizer = optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)
    )

    # --- Training ---
    model.train()
    t0 = time.time()

    n_steps      = 5000
    log_every    = 1000
    loss_window  = 0.0
    loss_count   = 0

    import random as _random
    _random.seed(2024)

    for step in range(n_steps):
        # Sample t uniformly from [0.05, 0.95] for full MDLM coverage
        t_val = _random.uniform(0.05, 0.95)

        loss, _, _ = mdlm_simple_loss(model, data, t=t_val)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_window += loss.item()
        loss_count  += 1

        if (step + 1) % log_every == 0:
            avg_loss = loss_window / loss_count
            elapsed  = time.time() - t0
            print(f"  step={step+1:5d}/{n_steps}  loss={avg_loss:.4f}  "
                  f"bpd={avg_loss/math.log(2):.4f}  elapsed={elapsed:.1f}s")
            loss_window = 0.0
            loss_count  = 0

    elapsed_train = time.time() - t0
    print(f"  Training done in {elapsed_train:.1f}s  "
          f"({n_steps / elapsed_train:.0f} steps/s)")

    # --- Evaluation ---
    model.eval()

    # Accuracy at t=0.5 (50% masked) averaged over 30 independent mask samples
    acc = eval_masked_accuracy(model, data, t=0.5, n_repeats=30, seed=99)

    # Also compute BPD estimate
    bpd = eval_bpd(model, data, n_t_samples=30)

    # Sampling sanity: sample a sequence and verify it's fully unmasked
    sample = iterative_unmask(model, seq_len=seq_len, n_steps=10, device="cpu", greedy=True)
    assert sample.shape == (1, seq_len), f"Bad sample shape: {sample.shape}"
    assert (sample != config.mask_token_id).all(), "Sample contains MASK tokens"

    print(f"\n  Masked-token accuracy @ t=0.5  : {acc:.4f}  (threshold: > 0.90)")
    print(f"  Estimated BPD                  : {bpd:.4f}")
    print(f"  Sample (first 16 tokens)       : {sample[0, :16].tolist()}")

    assert acc > 0.90, (
        f"SMOKE FAIL: masked-token accuracy {acc:.4f} <= 0.90 after {n_steps} steps. "
        f"Possible causes: LR too low, bug in loss/masking, or model too small."
    )

    print("\n  SMOKE PASS")


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

TESTS = [
    test_masking_process,
    test_loss_finite,
    test_loss_decreases,
    test_sampling_shape,
    test_smoke_gate,
]


def main() -> None:
    print(f"Running {len(TESTS)} tests (device=cpu, no pytest)\n")
    t_global = time.time()
    passed = 0
    failed = 0
    failures: list[str] = []

    for test_fn in TESTS:
        t_start = time.time()
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            msg = f"FAIL  {test_fn.__name__}: {e}"
            failures.append(msg)
            print(f"\n  *** {msg}\n")
        except Exception as e:
            failed += 1
            import traceback
            tb = traceback.format_exc()
            msg = f"ERROR {test_fn.__name__}: {e}"
            failures.append(msg)
            print(f"\n  *** {msg}\n{tb}\n")
        finally:
            t_end = time.time()
            if failed == 0 or (passed + failed) > 1:
                pass  # timing printed inline by smoke gate
        _ = time.time() - t_start  # noqa: F841

    elapsed = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(TESTS)} passed  ({elapsed:.1f}s total)")
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
