"""
MDLM Trainer — masked-diffusion language model training loop with CLI.

Usage examples
--------------
# Smoke-test / CPU overfit (synthetic data, tiny model):
  C:\\Python311\\python.exe scripts\\diffusion_lm\\mdlm_train.py \\
      --config smoke --dataset synthetic --steps 5000 \\
      --device cpu --out .docs/diffusion_lm/smoke

# Self-test mode (runs smoke gate inline, then exits):
  C:\\Python311\\python.exe scripts\\diffusion_lm\\mdlm_train.py --self-test

# Dense 117M WikiText-103 de-risk (GPU, queue for later):
  C:\\Python311\\python.exe scripts\\diffusion_lm\\mdlm_train.py \\
      --config derisk117m --dataset wikitext103 --tokens 300e6 \\
      --device cuda --batch 8 --seqlen 512 \\
      --lr 3e-4 --warmup-steps 2000 \\
      --out .docs/diffusion_lm/derisk117m \\
      --eval-every 10000 --ckpt-every 50000

Kill criterion for de-risk: BPD > 2.5 after 100M training tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Resolve imports regardless of how the script is invoked
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mdlm_model import (
    CONFIGS,
    MDLMConfig,
    MDLMModel,
    apply_mask,
    eval_bpd,
    eval_masked_accuracy,
    iterative_unmask,
    mdlm_simple_loss,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticDataset:
    """
    Synthetic token dataset for CPU smoke testing and overfitting checks.

    Generates a fixed set of random sequences and cycles through them.
    Tokens are integers in [0, vocab_size - 2] (leaving vocab_size - 1 for MASK).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        n_sequences: int = 8,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        # Tokens from [0, vocab_size - 2] so mask_token_id = vocab_size-1 is never a clean token
        self.data = torch.randint(
            0, vocab_size - 1, (n_sequences, seq_len),
            generator=rng, dtype=torch.long, device=device,
        )
        self._idx = 0

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Return next batch of [batch_size, seq_len] tokens, cycling over data."""
        n = len(self.data)
        indices = [(self._idx + i) % n for i in range(batch_size)]
        self._idx = (self._idx + batch_size) % n
        return self.data[indices]


class WikiText103Dataset:
    """
    WikiText-103 streaming dataset for the 117M de-risk GPU run.

    Requires WikiText-103 to be available on disk (HF datasets format or raw text).
    Tokenizes in row batches to keep peak RAM bounded (no single giant string
    allocation).  Token IDs are stored as int32 (4 bytes each) and cast to
    int64 only at batch-sample time.

    Peak RAM: ~2x the corpus token storage (chunks list + concatenated array)
    plus the final int32 tensor.  For the full ~130M-token corpus that is well
    under 2 GB.

    For the GPU de-risk run use:
        --dataset wikitext103 --tokenizer microsoft/bitnet-b1.58-2B-4T-bf16
    """

    def __init__(
        self,
        seq_len: int,
        tokenizer_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16",
        split: str = "train",
        device: str = "cuda",
        local_files_only: bool = True,
        max_corpus_tokens: int = 200_000_000,
        tokenize_batch_rows: int = 2000,
    ) -> None:
        import numpy as np
        from transformers import AutoTokenizer
        from datasets import load_dataset

        self.seq_len = seq_len
        self.device  = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=local_files_only
        )
        print(f"Loading WikiText-103 {split} split...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        print(f"Tokenizing in batches of {tokenize_batch_rows} rows "
              f"(cap: {max_corpus_tokens:,} tokens)...")

        # Accumulate small numpy int32 chunks; one concatenation at the end.
        # This avoids the O(corpus) peak from a single tokenizer.encode() call
        # on the full joined string, which previously required ~8.6 GB.
        chunks: list[np.ndarray] = []
        total_tokens = 0
        pending_rows: list[str] = []

        for row in ds:
            text = row["text"]
            if not text.strip():
                continue
            pending_rows.append(text)
            if len(pending_rows) >= tokenize_batch_rows:
                chunk_ids = self.tokenizer.encode(
                    "\n\n".join(pending_rows), add_special_tokens=False
                )
                chunks.append(np.array(chunk_ids, dtype=np.int32))
                total_tokens += len(chunk_ids)
                pending_rows = []
                if total_tokens >= max_corpus_tokens:
                    break

        # Flush any remaining rows (only if cap not already reached)
        if pending_rows and total_tokens < max_corpus_tokens:
            chunk_ids = self.tokenizer.encode(
                "\n\n".join(pending_rows), add_special_tokens=False
            )
            chunks.append(np.array(chunk_ids, dtype=np.int32))
            total_tokens += len(chunk_ids)

        # Single O(n) concatenation, then trim to cap
        all_ids = np.concatenate(chunks)[:max_corpus_tokens]
        del chunks  # release intermediate arrays immediately

        # Store as int32 torch tensor (4 bytes/token vs 8 for int64).
        # Converted to long on each get_batch call.
        self._tokens = torch.from_numpy(all_ids)  # shares numpy buffer
        self._pos = 0
        print(f"  {len(self._tokens):,} tokens loaded from {split} split "
              f"(cap={max_corpus_tokens:,}).")

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Return [batch_size, seq_len] random chunks from the token stream."""
        L = self.seq_len
        T = len(self._tokens)
        starts = torch.randint(0, T - L, (batch_size,))
        batch  = torch.stack([self._tokens[s : s + L] for s in starts]).long()
        return batch.to(self.device)


class WikipediaCachedDataset:
    """
    Uses the wikimedia/wikipedia dataset already in HF cache.
    Suitable for medium-size CPU or GPU experiments without requiring WikiText-103.
    """

    def __init__(
        self,
        seq_len: int,
        tokenizer_name: str = "microsoft/bitnet-b1.58-2B-4T-bf16",
        lang: str = "20231101.en",
        split: str = "train",
        device: str = "cpu",
        max_articles: int = 10000,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoTokenizer
        from datasets import load_dataset

        self.seq_len = seq_len
        self.device  = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, local_files_only=local_files_only
        )
        print(f"Loading wikimedia/wikipedia ({lang}) from cache...")
        ds = load_dataset("wikimedia/wikipedia", lang, split=split,
                          streaming=False, trust_remote_code=True)
        texts = []
        for i, item in enumerate(ds):
            if i >= max_articles:
                break
            texts.append(item["text"])
        text = "\n\n".join(texts)
        print("Tokenizing...")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self._tokens = torch.tensor(ids, dtype=torch.long)
        self._pos = 0
        print(f"  {len(self._tokens):,} tokens from {max_articles} articles.")

    def get_batch(self, batch_size: int) -> torch.Tensor:
        L = self.seq_len
        T = len(self._tokens)
        starts = torch.randint(0, T - L, (batch_size,))
        batch  = torch.stack([self._tokens[s : s + L] for s in starts])
        return batch.to(self.device)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: MDLMModel,
    optimizer: optim.Optimizer,
    step: int,
    tokens_seen: int,
    metrics: dict,
) -> None:
    """Save model + optimizer state to path/ckpt_{step:07d}.pt and update latest symlink."""
    path.mkdir(parents=True, exist_ok=True)
    ckpt_path = path / f"ckpt_{step:07d}.pt"
    torch.save(
        {
            "config":      model.config.as_dict(),
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "step":        step,
            "tokens_seen": tokens_seen,
            "metrics":     metrics,
        },
        ckpt_path,
    )
    # Update 'latest' pointer
    latest = path / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(ckpt_path.name)
    except OSError:
        # Windows may not allow symlinks without special privileges — just copy
        import shutil
        shutil.copy2(ckpt_path, latest)
    print(f"  [ckpt] saved {ckpt_path}")


def load_checkpoint(
    path: Path,
    model: MDLMModel,
    optimizer: Optional[optim.Optimizer] = None,
    device: str = "cpu",
) -> tuple[int, int, dict]:
    """Load latest checkpoint from path. Returns (step, tokens_seen, metrics)."""
    latest = path / "latest.pt"
    if not latest.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    step        = ckpt.get("step", 0)
    tokens_seen = ckpt.get("tokens_seen", 0)
    metrics     = ckpt.get("metrics", {})
    print(f"  [ckpt] resumed from step={step}, tokens_seen={tokens_seen:,}")
    return step, tokens_seen, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def make_lr_schedule(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(
    model: MDLMModel,
    dataset,
    *,
    n_steps: int,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    warmup_steps: int = 1000,
    eval_every: int = 500,
    ckpt_every: int = 5000,
    out_dir: Optional[Path] = None,
    device: str = "cpu",
    start_step: int = 0,
    tokens_seen: int = 0,
    optimizer: Optional[optim.Optimizer] = None,
    t_min: float = 0.05,
    t_max: float = 0.95,
    verbose: bool = True,
) -> dict:
    """
    Main training loop.

    Returns: dict of final metrics.
    """
    model.to(device)
    model.train()

    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.999), eps=1e-8,
        )

    scheduler = make_lr_schedule(optimizer, warmup_steps, n_steps)

    metrics_log: list[dict] = []
    t0 = time.time()
    loss_accum  = 0.0
    n_accum     = 0
    best_bpd    = float("inf")
    seq_len     = model.config.max_seq_len

    for step in range(start_step, n_steps):
        batch = dataset.get_batch(batch_size).to(device)  # [B, L]
        loss, t_val, mask_frac = mdlm_simple_loss(
            model, batch, t_min=t_min, t_max=t_max
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        tokens_seen += batch_size * seq_len
        loss_accum  += loss.item()
        n_accum     += 1

        # ---- periodic eval ----
        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            avg_loss = loss_accum / n_accum
            bpd      = avg_loss / math.log(2)
            lr_now   = scheduler.get_last_lr()[0]
            tok_M    = tokens_seen / 1e6

            # Masked-token accuracy on the last batch (quick proxy)
            model.eval()
            acc = eval_masked_accuracy(model, batch, t=0.5, n_repeats=3)
            model.train()

            entry = {
                "step":        step + 1,
                "tokens_M":    round(tok_M, 2),
                "loss":        round(avg_loss, 4),
                "bpd":         round(bpd, 4),
                "mask_acc":    round(acc, 4),
                "lr":          round(lr_now, 6),
                "elapsed_s":   round(elapsed, 1),
            }
            metrics_log.append(entry)

            if verbose:
                print(
                    f"  step={step+1:6d} | tokens={tok_M:.1f}M | "
                    f"loss={avg_loss:.4f} | bpd={bpd:.4f} | "
                    f"mask_acc={acc:.3f} | lr={lr_now:.2e}"
                )

            if out_dir is not None:
                (out_dir / "metrics.jsonl").open("a").write(
                    json.dumps(entry) + "\n"
                )

            if bpd < best_bpd:
                best_bpd = bpd

            loss_accum = 0.0
            n_accum    = 0

        # ---- periodic checkpoint ----
        if out_dir is not None and (step + 1) % ckpt_every == 0:
            save_checkpoint(
                out_dir, model, optimizer, step + 1, tokens_seen,
                metrics={"bpd": best_bpd}
            )

    final_metrics = {
        "best_bpd":    best_bpd,
        "tokens_seen": tokens_seen,
        "steps":       n_steps,
        "log":         metrics_log,
    }
    if out_dir is not None:
        (out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2))
        save_checkpoint(out_dir, model, optimizer, n_steps, tokens_seen,
                        metrics={"bpd": best_bpd})
    return final_metrics


# ---------------------------------------------------------------------------
# Self-test / smoke gate (inline)
# ---------------------------------------------------------------------------

def run_self_test(device: str = "cpu") -> bool:
    """
    Smoke gate: overfit tiny model on synthetic sequences.
    Returns True if masked-token accuracy > 0.9, False otherwise.
    """
    print("=== MDLM self-test (smoke gate) ===")
    torch.manual_seed(42)

    config = CONFIGS["smoke"]
    model  = MDLMModel(config)
    print(f"  Model: {config.n_layers}L {config.d_model}d {config.n_heads}H "
          f"vocab={config.vocab_size}  params={model.n_params:,}")

    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
        n_sequences=8,
        seed=0,
        device=device,
    )

    print("  Training 5000 steps on 8 synthetic sequences...")
    t0 = time.time()
    metrics = train(
        model, dataset,
        n_steps=5000,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=100,
        eval_every=1000,
        ckpt_every=999999,   # no checkpoints in self-test
        out_dir=None,
        device=device,
        t_min=0.05,
        t_max=0.95,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s  ({5000/elapsed:.0f} steps/s)")

    # Final eval: accuracy on the memorized dataset at t=0.5
    model.eval()
    all_data = dataset.data  # [8, 32]
    acc = eval_masked_accuracy(model, all_data, t=0.5, n_repeats=20, seed=99)
    bpd = eval_bpd(model, all_data, n_t_samples=20)

    print(f"\n  Final masked-token accuracy (t=0.5): {acc:.4f}")
    print(f"  Final estimated BPD:                 {bpd:.4f}")

    # Sampling test
    sample = iterative_unmask(model, seq_len=32, n_steps=10, device=device, greedy=True)
    assert sample.shape == (1, 32), f"Bad sample shape: {sample.shape}"
    assert (sample != config.mask_token_id).all(), "Sample contains MASK tokens"
    assert sample.min() >= 0 and sample.max() < config.vocab_size, "Sample OOV"
    print(f"  Sample shape OK: {tuple(sample.shape)}  tokens in [{sample.min()}, {sample.max()}]")

    passed = acc > 0.9
    status = "SMOKE PASS" if passed else "SMOKE FAIL"
    print(f"\n  {status}  (accuracy={acc:.4f}, threshold=0.90)")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MDLM trainer (masked diffusion language model)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="smoke",
        choices=list(CONFIGS.keys()),
        help="Named model config (smoke/tiny/derisk117m)",
    )
    p.add_argument(
        "--dataset", default="synthetic",
        choices=["synthetic", "wikitext103", "wikipedia"],
        help="Training dataset",
    )
    p.add_argument(
        "--tokenizer", default="microsoft/bitnet-b1.58-2B-4T-bf16",
        help="HF tokenizer name/path (used with wikitext103/wikipedia datasets)",
    )
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Override vocab_size from config (set from tokenizer if None)")
    p.add_argument("--tokens", type=float, default=None,
                   help="Training token budget (e.g. 300e6). Overrides --steps.")
    p.add_argument("--steps", type=int, default=5000,
                   help="Training steps (used if --tokens not given)")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--seqlen", type=int, default=None,
                   help="Sequence length (defaults to config.max_seq_len)")
    p.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--out", default=None, type=Path,
                   help="Output directory for checkpoints + metrics")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument(
        "--self-test", action="store_true",
        help="Run smoke gate and exit (ignores all other flags)",
    )
    p.add_argument(
        "--max-corpus-tokens", type=float, default=200e6,
        help="Cap on tokens loaded from wikitext103 corpus (default 200e6). "
             "Training will repeat/cycle over however many tokens are loaded. "
             "Set lower (e.g. 2e6) for fast validation; 200e6 covers the full "
             "~130M-token WikiText-103 train split.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.self_test:
        passed = run_self_test(device="cpu")
        sys.exit(0 if passed else 1)

    # ---- Build config ----
    config = CONFIGS[args.config]
    if args.seqlen:
        config.max_seq_len = args.seqlen

    # ---- Dataset + tokenizer vocab ----
    if args.dataset == "synthetic":
        if args.vocab_size:
            config.vocab_size    = args.vocab_size
            config.mask_token_id = args.vocab_size - 1
        dataset = SyntheticDataset(
            vocab_size=config.vocab_size,
            seq_len=config.max_seq_len,
            n_sequences=8,
            device=args.device,
        )
    elif args.dataset == "wikitext103":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True
        )
        config.vocab_size    = tok.vocab_size
        config.mask_token_id = tok.vocab_size - 1
        dataset = WikiText103Dataset(
            seq_len=config.max_seq_len,
            tokenizer_name=args.tokenizer,
            device=args.device,
            max_corpus_tokens=int(args.max_corpus_tokens),
        )
    elif args.dataset == "wikipedia":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.tokenizer, local_files_only=True
        )
        config.vocab_size    = tok.vocab_size
        config.mask_token_id = tok.vocab_size - 1
        dataset = WikipediaCachedDataset(
            seq_len=config.max_seq_len,
            tokenizer_name=args.tokenizer,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ---- Step budget ----
    if args.tokens is not None:
        n_steps = int(args.tokens / (args.batch * config.max_seq_len))
        print(f"  Token budget {args.tokens:.0e} → {n_steps:,} steps "
              f"(batch={args.batch}, seqlen={config.max_seq_len})")
    else:
        n_steps = args.steps

    # ---- Model ----
    # Re-apply post_init after possible vocab_size mutation
    config.__post_init__()
    model = MDLMModel(config)
    print(f"Model: {config.n_layers}L {config.d_model}d {config.n_heads}H "
          f"vocab={config.vocab_size}  params={model.n_params:,}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # ---- Resume ----
    start_step  = 0
    tokens_seen = 0
    if args.resume and args.out and (args.out / "latest.pt").exists():
        start_step, tokens_seen, _ = load_checkpoint(
            args.out, model, optimizer, device=args.device
        )

    # ---- Output dir ----
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "config.json").write_text(json.dumps(config.as_dict(), indent=2))

    # ---- Train ----
    metrics = train(
        model, dataset,
        n_steps=n_steps,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        ckpt_every=args.ckpt_every,
        out_dir=args.out,
        device=args.device,
        start_step=start_step,
        tokens_seen=tokens_seen,
        optimizer=optimizer,
        verbose=True,
    )

    print(f"\nDone. Best BPD: {metrics['best_bpd']:.4f}")


if __name__ == "__main__":
    main()
