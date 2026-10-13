# Strix Teacher Dump — Runbook

Offline KD (knowledge distillation) workflow: run the teacher model ONCE on
Strix (128 GB AMD, non-CUDA), cache its top-K logits to disk, then train the
MoTE student on the RTX 3060 reading the cache — no teacher compute during
training.

> **Coordinate before running on Strix.** Strix is the user's active dev box.
> Tell the user which command you intend to run and wait for confirmation that
> the machine is idle before starting the teacher dump (it will use many GB of
> RAM for several hours on a large corpus).

---

## Step 1 — Build the corpus (on the 3060 host, CPU-light)

Run on your local Windows box (no GPU needed — pure tokenisation):

```powershell
python scripts/lora/materialize_kd_corpus.py `
    --mix ja_en `
    --n-seqs 4096 `
    --seq-len 512 `
    --out .docs/kd/corpus_ja_en
```

This writes two files:
- `.docs/kd/corpus_ja_en/ids.npy`   — int32 `[4096, 512]`
- `.docs/kd/corpus_ja_en/meta.json` — tokenizer, seq_len, n_seqs, pad_id, mix, seed

Check stdout: decoded sample rows should show Japanese (CJK characters) and
English sentences interleaved.

**Adjust `--n-seqs` as needed.** 4096 seqs × 512 tokens = 2 M tokens.
For a larger distillation run use 16384 or more.

---

## Step 2 — Copy the corpus to Strix

From a PowerShell on the 3060 host:

```powershell
# scp the corpus dir to Strix (adjust remote path to taste)
scp -r .docs/kd/corpus_ja_en strix:~/dotllm_kd/corpus_ja_en
```

Or with rsync from WSL/Bash:

```bash
rsync -av --progress .docs/kd/corpus_ja_en strix:~/dotllm_kd/corpus_ja_en
```

`ssh strix` is the configured alias (see `OTHER_MACHINES.md` for host details).
Strix is Windows 11 — remote paths use forward slashes with the OpenSSH server.

---

## Step 3 — Run the teacher dump on Strix (CPU)

SSH into Strix and navigate to the repo:

```bash
ssh strix
cd ~/dotllm_kd   # or wherever you put the scripts
```

**Smoke test first (8 seqs, validates end-to-end in minutes):**

```bash
python scripts/lora/strix_teacher_dump.py \
    --corpus ~/dotllm_kd/corpus_ja_en \
    --out    ~/dotllm_kd/cache_ja_en \
    --max-seqs 8 \
    --batch-size 2 \
    --device cpu
```

Check that:
- Model loads without errors (will print `teacher ready: vocab_size=...`)
- `topk_idx.npy` shape = `(8, 512, 64)`, dtype int32
- `topk_logit.npy` shape = `(8, 512, 64)`, dtype float16
- `meta.json` and `progress.json` appear in `~/dotllm_kd/cache_ja_en/`

**Full pass (resumable):**

```bash
python scripts/lora/strix_teacher_dump.py \
    --corpus ~/dotllm_kd/corpus_ja_en \
    --out    ~/dotllm_kd/cache_ja_en \
    --batch-size 4 \
    --device cpu \
    --resume
```

Notes:
- **Resumable.** If the job is interrupted, re-run the same command with
  `--resume`; it reads `progress.json` and skips already-written rows.
- Default model: `microsoft/bitnet-b1.58-2B-4T`. If the ternary weights cause
  transformers issues on Strix, substitute `microsoft/bitnet-b1.58-2B-4T-bf16`
  (the dequantised bf16 variant) via `--model microsoft/bitnet-b1.58-2B-4T-bf16`.
- Default dtype is `--dtype bf16` (halves RAM vs fp32). Modern AMD CPUs
  support bfloat16 compute. If you see NaN outputs, switch to `--dtype fp32`.
- Expected RAM: ~4 GB model (bf16) + ~256 MB per batch of logits.
  Strix (128 GB) has ample headroom.
- Expected throughput: ~10–30 s per batch of 4 seqs on a fast AMD CPU.
  4096 seqs / 4 per batch = 1024 batches; estimate 3–8 hours.

---

## Step 4 — Copy the cache back to the 3060 host

```powershell
scp -r strix:~/dotllm_kd/cache_ja_en .docs/kd/cache_ja_en
```

Or with rsync (skips unchanged files, useful if copying in stages):

```bash
rsync -av --progress strix:~/dotllm_kd/cache_ja_en .docs/kd/cache_ja_en
```

Verify the downloaded files:
```python
import numpy as np, json
idx  = np.load(".docs/kd/cache_ja_en/topk_idx.npy",   mmap_mode="r")
logit = np.load(".docs/kd/cache_ja_en/topk_logit.npy", mmap_mode="r")
meta = json.load(open(".docs/kd/cache_ja_en/meta.json"))
print(idx.shape, idx.dtype)    # (4096, 512, 64) int32
print(logit.shape, logit.dtype) # (4096, 512, 64) float16
print(meta)
```

---

## Step 5 — Train the MoTE student with the cached logits

Pass both the corpus dir and the cache dir to the trainer:

```powershell
python scripts/lora/mote_train.py `
    --config   c1_kd `
    --n-experts 4 --top-k 1 --shared fp `
    --layers   upper-half `
    --tokens   4e8 `
    --kd-weight 0.5 `
    --mix      ja_en `
    --device   cuda `
    --teacher-cache .docs/kd/cache_ja_en `
    --kd-corpus     .docs/kd/corpus_ja_en `
    --out      .docs/mote/c1_kd
```

The trainer reads `ids.npy` and `topk_idx.npy` / `topk_logit.npy` for each
batch, computes top-K KL distillation loss, and never loads the teacher model
on the GPU.

---

## Notes on KD weight for Japanese

The base teacher (`microsoft/bitnet-b1.58-2B-4T`) is a **dense English-centric
model** trained primarily on English chat data. Its Japanese perplexity is high
(~74 on wiki_ja vs ~13 on no_robots), which means the teacher signal is weak
for Japanese tokens.

Recommendations:
- **For the JA-lm_head adaptation pass** (specialising the output vocabulary
  for Japanese), the `--self-kd` path in the trainer is preferred — it distils
  from the same model's own non-MoTE forward rather than from this offline
  cache, avoiding the teacher's JA blind spot.
- **For full-MoTE-capacity KD** (training router + routed experts to match the
  teacher's distribution on mixed content where the teacher is competent),
  this cache is appropriate. Consider lowering `--kd-weight` to 0.2–0.3 for
  the JA-heavy mix, or filtering the cache to EN-only rows.
- `--kd-weight 0.5` is the default and appropriate for the EN-dominant portion
  of the corpus where the teacher is reliable.

---

## File format contract (shared between producer and consumer)

| File | dtype | shape | Notes |
|------|-------|-------|-------|
| `corpus/ids.npy` | int32 | `[N, L]` | Right-padded with `pad_id` |
| `corpus/meta.json` | — | — | `tokenizer, seq_len, n_seqs, pad_id, mix, seed` |
| `cache/topk_idx.npy` | int32 | `[N, L, K]` | Top-K vocab indices |
| `cache/topk_logit.npy` | float16 | `[N, L, K]` | Raw (not softmaxed) logits |
| `cache/meta.json` | — | — | `model_id, K, seq_len, n_seqs, vocab_size` |
| `cache/progress.json` | — | — | `{"next_row": int}` — resume marker |

Alignment: `topk_idx[i, t, :]` = teacher top-K at position `t`, predicting
`ids[i, t+1]` (standard teacher-forcing). Final position (`t = L-1`) predicts
pad and is ignored by the trainer.
