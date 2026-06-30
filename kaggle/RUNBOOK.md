# Track-M MoTE Grid — Kaggle Offload Runbook

This runbook covers everything you need to run the Track-M MoTE ablation grid across
multiple Kaggle GPU sessions.  Each session handles exactly one grid cell.

## Architecture summary

| Step | What happens |
|------|-------------|
| Open notebook, set CELL_ID | Session owns that cell |
| Cell 1 installs deps | transformers, peft, datasets, accelerate, bitsandbytes |
| Cell 3 clones repo | jamesburton/dotLLM @ issue/trackM-mote |
| Cell 4 runs run_cell.py | train -> eval -> push_results |
| push_results.py | pushes metrics.json + eval.json to kaggle-results branch, marks cell "done" |

KD (kd_weight=0.5) runs on Kaggle because it needs two GPU-resident models.
LM-only runs (kd_weight=0) run locally on the RTX 3060 where VRAM is tighter.

---

## (a) Add GH_PAT as a Kaggle Secret

You need a GitHub Personal Access Token (classic or fine-grained) with **Contents: write** on
`jamesburton/dotLLM`.

1. Go to https://github.com/settings/tokens and create a token with `repo` scope.
2. In Kaggle: open the notebook -> **...** menu -> **Settings** -> **Secrets** ->
   **Add New Secret**.
3. Name: `GH_PAT`, Value: your token string (`ghp_...`).
4. Enable the secret for this notebook.

The notebook reads it in Cell 2 via `UserSecretsClient().get_secret("GH_PAT")` and writes it
to `os.environ["GH_PAT"]` so push_results.py can find it.

---

## (b) Running N sessions in parallel — one cell-id per session

Each Kaggle notebook session maps to exactly one grid cell.  Cells c0-c4 can run in parallel;
c5 must wait for the 4-expert winner.

### Steps for each session

1. Open a **new** Kaggle notebook session (or fork the template notebook).
2. In **Cell 2**, set `CELL_ID = "cN"` (one of c0/c1/c2/c3/c4).
   - Do NOT pick a cell-id another session is already running.
3. Enable **GPU T4 x2** accelerator (Settings -> Accelerator).
4. Enable **Internet** (Settings -> Internet -> On).
5. Enable the **GH_PAT** secret (Settings -> Secrets -> toggle on).
6. Run all 4 cells top-to-bottom.

### Parallel split example (two Kaggle accounts / two sessions)

| Session A | Session B |
|-----------|-----------|
| c0 (4e top1 ternary) | c2 (4e top2 fp) |
| c1 (4e top1 fp) | c3 (4e top1 none) |
| (wait for A+B, pick winner) | c4 (4e top2 none) |
| c5 (8e scale-up) | — |

You can mix cells across sessions freely as long as no two sessions share a CELL_ID.

---

## (c) What each session does and where results land

### During the session

- `run_cell.py` calls `mote_train.py` with the cell's hyperparams plus:
  - `--device cuda --teacher-device cuda` (both on GPU)
  - `--optim adamw8bit` (paged 8-bit AdamW from bitsandbytes)
  - `--checkpoint-every 500` (saves every 500 steps to `.docs/mote/<id>/checkpoint/`)
  - `--out .docs/mote/<id>` (adapter + metrics go here)
- After training, `mote_eval.py` runs on the same adapter.
- `push_results.py` pushes to the `kaggle-results` branch of `jamesburton/dotLLM`.

### Where results land

```
jamesburton/dotLLM  (branch: kaggle-results)
  results/
    c0/
      metrics.json        <- final losses + expert histogram
      eval.json           <- PPL, entropy, router dependence
      mote_config.json    <- hyperparams used
      checkpoint/
        state.json        <- step + tokens_seen (for resume)
    c1/
    ...
  grid_manifest.json      <- cell statuses (pending -> done)
```

---

## (d) Aggregating results after all cells finish

```bash
# Pull the results branch locally
git fetch origin kaggle-results
git checkout kaggle-results

# Quick summary: print eval.json for each cell
for cell in c0 c1 c2 c3 c4; do
    echo "=== $cell ==="
    python -c "import json; d=json.load(open('results/$cell/eval.json')); \
        print(f'  PPL mote={d[\"ppl_mote\"]:.2f} dense={d[\"ppl_dense\"]:.2f} \
delta={d[\"ppl_delta\"]:+.2f}  H_norm={d[\"expert_entropy_H_norm\"]:.3f}')"
done
```

Key metrics in eval.json:
- `ppl_mote` / `ppl_delta` — lower is better; target delta < +0.5 ppl points vs dense
- `expert_entropy_H_norm` — higher is better; flag if < 0.5 (router collapse)
- `router_dependence` — fraction of experts used as argmax; should be > 0.5

Pick the 4-expert winner (best ppl_delta with H_norm > 0.5), then update c5 in
`grid_manifest.json` to match its `shared` and `top_k` before running c5.

---

## (e) Session limit (~9-12 hours) and how resume works

Kaggle GPU sessions are time-limited to roughly 9-12 hours (idle timeout also applies).
With 5M tokens at ~1-2 steps/sec on T4, a full cell run takes about 40-80 minutes, so
a single session comfortably fits multiple cells.

### If a session dies mid-training

Resume is built into the harness:

1. In Cell 2, set `RESUME = True`.
2. `run_cell.py` will:
   - Clone `kaggle-results` branch
   - Copy `results/<id>/checkpoint/` (state.json + adapter_weights.pt) locally
   - Pass `--resume-from .docs/mote/<id>/checkpoint/` to `mote_train.py`
3. `mote_train.py` loads the checkpoint and continues from the saved step.
4. After training completes, push happens normally (overwrites prior partial results).

Note: checkpoints are only pushed at the end of a successful run (via push_results.py).
Mid-training checkpoints live on the Kaggle ephemeral disk.  If the session dies before
push completes, there is no checkpoint to resume from — restart from scratch.

To guarantee checkpoint persistence across session interruptions, add a `--push-adapter`
flag to the push command inside `run_cell.py` and run `push_results.py` manually after
each checkpoint interval.  This is not the default because adapter weights are large (~2 GB).

---

## (e2) T4 / P100 OOM: auto teacher-device selection

`run_cell.py` now **auto-detects GPU VRAM** and picks the teacher device automatically:

| GPU | VRAM | Auto teacher device | KD speed |
|-----|------|---------------------|----------|
| L4 | 24 GB | `cuda` | Fast (on-GPU) |
| A100 | 40/80 GB | `cuda` | Fast (on-GPU) |
| P100 | 16 GB | **`cpu`** | Slower (cross-device) |
| T4 | ~14.6 GB | **`cpu`** | Slower (cross-device) |

Threshold: **< 20 GB → teacher on CPU** to leave room for student + 8-bit optimizer.
A notice is printed when CPU fallback activates:

```
[run_cell] GPU 14.6GB < 20GB -> teacher on CPU to fit (KD slower). For fast KD use an L4 (24GB).
```

**Recommendation: use an L4 (24 GB) accelerator** for fast KD (teacher stays on GPU).
T4 and P100 will work with auto CPU teacher, but KD is noticeably slower.

### Manual override

Pass `--teacher-device {auto,cpu,cuda}` to bypass detection:

```bash
# Force CPU teacher even on L4 (e.g. to save VRAM for a very large batch):
python kaggle/run_cell.py --cell-id c1 --teacher-device cpu

# Force CUDA teacher on T4 if you know it fits (risky):
python kaggle/run_cell.py --cell-id c1 --teacher-device cuda

# Default auto-detection (recommended):
python kaggle/run_cell.py --cell-id c1
```

### OOM safety net

If the train subprocess exits non-zero and the output contains `CUDA out of memory` or
`OutOfMemoryError`, `run_cell.py` automatically **retries once** with `--teacher-device cpu`
and logs a clear message. This catches the case where auto-detection guessed wrong or an
explicit `--teacher-device cuda` was passed but VRAM was tighter than expected at runtime.

---

## (f) Coordination rule — one session per cell-id

The manifest is the claim board.  Protocol:

1. **Claim**: before starting a session for cell cN, check `grid_manifest.json` on the
   `kaggle-results` branch.  If status is `"pending"`, you can claim it.
2. **No in-flight marker**: marking "running" requires a push with PAT from within the
   session.  If two sessions race to the same cell, the later push wins and overwrites
   the earlier one — this is harmless but wasteful.  Coordinate via Discord/chat instead.
3. **Done**: `push_results.py` sets status to `"done"` on completion.  Do not re-run a
   "done" cell unless you intentionally want to overwrite results.
4. **c5 gate**: c5 status is `"blocked"` in `grid_manifest.json` and `run_cell.py` will
   refuse to run it until the status is changed.  After c0-c4 complete: pick the 4-expert
   winner, update `c5.shared` and `c5.top_k` to match, change its status to `"pending"`,
   commit, then start c5's session.

---

## KD-on-Kaggle vs LM-only-on-3060

| Mode | Location | Why |
|------|----------|-----|
| `kd_weight=0.5` (KD + LM) | Kaggle T4 | Teacher + student both on CUDA; teacher alone is ~5 GB |
| `kd_weight=0.0` (LM only) | RTX 3060 12 GB | Single model fits; no teacher needed |

The ablation cells in this harness all use `kd_weight=0.5` (Kaggle).  If you want a
matched LM-only run for comparison, reduce `kd_weight` to 0 and run locally:

```bash
python scripts/lora/mote_train.py --config c1_lmonly --n-experts 4 --top-k 1 \
    --shared fp --layers 24-29 --tokens 5000000 --kd-weight 0 \
    --device cuda --optim adamw8bit --out .docs/mote/c1_lmonly
```
