# Elevated `ncu --set full` re-profile — post-#197/#198, 2026-07-30

Follow-up to `.perf-runs/ncu-2026-07-27/README.md` (the original diagnosis behind issues #199/#200).
#197 (tuned split-KV heuristic) + #198 (GQA-group warp-sharing) landed together via PR #201. This
session re-profiled `attention_f32` decode on the RTX 3060 (elevated PowerShell/UAC, per
`[[ncu-elevation-workflow]]`) to check #199's own stated precondition before starting its FP16
tensor-core rewrite: "implement after #198 lands and the kernel is out of the latency-bound regime."

Full findings and the corrected comparison table are in `docs/CUDA.md`'s Flash Attention Future Work
entry (search for "2026-07-30 re-profile") — that's the authoritative, in-tree location. This README
records the raw capture inventory and the methodology mistakes made getting there, since both are
useful context for whoever picks this up next.

## Methodology, including two real mistakes this session

1. **First attempt** (`attn_f32_post197198.ncu-rep`) used the 2026-07-27 session's `--launch-skip 20
   --launch-count 8` recipe verbatim. Wrong: post-#197/#198 the launch ordering differs, and this
   captured `pq2_0_repack_split_f16`, not `attention_f32` at all.
2. **Second attempt** (`A_baseline_decode.ncu-rep`, `B_gqasplit_decode.ncu-rep`) switched to
   `--kernel-name` filtering, which is more robust — but ran with the default (very shallow,
   `--depth` unset) bench repro, and without checking that `attention_f32`'s grid is literally
   `seq_q * num_heads`. With an 8-token prompt, the first 8 name-matched launches were **prefill**
   (`seq_q=8`, grid=192), not decode (`seq_q=1`, grid=24) — a live instance of the exact
   "verify a flag/kernel is actually wired into the code path you're testing" lesson from the BitNet
   session's GQA-split false-correlation (see `[[bitnet-support]]` memory). Capture B additionally
   found `DOTLLM_ATTN_GQA_SPLIT=1` never actually dispatched `attention_f32_gqa_split_kv` at this
   depth — `ncu`'s own "Available Kernels" list confirmed the kernel never ran a single time — because
   `AttentionGqaSplitMinSeqKv` defaults to 256 and the shallow repro's `seq_kv` never reached it.
3. **Third attempt** (`C_baseline_d512.ncu-rep`, `D_gqasplit_d512.ncu-rep`) — the one that actually
   worked — added `--depth 512` (clears the 256-token gate) and verified **by captured grid shape**,
   not by assumption, which launches were real decode: `C` (`attention_f32`, `--launch-count 80`, no
   flags) came back with launches at three distinct grid shapes (192 = prefill; 6144 = the `--depth`
   seed-fill chunking; 24 = real decode — 32 of the 80 launches), and `D`
   (`attention_f32_gqa_split_kv`, `DOTLLM_ATTN_GQA_SPLIT=1`) came back with all 8 launches at grid
   `(4, 8, 1)` — confirming the runtime heuristic chose an 8-way KV split for this shape (`numKvHeads
   × splitFactor = 4 × 8 = 32` total blocks). Only launches matching the expected decode grid shape
   were used for the comparison table in `docs/CUDA.md`.

## Files (not committed — binary, ~250 MB combined; this README is the durable record)

- `A_baseline_decode.*`, `B_gqasplit_decode.*` — superseded by C/D (shallow-depth mistake, kept for
  the record of what went wrong, not as a data source).
- `C_baseline_d512.ncu-rep` / `C_details_utf8.txt` — 80-launch capture, `attention_f32`, default
  flags, `--depth 512`. Decode-shaped launches: grid `(24, 1, 1)`, 32 of 80.
- `D_gqasplit_d512.ncu-rep` / `D_details_utf8.txt` — 8-launch capture, `attention_f32_gqa_split_kv`,
  `DOTLLM_ATTN_GQA_SPLIT=1`, `--depth 512`. All 8 launches at grid `(4, 8, 1)`.
- `attn_f32_post197198.*` — first (wrong-kernel) attempt, kept for the record only.

## Open question for whoever picks up #199 next

Don't start the FP16 tensor-core rewrite without first checking whether the GQA-split kernel's
now-dominant **CTA-barrier stall** (44.2% of 15.24 cycles, "diverging code paths before a barrier"
per `ncu`) is fixable more cheaply — e.g. by reducing warp divergence in the grouped-warp code before
its `__syncthreads()` calls. This wasn't investigated at the SASS/source level this session (ran out
of scope for one sitting) — treat the barrier hypothesis with the same skepticism #218 gave a
superficially similar `ncu`-suggested hypothesis (which SASS inspection proved wrong) before
committing to either path.
