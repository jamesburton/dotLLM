# Usable 128k continuing sessions on Bonsai-2 (12 GB RTX 3060) — design

Date: 2026-10-21. Status: proposed.

## Intent

One continuing chat / agent session on **Bonsai-2 27B PQ2_0-MTP** (`CudaQwen3HybridDenseTransformerModel`) on an RTX 3060
12 GB should grow turn by turn to **at least 64k tokens and preferably 128k**, with each turn prefilling only what it adds
(the earlier conversation restored from the recurrent-prefix snapshot), so time to first token stays short late in the
session. Every change here should also serve Nemotron-H where the code is shared; Nemotron-only ports (quantized KV,
served-context fit, in-place KV growth with a shrinking expert cache, prefix snapshots) come **after** a review of this
phase's results with the user.

Standing rule from the user for this work: **copying or expanding quantized data (weights or KV) into wider types is a
critical-failure risk** — it blows up VRAM and blocks models or contexts that should fit. Treat any such expansion as a
defect to remove, not a cost to budget around, and state its byte cost in reviews and PRs.

## Measured baseline (dev `1a99d67e`, 2026-10-21, display on the Arc iGPU so the 3060 is otherwise idle)

Fitted window at `serve` start:

| KV | MTP | prefill tile | fitted window | per token |
|---|---|---|---|---|
| q8_0 | on | default | 28,672 | 54 KiB |
| q8_0 | off | default | 54,272 | 38 KiB |
| q4_0 | on | default | 41,984 | 38 KiB |
| q4_0 | off | default | 93,184 | 22 KiB |
| q4_0 | off | `DOTLLM_CUDA_PREFILL_CHUNK=512` | **110,592** | 22 KiB (2,782 MiB free after load) |

q4_0 per-token cost = 18 KiB KV + **4 KiB F16 dequant scratch**. With MTP add 16 KiB (the MTP head's F32 KV, 2 × 8 KiB).

Growing one session (+~8k wikitext tokens per turn, `max_tokens` 64, greedy, q4_0 / no MTP / tile 512 / `--ctx-size 131072`):

| context | cached | TTFT (prefill) | prefill tok/s | decode tok/s | VRAM |
|---|---|---|---|---|---|
| 7.8k | 0 | 51 s | 152 | 20.5 | 8.8 GB |
| 32.9k | 17.0k | 114 s | 140 | 18.8 | 9.5 GB |
| 64.3k | 48.8k | 124 s | 125 | 18.0 | 10.2 GB |
| 104.1k | 89.1k | 137 s | 110 | 16.0 | 11.0 GB |
| 112.0k | **0** | **901 s** | 124 | 14.9 | 11.0 GB |
| 120.7k | — | refused cleanly (KV +192 MiB, 102 MiB usable) | | | |

No spill into shared memory at any point. Scripts: `fitprobe.ps1`, `growth.py` (session scratchpad; copy into
`scripts/` as part of this work, see C5).

Three defects explain most of the gap to the goal:

1. **The snapshot lags one turn.** Every hit reused the prompt from *two* turns back (turn 3 cached 7,822 = turn 1's
   prompt), so each turn re-prefilled ~16k instead of its own ~8k. Cause: a chat prompt ends in the template's
   generation prompt (`<|im_start|>assistant\n`, plus a thinking opener on thinking templates); the next turn renders that
   assistant turn differently, so the previous prompt is never an exact prefix of the next.
   - `TextGenerator.ResolveRecurrentPrefix` (single snapshot): `extendsPrevious` is false, so it falls back to the shared
     prefix with the *previous* prompt, which is only known one turn late.
   - `TextGenerator.RecurrentSnapshots` (tiered, `--prefix-snapshots N`): snapshots at `promptLen - 1`, which includes the
     generation prompt, so it never matches the next turn.
2. **A refused growth with headroom becomes a silent full re-prefill.** At 112k, `EnsureRecurrentPrefixKvCapacity` asked
   for `needed + clamp(needed/8, …, 2048)`, the admission refused, the hit turned into a miss, and the miss path then
   allocated a fresh KV for `needed + small headroom` and re-prefilled all 112k tokens (15 minutes).
3. **Flash prefill expands the whole quantized KV history to F16.** `ResolveKvF16` dequantizes rows `[0, seqKv)` of K and
   V for every attention layer, on every prefill tile, into a shared scratch that grows with the context (4 KiB/token,
   ~0.5 GiB at 128k). Decode already reads quantized KV directly (`attention_flash_mma_decode_gqa_split_kvq.cu`, #956),
   and multi-row verifies do too (#1078). The time cost is small; the VRAM cost is what keeps 128k out of reach.

dev has since gained #1091 (MTP decodes on prefix hits, the first turn is snapshotted), which makes an MTP-on long
session worth supporting, hence C4's `-mtp` form. **The baseline above predates #1091.** Reading the code suggests that
on current dev, with a template whose history diverges from the generation prompt, the first-turn snapshot at
`promptLen - 1` may never advance (the next boundary equals it, and a hit only moves the snapshot ≥ 16 tokens forward) —
inferred, not measured. Step 0 of the plan re-measures the growth run on current dev before any change.

A fourth expansion was found in review: the **MTP head's F32 KV grows by copying** (`CudaMtpState`), so the fit counts it
twice (16 KiB/token = 2 × 8 KiB). It is not quantized data, but it is the same copy-on-grow pattern #1080 removed from the
main KV, and it decides how far `[Nk-mtp]` reaches; see C6.

## Goals

- G1. In a continuing chat session, each turn's cached tokens cover everything up to the start of that turn's new
  content. Agent and chat clients through the OpenAI and Anthropic endpoints both benefit.
- G2. A turn never silently degrades from a prefix hit to a full re-prefill because a growth with headroom was refused;
  it either grows exactly enough or fails with `context_length_exceeded`.
- G3. Flash prefill over q8_0 / q4_0 (and mixed) KV allocates no context-sized F16 copy. On this box q4_0 / no MTP / tile
  512 then fits ≥ 128k (≈135k by arithmetic: (2,782 − 256 − 149) MiB ÷ 18 KiB).
- G4. A model id may carry a window hint — `model[128k]` or `model[128k-mtp]` — and the server chooses the settings that
  reach it, advertises it, and refuses up front when it cannot.
- G5. Repeat the growth run to 128k and record the step table above for both `[128k]` and the largest `-mtp` window that
  fits, in `docs/SERVER.md`.

## Non-goals

- Raw prefill throughput (110–150 tok/s). It is the next limit, but separate work.
- Nemotron-H changes (next phase, after review). Shared engine code (C1, C2, C4) must not be Bonsai-specific.
- Vulkan / HIP quantized-KV prefill. Not a correctness bug, so the cross-backend mandate does not apply; note it.
- Mapping ollama `num_ctx` to the window hint (possible later; mention in docs only if trivial).
- Quantizing the MTP head's F32 KV (16 KiB/token). It is a wide store the user's rule flags; record it as the first
  follow-up because it decides how far `[Nk-mtp]` reaches, but do not do it in this phase unless C3's kernel makes it free.

## Anti-goals (would make this a failure even if it "works")

- Any new full-tensor widen/copy of quantized KV or weights, including in a fallback that the Bonsai shape reaches.
- Spilling into WDDM shared memory instead of refusing.
- Widening a numerical tolerance to make a parity test pass.
- A window hint that advertises a window the server then cannot serve (OOM mid-session).

## Constraints

- Correctness first: greedy output and logits must match the existing paths within the existing tolerances.
- Zero allocations on the hot path; GPU interop rules from `CLAUDE.md` (flat C API, `[LibraryImport]`, PTX per arch;
  `*.ptx` are tracked build artifacts and must be rebuilt for kernel changes).
- Release builds with analyzers on: `-m:1 -nr:false -p:UseSharedCompilation=false -p:EnableSourceControlManagerQueries=false`.
- GPU tests and benches on this box are serialized; wait for VRAM to drain between runs.
- Issue-driven workflow: GitHub issue(s), branch `issue/{n}-…` from `dev`, `(#n)` in commits, PR to `main`, merge into
  `dev` per the repo's dev-branch rules.

## Changes

### C1. Snapshot at the start of the generation prompt (top priority; engine, shared)

Decided:
- The snapshot boundary for a chat request is **the prompt length minus the chat template's generation-prompt tokens**:
  render the same messages with `add_generation_prompt` true and false; the token-count difference is the generation
  prompt. The server (which owns the template) passes that boundary to `TextGenerator` as a hint alongside the prompt.
- Both the single-snapshot path and the tiered path use the hint when present. Without a hint (raw `/v1/completions`,
  a template that cannot render both ways) keep today's behaviour.
- The `extendsPrevious` rule stays for prompts that do extend the previous one exactly. When both apply, the **later**
  boundary wins (exact extension gives `promptLen - 1`, which is longer). The generation-prompt boundary matters when the
  next turn's rendering diverges — e.g. Bonsai's template when the client drops `reasoning_content`, where an empty
  `<think>\n\n</think>` re-tokenizes differently. (When the client sends reasoning back with `preserve_thinking` on, or
  with plain ChatML, dev already extends exactly.)
- The snapshot must be able to advance turn after turn: the #1091 first-turn snapshot and the ≥16-token move-forward
  rule must not pin it at turn 1's boundary. (Builder's call: is the generation-prompt boundary always ≥ 16 tokens
  past the previous snapshot, given that a turn adds a user message?)

Left to the builder: the exact API shape for passing the hint (an optional field on the generation request/options is
the obvious one); whether to tokenize the no-generation-prompt rendering or count the suffix tokens directly — whichever
is exact for templates whose generation prompt merges with the preceding text under BPE (verify with Bonsai's and
Nemotron-H's templates; if a boundary is not a token boundary, step back to the nearest one that is).

Acceptance: a regression test that fails on dev and passes after. It runs a four-turn chat through Bonsai's real chat
template, with the client dropping reasoning so the history diverges. It checks that turn N+1's `cached_tokens` is at
least turn N's prompt length minus its generation prompt, and that the snapshot advances every turn, on both snapshot
paths. A ChatML / reasoning-preserved case asserts no regression (cached is still at least `promptLen - 1` of the previous turn). Real-model
check in C5.

Tiered path scope (my call): the tiered path (`--prefix-snapshots N`) gets C1 and a clean failure, and must stay correct.
Reaching 128k with it is **not** a goal: it allocates a fresh KV per request, and a device-tier snapshot is a second
full copy of the prefix KV, so at long context it needs the host tier. The docs must say so. The default single-snapshot
path is the one that must reach 128k.

### C2. Grow exactly before giving up; never silently re-prefill (engine, shared)

Decided: when growing a hit's KV with headroom is refused, retry the growth at exactly `needed` (capped at the window).
Only if that is refused too does the request fail with `context_length_exceeded` (HTTP 400) — not a miss followed by a
full re-prefill of a long prompt. This applies to the single-snapshot path's grow-on-hit. The tiered path never grows
(see C1); there, a request whose fresh KV cannot be admitted fails with the same context error.

Left to the builder: whether a miss is still acceptable for short prompts (a re-prefill under a few thousand tokens is
cheap); the threshold, if any, must be a named constant with a comment.

Acceptance: a unit test with a fake KV whose `TryGrowTo` refuses above a limit, showing (a) exact-size growth is tried and
the hit is kept, (b) refusal of the exact size surfaces as the context error, not a miss.

### C3. Flash prefill reads quantized KV directly (CUDA kernel; Bonsai now, Nemotron-H later)

Decided:
- Add a quantized-KV variant of `attention_flash_mma_prefill_hd256.cu` for K/V ∈ {q8_0, q4_0} (all four pairs, as the
  decode kernel has), dequantizing tiles in the load stage (registers/shared memory), reusing the block-decode helpers of
  `attention_flash_mma_decode_gqa_split_kvq.cu` rather than duplicating them.
- When both sides are quantized and the shape is eligible, `CudaQwen3HybridDenseTransformerModel` routes prefill to it
  and `ResolveKvF16` is not called; `ContextBytesPerToken` stops counting the scratch on that path, so the served fit
  rises automatically.
- No new full-history widen anywhere on the Bonsai path. Any remaining path that still needs `ResolveKvF16` (ineligible
  shapes, research toggles) keeps working but is listed in the PR with its per-token cost.
- MTP verifies of 9 rows (`MtpCostModel.MaxK = 8` + 1) exceed `PerRowKvqMaxRows = 8`, so today they still reach
  `ResolveKvF16`. They must route to the direct-read path too, either through the new prefill kernel or by raising the
  per-row limit. Builder's choice, stated in the PR.

Left to the builder: tile sizes and the dequant placement; whether one templated kernel or per-pair entry points
(match the decode kernel's convention).

Acceptance: kernel parity against the F16-scratch path within the existing flash-prefill tolerance (no widening), for all
four pairs, a non-degenerate GQA shape, positions offset > 0 and lengths not a multiple of the tile; a model-level test
that prefill over quantized KV allocates no F16 scratch (`DebugKvDequantCallsForTest == 0`); fitted window ≥ 128k on
this box with q4_0 / no MTP / tile 512.

### C4. Window hint in the model id: `model[128k]`, `model[128k-mtp]` (server, shared)

Decided:
- Grammar: `<model>[<size>]` or `<model>[<size>-mtp]`, size `<n>k` (×1024), `<n>m` (×1,048,576) or a plain integer,
  case-insensitive. It works on any model reference a request or `serve` accepts (path, `owner/repo[:quant]`, local
  name, profile). Plain `[Nk]` means **MTP off**; `-mtp` keeps MTP on.
- The variant is its own model id. When loaded, `GET /v1/models` and `/props` report it with its real window: add
  `max_sequence_length` to `ModelInfoDto`, matching `/props`. Unloaded variants are not listed; the hint works on
  demand. (My call.)
- Setting choice, most faithful first, stopping at the first that fits the requested window: KV **F16 → q8_0 → q4_0**
  (both sides; F16 is today's default), then prefill tile default → 512. MTP is set only by the suffix. (Ladder starting at F16 is my call.)
- **The hint wins (user decision).** For a hinted id, the hint's window and MTP choice override server-wide `serve` flags
  (`--ctx-size`, `--no-mtp`, `--mtp`). Ids without a hint keep using the flags. An explicit `--kv-quant-k/-v` is a
  ceiling on fidelity: the ladder may go below it but never above. Rationale: a hinted id must always get exactly the
  window and MTP setting it names, or be refused.
- **Predict before evicting (user decision).** The first time a model is loaded, persist its measured fit (free after
  load, fixed bytes, per-token terms per KV type, MTP and tile, plus the device and driver), as
  `%LOCALAPPDATA%\dotllm\moe-calibration.json` does for MoE. A hinted request is checked against that record before
  anything is unloaded. If it can't fit, refuse with a message naming the largest window each setting reaches, e.g.
  "128k-mtp does not fit: q4_0 + MTP reaches 93k; [128k] (MTP off) fits". A model with no record loads once to measure.
  Re-measure when the device, driver or model file changes.
- **Reconfigure in place (user decision).** Variants that share a weights file share one loaded model. Switching between
  `bonsai`, `bonsai[128k]` and `bonsai[64k-mtp]` keeps the weights resident and only reconfigures KV type, window, prefill
  tile and MTP. It drops the KV and the prefix snapshots, which are tied to the KV layout, and must leave no orphaned
  device buffers (check free VRAM before and after in a test). Residency and keep-alive therefore key by weights, with
  the active variant as state. Builder: decide how a request for a different variant interacts with an in-flight
  request on the same weights. Expected: wait for it, as a model switch does today.
- **Hint at `serve` start that can't fit: exit with an error (user decision)**, naming the largest window each setting
  reaches.
- The chosen settings are logged and reported in `/props`.

Left to the builder: whether choosing settings needs one measured fit plus arithmetic (KV type and MTP change only the
per-token cost and known fixed terms) or re-measuring; it must not reload the weights more than once. Making the prefill
tile a real option instead of only an environment variable is expected. Models without a fit measurement (non-CUDA
hybrid) accept the hint as an explicit `--ctx-size` equivalent with today's warnings.

Acceptance: unit tests for parsing (valid and invalid forms, `-mtp`, units, precedence with explicit flags) and for the
setting ladder against synthetic fit numbers; an endpoint test that `/v1/models` lists the variant with its window.

### C6. MTP head KV grows in place (CUDA; my call, follows the user's no-expansion rule)

The MTP head's F32 KV grows by copying, so old and new storage coexist and the fit counts 16 KiB/token. Grow it in place
with the same CUDA VMM approach #1080 used for the main KV (keeping the copying fallback where VMM is unavailable), and
count 8 KiB/token. By arithmetic, `-mtp` then reaches about 93k on this box (q4_0, tile 512). Quantizing the head's KV,
which would take it further, stays a follow-up.

Acceptance: a test that growing the head's KV allocates no second copy (VMM path) and keeps its rows; the fit's per-token
term drops by 8 KiB with MTP on; MTP greedy parity unchanged.

### C5. Measure and document

- **Step 0, before any change:** re-run the growth table on current dev (#1091 included) to confirm the
  snapshot-advance behaviour and set the true baseline.
- Move `growth.py` / `fitprobe.ps1` into `scripts/` (parameterised: URL, model id, turn size, turn count, output).
- Re-run the growth table to 128k with `bonsai[128k]` and with the largest `-mtp` window that fits; record TTFT, cached,
  prefill/decode tok/s, VRAM and shared memory per step in `docs/SERVER.md`, replacing the stale numbers where they
  conflict.
- Update `docs/SERVER.md` (window hint, snapshot boundary, growth behaviour) and `docs/ATTENTION.md` / `docs/KV_CACHE.md`
  (quantized-KV prefill, the scratch removal and its byte cost).

## Expected outcome on this box (arithmetic, to be confirmed by C5)

- `[128k]`: ~135k fits; each 8k turn prefills ~8k instead of ~16k, roughly halving TTFT (~60–75 s per 8k turn at depth;
  a typical 1–4k agent turn ~10–35 s). Decode ~15–20 tok/s.
- `[64k-mtp]`: fits (q4_0 + MTP ≈ 26 KiB/token after C3 + C6 → ~93k). `[128k-mtp]` will be refused with the
  explanation above until the MTP head's KV is quantized (first follow-up).

## Follow-ups (after review with the user)

1. Quantize the MTP head's F32 KV (decides `[128k-mtp]`).
2. Nemotron-H: q8_0/q4_0 KV, served-context fit, in-place KV growth with the VRAM expert cache shrinking as the KV
   grows, prefix snapshots — measured with the same growth script in steps to 128k.
3. Prefill throughput.
