# Speculative Decoding — dotLLM

## Overview

A small **draft model** proposes K candidate tokens; the larger **target model** verifies them in a single forward pass. Achieves 2-3× speedup while (with pipeline-aware acceptance) producing output drawn from the target model's distribution.

## Current Status

Speculative decoding is **greedy-only** in the current implementation. `SpeculativeDecoder`'s constructor rejects `greedy: false` and `TextGenerator` only engages speculative decoding when the sampler is effectively argmax — `Temperature <= 0` and `RepetitionPenalty == 1.0`. Requests that don't meet both fall through to the regular (non-speculative) decode path; they are never sampled from an incorrect distribution.

The probabilistic (modified rejection sampling) path described below requires `q` and `p` to be drawn from the same post-transform distribution the [sampler pipeline](SAMPLING.md) actually samples from (temperature / top-k / top-p / min-p / repetition penalty). The current code computes them from raw softmax over constraint-masked logits, which only coincides with the pipeline for argmax. Tracked in Wave 8 ([issue #121](https://github.com/kkokosa/dotLLM/issues/121)).

## Algorithm

### Draft Phase
Draft model generates K tokens autoregressively (K typically 3-5):
```
for i in 1..K:
  draft_logits = draft_model.forward(token)
  q[i] = softmax(draft_logits)   // draft probability
  token[i] = sample(q[i])
```

### Verification Phase
Target model processes all K candidates in one batched forward pass:
```
target_logits[1..K] = target_model.forward([token[1], ..., token[K]])
p[i] = softmax(target_logits[i])  // target probability for each position
```

### Acceptance (Modified Rejection Sampling)
Accept tokens left-to-right:
```
for i in 1..K:
  r = random_uniform(0, 1)
  if r < min(1, p[i][token[i]] / q[i][token[i]]):
    accept token[i]
  else:
    // Reject: sample corrected token from adjusted distribution
    corrected = sample(normalize(max(0, p[i] - q[i])))
    output corrected, discard tokens[i+1..K]
    break

if all K accepted:
  // Bonus: sample one more token from p[K+1]
  bonus = sample(p[K+1])
  output bonus
```

**Key property**: When `q` and `p` are computed from the same post-transform distribution the sampler pipeline actually draws from, this scheme produces samples from the target model's distribution exactly — not an approximation. See [Current Status](#current-status) for the current limitation: the probabilistic path is not yet pipeline-aware, so the implementation accepts only greedy (argmax) mode today.

## ISpeculativeDecoder Interface

```
ISpeculativeDecoder:
  DraftAndVerify(targetModel, draftModel, kvCacheTarget, kvCacheDraft,
                 pipeline, generatedIds, constraint, position,
                 targetVocabSize, draftVocabSize, numCandidates,
                 outputBuffer) → SpeculativeResult { AcceptedCount, DraftTicks, VerifyTicks, DraftedCount }
```

All buffers are caller-owned or pool-rented — zero per-call heap allocation on the hot path. The `outputBuffer` is a `Span<int>` (backed by a reusable `ArrayPool` rental in `TextGenerator`).

## Draft Model Options

| Type | Description | Trade-off |
|------|-------------|-----------|
| Separate small model | e.g., Llama 1B drafting for Llama 70B | Must share vocab. Extra memory. Best acceptance rate. |
| Layer subset | First N layers of target model | No extra params. Lower acceptance rate. |
| Speculative head | Small MLP trained alongside target | Minimal overhead. Model-specific. |

Draft and target **must share the same base tokenizer** — the acceptance scheme requires comparing probabilities over the same token space. A small vocab size difference (up to 128 tokens) is tolerated; see [Vocabulary Compatibility](#vocabulary-compatibility) below.

## KV-Cache Rollback

Speculated tokens that are rejected need their KV-cache entries invalidated:
- Draft model KV-cache: roll back to pre-speculation position.
- Target model KV-cache: only keep entries for accepted tokens.
- With PagedAttention: simply update the sequence length counter in the block table (blocks are reused, data overwritten on next append).

## Recurrent (GDN) Trunk State Rollback

Position-indexed attention KV-cache is not the only per-sequence state a verify round can touch.
Hybrid architectures with Gated DeltaNet layers (`Qwen3HybridDense` and siblings) carry a
**recurrent trunk state** (`IGdnState`) that has no position addressing — every token forwarded
through the trunk mutates it in place, in call order, regardless of the position label attached to
that call. A batched verify forward issues ALL K drafted tokens before any accept/reject decision
is made, so a rejected token's contribution to that recurrent state has already happened by the
time the KV-cache is rolled back — and unlike the KV-cache, there was historically no way to
address it away (issue [#287](https://github.com/kkokosa/dotLLM/issues/287)).

**Fix**: `IModel` exposes an opt-in checkpoint/restore pair —
`SupportsRecurrentStateCheckpoint` / `CheckpointRecurrentState()` / `RestoreRecurrentState(checkpoint)`.
Both `SpeculativeDecoder` and `MtpSpeculativeDecoder` checkpoint the target model's recurrent state
immediately before a batched verify forward that might get partially rejected. On rejection, they
restore the checkpoint and replay exactly the draft tokens that turned out to be genuinely
accepted (`acceptedCount - 1` of them — the trailing output token is always a corrected/bonus
substitute that was never itself fed through the trunk), bringing the recurrent state to precisely
what it would be had the rejected tokens never been drafted. On full acceptance, no restore is
needed — the state already reflects exactly the accepted history.

Implemented for `Qwen3HybridDenseTransformerModel` (CPU, deep-copies `GdnStateCache` via
`Buffer.MemoryCopy`) and `CudaQwen3HybridDenseTransformerModel` (CUDA, device-to-device via
`cuMemcpyDtoD_v2`) — the two models with a real speculative-decoding call path today (MTP
self-speculation requires `SupportsMtp`, currently only true for this family). Other
`RequiresPerSequenceState` architectures (`Qwen3MoeHybridTransformerModel`, `Mamba3TransformerModel`)
have not implemented the checkpoint pair yet — `SupportsRecurrentStateCheckpoint` defaults to
`false` for them, so they keep prior (uncorrected) behavior rather than silently corrupting an
un-audited model; extending this fix to them is follow-up work if/when they gain a real
speculative-decoding caller.

LFM2 CPU and Vulkan models also implement this checkpoint pair for short-convolution
history (#1028), with independent per-sequence KV+convolution prefix snapshots described
in [KV_CACHE.md](KV_CACHE.md). This does not add an MTP head or per-row snapshots to LFM2.

**An asymmetry between the two decoders, worth knowing before touching either rollback path
again.** `MtpSpeculativeDecoder` forwards `lastToken` via a separate, always-legitimate "catchup"
call BEFORE taking its checkpoint, so on rollback only the accepted draft-token prefix needs
replaying. `SpeculativeDecoder`'s verify batch instead forwards `lastToken` itself as row 0
(`verifyTokens[0]`) — the checkpoint taken immediately before that SAME batched call therefore
predates `lastToken`'s own trunk processing too, so its replay must re-forward `lastToken` FIRST,
then the accepted draft-token prefix. Getting this wrong (replaying only the draft tokens, mirroring
`MtpSpeculativeDecoder`) was caught during development by
`SpeculativeDecoderGdnStateTests` — the resulting state omitted `lastToken`'s contribution entirely
and produced ~1% relative logit error. That test's regression assertion uses a 1e-4 relative
tolerance rather than `MtpSpeculativeDecoderGdnStateTests`' byte-exact one: composing three
checkpoint/restore/re-batch cycles back-to-back leaves a benign float32-ULP-scale residual (IEEE 754
addition is not associative across differently-shaped batched re-computation), four orders of
magnitude below what either real bug produced.

### Per-row recurrent snapshots (issue #473)

Checkpoint + replay costs a second trunk forward on every partially rejected round, plus a full
state copy before every verify. A model that reports `SupportsRecurrentRowSnapshots` instead offers
`ForwardWithRecurrentSnapshots` (a normal forward that also records the recurrent state after each
row `0..S-2`; the state after the last row is the live state) and `RestoreRecurrentStateToRow(n)`.
This is llama.cpp's `n_rs_seq` snapshot ring. `MtpSpeculativeDecoder` uses it when available: it
verifies `[lastToken, d1..dK]` through the snapshot forward, skips the checkpoint, and on a
rejection at `accepted` restores row `accepted`. There is no trunk replay. Both the GDN matrix state and
the conv1d window are restored, the latter from `ConvInput` rows `t+1 .. t+dConv-1`.

**What is stored (#1194): no state at all.** The verify is **read-only** on the recurrent state, the design #1237 used for
Nemotron-H. The verify's scan reads the live GDN state and never writes it, and the conv-state save is skipped. Each row's scan
inputs are recorded, plus the `ConvInput` rows `1 .. S+dConv-2` that every row's conv window lies in. After the decoder decides,
the accepted rows `0..a` are replayed **into the live state in place**: the conv window after row `a`, then the same recurrence
over the recorded rows. A full accept replays every row. Without a `RestoreRecurrentStateToRow` call, the next forward, checkpoint or
prefix capture commits every row. The replay runs the scan's own operations on the same inputs, so the state is **bit-identical**
to what the writing scan (and so plain decoding) leaves. There is no per-row state copy and no pre-verify copy.

History: per-row F32 states (#473; 1047 MiB on Bonsai-2 27B at DFlash K=7), then one pre-verify copy plus row inputs (174 MiB, PR
#1231), now row inputs only.

- **Implemented:** dense `qwen35` on all three backends.
  - CPU `Qwen3HybridDenseTransformerModel`: `GatedDeltaNetScan.Execute(readOnlyState: true)` walks the tokens per head in a
    one-head scratch (`DState²` floats, 64 KiB, pooled) and records each row's delta `β(v − Sᵀk)`.
    `GatedDeltaNetScan.Replay` commits `S = S·g + k⊗d` in place (#842).
  - CUDA `CudaQwen3HybridDenseTransformerModel`: `gdn_scan_multi_f32` takes `write_state` (it replaces the unused snapshot output).
    The commit is `gdn_scan_replay_f32_d128` with the live state as its own source. Each thread owns one column, so in-place is
    safe. It needs the register scan (`DState` 128). Under the opt-in split4 scan, or another head size,
    `SupportsRecurrentRowSnapshots` is false and the decoder uses checkpoint + replay.
  - Vulkan `VulkanQwen3HybridDenseTransformerModel`: the register/LDS scan arms (`rs`, the default, and its `lds_fused` fallback)
    take a `write_state` push constant. The commit is recorded at the head of the next forward's first command buffer (no extra
    submit), as the same multi-token dispatch over the staged rows. The global-state A/B arms (`base`, `fused`, `lds*`) cannot run
    read-only, so they fall back to checkpoint + replay.
  - Still a full state per row: the Vulkan `qwen35moe` and Qwen4Exp hosts (`Q4GdnRowSnapshots`, #1188; #1221 ports this). Their
    `rs` / `lds_fused` arms now accept `write_state`. The CPU Qwen4Exp host replays (#842).
- **Opt-out:** `DOTLLM_MTP_GDN_SNAPSHOTS=0`, or `MtpSpeculativeDecoder.UseRecurrentRowSnapshots`
  in-process. `Replays` and `ReplaysAvoided` count each path.
- **Memory:** per row and GDN layer, the replay inputs in decode's own dtype (F32). CUDA: k, v, g, β. Vulkan: q, k, v, g, β (the
  replay is the shipping scan, which reads q). CPU: k, g, delta. Plus `(S + dConv − 2) × convDim` per layer of conv rows. Grown to
  the largest verify seen and kept. Bonsai-2 27B at DFlash K=7 (8 rows): **~31 MiB** on CUDA (was 174 MiB): 12 MiB of inputs and
  19 MiB of conv rows. Vulkan is ~34 MiB, CPU ~31 MiB. Measured on the 3060: peak dedicated VRAM fell by 140–142 MiB at MTP K=3 and
  DFlash K=5/7.
- **Cost:** a full accept now replays every row (before, the live state was already right). The replay is one register-resident
  pass per layer, so it reads and writes the state once. In exchange the verify never writes the state.
  On Strix Vulkan (8060S), DFlash K=7 medians went 20.61 → 21.09 tok/s and MTP K=3 went 33.99 → 35.71 (3 alternated rounds each), all token-identical.
- **Why not 16-bit snapshots (#1107):** a BF16/F16 snapshot restores a rounded state, so a rejected
  round would leave a different recurrence than plain decode. That breaks greedy token identity, the
  hard gate. Storing inputs instead is exact and smaller than 16-bit states for any K ≥ 1. The live
  state's own precision is #1144.
- **Numerics:** a snapshot comes from inside the K+1-row batch, and a replay is a shorter batch, so
  the two differ at ULP scale (CPU GEMM tiles by n, Vulkan picks its matmul pipeline by width).
  Greedy output is identical. `MtpSpeculativeDecoderGdnStateTests`' byte-exact serial comparison
  is pinned to the replay path for that reason, and `MtpRecurrentRowSnapshotTests` compares the
  two paths within tolerance.
- **Measured (Bonsai 2 27B, Vulkan, gfx1151, same-session order-reversed A/B):** vs plain decode,
  K=2 runs at 1.168x with snapshots and 1.013x with replay; K=3 runs at 1.117x and 0.930x. Round
  time falls from 180 to 156 ms at K=2 and from 232 to 193 ms at K=3. The snapshot writes add about
  4 ms to each verify.

## Constraint Interaction

When constrained decoding is active:
1. **Before speculation**: Clone the constraint state via `IDecodingConstraint.Clone()`.
2. **During drafting**: Each draft token advances the cloned constraint state. Draft must respect the constraint mask (invalid tokens excluded from draft sampling).
3. **On rejection**: Restore constraint state from the clone at the rejection point.
4. **On corrected token**: Advance constraint from the restored state.

## Performance Characteristics

- Speedup scales with **acceptance rate** (how often draft matches target).
- Acceptance rate depends on: model similarity, task difficulty, temperature.
- Typical: 60-80% acceptance at low temperature → 2-3× effective speedup.
- Higher temperature → lower acceptance → less benefit.
- K (candidates per iteration): diminishing returns past ~5. Optimal K depends on acceptance rate.

## Vocabulary Compatibility

Draft and target models must share the same base tokenizer. A small vocabulary size difference (up to 128 tokens) is tolerated — matching llama.cpp's `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`. The extra tokens are typically padding/reserved IDs that never appear in normal generation.

When vocab sizes differ, probability comparison uses the shared range (`Math.Min(targetVocab, draftVocab)`). Tokens beyond the draft's vocab can only be produced by the target model (as corrected or bonus tokens).

| Compatibility | Condition | Status |
|---------------|-----------|--------|
| Exact match | `targetVocab == draftVocab` | Best — no clamping needed |
| Close match | `abs(diff) <= 128` | Supported — shared range comparison |
| Incompatible | `abs(diff) > 128` | Rejected — different tokenizer family |

## CLI & Server Usage

```bash
# CLI: run with speculative decoding
dotllm run model.gguf --speculative-model draft.gguf --speculative-k 5 -p "Hello"

# Interactive chat with speculative decoding
dotllm chat model.gguf --speculative-model draft.gguf --speculative-k 5

# Serve: pass at startup
dotllm serve model.gguf --speculative-model draft.gguf --speculative-k 5

# Serve: select draft model from the web UI's Load Model modal

# CLI: MTP self-speculative decoding — auto-detected from the GGUF, no flag needed
dotllm run qwen3.6-27b-mtp.gguf -p "Hello"

# CLI: opt out of auto-detected MTP (e.g. for exact-timing comparisons or debugging)
dotllm run qwen3.6-27b-mtp.gguf --no-mtp -p "Hello"

# Serve: MTP is ON by default for models with an embedded head (#757) — adaptive gate, concurrency < 5, no external draft.
# It takes the continuous-batch scheduler offline for that model; opt out with --no-mtp (--mtp is accepted but redundant)
dotllm serve qwen3.6-27b-mtp.gguf
dotllm serve qwen3.6-27b-mtp.gguf --no-mtp

# Prompt lookup (n-gram drafting, #1293): no draft model; run, chat and serve
dotllm run model.gguf --prompt-lookup --lookup-draft 8 --lookup-ngram 6 -p "Rewrite this function: ..."
dotllm serve model.gguf --prompt-lookup
```

`--draft-model` and `--draft-tokens` are accepted as aliases of `--speculative-model` and
`--speculative-k` on `run`, `chat`, and `serve`, and by the standalone server's argument
parser (`ServerOptions.Parse`). `ServerOptions.SpeculativeModel` / `SpeculativeCandidates`
can also be bound from `appsettings.json` by a host that binds `ServerOptions` from
configuration. Vocabulary compatibility (above) is validated at startup with a clear error
on all three surfaces.

Note: when a draft model is configured, `serve` uses the single-request `TextGenerator`
path — the continuous-batch scheduler does not support draft models yet and is not started.

The serve UI shows three-state compatibility feedback when selecting a draft model:
- **Green**: exact vocab match
- **Yellow**: compatible (within 128-token tolerance)
- **Red**: incompatible (different tokenizer family)

Speculative metrics (`acceptance rate`, `drafted`, `accepted`) appear in the stats bar and the hover card.

## Prompt-lookup (n-gram) drafting (issue #1293)

`--prompt-lookup` (alias `--lookup`) drafts without a draft model: the longest suffix of prompt + output (up to
`--lookup-ngram`, default 6, down to 2 tokens) is looked up in earlier context, and the up to `--lookup-draft` (default 8)
tokens that followed its latest occurrence are proposed. When the match overlaps the suffix (a periodic run) the
continuation repeats the period. The drafts go through `SpeculativeDecoder`'s verify/accept path unchanged
(`LookupAndVerify` → shared `Verify`), so greedy output is token-identical to plain decode; a lookup miss is a one-row
plain step, with no recurrent checkpoint.

- **Index** (`PromptLookupDrafter`): a hash chain keyed by the 2-gram ending at each position (`head[bucket]` = latest
  position, `prev[i]` = the previous one in the bucket). Append is O(1); a lookup walks at most 64 candidates newest first
  and confirms each by comparing tokens, so bucket collisions never match. Buffers are pooled and sized once to the KV
  capacity; the decode loop allocates nothing.
- **When it engages**: greedy requests with no logprobs, no draft model and no MTP round (both take priority), on a backend
  whose verify forward returns every row's logits (`IModel.MaxAllRowLogitsLength > 1` after `TrySetAllRowLogitsLimit`). The
  dense CUDA model (`CudaTransformerModel`) computes the LM head on the last row only, so lookup stays off there; its
  hybrid models, CPU and Vulkan qualify. `serve` takes the continuous-batch scheduler offline while it is on, as for a draft
  model. Settings: `ServerOptions.PromptLookup`; per-request override not yet.
- **Stats**: `speculative_draft_tokens` / `speculative_accepted_tokens` count drafts and accepted drafts only (the round's
  bonus/correction token is the target's own); this now holds for the draft-model loop too.

**Measured** (greedy, 200 tokens, Llama-3.2-1B-Instruct Q8_0, prompt "rename `items` → `records` in this 15-line Python
function", every run token-identical to plain):

| Strix Halo CPU (16 threads) | decode tok/s | drafted / accepted | acceptance |
|---|---|---|---|
| plain | 58.4–60.3 | – | – |
| lookup K=3 | 99.3–100.0 | 159 / 126 | 0.79 |
| lookup K=5 | 102.9–107.7 | 189 / 141 | 0.75 |
| lookup K=8 (default) | 102.1–108.3 | 252 / 147 | 0.58 |
| lookup K=12 | 97.8–104.7 | 288 / 154 | 0.53 |

A story prompt (prose, little copying): plain 53.6–54.8, K=5 67.5–71.8 (0.69), K=8 68.2–69.9 (0.60). On CUDA (local RTX 3060)
Ternary-Bonsai-2-2B PQ2_0 (`qwen35` hybrid, `--no-mtp`) was token-identical but a **net loss**: plain ~186–191 tok/s (graph
replay) vs ~57–90 at K=8 and ~88–191 at K=4, noisy host. The verify is eager and a rejection restores + replays the GDN
state, which a 2B model's ~5 ms step cannot amortise. An adaptive gate (as for MTP) is the follow-up.

**Not yet**: combining with a draft model / MTP / DFlash (azllm's lookup-first chain under the tree root); an adaptive gate;
all-row logits for the dense CUDA verify; a per-request server switch.

## When NOT to Use

- Very short generations (overhead of draft exceeds benefit)
- Very high temperature (low acceptance rate)
- No suitable draft model available
- Memory-constrained (draft model requires additional memory)

## Future Considerations

### Universal Assisted Generation (UAG)

HuggingFace Transformers v4.46.0 introduced UAG, which enables speculative decoding across model families with **different tokenizers**. The approach: draft tokens are decoded to text, re-tokenized with the target tokenizer, and aligned via longest common subsequence. This removes the vocabulary matching requirement entirely but adds tokenization overhead per speculation step. Reported speedups: 1.5-2× across model families. See [HuggingFace blog](https://huggingface.co/blog/universal_assisted_generation).

### Layer-Subset Drafting

Use the first N layers of the target model itself as a draft — no separate model needed. Lower acceptance rate than a dedicated draft model, but zero extra memory and guaranteed vocabulary compatibility.

## Multi-Token Prediction (MTP) Self-Speculative Decoding

Issue #253. MTP ships a lightweight extra prediction head **in the same GGUF checkpoint** as the
target model — a single extra transformer block that predicts several future tokens from the
target's own final hidden state. The target model verifies the drafted tokens in one extra
batched forward pass, exactly like the two-model scheme above, but there is no second `IModel`
and no second full-model KV-cache: only a small additional per-layer KV-cache for the MTP block
itself. First observed for GGUF at scale in Qwen3.5/3.6 (`froggeric/Qwen3.6-27B-MTP-GGUF`,
`ggml-org/Qwen3.6-27B-MTP-GGUF`), sharing `Architecture.Qwen3HybridDense` with PrismML's
Bonsai-27B.

### Research source

Confirmed directly against llama.cpp PR [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673)
("llama + spec: MTP Support", merged 2026-05-16) — the actual diff, not a secondary write-up:
`gguf-py/gguf/constants.py`, `src/models/qwen35.cpp` (`load_block_mtp` / `graph_mtp`),
`src/llama-hparams.{h,cpp}`, `conversion/qwen.py`, and `common/speculative.{h,cpp}`
(`common_speculative_state_draft_mtp`). Key facts pulled from the real source:

- **Tensor naming** reuses the pre-existing DeepSeek-V3 "NextN" GGUF tensor group (unrelated to
  Qwen, defined for MLA/MoE architectures well before this PR): per MTP block `n`,
  `blk.{n}.nextn.eh_proj.weight` `[2·hidden, hidden]`, `blk.{n}.nextn.enorm.weight` `[hidden]`,
  `blk.{n}.nextn.hnorm.weight` `[hidden]`, and optional `blk.{n}.nextn.embed_tokens.weight` /
  `blk.{n}.nextn.shared_head_head.weight` / `blk.{n}.nextn.shared_head_norm.weight` (fall back to
  the trunk's own `token_embd.weight` / `output.weight` / `output_norm.weight` when absent). A new
  hparam key, `{arch}.nextn_predict_layers`, gives the trailing MTP block count (1 for Qwen3.5/3.6
  today — `GGML_ASSERT(nextn_predict_layers == 1)` in the merged source).
- **Layer placement**: `convert_hf_to_gguf.py` sets `block_count = num_hidden_layers +
  mtp_num_hidden_layers` — the MTP block(s) are appended as extra trailing entries in the layer
  stack, using the *same* per-layer tensor conventions as a normal full-attention decoder block
  (`attn_q/k/v/output`, gated QKV, dense SwiGLU FFN) plus the four `nextn.*` tensors wrapped
  around it.
- **MTP block forward** (`graph_mtp`): `h_norm = RMSNorm(trunk_hidden, nextn.hnorm)`,
  `e_norm = RMSNorm(embed(token), nextn.enorm)`, `cur = eh_proj @ concat(e_norm, h_norm)`, then a
  full gated-attention + SwiGLU-FFN decoder block (identical math to a normal Qwen3.5/3.6
  full-attention layer — dotLLM's CPU implementation reuses the trunk's own
  `Qwen3FullAttnWeights`/`ForwardFullAttnBody`-equivalent math directly), then
  `shared_head_norm` → `shared_head_head` (or the trunk fallbacks) → logits. The block's own
  post-FFN hidden state becomes the seed for the *next* autoregressive MTP step.
- **KV-cache**: the MTP block has its **own** KV-cache, separate from the trunk
  (`kv_only_nextn` hparam flag in llama.cpp), sized for just that one block — this is the "small
  additional KV-cache extension" the issue references, not a second full model's cache.
- **Draft loop**: llama.cpp's own merged MTP draft sampler is `top_k = 1` (greedy/argmax) with an
  explicit `// TODO: re-enable top_k == 10 and utilize p_min spec param` — i.e. upstream's own MTP
  implementation is greedy-only today, same restriction this project's existing
  `SpeculativeDecoder` already has (Wave 8 / issue #121).

### Design decision: parallel interface, not an `ISpeculativeDecoder` overload

`ISpeculativeDecoder.DraftAndVerify(targetModel, draftModel, kvCacheTarget, kvCacheDraft, ...)` is
built around two independent models with two independent full-model KV-caches. MTP has no second
`IModel` — the "draft" is a single extra transformer block sharing the target model's own weights
file, and its state (`IMtpState`) is a KV-cache sized for just that one block, not a second
`IKvCache`. Forcing MTP through the two-model signature would mean either threading a fake "draft
model" wrapper whose forward pass needs the *target's* hidden state as an input no `IModel.Forward`
overload exposes as an output, or overloading `kvCacheDraft`'s meaning to sometimes be a full
`IKvCache` and sometimes a tiny per-layer `IMtpState` — both erode the existing interface's clarity
for its actual two-model use case. dotLLM instead adds a **parallel interface**,
`IMtpSpeculativeDecoder.DraftAndVerify(targetModel, kvCacheTarget, mtpState, ...)`, sharing the
same `SpeculativeResult` return shape and the same greedy-only correctness gate as
`SpeculativeDecoder`.

`IModel` gained matching capability members (all default to "off", zero behavior change for every
model that doesn't override them):

```
bool SupportsMtp => false;
IMtpState? CreateMtpState() => null;
ITensor Forward(..., IMtpState? mtpState) => Forward(..., ) // default ignores mtpState
ITensor ForwardMtp(IMtpState state, int tokenId, int position) => throw NotSupportedException;
```

`Forward(..., IMtpState? mtpState)` captures the trunk's pre-final-norm hidden state (one row per
input position) into `mtpState` as a pure side effect — the returned logits are byte-identical to
the non-MTP overload. `ForwardMtp` runs one autoregressive MTP draft step against the model's own
tiny KV-cache.

### The "catchup" forward — a correctness subtlety worth documenting

The MTP head's first draft step needs `mtpState` seeded with the trunk's hidden state *after*
processing `lastToken` (the pairing invariant confirmed against `graph_mtp`: `h` after token `T`
pairs with `embed(T)` to predict `T+1`). Whichever token becomes `lastToken` for a new speculation
round — a corrected token (its argmax differed from what the previous round's verify batch fed it)
or a bonus token (sampled from logits, never fed as an input at all) — has, by construction,
**never been forwarded through the trunk as an input**: no row in the previous round's verify
batch reflects it. `MtpSpeculativeDecoder` therefore starts every round with a single-token
"catchup" forward of `lastToken` (with `mtpState` capture) purely to obtain that hidden state
before drafting can start. This re-forward is safe and idempotent (`IKvCache` is position-indexed,
not an append-cursor — re-writing the same token at the same position is a no-op on cache
contents) but costs one extra single-token trunk forward per round versus a maximally-optimized
implementation that reuses the catchup call's own logits as the verify batch's row-0 comparison
basis — documented here as a known, correctness-first simplification.

> Superseded by #469: there is no catchup forward. One verify forward runs over
> `[lastToken, d1..dK]` with the reference pairing `(h_{p-1}, x_p)`, so the MTP decoder now
> behaves like `SpeculativeDecoder` in the rollback asymmetry described under
> "Recurrent (GDN) Trunk State Rollback": `lastToken` is row 0 of the verify batch.

### MTP head's own KV-cache lifetime — a second documented simplification

llama.cpp's MTP draft context keeps a KV-cache that persists across speculation rounds with
partial rollback. dotLLM's `MtpSpeculativeDecoder` resets `IMtpState`'s own tiny KV-cache to empty
at the start of every round instead: each round's first draft step re-seeds entirely from the
target model's own just-verified hidden state, so the MTP head only ever needs causal
self-attention over the *current* round's own K draft steps, never across rounds. Simpler to
reason about for a first CPU implementation; does not affect correctness.

> Superseded by #469: the head's KV-cache is now persistent and position-indexed, and every trunk
> `Forward` of an MTP sequence (prefill and verify) *absorbs* its tokens into it — see below.

### Absorbing trunk batches into the head (#469, #472)

Slot `p` of the head's KV-cache holds the pair `(h_{p-1}, x_p)`: token `i` of a trunk batch pairs
with the carried row for `i == 0` and captured row `i - 1` otherwise. An absorbed step's output
hidden is never used (the next draft seeds from a trunk row), so the absorb only needs the K/V
rows. Since #472 every backend absorbs a contiguous batch in ONE KV-only pass — embed,
`enorm`/`hnorm`, `eh_proj`, `attn_norm`, K/V projections (n = S), K-norm, RoPE, one slab write — and
skips attention, the O-projection and the FFN entirely. `DOTLLM_MTP_ABSORB_PER_TOKEN=1` restores the
per-token loop. CUDA (#478) batches the per-row kernels but keeps the three projections as S
single-row GEMVs: its `Gemm` routes n > 1 through dequant-to-F16 + HGEMM, which would change the
head's K/V bits on quantized weights, so the CUDA batched absorb stays bit-identical to its
per-token loop (one stream sync per 64-row chunk instead of one per token). Bonsai 2 27B, Vulkan: S=3 (a K=2 verify) 12.9 → 1.7 ms; S=256
prefill 1171 → 5 ms. CPU: S=3 178 → 15 ms.

### The head's KV in the trunk's type (CUDA, #1095)

`CudaMtpState` stores the head's K/V in the trunk's KV type per side (Q8_0 / Q4_0 where the fused quantized-KV decode kernel
takes the head's shape, F16 otherwise; never F32, #1107) and grows it in place with CUDA virtual memory. The absorb and the
draft step compute K/V in a bounded F32 scratch (one row, or a 64-row absorb chunk, as the trunk does), apply K-norm and RoPE, then store them through the trunk's own
write (F32 -> F16 -> `quant_f16_to_q{8,4}_0`). A draft step reads the history in place with the fused kvq decode kernel (or
the F16 tensor-core kernel). Verified on Bonsai-2 27B (`CudaQwen3HybridDenseMtpKvFormatTests`, 600-token prefix, 3 chained
draft steps):

- the fused read's draft logits equal dequant-to-F16-scratch + the F16 kernel **bit for bit** (q4_0, q8_0, k8v4);
- the stored rows equal the F32 head's rows stored through the trunk's write, **byte for byte** (and, for F16, the CPU's own
  half conversion);
- drift against the F32 head (reported, not asserted): max |dlogit| 0.009-0.010 (f16), 0.04-0.06 (q8_0), 0.33-0.60 (q4_0) on
  logits up to ~9.8; the draft argmax matched the F32 head on every step.

The trunk verifies every draft, so greedy output cannot change; only acceptance can (measured below / in the PR).
`DOTLLM_CUDA_MTP_KV=f32` restores F32 storage as the A/B reference arm.

### Correctness (demonstrated, not asserted)

`MtpSpeculativeDecoderTests` proves token-for-token equivalence between MTP self-speculative
decoding and plain greedy decode of the target model alone, using a synthetic MTP-capable mock
model:

- `DraftAndVerify_AllAccepted_MatchesPlainGreedyDecode` — MTP always agrees with the target.
- `DraftAndVerify_WithDisagreements_StillMatchesPlainGreedyDecode` — MTP is *deliberately wrong*
  at half the tokens (forcing rejections every other round); the final accepted sequence still
  exactly matches plain greedy decode of the target function alone. MTP never gets to inject a
  token the target didn't independently agree with — the same guarantee the two-model decoder's
  greedy mode provides.

`Qwen3HybridDenseMtpTests` separately covers the real (if synthetic-GGUF) MTP head forward math:
GGUF detection/loading (`SyntheticQwen35HybridDenseMtpGguf`, built from the confirmed llama.cpp
tensor layout above — no real Qwen3.6-MTP-GGUF fixture is cached locally, see the issue), the
zero-behavior-change guarantee for non-MTP checkpoints, hidden-state-capture-is-a-pure-side-effect
(byte-identical logits with/without `mtpState`), determinism, and the trunk-fallback path when a
checkpoint omits the optional head-local `nextn.*` tensors.

### CLI & server wiring (issue #253, CLI/server pass)

`TextGenerator` gained an `mtpEnabled` constructor parameter (default `true`) dispatching to
`MtpSpeculativeDecoder` in both `Generate` and `GenerateStreamingTokensAsync` — a third decode-loop
branch alongside the existing standard and two-model-speculative loops, gated by
`_mtpEnabled && _draftModel is null && _model.SupportsMtp && !captureLogprobs &&
IsEffectivelyGreedy(options)` (mutually exclusive with an explicit two-model draft, which always
takes priority; same greedy/no-logprobs gate the two-model path already uses). MTP's K reuses
`speculativeCandidates` — the same "candidates per round" knob, just drafted by the model's own head
instead of a second model. Metrics flow through the existing generic `SpeculativeDraftTokens` /
`SpeculativeAcceptedTokens` / `SpeculativeAcceptanceRate` fields on `InferenceTimings` — no new
fields needed, since those were never two-model-specific in the first place.

**`dotllm run`/`dotllm chat` (`RunCommand.cs`): auto-detect, opt-out via `--no-mtp`.** MTP engages
automatically whenever the loaded GGUF carries an MTP head and decoding is effectively greedy — no
flag needed to get the speedup. This mirrors the project's existing precedent for "safe machinery
on by default with an escape hatch" (`--no-prompt-cache`, `--no-warmup` on `serve`) rather than
requiring an opt-in flag like the two-model `--speculative-model` (which is inherently opt-in
because a second model path must be supplied). Two considerations specifically favor auto-detect
here over opt-in: (1) the correctness guarantee is already demonstrated (`MtpSpeculativeDecoderTests`)
so auto-enabling can't silently change output, matching the same "provably behavior-preserving, so
default on" reasoning as prompt-cache/warm-up; (2) the population that could be surprised is
inherently self-selected — only users who deliberately downloaded an MTP-branded GGUF variant are
affected at all, and they almost certainly picked that file *for* the MTP speedup. `RunCommand`
prints a one-line status note (`MTP self-speculative decoding: K=...`) when it engages, same as the
two-model path's own status line.

**`dotllm serve` (`ServeCommand.cs`/`ServerOptions.cs`/`ServerStartup.cs`): opt-in via `--mtp`,
default off** — the opposite default from `run`/`chat`, deliberately. Enabling MTP also takes the
continuous-batch scheduler offline for that model (same restriction the two-model
`--speculative-model` flag already has — MTP's single-sequence self-speculative loop doesn't support
multi-request batching in this iteration; see `ServerStartup.LoadModel`'s `mtpActive` gate next to
the existing `draftModel is null` scheduler condition). Silently trading concurrent-request
throughput for single-request MTP speedup purely because a particular GGUF happened to load would be
a surprising regression for shared server traffic — unlike `run`'s single-shot case, there's a real
downside to weigh, so `serve` requires the explicit flag. `GET /props` reports whether MTP is
actually active for the loaded model (`mtp_active`) — `true` only when both `--mtp` was passed *and*
the checkpoint carries an MTP head.

**Chat completions.** `speculative_draft_tokens` / `speculative_accepted_tokens` /
`speculative_acceptance_rate` on the streaming chunk's timing block (already present for the
two-model path) report MTP's numbers identically when MTP is the mode that ran — no new
wire-format fields needed.

### Status / what's left

- **CPU and CUDA implemented** (`Qwen3HybridDenseTransformerModel`/`CudaQwen3HybridDenseTransformerModel`);
  the partial-offload hybrid split path (`HybridQwen3HybridDenseTransformerModel`) does not support
  MTP (`SupportsMtp` is always `false` there) and falls through to the standard decode loop.
- **Wired into `TextGenerator`/CLI/server** — see above. Not wired into `ChatCommand.cs` with its
  own `--no-mtp` flag (only `RunCommand.cs` per the issue's CLI scope); `ChatCommand` still gets MTP
  automatically via `TextGenerator`'s `mtpEnabled: true` default (no escape hatch there yet — a
  small documented follow-up, not a functional gap).
- **Real-model validation**: `froggeric/Qwen3.6-27B-MTP-GGUF` (Q4_K_M) fetched to
  `~/.dotllm/test-cache` this session — real end-to-end CLI run confirmed via `dotllm run` with the
  new `--no-mtp`-gated auto-detect path (see the issue #253 CLI-wiring session notes for the actual
  generation output and acceptance rate). Full interleaved A/B throughput benchmarking is still
  `MtpBenchProfile`-only.
- **Only `nextn_predict_layers == 1`** is supported, matching llama.cpp's own current QWEN35
  assertion — multi-block MTP is out of scope until a real checkpoint needs it.
### Separate `mtp-*.gguf` head for `qwen35` (issue #1185)

`ggml-org/Qwen3.8-27B-GGUF` converts the trunk with `--no-mtp` and ships the head beside it as `mtp-Qwen3.8-27B-{Q4_0,Q8_0,BF16}.gguf`:
18 tensors, `blk.64.*` (full attention + dense FFN, the same block an embedded head has) plus `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`,
and its own copies of `token_embd` / `output`.

- `Qwen3HybridDenseTransformerModel` (CPU) and `VulkanQwen3HybridDenseTransformerModel` implement `IMtpHeadAttachable`, so the existing
  `mtp-*.gguf` auto-detect (`Qwen4ExpMtpHeadResolver`), `--mtp-head` and the server attach it. The head file is validated against the trunk
  (`ValidateMtpHeadFile`: arch, `nextn_predict_layers == 1`, `block_count == trunk + 1`, hidden / heads / KV heads / head dim / FFN width) and
  loaded through the same `LoadMtpHeadIfPresent` as an embedded head, so draft, absorb and verify are unchanged.
- **No weight duplication:** the file's `token_embd` / `output` copies are never read or uploaded (the trunk's are used); the head's own
  projections stay in their stored quant (CPU: read from the head file's mapping; Vulkan: uploaded as-is, ~0.4 GiB for the Q8_0 block).
- Synthetic coverage: `SyntheticQwen35HybridDenseMtpGguf.Build(headOnly: true)` splits one seeded checkpoint into trunk + head files; the
  split model drafts bit-identically to the embedded one and speculates to plain greedy (`Qwen3HybridDenseMtpHeadAttachTests`,
  `VulkanQwen3HybridDenseMtpHeadAttachTests`).

**Verify cost: 2..8-row GEMVs on the dense Vulkan model.** The first Strix run accepted 91 % of drafts and still decoded at 0.4x plain:
every projection of a 2..8-row verify fell to the 128x128 coopmat GEMM tile, so a 2-row forward cost 4.5 single-row forwards
(Qwen3.8-27B Q4_K_M: 105 ms at 1 row, 477 ms at 2, 512 ms at 4, 480 ms at 8). The dense model now takes the multi-column GEMVs the MoE
host has used since #876 / #902 (Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_S / F16 / F32 on float activations; Q8_0 via the Q8_1 MMVQ multi
kernel), `DOTLLM_VK_SMALLROW_GEMV=0` to disable: 113 ms at 2 rows, 150 ms at 4.

**Exactness needs FP32-activation verify GEMVs.** The MoE host's Q8_0 small-row path (Q8_1 MMVQ multi) quantises the activations, which
the 1-row FP32 decode GEMV does not: on Qwen3.8-27B it flipped a near-tie (`InvalidOperationException` -> `ArgumentException` in a C# prompt).
The dense model therefore uses a new `matmul_q8_0_gemv_multi` (the coalesced 1-row Q8_0 GEMV with NC accumulators, same per-column
reduction order; `VulkanSmallRowKernelTests.Q8F32Multi_EachColumnBitIdenticalToCoalescedGemv` asserts bit equality), and only when the decode
GEMV is the coalesced variant. Every kernel the verify uses is then the 1-row kernel's arithmetic per column.

**Measured (Strix Halo 8060S, Vulkan, `ggml-org` Qwen3.8-27B Q4_K_M + `mtp-Qwen3.8-27B-Q8_0.gguf`, 4 prompts x 128 greedy tokens, CLI `run`,
llama.cpp master `0ae0e1976` `llama-server --spec-type draft-mtp -md <head>`, same prompts, temperature 0):**

| arm | decode tok/s (mean of 4) | per prompt | x plain | peak private / GPU shared |
|---|---|---|---|---|
| dotLLM plain | 10.57 | 10.6 / 10.6 / 10.4 / 10.6 | 1.00 | 23.3 / 22.6 GB |
| dotLLM MTP K=2 | 16.82 | 16.5 / 18.0 / 14.3 / 18.5 | 1.59 | 24.2 / 23.3 GB |
| dotLLM MTP K=3 | 17.14 | 15.9 / 19.9 / 14.0 / 18.8 | 1.62 | 24.3 / 23.5 GB |
| dotLLM MTP K=4 | 16.99 | 15.9 / 19.8 / 14.3 / 18.0 | 1.61 | 24.4 / 23.6 GB |
| llama.cpp plain | 11.00 | 10.9 / 11.0 / 11.0 / 11.0 | 1.00 | 20.6 / 18.6 GB |
| llama.cpp MTP K=2 | 22.21 | 19.4 / 25.3 / 21.5 / 22.7 | 2.02 | 24.7 / 22.1 GB |
| llama.cpp MTP K=3 | 24.27 | 21.4 / 28.5 / 21.8 / 25.5 | 2.21 | 24.8 / 22.2 GB |
| llama.cpp MTP K=4 | 20.56 | 15.8 / 26.7 / 17.5 / 22.2 | 1.87 | 25.0 / 22.4 GB |

- **Greedy output: dotLLM MTP equals dotLLM plain on 12 / 12 runs** (4 prompts x K 2..4, 128 tokens); llama.cpp's MTP equals its plain decode
  on 7 / 12. CPU (Strix, 48 tokens): identical, 4.41 vs 3.73 tok/s.
- The head costs ~0.95 GB (Q8_0 block uploaded as-is, 4 KiB/position F32 head KV, scratch); llama.cpp's head costs ~3.6 GB because it also
  loads the head file's own `token_embd` / `output` copies. dotLLM's plain baseline is ~4 GB above llama.cpp's (a trunk-side gap, not MTP).
- llama.cpp drafts 74 % accepted at K=3. dotLLM's verify (150 ms for 4 rows vs 105 ms decode) and per-step head cost (LM head over 248k rows
  per draft step) are the remaining gap to llama.cpp's 2.2x.

**Verify and decode on the same kernels (issue #1220).** The 4-row verify cost 150 ms against a 105 ms decode, and every verify column only
matched decode by luck: the multi-column GEMVs reduced with `subgroupAdd`, whose order changes with the subgroup width a driver picks for
each pipeline. Intel picks SIMD8/16/32 by register pressure, so the multi kernels differed from decode at NR = 4 on Arc. The row-split GDN
decode scan and the LDS snapshot scan used for verify re-associate the 128-row sums differently. The dense `qwen35` model now decodes and
verifies on the same deterministic kernels:

- **Q4_K / Q8_0 / Q6_K rows-per-workgroup GEMVs** (`matmul_q4_k_gemv_f32_multi`, `matmul_q8_0_gemv_multi`, `matmul_q6_k_gemv_f32_rows`):
  NR output rows per workgroup share the activation loads, there is an explicit fma chain under `precise`, and a fixed shared-memory
  reduction tree (no `subgroupAdd`). A row's bits do not depend on NR or on the column count, so decode (NC = 1, NR = 2) and a verify column
  (NC = 2..8, NR = 4) are bit-identical (`VulkanDenseGemvRowsTests`, also across NR). Q4_K dequantises each weight once per row
  (`w = sc * q - min`), leaving 8 FMAs per column instead of a dot product, a sum of x and two scale FMAs: at 4 columns the kernel went from
  1.5-1.75x a single-row GEMV to 1.2-1.4x. #1193's `Q4_K_R4` (the same row sharing, `subgroupAdd`) is replaced. The knobs
  `DOTLLM_VK_{Q4K,Q8,Q6K}_ROWS` / `_ROWS_MULTI` (1..8, default 2 / 4) and `DOTLLM_VK_{Q4K,Q8,Q6K}_ROWS_DECODE=0` (the old 1-row decode
  kernel) are kept for A/B.
- **Row-split GDN snapshot scan** (`gdn_scan_multi_token_rs_snap_f32`): the snapshot kernel mirrors the decode arm, so every snapshot and
  the final state are bit-identical to token-by-token decode (`VulkanGdnScanSnapshotRowSplitTests`); the scan in a 4-row verify went from
  9.8 to 3.8 ms. The dense model has since replayed instead of snapshotting (#1194, on the decode scan itself, so exact too); the
  `qwen35moe` host still snapshots and now uses this kernel. The numbers below were measured on the snapshot path, before #1194 merged.
- Strix kernel bench (`DOTLLM_GEMV_ROWS_BENCH=1`, `Bench_RowsPerWorkgroup`): at one column every rows kernel streams at ~220-227 GB/s, the
  same as the coalesced decode GEMV it replaces. Q4_K at 4 columns: 278-324 µs vs 225 µs at one column (was 409 µs).

Measured and rejected (Strix): staging the activations in LDS for 4 row groups (slower than the global-load kernel once Q4_K dequantised
once), a fused residual add + RMSNorm (neutral: the per-layer barriers are not the cost), writing the LM head to device memory before
copying it to the host-cached logits buffer (neutral), and NR = 8 (no better than 4).

**Measured (Strix Halo 8060S, Qwen3.8-27B Q4_K_M + `mtp-Qwen3.8-27B-Q8_0.gguf`, same 4 prompts x 128 greedy tokens as above, K = 3,
`DOTLLM_MTP_ADAPTIVE=0`, dotLLM and llama.cpp `0ae0e1976` interleaved A-B-A-B, 2 reps; the host had unrelated background load):**

| arm | tok/s (mean of 8) | per prompt (rep a / rep b) | x plain | peak private / GPU shared |
|---|---|---|---|---|
| dotLLM plain | 10.57 | 10.4/10.2, 10.8/10.8, 10.6/10.5, 10.6/10.5 | 1.00 | 19.2 / 18.4 GB |
| **dotLLM MTP K=3** | **20.82** | 18.3/18.2, 25.3/25.9, 17.2/17.4, 21.7/22.6 | **1.97** | 20.2 / 19.3 GB |
| llama.cpp plain | 10.73 | 10.3/10.9, 10.9/10.5, 10.8/10.7, 10.9/10.8 | 1.00 | 19.6-20.6 / 18.7 GB |
| llama.cpp MTP K=3 | 23.42 | 20.2/21.0, 27.6/26.6, 21.2/21.4, 24.6/24.8 | 2.18 | 23.8-24.8 / 22.4 GB |

- After merging #1194 (dense verify replays instead of snapshotting), interleaved against `dev` at that point (8 runs each): MTP K=3
  **21.65** vs 19.79 tok/s, plain **10.82** vs 10.57, MTP memory 19.8 / 19.0 GB; 8 / 8 token-identical. Against the llama.cpp rows
  above that is 0.92x for MTP and 1.01x for plain decode.
- Before (#1195 build, same session): plain ~10.4, MTP K=3 ~17.1 (1.62x). MTP is now 0.89x llama.cpp and plain decode 0.985x, with less
  memory than llama.cpp in both modes (plain 18.4 vs 18.7 GB GPU shared; MTP 19.3 vs 22.4 GB).
- **Greedy output: dotLLM MTP equals dotLLM plain on 20 / 20 runs** (K 2..4 x 4 prompts, and the 8 K = 3 runs above).
- Round split (`DOTLLM_MTP_TRACE=1`, prompt 0): draft 7.2 ms per step, verify 113 / 121.5 / 156 ms for 3 / 4 / 5 rows against a ~92 ms
  decode, ~2 ms restore and absorb per round. K = 3 stays best (K = 2: 19.0, K = 4: 16.0 tok/s on prompt 0).
- What is left: the 4-row verify is still ~1.3x a decode (Q4_K FFN at 4 columns is the largest share, then the Q8_0 GDN projections), and
  the 5-row verify jumps to 156 ms, which caps K at 3.

### qwen4exp MTP head (issue #820, CPU oracle)

Qwen3.8-Flash-Next (`qwen4exp`) ships its MTP head as a **separate 49-block GGUF** (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 4.1 GB: `blk.48.*`
QSA + 512-expert MoE block, `blk.48.nextn.{eh_proj,enorm,hnorm,hc_head_*}`; the file's own `token_embd`/`output` are not used — the
trunk's are, `mtp_use_dedicated_embeddings = false`). The trunk GGUF has no `nextn_predict_layers`, so:

- `Qwen4ExpTransformerModel.AttachMtpHead(GgufFile | path)` loads the head; `Qwen4ExpMtpHeadResolver` auto-detects an `mtp-*.gguf` in
  the model's directory and the one above it (the HF snapshot layout), `dotllm run --mtp-head <path>` overrides, `--no-mtp` skips it.
- **The head reads the 4-stream residual**, captured BEFORE the head mixer (`hc * hidden` = 10240 floats per position), not the mixed hidden
  state the dense hybrids use. `Qwen4ExpMtpState.HiddenSize` is therefore 10240.
- Cell for token `p` = `(R_{p-1}, token_p)` (the #469 pairing; Strata's "cell i = (R_i, token_{i+1})" shifted by one), at RoPE position `p - 1`.
  Token 0 owns no cell (no preceding residual), so `n` absorbed tokens hold `n - 1` cells.
- Per cell: `R' = eh_proj([enorm(embed(token)) ; hnorm(R[s])])` per stream (`eh_proj = cat(fc_embedding, fc_hidden)`, one GEMM over `4 x 5120`),
  then GR-read -> QSA -> GR-write -> GR-read -> MoE -> GR-write, then the head's own mixer + the trunk's LM head. The next draft step's
  input residual is `R'` (before the mixer). All norm gammas are folded (`1 + w`): measured against the HF tensors, enorm/hnorm included.
- Absorbing a trunk batch computes **K/V only** (the cell's attention, MoE and mixer output would be discarded), which is why prefill
  absorb is cheap. The head's attention is **dense** over all cells with no indexer: exact below 2051 positions, a draft-only
  approximation beyond (Strata does the same), and never a correctness matter since the verify forward decides every token. Top-k reuse
  across MTP steps / an indexer cache for the head belongs to sparse QSA (#819).
- `DOTLLM_MTP_HNORM=stream|joint` selects the `pre_fc_norm_hidden` convention (one RMS per residual stream — llama.cpp's reading, the
  default — vs one RMS over all four streams, Strata's default). Chosen by measurement on the real file (UD-Q4_K_XL trunk + Q8_0 head,
  CPU, K=4, 6 chat/code prompts x 128 greedy tokens): tokens per verify round **3.05 per-stream vs 2.91 joint** (per-stream ahead on 6 of 6).
  Strata's reported band at 4 drafts is 2.4-3.6.
- **Exactness.** Greedy output was IDENTICAL to plain decode in all 12 real-file runs (128/128 tokens each). That is not luck: on the CPU
  the real file's logits for a 5-row forward equal five 1-row forwards bit for bit (max |diff| 0 over all 248320 logits of every row),
  so the verify shape cannot flip an argmax.
- The pinned llama.cpp (`3cf0325`) has no qwen4exp MTP graph (no `nextn` tensors in `qwen4exp.cpp`, no `--spec-type draft-mtp` for this
  architecture), so there is no llama.cpp acceptance number; Strata (`MtpDrafter`) is the algorithmic reference.

#### Vulkan go/no-go: verify cost vs rows (issue #820 stage 2 guard)

MTP only pays if verifying `k + 1` rows costs far less than `k + 1` single-row forwards (the Tev1 refutation was exactly that: linear row
cost). Measured on the real `UD-Q4_K_XL` file, Strix Halo (Radeon 8060S), Vulkan, natural-text tokens, median of 8, full logits for every
row, `VulkanQwen4ExpTransformerModel.Forward` (same session as a Llama-3.1-8B Q4_K_M bench at 32.5 / 36.8 tok/s before / after):

| rows | ms | x 1 row |
|---|---|---|
| 1 | 57 | 1.00 |
| 2 | 166 | 2.9 |
| 3 | 187 | 3.3 |
| 4 | 207 | 3.6 |
| 5 | 225 | 3.9 |
| 6 | 247 | 4.3 |

A 4-draft round verifies 5 rows (225 ms) and, at the CPU-measured 3.05 tokens per round, delivers about 12 tok/s against ~17 tok/s plain
decode: **0.6-0.7x, a slowdown, with the current kernels.** The shape of the curve says why and what would change it:

- Per-half-layer GPU waits (36 GDN / 12 QSA / 48 MoE halves, split mode), ms per forward at 1 row -> 2 rows -> 6 rows: GDN 19.9 -> 48.5 -> 51,
  QSA 5.9 -> 18.7 -> 19, MoE 30.2 -> 79.4 -> 138, head 3.3 -> 6.1 -> 6.2. Every multi-row path is ~2.5x the T=1 fast path at 2 rows
  (a fixed ~100 ms step), after which only the MoE half grows (~15 ms per row). The dense paths are flat in rows.
- It is not the per-half-layer submit splitting (`T > 1` splits: 98 submit-waits instead of 2 costs about 6 ms at 2 rows) and not the host
  n-gram branch (4-10 ms).
- So the lever is a small-T (2..8 rows) fast path for the dense projections (multi-column GEMV), the GDN / QSA token mixers and the MoE
  (expert-grouped, reading each routed expert once per window). With the dense halves held near their T=1 cost and the MoE growing at the
  measured ~15 ms/row, 5 rows would land near 100 ms (1.7x) and K=4 MTP near 1.4x; that is a kernel project of its own and a prerequisite
  to building the Vulkan head, recurrent-state checkpoint/restore and the draft block on that backend.

#### Vulkan qwen4exp MTP (issue #820, stage 2)

`VulkanQwen4ExpTransformerModel` now carries the head (`AttachMtpHead`, auto-attached by `Qwen4ExpMtpHeadResolver` / `dotllm run` through
`IMtpHeadAttachable`), per-row recurrent snapshots and the verify path; the engine's `MtpSpeculativeDecoder` drives it unchanged.

**Design.** The head block runs on a one-layer `VulkanQwen3MoeHybridTransformerModel` (same QSA / 512-expert MoE kernels, own scratch, own dense
K/V cells), the gated-residual plumbing on the trunk model's. `eh_proj` is split once at load into its embedding and hidden halves (row-wise block
copies of the Q8_0 matrix) so the front is two plain matmuls + a broadcast + an add instead of a concatenation. The trunk residual rows never visit
the host: the head state holds `Pending` (the head's next input, before its mixer), `Carry` (`R_{p-1}` for the next absorb, the #469 pairing) and the
last 32 trunk rows; `SeedFromCapturedRow` is a queued device copy resolved by the next command buffer. Absorb (after every trunk forward that carries
the state) writes K/V cells only (attention is computed and discarded; the head's MoE/mixer never run for absorbed cells).

**Verify rows and exactness.** Verify runs `ForwardWithRecurrentSnapshots`: the GDN scan is swapped for its snapshot twin (bit-identical state
evolution), each row's conv window is copied out of `ConvInput`, the n-gram branch records its hash window / conv history per row, and a rejection
restores row `accepted` (a device copy, ~1.5 ms) instead of replaying a forward. **Accept rule (chosen):** the decoder emits the verify row's argmax
for every position (a draft is accepted only when it equals it), so the output is the *verify path's* greedy. The multi-row kernels are not
bit-identical to the 1-row decode (KL 2e-3..1.4e-2, #876), so a near-tie can resolve differently. Measured on the real file (stand-in head, so every
round rejects and restores: the harshest case): identical to plain 1-row greedy in 6 of 10 runs (64 tokens each, 4 short prompts + 1K / 1.9K
text, K = 2-4); the 4 divergences are near-ties of the 1-row logits (gaps 0.14, 0.14, 0.36 and 0.75 logits of a 31-38 logit range, 0.4-2.3 %;
the two 0.14 ones are the same position reached at K = 4 and K = 3). Re-checking with a 1-row forward would cost a full decode per round, so it is
not done.

**Cost on the real file (UD-Q4_K_XL, Strix Halo, same process, plain and speculative interleaved, ratios not absolutes).** Draft step 4.5-5.5 ms
(proxy measured before building: 4.2 ms + ~0.6 ms for `eh_proj` / Q8_0 experts; the 5.5 ms includes host argmax and the submit). Verify
(`snap+mtp`, 5 rows, all logits): 107 ms at ctx < 100, 128 ms at 1K, 137 ms at 1.9K, against a 62 ms 1-row step (1.7x / 2.1x / 2.2x). Absorb 1-3 ms,
restore 1.5 ms, snapshots + MTP bookkeeping +7 ms on the 5-row forward. Round times (K=4 / K=3 / K=2): 128 / - / - ms at short context, 152 / 128 / 110
ms at 1K, 162 / - / 126 ms at 1.9K.

**Projected throughput** (round times above; tokens per round from the CPU oracle's acceptance, 3.05 at K=4, and its geometric fit 2.73 / 2.31 at
K=3 / 2, p = 0.75; the Vulkan head reproduces the oracle's drafts, argmax 16/16 on the synthetic tests, so acceptance transfers):

| context | K=4 | K=3 | K=2 |
|---|---|---|---|
| short | 1.44x | - | - |
| 1K | 1.27x (20.1 vs 15.9 tok/s) | 1.32x | 1.30x |
| 1.9K | 1.19x | - | 1.15x |

The 1.5x target is **not met on this box**. Where it goes (K=4, 1K): 21 ms drafting, 128 ms verifying 5 rows, 3 ms absorb, 2 ms restore, of a 152
ms round that must beat 3.05 x 62 = 189 ms. The verify rows are the shortfall: MoE rows activate mostly different experts (10 of 512), so the expert
read traffic is roughly linear in rows (`gate_up` + `down` +38 ms per 5-row forward in the split-submit profile), and the 12 QSA layers fall off the
split-KV decode kernel onto the prefill flash kernel at `seqQ > 1` (up to +20 ms at 1K, more at 1.9K). Levers not taken: a split-KV attention variant
with per-row query offsets, an expert-grouped verify GEMM, a multi-column Q6_K LM-head GEMV. The adaptive gate (`MtpGatePriorBytes`, re-measured per
request) keeps MTP off whenever it is slower than plain decode.

**An avoidable cost found on the way.** Reading back five logit rows from a device-local scratch went through a freshly allocated staging buffer on
every call: +26 ms per verify (141 -> 115 ms for 5 rows). The all-rows logits scratch is now a host-readback buffer.

**The real head does not fit next to this trunk on this box.** `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` needs ~2.6 GiB of device memory (without its own
embedding / LM head) plus ~0.45 GiB of verify snapshots, and UD-Q4_K_XL (69.3 GiB) fills the 69.8 GiB device-local heap; measured headroom after
the load is under 1 GiB and varies from run to run. A failed upload does not just fail: afterwards *every* allocation fails, including a 113 MiB
sequence state. `AttachMtpHead` therefore refuses before touching the device, with numbers, when trunk + head + snapshots + KV/scratch exceed the
device-local heap (`DOTLLM_VK_ALLOW_OVERCOMMIT=1` overrides); the CLI prints a warning and decodes plain. The residency work (#874, #880) is what
would make the real head fit; until then the round times above come from a **stand-in head** (`AttachStandInMtpHead`, benchmarking only): the
trunk's own QSA layer 3 with a random `eh_proj`, same per-step work as the real head, noise drafts (acceptance 1.0 token per round). Draft quality
and the acceptance rate on Vulkan numerics are therefore *not* measured on the real file; the synthetic 512-expert tests pin Vulkan against the CPU
oracle (draft logits relL2 5e-5 F32 / 1.4e-2 Q8, argmax 16/16, identical accepted counts per round).

#### Making MTP a default win on qwen4exp (issue #900)

Measured on Vulkan / Strix Halo with the real `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` head, plain and speculative runs interleaved in one process
(`benchmarks/Qwen4ExpRowProbe`: `specp`, `spec`, `mcbench`), four short chat/code prompts at 128 tokens, greedy. Ratios are within-process; the box drifts
10-15 % between runs, so absolute tok/s move more than the ratios.

**Where the round time was.** The first re-measure (Q8_0 head as shipped) put a K = 4 round at ~150 ms for 3.65 tokens: verify 100 ms (5 rows), but
**draft steps of 12-14 ms each, not the 4.5-5.5 ms of the stand-in head**. Perturbation (`DOTLLM_VK_MTP_DRAFT_SKIP`, probe `set`) attributed 10.5 ms of
the 13.9 ms step to the head's MoE: the head's routed experts are Q8_0, which the fused single-token chain does not take (it needs Q4_K or IQ gate/up),
so the head ran the general serial MoE path. Re-quantising the head's routed banks at attach (gate/up Q4_K, down Q5_1; `RequantBank`) puts it on the
fused chain: **draft step 13.9 -> 4.1 ms, with the same drafts** (tokens per round 3.26 / 3.10 / 4.23 / 4.10 against 3.26 / 3.02 / 4.23 / 4.10) and
the head 1.0 GiB smaller. UD-IQ4_XS, K = 4: 1.17x -> 1.56x of plain decode (K = 3: 1.38x). `DOTLLM_MTP_HEAD_REQUANT=0` keeps the file's quantisation.

**Adaptive gate (`MtpAdaptiveGate`, `MtpCostModel`, `MtpRoundController`).**
- *Honest accounting.* The gate used draft + verify ticks, which omit the post-verify state restore; it now uses the round's wall time
  (`SpeculativeResult.RoundTicks`). Plain decode is measured forward + sampler. Greedy and sampled requests have separate gates (their acceptance and
  sampler cost differ).
- *Per-round K.* `MtpCostModel` keeps the acceptance of every draft position (running mean, then slow EMA) and the wall time of a round per K; the
  controller warms up at the configured K and then runs the K with the lowest predicted ms per generated token, re-measuring a neighbouring K every
  16 rounds. `IModel.MtpPreferredDraftTokens` (qwen4exp: 4) raises the default depth of 3 when the caller did not choose one.
- *Mid-request fall-back.* After 5 rounds, if the best K is more than 4 % slower per token than plain decode, the request finishes on the plain loop
  (output unchanged: the state after a round is exactly the committed state). The decision needs a plain reference.
- *Cold start (#906).* The first request of a process has no plain sample, so before #906 it was not cut short and a losing configuration paid a whole
  request. A request of at least `MtpRoundController.MinTokensToCalibrate` (48) tokens on a gate with no plain sample now opens with 8 **zero-draft
  rounds** (`MtpSpeculativeDecoder.DraftAndVerify(..., allowZeroDraft: true)` with 0 candidates: no draft, a one-row verify of the last token, one committed
  token, MTP state kept in sync, so the output is exactly plain greedy/sampled decode). The first 2 are cold and not counted; the median of the other 6 is the
  request's own plain reference for the fall-back and is adopted as the gate's first plain sample, so after ONE request both arms are measured (before: a second
  request was spent exploring plain). Warm, a zero-draft round costs what a plain step costs (44.5-45.2 vs 43.9-50.9 ms median, UD-IQ4_XS, 16 rounds each: the head's
  one-row absorb is ~1 ms); the loss on a losing configuration is bounded by those 8 plain-speed tokens plus the 5 rounds before the fall-back. The reference
  errs towards keeping MTP (the absorb, and any cold rounds that slip into the median, make it slower than true plain decode); the re-probes replace it with a true
  plain sample. `DOTLLM_MTP_CALIBRATE=0` turns it off; `DOTLLM_MTP_TRACE=1` prints every calibration round.
  Measured in fresh processes (UD-IQ4_XS, greedy, 192 tokens, box otherwise idle): calibration rounds 43-62 ms (reference 44-48 ms/token; plain decode measures 47);
  the first request took 5.7 s with calibration against 5.2 s without (+0.5 s once, because 8 tokens run at plain speed instead of MTP speed), and the second
  request then runs the faster arm straight away. The fall-back itself is covered by the latency-simulating mock (`TextGeneratorMtpGateTests`: a losing config
  drops MTP after the calibration plus 5 rounds, a winning one keeps its win, both with output identical to plain); on the real files the one mild loser tried
  (ISTA IQ2_XS with the native Q8_0 head at K = 8, 0.9x) is handled by the controller picking a shorter K, not by the fall-back, so it neither gained nor lost
  from the calibration beyond noise. A stall in the first MTP rounds (measured once, while the box was building code: 700 ms per round) can trigger a spurious
  fall-back in the first request; the next re-probe (up to 16 requests later) recovers it.
- *Re-probing.* A losing arm is re-measured every 40 decisions; a clear loss (> 10 %) doubles the interval up to 640.
- *serve attaches the head.* `serve` never attached the separate qwen4exp head (only `run` did), so the default-on serve path never speculated on this
  model. It now attaches an auto-detected `mtp-*.gguf` and logs the outcome; a head that does not fit the device is declined with the numbers and serving
  continues. `--mtp-head <path>` forces it.

**Sampled requests speculate too.** The draft is the head's argmax (a point mass), so exact speculative sampling is cheap: accept a draft with
probability `p(d)` under the prepared distribution (processors, temperature, top-k/p/min-p), resample a rejection from `p` with `d` removed and
renormalised, sample the bonus row from its `p`. Every emitted token is distributed exactly as plain sampling; greedy pipelines reduce to accept-iff-argmax.
`MtpSpeculativeDecoderTests` checks the 27 three-token continuations of a soft Markov target against their analytic probabilities (30 000 runs), and three
mutants (always accept; keep the draft in the residual; drop the history from the penalty processors) each fail a test.

**Greedy identity.** The emitted token is always the verify row's argmax, but the multi-row kernels are not bit-identical to the 1-row decode, so a
near-tie can resolve differently. Across the runs above (4 prompts x K = 2/3/4 per file, 128 tokens each) the MTP text equalled plain 1-row greedy in
23-46 % of the runs and diverged in the rest, always at a position where the two paths' logits for the top two tokens differ by 0.01-0.6 of a 25-35 logit
range (once 1.6, on the 2-bit ISTA IQ2_XS file). Re-checking every committed token with a 1-row forward would cost a decode step per round, which is the
whole saving, so it is not done; sampled output is unaffected in distribution. Why a cheaper fix does not exist is measured under #909 below.

**Greedy identity, investigated (#909): the noise floor is set by the first non-bit-identical op, so partial unification buys nothing.** Tools
(`benchmarks/Qwen4ExpRowProbe`, UD-IQ4_XS, 4 prompts x 17 windows): `rowident <prompt|all> <gen> <rows> <stride> [plain|prefill]` compares the logits of a 5-row
verify forward with the plain 1-row run's logits at the same positions; `layerdiv <prompt> <rows> <skip>` compares every layer's residual of row j of a verify
forward with the same token run 1-row on an identical state and prints the first divergent layer (`=` bit-identical, `u` < 1e-6 relative, `m` < 1e-4, `X` larger).

1. *The state matters as much as the kernel.* Rebuilding the state with one prefill over the generated tokens (the >= 9-row path, unquantised activations) and then
   replaying the tokens 1-row already differs from the plain run by a mean max |dlogit| of 0.74-1.0 (argmax equal 92.6-100 % per row): committed state built by a
   different kernel family is a source of its own. On the plain run's OWN state a 1-row replay is bit-identical (340/340 rows, so the measurement is deterministic).
2. *The 5-row verify forward is not bit-identical to the 1-row forward even on that state*: 0/340 rows bit-identical, mean max |dlogit| 0.54-0.63, argmax equal
   97.1-100 % per row (row 0 included; ~0.9 % of rows flip, and 0.991^128 = 31 % is the observed 25-50 % identity). The garbage-draft runs that "reproduced plain 5
   of 6 times" were too few to see a 1 % flip rate.
3. *First-order sources found by `layerdiv`.* Layer 0 (a GDN layer) already differs by 1.8e-3 relative; with the fused single-token MoE chain and its fused combine
   off (`DOTLLM_VK_Q4E_MOE_FUSED=0 DOTLLM_VK_Q4E_COMBINE_FUSED=0`, i.e. the 1-row path takes the same general MoE path as the rows) it is 1.9e-7: the fused decode
   chain is the big difference. The first QSA layer (3) then jumps to 7.8e-4: the f16 hi/lo coopmat flash attention that every `seqQ > 1` forward takes against the
   f32 split-KV kernel of the 1-row path (also the +20 ms per 1K tokens of verify cost noted above; replaced by a per-row split-KV kernel in #910, below). Switching the 1-row path to the verify path's attention and MoE
   (or the other way round: `DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1 DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1 DOTLLM_VULKAN_DISABLE_SPLIT_DECODE=1`) leaves layers 0-3 (one
   window: 0-7) at ULP level (`u`), and from layer 4-8 on every residual is > 1e-4 away again: the Q8 activation quantisation (#873) turns the first 1-ULP
   difference into a quantisation-step difference. In one window row 1 of the batch differs by > 1e-4 already at layer 0 (the GDN conv state and recurrence run
   across rows; not isolated).
4. *Unifying ops one by one does not move the logits.* Mean max |dlogit| of the verify rows with attention unified 0.51-0.67, plus the fused MoE chain off 0.51-0.68,
   plus fused gated-residual reads, decode multi-row MoE MMVQ and wide Q8 twins off 0.54-0.72: the same ~0.6 as the default (0.54-0.63). Only a configuration in which
   EVERY row-count-dependent op is bit-identical would change the answer, and the ULP-level (`u`) differences that remain in layers 0-7 of the fully switched
   configuration show that this is not one switch away (the multi-column GEMV shaders share the single-row accumulation order but not necessarily its FMA contraction).
5. *Re-scoring borderline rows cannot reach identity either.* A 1-row recompute of a row with a small top-2 gap runs on the KV and GDN state that the verify
   kernels built for the accepted rows before it, so it is the "different state" case of (1), not the plain run. Only re-running the committed tokens 1-row restores
   identity, and that is plain decode.

Decision (rule from #820: default only if the tok/s loss is < 3 % AND identity > 90 % at 128 tokens): neither is reachable cheaply, nothing was changed. Output
quality is not what differs (both paths are equally valid draws from the same near-ties, #873; sampled MTP is exact in distribution). What a real fix would take:
bit-identical single-row and batched variants of an estimated ~15 kernels (fused vs general MoE, split-KV with per-row offsets for attention, GDN conv/scan, the multi- and
single-column GEMV codegens, router top-k, LM head), each with a bit-exactness test, with `layerdiv` as the harness (the target is a string of `=`).

**Verify-row attention: per-row split-KV (issue #910), dense regime.** The attention of a 2..8-row verify forward (it was the f16 hi/lo coopmat flash attention, 13 ms
of a 5-row forward at 1K context and 25 ms at 2K, 1-2 ms of a 1-row forward) is now `attention_f32_splitkv_rows.comp` + `_rows_merge.comp`
(`VulkanSplitKvRowsAttentionKernel`): one workgroup per (row, head, split), row r at `positionOffset + r` attending to its own `positionOffset + r + 1` keys, split
with the split count and chunk boundaries the 1-row decode kernel uses at that length (the shader is the 1-row one with the row dimension added). It serves F32 KV
caches, 2..8 rows (K <= 7), consecutive positions, contexts where the first row has > 16 keys; anything else falls back to flash. `DOTLLM_VK_Q4E_SPLITKV_ROWS=0`
restores the flash path. An F16 KV cache (`DOTLLM_VK_QWEN4EXP_KV=f16`, opt-in) and the sparse QSA regime past 2051 tokens never reach this branch (both go through the QSA
gather kernel for every row count), so they are unchanged.
- *Bit-identity of the kernel itself* is pinned by `VulkanSplitKvRowsAttentionKernelTests`: every row equals the shipping 1-row kernel at that row's length bit for bit
  (6 shapes, including batches whose rows have different split counts and the 256-wide head dim). The wiring (one dispatch per dense QSA layer of a 2..8-row forward,
  none for 1 or 9+ rows or with the switch off, logits on the CPU oracle) is `VulkanQwen4ExpVerifyRowsAttentionTests`.
- *Attention stage* (`stage` with GPU timestamps, 12 attention layers summed, UD-IQ4_XS; flash -> per-row split-KV): 5 rows 7.0 -> 2.6 ms at 512, 13.3 -> 2.9 ms at 1K,
  25.3 -> 5.0 ms at 2K; 2 rows 6.7 -> 1.0, 13.0 -> 1.9, 24.8 -> 2.9 ms. The 1-row decode kernel is 0.8 / 1.1 / 1.9 ms. UD-Q4_K_XL, 5 rows: 16.6 -> 3.1 ms at 1K, 27.8 -> 5.2 ms at 1.9K
  (forward stage sum 102.0 -> 90.0 and 116.0 -> 91.8 ms, trunk only).
- *End to end* (`spec <ctx> 96 4`, real head, plain and MTP interleaved, switch alternated in one process, 2 repeats): verify ms/round 108.4 -> 106.4 at 512 (neutral),
  117.4 / 116.3 -> 107.0 / 109.5 at 1K, 137.8 / 128.1 -> 112.1 / 111.9 at 1.9K on UD-IQ4_XS; UD-Q4_K_XL (attach forced with `DOTLLM_VK_ALLOW_OVERCOMMIT=1`, the canary
  declined by 0.23 GiB with 0.9 GiB held by other processes) 114.9 / 112.3 -> 105.5 / 107.5 at 1K and 131.3 / 129.0 -> 103.6 / 104.1 at 1.9K. Where the tokens per round were
  equal in both arms (Q4_K_XL: 3.65 and 3.80) the MTP rate moved 26.8 / 27.6 -> 28.3 / 28.2 tok/s at 1K (+3-5 %) and 24.7 / 25.0 -> 30.5 / 30.4 tok/s at 1.9K (+22 %), i.e. 1.30 / 1.20 -> 1.46x of
  plain at 1.9K. On UD-IQ4_XS the acceptance sequences differed between the arms (3.65 vs 4.13 tokens/round at 1K, same prompt), so its tok/s ratios (1.35 -> 1.66x at 1K) overstate the
  time effect; use the verify ms/round. The 4-prompt short-context run (ctx <= 70) moves nothing but the trajectories, as predicted (attention is ~1 ms there).
- *Identity and KL: unchanged.* `rowident` (5-row verify vs plain 1-row, 4 prompts x 13 windows x 5 rows, UD-IQ4_XS) with a 1000-token prefix: argmax equal 260/260 in both arms, mean max |dlogit|
  0.68 -> 0.71, mean KL(1-row || verify row) 6.2e-4 -> 7.1e-4; 1700-token prefix: 258/260 both, 0.61 -> 0.64, 6.4e-4 -> 7.6e-4; short prompts: 256/260 both, 0.65 -> 0.63, 5.6e-3 -> 2.3e-3.
  Row KLs are heavy-tailed (2e-4 .. 1.8e-2 between rows), so these differences are noise; there is no improvement in identity either. `layerdiv` (5 rows) shows why: with the fused MoE chain on (the
  default) layer 0, a GDN layer, is already `X` in both arms; with it off the first QSA layer (3) goes `X` -> `u` for rows 0-2 and the first `X` moves from layer 3 to layers 8 / 5 / 4, but
  rows 3-4 stay `uuX` / `uX` (the GDN recurrence across rows is out of this kernel's reach). One of the ~15 kernels now matches its 1-row twin exactly; the logit floor is
  set by the others, as written above. Not extended to the sparse regime: the same per-row offsets would go into `qsa_attention_f32.comp`'s split dispatch.

**The ISTA GSQ-RCO files lost for another reason: 2..8-row dense projections (issue #902).** On ISTA IQ3_XXS / IQ2_XS the 2-row forward cost 100-109 ms
against 41-45 ms for one row (UD-IQ4_XS: 57 ms), so a K = 4 round was 200+ ms and MTP ran at 0.65-0.82x. The split-submit profile put the jump in the dense
projections (gdn_proj +24 ms, gdn_outproj +9, attn_qproj +7): those files keep them in IQ4_XS, IQ3_S, Q4_K, Q5_K and Q6_K, and the #878 multi-column GEMVs
cover only F32 / F16 / Q8_0, so everything else fell to the coopmat / tiled GEMM at any row count from 2. New multi-column F32-in GEMVs
(`matmul_{q4_k,q5_k,q6_k}_gemv_f32_multi`, `matmul_iq4_xs_f32_gemv_multi`, `matmul_iq3_s_f32_gemv_multi`, `MatMulQuantGemvMultiKernel`) decode every weight once
and FMA it against up to 8 activation rows. Two layout lessons from `mcbench`: array accumulators indexed by a specialization-constant loop spilled from 5
columns (IQ3_S 130 -> 430 us), and a thread-per-sub-block layout whose activation loads are 128 bytes apart per lane left the L1 from 5 columns (IQ4_XS
10240x2560: 133 us at n = 4, 582 us at n = 5, 2 ms at n = 8); the IQ shaders are generated (`native/vulkan/gen_iq_gemv_multi.py`) with one scalar accumulator
per column and the IQ4_XS one uses the coalesced lane = K layout (n = 2..8: 46-160 us). Forward cost by rows on the ISTA IQ3_XXS file, 1/2/3/5 rows:
43 / 100 / 108 / 128 ms before, **43 / 56 / 69 / 92 ms after**; IQ2_XS 41 / 109 / 118 / 138 before, 40 / 53 / 64 / 87 after. Parity tests
(`VulkanQuantGemvMultiKernelTests`) hold every column to the single-row GEMV of the same format.

**Per-quantisation result (final build, same-process interleaved plain / MTP, 4 short prompts at 128 tokens, greedy; K = 3 is the library default depth,
K = 4 the qwen4exp preferred depth).** The last column is the 1K-token context run at K = 4.

| file | plain tok/s | MTP K=3 | MTP K=4 | tokens/round K=4 | 1K ctx K=4 | greedy text identical to 1-row (K=3 and K=4 runs) |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL (79 GiB; measured with `DOTLLM_VK_ALLOW_OVERCOMMIT=1`; since #905 the headroom probe decides, see below) | 22.3 | 30.9 (1.38x) | 28.8 (1.33x) | 3.57 | 30.6 vs 21.5 (1.42x) | 1 / 8 |
| UD-IQ4_XS (60 GiB) | 22.7 | 32.3 (1.42x) | 31.1 (1.37x vs the 22.7 baseline; 19.9 in that run's cold first prompt) | 3.67 | 30.2 vs 21.8 (1.39x) | 2 / 8 |
| ISTA IQ3_XXS (44 GiB) | 23.2 | 31.6 (1.36x) | 30.3 (1.35x) | 3.60 | 31.7 vs 22.3 (1.42x) | 2 / 8 |
| ISTA IQ2_XS (37 GiB) | 24.5 | 32.9 (1.34x) | 30.8 (1.30x) | 3.39 | 26.8 vs 23.4 (1.14x) | 4 / 8 |

(Before #900 / #902 the same files measured 1.17x (IQ4_XS, native Q8_0 head), 0.65-0.82x (ISTA IQ3_XXS / IQ2_XS) and "head does not fit" (Q4_K_XL).)
The depth plateau is broad (K = 2 is 1.30-1.41x on the same files), so the controller's choice matters by a few percent; it picks K per round from the
measured acceptance and round cost.

**UD-Q4_K_XL fit and the attach guard (#905).** The re-quantised head adds 2.08 GiB of resident memory (77.86 -> 79.95 GiB, measured), 1.0 GiB less than the
Q8_0 head. The guard used to decline it by 1.0 GiB (82.5 GiB estimated against an 81.4 GiB limit) although the attach is stable with the guard overridden. What
was wrong, and what replaced it:

- *The arithmetic double-counted.* The per-forward scratch (`DOTLLM_VK_PLANNED_ROWS`, ~1 GiB) is allocated at load and already in the live bytes, yet the
  estimate added a flat 1 GiB for it again (and K/V at full capacity without the indexer cache). `Qwen4ExpMtpAttachEstimate` is now the itemised sum of what the
  attach newly allocates: live bytes (trunk + scratch + any KV already grown) + head weights after the re-quantisation + the 4 verify snapshots + K/V and indexer
  growth to the context capacity + the head's own K/V + absorb scratch + 0.25 GiB margin + other processes' GPU memory. For UD-Q4_K_XL: 77.86 + 2.00 + 0.46 +
  0.02 + 0.25 + 1.22 = 81.8 GiB. That still exceeds the assumed limit (0.64 x RAM = 81.37 GiB) by 0.4 GiB, and it should: the limit is not a constant.
- *The limit drifts, and a formula cannot see it.* Allocate-and-touch on a fresh device (`capscan`, `memtest` in `Qwen4ExpRowProbe`) reached 81.43 GiB in one
  session (0.650 x RAM, 3.5 GiB above the Q4_K_XL trunk) and 73.4-74.8 GiB an hour later on the same idle box with nothing of ours running. The only change found
  is that the WSL2 VM's working set had grown from 5.1 to 10.1 GiB (`.wslconfig`: `autoMemoryReclaim=disabled`, so it never gives it back); the drop matches it
  in size, and the limit came back when the VM's working set fell to 3.4 GiB, but that is a correlation, not a measured cause (a 10 GiB host-RAM hog in another process did NOT move the limit). After the drop UD-Q4_K_XL no
  longer loads at all (`VK_ERROR_OUT_OF_DEVICE_MEMORY` at ~70 GiB) and UD-IQ4_XS has 12.4 GiB of headroom instead of ~20. The allocation
  that crosses the limit succeeds; the NEXT submit fails, and from then on **every allocation on that `VkDevice` fails, even after the memory is freed**
  (measured: a 16-row forward after the OOM and after releasing everything). A failed submit on a SECOND `VkDevice` of the same process leaves the first one
  working (measured), which is what makes a safe probe possible.
- *The canary (`VulkanHeadroomCanary`).* Before the head is uploaded, a second device bound to the same GPU allocates what the attach would add (3.0 GiB for the
  real file: itemised bytes + margin + 0.25 GiB slack, in 128 MiB chunks, each followed by a GPU copy over the whole chunk so the driver must make it
  resident) and frees it again (~1 s). The probe sees what no formula can: the current OS limit, this process's other devices and other processes. Success
  attaches (even where the arithmetic would decline); an out-of-memory refuses (even where the arithmetic would fit); only an unavailable probe falls back to
  the arithmetic. `DOTLLM_MTP_HEADROOM_CANARY=0` leaves the arithmetic alone; `DOTLLM_MTP_TRACE=1` prints the itemised sum, the probe and the verdict;
  `DOTLLM_MTP_HEADROOM_EXTRA_MIB` asks the probe for more (a diagnostic that makes the refusal reproducible).
- *Validated against real out-of-memory (UD-IQ4_XS, 61.4 GiB resident, head needs 2.98 GiB, a competing process allocates AFTER the load).* A process that
  allocates before the load does not take memory from a loaded process (its idle allocations are evicted), so the competitor has to arrive late:

  | competitor | headroom the probe found | guard | what the same attach does when forced (`DOTLLM_VK_ALLOW_OVERCOMMIT=1`) |
  |---|---|---|---|
  | none | all 2.98 GiB | attaches | attach +2.08 GiB, 1K prefill and decode fine, 8 fresh-process CLI cycles with and without `DOTLLM_VULKAN_MEM_TRACE` all attached and ran MTP (21.6-24.4 tok/s) |
  | 9 GiB | all 2.98 GiB | attaches | attach and a 1K-token forward fine |
  | 10 GiB | 2.38 GiB | **refuses** | attach succeeds (+2.08), the 1K-token forward then runs out of memory and the device is dead |
  | 11 GiB | 1.25 GiB | **refuses** | the upload itself runs out of memory, device dead |
  | 11.9 GiB | 1.38 GiB | **refuses** | device dead |

  The 10 GiB row is the point of the 3.0 GiB (not 2.1 GiB) request: the head fits, the first real forward after it does not.
  A refusal is a `NotSupportedException` with the numbers; `run`, `chat` (also for an explicit `--mtp-head`) and `serve` log a warning and decode without
  speculation, and the model's device stays usable (a 16-row forward after the refusal ran).
- *UD-Q4_K_XL itself, default settings (no `DOTLLM_VK_ALLOW_OVERCOMMIT`), once the box had its memory back* (the WSL VM's working set fell to 3.4 GiB and the
  limit recovered; an hour earlier the same file could not even load): three fresh-process load + attach + 1K-token prefill + 64-token decode cycles, two without and
  one with `DOTLLM_VULKAN_MEM_TRACE=1`, all attached (probe: 2.98 GiB allocated and touched; an oversized probe measured 3.88 GiB actually available, so 0.9 GiB beyond
  the itemised worst case with K/V at the full 8K context, ~1.7 GiB at a 1K context) and ran MTP at 24.8-25.3 tok/s (19.4 for the very first request of a process, which
  is cold) against 17.8 tok/s plain decode in a fresh `--no-mtp` process on the same prompt. The same file with the diagnostic oversize
  (`DOTLLM_MTP_HEADROOM_EXTRA_MIB=30000`): the probe reached 3.88 GiB of 32.28 GiB, the CLI logged `WARNING: MTP head not attached (NotSupportedException: ...)` with the
  numbers and decoded the same prompt plain at 20.0 tok/s: a clean fallback, not a crash. The itemised arithmetic alone (81.8 GiB against the assumed 81.37 GiB) would
  still have refused in that state, which is why the probe is the authority.

**End to end through `dotllm serve` (default settings, streamed chat, 200 tokens, `--mtp-head` for the ISTA files; tok/s from the final chunk's timings).** The gate
explores plain decode once after the first MTP request and then follows the faster arm:

| file | greedy plain | greedy MTP | sampled (T 0.7, top-k 40) plain | sampled MTP |
|---|---|---|---|---|
| UD-IQ4_XS | 23.1 | 29.1-31.6 (1.26-1.37x) | 22.4 | 26.8-32.4 |
| ISTA IQ3_XXS | 23.3 | 28.9-31.7 (1.24-1.36x) | 23.2 | 24.2-27.1 |
| ISTA IQ2_XS | 24.5 | 33.0-35.7 (1.35-1.45x) | 23.6 | 27.3-28.5 |

**Never a loss: the stress case.** The native Q8_0 head (draft step 12-14 ms, `DOTLLM_MTP_HEAD_REQUANT=0`) on ISTA IQ2_XS makes MTP break even (+5 % greedy,
-8 % sampled). With `DOTLLM_MTP_TRACE=1` the gate shows what it does: greedy request 1 on MTP (37.6 ms/token), request 2 plain (39.7), request 3 back on MTP,
which then fell back to plain mid-request (103 of 199 tokens on MTP: the round model said MTP had slipped 4 % behind plain); the sampled gate measured MTP
43.2 vs plain 40.0 ms/token and stayed on plain. Every request landed within 3 % of plain decode except the single exploratory sampled request (-8 %).

**n-gram table prefetch (#822) on Vulkan.** `PleRowPrefetcher` is called at the top of every forward (before the first embedding record) and, for
MTP, after each draft step with the verify chunk's known prefix `[last token, d1 .. di]`. Cold 1K-token prefill of fresh ids, A/B/B/A in one process:
**7.7-7.9 s with the prefetch vs 14.4-14.6 s without** (-6.7 s, 1.85x), matching the CPU measurement of 6.8 s of demand faults per 1K chunk.
`DOTLLM_VK_Q4E_PREFETCH=0` turns it off (diagnostic).

## DFlash / DFlash2 block-diffusion drafting (issue #1175)

DFlash (z-lab, arXiv 2602.06036) and DFlash2 (Inco AI) drafters are small (5-layer, ~1.9B) models that condition on the
target's intermediate residual streams and draft a whole block of tokens in **one** forward pass. dotLLM loads llama.cpp's
`dflash` GGUF (PR #22105 / #27342 format, e.g. `z-lab/Qwen3.8-27B-DFlash2-GGUF`, the Bonsai-2 refit
`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`) and attaches it to a `qwen35` target with `--dflash <gguf>` (run, chat, serve).

**How it plugs in.** The drafter takes the target's MTP slot (`IDFlashDraftAttachable`), so the unmodified
`MtpSpeculativeDecoder` verifies its drafts — KV rollback, recurrent per-row snapshots, the adaptive gate and streaming all
come for free:

| MTP protocol call | DFlash meaning |
|---|---|
| `Forward(..., mtpState)` | captures the target's layer inputs at `dflash.target_layers` (F16) and injects them into the drafter's context K/V (`fc` → `enc.output_norm` → per-layer `attn_k`/`attn_v`, k-norm, RoPE) |
| `ForwardMtpArgMax(state, last, pos)` | at the context end: runs the block `[last, mask × (B-1)]` once and returns draft 1; later calls serve drafts 2..B-1 |
| `IMtpState.Rollback(len)` | crops the drafter context like the MTP head's KV |

`dflash.target_layers` are **layer inputs** (HF `target_layer_ids + 1`; Qwen3.8: 6/20/34/48/62 = outputs of layers
5/19/33/47/61). The drafter attends non-causally over its context (sliding window 2048) plus the block; DFlash2 adds the
grouped dynamic causal convolutions (kernel 2, group 16) and the rank-256 / top-16 candidate selector, walked greedily from
the anchor. The block embedding and the lm-head are the **target's** (no copy).

**Backends.** The drafter's forward runs on an `IDFlashBackend`. CPU (`Qwen3HybridDenseTransformerModel`) uses the CPU backend.
CUDA (`CudaQwen3HybridDenseTransformerModel`) and Vulkan (`VulkanQwen3HybridDenseTransformerModel`, #1193) run the whole drafter on
the GPU: weights uploaded as stored (1.1 GB for the Q4_K_M drafters; on a UMA Vulkan device imported zero-copy from the mmap), the
context ring in device F16, the trunk's projection / norm / SwiGLU / add kernels and the target's own embedding and lm-head. Their own
kernels are the DFlash2 convolution, the ring write, a fused per-head RMSNorm + NeoX RoPE (cos/sin from a host table in double, as the
CPU) and the block attention over the ring (`dflash_f32.ptx`; `dflash_*.comp` on Vulkan, where the attention is a tiled online softmax
so the 2048-position window fits Strix Halo's 32 KB of shared memory). On Vulkan the verify's tap injection and the next draft record
into one command buffer: a draft is one submit for the layers (read back for the selector) and one for the head. The verify forward
copies the tapped rows to host memory (F16 in the state) on both GPU hosts.

**Measured** (Bonsai-2 27B PQ2_0 abliterated + the `naklitechie` r3 drafter, RTX 3060, 128 tokens greedy, code prompt; the
host CPU was shared with other jobs, hence ranges). Every run was token-identical to plain decode.

| | tok/s | tokens / round | peak VRAM over idle |
|---|---|---|---|
| dotLLM plain | 11.7–14.3 | 1 | 8.15 GB |
| dotLLM MTP K=3 | 19.3–21.9 | 3.63 | 8.6 GB |
| dotLLM DFlash K=7 (GPU drafter) | 31.7–33.5 (15–18 under heavy host load) | 6.68 | 10.3 GB (9.4 GB with `DOTLLM_MTP_GDN_SNAPSHOTS=0`) |
| llama.cpp PrismML fork plain / MTP K=3 | 15.6 / 22.2 | – / 3.6 | 7.6 / 8.3 GB |
| llama.cpp PrismML fork DFlash K=7 | did not fit: card full, 0.38 tok/s | 6.9 | – |

The extra DFlash VRAM was the drafter (1.1 GB) plus the per-row GDN snapshots of a 7-row verify (#473, ~0.9 GB, F32). Since
#1194 a verify keeps only the rows' scan inputs (1047 → 174 → ~31 MiB at K=7), so the DFlash rows above
overstate today's VRAM by ~1.0 GB. On the
CPU (Strix Halo, Bonsai-2) DFlash breaks even (3.3 vs 3.4 tok/s): an 8-row CPU verify costs ~7× a decode step there.

**Strix Halo (8060S, Vulkan, #1193).** 128 tokens greedy, `DOTLLM_MTP_ADAPTIVE=0`, `DOTLLM_DFLASH_DRAFT=K`, four prompts (code,
prose, Q&A, C#); llama.cpp master (0ae0e1976) Vulkan `llama-server` on the same box and files. Every dotLLM run was token-identical to
its plain decode. tok/s per prompt:

| Qwen3.8-27B Q4_K_M | code | prose | Q&A | C# | peak private / GPU-shared |
|---|---|---|---|---|---|
| dotLLM plain | 10.3 | 10.5 | 10.4 | 10.5 | 18.7 / 17.9 GiB |
| dotLLM MTP K=3 (Q8_0 head) | 20.6 | 16.1 | 15.8 | 19.5 | 19.7 / 18.9 GiB |
| dotLLM DFlash K=3 (incoai DFlash2) | 22.5 | 17.0 | 14.3 | 20.3 | 20.3 / 19.5 GiB |
| dotLLM DFlash K=7 | 25.6 | 11.1 | 9.3 | 17.4 | 20.9 / 20.1 GiB |
| llama.cpp plain | 10.8 | 10.9 | 11.0 | 11.0 | 19.2 / 18.3 GiB |
| llama.cpp MTP K=3 | 28.1 | 20.9 | 20.5 | 24.6 | 23.4 / 21.9 GiB |
| llama.cpp DFlash K=3 | 31.5 | 22.2 | 20.3 | 27.3 | 21.8 / 20.6 GiB |
| llama.cpp DFlash K=7 | 36.7 | 13.9 | 14.2 | 21.7 | 22.4 / 21.2 GiB |

DFlash tokens per round (K=3 / 7): code 3.9 / 7.1, prose 2.8 / 2.8, C# 3.5 / 4.6. The gap to llama.cpp is the target's multi-row
verify (shared with MTP), not the drafter: a draft costs ~19 ms at K=3 on Qwen3.8 (layers ~12 ms, the 248k-row Q6_K head ~7 ms).

| Bonsai-2 27B PQ2_0 (naklitechie r3) | code | prose | C# | peak private / GPU-shared |
|---|---|---|---|---|
| dotLLM plain | 23.6 | 23.6 | 23.5 | 8.4 / 7.6 GiB |
| dotLLM MTP K=3 (embedded head) | 35.3 | 25.9 | 37.0 | 8.8 / 8.0 GiB |
| dotLLM DFlash K=3 | 33.2 | 27.2 | 41.3 | 10.0 / 9.1 GiB |
| dotLLM DFlash K=5 | 44.2 | 26.3 | 45.3 | 10.2 / 9.4 GiB |

llama.cpp master cannot load PQ2_0 (the PrismML fork is CUDA/CPU only), so Bonsai-2 has no comparator on Strix. The Q&A prompt ends at
once on this abliterated checkpoint (EOS first) and is left out.

**Exactness.** Greedy output is token-identical to plain decode (`DFlashDrafterTests`, `CudaDFlashTests`,
`VulkanDFlashTests`; real Bonsai-2 27B on the RTX 3060). The selector's greedy draft is a point mass, so sampled decoding
uses the MTP decoder's exact accept/residual rule unchanged.

**Memory.** Drafter weights stay in their stored quant (mmap). Context ring: `layers × (2048 + 2·B) × kvDim × 2 × 2 B`
(42 MB for Qwen3.8-27B-DFlash2). Taps: `rows × 5 × 5120 × 2 B` F16 per verify (400 KB at B = 8), capped at the ring for a
long prefill and released after it.

**Knobs.** `DOTLLM_DFLASH_DRAFT=n` caps drafts per round (default B-1 = 7; it can only lower it, so a DFlash verify is never more than 8 rows); `DOTLLM_DFLASH_TRACE=1` prints draft / inject wall
time per request (and, on Vulkan, the drafter's submit count and wait). Vulkan small-row GEMVs: `DOTLLM_VK_Q4K_ROWS` / `DOTLLM_VK_Q4K_ROWS_MULTI` (1..4 output
rows per workgroup for decode / verify, default 2 / 4, #1220; this replaced #1193's `DOTLLM_VK_Q4K_MULTI_ROWS` / `_R4` kernel) and `DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS` (default 3).

**Not yet.** Tree (DDTree) verification (gated on the verify-row cost curve, measured: "Verify-row cost curve (issue #1359, stage 1)" below); reduced-vocab (`d2t`) and DSpark drafters (LiLiCorr: #1237 below);
sampled-selector proposals (llama.cpp's DFlash2 sampled mode).

### LiLiCorr drafter for Nemotron-3.5 Bolt (issue #1237)

NVIDIA's `*-LiLiCorr` GGUFs (e.g. `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF`, arXiv 2608.20530) are llama.cpp
`dflash` drafters: the **same drafter contract** as DFlash2 (target layer-input taps, here `[2,19,36,52,69,86]` of Bolt's 88 layers;
`fc` → `enc.output_norm` → KV injection; a non-causal block of B = 8 `[anchor, mask(990) × 7]`; sliding window 1024; DFlash2
convolutions) **without the low-rank selector**, plus a **lattice head** (`lilicorr.*`). Run it with
`dotllm run|serve <bolt.gguf> --dflash <lilicorr.gguf>` on CPU or Vulkan (`nemotron_h` / `nemotron_h_moe` targets).

**The head** (llama.cpp `build_lilicorr_scores` + the walk in `common/speculative.cpp`): for each of the 7 drafted slots it takes the
top-8 candidates of the drafter's logits (sorted by logit) and builds one node per (slot, candidate) from
`token_proj(target embedding)` + `pass_hidden_proj(slot's normed drafter hidden)` + an MLP over five features
`[log p, p, log p − log p₀, rank/7, rank==0]` (LayerNorm first) + slot and rank embeddings. Two pre-RMSNorm transformer layers run over
all 56 nodes (8 heads, an additive per-head slot-distance bias, SiLU MLP, biases everywhere). Factors `SiLU(W[x, a, x·a])` (a =
`context_proj` of the fused context feature of the position before the anchor) give unit `out`/`in` vectors; the transition score is
`8·⟨inₛ,ⱼ, outₛ₋₁,ₚ⟩` (slot 0 against the anchor), and the draft is the greedy walk. The drafter always runs the full trained block;
`DOTLLM_DFLASH_DRAFT=n` only shortens the walk. The head runs on the host (56 nodes, ~70 MB of F32 / NVFP4 weights read in place
from the mmap) for both backends; the trunk drafter runs on the target's backend (Vulkan: on the GPU, weights imported zero-copy).

Backbone additions this needed: NVFP4 weights with a per-tensor `X.scale` (`DFlashDrafter.Matrix.Scale`), BF16 projections, and
**attention sinks** (`blk.*.attn_sinks`: one more logit in the softmax denominator; CPU and `dflash_attention_f32.comp`, which
starts its online softmax at `m = sink, l = 1`). A per-sequence ring of `B + 1` fused context features (F32, 144 KiB on Bolt) feeds
the anchor.

**Exactness (Vulkan).** A batch of ≤ 16 rows (a verify, its replay, a short prompt) runs:

- every projection through the **decode GEMV over its rows**: `matmul_q8_0[_coalesced]`, `matmul_nvfp4_*gemv*`,
  `matmul_iq2_xxs_f32_gemv` and `matmul_f32_gemv` take a `cols` push constant. Workgroup `w` computes row `w / cols` of column
  `w % cols` with the one-column arithmetic, and decode dispatches `cols = 1` through the same pipeline. The `cols` workgroups of a
  row are adjacent, so each weight row is read from memory once per verify;
- the attention one row at a time through the decode kernels (split-KV or per-token) over the cache prefix that row sees;
- the routed experts through the per-(token, slot) indexed MMVQ (never the grouped prefill).

Every row is then **bit-identical** to a decode step (`VulkanNemotronHDFlashTests`: 256/256 logits equal; with the path switched
off, 256/256 differ), so greedy speculation reproduces plain decoding token for token. The CPU host runs batches of ≤ 16 rows one
row at a time (an oracle, not a fast path).

**Rollback (Vulkan): no copy of the state.** The verify's selective scan (the register-resident scan, which serves Bolt's
d_state 128) **reads** the live Mamba-2 state and never writes it back. It also skips the conv-state save. Each SSM layer's scan
inputs (x, dt, B, C) and conv-input rows are kept, in F32, which is what decode itself produces and consumes. After the decoder
decides, the accepted rows are replayed into the live state at the head of the next forward's command buffer: the conv window first,
then the same scan dispatch, in place, over rows 0..a. This holds for a full accept too, and it needs no extra submit and no replay
forward. Per round the state is read by the verify and read + written by the replay; nothing else touches it, so the double-buffer
fallback was not needed. Exactness comes from the scan walking rows in order inside one dispatch: a replay over rows 0..a leaves
bit-for-bit the state the verify passed through after row a.

The pre-verify copy is kept as a fallback for scans that keep their state in memory (the non-register scan) and for A/B
(`DOTLLM_NEMOTRON_VERIFY_COPY=1`; F32, `layers × (d_inner·d_state + 3·conv_dim) × 4 B`, ≈ 4.1 MiB per SSM layer on Bolt). The CPU
oracle uses a checkpoint copy plus the decoder's replay forward. `DOTLLM_MTP_GDN_SNAPSHOTS=0` selects checkpoint + replay on Vulkan
too. With a drafter attached the adaptive gate tries speculation first (`MtpGatePriorBytes`, #1223).

**Memory (Bolt).** The drafter weights are stored as-is (~1.0 GB, imported zero-copy on UMA). The extra scratch:

| Scratch | Size on Bolt |
|---|---|
| Per-row scan inputs and conv rows (F32, decode's own dtype; kept for 16 rows) | `16 × (d_inner + n_head + 2·n_group·d_state + conv_dim) × 4 B` ≈ 1.3 MiB per SSM layer |
| Anchor ring (F32) | 144 KiB |
| Tap capture (F16) | `rows × 6 × 4096 × 2 B` |

#1194 applies the same read-only verify + replay to the dense `qwen35` GDN path on CPU, CUDA and Vulkan (174 → ~31 MiB at DFlash K=7).

**Measured.** Real Bolt 24GB on CPU (Core Ultra 7 155H), 24 tokens greedy: speculative output identical to plain, ~5.75 tokens/round.
Strix (Vulkan):

These runs used Bolt 24GB with the real LiLiCorr drafter on Strix Halo 8060S. Each figure is the median of 3 interleaved reps:
128 greedy tokens, `DOTLLM_MTP_ADAPTIVE=0`, K = 7 drafts (`--spec-draft-n-max 7` for llama.cpp master 0ae0e1976 Vulkan).
p0 is the high-acceptance prompt and p1 the low-acceptance one. Reproduced in #1368: the prose prompt "The history of the printing press began in" gives 3.25-3.75 tokens/round and 21-22 tok/s on every build from #1237 to dev (so it is p1), and a code prompt ("def quicksort(arr):") gives 7.9 tokens/round and ~41 tok/s (p0). The p0/p1 labels in these tables originally said the opposite; the numbers are right.

| Strix 8060S, Bolt 24GB | p0 tok/s | p1 tok/s | tokens/round | peak private / GPU-shared MiB |
|---|---|---|---|---|
| dotLLM plain | 30.05 | 29.72 | 1 | 21,841 / 21,359 |
| dotLLM + LiLiCorr | 25.86 | 12.86 | p0 7.44, p1 3.3–3.75 | 22,883 / 22,353 |
| llama.cpp plain | 28.88 | 28.89 | 1 | 21,872 / 21,424 |
| llama.cpp `draft-dflash` + LiLiCorr | 58.14 | 28.04 | acceptance p0 0.89, p1 0.33 | 24,972 / 24,143 |

- **Exactness:** dotLLM LiLiCorr output == dotLLM plain on **6/6** runs. On these two prompts llama.cpp `draft-dflash` also matched
  its own plain output.
- **Memory:** speculative dotLLM uses **2.1 GiB less private / 1.8 GiB less GPU memory** than llama.cpp.
- **Speed:** dotLLM speculation is **slower than plain**. The drafts are as good as llama.cpp's (p0: 7.44 tokens/round), but a
  round costs ~285 ms: verify 187 ms plus draft 93 ms.
- **The verify is the gap.** `DOTLLM_VK_NEMOTRON_PROFILE` on an 8-row verify gave 174 ms wall:

| Bucket | Time | What it is |
|---|---|---|
| `proj_gdn` | 63 ms | NVFP4 SSM projections |
| `moe_dense` | 54 ms | IQ2_XXS shared expert, Q8_0 latent, F32 router |
| `moe_routed` | 23 ms | routed experts |
| `lm_head` | 8.5 ms | LM head |

  At 3 rows the same buckets are 25.5 / 19 / 10.6 ms. So the column GEMVs scale linearly with rows: each workgroup decodes the
  weights again for its own column. The weight bytes are shared through the cache, but the NVFP4/IQ2_XXS dequant ALU work is not.
- **Draft:** 93 ms per round. About 32 ms is the host-side LiLiCorr head, 17 ms the 7-row LM head plus its 3.6 MB logits readback,
  and the rest the drafter trunk.

**Multi-column exact GEMVs (#1244).** Decode and every batch of 1..16 rows of an NVFP4, IQ2_XXS or Q8_0 projection now run
`matmul_nvfp4_gemv_multi`, `matmul_iq2_xxs_gemv_multi` or `matmul_q8_0_gemv_multi` (`MatMulExactGemvMultiKernel`). Each lane decodes
its weights once per row and FMAs them against every column. A column's arithmetic (explicit fma chains under `precise`, a fixed
shared-memory reduction tree, never `subgroupAdd`) does not depend on the column count or on the rows per workgroup, so a verify row
is bit-identical to the one-column decode dispatch. Rows per workgroup for verify: NVFP4 4, IQ2_XXS 2, Q8_0 4; decode 1
(`DOTLLM_VK_NEMOTRON_GEMV_ROWS=decode,verify` overrides; `DOTLLM_VK_NEMOTRON_EXACT_GEMV=0` restores #1237's per-column GEMVs). The
opt-in NVFP4 MMVQ keeps NVFP4 decode, as before. No new buffers.

Strix 8060S, Bolt 24GB + LiLiCorr, K = 7, 128 greedy tokens, medians of 3 interleaved reps (p0 code, p1 prose; labels corrected in #1368):

| `DOTLLM_VK_NEMOTRON_PROFILE`, ms over all layers | 8-row verify before | after | 3-row verify before | after | decode before | after |
|---|---|---|---|---|---|---|
| wall | 172.4 | 83.7 | 81.1 | 58.2 | 38.7 | 39.0 |
| `proj_gdn` (NVFP4) | 58.4 | 23.2 | 25.7 | 18.8 | 12.7 | 12.8 |
| `moe_dense` (IQ2_XXS shexp, Q8_0 latent, F32 router) | 50.7 | 13.8 | 19.0 | 8.5 | 7.4 | 7.3 |
| `moe_routed` (indexed experts, unchanged) | 23.1 | 24.4 | 10.8 | 10.9 | 4.4 | 4.3 |
| `lm_head` (Q8_0) | 17.0 | 3.6 | 8.3 | 3.9 | 2.6 | 2.6 |

| Strix, Bolt 24GB | p0 tok/s | p1 tok/s | round (draft + verify) | peak private / GPU-shared MiB |
|---|---|---|---|---|
| dotLLM plain, before / after | 30.75 / 30.66 | 30.50 / 30.36 | | 21,843 / 21,179 before, 21,853 / 21,180 after |
| dotLLM + LiLiCorr, before | 27.44 | 13.78 | 260 ms (95 + 160) | 22,889 / 22,178 |
| dotLLM + LiLiCorr, after | **44.27** | **23.00** | 164 ms (82 + 81) | 22,894 / 22,178 |
| dotLLM + LiLiCorr, after, `DOTLLM_NEMOTRON_VERIFY_COPY=1` | 43.57 | 22.15 | 167 ms (82 + 83) | 23,078 / 22,347 |
| llama.cpp plain | 29.59 | 29.66 | | 21,865 / 21,359 |
| llama.cpp `draft-dflash` + LiLiCorr | 60.22 | 29.38 | | 24,970 / 24,068 |

Every dotLLM run (plain and speculative, before and after, both verify modes) produced the same text. Speculation now beats plain
on code (1.44x) but not on prose (3.3-3.75 tokens/round). The round is now half draft: about 45 ms of each draft is the host-side
LiLiCorr head (`select` in `DOTLLM_DFLASH_TRACE`). The verify (81 ms) is 2.5x a 32.6 ms decode step; the rest of it is the indexed
routed experts, which barely share expert reads across 8 tokens (top-k routing over 240 experts). The read-only verify beats the
pre-verify copy by ~2 ms per round and 180 MiB.

**IQ2_XXS 14..16-column cliff (#1366).** With 2 rows per workgroup `matmul_iq2_xxs_gemv_multi` was 2.5x slower at 14..16 columns than at 13
(Bolt `moe_dense`, ms over 88 layers: 19.7 at 13, 46-58 at 14..16, slower than the 32-row GEMM path's 37). 4 rows per workgroup from 14 columns
(`MatMulExactGemvMultiKernel.Iq2WideColumns`) gives 26.0 / 28.2 / 30.4 at 14 / 15 / 16, bit-identical (a row's bits do not depend on NR).
Verify medians (ms, 3 reps each, same session): 14 rows 141 -> 117, 16 rows 151-153 -> 129. NR=1 and NR=2 both cliff; 4 is the largest NR with NR x columns <= 64.

**Next (follow-ups):**

1. The LiLiCorr head and the top-k/softmax on the GPU (no 3.6 MB logits readback); now the largest cost per round.
2. Read-only verify + replay for the Vulkan `qwen35moe` / Qwen4Exp GDN hosts (#1221; #1194 did the dense `qwen35` path).

**Qwen3.8-27B / Bonsai-2 27B.** No LiLiCorr head is published for `qwen35` targets. The head is a cheap re-scorer (~70 MB) trained
on a frozen DFlash backbone with the target's hidden states, so one could be trained for the z-lab / incoai DFlash2 drafters.
Converting is not possible: DFlash2's selector (rank-256 `prev ⊙ gate · next` scores) is a different parametrisation. The expected
gain over DFlash2 is modest. Both re-rank the drafter's own top-k with pairwise transitions; LiLiCorr adds attention across slots and
rank/probability features, which mostly helps the long tail of the block. On Qwen3.8 the measured gap to llama.cpp is the target's
multi-row verify, not draft quality (see the Strix table above), so a better head would not close it. Not worth building until a
checkpoint exists.

### Verify-row cost curve (issue #1359, stage 1)

Question: is a 16- or 32-node tree verify (DDTree / Weaver-style, Mirai "Trees from Marginals", arXiv 2607.06763) worth building here? That
depends on what extra verify rows cost. No new kernels: `benchmarks/VerifyRowCurve` loads a model once, prefills 64 tokens (32 on CPU), then
times one target forward over R rows (random token ids, content is irrelevant to cost), rolls the KV and the recurrent state back, repeats.
`verify` = `ForwardWithRecurrentSnapshots` (the call `MtpSpeculativeDecoder` makes; capped at 16 rows by `GemvColumns.MaxColumns`), `fwd` = plain
`Forward` (used for R = 24, 32). No drafter / tap capture in the timed call. Median of 3 interleaved reps x 5 timings (even reps ascend R, odd reps
descend, 2 discarded warm-up passes over every R first), under `scripts/gpu-lock.sh`, no human GPU process before or after (ollama idle).
Same build (`dev` @ c9436002 + the harness; the build recompiled 30 `.spv` blobs with the local glslc, which differ byte-wise from the checked-in
ones, so the shaders measured are the local build's), 2026-10-28. Ranges are min-max over the 15 timings. Strix Halo 8060S, Vulkan unless stated.

| rows | Qwen3.8-27B Q4_K_M ms | Bonsai-2 27B PQ2_0 ms | Bolt 24GB ms | CPU, Bonsai-2 PQ2_0 ms |
|---|---|---|---|---|
| 1 (decode) | 98.3 (97.4-101.5) | 44.9 (43.6-45.7) | 38.8 (38.2-44.9) | 247 (234-276) |
| 2 | 107.6 (106.6-116.3) | 53.5 (51.8-56.5) | 45.1 (44.2-49.6) | 1335 (1271-2282) |
| 4 | 121.6 (119.9-131.4) | 69.3 (66.2-74.8) | 58.0 (56.9-68.2) | 1604 (1527-1659) |
| 8 | 147.7 (145.5-151.7) | 106.4 (100.5-111.1) | 81.5 (80.2-86.3) | 2145 (2085-2172) |
| 9 | **578.3** (575.1-638.4) | **346.0** (344.3-350.5) | 85.6 (85.0-88.6) | not run |
| 16 | 592.6 (589.2-660.0) | 365.1 (362.2-368.9) | 164.2 (162.9-217.1) | 3462 (3355-4707) |
| 32 (`fwd`) | 553.5 (550.8-580.1) | 404.0 (402.1-405.5) | 253.0 (249.6-276.8) | 6752 (6398-6858) |

Intermediate rows (Qwen3.8 / Bonsai-2 / Bolt, ms): 3 = 113.9 / 63.2 / 52.0, 6 = 128.9 / 88.9 / 70.1, 12 = 585.3 / 354.5 / 110.4, 24 (`fwd`) = 542.8 /
386.6 / 234.7. The Qwen3.8 and Bonsai-2 numbers come from the second, finer run (13 and 11 row counts); the earlier 1-32 run (Qwen3.8 only) agreed within
3% for every row it shared (e.g. 8 rows 150.9, 16 rows 610.6 ms). Bonsai-2, Bolt and CPU were run once (3 interleaved reps). The Bolt numbers are the unprofiled run; its first rep-2 outlier (12 and 16 rows) is why the maxima are high.

**Marginal row cost and ratio vs a decode step** (slope between the quoted rows, ms per extra row, in brackets as % of the 1-row decode):

| cell | 1 -> 8 rows | 8 -> 16 rows | 16 -> 32 rows (`fwd`) | 8 rows / decode | 16 rows / decode | 32 rows / decode |
|---|---|---|---|---|---|---|
| Strix Vulkan Qwen3.8 Q4_K_M | 7.1 (7.2%) | 55.6 (57%) | -2.4 | 1.50x | 6.03x | 5.63x |
| Strix Vulkan Bonsai-2 PQ2_0 | 8.8 (19.6%) | 32.3 (72%) | 2.4 | 2.37x | 8.13x | 9.00x |
| Strix Vulkan Bolt 24GB | 6.1 (15.7%) | 10.3 (26.6%) | 5.5 (14%) | 2.10x | 4.23x | 6.52x |
| CPU, Bonsai-2 PQ2_0 | 2 -> 8: 135 (55%) | 164 (66%) | 206 (83%) | 8.7x | 14.0x | 27.3x |
| RTX 3060 CUDA (T5500) | not run, needs user go-ahead (shared-GPU rule) | | | | | |

Reading: Qwen3.8 and Bonsai-2 have a **cliff between 8 and 9 rows** (4x and 7x of the 8-row cost; the multi-column GEMVs cap at 8 columns and the
next path is a fixed ~340-570 ms GEMM, issue #1365), so 16/32-row trees cannot be priced from the 1..8 slope. Bolt has no cliff but steepens past 12
rows (issue #1366). On CPU the cliff is between 1 and 2 rows (issue #1367). **Fixed in #1367** (CPU only): the 2..8-row GEMM took the scalar float PQ2_0 tier (per-element unpack + f32 dot) and the GDN scan was single-threaded scalar. Now small-n PQ2_0 takes the W2A8 tier in 4-token tiles (bit-identical per token to the decode GEMV) and the scan is head-parallel + Vector256 (bit-identical to the scalar walk). CPU Bonsai-2 PQ2_0, same harness, ms (min-max of 9 timings): 1 row 164-387 (noisy; was 234-276), 2 rows 295-363 (was 1271-2282), 4 rows 305-372 (1527-1659), 8 rows 493-639 (2085-2172), 16 rows 946-1196 (3355-4707), 32 rows 1822-1997 (6398-6858). Marginal row cost 8->16 is now ~55 ms (about 25% of a decode step) against 164 before.

**Break-even tau gain.** A tree of the same depth as a K=7 chain pays the same draft but verifies N rows instead of 8. Break-even needs
`g* = (draft + V(N)) / (draft + V(8)) - 1` more tokens per round than the chain. Draft = measured in-engine draft ms of the same run (below;
K=7 DFlash/LiLiCorr: Qwen3.8 31.1, Bonsai-2 32.3, Bolt 76.0; CPU not measured, so `draft = 0` is used, which is a lower bound on `g*`). The
net columns are `(1+gain) / (1+g*)`, the end-to-end speed relative to the chain, for three tau gains over a chain of the same depth. **Those
gains are the paper's numbers (Weaver paper sec 5, B200, Qwen3.6-27B, T=1), not ours: +20% (~ Weaver over DDTree at equal size, paper +22.3%), +35% (~ DDTree-64 over a chain, 5.21 -> 6.98, paper +34%),
+75% (~ Weaver over a chain, paper +73.6%).** We have measured no tau gain for a tree.

| cell | N | g* | net at +20% | net at +35% | net at +75% | gate |
|---|---|---|---|---|---|---|
| Qwen3.8 Q4_K_M | 16 | +249% | 0.34x | 0.39x | 0.50x | no-go |
| Qwen3.8 Q4_K_M | 32 | +227% | 0.37x | 0.41x | 0.54x | no-go |
| Bonsai-2 PQ2_0 | 16 | +187% | 0.42x | 0.47x | 0.61x | no-go |
| Bonsai-2 PQ2_0 | 32 | +215% | 0.38x | 0.43x | 0.56x | no-go |
| Bolt + LiLiCorr | 16 | +53% | 0.79x | 0.89x | 1.15x | no-go (only +75% pays) |
| Bolt + LiLiCorr | 32 | +109% | 0.57x | 0.65x | 0.84x | no-go |
| CPU Bonsai-2 (draft = 0) | 16 | >= +61% | <= 0.74x | <= 0.84x | <= 1.08x | no-go |
| CPU Bonsai-2 (draft = 0) | 32 | >= +215% | <= 0.38x | <= 0.43x | <= 0.56x | no-go |
| CUDA RTX 3060 | - | not run, needs user go-ahead | | | | |

Hypothetical, not measured: if the 8-column cliff were removed and rows 9..16 cost the 1..8 slope, a 16-row verify would be ~204 ms (Qwen3.8) and
~177 ms (Bonsai-2), i.e. `g*` = +32% and +51%. Still above the +20% gate; it would only be break-even against the paper's training-free tree
gain (+34%, which the paper itself shows is end-to-end neutral: speedup 3.51x -> 3.56x), so a cliff fix alone does not justify the tree.

**Go / no-go (issue gate: marginal row cost < ~10% of a decode step, or break-even tau gain < 20%):**

- **Qwen3.8 Q4_K_M, Vulkan: NO-GO.** The 1..8 slope (7.2%) passes the 10% gate, but that is the chain, not a tree: at 16/32 rows the cliff makes `g*`
  +230-250%. Even cliff-free it is ~+32%. Revisit only with Weaver-class gains, and only after #1365.
- **Bonsai-2 PQ2_0, Vulkan: NO-GO.** 19.6% per row already at 1..8, cliff at 9, `g*` +187% at 16.
- **Bolt 24GB + LiLiCorr, Vulkan: NO-GO** on the gate (15.7% per row; `g*` +53% at 16). It is the closest: a Weaver-class +75% would net 1.15x at
  16 nodes, the LiLiCorr lattice is a ready candidate pool, and the recurrence is diagonal. But code tau is draft-quality bound (3.3-3.75), and
  the draft is already half the round (76 of 161 ms).
- **CPU: NO-GO** (decode-to-2-row cliff, every extra row 55-83% of a decode step).
- **CUDA: not run, needs user go-ahead.**

**Vulkan DFlash round breakdown** (`DOTLLM_MTP_ADAPTIVE=0 DOTLLM_DFLASH_DRAFT=K DOTLLM_MTP_TRACE=1 DOTLLM_DFLASH_TRACE=1`, `dotllm run ... -n 128 -t 0`, one
run per row, code prompt "Write a Python function that implements quicksort, with comments:" except Bolt, prose "The history of the printing
press began in"; mean over the trace's last window). `round - draft - verify` is the host work between GPU submits (accept walk, argmax over the rows'
logits, KV rollback, state commit), i.e. the "idle gap" that uzu's accept-behind-draft (#984: 72.8 -> 70.8 ms/pass, -2.7%) removed:

| run | tokens/round | draft ms | verify ms | round ms | gap ms (% of round) | draft detail |
|---|---|---|---|---|---|---|
| Qwen3.8 + DFlash2 K=7 | 7.31 | 31.1 | 184.3 | 217.5 | 2.1 (1.0%) | 18 drafts, 40 submits, 406 ms submit wait (10 ms each), head 9 ms/draft, `select` 8.8 ms/draft |
| Qwen3.8 + DFlash2 K=3 | 3.88 | 17.5 | 129.1 | 147.2 | 0.6 (0.4%) | 33 drafts, 70 submits, 541 ms wait, head 7 ms/draft, `select` 1 ms/draft |
| Bonsai-2 + r3 DFlash2 K=7 | 5.62 | 32.3 | 100.6 | 133.9 | 1.0 (0.7%) | 21 drafts, 46 submits, 310 ms wait, head 3.8 ms/draft, `select` 14 ms/draft |
| Bolt + LiLiCorr K=7 | 3.28 | 76.0 | 84.4 | 161.3 | 0.9 (0.6%) | 38 drafts, 118 submits, 1350 ms wait; host LiLiCorr `select` 36 ms/draft |

The between-round gap (host accept walk, rollback, commit) is 0.6-2.1 ms, <= 1.0% of a round, against uzu's 2.7% on M5 Pro. Even if
accept-behind-draft hid all of it that is the ceiling, so **the inter-round idle gap is not worth chasing on any Vulkan cell.** The idle that is
visible is *inside* the draft: a draft is two submits (layers, read back for the selector, then the head) and the host `select` stage at K=7 is
8.8 ms (Qwen3.8, 4% of the round), 14 ms (Bonsai-2, 10%) and 36 ms (Bolt LiLiCorr head, 22%, follow-up #1 above), while the K=3 selector is
1 ms. `select` is wall time and includes the readback wait, so it is an upper bound on GPU idle, not a measurement of it (submit waits
account for 72% / 94% / 50% / 49% of the draft wall in the four rows). A GPU-side selector walk is the lever there, not the accept gap; measure
GPU idle with timestamps before building it. Caveats: the in-engine verify (with tap capture and drafter injection) was 184 ms for 8
rows on Qwen3.8 against 148 ms bare, 100.6 vs 106.4 on Bonsai-2 and 84.4 vs 81.5 on Bolt, from single runs whose first trace window includes warm-up (not
reproduced; not trusted). The Bolt row is also at 3.28 tokens/round where the table above records 7.44 on the same prompt (issue #1368). Nothing here
changes the table's go/no-go: a 36 ms swing in verify is smaller than the cliff.

**Takeaways for stage 2 (not started, gate not passed):** (1) the verify-row curve, not draft quality, still caps speculation on Qwen3.8 and
Bonsai-2: fix the 9-row cliff (#1365) first (it unblocks MTP K>=8, `--lookup-draft` >= 8 and tree verifies of 9..16 nodes; it does NOT enable DFlash K>=8, which the shipped drafters cannot produce, see #1378); (2) Bolt's 12..16-row `moe_dense` kernel choice (#1366) is a
cheap cell-level win independent of trees; (3) a tree verify needs `ForwardWithRecurrentSnapshots` above 16 rows (cap `GemvColumns.MaxColumns`).

#### Verify-cliff fix: 9..16-column GEMVs (issue #1365)

Profile first (`DOTLLM_VULKAN_HYBRID_PROFILE=1`, 9 rows, Qwen3.8 Q4_K_M, per-category GPU timestamps): proj_ffn 501 ms vs 95 at 8 rows, proj_gdn 156 vs 36,
lm_head 75 vs 8, proj_attn 57 vs 11; everything else (gdn scan, attention, norms) is flat at 2-3 ms. So the cliff is the projections, all of them:
the GEMV caps at 8 columns and the next path is the 128x128 coopmat GEMM. The weight mix of that file is Q4_K (FFN), Q8_0 (GDN / attn q,k,v and the
48-row ssm_alpha/beta) and Q6_K (attn_output, lm_head), so Q4_K alone (the issue's title) is a third of it; Bonsai-2 is PQ2_0 everywhere plus BF16
ssm_alpha/beta. Fix, no new storage and no expansion (the weights are read as blocks in the tile load, as before):

- `matmul_q4_k_gemv_f32_multi`, `matmul_q6_k_gemv_f32_rows`, `matmul_q8_0_gemv_multi` are already specialisation-constant in NC and NR, so the 9..16-column
  pipelines are the same SPIR-V with `NC = 9..16, NR = 4` (`MatMulQuantGemvMultiKernel.WideMaxColumns`, `MatMulQ8_0GemvMultiKernel.WideMaxColumns`;
  `DOTLLM_VK_{Q4K,Q6K,Q8}_ROWS_WIDE` override NR, 1..4 because the shaders store one output per lane and have 64 lanes). A row's bits do not depend on NR
  or NC, so **every column of a 9..16-row verify is bit-identical to the one-column decode dispatch**, as it is for 2..8.
- PQ2_0: eight new `matmul_pq2_0_f32_gemv_mr_r4_c{9..16}_b1_w64` variants of the #474 multi-row family (`PQ2_0SmallNDispatch` default max `n` 8 -> 16).
- BF16: `matmul_bf16_gemv_multi_f32` `MAX_N` 8 -> 16 (same code; the 2 and 4/8-row Bonsai-2 verify did not move, 4 x 5 interleaved timings each way).
- Only the dense qwen35 host takes the wide path (`VulkanQwen3HybridDenseTransformerModel.SmallRow.cs`); the MoE host and `VulkanTransformerModel` keep their 8 cap
  (not validated above 8 rows; follow-up). Q5_K, Q6_K (non-Rows), IQ4_XS, IQ3_S multi kernels stay at 8 (`MaxColumnsFor`).
- Escape hatch / A/B: `DOTLLM_VK_SMALLROW_WIDE=0` plus `DOTLLM_VK_PQ2_0_MULTICOL_MAX_N=8` restores the old dispatch.

NR matters a lot here (every activation vec4 load is shared by NR weight rows; at 16 columns the loads, not the weight bytes, are the cost). Whole verify forward,
ms, 9 / 12 / 16 rows: Qwen3.8 Q4_K_M NR 1 = 276 / - / 410, NR 2 = 210 / 303 / 404, NR 4 = 158 / - / 250 (12 rows not taken for NR 1 and 4); Bonsai-2 PQ2_0 rows per workgroup 1 = 172 / 240 / 341,
2 = 123 / 159 / 210, 4 = 103 / 127 / 163 (one 3-sample pass each, `VerifyRowCurve`; NR 4 x 16 columns is also all a 64-lane workgroup can store).

Result, `ForwardWithRecurrentSnapshots` ms, `VerifyRowCurve ... ab` (old = caps at 8, new = up to 16, **in the same process**, every timing taken both ways,
order alternating, 3 reps x 5; median (min-max); Strix Halo 8060S Vulkan, ctx 64, `dev` + this change, checked-in `-O` shaders, 2026-10-28):

| rows | Qwen3.8 old | Qwen3.8 new | Bonsai-2 old | Bonsai-2 new |
|---|---|---|---|---|
| 2 | 102.3 | 102.4 | 49.8 | 49.8 |
| 8 | 135.0 | 135.1 | 98.4 | 98.4 |
| 9 | 317.8 (312-338) | **145.9** (141-150) | 319.6 (305-334) | **107.0** (102-112) |
| 10 | 320.7 | 156.9 | 314.5 | 113.8 |
| 12 | 320.3 | 180.7 (176-186) | 318.9 | 131.8 |
| 14 | 322.9 | 210.1 | 322.6 | 151.3 |
| 16 | 329.5 (320-355) | **240.3** (233-256) | 325.5 (322-346) | **167.2** (165-189) |

The "old" column is the GEMM path and moves between invocations (462 ms at 9 rows in an earlier invocation the same hour, 578 in the issue's run, whose build
recompiled shaders without `-O`); the new column and the 1..8 rows reproduced within a few percent. Marginal cost 8 -> 16 rows is now 13.1 ms/row (Qwen3.8)
and 8.6 (Bonsai-2) against the same-run 1..8 slope of ~5.4 and ~7.9, i.e. 16 rows cost 1.7-1.8x an 8-row verify and 2.5x / 3.9x a decode step, not the 3.4x / 7.6x of the
old arm. Qwen3.8's remaining slope is FFN FMA/load issue (proj_ffn 164 of 244 attributed ms at 16 rows, profiling on); getting under it means an int8 dot or coopmat tile on the
quantized blocks, not more columns per lane (not tried). 17..32 rows are unchanged (the GEMM, ~320 ms).

End to end on the CLI (`--mtp-head`/embedded MTP, `DOTLLM_MTP_ADAPTIVE=0`, code prompt, `-n 128 -t 0`, decode tok/s, old/new/new/old order, two runs each):
Qwen3.8 + `mtp-Qwen3.8-27B-Q8_0` head, K = 7: old 33.6 / 35.3, new 34.8 / 33.9 (no change, as expected: 8 rows); K = 11: old 20.9 / 26.5, new 35.6 / 38.4; K = 15:
old 28.4 / 30.3, new 38.0 / 36.6. Bonsai-2 + its embedded MTP head: K = 7 24.3 / 21.1 vs 24.5 / 24.8; K = 11 9.2 / 9.2 vs 18.3 / 18.2 (in-engine verify 318 -> 134 ms);
K = 15 8.3 / 8.6 vs 13.9 / 13.8 (verify 331-343 -> 178 ms). Greedy text is identical old vs new at K = 7, 11, 15 on both models. Acceptance on this prompt is low
(3.06 tokens/round for Bonsai-2 at any K), so K >= 8 is not faster than K = 7 *here*; the fix removes the penalty, it does not make long chains pay on a low-acceptance prompt.
**DFlash K = 8 / 11 / 15 is not reachable with the shipped drafters**: `DOTLLM_DFLASH_DRAFT` only caps the block's `BlockSize - 1 = 7` drafts, so the DFlash
numbers above stay at 8 rows; the cliff bites MTP K >= 8, prompt-lookup `--lookup-draft` >= 8 and any tree verify of 9..16 nodes.

Updated tree arithmetic (same formula, same-run V): Qwen3.8 N = 16 `g*` = (31.1 + 240.3) / (31.1 + 135.1) - 1 = +63%; Bonsai-2 (32.3 + 167.2) / (32.3 + 98.4) - 1 = +53%.
That is worse than the "+32% / +51% if the cliff were removed and the 1..8 slope held" extrapolation above, because the 9..16 slope is steeper than the 1..8 one. The tree
gate does not open; what #1365 changes is that 9..16-row verifies no longer cost 3-5x.

Parity: `VulkanDenseGemvRowsTests.{Q4KWide,Q8Wide,Q6KWide}_*` (every column bit-identical to the one-column dispatch and within 1e-3 + 1e-4 rel of an f64 dot
over the dequantised weights; ragged M, K of 3 blocks, M below NR), `VulkanMatMulPQ2_0GemvF32KernelTests.{RecordColumns,MultiRowVariants}_*` extended to 16 columns
(vs the looped #470 kernel at 1e-5 and the scalar reference, sentinel past the live columns), `VulkanMatMulBf16GemvMultiF32KernelTests` at 9/12/13/16.
CUDA has the same 8-column caps (`Pq2_0GemvMultiMaxColumns`, `CudaQ8_0RbGemv.MaxColumns`, `CudaSmallSGemvDispatch.MaxColumns`); not measured here (#1375). The shared
`PQ2_0SmallNDispatch` cap stays 8 on the MoE host and `VulkanTransformerModel` (their 9..16-row PQ2_0 path is unmeasured, #1374 below). Remaining: 17..32 rows (#1376), Q5_K / Q6_K / IQ4_XS / IQ3_S wide kernels (#1377), (the DFlash K >= 8 premise in the stage-1 takeaways was wrong and is corrected above, #1378).

#### 17..32-row verify (issue #1376): measured, not built

`VerifyRowCurve` (Qwen3.8-27B Q4_K_M, ctx 64, 3 reps x 3, `dev` @ cc686e1d, both locks held, ollama idle), `fwd` median (min-max) ms: 1 row 102.9 (101.4-106.5),
8 rows 164.3 (159.3-167.8), 16 rows 276.0 (271.9-282.1; `verify` 277.5), **17 rows 997.6 (979.9-1023.4), 20 rows 990.4, 24 rows 1005.6, 32 rows 1019.3 (973.2-1029.4)**.
So the 16 -> 17 step is a 3.6x cliff and 17..32 rows are flat at ~1 s (the GEMM's fixed cost: 128x128 coopmat tile + per-projection F16 dequant, #495/#1107 rule 2);
the 553 ms quoted for 32 rows in the stage-1 table was a different build and the GEMM path has not been stable between builds (320 / 462 / 578 / ~1000 ms).

Why nothing was built: the only way to serve 17..32 rows on the existing GEMVs is two dispatches per projection (columns 0..15 and 16..n-1), which needs
column-offset descriptors/push constants in every multi-column kernel (the `Record` APIs take whole buffers and the descriptor cache is keyed on handles). The
ceiling of that is ~16-row cost x 2 = ~550 ms (13 ms/row marginal at 9..16 rows, so 32 rows ~ 135 + 24 x 13 = ~450-550 ms): 4.4-5.4x a decode step. Break-even
needs `(1 + a) / (1 + C/decode_step)`: at 32 rows that is ~5.4 accepted tokens per round just to match plain decode, against the measured 3.05-3.06 tokens/round
(Qwen3.6 CPU head, Bonsai-2 MTP). The 16-row tree already did not open the gate (+53..63 % `g*` from the arithmetic above needs acceptance the drafters do not
deliver), and nothing calls a 17..32-row verify: `ForwardWithRecurrentSnapshots` is capped at `GemvColumns.MaxColumns = 16`, so only plain `Forward` (e.g. prefill
of a short chunk) lands there. The real fix for that range would be an int8-dot / coopmat tile that reads the quantized blocks (dequant in the tile load), a new GEMM
family, KL-checked; it is only worth building if a drafter with >= 6 accepted tokens/round or a tree consumer appears. Tracked in #1376; reopen then.

#### Verify-cliff on the MoE host and `VulkanTransformerModel` (issue #1374)

`VulkanQwen3MoeHybridTransformerModel` now serves 9..16 rows on the same wide multi-column GEMVs through the `AcceptsColumns` gate: Q4_K, Q6_K_Rows
(preferred above 8; the older non-Rows Q6_K otherwise), Q8_0 (the F32-activation `MatMulQ8Multi`, since the Q8_1 MMVQ multi stops at 8; so 9..16 rows are
ULP-close, not bit-equal, to the 2..8-row Q8_1 path) and BF16. `DOTLLM_VK_SMALLROW_WIDE=0` restores the 8-column cap (pair it with `DOTLLM_VK_PQ2_0_MULTICOL_MAX_N=8`
for PQ2_0). Routed-expert paths are unchanged.

Qwen3.6-35B-A3B UD-Q4_K_M (Q8_0 attention / GDN / shared-expert projections; Q4_K / Q5_K routed banks), Strix Halo Vulkan, ctx 64, `VerifyRowCurve ab`
(old = 8-column cap, new = wide; interleaved in-process, alternating order, 5 reps x 5, `ForwardWithRecurrentSnapshots`), median [min-max] ms:

| rows | old | new |
|---|---|---|
| 2 | 28.6 [28.1-30.7] | 28.8 [28.0-30.4] |
| 8 | 53.0 [50.8-54.2] | 53.0 [51.6-56.1] |
| 9 | 75.4 [74.1-80.9] | 60.3 [59.3-62.4] |
| 10 | 78.6 [76.4-88.3] | 64.1 [62.8-66.0] |
| 12 | 84.5 [83.5-88.5] | 72.1 [70.3-74.2] |
| 14 | 91.5 [89.6-94.0] | 79.9 [78.4-89.3] |
| 16 | 88.2 [85.4-90.2] | 78.7 [76.2-81.6] |

1..8 rows unchanged; 9 rows -20%, 16 rows -11%. The MoE host's cliff was small (8 -> 9 rows: 53 -> 75 ms, 1.4x, not the 3-8x of the dense hosts) because
its dense projections are a minor share; the wide GEMVs recover 14-15 ms of it. A step remains at 8 -> 9 (53 -> 60 ms): the routed-expert and attention paths.

`VulkanTransformerModel` (Llama-3.1-8B Q4_K_M, plain `Forward`, 5 x 3): 1 row 25.3, 4 rows 69.2, 8 rows 79.7, 9 rows 82.5, 12 rows 82.4, 16 rows 82.9 ms. No 8 -> 9 cliff
(its Q4_K goes to MMQ, Q6_K / Q8_0 multi-column GEMVs are deliberately capped at 4 rows by #996), so nothing to widen. Its 2..4-row cost (2.7x a decode step at 4 rows) is a different matter.

**PQ2_0 default (16, #1365) is NOT carried over.** No PQ2_0 model runs on the MoE host or `VulkanTransformerModel` locally, so 9..16-row PQ2_0 gains / regressions there
are unmeasurable; both hosts pass `hostMaxColumns: PQ2_0SmallNDispatch.UnvalidatedHostMaxColumns` (8) to `PQ2_0SmallNDispatch.Record`, which is the pre-#1365 behaviour. The
dense qwen35 host keeps 16 (measured, #1365). Raise it per host once a PQ2_0 model is measured there.

#### Q5_K / Q6_K / IQ4_XS / IQ3_S wide GEMVs (issue #1377)

Q5_K, non-Rows Q6_K and IQ4_XS multi GEMVs now serve 9..16 columns (Q5_K / Q6_K already used `acc[NC]` arrays: only the pipelines were missing; the IQ4_XS generator
`gen_iq_gemv_multi.py` emits 16 scalar accumulators). Every column keeps the single-row kernel's accumulation order but is **not bit-equal** across NC
(per-NC specialisation changes FMA contraction: ~1e-7 relative); the parity tests compare to the single-row kernels (2e-3 + 1e-4 rel) and to the 8-column dispatch (1e-4 + 1e-5 rel).
Kernel timing (`Bench_WideColumns`, `DOTLLM_GEMV_WIDE_BENCH=1`, device-local weights, min of 5 x 20 dispatches; us):

| format, shape | 1x8 | 2x8 | 1x12 | 1x16 |
|---|---|---|---|---|
| Q5_K 17408x5120 | 768 | 1497 | 1016 | 1299 |
| Q6_K 5120x6144 | 288 | 542 | 373 | 531 |
| IQ4_XS 17408x5120 | 712 | 1441 | 894 | 1390 |
| IQ3_S 17408x5120 (probe only) | 1937 | 3793 | 18248 | 23352 |

Q5_K / Q6_K / IQ4_XS 16 columns cost about the same as two 8-column dispatches (4-13% less) and avoid the GEMM fallback. **IQ3_S was capped at 8**: 12 / 16 columns were 10x slower than two
8-column dispatches. #1389 found the cause: NOT a register spill (`VK_AMD_shader_info`: 127 / 256 VGPRs, 0 bytes scratch at 16 columns) but the
per-thread layout (32 contiguous floats per column per lane, 128 B apart across lanes) that thrashes the L1 as the working set grows with the column count, the same
failure IQ4_XS had before its coalesced rewrite (#902). IQ3_S now uses the coalesced K-position layout (one 32-lane workgroup per row, lane = 8 consecutive elements, two
vec4 loads per column), and serves 16 columns: 1x8 700, 2x8 1450, 1x12 1250, 1x16 1310 us (was 1941 / 3833 / 18389 / 23636 us), so 8 columns got 2.8x faster too.
The end-to-end 9..16-row verify on a Q5_K / IQ4_XS file was not measured (no such hybrid model locally); the kernel-level numbers above are the evidence.

### `qwen35moe` MTP head (issue #1188)

Qwen3.6-35B-A3B ships its MTP block embedded (`blk.40`, `nextn_predict_layers = 1`; `ggml-org/Qwen3.6-35B-A3B-MTP-GGUF`,
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`). The head is the dense `qwen35` head with the trunk's MoE FFN (llama.cpp `qwen35moe.cpp` `graph_mtp`):
`eh_proj([enorm(embed) ; hnorm(h)])` -> gated attention over the head's own KV -> `post_attention_norm` -> routed top-k + sigmoid-gated shared
expert -> `nextn.shared_head_norm` -> the trunk's LM head. Absorb computes K/V only, so the experts run on draft steps only.

- **CPU oracle** (`Qwen3MoeHybridTransformerModel.Mtp.cs`): head loader, capture/absorb, `ForwardMtp`, and the pooled GDN checkpoint pair the
  MoE hybrid lacked. Synthetic tests: trunk logits unchanged by capture/absorb, absorb K/V == the draft step's, checkpoint/restore
  discriminates, greedy speculation == plain greedy at K = 1/3/5. Real UD-Q4_K_M on the Strix CPU: identical output (64 tokens); 6.65 vs
  10.17 tok/s (the CPU MoE verify reads K+1 rows' experts and replays on rejection; the adaptive gate keeps CPU on plain decode by its size prior).
- **Vulkan** (`VulkanQwen3MoeHybridTransformerModel`): GDN checkpoint pair, per-row recurrent snapshots (the qwen4exp host's snapshot scan;
  removed a ~30 ms replay from every partial rejection), every-row logits for batches of up to 16 rows, and the head (resident routed banks,
  one submit per draft step, ~2.5 ms). Drafts match the CPU oracle (argmax exact) and speculation equals plain greedy on the synthetic fixture.
  **Opt-in (`DOTLLM_VK_QWEN35MOE_MTP=1`)**: measured on Strix (UD-Q4_K_M, 4 prompts x 128 tokens) it is neither exact nor a clear win yet:

| arm | decode tok/s (mean of 4) | x plain | peak private / GPU shared |
|---|---|---|---|
| dotLLM plain | 50.7 | 1.00 | 24.6 / 23.2 GB |
| dotLLM MTP K=2 / 3 / 4 | 49.0 / 52.1 / 55.1 | 0.97 / 1.03 / 1.09 | 24.7-24.8 / 23.3-23.4 GB |
| llama.cpp `0ae0e1976` plain | 57.5 | 1.00 | 22.6 / 21.2 GB |
| llama.cpp MTP K=2 / 3 / 4 | 72.2 / 84.2 / 64.6 | 1.26 / 1.46 / 1.12 | 23.8-24.0 / 22.0-22.1 GB |

  Round anatomy (K=3, with snapshots): draft 7.5 ms, verify 38 ms (4 rows; 20 ms for 1), a fully accepted round ~47 ms for 4 tokens. Greedy
  output equalled plain decode on 2 of 12 runs (llama.cpp: 1 of 4 at K=3): the multi-row routed-MoE / Q8_1 kernels are not bit-identical to the
  fused single-token decode chain, so near-ties resolve differently. Making the verify exact and cheaper is the follow-up.

## Gumbel-coupled sampled verification (#1358, CPU reference)

Opt-in (`InferenceOptions.GumbelSampling` or `DOTLLM_GUMBEL_SAMPLING=1`; default off because it changes which tokens a seed yields). A stochastic token is drawn as `argmax_i(prepared_logit_i + G(seed, n, i))`, n = generation index (`generatedIds.Count`), G = Philox4x32-10 noise from a 24-bit uniform (`GumbelNoise`; a 32-bit conversion rounds to u = 1 and gives +inf noise). `prepared` is the output of `SamplerPipeline.PrepareDistribution` (penalties, temperature, top-k/p, min-p, constraint mask), so excluded tokens never win and the draw is exactly softmax(prepared). A verify row accepts draft d iff d equals the row own sample: P(accept) = p(d), same as rejection sampling for point-mass drafts, and the emitted token is the one plain decoding draws for that seed, so speculative text equals plain text (tests: GumbelSpeculativeSamplingTests). T = 0 unchanged. The noise uses a deterministic F32 `GumbelNoise.Log` (IEEE add/multiply only, degree-9 polynomial) instead of `MathF.Log`, so a device kernel reproduces it bit for bit (GLSL `log` promises only an absolute error near 1, which near u = 1 is several times the value).

### Device-side Gumbel draw on Vulkan (#1382)

`gumbel_argmax_f32.comp` (`GumbelArgMaxKernel`) draws `argmax(logits * invT + G(seed, n, i))` per verify row, reading the LM-head logits in place and writing one id per row; `IModel.ForwardGumbelSampled` (implemented by `VulkanQwen3HybridDenseTransformerModel`) runs the verify forward with that dispatch instead of the F32 logits download, and `MtpSpeculativeDecoder` uses it when `DOTLLM_GUMBEL_SAMPLING=1`, the model supports it, there is no constraint and `SamplerPipeline.DeviceGumbelInvTemperature` is non-null. Off switches: `DOTLLM_DEVICE_GUMBEL=0` (decoder) or `DOTLLM_VULKAN_DEVICE_GUMBEL=0` (model).

- **The shader takes RAW logits and applies only the temperature scale** (`l * (1f/T)`, the same reciprocal the CPU step multiplies by). Pre-masked logits would need the readback this removes; top-k / top-p / min-p / penalties / constraint masks need host history and constraint state (unknown for rows beyond the first rejection) or a per-row k-th-largest / sort with the CPU's index-order tie rule. Any such pipeline keeps the host path. Follow-up: device top-k / min-p threshold pass (issue filed).
- **Bit-exactness** (tests `VulkanGumbelArgMaxKernelTests`): noise equals the CPU for all 2^24 uniforms and for Philox counters; the sampled id equals `GumbelNoise.Sample` on 320 real-vocabulary rows; chi-square vs softmax on the device draw. Every float op in the shader is `precise` (no FMA).
- **Batched vs single-row logits are NOT bit-identical** on Tev1-4B Q4_K_M (Vulkan): in 240 verify rows (4 rows, 60 positions) not one row equalled the plain single-row decode (99.9% of elements differ, max abs diff 2.5e-3 on logits ~10-20, same size for row 0). Consequence for the identity guarantee: the Gumbel argmax flips between plain and speculative in **3-6e-5 of draws** (T=0.7: 24/480,000; T=1.0: 30/480,000 and 7/240,000 with 8 rows), i.e. about one differing token per 16-30k; greedy argmax flipped 0/240 rows. This bounds "speculative text == plain text" on the GPU; end to end 9/9 runs (763 tokens) were identical.
- **Cost** (VerifyRowCurve-style paired A/B, ctx 64, 3 reps x 5, order alternated, both locks): a verify forward plus host `PrepareDistribution` + `GumbelNoise.Sample` is dominated by the CPU Gumbel loop (~17.5 ms per row at V = 248k), not the readback: 2 rows 35.3 ms host vs 17.8 device (readback-only floor 17.6); 4 rows 56.7 vs 21.2 (21.7); 8 rows 96.5 vs 25.5 (26.5). End to end (TextGenerator, MTP, T=0.7, 9 runs): tok/s plain 59.9-62.3, host-Gumbel 32.5-43.1, device-Gumbel 44.0-71.3 (median 61.8).
- Only the dense qwen35 Vulkan host is wired; Gemma4/Qwen4Exp/NemotronH/MoE hosts, the continuous-batch scheduler and `EntropyBoundSampler` (passes an empty history) are not covered.
