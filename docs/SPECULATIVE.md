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
rejection at `accepted` restores row `accepted`. There is no replay. Both the GDN matrix state and
the conv1d window are captured, the latter as `ConvInput` rows `t+1 .. t+dConv-1`.

- **Implemented:** CPU `Qwen3HybridDenseTransformerModel` (`GatedDeltaNetScan.Execute` copies
  the state out after each row) and `VulkanQwen3HybridDenseTransformerModel` (the
  `gdn_scan_multi_token_lds_fused_snap_f32` twin of the shipping scan writes each row's state from
  LDS), and since #478 `CudaQwen3HybridDenseTransformerModel` (no new kernel: the CUDA scan is
  already one launch per token, so each row's state is a D2D copy right after its launch; the
  scratch is laid out `[row][layer]` so a restore is two copies). The CUDA checkpoint is pooled
  (one spare) like the CPU and Vulkan ones (#469).
- **Opt-out:** `DOTLLM_MTP_GDN_SNAPSHOTS=0`, or `MtpSpeculativeDecoder.UseRecurrentRowSnapshots`
  in-process. `Replays` and `ReplaysAvoided` count each path.
- **Memory:** `K x GDN layers x (NVHead*DState^2 + conv)` floats, grown to the largest K seen and
  kept. That is about 150 MiB per row on Bonsai 2 27B (449 MiB at K=3).
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
