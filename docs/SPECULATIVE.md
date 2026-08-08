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

# Serve: MTP is opt-in (default off) — takes the continuous-batch scheduler offline when enabled
dotllm serve qwen3.6-27b-mtp.gguf --mtp --speculative-k 5
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

### MTP head's own KV-cache lifetime — a second documented simplification

llama.cpp's MTP draft context keeps a KV-cache that persists across speculation rounds with
partial rollback. dotLLM's `MtpSpeculativeDecoder` resets `IMtpState`'s own tiny KV-cache to empty
at the start of every round instead: each round's first draft step re-seeds entirely from the
target model's own just-verified hidden state, so the MTP head only ever needs causal
self-attention over the *current* round's own K draft steps, never across rounds. Simpler to
reason about for a first CPU implementation; does not affect correctness.

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