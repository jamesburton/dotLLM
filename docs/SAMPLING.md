# Sampling Pipeline — dotLLM

## Composable ISamplerStep Chain

The sampler pipeline is a sequence of `ISamplerStep` operations applied to raw logits before final token selection. Steps are ordered and extensible.

```
ISamplerStep:
  Apply(Span<float> logits, SamplerContext ctx) → void
```

## Default Pipeline Order

### 1. Logit Bias (`LogitBiasStep`)
Per-token additive bias from request `logit_bias` map: `logits[token_id] += bias`.
OpenAI API compatible: `{token_id: float_value}`.

### 2. Constraint Mask (`ConstraintMaskStep`)
Apply `IDecodingConstraint.GetAllowedTokens()` mask if structured output active.
Invalid tokens → `-∞`. See [CONSTRAINED_DECODING.md](CONSTRAINED_DECODING.md).

### 3. Repetition Penalties

Four independent, combinable logit processors (`ILogitProcessor`), auto-added by `SamplerPipeline`
when their controlling `InferenceOptions` field is non-default:

- **Repetition penalty** (`RepetitionPenaltyProcessor`, multiplicative): For tokens in history,
  `logit = logit > 0 ? logit/penalty : logit*penalty`. Common in open models.
  `InferenceOptions.RepetitionPenalty` (1.0 = disabled) + `RepetitionPenaltyWindow` (0 = full history).
- **Frequency penalty** (`FrequencyPresencePenaltyProcessor`, additive, proportional):
  `logit -= frequency_penalty × count(token)`. OpenAI API (`frequency_penalty`).
- **Presence penalty** (same processor, additive, binary): `logit -= presence_penalty × (count(token) > 0 ? 1 : 0)`.
  OpenAI API (`presence_penalty`). Runs over the *full* token history (no windowing), unlike repetition penalty.
- **Logit bias** (`LogitBiasProcessor`): `logits[token_id] += bias` from `InferenceOptions.LogitBias`
  (`IReadOnlyDictionary<int, float>`). OpenAI API (`logit_bias`, string token-id keys resolved to
  ints by `RequestConverter`). CLI: `--logit-bias|-l token_id=bias` (repeatable).
- **DRY (Don't Repeat Yourself)** (`DryProcessor`, llama.cpp `--dry-multiplier` family): detects
  whether the *next* token would continue a previously-seen repeated n-gram ending at the most
  recent token, and applies an exponentially growing penalty as the matched length increases —
  meaningfully better than plain repetition penalty at killing verbatim loops without flattening
  the rest of the distribution. Controlled by `InferenceOptions.DryMultiplier` (0 = disabled),
  `DryBase` (default 1.75), `DryAllowedLength` (default 2 — minimum match length before penalizing),
  `DryPenaltyLastN` (0 = full history), and `DrySequenceBreakers` (token strings that reset n-gram
  matching, e.g. newline/punctuation — resolved to token ids by `SamplerPipeline` via the tokenizer
  passed to its constructor). CLI: `--dry-multiplier`, `--dry-base`, `--dry-allowed-length`,
  `--dry-penalty-last-n`, `--dry-sequence-breaker` (repeatable). Server API: `dry_multiplier`,
  `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`.

### 4. Temperature (`TemperatureStep`)
`logits /= temperature`. T=0 → greedy (argmax). T=1 → unmodified. T>1 → more random.

### 5. Top-K (`TopKStep`)
Keep only K highest-probability tokens. Set rest to `-∞`.

### 6. Top-P / Nucleus (`TopPStep`)
Sort by probability descending. Keep smallest set where cumulative probability ≥ P.

### 7. Min-P (`MinPStep`)
Keep tokens with `probability ≥ min_p × max_probability`. More stable than top-p across distributions.

### 7b. Top-nσ (`TopNSigmaSampler`)
llama.cpp `--top-nsigma`. Keeps tokens with `logit ≥ max(logits) - n × stddev(logits)`, where mean/stddev
are computed over the raw (pre-temperature) logit distribution, skipping tokens already masked to `-∞`
by an earlier step. Unlike top-p/min-p, the threshold comes from the distribution's shape rather than
cumulative probability mass — the "Top-nσ" paper argues this is more robust to temperature scaling.
`InferenceOptions.TopNSigma` (negative = disabled, the default). When enabled it also disables the
`SamplerPipeline` "fast top-k" shortcut so the shape-based masking always runs. CLI: `--top-nsigma`.
Server API: `top_n_sigma`.

### 8. Categorical Sample (`CategoricalSampleStep`)
Convert logits to probabilities (softmax), sample. Argmax if temperature was 0.

## Custom Logit Processors

Users can inject arbitrary processing at any pipeline position:

```
ILogitProcessor:
  Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext ctx) → void
```

Use cases: classifier-free guidance, contrastive decoding, custom penalty schemes.

## RoPE Scaling Override

`RoPEScalingType` (None/Linear/YaRN/NTK/DynamicNTK/Su) and the YaRN math are fully implemented
(`GgufModelConfigExtractor`) and normally auto-derived from GGUF metadata. When that metadata is
wrong or absent (common on community context-extended quants), `GgufModelConfigExtractor.ApplyRoPEOverride`
lets the caller override individual `RoPEConfig` fields after extraction — pure override plumbing,
no new scaling math. Applies to both `ModelConfig.RoPEConfig` and `ModelConfig.GlobalRoPEConfig`
(Gemma 4's dual-RoPE global-attention config) when present.

Matches llama.cpp's `--rope-scaling`/`--rope-freq-base`/`--rope-scale`/`--yarn-*` flag set:

| `RoPEOverrideOptions` field | Overrides | CLI flag (`dotllm run`/`serve`) | Server API (`ModelLoadRequest`) |
|---|---|---|---|
| `ScalingType` | `RoPEConfig.ScalingType` | `--rope-scaling {none,linear,yarn,ntk,dynamic}` | `rope_scaling` |
| `FreqBase` | `RoPEConfig.Theta` | `--rope-freq-base` | `rope_freq_base` |
| `ScalingFactor` | `RoPEConfig.ScalingFactor` | `--rope-scale` | `rope_scale` |
| `OrigMaxSeqLen` | `RoPEConfig.OrigMaxSeqLen` | `--yarn-orig-ctx` | `yarn_orig_ctx` |
| `AttnFactor` | `RoPEConfig.AttnFactor` | `--yarn-attn-factor` | `yarn_attn_factor` |
| `BetaFast` | `RoPEConfig.BetaFast` | `--yarn-beta-fast` | `yarn_beta_fast` |
| `BetaSlow` | `RoPEConfig.BetaSlow` | `--yarn-beta-slow` | `yarn_beta_slow` |

Every field is optional — a null field leaves the GGUF-derived value unchanged, and passing no
override fields at all is a no-op (`ApplyRoPEOverride` returns the input config unmodified).
This is model-load-time configuration (`ServerOptions.RopeOverride`), not a per-request sampling
parameter — set it via `dotllm run`/`dotllm serve` CLI flags or `POST /v1/models/load`.

## Beam Search

Alternative to sampling. Maintains N candidate beams:

1. Each step: expand each beam by top-M tokens → N×M candidates.
2. Score by cumulative log-probability with length normalization.
3. Keep top N beams.
4. Stop when all beams hit EOS or max length.
5. Return top-K completed sequences by normalized score.

**KV-cache**: Beams sharing prefix use copy-on-write (PagedAttention COW blocks).
**Constraints**: Each beam clones its `IDecodingConstraint` state at branch points.
**Configured via**: `n` parameter in API request (n > 1 triggers beam search).

## Stop Conditions — IStopCondition

Multiple conditions active simultaneously. First match wins.

```
IStopCondition:
  ShouldStop(tokenId, generatedTokens, decodedText) → StopResult

StopResult: Continue | Stop | StopInclude
```

### Built-in Conditions

- **EOS token** — Always active. Model's end-of-sequence token.
- **Max tokens** — Hard limit on generated tokens.
- **Stop strings** — Text patterns that terminate generation (e.g., `"\n\nHuman:"`, `"END"`). Rolling buffer of decoded text, check suffix matches. Stop string excluded from output.
  - **Known limitation (Wave 8 / [issue #121](https://github.com/kkokosa/dotLLM/issues/121))**: the detector fires at token boundaries and the entire last token is removed from the output, not just the matched suffix. For BPE tokenizers that merge preceding text into the same token as the stop sequence, valid content may be lost. Use **stop token sequences** (below) when exact boundary control matters.
- **Stop token sequences** — Token ID sequences (avoids tokenization ambiguity).
- **Custom predicate** — Arbitrary `IStopCondition` implementation.

OpenAI API: `stop: ["str1", "str2"]` maps to stop string conditions.