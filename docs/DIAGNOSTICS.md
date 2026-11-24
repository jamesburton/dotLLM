# Diagnostics & Interpretability — dotLLM

## Hook System

### Hook Points

Fired at well-defined pipeline locations with the activation tensor:

| Hook Point | Location | Typical Use |
|-----------|----------|-------------|
| `PostEmbedding` | After token embedding, before layer 0 | Input analysis |
| `PreAttention(layer)` | After pre-attention norm | Attention input study |
| `PostAttention(layer)` | After attention, before residual | Attention output analysis |
| `PreFfn(layer)` | After post-attention norm | FFN input study |
| `PostFfn(layer)` | After FFN, before residual | FFN output analysis |
| `PostLayer(layer)` | After residual add (residual stream) | Layer-wise analysis, SAE |
| `PreLmHead` | Final hidden state | Embedding analysis |
| `PostLmHead` | Raw logits | Logit analysis |

### IInferenceHook Interface

```
IInferenceHook:
  HookPoint → HookPoint
  OnActivation(ReadOnlySpan<float> activation, HookContext ctx) → HookResult

HookResult:
  Continue     — Read-only inspection, no modification
  Replace(Span<float>) — Replace activation (interventions, steering, ablation)

HookContext:
  LayerIndex: int
  TokenPosition: int
  SequenceId: int
  CurrentStep: int
```

### In-place residual modification (`IResidualModifier`)

`IInferenceHook` (and `HookResult.Replace(float[])`) allocates per call, so the forward passes do not call it: the generic
hook table above is the design, not an implemented hook point per model. What is wired is narrower and cheaper: read-only
[hidden-state taps](#hidden-state-taps) (observation, with an optional abort) and the in-place `IResidualModifier` below
(intervention). For allocation-free `PostLayer` interventions (e.g. ternary steering adapters, see
[TERNARY_ADAPTER.md](TERNARY_ADAPTER.md)) the CPU `TransformerModel` exposes `ResidualModifier`
(`DotLLM.Core.Diagnostics.IResidualModifier`): after each layer it gets the `[tokens x hidden]` residual span to
mutate in place. Null by default; unset cost is one branch per layer.

### Hidden-state taps

Read-only observation of the residual stream on every backend, with an optional early exit (#1920). A model that implements
`IHiddenStateTapHost` (`DotLLM.Core.Diagnostics`) accepts taps; a model that does not is refused by type.

```csharp
var host = (IHiddenStateTapHost)model;
using var tap = host.RegisterTap(
    new HiddenStateTapSubscription(Layers: [12, 24], Points: [HookPoint.PostLayer, HookPoint.PreLmHead], Rows: TapRows.Last),
    sample => { /* sample.Data is a hidden-size F32 row, valid only inside the callback */ });
```

**Points.** `PostEmbedding`, `PostLayer` (the residual leaving the layer; `Layers` null means every layer), `PreFinalNorm` and `PreLmHead`.
Any other `HookPoint` throws `NotSupportedException` at registration (those sit inside fused kernels and are never silently ignored);
an out-of-range layer or an empty point list throws `ArgumentException`.

**`PreFinalNorm` vs `PreLmHead`.** `PreFinalNorm` is the residual leaving the last layer (the same rows as `PostLayer(last)`).
`PreLmHead` is *post* final norm: the normed rows the LM head reads (`rmsnorm(PreFinalNorm) * output_norm.weight`, asserted per row against a host
reference on every backend), taken before any Hadamard rotation the head applies; on a Hadamard-rotated hybrid it is therefore the pre-rotation
norm output, not the literal head input. **One meaning of `PreLmHead` + `TapRows.All` on every host: every trunk row of the forward**, even when
the head projects only the last (`lastTokenLogitsOnly`): the final norm is widened to every row while such a tap is attached, at the cost of one
extra norm over those rows. `TapRows.Last` delivers the last row.

**`PostLayer` vs `IResidualModifier`.** `PostLayer(L)` fires **after** the layer's `IResidualModifier` on every host: a probe sees the steered
stream, not the pre-steering one. A consumer that steers and probes at the same layer gets the post-steering values.

**Rows.** `TapRows.Last` (default) delivers one sample per forward (a prefill copies one row); `TapRows.All` delivers one sample per input
row, `RowIndex` the row within the forward and `TokenPosition` its absolute position. `HiddenStateSample.SequenceId` is the request index of
a `ForwardBatch`, `Step` the model's count of tapped forwards. `BufferProvider` lets the caller own the destination row.

**Taps never write.** The residual is not modified, so a tapped forward's logits and KV are those of an untapped one (bit for bit where the
tapped and untapped runs take the same kernels; see the per-backend notes). To modify the stream use `IResidualModifier`.

**Abort.** `RegisterAbortableTap` takes a callback returning `HookResult.Continue` or `HookResult.AbortForward` (`Replace` throws
`InvalidOperationException`). On abort the remaining layers and the LM head are skipped, the KV-cache is rolled back to its length at entry
(the cache must support `Rollback`, probed before anything is written) and recurrent (GDN) state is restored from a checkpoint taken at
entry. `TryForward(...)` reports the abort as `false` plus a `ForwardAbortInfo`; the legacy `Forward(...)` throws `ForwardAbortedException`
after the same rollback (the exception implements `DotLLM.Core.Models.IForwardStateRestored`: the model already restored KV and recurrent state,
so a caller's bookkeeping needs no recovery). Use `TryForward` for per-token early exit on a hot path. `TryForward` is decode-shaped: it has no
`adapter`, `lastTokenLogitsOnly` or `mtpState` parameter (a prefill through it returns the full `[seqLen, vocab]` logits on CPU, and a LoRA
adapter cannot be applied on this path); use `Forward` with an abortable tap when you need those.

**A throwing callback (any tap).** A callback that throws (a user observer, a `BufferProvider`, the registry's own validation) leaves KV and
recurrent state as they were at forward entry on every backend: the trunk is wrapped so the KV-cache is rolled back to its entry length, the
recurrent state is restored from the entry checkpoint, and the exception is rethrown. A hybrid therefore takes the entry GDN checkpoint for a
read-only tap too (one state copy per tapped forward; a tapped forward already leaves the fused/graph paths). Two cases cannot be covered by a
checkpoint, and a throwing read-only callback there leaves the recurrent state undefined (KV is still rolled back): a forward that carries an
MTP/DFlash state, and a forward while verify-row snapshots are pending. A windowed cache that cannot truncate is best effort for a read-only tap.

**Abortable-tap refusals (all before any state is touched, identical on every backend).**
- *Cache kind, not model config (ruling 2026-11-23).* An abortable tap is refused only when the cache actually DROPS rows and so cannot roll
  back: `IKvCache.SupportsAbortRollback` is false (`SimpleKvCache`/`VulkanKvCache` with compact sliding windows, `QuantizedKvCache` and
  `CudaQuantizedKvCache` with an FP window). A model with sliding-window attention layers (e.g. Mistral) running on a **full-history** cache
  (`KvGeometry.FromConfig(config)`, rows never dropped) may use early exit. The refusal is **independent of the current length** and names the
  cache type, so a session never fails mid-generation at token N. It is raised (a) up front by
  `RegisterAbortableTap(subscription, callback, IKvCache cache)` (default interface method, also implemented on all nine hosts), which
  validates the cache kind before registering, and (b) for callers of the plain two-argument overload, at the first abortable forward. The older
  per-forward check (`IKvCache.CanRollbackAfterAppend`) stays as a backstop. Use a full-history cache or a read-only tap.
- *`ForwardBatch`.* An abortable tap is refused in `ForwardBatch` (an abort of request i would leave requests 0..i-1 committed and the rest unrun);
  call `Forward`/`TryForward` per sequence. Read-only taps are unchanged.
- *MTP/DFlash state.* An abortable tap on a forward that carries an MTP/DFlash state is refused (the checkpoint does not cover the head). A
  **read-only** tap with an MTP state works on every backend: it only reads, and is delivered after the capture.
- *Pending verify rows.* On a hybrid, an abortable forward while verify-row snapshots are pending (after `ForwardWithRecurrentSnapshots`, before
  `RestoreRecurrentStateToRow`) is refused: the entry checkpoint would commit the rejected draft rows.

**Delivery timing.**

| Backend | Read-only tap | Abortable tap |
|---------|---------------|---------------|
| CPU | at the site, in place | at the site |
| Vulkan | after the forward's last submit (host-visible staging, 64 MiB cap) | submit-and-wait at the site; a plan over the cap also goes site by site |
| CUDA | after the stream synchronize that ends the forward (async D2H into pinned staging, 64 MiB cap) | stream synchronize at the site; a plan over the cap also goes site by site |

Samples of one forward arrive in forward order: embedding, layers ascending, `PreFinalNorm`, `PreLmHead`. Callbacks run on the inference thread
before `Forward` returns.

**Zero cost unattached.** A forward reads the registry once into a local; with no tap it is a null check with no copy, sync or allocation, and
no counter moves. This is asserted against an **absolute baseline**, not against another unattached run: the untapped decode must allocate exactly
the managed bytes recorded at the merge base (d1275eef; CPU 1664 B per 16 decodes, 21632 B for the MoE hybrid; Vulkan 1872 B), and on Vulkan do
exactly the merge base's device work per 16 decodes (dispatches, copies, barriers, descriptor writes/allocs, queue submits and fence waits); on CUDA
zero tap copies are enqueued.

**Fast paths a tap bypasses.** A tap needs layer boundaries, which fused paths do not have, so a tapped forward runs the eager path and says so
once on stderr (`PathCounters` below). The bypass costs what the fast path saved; no tap-specific number is measured yet, the sources below are the
reasons the fast paths exist:

- CUDA decode graphs (dense `CudaTransformerModel` standard/quantized and MoE graphs, hybrid-dense `TryForwardDecodeGraph`, including its 2..8 row
  verify graphs). The hybrid-dense graph exists because eager decode there was launch-bound (about 27 ms GPU-busy of about 50 ms per token on an
  RTX 3060, see `CudaQwen3HybridDenseTransformerModel.DecodeGraph.cs`); a tapped decode runs those launches eagerly. A graph captured ahead is dropped.
- Vulkan fused decode on the MoE hybrid (no layer boundaries inside it).
- Packed `ForwardBatch`: CPU dense and Vulkan dense/hybrid-dense run each request on its own with a tap attached (the request index is the
  sample's `SequenceId`). CUDA has no packed-matmul batch; with a tap its deferred pinned-logits batch (one synchronize for all requests) becomes one
  synchronous forward per request.
- A long CUDA hybrid-dense prefill is still tiled (`PrefillChunkTokens`) for a last-row tap, which fires in the final tile; an every-row tap runs it
  as one tile, so its working set is no longer bounded by the tile.

**PathCounters** (`HiddenStateTapRegistry.PathCounters`, process-wide, `Reset()` for tests): `TapForwards`, `CudaGraphBypassed`,
`VulkanFusedDecodeBypassed`, `VulkanRowSliceBypassed`, `PackedBatchFallbacks`, `AbortedForwards`, `AbortCheckpoints`. The bypass notice on stderr is
logged once per kind of bypass. A bypass counter moves only when the fast path would
otherwise have been taken.

**Supported models.**

| Host | CPU | Vulkan | CUDA |
|------|-----|--------|------|
| Dense / GQA / MoE trunk | `TransformerModel` | `VulkanTransformerModel` | `CudaTransformerModel` (F16 residual; BitNet reads its F32 residual; Flint's scaled stream is unscaled on the host) |
| GDN-hybrid dense (Qwen3.5/3.6) | `Qwen3HybridDenseTransformerModel` | `VulkanQwen3HybridDenseTransformerModel` | `CudaQwen3HybridDenseTransformerModel` |
| GDN-hybrid MoE | `Qwen3MoeHybridTransformerModel` | `VulkanQwen3MoeHybridTransformerModel` | `CudaQwen3MoeHybridTransformerModel` (abortable taps need `IModel.SupportsRecurrentStateCheckpoint`, from #1919; refused with `NotSupportedException` until then) |

Not supported (registration throws `NotSupportedException`): Gemma-3n/4, Gemma / Gemma-2 on CUDA, MLA, diffusion and bidirectional towers,
trimmed pipeline stages and head-only offload instances, and an abortable tap combined with an MTP/DFlash state. Backend differences: the dense
Vulkan host refuses a tapped forward while all-row logits are widened (`TrySetAllRowLogitsLimit`, perplexity scoring); the hybrids do not. Other architectures (Mamba,
Nemotron-H, LFM2, ...) do not implement `IHiddenStateTapHost` and are refused by type.

**Why the sample is F32.** The public contract is a hidden-size F32 row, so a consumer needs no per-backend decoding. It is host-side only:
device residuals stay F16/F32 as the model stores them (the F16 ones are widened on the host after the copy), and no device F32 buffer is added
for a tap.

### Zero-Cost When Disabled

Hooks disabled by default. Implementation:
```csharp
// Hot path — simple null/flag check, no allocation
if (_hooks is not null && _hooks.HasHookAt(HookPoint.PostLayer, layer))
    _hooks.Fire(HookPoint.PostLayer, layer, activation, ctx);
```

No event invocation, no delegate allocation, no virtual dispatch when hooks are unregistered.

### Performance Impact When Enabled

Enabling hooks requires materializing activations at hook points (data may live on GPU). Cost:
- GPU → CPU copy for inspection hooks
- Memory allocation for captured tensors
- Synchronous execution on inference thread

Intended for research/debugging, not production serving.

## Logprobs

### API Support

OpenAI-compatible `logprobs` and `top_logprobs` fields on `/v1/chat/completions`:

```
Request:
  logprobs: true
  top_logprobs: 5  (1-20, default 5)

Response (per token in choices[].logprobs.content):
  token: "Hello"
  logprob: -0.0012
  top_logprobs: [
    { token: "Hello", logprob: -0.0012 },
    { token: "Hi", logprob: -2.34 },
    { token: "Hey", logprob: -3.12 },
    ...
  ]
```

Implementation: after LM head produces logits, apply log-softmax and capture top-k before sampling. No hook system dependency — operates directly on the sampling pipeline output. Works in both non-streaming and SSE streaming modes.

### Chat UI Visualization

Opt-in in the built-in web chat (`dotllm serve`):

- **Color-coded token confidence**: green (>90%), lime (>70%), yellow (>50%), orange (>30%), red (<30%)
- **Hover**: shows top-k alternative tokens with probabilities
- **Three diagnostic cues**:
  - Low confidence: token probability < 10%
  - Ambiguity: gap between top-2 tokens < 0.15
  - Sampling effect: chosen token ≠ argmax (sampling diverged from greedy)

Inspired by the logprobs visualization approach from [kokosa.dev/blog/2026/logprobs](https://kokosa.dev/blog/2026/logprobs/).

### Sample Project

`DotLLM.Sample.Logprobs` — console application demonstrating logprobs API usage with colored terminal output, confidence analysis, and ambiguity detection.

## Built-in Diagnostic Tools

### Activation Capture (`CaptureHook`)

Collects activations at specified layers/positions into a buffer:
```
var capture = new CaptureHook(HookPoint.PostLayer, layers: [0, 15, 31]);
engine.RegisterHook(capture);
// ... run inference ...
var activations = capture.GetCaptured();  // Dictionary<(layer, position), float[]>
```

Configurable: capture all tokens or specific positions, max buffer size.

### Logit Lens

Projects intermediate hidden states through the LM head at each layer:
```
For layer L:
  hidden = capture at PostLayer(L)
  logits = hidden @ lm_head_weight
  probs = softmax(logits)
  top_tokens = argmax(probs, k=10)
```

Reveals how the model's "belief" about the next token evolves through layers. Useful for understanding which layers are most important for specific predictions.

**Chat UI visualization** (opt-in in `dotllm serve`):
- Layer-by-layer prediction heatmap showing top predicted tokens at each layer
- Click a token to see how the prediction evolved through layers
- Per-layer confidence visualization (how certain the model was at each depth)

Sample project: `DotLLM.Sample.LogitLens`.

### Attention Pattern Capture

When enabled, attention mechanism exports full weight matrix (softmax output):
```
attention_weights[layer][head][query_pos][key_pos]
```

Expensive: O(n²) per head per layer. Use selectively (specific layers, specific tokens).

### Sparse Autoencoder (SAE) Integration

SAEs decompose residual stream activations into interpretable sparse features.

**Workflow**:
1. Register `Replace` hook at `PostLayer(layer)`.
2. Hook encodes activation: `features = relu(activation @ W_enc + b_enc)` → sparse vector.
3. Analyze features: log which features are active, their magnitudes.
4. Optionally modify: zero out features (ablation), amplify features (steering).
5. Decode back: `modified_activation = features @ W_dec + b_dec`.
6. Return modified activation to inference pipeline.

**SAE Loading**: Pre-trained SAEs loaded from SafeTensors. Each SAE targets a specific layer and has: encoder weight, encoder bias, decoder weight, decoder bias, feature labels (optional).

**SAE Format**: SafeTensors with `cfg.json` (SAELens convention). The `cfg.json` specifies architecture parameters (`d_in`, `d_sae`, activation function, hook point, layer index).

**SAE Discovery**: Manual path specification is the primary method. Optional curated registry: a shipped JSON file mapping base models to known SAE repositories (EleutherAI, Goodfire, Llama Scope). No unified programmatic API exists for SAE search across providers.

**Neuronpedia Integration**: [Neuronpedia](https://neuronpedia.org/) (MIT-licensed) provides auto-generated feature explanations for 50M+ features across 30+ models. `GET /api/feature/{modelId}/{source}/{index}` returns explanation text, top/bottom logits, and activation examples (no auth required). `POST /api/search-topk-by-token` finds top-k active features per token. Bulk S3 export available for offline label cache. Covers Llama 3.1-8B, Gemma 2/3, Qwen 2.5/3, DeepSeek R1. Integration is optional — dotLLM works offline with local SAEs; Neuronpedia enriches with human-readable labels when available.

```
ISparseAutoencoder:
  Encode(activation) → SparseFeatures
  Decode(features) → activation
  FeatureCount → int
```

Sample project: `DotLLM.Sample.Interpretability` demonstrates full workflow.

**Chat UI visualization** (opt-in in `dotllm serve`):
- SAE configuration panel: load SAE for a specific layer, browse active features and their labels
- Feature visualization in generated text: highlight tokens by active feature magnitudes
- Feature steering controls: amplify or suppress specific features directly from the UI
