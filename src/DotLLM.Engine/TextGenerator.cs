using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Sampling;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Engine.Strategies;
using DotLLM.Telemetry;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Autoregressive text generator: encodes a prompt, runs prefill + decode loop
/// with sampling and stop conditions, and returns the generated text.
/// </summary>
public sealed partial class TextGenerator
{
    private readonly IModel _model;
    private readonly ITokenizer _tokenizer;
    private readonly Func<ModelConfig, int, Core.Attention.IKvCache>? _kvCacheFactory;
    private readonly PrefixCache? _prefixCache;
    private readonly PrefixTrieManager? _prefixTrieManager;
    private readonly IModel? _draftModel;
    private readonly Func<ModelConfig, int, Core.Attention.IKvCache>? _draftKvCacheFactory;
    private readonly int _speculativeCandidates;
    private readonly int _mtpCandidates;   // MTP depth: the model's preferred K when the caller kept the library default
    private readonly bool _mtpEnabled;
    private readonly HybridPrefillDecodeStrategy? _hybridStrategy;
    private readonly int _prefillChunkSize;
    private readonly MtpAdaptiveGate? _mtpGate;
    private readonly MtpAdaptiveGate? _mtpGateSampled;   // sampled requests accept fewer drafts and pay the sampler per verify row: own arms and cost model

    // Single-slot prefix cache for recurrent (GDN/SSM) models: tokens + the KV cache holding them + the recurrent state
    // checkpointed right after them. See ResolveRecurrentPrefix.
    private readonly bool _recurrentPrefixEnabled;
    private RecurrentPrefixEntry? _recurrentPrefix;
    // The previous request's prompt. Written by whichever recurrent path is active: the single-slot path (shared-prefix and
    // extension boundaries) or the tiered path (exact-extension check, issue #1093). Only one of them runs per generator.
    private int[]? _recurrentPrefixLastPrompt;
    private int? _recurrentPrefixLastSuffixBoundary;   // _recurrentPrefixLastPrompt's generation-prompt boundary (issue #1093)

    // True from ResolveKvCache until the request reaches a normal end (StoreInPrefixCache). Still set in the request's finally
    // means it failed or was abandoned, so the snapshot is dropped (PR #1026 re-review).
    private bool _recurrentPrefixRequestPending;

    /// <summary>Shortest shared prefix worth snapshotting (tokens).</summary>
    public const int RecurrentPrefixMinTokens = 16;

    /// <summary>Extra KV positions allocated beyond the request when a prefix snapshot is taken, so a slightly longer later prompt can still reuse it.</summary>
    private const int RecurrentPrefixKvHeadroom = 256;

    /// <summary>Most positions a recurrent-prefix KV growth adds beyond the request (PR #1026 review: the CUDA hybrid's storage never shrinks).</summary>
    private const int RecurrentPrefixMaxGrowthHeadroom = 2048;

    /// <summary>
    /// Longest prompt (tokens) for which a recurrent-prefix hit whose KV cannot grow, even to the exact size, may still fall
    /// back to a miss that re-prefills the whole prompt (issue #1093). At or above it the request fails with
    /// <see cref="DeviceMemoryAdmissionException"/> (HTTP 400 <c>context_length_exceeded</c>) instead.
    /// </summary>
    /// <remarks>
    /// Why 4096: a miss is still worth trying for a short prompt, because it frees before it allocates (a copying growth holds
    /// the old and the new storage at once), so it can fit where the growth did not. And re-prefilling 4k tokens costs about
    /// 30 s at the slowest measured long-context rate (Bonsai-2 on an RTX 3060, 110-150 tok/s), which is less than one turn of
    /// a long session already spends on its own new content (about 8k tokens). Past that the cost grows with the prompt and
    /// stops being a fallback: at 112k it was a silent 15-minute turn (916 s). Compared against the prompt length, the tokens a
    /// miss would re-prefill.
    /// </remarks>
    internal const int RecurrentPrefixRefillMissMaxTokens = 4096;

    // The head state that decoded the request which took the snapshot, when it decoded with MTP: its rows [0, Tokens.Length)
    // are the prefix's, so a hit resumes it at MtpMark and drafts without re-absorbing the prefix (issue #1088).
    private IMtpState? _resumedMtp;

    // Whether this model's MTP head states can be marked and resumed (IMtpState.MarkPrefix), learned from the first state made;
    // null until then. A plain turn never feeds a head that no hit could resume (PR #1091 review).
    private bool? _mtpStatesResumable;

    private sealed class RecurrentPrefixEntry(int[] tokens, Core.Attention.IKvCache kv, object? checkpoint, IMtpState? mtp, object? mtpMark)
    {
        public readonly int[] Tokens = tokens;
        public readonly Core.Attention.IKvCache Kv = kv;
        public readonly object? Checkpoint = checkpoint;

        /// <summary>The MTP head state whose rows <c>[0, Tokens.Length)</c> hold the prefix, or null (the request decoded plain).</summary>
        public readonly IMtpState? Mtp = mtp;

        /// <summary>The <see cref="IMtpState.MarkPrefix"/> mark of <see cref="Mtp"/> at the snapshot boundary.</summary>
        public readonly object? MtpMark = mtpMark;

        public void Dispose() => Release(keepKv: null, keepMtp: null);

        /// <summary>
        /// Releases the checkpoint, and the KV cache and head state unless a newer entry or the running request took them over.
        /// </summary>
        /// <param name="keepKv">A KV cache that stays in use: not disposed when it is this entry's.</param>
        /// <param name="keepMtp">A head state that stays in use: not disposed when it is this entry's.</param>
        public void Release(Core.Attention.IKvCache? keepKv, IMtpState? keepMtp)
        {
            (Checkpoint as IDisposable)?.Dispose();
            if (Mtp is not null && !ReferenceEquals(Mtp, keepMtp)) Mtp.Dispose();
            if (!ReferenceEquals(Kv, keepKv)) Kv.Dispose();
        }
    }

    /// <summary>
    /// Default draft tokens per speculative round (K), for MTP self-speculation and for a
    /// separate draft model. Issue #466: Bonsai 2 27B MTP peaks at K=3 on both CPU and Vulkan
    /// (Vulkan tok/s at K=2/3/4: 22.46/23.95/21.86). K=5 also held ~750 MiB of the #473
    /// per-row recurrent snapshots, against ~450 MiB at K=3. The CLI and server defaults
    /// reference this constant.
    /// </summary>
    public const int DefaultSpeculativeCandidates = 3;

    /// <summary>
    /// Creates a new text generator.
    /// </summary>
    /// <param name="model">The model to use for forward passes.</param>
    /// <param name="tokenizer">The tokenizer for encoding/decoding text.</param>
    /// <param name="kvCacheFactory">Optional factory for creating a KV-cache. When null, uses <see cref="SimpleKvCache"/>.
    /// Parameters: (config, maxSeqLen).</param>
    /// <param name="prefixCache">Optional prefix cache for reusing KV-cache state across calls.
    /// When provided, the KV-cache is kept alive between calls and only new suffix tokens are prefilled.</param>
    /// <param name="draftModel">Optional draft model for speculative decoding.</param>
    /// <param name="draftKvCacheFactory">Optional factory for creating the draft model's KV-cache.</param>
    /// <param name="speculativeCandidates">Number of draft tokens per speculative step (K). Default 5.
    /// Shared with MTP self-speculative decoding's own K (<paramref name="mtpEnabled"/>) — both are the
    /// same "candidates per round" concept, just drafted by a second model vs. the target's own head.</param>
    /// <param name="mtpEnabled">
    /// Enables Multi-Token Prediction (MTP) self-speculative decoding (issue #253) when
    /// <paramref name="model"/> reports <see cref="IModel.SupportsMtp"/> — the model's own
    /// lightweight extra head drafts candidates from its own hidden state, no second model needed.
    /// Defaults to <see langword="true"/>: MTP is auto-detected purely from the loaded checkpoint
    /// (mirrors this project's other GGUF-content-driven auto-behavior, e.g. hybrid-architecture
    /// dispatch) and is a no-op for every model that doesn't carry an MTP head, so the default is
    /// safe for all existing callers. Ignored (falls through to the standard or two-model
    /// speculative loop) whenever <paramref name="draftModel"/> is also supplied — the two are
    /// mutually exclusive per sequence; an explicit draft model takes priority. Only engages when
    /// decoding is effectively greedy and logprobs aren't requested, same gate as the two-model
    /// path (see <see cref="IsEffectivelyGreedy"/>).</param>
    /// <param name="hybridStrategy">Optional CPU-prefill / GPU-decode hybrid strategy. When set
    /// and the prompt length is below the strategy's crossover threshold, prefill runs on the
    /// strategy's CPU model and the KV state is handed off to <paramref name="model"/> (the
    /// decode model) before the decode loop. When the prompt exceeds the threshold, or when
    /// the strategy is null, the existing single-backend path runs unchanged.</param>
    /// <param name="prefixTrieManager">Optional cross-request prefix trie manager (Step 37).
    /// Takes precedence over <paramref name="prefixCache"/> when supplied — multiple sessions
    /// share KV blocks via the trie.</param>
    /// <param name="recurrentPrefixCache">When true and the model is recurrent with checkpoint support (Gated DeltaNet hybrids),
    /// the longest prefix a request shares with the previous one is snapshotted (KV + recurrent state) so later requests that
    /// start with it prefill only their suffix. Single slot; see <see cref="ClearRecurrentPrefixCache"/>.</param>
    /// <param name="mtpAdaptive">When true (and <paramref name="mtpEnabled"/>), MTP self-speculation engages only when
    /// measured faster than plain decode for this model and the request is long enough to benefit (see
    /// <see cref="MtpAdaptiveGate"/>). Default false: MTP engages whenever enabled and supported, as before.</param>
    /// <param name="prefillChunkSize">Maximum prompt tokens per prefill forward pass (llama.cpp
    /// <c>-ub</c> / micro-batch analog). When &gt; 0 the prompt is prefilled in chunks of at most
    /// this many tokens (bounding peak activation memory per forward); 0 (default) runs the whole
    /// uncached suffix in a single forward pass — behavior unchanged. Ignored on the hybrid
    /// CPU-prefill/GPU-decode path, whose <see cref="HybridPrefillDecodeStrategy.RunPrefill"/>
    /// processes the full prompt itself.</param>
    /// <param name="recurrentPrefixCacheOptions">Optional independent recurrent snapshot count and pinned host memory budget.</param>
    public TextGenerator(IModel model, ITokenizer tokenizer,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? kvCacheFactory = null,
                          PrefixCache? prefixCache = null,
                          IModel? draftModel = null,
                          Func<ModelConfig, int, Core.Attention.IKvCache>? draftKvCacheFactory = null,
                          int speculativeCandidates = DefaultSpeculativeCandidates,
                          bool mtpEnabled = true,
                          HybridPrefillDecodeStrategy? hybridStrategy = null,
                          PrefixTrieManager? prefixTrieManager = null,
                          int prefillChunkSize = 0,
                          bool mtpAdaptive = false,
                          bool recurrentPrefixCache = false,
                          RecurrentPrefixCacheOptions? recurrentPrefixCacheOptions = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _kvCacheFactory = kvCacheFactory;
        _prefixCache = prefixCache;
        _prefixTrieManager = prefixTrieManager;
        _draftModel = draftModel;
        _draftKvCacheFactory = draftKvCacheFactory;
        _speculativeCandidates = speculativeCandidates;
        _mtpCandidates = speculativeCandidates == DefaultSpeculativeCandidates && model.MtpPreferredDraftTokens > 0
            ? Math.Min(model.MtpPreferredDraftTokens, MtpCostModel.MaxK) : speculativeCandidates;
        _mtpEnabled = mtpEnabled;
        _hybridStrategy = hybridStrategy;
        _prefillChunkSize = prefillChunkSize;
        _recurrentPrefixEnabled = recurrentPrefixCache && draftModel is null
            && model.RequiresPerSequenceState && model.SupportsRecurrentStateCheckpoint;
        var snapshots = recurrentPrefixCacheOptions ?? new RecurrentPrefixCacheOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(snapshots.MaxEntries, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(snapshots.HostBudgetBytes);
        if (_recurrentPrefixEnabled && model is IRecurrentPrefixSnapshotProvider provider && (snapshots.MaxEntries > 1 || snapshots.HostBudgetBytes > 0 || snapshots.DiskBudgetBytes > 0))
            _tieredSnapshots = new RecurrentPrefixSnapshotCache(provider, snapshots);
        _mtpGate = mtpAdaptive && MtpAdaptiveGate.EnabledByEnvironment ? new MtpAdaptiveGate(model.MtpGatePriorBytes) : null;
        _mtpGateSampled = _mtpGate is null ? null : new MtpAdaptiveGate(model.MtpGatePriorBytes);

        if (hybridStrategy is not null
            && !ReferenceEquals(hybridStrategy.DecodeModel, model))
        {
            throw new ArgumentException(
                "When a HybridPrefillDecodeStrategy is supplied, its DecodeModel must be the same "
                + "instance as the TextGenerator's primary model (which runs the decode loop).",
                nameof(hybridStrategy));
        }
    }

    /// <summary>
    /// Configured prefill chunk size (llama.cpp <c>-ub</c> analog). 0 = single-pass prefill.
    /// </summary>
    public int PrefillChunkSize => _prefillChunkSize;

    /// <summary>
    /// Chunk size the prompt prefill actually uses: the configured <see cref="PrefillChunkSize"/>, else the model's
    /// <see cref="IModel.PreferredPrefillChunkTokens"/> (0 = one forward pass).
    /// </summary>
    internal int EffectivePrefillChunkSize => _prefillChunkSize > 0 ? _prefillChunkSize : Math.Max(0, _model.PreferredPrefillChunkTokens);

    /// <summary>
    /// Generates text from the given prompt using the specified options.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="onTokenGenerated">
    /// Optional callback invoked after each token is generated, receiving the token ID.
    /// <para>
    /// <b>This is the pre-trim, token-level stream.</b> It is a progress hook, not an output
    /// stream: the token that triggers a stop condition is never passed to it, and a token
    /// whose text is only <em>partly</em> emitted (e.g. it decodes to
    /// <c>"ld&lt;|im_end|&gt;"</c> for stop string <c>"&lt;|im_end|&gt;"</c>) cannot be
    /// expressed by an <see cref="Action{T}"/> over token IDs at all. Handing the ID over
    /// would leak the stop string into the consumer's text, which
    /// <see cref="StopStringCondition"/> exists to prevent.
    /// </para>
    /// <para>
    /// Consequence: re-rendering these IDs can produce a strict prefix of
    /// <see cref="InferenceResponse.Text"/>. Keep using this callback for things that are
    /// genuinely about token identity (logprob correlation, speculative-decoding
    /// instrumentation). For driving output text, use <paramref name="onTextGenerated"/>,
    /// which reproduces <see cref="InferenceResponse.Text"/> exactly.
    /// </para>
    /// </param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <param name="onTextGenerated">
    /// Optional <b>text-level</b> streaming callback (issue #424). Receives the decoded text
    /// fragments of the output, in order; concatenating every fragment reproduces
    /// <see cref="InferenceResponse.Text"/> exactly, including a final token whose
    /// stop-string suffix was trimmed at a character boundary, and never including a stop
    /// string. Fragments do not correspond one-to-one with tokens: text that could still turn
    /// out to be the start of a stop string is withheld until it is known to be output, so a
    /// single call may carry several tokens' worth of characters, or none.
    /// <para>
    /// The span is only valid for the duration of the call — copy it if you need to retain it.
    /// </para>
    /// </param>
    /// <param name="cancellationToken">
    /// Cancels generation cooperatively: between prefill chunks (see <see cref="IModel.PreferredPrefillChunkTokens"/>)
    /// and between decode steps. Throws <see cref="OperationCanceledException"/>.
    /// </param>
    /// <returns>The inference response with generated text, metadata, and timings.</returns>
    private InferenceResponse GenerateCore(string prompt, InferenceOptions? options = null,
        Action<int>? onTokenGenerated = null,
        ILoraAdapter? adapter = null,
        Action<ReadOnlySpan<char>>? onTextGenerated = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new InferenceOptions();

        int[] promptIds = _tokenizer.Encode(prompt);
        int promptLen = promptIds.Length;
        int maxTokens = options.MaxTokens;
        int vocabSize = _model.Config.VocabSize;

        // Guard: empty prompt — use BOS token as seed
        if (promptLen == 0)
        {
            promptIds = [_tokenizer.BosTokenId];
            promptLen = 1;
        }

        // Guard: MaxTokens=0 — return immediately, no generation
        if (maxTokens <= 0)
        {
            return new InferenceResponse
            {
                GeneratedTokenIds = [],
                Text = string.Empty,
                FinishReason = FinishReason.Length,
                PromptTokenCount = promptLen,
                GeneratedTokenCount = 0
            };
        }

        var telemetry = new TelemetryRecorder(_model.Config, options);

        // Build sampling pipeline
        var pipeline = new SamplerPipeline(options, _tokenizer);

        // Logprobs capture setup
        bool captureLogprobs = options.Logprobs;
        int topLogprobs = Math.Clamp(options.TopLogprobs, 0, 20);
        List<TokenLogprobInfo>? logprobsList = captureLogprobs ? new List<TokenLogprobInfo>(maxTokens) : null;

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => RegexConstraint.Create(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null
        };

        // Build stop conditions — use explicit list if provided, otherwise default set
        List<IStopCondition> stopConditions;
        if (options.StopConditions is not null)
        {
            stopConditions = new List<IStopCondition>(options.StopConditions);
        }
        else
        {
            stopConditions = new List<IStopCondition>
            {
                EndOfGenerationTokens.CreateStopCondition(_tokenizer),
                new MaxTokensStopCondition(maxTokens)
            };
            // StopStringCondition excludes the triggering token. Partial-token
            // suffix trimming would require text-level stop metadata alongside
            // token-level finish semantics.
            stopConditions.AddRange(StopStringCondition.CreateAll(options));
        }

        // Resolve KV-cache: reuse from prefix cache or allocate fresh
        var (kvCache, cachedTokenCount, ownsKvCache) = ResolveKvCache(promptIds, promptLen, maxTokens, out int snapshotAt, options.GenerationPromptSuffix, adapter);
        _recurrentPrefixRequestPending = _recurrentPrefixEnabled;

        // Hybrid mode is enabled when a strategy is wired up, the prompt is short enough,
        // and we have a clean cache (no prefix-cache reuse, no speculative draft model, no MTP).
        // MTP needs the whole sequence absorbed into its head from position 0 (issue #469). A
        // prefix-cache hit skips the cached prefix's forward, so it decodes with MTP only when the
        // snapshot kept the head state that absorbed that prefix (issue #1088).
        var mtpGate = GateFor(pipeline.IsStochastic);
        var resumedMtp = TakeResumedMtpState();
        bool useMtp = ShouldUseMtp(captureLogprobs) && (cachedTokenCount == 0 || resumedMtp is not null)
            && (mtpGate?.ShouldUseMtp(maxTokens) ?? true);
        TraceGate(mtpGate, pipeline.IsStochastic, useMtp, captureLogprobs, cachedTokenCount);
        if (useMtp)
            _model.TrySetAllRowLogitsLimit(_mtpCandidates + 1);
        bool useHybrid = _hybridStrategy is not null
            && _hybridStrategy.ShouldRunHybrid(promptLen)
            && cachedTokenCount == 0
            && _draftModel is null
            && !useMtp;

        // Stop-check scratch buffer: rented up-front and returned in the outer finally to preserve
        // the zero-GC-pressure guarantee on the inference hot path.
        int stopTailSize = ComputeStopTailSize(stopConditions);
        char[] stopScratch = ArrayPool<char>.Shared.Rent(stopTailSize);
        DotLLM.Core.Models.IMtpState? mtpState = AcquireRequestMtpState(useMtp, snapshotAt, cachedTokenCount, resumedMtp, useHybrid, mtpGate, kvCache);

        try
        {
            var generatedIds = new List<int>(maxTokens);
            var finishReason = FinishReason.Length;
            long prefillTicks = 0;
            long decodeTicks = 0;
            long samplerTicks = 0;
            int cacheSize = kvCache.MaxLength;

            // Incremental detokenizer keeps stop-check cost O(1) amortized per token
            // instead of decoding the entire generated sequence each step (O(n²)).
            var detok = new IncrementalDetokenizer(_tokenizer, initialCapacity: Math.Max(64, maxTokens * 4));

            // Text-level streaming (issue #424). Inert unless onTextGenerated is supplied, so the
            // id-only path keeps its allocation profile. Sits alongside the id callback rather than
            // replacing it: only this one can express a final token whose stop-string suffix was
            // trimmed at a character boundary.
            var textEmitter = new StopAwareTextEmitter(onTextGenerated, stopConditions);

            // Emits the freshly decoded text of a token that stays in the output. A UTF-8 character
            // still split across tokens is held back until it completes (#1263).
            void EmitText()
            {
                if (textEmitter.IsActive)
                    textEmitter.Append(detok.TakeDelta(flushIncomplete: false));
            }

            // Terminal emission for endings that keep every generated token (length limit, KV-cache
            // exhaustion, a terminal constraint): releases a dangling incomplete character, then
            // anything withheld as a possible stop-string prefix. Idempotent once both are drained.
            void FlushText()
            {
                if (textEmitter.IsActive)
                    textEmitter.Append(detok.TakeDelta(flushIncomplete: true));
                textEmitter.Flush();
            }

            // Terminal emission. The three endings differ in what the last token contributes:
            //  • Stop + stop-string match — the token stays in the id list but only its prefix is
            //    output, so append its text and trim the matched suffix off. This is precisely the
            //    case the id-level callback cannot express (issue #424).
            //  • Stop without a stop-string match (EOS and friends) — the token is dropped from the
            //    id list, so its text is never emitted; only previously withheld text is released
            //    (including an incomplete character held back before the dropped token, #1263).
            //  • StopInclude — the token is part of the output in full.
            void EmitTextForStop(StopResult result, bool isStopStringMatch)
            {
                if (!textEmitter.IsActive)
                    return;

                if (result == StopResult.Stop && !isStopStringMatch)
                {
                    textEmitter.Append(detok.TakeWithheld());
                    textEmitter.Flush();
                    return;
                }

                if (result == StopResult.Stop)
                {
                    textEmitter.FlushTrimmingStopSuffix(detok.TakeDelta(flushIncomplete: true));
                }
                else
                {
                    textEmitter.Append(detok.TakeDelta(flushIncomplete: true));
                    textEmitter.Flush();
                }
            }

            // Local helper: snapshot log-softmax before sampling (which modifies logits in-place),
            // sample a token, then build logprob info.
            (int tokenId, TokenLogprobInfo? logprob) SampleWithLogprobs(Span<float> logitSpan)
            {
                float[]? lsBuf = captureLogprobs ? LogprobsCapture.ComputeLogSoftmax(logitSpan) : null;
                int tokenId = pipeline.Sample(logitSpan, generatedIds);
                TokenLogprobInfo? info = null;
                if (lsBuf != null)
                {
                    info = LogprobsCapture.BuildInfo(lsBuf.AsSpan(0, vocabSize), vocabSize, tokenId, topLogprobs, _tokenizer);
                    ArrayPool<float>.Shared.Return(lsBuf);
                }
                return (tokenId, info);
            }

            // Prefill: run only new suffix tokens through the model
            int prefillStart = cachedTokenCount;
            int prefillLen = promptLen - prefillStart;

            int firstTokenId;
            long ts0 = Stopwatch.GetTimestamp();

            using (var prefillSpan = telemetry.StartPrefill())
            {
                if (useHybrid)
                {
                    // ── Hybrid prefill: CPU model populates a SimpleKvCache, then we hand
                    //    off into the decode-side cache (kvCache) and sample the first
                    //    token from the CPU-produced logits. The decode loop below sees a
                    //    fully populated decode KV cache exactly as if pure-GPU prefill ran.
                    var handoff = _hybridStrategy!.RunPrefill(promptIds.AsSpan(0, promptLen), cacheSize);
                    try
                    {
                        _hybridStrategy.Handoff(handoff.HostCache, kvCache);
                        prefillTicks = handoff.PrefillTicks;

                        using var sampleSpan = telemetry.StartSample();
                        long samplerStart = Stopwatch.GetTimestamp();
                        var logitSpan = handoff.LastLogits.AsSpan(0, vocabSize);
                        if (constraint != null)
                            TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                        var (tid, lp) = SampleWithLogprobs(logitSpan);
                        firstTokenId = tid;
                        if (lp.HasValue) logprobsList!.Add(lp.Value);
                        samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                    }
                    finally
                    {
                        handoff.HostCache.Dispose();
                    }
                }
                else if (prefillLen > 0)
                {
                    // Prefill suffix tokens — chunked when a prefill chunk size is configured
                    // (llama.cpp -ub analog); otherwise a single forward pass, as before.
                    using (ITensor prefillLogits = ForwardPrefill(promptIds, prefillStart, prefillLen, kvCache, adapter, mtpState,
                        snapshotAt, () => { TakeRecurrentSnapshot(promptIds, snapshotAt, kvCache, mtpState); if (_tieredSnapshots is null) ownsKvCache = false; },
                        cancellationToken))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            // GPU/hybrid models return [1, vocabSize] (last token only);
                            // CPU model returns [seqLen, vocabSize]. Use actual shape to index.
                            float* logitPtr = (float*)prefillLogits.DataPointer;
                            int logitRows = prefillLogits.Shape[0];
                            var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            var (tid, lp) = SampleWithLogprobs(logitSpan);
                            firstTokenId = tid;
                            if (lp.HasValue) logprobsList!.Add(lp.Value);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }
                }
                else if (promptLen > 0)
                {
                    // 100% cache hit — re-forward last prompt token to get logits
                    using (ITensor logits = _model.Forward([promptIds[^1]], [promptLen - 1], deviceId: -1, kvCache, adapter))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            var (tid, lp) = SampleWithLogprobs(logitSpan);
                            firstTokenId = tid;
                            if (lp.HasValue) logprobsList!.Add(lp.Value);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }
                }
                else
                {
                    // Unreachable: empty prompt guard ensures promptLen >= 1
                    throw new InvalidOperationException("Prompt is empty after guard.");
                }

                if (prefillSpan is { IsAllDataRequested: true })
                {
                    prefillSpan.SetTag(TelemetryTags.PrefillTokenCount, prefillLen);
                    prefillSpan.SetTag(TelemetryTags.PrefillDurationMs, prefillTicks * 1000.0 / Stopwatch.Frequency);
                }
            }

            telemetry.RecordFirstToken();
            constraint?.Advance(firstTokenId);

            // Check stop conditions for first token
            generatedIds.Add(firstTokenId);
            detok.Append(firstTokenId);

            ReadOnlySpan<char> firstTail = detok.GetTailView(stopTailSize, stopScratch);
            var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds, firstTail);
            if (stopResult != StopResult.Continue)
            {
                bool isStopStringMatch = HasStopStringSuffix(firstTail, stopConditions);
                if (stopResult == StopResult.Stop)
                {
                    if (!isStopStringMatch)
                        generatedIds.RemoveAt(generatedIds.Count - 1);
                    // Stop-string match: keep the last token in the id list; BuildResponse
                    // will trim the matched stop-string suffix at the character boundary.
                }
                else
                    onTokenGenerated?.Invoke(firstTokenId);

                EmitTextForStop(stopResult, isStopStringMatch);

                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                telemetry.Complete(promptLen, cachedTokenCount, generatedIds.Count,
                    prefillTicks * 1000.0 / Stopwatch.Frequency,
                    decodeTicks * 1000.0 / Stopwatch.Frequency,
                    finishReason);
                return BuildResponse(promptLen, generatedIds, finishReason,
                    prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache), cachedTokenCount,
                    logprobs: logprobsList?.ToArray(),
                    stopConditionsForSuffixTrim: stopConditions);
            }

            onTokenGenerated?.Invoke(firstTokenId);
            EmitText();

            // A complete constraint that only permits EOS (e.g. a one-letter answer) has nothing left to say:
            // stop now instead of spending a forward pass to sample the forced EOS (~23 ms on Tev1-4B/Vulkan).
            if (ConstraintIsTerminal(constraint))
            {
                FlushText();
                StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                telemetry.Complete(promptLen, cachedTokenCount, generatedIds.Count,
                    prefillTicks * 1000.0 / Stopwatch.Frequency,
                    decodeTicks * 1000.0 / Stopwatch.Frequency,
                    FinishReason.Stop);
                return BuildResponse(promptLen, generatedIds, FinishReason.Stop,
                    prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache), cachedTokenCount,
                    logprobs: logprobsList?.ToArray(),
                    stopConditionsForSuffixTrim: stopConditions);
            }

            int specDrafted = 0, specAccepted = 0;
            bool useDraftSpec = _draftModel != null && !captureLogprobs && IsEffectivelyGreedy(options);
            int mtpFallbackStep = 0;                // > 0: the MTP rounds measured slower than plain decode; plain decode resumes at this step (#900)
            long mtpTicks = 0; int mtpTokens = 0;   // MTP-segment wall ticks / committed tokens, for the adaptive gate

            // Decode loop (speculative decode disabled when logprobs requested — no per-position logit access;
            // also disabled when sampling isn't effectively greedy, since non-greedy acceptance is not yet
            // distributionally correct under the sampler pipeline — see Wave 8 / issue #121).
            if (useDraftSpec)
            {
                // ── Speculative decode loop ──
                var specDecoder = new SpeculativeDecoder(
                    greedy: true, seed: options.Seed);
                Core.Attention.IKvCache draftKvCache = AllocateDraftKvCache(cacheSize);
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_speculativeCandidates + 1);
                try
                {
                    // Prefill draft model with prompt
                    PrefillDraftModel(promptIds, draftKvCache);

                    int step = 1;
                    while (step < maxTokens)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        int remaining = maxTokens - step;
                        int k = Math.Min(_speculativeCandidates, remaining);

                        var result = specDecoder.DraftAndVerify(
                            _model, _draftModel!, kvCache, draftKvCache,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, _draftModel!.Config.VocabSize, k, specBuffer);

                        if (result.AcceptedCount == 0) break;

                        decodeTicks += result.DraftTicks + result.VerifyTicks;
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        // Only count tokens that actually make it into output (stop conditions may discard some).
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            detok.Append(tokenId);

                            ReadOnlySpan<char> specTail = detok.GetTailView(stopTailSize, stopScratch);
                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds, specTail);
                            if (stopResult != StopResult.Continue)
                            {
                                bool isStopStringMatch = HasStopStringSuffix(specTail, stopConditions);
                                if (stopResult == StopResult.Stop)
                                {
                                    if (!isStopStringMatch)
                                        generatedIds.RemoveAt(generatedIds.Count - 1);
                                    // Stop-string match: keep the last token; trim suffix in BuildResponse.
                                }
                                else
                                {
                                    specAccepted++;
                                    onTokenGenerated?.Invoke(tokenId);
                                }

                                EmitTextForStop(stopResult, isStopStringMatch);

                                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                shouldBreak = true;
                                break;
                            }

                            specAccepted++;
                            onTokenGenerated?.Invoke(tokenId);
                            EmitText();
                            step++;
                        }

                        if (shouldBreak) break;
                    }
                }
                finally
                {
                    draftKvCache.Dispose();
                    ArrayPool<int>.Shared.Return(specBuffer);
                }
            }
            else if (useMtp)
            {
                // ── MTP self-speculative decode loop (issue #253): the target model's own
                //    lightweight head drafts candidates from its own hidden state — no second
                //    model, no second full-model KV-cache, just the model's own IMtpState. ──
                var mtpDecoder = new MtpSpeculativeDecoder();
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_mtpCandidates + 1);
                var mtpCtl = mtpGate is { } mtpG ? new MtpRoundController(mtpG.CostModel, _mtpCandidates, calibratePlain: mtpG.NeedsPlainReference(maxTokens - 1), gate: mtpG) : null;
                try
                {
                    int step = 1;
                    while (step < maxTokens)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        // The measured MTP round cost per token is above plain decode's: finish this request on the plain loop below.
                        if (mtpCtl is not null && mtpCtl.ShouldFallBack(mtpGate!.PlainMsPerToken)) { mtpFallbackStep = step; break; }

                        int remaining = maxTokens - step;
                        int k = mtpCtl?.NextK(remaining) ?? Math.Min(_mtpCandidates, remaining);

                        var result = mtpDecoder.DraftAndVerify(
                            _model, kvCache, mtpState!,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, k, specBuffer, allowZeroDraft: true);

                        if (result.AcceptedCount == 0) break;

                        // Whole-round wall time (draft + verify + restore/absorb): the draft + verify sum omits the post-verify state restore.
                        decodeTicks += result.RoundTicks;
                        // Zero-draft calibration rounds are plain decode steps: they stay out of the MTP segment, so the end-of-request record
                        // hands them to the gate as the plain sample.
                        if (result.DraftedCount > 0)
                        {
                            mtpTicks += result.RoundTicks;
                            mtpTokens += result.AcceptedCount;
                        }
                        mtpCtl?.Record(result.DraftedCount, result.AcceptedCount, result.RoundTicks * 1000.0 / Stopwatch.Frequency);
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            detok.Append(tokenId);

                            // The MTP loop postdates #296, so it never received that PR's
                            // keep-the-token treatment; apply it here too, otherwise this path
                            // alone still drops the whole partially-output final token.
                            ReadOnlySpan<char> mtpTail = detok.GetTailView(stopTailSize, stopScratch);
                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds, mtpTail);
                            if (stopResult != StopResult.Continue)
                            {
                                bool isStopStringMatch = HasStopStringSuffix(mtpTail, stopConditions);
                                if (stopResult == StopResult.Stop)
                                {
                                    if (!isStopStringMatch)
                                        generatedIds.RemoveAt(generatedIds.Count - 1);
                                    // Stop-string match: keep the last token; trim suffix in BuildResponse.
                                }
                                else
                                {
                                    specAccepted++;
                                    onTokenGenerated?.Invoke(tokenId);
                                }

                                EmitTextForStop(stopResult, isStopStringMatch);

                                finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                shouldBreak = true;
                                break;
                            }

                            specAccepted++;
                            onTokenGenerated?.Invoke(tokenId);
                            EmitText();
                            step++;
                        }

                        if (shouldBreak) break;
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(specBuffer);
                }
            }

            // Plain decode: the default, and the tail of a request whose MTP rounds fell back (mtpFallbackStep).
            if (!useDraftSpec && (!useMtp || mtpFallbackStep > 0))
            {
                // ── Standard decode loop: one token at a time ──
                for (int step = mtpFallbackStep > 0 ? mtpFallbackStep : 1; step < maxTokens; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int pos = promptLen + step - 1;
                    if (pos >= cacheSize)
                        break;

                    using var decodeStepSpan = telemetry.StartDecodeStep(step);

                    int lastToken = generatedIds[^1];
                    int nextTokenId;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using (ITensor logits = _model.Forward([lastToken], [pos], deviceId: -1, kvCache, adapter))
                    {
                        decodeTicks += Stopwatch.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            var (tid, lp) = SampleWithLogprobs(logitSpan);
                            nextTokenId = tid;
                            if (lp.HasValue) logprobsList!.Add(lp.Value);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }

                    constraint?.Advance(nextTokenId);

                    generatedIds.Add(nextTokenId);
                    detok.Append(nextTokenId);

                    ReadOnlySpan<char> decTail = detok.GetTailView(stopTailSize, stopScratch);
                    stopResult = CheckStopConditions(stopConditions, nextTokenId, generatedIds, decTail);
                    if (stopResult != StopResult.Continue)
                    {
                        bool isStopStringMatch = HasStopStringSuffix(decTail, stopConditions);
                        if (stopResult == StopResult.Stop)
                        {
                            if (!isStopStringMatch)
                                generatedIds.RemoveAt(generatedIds.Count - 1);
                            // Stop-string match: keep the last token; trim suffix in BuildResponse.
                        }
                        else
                            onTokenGenerated?.Invoke(nextTokenId);

                        EmitTextForStop(stopResult, isStopStringMatch);

                        finishReason = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                        break;
                    }

                    onTokenGenerated?.Invoke(nextTokenId);
                    EmitText();

                    if (ConstraintIsTerminal(constraint))
                    {
                        finishReason = FinishReason.Stop;
                        break;
                    }
                }
            }

            // Releases text still withheld because it could have begun a stop string, for every
            // ending that did not go through EmitTextForStop (length limit, KV-cache exhaustion,
            // a speculative round accepting nothing), plus any incomplete character held back.
            // Idempotent once the buffers are empty.
            FlushText();

            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
            telemetry.Complete(promptLen, cachedTokenCount, generatedIds.Count,
                prefillTicks * 1000.0 / Stopwatch.Frequency,
                decodeTicks * 1000.0 / Stopwatch.Frequency,
                finishReason);
            RecordMtpGate(mtpGate, useMtp, generatedIds.Count, mtpTicks, mtpTokens, decodeTicks, samplerTicks);
            return BuildResponse(promptLen, generatedIds, finishReason,
                prefillTicks, decodeTicks, samplerTicks, GetKvCacheBytes(kvCache), cachedTokenCount,
                specDrafted, specAccepted, logprobsList?.ToArray(),
                stopConditionsForSuffixTrim: stopConditions);
        }
        finally
        {
            DropRecurrentPrefixIfRequestFailed();
            ArrayPool<char>.Shared.Return(stopScratch);
            ReleaseRequestMtpState(mtpState);
            if (ownsKvCache)
                kvCache.Dispose();
            telemetry.RequestSpan?.Dispose();
        }
    }

    /// <summary>
    /// Streams generated tokens as an async enumerable, yielding each token with incremental text,
    /// finish reason, and timings on the final token.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <returns>An async enumerable of <see cref="GenerationToken"/> values.</returns>
    private async IAsyncEnumerable<GenerationToken> GenerateStreamingTokensCoreAsync(
        string prompt,
        InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        ILoraAdapter? adapter = null)
    {
        options ??= new InferenceOptions();

        int[] promptIds = _tokenizer.Encode(prompt);
        int promptLen = promptIds.Length;
        int maxTokens = options.MaxTokens;
        int vocabSize = _model.Config.VocabSize;

        // Guard: empty prompt — use BOS token as seed
        if (promptLen == 0)
        {
            promptIds = [_tokenizer.BosTokenId];
            promptLen = 1;
        }

        // Guard: MaxTokens=0 — yield nothing
        if (maxTokens <= 0)
            yield break;

        cancellationToken.ThrowIfCancellationRequested();

        var telemetry = new TelemetryRecorder(_model.Config, options);

        // Build sampling pipeline
        var pipeline = new SamplerPipeline(options, _tokenizer);

        // Logprobs capture setup
        bool captureLogprobs = options.Logprobs;
        int topLogprobs = Math.Clamp(options.TopLogprobs, 0, 20);

        // Build decoding constraint for structured output
        IDecodingConstraint? constraint = options.ResponseFormat switch
        {
            ResponseFormat.JsonObject => new JsonConstraint(_tokenizer),
            ResponseFormat.JsonSchema js => new JsonSchemaConstraint(_tokenizer, js.Schema),
            ResponseFormat.Regex rx => RegexConstraint.Create(_tokenizer, rx.Pattern),
            ResponseFormat.Grammar gr => new GrammarConstraint(_tokenizer, gr.GbnfGrammar),
            _ => null
        };

        // Build stop conditions
        List<IStopCondition> stopConditions;
        if (options.StopConditions is not null)
        {
            stopConditions = new List<IStopCondition>(options.StopConditions);
        }
        else
        {
            stopConditions = new List<IStopCondition>
            {
                EndOfGenerationTokens.CreateStopCondition(_tokenizer),
                new MaxTokensStopCondition(maxTokens)
            };
            stopConditions.AddRange(StopStringCondition.CreateAll(options));
        }

        // Resolve KV-cache: reuse from prefix cache or allocate fresh
        var (kvCache, cachedTokenCount, ownsKvCache) = ResolveKvCache(promptIds, promptLen, maxTokens, out int snapshotAt, options.GenerationPromptSuffix, adapter);
        _recurrentPrefixRequestPending = _recurrentPrefixEnabled;
        long kvBytes = GetKvCacheBytes(kvCache);

        // Hybrid mode: same gating as the non-streaming path, MTP on a prefix hit included (issue #1088).
        var mtpGate = GateFor(pipeline.IsStochastic);
        var resumedMtp = TakeResumedMtpState();
        bool useMtp = ShouldUseMtp(captureLogprobs) && (cachedTokenCount == 0 || resumedMtp is not null)
            && (mtpGate?.ShouldUseMtp(maxTokens) ?? true);
        TraceGate(mtpGate, pipeline.IsStochastic, useMtp, captureLogprobs, cachedTokenCount);
        if (useMtp)
            _model.TrySetAllRowLogitsLimit(_mtpCandidates + 1);
        bool useHybrid = _hybridStrategy is not null
            && _hybridStrategy.ShouldRunHybrid(promptLen)
            && cachedTokenCount == 0
            && _draftModel is null
            && !useMtp;

        // Stop-check scratch buffer: rented up-front and returned in the outer finally. try/finally
        // is preserved across yield points by the async-iterator state machine, so Return runs on
        // normal completion, exception, or consumer-side cancellation (Dispose of the enumerator).
        int stopTailSize = ComputeStopTailSize(stopConditions);
        char[] stopScratch = ArrayPool<char>.Shared.Rent(stopTailSize);
        DotLLM.Core.Models.IMtpState? mtpState = AcquireRequestMtpState(useMtp, snapshotAt, cachedTokenCount, resumedMtp, useHybrid, mtpGate, kvCache);
        var generatedIds = new List<int>(maxTokens);
        long prefillTicks = 0;
        long decodeTicks = 0;
        long mtpTicks = 0; int mtpTokens = 0;   // MTP-segment wall ticks / committed tokens, for the adaptive gate (read in the finally)
        long samplerTicks = 0;

        try
        {
            int cacheSize = kvCache.MaxLength;

            // Incremental detokenizer: O(1) amortized per token for stop-check + streaming delta,
            // instead of decoding the full generated sequence at every step.
            var detok = new IncrementalDetokenizer(_tokenizer, initialCapacity: Math.Max(64, maxTokens * 4));

            // Local helper: snapshot log-softmax before sampling (which modifies logits in-place),
            // sample a token, then build logprob info.
            (int tokenId, TokenLogprobInfo? logprob) SampleWithLogprobs(Span<float> logitSpan)
            {
                float[]? lsBuf = captureLogprobs ? LogprobsCapture.ComputeLogSoftmax(logitSpan) : null;
                int tokenId = pipeline.Sample(logitSpan, generatedIds);
                TokenLogprobInfo? info = null;
                if (lsBuf != null)
                {
                    info = LogprobsCapture.BuildInfo(lsBuf.AsSpan(0, vocabSize), vocabSize, tokenId, topLogprobs, _tokenizer);
                    ArrayPool<float>.Shared.Return(lsBuf);
                }
                return (tokenId, info);
            }

            // Prefill: run only new suffix tokens through the model
            int prefillStart = cachedTokenCount;
            int prefillLen = promptLen - prefillStart;

            int firstTokenId;
            TokenLogprobInfo? firstLogprobInfo = null;
            long ts0 = Stopwatch.GetTimestamp();

            using (var prefillSpan = telemetry.StartPrefill())
            {
                if (useHybrid)
                {
                    // ── Hybrid prefill: CPU populates a SimpleKvCache, hand off to kvCache.
                    var handoff = _hybridStrategy!.RunPrefill(promptIds.AsSpan(0, promptLen), cacheSize);
                    try
                    {
                        _hybridStrategy.Handoff(handoff.HostCache, kvCache);
                        prefillTicks = handoff.PrefillTicks;

                        using var sampleSpan = telemetry.StartSample();
                        long samplerStart = Stopwatch.GetTimestamp();
                        var logitSpan = handoff.LastLogits.AsSpan(0, vocabSize);
                        if (constraint != null)
                            TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                        (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                        samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                    }
                    finally
                    {
                        handoff.HostCache.Dispose();
                    }
                }
                else if (prefillLen > 0)
                {
                    // Prefill suffix tokens — chunked when a prefill chunk size is configured
                    // (llama.cpp -ub analog); otherwise a single forward pass, as before.
                    using (ITensor prefillLogits = ForwardPrefill(promptIds, prefillStart, prefillLen, kvCache, adapter, mtpState,
                        snapshotAt, () => { TakeRecurrentSnapshot(promptIds, snapshotAt, kvCache, mtpState); if (_tieredSnapshots is null) ownsKvCache = false; },
                        cancellationToken))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            // GPU/hybrid models return [1, vocabSize] (last token only);
                            // CPU model returns [seqLen, vocabSize]. Use actual shape to index.
                            float* logitPtr = (float*)prefillLogits.DataPointer;
                            int logitRows = prefillLogits.Shape[0];
                            var logitSpan = new Span<float>(logitPtr + (long)(logitRows - 1) * vocabSize, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }
                }
                else if (promptLen > 0)
                {
                    // 100% cache hit — re-forward last prompt token to get logits
                    using (ITensor logits = _model.Forward([promptIds[^1]], [promptLen - 1], deviceId: -1, kvCache, adapter))
                    {
                        long ts1 = Stopwatch.GetTimestamp();
                        prefillTicks = ts1 - ts0;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (firstTokenId, firstLogprobInfo) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }
                }
                else
                {
                    // Unreachable: empty prompt guard ensures promptLen >= 1
                    throw new InvalidOperationException("Prompt is empty after guard.");
                }

                if (prefillSpan is { IsAllDataRequested: true })
                {
                    prefillSpan.SetTag(TelemetryTags.PrefillTokenCount, prefillLen);
                    prefillSpan.SetTag(TelemetryTags.PrefillDurationMs, prefillTicks * 1000.0 / Stopwatch.Frequency);
                }
            }

            telemetry.RecordFirstToken();
            constraint?.Advance(firstTokenId);

            // Check stop conditions for first token
            generatedIds.Add(firstTokenId);
            detok.Append(firstTokenId);

            var stopResult = CheckStopConditions(stopConditions, firstTokenId, generatedIds,
                detok.GetTailView(stopTailSize, stopScratch));
            if (stopResult != StopResult.Continue)
            {
                var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                if (stopResult == StopResult.Stop)
                {
                    generatedIds.RemoveAt(generatedIds.Count - 1);
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, detok.TakeWithheld(), fr, timings, firstLogprobInfo, GeneratedTokenCount: generatedIds.Count);
                }
                else
                {
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    string text = detok.TakeDelta(flushIncomplete: true);
                    yield return new GenerationToken(firstTokenId, text, fr, timings, firstLogprobInfo, GeneratedTokenCount: generatedIds.Count);
                }
                yield break;
            }

            // Yield first token — check if it's also the last (maxTokens == 1)
            {
                bool firstIsLast = maxTokens <= 1;
                string text = detok.TakeDelta(flushIncomplete: firstIsLast);
                if (firstIsLast)
                {
                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount);
                    yield return new GenerationToken(firstTokenId, text, FinishReason.Length, timings, firstLogprobInfo, GeneratedTokenCount: generatedIds.Count);
                    yield break;
                }
                yield return new GenerationToken(firstTokenId, text, null, Logprobs: firstLogprobInfo);
            }

            int specDrafted = 0, specAccepted = 0;
            bool useDraftSpec = _draftModel != null && !captureLogprobs && IsEffectivelyGreedy(options);
            int mtpFallbackStep = 0;                // > 0: the MTP rounds measured slower than plain decode; plain decode resumes at this step (#900)

            // Speculative decode disabled when logprobs requested — no per-position logit access.
            // Also disabled when sampling isn't effectively greedy (see Wave 8 / issue #121).
            if (useDraftSpec)
            {
                // ── Speculative decode loop ──
                var specDecoder = new SpeculativeDecoder(
                    greedy: true, seed: options.Seed);
                Core.Attention.IKvCache draftKvCache = AllocateDraftKvCache(cacheSize);
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_speculativeCandidates + 1);
                try
                {
                    PrefillDraftModel(promptIds, draftKvCache);

                    int step = 1;
                    while (step < maxTokens)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        int remaining = maxTokens - step;
                        int kk = Math.Min(_speculativeCandidates, remaining);

                        var result = specDecoder.DraftAndVerify(
                            _model, _draftModel!, kvCache, draftKvCache,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, _draftModel!.Config.VocabSize, kk, specBuffer);

                        if (result.AcceptedCount == 0) break;

                        decodeTicks += result.DraftTicks + result.VerifyTicks;
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        // Only count tokens that actually make it into output.
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            detok.Append(tokenId);

                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds,
                                detok.GetTailView(stopTailSize, stopScratch));
                            if (stopResult != StopResult.Continue)
                            {
                                var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                if (stopResult == StopResult.Stop)
                                {
                                    generatedIds.RemoveAt(generatedIds.Count - 1);
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, detok.TakeWithheld(), fr, timings, GeneratedTokenCount: generatedIds.Count);
                                }
                                else
                                {
                                    specAccepted++;
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    string text = detok.TakeDelta(flushIncomplete: true);
                                    yield return new GenerationToken(tokenId, text, fr, timings, GeneratedTokenCount: generatedIds.Count);
                                }
                                shouldBreak = true;
                                yield break;
                            }

                            specAccepted++;

                            // Yield each accepted token
                            {
                                bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                                string text = detok.TakeDelta(flushIncomplete: isLastStep && i == result.AcceptedCount - 1);
                                if (isLastStep && i == result.AcceptedCount - 1)
                                {
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, text, FinishReason.Length, timings, GeneratedTokenCount: generatedIds.Count);
                                    shouldBreak = true;
                                    break;
                                }
                                yield return new GenerationToken(tokenId, text, null);
                            }

                            step++;
                        }

                        if (shouldBreak) yield break;
                    }
                }
                finally
                {
                    draftKvCache.Dispose();
                    ArrayPool<int>.Shared.Return(specBuffer);
                }
            }
            else if (useMtp)
            {
                // ── MTP self-speculative decode loop (issue #253) — streaming variant. ──
                var mtpDecoder = new MtpSpeculativeDecoder();
                int[] specBuffer = ArrayPool<int>.Shared.Rent(_mtpCandidates + 1);
                var mtpCtl = mtpGate is { } mtpG ? new MtpRoundController(mtpG.CostModel, _mtpCandidates, calibratePlain: mtpG.NeedsPlainReference(maxTokens - 1), gate: mtpG) : null;
                try
                {
                    int step = 1;
                    while (step < maxTokens)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int pos = promptLen + step - 1;
                        if (pos >= cacheSize) break;

                        // See the non-streaming loop: fall back to plain decode when MTP measures slower.
                        if (mtpCtl is not null && mtpCtl.ShouldFallBack(mtpGate!.PlainMsPerToken)) { mtpFallbackStep = step; break; }

                        int remaining = maxTokens - step;
                        int kk = mtpCtl?.NextK(remaining) ?? Math.Min(_mtpCandidates, remaining);

                        var result = mtpDecoder.DraftAndVerify(
                            _model, kvCache, mtpState!,
                            pipeline, generatedIds, constraint,
                            pos, vocabSize, kk, specBuffer, allowZeroDraft: true);

                        if (result.AcceptedCount == 0) break;

                        decodeTicks += result.RoundTicks;
                        // Zero-draft calibration rounds are plain decode steps: they stay out of the MTP segment, so the end-of-request record
                        // hands them to the gate as the plain sample.
                        if (result.DraftedCount > 0)
                        {
                            mtpTicks += result.RoundTicks;
                            mtpTokens += result.AcceptedCount;
                        }
                        mtpCtl?.Record(result.DraftedCount, result.AcceptedCount, result.RoundTicks * 1000.0 / Stopwatch.Frequency);
                        specDrafted += result.DraftedCount;

                        // Constraint is already advanced inside DraftAndVerify — do NOT advance again here.
                        bool shouldBreak = false;
                        for (int i = 0; i < result.AcceptedCount; i++)
                        {
                            int tokenId = specBuffer[i];
                            generatedIds.Add(tokenId);
                            detok.Append(tokenId);

                            stopResult = CheckStopConditions(stopConditions, tokenId, generatedIds,
                                detok.GetTailView(stopTailSize, stopScratch));
                            if (stopResult != StopResult.Continue)
                            {
                                var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;
                                if (stopResult == StopResult.Stop)
                                {
                                    generatedIds.RemoveAt(generatedIds.Count - 1);
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, detok.TakeWithheld(), fr, timings, GeneratedTokenCount: generatedIds.Count);
                                }
                                else
                                {
                                    specAccepted++;
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    string text = detok.TakeDelta(flushIncomplete: true);
                                    yield return new GenerationToken(tokenId, text, fr, timings, GeneratedTokenCount: generatedIds.Count);
                                }
                                shouldBreak = true;
                                yield break;
                            }

                            specAccepted++;

                            // Yield each accepted token
                            {
                                bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                                string text = detok.TakeDelta(flushIncomplete: isLastStep && i == result.AcceptedCount - 1);
                                if (isLastStep && i == result.AcceptedCount - 1)
                                {
                                    StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                                    var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                                    yield return new GenerationToken(tokenId, text, FinishReason.Length, timings, GeneratedTokenCount: generatedIds.Count);
                                    shouldBreak = true;
                                    break;
                                }
                                yield return new GenerationToken(tokenId, text, null);
                            }

                            step++;
                        }

                        if (shouldBreak) yield break;
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(specBuffer);
                }
            }

            // Plain decode: the default, and the tail of a request whose MTP rounds fell back (mtpFallbackStep).
            if (!useDraftSpec && (!useMtp || mtpFallbackStep > 0))
            {
                // ── Standard decode loop: one token at a time ──
                for (int step = mtpFallbackStep > 0 ? mtpFallbackStep : 1; step < maxTokens; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int pos = promptLen + step - 1;
                    if (pos >= cacheSize)
                        break;

                    Activity? decodeStepSpan = telemetry.StartDecodeStep(step);

                    int lastToken = generatedIds[^1];
                    int nextTokenId;
                    TokenLogprobInfo? tokenLogprob;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using (ITensor logits = _model.Forward([lastToken], [pos], deviceId: -1, kvCache, adapter))
                    {
                        decodeTicks += Stopwatch.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            using var sampleSpan = telemetry.StartSample();
                            long samplerStart = Stopwatch.GetTimestamp();
                            var logitSpan = new Span<float>((void*)logits.DataPointer, vocabSize);
                            if (constraint != null)
                                TokenMaskApplier.Apply(logitSpan, constraint.GetAllowedTokens());
                            (nextTokenId, tokenLogprob) = SampleWithLogprobs(logitSpan);
                            samplerTicks += Stopwatch.GetTimestamp() - samplerStart;
                        }
                    }

                    decodeStepSpan?.Dispose();
                    constraint?.Advance(nextTokenId);

                    generatedIds.Add(nextTokenId);
                    detok.Append(nextTokenId);

                    stopResult = CheckStopConditions(stopConditions, nextTokenId, generatedIds,
                        detok.GetTailView(stopTailSize, stopScratch));
                    if (stopResult != StopResult.Continue)
                    {
                        var fr = stopResult == StopResult.StopInclude ? FinishReason.Length : FinishReason.Stop;

                        if (stopResult == StopResult.Stop)
                        {
                            generatedIds.RemoveAt(generatedIds.Count - 1);
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                            yield return new GenerationToken(nextTokenId, detok.TakeWithheld(), fr, timings, tokenLogprob, GeneratedTokenCount: generatedIds.Count);
                        }
                        else
                        {
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                            string text = detok.TakeDelta(flushIncomplete: true);
                            yield return new GenerationToken(nextTokenId, text, fr, timings, tokenLogprob, GeneratedTokenCount: generatedIds.Count);
                        }
                        yield break;
                    }

                    // Yield token — attach finish reason if this is the last iteration
                    {
                        bool isLastStep = (step + 1 >= maxTokens) || (promptLen + step >= cacheSize);
                        string text = detok.TakeDelta(flushIncomplete: isLastStep);
                        if (isLastStep)
                        {
                            StoreInPrefixCache(kvCache, promptIds, generatedIds, ref ownsKvCache);
                            var timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvBytes, cachedTokenCount, specDrafted, specAccepted);
                            yield return new GenerationToken(nextTokenId, text, FinishReason.Length, timings, tokenLogprob, GeneratedTokenCount: generatedIds.Count);
                            yield break;
                        }
                        yield return new GenerationToken(nextTokenId, text, null, Logprobs: tokenLogprob);
                    }
                }
            }
        }
        finally
        {
            telemetry.Complete(promptLen, cachedTokenCount, generatedIds.Count,
                prefillTicks * 1000.0 / Stopwatch.Frequency,
                decodeTicks * 1000.0 / Stopwatch.Frequency,
                FinishReason.Length);
            RecordMtpGate(mtpGate, useMtp, generatedIds.Count, mtpTicks, mtpTokens, decodeTicks, samplerTicks);
            DropRecurrentPrefixIfRequestFailed();
            ArrayPool<char>.Shared.Return(stopScratch);
            ReleaseRequestMtpState(mtpState);
            if (ownsKvCache)
                kvCache.Dispose();
        }
    }

    /// <summary>
    /// Streams generated text as an async enumerable, yielding incremental text fragments.
    /// This is a convenience wrapper over <see cref="GenerateStreamingTokensAsync"/>.
    /// </summary>
    /// <param name="prompt">Input text prompt.</param>
    /// <param name="options">Inference options controlling sampling and stopping. Null uses defaults.</param>
    /// <param name="cancellationToken">Token to cancel generation cooperatively between decode steps.</param>
    /// <param name="adapter">Optional LoRA adapter to apply during the forward passes (Phase 4c).</param>
    /// <returns>An async enumerable of incremental text strings.</returns>
    public async IAsyncEnumerable<string> GenerateStreamingAsync(
        string prompt,
        InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        ILoraAdapter? adapter = null)
    {
        await foreach (var token in GenerateStreamingTokensAsync(prompt, options, cancellationToken, adapter))
            yield return token.Text;
    }

    /// <summary>
    /// Resolves the KV-cache to use: either from the prefix cache (on hit) or freshly allocated.
    /// Returns the cache, number of cached tokens, and whether the caller owns (should dispose) the cache.
    /// </summary>
    private (Core.Attention.IKvCache KvCache, int CachedTokenCount, bool OwnsKvCache) ResolveKvCache(
        int[] promptIds, int promptLen, int maxTokens, out int snapshotAt, string? generationPromptSuffix,
        ILoraAdapter? adapter = null)
    {
        snapshotAt = -1;
        _resumedMtp = null;
        LastPrefixCacheTier = null; LastPrefixRestoreMilliseconds = 0;
        if (adapter is not null) ClearRecurrentPrefixCacheCore();
        // A recurrent (Gated DeltaNet / SSM) model carries per-sequence state that has no position
        // addressing: reusing only the KV half of a cached prefix would leave that state at the END of
        // the previous request. So recurrent models never take a KV-only prefix hit, and every
        // generation starts from zeroed state (the model-owned state is otherwise inherited from
        // whichever request ran last — silently wrong logits, measured on Tev1-0.8B: the same prompt
        // scored logprob -0.111 alone and -0.234 after an unrelated request).
        bool recurrent = _model.RequiresPerSequenceState;
        if (recurrent)
        {
            if (_recurrentPrefixEnabled && adapter is null && _tieredSnapshots is not null)
                return ResolveTieredRecurrentPrefix(promptIds, promptLen, maxTokens, generationPromptSuffix, out snapshotAt);
            if (_recurrentPrefixEnabled && adapter is null)
                return ResolveRecurrentPrefix(promptIds, promptLen, maxTokens, generationPromptSuffix, out snapshotAt);
            _model.ResetSequenceState();
        }

        // Cross-request prefix trie (Step 37) takes priority — multiple sessions share blocks.
        if (_prefixTrieManager != null && !recurrent)
        {
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var admission = _prefixTrieManager.Admit(promptIds, cacheSize);
            return (admission.Cache, admission.CachedTokens, true);
        }

        if (_prefixCache != null && !recurrent)
        {
            var (entry, matchedTokens) = _prefixCache.FindMatch(promptIds);

            if (entry != null && matchedTokens > 0)
            {
                // Cache hit — reuse existing KV-cache, truncate to matched prefix
                switch (entry.KvCache)
                {
                    // A compact sliding-window cache (#1169) can only cut back while it still holds the rows the
                    // next query needs; past that the hit is a miss.
                    case SimpleKvCache simpleCache when simpleCache.CanTruncateTo(matchedTokens):
                        simpleCache.SetCurrentLength(matchedTokens);
                        break;
                    case KvCache.PagedKvCache pagedCache:
                        pagedCache.SetCurrentLength(matchedTokens);
                        break;
                    default:
                        // Unsupported cache type for prefix reuse — fall through to allocate fresh
                        goto cacheMiss;
                }

                // Verify the cache is large enough for the new prompt + generation.
                // Reusing a cache that can only fit the prompt but not the requested generation
                // causes silent truncation once pos >= MaxLength in the decode loop.
                int requiredSize = promptLen + maxTokens;
                if (entry.KvCache.MaxLength >= requiredSize)
                    return (entry.KvCache, matchedTokens, false);

                // Cache too small — fall through to allocate fresh
            }
            cacheMiss:

            // Cache miss or incompatible — allocate with full model context for future reuse
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, false); // ownsKvCache=false: will be transferred to prefix cache
        }

        // No prefix cache — allocate normally, caller owns
        {
            int cacheSize = Math.Min(promptLen + maxTokens, _model.Config.MaxSequenceLength);
            var kvCache = AllocateKvCache(cacheSize);
            return (kvCache, 0, true);
        }
    }

    /// <summary>
    /// Allocates a fresh KV-cache using the factory or default SimpleKvCache.
    /// </summary>
    private Core.Attention.IKvCache AllocateKvCache(int cacheSize)
    {
        return _kvCacheFactory != null
            ? _kvCacheFactory(_model.Config, cacheSize)
            // KvGeometry.FromConfig is byte-identical to the scalar
            // (NumLayers, NumKvHeads, HeadDim) form for every uniform model and
            // supplies distinct per-layer strides for Gemma-4 (sliding vs global).
            : new SimpleKvCache(
                Core.Attention.KvGeometry.FromConfig(_model.Config, _model.SupportsCompactSlidingKv),
                cacheSize);
    }

    /// <summary>
    /// Stores the KV-cache in the prefix cache after generation completes.
    /// Transfers ownership so the cache is not disposed by the caller.
    /// </summary>
    private void StoreInPrefixCache(Core.Attention.IKvCache kvCache, int[] promptIds,
        List<int> generatedIds, ref bool ownsKvCache)
    {
        // Every normal end of a request passes through here (before its terminal token is yielded): the request succeeded.
        _recurrentPrefixRequestPending = false;

        // Recurrent models never reuse a KV-only prefix (see ResolveKvCache): nothing to store.
        if (_model.RequiresPerSequenceState)
            return;

        // Cross-request trie (Step 37): record completion so freshly-computed
        // blocks become available to future sequences, then let Dispose run.
        if (_prefixTrieManager != null && kvCache is KvCache.PagedKvCache paged)
        {
            int total = promptIds.Length + generatedIds.Count;
            var full = ArrayPool<int>.Shared.Rent(total);
            try
            {
                Array.Copy(promptIds, full, promptIds.Length);
                CollectionsMarshal.AsSpan(generatedIds).CopyTo(full.AsSpan(promptIds.Length));
                _prefixTrieManager.RecordCompletion(paged, full.AsSpan(0, total));
            }
            finally
            {
                ArrayPool<int>.Shared.Return(full);
            }
            // ownsKvCache stays unchanged — caller disposes the cache, the trie has
            // already promoted the new blocks to "trie-owned".
            return;
        }

        if (_prefixCache == null)
            return;

        // Build full token sequence: prompt + generated
        var fullSequence = new int[promptIds.Length + generatedIds.Count];
        Array.Copy(promptIds, fullSequence, promptIds.Length);
        CollectionsMarshal.AsSpan(generatedIds).CopyTo(fullSequence.AsSpan(promptIds.Length));

        _prefixCache.Store(fullSequence, kvCache);
        ownsKvCache = false;
    }

    private static StopResult CheckStopConditions(
        List<IStopCondition> conditions, int tokenId,
        IReadOnlyList<int> generatedTokens, ReadOnlySpan<char> decodedTail)
    {
        for (int i = 0; i < conditions.Count; i++)
        {
            var result = conditions[i].ShouldStop(tokenId, generatedTokens, decodedTail);
            if (result != StopResult.Continue)
                return result;
        }
        return StopResult.Continue;
    }

    /// <summary>
    /// True when some registered <see cref="StopStringCondition"/>'s stop string is a
    /// suffix of <paramref name="decodedTail"/>. Determines whether the last token is
    /// kept in <c>generatedIds</c> (true — its text carries a partial overlap with the
    /// stop string and must be character-trimmed in <c>BuildResponse</c>) or removed
    /// (false — EOS / similar single-token termination where the token's text is
    /// conceptually the terminator itself).
    /// </summary>
    /// <remarks>
    /// Deliberately independent of <em>which</em> condition <see cref="CheckStopConditions"/>
    /// matched first: that would make the outcome depend on the order conditions were
    /// registered, so an EOS condition placed ahead of a stop-string condition matching
    /// the same tail would drop the whole last token and resurrect the over-trim bug.
    /// The suffix test here uses exactly the predicate <see cref="StopStringCondition"/>
    /// itself uses (ordinal <c>EndsWith</c> on the same tail window).
    /// </remarks>
    private static bool HasStopStringSuffix(ReadOnlySpan<char> decodedTail, List<IStopCondition> conditions)
        => StopSuffixTrimmer.MatchedSuffixLength(decodedTail, conditions) > 0;

    // Tail window passed to stop conditions. Must cover the longest stop string currently
    // registered; a safety cushion absorbs future stop strings added via custom conditions.
    private static int ComputeStopTailSize(List<IStopCondition> conditions)
        => Math.Max(64, StopSuffixTrimmer.TailWindowSize(conditions));

    // Speculative decoding's greedy acceptance path matches the target pipeline only when the pipeline
    // itself is effectively argmax. Temperature <= 0 forces argmax selection; repetition penalty can
    // shift which token is argmax, so it must also be neutral. Top-k/p/min-p prune low-probability
    // tokens and never mask the argmax, so they don't need to be checked. Wave 8 / issue #121 lifts
    // this restriction by making q/p pipeline-aware.
    private static bool IsEffectivelyGreedy(DotLLM.Core.Configuration.InferenceOptions options)
        => options.Temperature <= 0f && options.RepetitionPenalty == 1.0f;

    /// <summary>
    /// Gates MTP self-speculative decoding (issue #253) for one <c>Generate</c>/
    /// <c>GenerateStreamingTokensAsync</c> call: enabled, the model actually carries an MTP head,
    /// no explicit two-model draft is configured (mutually exclusive — an explicit draft model
    /// always wins), no logprobs requested (no per-position logit access in the draft loop), and
    /// decoding is effectively greedy (same distributional-correctness gate as the two-model path,
    /// see <see cref="IsEffectivelyGreedy"/>).
    /// </summary>
    /// <summary>
    /// Creates the MTP state for one request, sized to the KV-cache: the head's cache is indexed by
    /// sequence position, so it must reach as far as the trunk's (issue #469).
    /// </summary>
    private DotLLM.Core.Models.IMtpState CreateMtpState(Core.Attention.IKvCache kvCache)
        => _model.CreateMtpState(kvCache.MaxLength)
           ?? throw new InvalidOperationException(
               $"{_model.GetType().Name}.SupportsMtp is true but CreateMtpState() returned null.");

    /// <summary>
    /// Recurrent-model prefix reuse. HIT: the new prompt starts with the snapshotted prefix, so restore the recurrent
    /// state, roll the KV back to the prefix and prefill only the suffix. MISS: reset, allocate a KV with headroom, and
    /// (when the prompt shares at least <see cref="RecurrentPrefixMinTokens"/> tokens with the previous one) ask the
    /// prefill to snapshot at that boundary. The shared prefix of two consecutive requests is, for a classifier, the
    /// fixed system prompt, so the cost is one extra checkpoint and the saving is that prefix's prefill on every later request.
    /// </summary>
    /// <remarks>
    /// <para>Agent loops (issue #1025), where each prompt extends the previous one by an assistant turn and tool results:</para>
    /// <list type="bullet">
    /// <item>A hit whose KV is too small for this request grows it (<see cref="Core.Attention.IKvCache.TryGrowTo"/>, rows
    /// kept). It used to miss: the KV had 256 positions of headroom and an agent step adds more, so every step re-prefilled
    /// the whole conversation. A refused growth is retried at the exact size; when that is refused too, a long prompt fails with
    /// <see cref="DeviceMemoryAdmissionException"/> rather than re-prefilling (<see cref="RecurrentPrefixRefillMissMaxTokens"/>, issue #1093).</item>
    /// <item>A hit also moves the snapshot forward, so the next step does not re-prefill everything since the first snapshot.
    /// When the prompt extends the whole previous one (an agent loop) the snapshot goes at its own end and the next step
    /// prefills only its new turn (issue #1080); otherwise at what it shares with the previous prompt.</item>
    /// <item>A miss on a KV without exclusive storage (<see cref="Core.Attention.IKvCache.HasExclusiveStorage"/>) drops the
    /// snapshot. This request's forwards overwrite those rows, and a later hit would attend over them.</item>
    /// </list>
    /// </remarks>
    private (Core.Attention.IKvCache KvCache, int CachedTokenCount, bool OwnsKvCache) ResolveRecurrentPrefix(
        int[] promptIds, int promptLen, int maxTokens, string? generationPromptSuffix, out int snapshotAt)
    {
        snapshotAt = -1;
        int needed = promptLen + maxTokens;
        int[]? previous = _recurrentPrefixLastPrompt;
        int? previousSuffixBoundary = _recurrentPrefixLastSuffixBoundary;
        int shared = previous is null ? 0 : promptIds.AsSpan().CommonPrefixLength(previous);
        int? suffixBoundary = RecurrentSnapshotBoundary.FromGenerationSuffix(promptIds, _tokenizer, generationPromptSuffix);

        // Leave at least one suffix token to run (it produces the logits), so an exact repeat of the previous prompt
        // snapshots one token short of its end. A prompt that extends the whole previous one is an agent loop, whose next
        // prompt will extend this one: snapshot at this prompt's end, so the next turn prefills only what it adds. Snapshotting
        // at the shared prefix made every agent hit reuse the prompt from two turns back (issue #1080).
        // A chat turn that continues the previous one but re-renders its generation prompt differently (issue #1093) does
        // not extend it: snapshot just before this prompt's generation prompt, which the next turn will share.
        bool extendsPrevious = previous is not null && shared == previous.Length;
        int boundary = RecurrentSnapshotBoundary.Choose(
            extendsPrevious ? promptLen - 1 : Math.Min(shared, promptLen - 1),
            RecurrentSnapshotBoundary.ForContinuation(suffixBoundary, shared, previousSuffixBoundary),
            promptLen);

        var entry = _recurrentPrefix;
        bool hit = entry is not null
            && promptLen > entry.Tokens.Length                      // at least one suffix token must run to produce logits
            && promptIds.AsSpan(0, entry.Tokens.Length).SequenceEqual(entry.Tokens);
        if (entry is not null && hit && !EnsureRecurrentPrefixKvCapacity(entry.Kv, needed))
        {
            // Issue #1093: the growth was refused even at the exact size. A long prompt fails here, before anything changed, so
            // the snapshot (whose KV did not grow) stays for a request that fits; a short one misses as before.
            ThrowIfRecurrentRefillTooLong(promptLen, Math.Min(_model.Config.MaxSequenceLength, needed), entry.Kv.MaxLength);
            hit = false;
        }

        // Recorded only once the request is going ahead, so a refused one leaves the next request's boundaries unchanged.
        _recurrentPrefixLastPrompt = promptIds;
        _recurrentPrefixLastSuffixBoundary = suffixBoundary;

        if (entry is not null && hit)
        {
            long restoreStart = Stopwatch.GetTimestamp();
            entry.Kv.Rollback(entry.Tokens.Length);
            _model.RestoreRecurrentState(entry.Checkpoint);
            // The head rolls back with the trunk. TryGrowTo may have grown the KV above, so the head's ceiling follows it.
            if (entry.Mtp is { } mtp && entry.MtpMark is { } mark && mtp.TryResumeAt(mark, entry.Kv.MaxLength))
                _resumedMtp = mtp;
            LastPrefixCacheTier = "vram";
            LastPrefixRestoreMilliseconds = Stopwatch.GetElapsedTime(restoreStart).TotalMilliseconds;
            _legacyPrefixHits++;
            if (boundary >= entry.Tokens.Length + RecurrentPrefixMinTokens)
                snapshotAt = boundary;
            return (entry.Kv, entry.Tokens.Length, false);
        }

        _legacyPrefixMisses++;
        _model.ResetSequenceState();
        if (entry is not null && !entry.Kv.HasExclusiveStorage)
        {
            _recurrentPrefix = null;
            entry.Dispose();
        }

        // Issue #1088: with no snapshot held there is nothing to evict. A prompt that shares nothing usable with the previous
        // one starts a conversation (the first request, or a new client), so snapshot its own end, betting that the next
        // prompt extends it: an agent's second turn then hits instead of re-prefilling its first. A prompt that does share a
        // prefix keeps that boundary (the classifier case above). With a generation-prompt hint the bet is the start of the
        // generation prompt instead (issue #1093): the next turn shares that much whether or not it re-renders the rest.
        if (_recurrentPrefix is null && shared < RecurrentPrefixMinTokens)
            boundary = suffixBoundary ?? promptLen - 1;

        bool snapshot = boundary >= RecurrentPrefixMinTokens;
        int cacheSize = Math.Min(_model.Config.MaxSequenceLength, needed + (snapshot ? RecurrentPrefixKvHeadroom : 0));
        var kv = AllocateKvCache(cacheSize);
        if (snapshot)
            snapshotAt = boundary;
        return (kv, 0, true);
    }

    /// <summary>
    /// True when <paramref name="kv"/> holds <paramref name="needed"/> positions, growing it when needed. The headroom is an
    /// eighth of the request, at least <see cref="RecurrentPrefixKvHeadroom"/> and at most <see cref="RecurrentPrefixMaxGrowthHeadroom"/>,
    /// so a growing agent conversation does not reallocate on every step. It is capped because model-owned storage (the CUDA hybrid)
    /// never shrinks. Capped at the context window too, as is <paramref name="needed"/>: past the window the request decodes
    /// until the window is full, as a miss sized to the window would. A cache may refuse the headroom (no room in free VRAM);
    /// the growth is then retried at exactly the needed size (issue #1093: at 112k the headroom was refused, the hit missed and
    /// re-prefilled everything, though the exact size fitted). False only when that is refused too.
    /// </summary>
    private bool EnsureRecurrentPrefixKvCapacity(Core.Attention.IKvCache kv, int needed)
    {
        int maxSeq = _model.Config.MaxSequenceLength;
        int exact = Math.Min(maxSeq, needed);
        if (kv.MaxLength >= exact) return true;
        int headroom = Math.Clamp(needed / 8, RecurrentPrefixKvHeadroom, RecurrentPrefixMaxGrowthHeadroom);
        int target = (int)Math.Min(maxSeq, (long)needed + headroom);
        return kv.TryGrowTo(target) || (target > exact && kv.TryGrowTo(exact));
    }

    /// <summary>
    /// Fails a request whose recurrent-prefix KV could not grow to <paramref name="needed"/> positions when the fallback, a miss
    /// that re-prefills the whole prompt, would be long (<see cref="RecurrentPrefixRefillMissMaxTokens"/>). Returns normally for
    /// a short prompt, whose caller then misses. Shared by the single-slot path and the continuous-batch scheduler.
    /// </summary>
    /// <param name="promptLen">The request's prompt length: what a miss would re-prefill.</param>
    /// <param name="needed">The positions the request needs (prompt + max_tokens, capped at the window).</param>
    /// <param name="kvMaxLength">The positions the KV holds, which it could not grow past.</param>
    /// <exception cref="DeviceMemoryAdmissionException">The prompt is at least <see cref="RecurrentPrefixRefillMissMaxTokens"/> tokens.</exception>
    internal static void ThrowIfRecurrentRefillTooLong(int promptLen, int needed, int kvMaxLength)
    {
        if (promptLen < RecurrentPrefixRefillMissMaxTokens) return;
        throw new DeviceMemoryAdmissionException(
            $"The KV cache could not grow to the {needed} positions this request needs (prompt + max_tokens); it holds {kvMaxLength}. " +
            $"Re-prefilling the {promptLen}-token prompt instead would be slow, so the request is refused: shorten the prompt or " +
            "max_tokens, or free device memory.");
    }

    /// <summary>
    /// Snapshots the recurrent state at <paramref name="boundary"/>, which the prefill has just reached. When the request
    /// carries the MTP head (it decodes with MTP, or <see cref="TracksMtpHead"/>), the head has absorbed exactly
    /// <c>[0, boundary)</c> too, so the entry keeps that head state and a mark of it (issue #1088). The head's rows below the mark stay valid for the same reason the trunk's KV rows do: every
    /// later write of this request, and of a hit that resumes it, is at or beyond the boundary. The tiered cache copies
    /// only the trunk, so its hits decode plain.
    /// </summary>
    private void TakeRecurrentSnapshot(int[] promptIds, int boundary, Core.Attention.IKvCache kvCache, IMtpState? mtpState)
    {
        if (_tieredSnapshots is not null) { _tieredSnapshots.TryCapture(promptIds, kvCache, boundary); return; }
        object? checkpoint = _model.CheckpointRecurrentState();
        object? mtpMark = mtpState is not null && mtpState.CurrentLength == boundary ? mtpState.MarkPrefix() : null;
        var old = _recurrentPrefix;
        _recurrentPrefix = new RecurrentPrefixEntry(promptIds.AsSpan(0, boundary).ToArray(), kvCache, checkpoint,
            mtpMark is null ? null : mtpState, mtpMark);
        // A hit that moves its snapshot forward hands the same KV (and, decoding with MTP, the same head state) to the new
        // entry, and the request still runs on both: release only what nothing uses any more.
        old?.Release(keepKv: kvCache, keepMtp: mtpState);
    }

    /// <summary>
    /// True when a request that decodes plain should still carry the MTP head through its prefill (issue #1088): it takes a
    /// single-slot prefix snapshot, and a later hit on that snapshot can decode with MTP only if the head absorbed the prefix
    /// too. Without this, an adaptive gate that sends the first turn of an agent session to plain decode never sees MTP again,
    /// because every later turn hits. A hit can carry the head only when it resumed one: the head cannot start absorbing
    /// mid-sequence.
    /// </summary>
    /// <remarks>
    /// The feed is not free: it adds the head's absorb to this request's prefill, so to its time to first token (about 0.2 ms
    /// per token, 3% of the trunk prefill, for Bonsai-2 27B on an RTX 3060; PR #1091 review). So a plain turn feeds the head only
    /// when the adaptive gate forecasts MTP for a coming request (<see cref="MtpAdaptiveGate.LikelyUsesMtpNext"/>). Once plain
    /// decode measures faster, the session's later hits skip the feed and stay plain, as they did before #1088; the next miss
    /// starts a fresh head. Its KV is already counted by the context fit when MTP is on. A backend whose head states cannot
    /// resume (<see cref="IMtpState.MarkPrefix"/> returns null, the Vulkan states today) never feeds one.
    /// </remarks>
    /// <param name="snapshotAt">The prefill position at which this request snapshots, or -1.</param>
    /// <param name="cachedTokenCount">Prompt tokens served from the prefix cache.</param>
    /// <param name="resumedMtp">The head a prefix hit resumed, or null.</param>
    /// <param name="useHybrid">Whether this request runs the hybrid prefill strategy.</param>
    /// <param name="gate">The adaptive gate for this request's sampling kind, or null when MTP is not adaptive.</param>
    /// <returns>True when the head should ride along this plain request's prefill.</returns>
    private bool TracksMtpHead(int snapshotAt, int cachedTokenCount, IMtpState? resumedMtp, bool useHybrid, MtpAdaptiveGate? gate)
        => snapshotAt >= 0 && _tieredSnapshots is null && !useHybrid
           && _mtpEnabled && _draftModel is null && _model.SupportsMtp && _mtpStatesResumable != false
           && (cachedTokenCount == 0 || resumedMtp is not null)
           && (gate?.LikelyUsesMtpNext() ?? true);

    /// <summary>
    /// The MTP head state a request runs with: the one a prefix hit resumed, else a new one, when the request decodes with MTP
    /// or carries the head for its snapshot (<see cref="TracksMtpHead"/>); otherwise null. The first state made tells whether
    /// this model's states can resume. A head made only for the snapshot is dropped at once when they cannot.
    /// </summary>
    /// <param name="useMtp">Whether the request decodes with MTP.</param>
    /// <param name="snapshotAt">The prefill position at which this request snapshots, or -1.</param>
    /// <param name="cachedTokenCount">Prompt tokens served from the prefix cache.</param>
    /// <param name="resumedMtp">The head a prefix hit resumed, or null.</param>
    /// <param name="useHybrid">Whether this request runs the hybrid prefill strategy.</param>
    /// <param name="gate">The adaptive gate for this request's sampling kind, or null.</param>
    /// <param name="kvCache">The request's trunk KV cache, which sizes a new head.</param>
    /// <returns>The head state, or null when the request runs without one.</returns>
    private IMtpState? AcquireRequestMtpState(bool useMtp, int snapshotAt, int cachedTokenCount, IMtpState? resumedMtp,
        bool useHybrid, MtpAdaptiveGate? gate, Core.Attention.IKvCache kvCache)
    {
        if (!useMtp && !TracksMtpHead(snapshotAt, cachedTokenCount, resumedMtp, useHybrid, gate))
            return null;
        if (resumedMtp is not null)
            return resumedMtp;
        var state = CreateMtpState(kvCache);
        _mtpStatesResumable ??= state.MarkPrefix() is not null;
        if (useMtp || _mtpStatesResumable == true)
            return state;
        state.Dispose();
        return null;
    }

    /// <summary>Hands the head state a prefix hit resumed to the request that resolved it (once), or null.</summary>
    private IMtpState? TakeResumedMtpState()
    {
        var mtp = _resumedMtp;
        _resumedMtp = null;
        return mtp;
    }

    /// <summary>Disposes a request's MTP state unless the recurrent prefix snapshot kept it for the next hit.</summary>
    private void ReleaseRequestMtpState(IMtpState? mtpState)
    {
        if (mtpState is not null && !ReferenceEquals(_recurrentPrefix?.Mtp, mtpState))
            mtpState.Dispose();
    }

    /// <summary>
    /// Drops the recurrent prefix snapshot when the request that used or produced it did not finish normally: a throw (for
    /// example an OOM growing the KV after the free-VRAM admission had passed) or an abandoned stream. A grow-hit has already
    /// raised the handle's MaxLength, so keeping the snapshot would make the next same-prefix request skip the admission and
    /// retry the same failing allocation on every agent step (PR #1026 re-review).
    /// </summary>
    private void DropRecurrentPrefixIfRequestFailed()
    {
        if (!_recurrentPrefixRequestPending) return;
        _recurrentPrefixRequestPending = false;
        ClearRecurrentPrefixCacheCore();
    }

    /// <summary>Releases the recurrent prefix snapshot (its KV cache and state checkpoint). Call when the model is unloaded.</summary>
    public void ClearRecurrentPrefixCache()
    {
        _generationGate.Wait();
        try { ClearRecurrentPrefixCacheCore(); }
        finally { _generationGate.Release(); }
    }

    private void ClearRecurrentPrefixCacheCore()
    {
        _tieredSnapshots?.Clear();
        var entry = _recurrentPrefix;
        _recurrentPrefix = null;
        _recurrentPrefixLastPrompt = null;
        _recurrentPrefixLastSuffixBoundary = null;
        entry?.Dispose();
    }

    /// <summary>The adaptive MTP gate, or null when <c>mtpAdaptive</c> was not requested (or disabled by environment).</summary>
    public MtpAdaptiveGate? MtpGate => _mtpGate;

    /// <summary>
    /// True when <paramref name="constraint"/> is satisfied and permits no token except EOS, i.e. the next forward pass
    /// could only sample EOS. Cheap (a popcount over the cached mask) and only evaluated once the constraint reports
    /// completion.
    /// </summary>
    private bool ConstraintIsTerminal(DotLLM.Core.Constraints.IDecodingConstraint? constraint)
    {
        if (constraint is null || !constraint.IsComplete())
            return false;

        var mask = constraint.GetAllowedTokens();
        int allowed = 0;
        foreach (long word in mask.AsSpan())
            allowed += System.Numerics.BitOperations.PopCount((ulong)word);

        int eos = _tokenizer.EosTokenId;
        if (eos >= 0 && eos < _model.Config.VocabSize && mask.IsAllowed(eos))
            allowed--;
        return allowed <= 0;
    }

    /// <summary>Clears the adaptive MTP gates' measurements (greedy and sampled); the warm-up calls this so its tiny cold requests do not set the first verdict.</summary>
    public void ResetMtpGate()
    {
        _mtpGate?.Reset();
        _mtpGateSampled?.Reset();
    }

    private void TraceGate(MtpAdaptiveGate? gate, bool sampled, bool useMtp, bool captureLogprobs, int cachedTokenCount)
    {
        if (!MtpAdaptiveGate.TraceEnabled || !_model.SupportsMtp) return;
        Console.Error.WriteLine($"[mtp] {(sampled ? "sampled" : "greedy")} request: {(useMtp ? "MTP" : "plain")} (enabled={_mtpEnabled}, logprobs={captureLogprobs}, cachedTokens={cachedTokenCount}, K={_mtpCandidates}); {gate?.Describe() ?? "no gate"}");
    }

    private MtpAdaptiveGate? GateFor(bool sampled) => sampled ? _mtpGateSampled : _mtpGate;

    /// <summary>
    /// Feeds one finished request to the adaptive gate. A plain request contributes one plain sample (forward + sampler); an MTP request contributes
    /// an MTP sample from its round wall time (the rounds already contain their sampling) and, when it fell back to plain decode mid-request, a plain
    /// sample from the tail.
    /// </summary>
    private static void RecordMtpGate(MtpAdaptiveGate? gate, bool usedMtp, int generated, long mtpTicks, int mtpTokens, long decodeTicks, long samplerTicks)
    {
        if (gate is null) return;
        if (MtpAdaptiveGate.TraceEnabled)
            Console.Error.WriteLine($"[mtp] record: used={(usedMtp ? "MTP" : "plain")} generated={generated} mtpTokens={mtpTokens} mtpMs={mtpTicks * 1000.0 / Stopwatch.Frequency:F0} decodeMs={decodeTicks * 1000.0 / Stopwatch.Frequency:F0} samplerMs={samplerTicks * 1000.0 / Stopwatch.Frequency:F0}");
        if (!usedMtp)
        {
            gate.Record(false, generated, (decodeTicks + samplerTicks) * 1000.0 / Stopwatch.Frequency);
            return;
        }
        gate.RecordSegment(true, mtpTokens, mtpTicks * 1000.0 / Stopwatch.Frequency);
        int plainTokens = generated - 1 - mtpTokens;
        if (plainTokens > 0)
            gate.RecordSegment(false, plainTokens, (decodeTicks - mtpTicks + samplerTicks) * 1000.0 / Stopwatch.Frequency);
    }

    // Sampled requests speculate too (#900): the decoder does exact speculative sampling against the prepared distribution. Logprob capture still
    // opts out (no per-position logits to read back).
    private bool ShouldUseMtp(bool captureLogprobs)
        => _mtpEnabled && _draftModel is null && _model.SupportsMtp && !captureLogprobs;

    /// <summary>
    /// Runs the prompt-suffix prefill forward pass(es) against <paramref name="kvCache"/>.
    /// When <see cref="_prefillChunkSize"/> is 0 the whole suffix runs in a single forward pass
    /// (behavior unchanged); otherwise the suffix is split into chunks of at most that many tokens
    /// (llama.cpp <c>-ub</c> analog — bounds peak activation memory per forward pass). Returns the
    /// logits tensor of the <b>last</b> chunk only (the caller samples from its final row); earlier
    /// chunks' logits are disposed here.
    /// Every chunk opts in to <c>lastTokenLogitsOnly</c> (issue #493) — only row
    /// <c>Shape[0] - 1</c> of the final chunk is ever read, so the caller must index the last row
    /// by shape and never assume one row per input token.
    /// <para>With no chunk size configured, the model's own <see cref="IModel.PreferredPrefillChunkTokens"/>
    /// applies (issue #1025). The engine then splits exactly where the model would split internally, so the
    /// result is unchanged, and <paramref name="cancellationToken"/> is checked before every chunk. Without
    /// that, a cancelled request (client gone, shutdown) ran a long prefill to completion.</para>
    /// </summary>
    private ITensor ForwardPrefill(int[] promptIds, int prefillStart, int prefillLen,
        Core.Attention.IKvCache kvCache, ILoraAdapter? adapter, DotLLM.Core.Models.IMtpState? mtpState = null,
        int snapshotAt = -1, Action? onSnapshotPoint = null, CancellationToken cancellationToken = default)
    {
        int configured = EffectivePrefillChunkSize;
        int chunkSize = configured > 0 ? Math.Min(configured, prefillLen) : prefillLen;
        int[] positionsArray = ArrayPool<int>.Shared.Rent(chunkSize);
        ITensor? logits = null;
        try
        {
            int offset = 0;
            while (offset < prefillLen)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int len = Math.Min(chunkSize, prefillLen - offset);
                // A recurrent-prefix snapshot needs a forward boundary exactly at snapshotAt.
                int absolute = prefillStart + offset;
                if (snapshotAt > absolute && snapshotAt < prefillStart + prefillLen)
                    len = Math.Min(len, snapshotAt - absolute);
                Span<int> positions = positionsArray.AsSpan(0, len);
                for (int i = 0; i < len; i++)
                    positions[i] = prefillStart + offset + i;

                logits?.Dispose();
                // Issue #493: this driver samples row Shape[0]-1 of the LAST chunk and nothing
                // else — every earlier chunk's logits are disposed unread, and within the last
                // chunk only the final row is read. So opt in to last-row-only logits. Models that
                // ignore the hint keep returning [len, vocab] and both call sites index
                // Shape[0]-1, so the behaviour is unchanged for them.
                logits = _model.Forward(promptIds.AsSpan(prefillStart + offset, len), positions,
                    deviceId: -1, kvCache, adapter, mtpState, lastTokenLogitsOnly: true);
                offset += len;
                if (onSnapshotPoint is not null && prefillStart + offset == snapshotAt)
                    onSnapshotPoint();
            }
            return logits!;
        }
        catch
        {
            logits?.Dispose();
            throw;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positionsArray);
        }
    }

    /// <summary>
    /// Prefills the draft model with the full prompt. Honors <see cref="_prefillChunkSize"/>
    /// the same way the target-model prefill does.
    /// </summary>
    private void PrefillDraftModel(int[] promptIds, Core.Attention.IKvCache draftKvCache)
    {
        int promptLen = promptIds.Length;
        int chunkSize = _prefillChunkSize > 0 ? Math.Min(_prefillChunkSize, promptLen) : promptLen;
        int[] positions = ArrayPool<int>.Shared.Rent(chunkSize);
        try
        {
            int offset = 0;
            while (offset < promptLen)
            {
                int len = Math.Min(chunkSize, promptLen - offset);
                for (int i = 0; i < len; i++)
                    positions[i] = offset + i;

                // The draft prefill discards its logits entirely — it runs only to populate the
                // draft KV-cache — so the last-row hint is always safe here (issue #493).
                using ITensor _ = _draftModel!.Forward(promptIds.AsSpan(offset, len),
                    positions.AsSpan(0, len), deviceId: -1, draftKvCache, lastTokenLogitsOnly: true);
                offset += len;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positions);
        }
    }

    /// <summary>
    /// Allocates a KV-cache for the draft model.
    /// </summary>
    private Core.Attention.IKvCache AllocateDraftKvCache(int cacheSize)
    {
        if (_draftKvCacheFactory != null)
            return _draftKvCacheFactory(_draftModel!.Config, cacheSize);

        return new SimpleKvCache(
            Core.Attention.KvGeometry.FromConfig(_draftModel!.Config, _draftModel.SupportsCompactSlidingKv),
            cacheSize);
    }

    private InferenceResponse BuildResponse(int promptLen, List<int> generatedIds,
        FinishReason finishReason, long prefillTicks, long decodeTicks, long samplerTicks,
        long kvCacheBytes = 0, int cachedTokenCount = 0,
        int specDrafted = 0, int specAccepted = 0,
        TokenLogprobInfo[]? logprobs = null,
        List<IStopCondition>? stopConditionsForSuffixTrim = null)
    {
        string text = generatedIds.Count > 0
            ? _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), stripBosSpace: false)
            : string.Empty;

        // Character-level stop-string suffix trim. When generation stopped because a
        // StopStringCondition matched, the last token is kept in `generatedIds` so the
        // user can see how many tokens were actually emitted, but its decoded text may
        // contain a partial overlap with the stop string (e.g. last token decodes to
        // "ld<|im_end|>", stop string "<|im_end|>"). Trim at the char boundary so the
        // returned text preserves the "ld" prefix and excludes the matched suffix.
        string? matchedStop = null;
        if (stopConditionsForSuffixTrim is not null && finishReason == FinishReason.Stop)
        {
            // Capture WHICH stop string matched before trimming it away — afterwards the text no
            // longer carries the evidence, and the Anthropic stop_sequence mapping needs it (#459).
            matchedStop = StopSuffixTrimmer.MatchedSuffix(text.AsSpan(), stopConditionsForSuffixTrim);
            text = StopSuffixTrimmer.TrimMatchedSuffix(text, stopConditionsForSuffixTrim);
        }

        return new InferenceResponse
        {
            GeneratedTokenIds = generatedIds.ToArray(),
            Text = text,
            FinishReason = finishReason,
            PromptTokenCount = promptLen,
            GeneratedTokenCount = generatedIds.Count,
            Timings = BuildTimings(promptLen, generatedIds.Count, prefillTicks, decodeTicks, samplerTicks, kvCacheBytes, cachedTokenCount, specDrafted, specAccepted),
            Logprobs = logprobs,
            MatchedStopSequence = matchedStop,
        };
    }

    private InferenceTimings BuildTimings(int promptLen, int generatedCount,
        long prefillTicks, long decodeTicks, long samplerTicks, long kvCacheBytes = 0,
        int cachedTokenCount = 0, int specDrafted = 0, int specAccepted = 0)
    {
        double tickFreq = Stopwatch.Frequency;
        int decodeSteps = generatedCount > 1 ? generatedCount - 1 : 0;

        return new InferenceTimings
        {
            PrefillTimeMs = prefillTicks / tickFreq * 1000.0,
            DecodeTimeMs = decodeTicks / tickFreq * 1000.0,
            SamplingTimeMs = samplerTicks / tickFreq * 1000.0,
            PrefillTokenCount = promptLen,
            DecodeTokenCount = decodeSteps,
            KvCacheBytes = kvCacheBytes,
            CachedTokenCount = cachedTokenCount,
            CacheTier = LastPrefixCacheTier,
            CacheRestoreMilliseconds = LastPrefixRestoreMilliseconds,
            SpeculativeDraftTokens = specDrafted,
            SpeculativeAcceptedTokens = specAccepted
        };
    }

    /// <summary>
    /// Extracts allocated bytes from a KV-cache, regardless of concrete type.
    /// </summary>
    internal static long GetKvCacheBytes(Core.Attention.IKvCache kvCache) => kvCache switch
    {
        KvCache.SimpleKvCache simple => simple.AllocatedBytes,
        KvCache.QuantizedKvCache quantized => quantized.AllocatedBytes,
        KvCache.PagedKvCache paged => paged.AllocatedBytes,
        _ => 0 // GPU caches — AllocatedBytes is on the concrete type, accessed by CLI directly
    };
}
