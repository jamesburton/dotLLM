using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Orchestrates the sampling pipeline: logit processors → sampler steps → final token selection.
/// Can be built automatically from <see cref="InferenceOptions"/> or composed explicitly
/// from individual <see cref="ISamplerStep"/> instances.
/// </summary>
public sealed class SamplerPipeline
{
    private readonly ILogitProcessor[] _processors;
    private readonly ISamplerStep[] _steps;
    private readonly ProcessorContext _processorContext;
    private readonly SamplerContext _samplerContext;
    private readonly Random _rng;
    private readonly bool _greedy;
    private readonly bool _fastTopKSampling;
    private readonly ulong _gumbelSeed;
    private readonly bool _gumbel;
    internal bool UnsafeUniform32ForTests { get; set; }

    /// <summary>Env switch that turns position-seeded Gumbel-max sampling on for every stochastic pipeline built from options (#1358).</summary>
    public const string GumbelEnvVar = "DOTLLM_GUMBEL_SAMPLING";

    /// <summary>
    /// True when a stochastic draw is <c>argmax(prepared + Gumbel(seed, generation index))</c> (#1358) instead of a draw from the
    /// pipeline's sequential <see cref="Random"/>. Speculative verification then reproduces the plain-decode text for the same seed.
    /// </summary>
    public bool UsesGumbel => _gumbel;

    /// <summary>
    /// Creates a composable sampling pipeline from explicit steps.
    /// Steps are applied in the order provided, followed by categorical sampling.
    /// </summary>
    /// <param name="steps">Sampler steps to apply in order (e.g., temperature → top-K → top-P → min-P).</param>
    public SamplerPipeline(params ISamplerStep[] steps)
        : this(processors: null, steps: steps, seed: null)
    {
    }

    /// <summary>
    /// Creates a composable sampling pipeline from explicit processors and steps.
    /// </summary>
    /// <param name="processors">Logit processors (e.g., repetition penalty). Applied before steps.</param>
    /// <param name="steps">Sampler steps to apply in order.</param>
    /// <param name="seed">Random seed for reproducible sampling. Null = non-deterministic.</param>
    public SamplerPipeline(
        IReadOnlyList<ILogitProcessor>? processors,
        IReadOnlyList<ISamplerStep> steps,
        int? seed = null)
    {
        _greedy = false;
        _fastTopKSampling = false;
        _rng = seed.HasValue ? new Random(seed.Value) : new Random();
        _processors = processors?.ToArray() ?? [];
        _steps = steps.ToArray();
        _processorContext = new ProcessorContext(1.0f, 0, SequenceId: 0);
        _samplerContext = default;
    }

    /// <summary>
    /// Creates a new sampling pipeline from the given inference options.
    /// When <see cref="InferenceOptions.SamplerSteps"/> is set, uses those explicit steps.
    /// Otherwise builds steps automatically from flat properties, skipping disabled ones.
    /// </summary>
    /// <param name="options">Inference options controlling the pipeline shape.</param>
    /// <param name="tokenizer">
    /// Optional tokenizer used to resolve <see cref="InferenceOptions.DrySequenceBreakers"/> strings
    /// to token ids when DRY is enabled and no explicit <see cref="InferenceOptions.LogitProcessors"/>
    /// list is supplied. Null = DRY runs without sequence breakers (matches can span the full window).
    /// </param>
    public SamplerPipeline(InferenceOptions options, ITokenizer? tokenizer = null)
    {
        _rng = options.Seed.HasValue ? new Random(options.Seed.Value) : new Random();
        bool wantGumbel = options.GumbelSampling || Environment.GetEnvironmentVariable(GumbelEnvVar) == "1";
        if (wantGumbel)
            _gumbelSeed = options.Seed.HasValue ? (ulong)(uint)options.Seed.Value * 0x9E3779B97F4A7C15UL + 0x1234567UL : (ulong)_rng.NextInt64();

        // Explicit steps provided — use composable path
        if (options.SamplerSteps is not null)
        {
            _greedy = false;
            _fastTopKSampling = false;
            _steps = options.SamplerSteps.ToArray();

            // Build processors: use explicit list if provided, otherwise auto-build from flat properties
            _processors = ResolveProcessors(options, tokenizer);

            _processorContext = BuildProcessorContext(options);
            _samplerContext = new SamplerContext(
                options.Temperature,
                options.TopK,
                options.TopP,
                options.MinP,
                options.Seed,
                options.TopNSigma);
            _gumbel = wantGumbel;
            return;
        }

        // Auto-build from flat properties
        _greedy = options.Temperature <= 0f;
        _gumbel = wantGumbel && !_greedy;
        _fastTopKSampling = !_greedy
            && options.TopK > 0
            && options.TopP >= 1.0f
            && options.MinP <= 0f
            && options.TopNSigma < 0f;

        // Build processor chain (only add if enabled)
        _processors = ResolveProcessors(options, tokenizer);

        // Build sampler step chain (only add if enabled)
        var steps = new List<ISamplerStep>();
        if (!_greedy)
        {
            if (!_fastTopKSampling && options.Temperature != 1.0f)
                steps.Add(new TemperatureSampler());
            if (!_fastTopKSampling && options.TopK > 0)
                steps.Add(new TopKSampler());
            if (options.TopP < 1.0f)
                steps.Add(new TopPSampler());
            if (options.MinP > 0f)
                steps.Add(new MinPSampler());
            if (options.TopNSigma >= 0f)
                steps.Add(new TopNSigmaSampler());
        }
        _steps = steps.ToArray();

        _processorContext = BuildProcessorContext(options);

        _samplerContext = new SamplerContext(
            options.Temperature,
            options.TopK,
            options.TopP,
            options.MinP,
            options.Seed,
            options.TopNSigma);
    }

    /// <summary>
    /// The explicit <see cref="InferenceOptions.LogitProcessors"/> (or the auto-built chain), preceded by a fresh
    /// <see cref="ThinkingBudgetProcessor"/> when <see cref="InferenceOptions.ThinkingBudget"/> is set (#1292). It is stateful,
    /// so it is created here, per pipeline (= per sequence), never shared through the options.
    /// </summary>
    private static ILogitProcessor[] ResolveProcessors(InferenceOptions options, ITokenizer? tokenizer)
    {
        var processors = options.LogitProcessors is not null
            ? options.LogitProcessors.ToArray()
            : BuildProcessors(options, tokenizer);
        return options.ThinkingBudget is { } budget && tokenizer is not null
            ? [new ThinkingBudgetProcessor(budget, tokenizer), .. processors]
            : processors;
    }

    /// <summary>
    /// Auto-builds the logit processor chain from flat <see cref="InferenceOptions"/> properties,
    /// only including processors whose parameters are actually enabled.
    /// </summary>
    private static ILogitProcessor[] BuildProcessors(InferenceOptions options, ITokenizer? tokenizer)
    {
        var processors = new List<ILogitProcessor>();
        if (options.RepetitionPenalty != 1.0f)
            processors.Add(new RepetitionPenaltyProcessor());
        if (options.LogitBias is { Count: > 0 })
            processors.Add(new LogitBiasProcessor());
        if (options.FrequencyPenalty != 0f || options.PresencePenalty != 0f)
            processors.Add(new FrequencyPresencePenaltyProcessor());
        if (options.DryMultiplier > 0f)
            processors.Add(new DryProcessor(ResolveDryBreakerTokens(options.DrySequenceBreakers, tokenizer)));
        return processors.ToArray();
    }

    private static ProcessorContext BuildProcessorContext(InferenceOptions options) => new(
        options.RepetitionPenalty,
        options.RepetitionPenaltyWindow,
        SequenceId: 0,
        options.FrequencyPenalty,
        options.PresencePenalty,
        options.LogitBias,
        options.DryMultiplier,
        options.DryBase,
        options.DryAllowedLength,
        options.DryPenaltyLastN,
        options.DrySequenceBreakers);

    /// <summary>
    /// Resolves DRY sequence-breaker strings to token ids via the tokenizer. Every token produced by
    /// encoding a breaker string is added — for the common case (single-token breakers like a bare
    /// newline or comma) this exactly matches the intended token; multi-token breakers degrade to
    /// "any of its constituent tokens also breaks a match", a reasonable approximation without full
    /// multi-token sequence tracking. Returns null when there's no tokenizer or no breakers configured.
    /// </summary>
    private static IReadOnlySet<int>? ResolveDryBreakerTokens(
        IReadOnlyList<string>? breakers, ITokenizer? tokenizer)
    {
        if (tokenizer is null || breakers is null || breakers.Count == 0)
            return null;

        var tokenIds = new HashSet<int>();
        foreach (var breaker in breakers)
        {
            if (string.IsNullOrEmpty(breaker))
                continue;
            foreach (var id in tokenizer.EncodeRaw(breaker))
                tokenIds.Add(id);
        }
        return tokenIds.Count > 0 ? tokenIds : null;
    }

    /// <summary>True when the pipeline draws from a distribution; false when it is a (processor-adjusted) argmax.</summary>
    public bool IsStochastic => !_greedy;

    /// <summary>True when at least one logit processor (repetition / frequency / presence penalty, logit bias, DRY) runs before the argmax or the steps.</summary>
    public bool HasProcessors => _processors.Length > 0;

    private ISamplerStep[]? _fastPrepSteps;

    /// <summary>
    /// Turns <paramref name="logits"/> in place into the exact logits <see cref="Sample"/> would draw from (processors, then temperature /
    /// top-k / top-p / min-p / ...), without drawing. Speculative sampling (<see cref="TryAcceptDraft(ReadOnlySpan{float}, int, out int)"/>) needs the distribution, not just a draw.
    /// A greedy pipeline only applies its processors: the distribution is the argmax of the result.
    /// </summary>
    /// <param name="logits">Logits of one position (modified in place).</param>
    /// <param name="previousTokens">Tokens generated so far, for the penalty processors.</param>
    public void PrepareDistribution(Span<float> logits, IReadOnlyList<int> previousTokens)
    {
        for (int i = 0; i < _processors.Length; i++)
            _processors[i].Process(logits, previousTokens, _processorContext);
        if (_greedy) return;

        if (_fastTopKSampling)
        {
            // Sample() skips the steps on this shape and draws straight from the top-k candidates; the equivalent distribution is
            // temperature then top-k, which is what SampleTopK computes.
            _fastPrepSteps ??= _samplerContext.Temperature != 1.0f
                ? [new TemperatureSampler(), new TopKSampler()]
                : [new TopKSampler()];
            for (int i = 0; i < _fastPrepSteps.Length; i++)
                _fastPrepSteps[i].Apply(logits, _samplerContext);
            return;
        }
        for (int i = 0; i < _steps.Length; i++)
            _steps[i].Apply(logits, _samplerContext);
    }

    /// <summary>Draws from logits already prepared by <see cref="PrepareDistribution"/> (argmax for a greedy pipeline).</summary>
    public int SamplePrepared(ReadOnlySpan<float> preparedLogits)
        => _greedy ? CategoricalSampler.ArgMax(preparedLogits) : CategoricalSampler.Sample(preparedLogits, _rng);

    /// <summary>
    /// As <see cref="SamplePrepared(ReadOnlySpan{float})"/>, for the token at generation index <paramref name="tokenIndex"/> (= the number of tokens
    /// generated before it). With <see cref="UsesGumbel"/> the draw is the position-seeded Gumbel-max, a pure function of (logits, seed, index).
    /// </summary>
    public int SamplePrepared(ReadOnlySpan<float> preparedLogits, int tokenIndex)
        => _gumbel ? GumbelNoise.Sample(preparedLogits, _gumbelSeed, tokenIndex, UnsafeUniform32ForTests) : SamplePrepared(preparedLogits);

    /// <summary>
    /// <see cref="TryAcceptDraft(ReadOnlySpan{float}, int, out int)"/> for the token at generation index <paramref name="tokenIndex"/>. With
    /// <see cref="UsesGumbel"/> the row's own Gumbel-max sample is drawn and the draft is accepted iff it equals that sample: P(accept) is
    /// <c>p(draft)</c>, as in rejection sampling, and the emitted token is always the sample plain decoding would have drawn.
    /// </summary>
    public bool TryAcceptDraft(ReadOnlySpan<float> preparedLogits, int draftToken, out int token, int tokenIndex)
    {
        if (!_gumbel)
            return TryAcceptDraft(preparedLogits, draftToken, out token);
        token = GumbelNoise.Sample(preparedLogits, _gumbelSeed, tokenIndex, UnsafeUniform32ForTests);
        return token == draftToken;
    }

    /// <summary>
    /// Speculative-sampling accept/reject of ONE deterministic draft token against the prepared distribution <c>p</c> (the draft distribution is a
    /// point mass, so the accept probability <c>min(1, p(d)/1)</c> is <c>p(d)</c>). Accepted: the token is <paramref name="draftToken"/>. Rejected:
    /// <paramref name="token"/> is drawn from <c>p</c> with the draft removed and renormalised. Either way the emitted token is distributed exactly
    /// as <c>p</c>, which is why the result is the same as sampling the target alone. A greedy pipeline accepts iff the draft is the argmax.
    /// </summary>
    /// <param name="preparedLogits">Output of <see cref="PrepareDistribution"/> for this position.</param>
    /// <param name="draftToken">The head's draft for this position.</param>
    /// <param name="token">The token to emit (the draft when accepted).</param>
    /// <returns>True when the draft was accepted.</returns>
    public bool TryAcceptDraft(ReadOnlySpan<float> preparedLogits, int draftToken, out int token)
    {
        if (_greedy)
        {
            token = CategoricalSampler.ArgMax(preparedLogits);
            return token == draftToken;
        }

        int vocab = preparedLogits.Length;
        float[] rented = System.Buffers.ArrayPool<float>.Shared.Rent(vocab);
        try
        {
            Span<float> probs = rented.AsSpan(0, vocab);
            System.Numerics.Tensors.TensorPrimitives.SoftMax(preparedLogits, probs);
            double pd = probs[draftToken];
            if (_rng.NextDouble() < pd)
            {
                token = draftToken;
                return true;
            }
            probs[draftToken] = 0f;
            double rest = 0;
            for (int i = 0; i < vocab; i++) rest += probs[i];
            if (rest <= 1e-12)
            {
                token = draftToken;     // all the mass sits on the draft: rejecting it is numerically meaningless
                return true;
            }
            double r = _rng.NextDouble() * rest, cumulative = 0;
            int last = draftToken;
            for (int i = 0; i < vocab; i++)
            {
                float p = probs[i];
                if (p <= 0f) continue;
                last = i;
                cumulative += p;
                if (r < cumulative) { token = i; return false; }
            }
            token = last;               // rounding: the last token with mass
            return false;
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Samples a token from the given logits, applying all enabled processors and steps.
    /// </summary>
    /// <param name="logits">Logit values to sample from (modified in-place).</param>
    /// <param name="previousTokens">Previously generated token IDs for repetition penalty.</param>
    /// <returns>The sampled token index.</returns>
    public int Sample(Span<float> logits, IReadOnlyList<int> previousTokens)
    {
        if (_gumbel)
        {
            // Position-seeded Gumbel-max (#1358): the generation index is previousTokens.Count.
            PrepareDistribution(logits, previousTokens);
            return GumbelNoise.Sample(logits, _gumbelSeed, previousTokens.Count, UnsafeUniform32ForTests);
        }

        // 1. Run logit processors (repetition penalty)
        for (int i = 0; i < _processors.Length; i++)
            _processors[i].Process(logits, previousTokens, _processorContext);

        // 2. Greedy: argmax, skip everything else
        if (_greedy)
            return CategoricalSampler.ArgMax(logits);

        // 3. Run sampler steps (temperature → top-k → top-p → min-p)
        if (_fastTopKSampling)
            return CategoricalSampler.SampleTopK(
                logits,
                _samplerContext.TopK,
                _samplerContext.Temperature,
                _rng);

        for (int i = 0; i < _steps.Length; i++)
            _steps[i].Apply(logits, _samplerContext);

        // 4. Categorical sample
        return CategoricalSampler.Sample(logits, _rng);
    }
}
