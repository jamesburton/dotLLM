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

        // Explicit steps provided — use composable path
        if (options.SamplerSteps is not null)
        {
            _greedy = false;
            _fastTopKSampling = false;
            _steps = options.SamplerSteps.ToArray();

            // Build processors: use explicit list if provided, otherwise auto-build from flat properties
            _processors = options.LogitProcessors is not null
                ? options.LogitProcessors.ToArray()
                : BuildProcessors(options, tokenizer);

            _processorContext = BuildProcessorContext(options);
            _samplerContext = new SamplerContext(
                options.Temperature,
                options.TopK,
                options.TopP,
                options.MinP,
                options.Seed,
                options.TopNSigma);
            return;
        }

        // Auto-build from flat properties
        _greedy = options.Temperature <= 0f;
        _fastTopKSampling = !_greedy
            && options.TopK > 0
            && options.TopP >= 1.0f
            && options.MinP <= 0f
            && options.TopNSigma < 0f;

        // Build processor chain (only add if enabled)
        _processors = options.LogitProcessors is not null
            ? options.LogitProcessors.ToArray()
            : BuildProcessors(options, tokenizer);

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
            foreach (var id in tokenizer.Encode(breaker))
                tokenIds.Add(id);
        }
        return tokenIds.Count > 0 ? tokenIds : null;
    }

    /// <summary>
    /// Samples a token from the given logits, applying all enabled processors and steps.
    /// </summary>
    /// <param name="logits">Logit values to sample from (modified in-place).</param>
    /// <param name="previousTokens">Previously generated token IDs for repetition penalty.</param>
    /// <returns>The sampled token index.</returns>
    public int Sample(Span<float> logits, IReadOnlyList<int> previousTokens)
    {
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
