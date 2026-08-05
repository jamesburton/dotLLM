using DotLLM.Core.Attention;
using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Regression coverage for issue #261: perplexity scoring leaked <em>model-owned recurrent state</em>
/// across sequence boundaries on every hybrid / SSM architecture.
/// </summary>
/// <remarks>
/// <para>Two distinct leaks, both silent — they moved the reported number without raising anything:</para>
/// <list type="number">
///   <item><description><see cref="BackendPerplexityModel.Probe"/> runs a throwaway two-token forward
///   to detect all-rows logits. On a recurrent model that forward advances the model-owned Gated
///   DeltaNet / SSM state, so the probe's two junk tokens were effectively prepended to the FIRST
///   scored window.</description></item>
///   <item><description>Perplexity treats every window as an independent sequence (positions restart
///   at 0), but the uncached <c>Forward</c> carries no state container and falls back to the same
///   model-owned state — so window <c>n</c> inherited window <c>n-1</c>'s recurrence.</description></item>
/// </list>
/// <para><b>What these tests can and cannot prove.</b> No known-good hybrid fixture exists at time of
/// writing — the two real hybrid checkpoints on hand have separate live numerics/memory defects
/// (issues #262 and #260) — so "perplexity gets closer to the true value" is not demonstrable. It is
/// also not the property that was broken. The property that was broken is <b>sequence
/// independence</b>, and that is provable without a numerically-correct model: it only needs a
/// deterministic one. Each test below therefore pins an invariant of the form "two paths that are
/// supposed to score the same sequence must produce bit-identical output", which is false before the
/// fix and true after it, regardless of whether the underlying numbers are any good.</para>
/// <para>The tiny synthetic <c>qwen35moe</c> fixture is the vehicle: layer 0 is a Gated DeltaNet
/// layer (real recurrence, a real <c>GdnStateCache</c>) and its weights come from a seeded PRNG, so
/// the model is fully deterministic.</para>
/// </remarks>
public sealed class RecurrentPerplexityStateTests : IDisposable
{
    private readonly string _scratch;
    private readonly string _ggufPath;

    // The CPU model reads its weights straight out of the mapped GGUF, so the file must outlive
    // every model loaded from it. Held here and released after all models are disposed.
    private readonly List<GgufFile> _openFiles = [];

    // Inside the fixture's 8-entry vocabulary. Two windows of 8 at stride 4 exercise the
    // window-to-window boundary, which one window could not.
    private static readonly int[] Corpus = [1, 2, 3, 4, 5, 6, 7, 0, 3, 1, 4, 1, 5, 2, 6, 0];

    private static readonly PerplexityOptions SlidingOptions =
        new(PerplexityMode.SlidingWindow, ContextLength: 8, Stride: 4);

    public RecurrentPerplexityStateTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-ppl-recurrent-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
        _ggufPath = SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "qwen35moe-tiny.gguf"));
    }

    public void Dispose()
    {
        foreach (var f in _openFiles) f.Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>Loads a fresh, never-forwarded CPU model from the synthetic hybrid fixture.</summary>
    private IModel LoadPristineModel()
    {
        var gguf = GgufFile.Open(_ggufPath);
        _openFiles.Add(gguf);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.True(model.RequiresPerSequenceState,
            "fixture must be a recurrent architecture or these tests prove nothing");
        return model;
    }

    /// <summary>
    /// THE load-bearing test: scoring the same corpus twice through the same model instance must
    /// give the same answer.
    /// </summary>
    /// <remarks>
    /// Perplexity is a pure function of (weights, tokens, window geometry) — nothing about the second
    /// run differs from the first. Before the fix it differed anyway, because run 1 left the GDN
    /// state populated and run 2's first window started from it. That inequality is the bug made
    /// directly observable, and it needs no assumption about whether the model's numerics are
    /// otherwise correct.
    /// </remarks>
    [Fact]
    public void ScoringTwiceThroughTheSameModel_GivesIdenticalPerplexity()
    {
        using IModel model = LoadPristineModel();
        var perplexityModel = new BackendPerplexityModel(
            model, deviceId: -1, BackendPerplexityModel.Probe(model, deviceId: -1));
        Assert.True(perplexityModel.ReturnsAllRows, "sliding-window mode needs all-rows logits");

        var first = PerplexityEvaluator.Evaluate(perplexityModel, Corpus, SlidingOptions);
        var second = PerplexityEvaluator.Evaluate(perplexityModel, Corpus, SlidingOptions);

        Assert.True(first.WindowCount > 1, "geometry must span a window boundary");
        Assert.Equal(first.ScoredTokens, second.ScoredTokens);
        Assert.Equal(first.MeanNegativeLogLikelihood, second.MeanNegativeLogLikelihood);
        Assert.Equal(first.Perplexity, second.Perplexity);
    }

    /// <summary>
    /// Each window must score identically whether it is reached as window 0 of a run or as window
    /// <c>n</c> of a longer run — i.e. windows must not see each other.
    /// </summary>
    /// <remarks>
    /// Stronger than the repeat test and complementary to it: the repeat test would still pass if
    /// every run were corrupted the SAME way, whereas this one compares the whole-corpus run against
    /// each window scored in isolation on a model that has seen nothing else. The per-window figures
    /// come from the evaluator's diagnostic <c>WindowObserver</c>, which is what makes the
    /// comparison per-window rather than aggregate — an aggregate could hide two windows' errors
    /// cancelling.
    /// </remarks>
    [Fact]
    public void EachWindowScoresTheSame_WhetherOrNotEarlierWindowsRanFirst()
    {
        var observed = new List<double>();
        using (IModel model = LoadPristineModel())
        {
            var perplexityModel = new BackendPerplexityModel(
                model, deviceId: -1, BackendPerplexityModel.Probe(model, deviceId: -1));
            PerplexityEvaluator.Evaluate(
                perplexityModel, Corpus, SlidingOptions,
                (_, windowPerplexity, _) => observed.Add(windowPerplexity));
        }

        Assert.True(observed.Count > 1, "geometry must span a window boundary");

        // Re-score each window alone, on a model that has never seen any other window.
        for (int w = 0; w < observed.Count; w++)
        {
            int start = w * SlidingOptions.Stride;
            int[] isolated = Corpus[start..(start + SlidingOptions.ContextLength)];

            using IModel model = LoadPristineModel();
            var perplexityModel = new BackendPerplexityModel(
                model, deviceId: -1, BackendPerplexityModel.Probe(model, deviceId: -1));
            var alone = PerplexityEvaluator.Evaluate(perplexityModel, isolated, SlidingOptions);

            Assert.Equal(1, alone.WindowCount);
            Assert.Equal(observed[w], alone.Perplexity, 12);
        }
    }

    /// <summary>
    /// <see cref="BackendPerplexityModel.Probe"/> must leave the model exactly as it found it.
    /// </summary>
    /// <remarks>
    /// Tested at the model level rather than through the evaluator on purpose: the evaluator now
    /// also resets before its first window, which would mask a probe that dirties state. Comparing a
    /// probed model's first forward against a pristine model's first forward isolates the probe.
    /// </remarks>
    [Fact]
    public void Probe_LeavesRecurrentStateUntouched()
    {
        int[] window = Corpus[..8];
        int[] positions = [0, 1, 2, 3, 4, 5, 6, 7];

        float[] pristine;
        using (IModel model = LoadPristineModel())
            pristine = ForwardToArray(model, window, positions);

        float[] afterProbe;
        using (IModel model = LoadPristineModel())
        {
            BackendPerplexityModel.Probe(model, deviceId: -1);
            afterProbe = ForwardToArray(model, window, positions);
        }

        Assert.Equal(pristine.Length, afterProbe.Length);
        Assert.Equal(pristine, afterProbe);
    }

    /// <summary>
    /// A model that declares <see cref="IModel.RequiresPerSequenceState"/> must not be able to
    /// inherit a silent no-op <see cref="IModel.ResetSequenceState"/>.
    /// </summary>
    /// <remarks>
    /// This is the "cannot be forgotten" guard. The whole defect class is a recurrent architecture
    /// whose model-owned state is never cleared; a no-op default would let the next one added
    /// reintroduce it, and — as issue #261 records — the symptom is a plausible wrong number, not a
    /// crash. So the default throws for recurrent models and only stateless models inherit it.
    /// </remarks>
    [Fact]
    public void ResetSequenceState_DefaultThrowsForARecurrentModelThatForgotToImplementIt()
    {
        // Through the interface: ResetSequenceState is a default interface member, so the guard
        // lives on IModel and is only reachable via an IModel-typed reference.
        using IModel forgetful = new StubModel(requiresPerSequenceState: true);
        var ex = Assert.Throws<NotSupportedException>(forgetful.ResetSequenceState);
        Assert.Contains(nameof(IModel.ResetSequenceState), ex.Message, StringComparison.Ordinal);

        using IModel stateless = new StubModel(requiresPerSequenceState: false);
        stateless.ResetSequenceState();   // no-op; must not throw
    }

    private static unsafe float[] ForwardToArray(IModel model, int[] tokens, int[] positions)
    {
        using ITensor logits = model.Forward(tokens, positions, deviceId: -1);
        var result = new float[logits.ElementCount];
        new ReadOnlySpan<float>((void*)logits.DataPointer, result.Length).CopyTo(result);
        return result;
    }

    /// <summary>Minimal <see cref="IModel"/> that implements nothing beyond the flag under test.</summary>
    private sealed class StubModel(bool requiresPerSequenceState) : IModel
    {
        public ModelConfig Config { get; } = new()
        {
            VocabSize = 4,
            NumLayers = 1,
            NumAttentionHeads = 1,
            NumKvHeads = 1,
            HiddenSize = 4,
            IntermediateSize = 8,
            HeadDim = 4,
            MaxSequenceLength = 8,
            Architecture = Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public bool RequiresPerSequenceState { get; } = requiresPerSequenceState;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => throw new NotSupportedException();

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                               IKvCache? kvCache)
            => throw new NotSupportedException();

        public void Dispose() { }
    }
}
