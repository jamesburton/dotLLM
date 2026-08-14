using DotLLM.Core.Configuration;
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
/// Coverage for issue #395's layer-cycling perplexity path: scoring a model by sliding a layer
/// window across the trunk, checkpointing the hidden state at each cut, instead of holding the whole
/// model on one device.
/// </summary>
/// <remarks>
/// <para><b>Why the fixture is a hybrid and not a dense transformer.</b> The failure this mode is
/// most exposed to is recurrent-state leakage between corpus windows — issue #261's bug, multiplied,
/// because each layer window replays the entire corpus. A dense transformer carries nothing across
/// an uncached forward, so a dense-only test passes identically whether the per-window reset is
/// present or absent and proves nothing. <see cref="SyntheticQwen35MoeGguf"/> alternates Gated
/// DeltaNet and full-attention layers from a seeded PRNG, so it is both recurrent and fully
/// deterministic; the tests below run it four layers deep so that GDN layers sit on <em>both</em>
/// sides of the layer cut.</para>
/// <para><b>Why the equality is asserted tight.</b> A cycled CPU run and a whole-model CPU run
/// execute the same kernels over the same values in the same order — the only difference is that the
/// residual stream makes a round trip through a host FP32 buffer at the cut, which is exact. So the
/// two agree to floating-point identity, and a loose tolerance here would hide a real defect. The
/// error-budget reasoning that scales a cross-backend tolerance with sqrt(K) applies to comparisons
/// between two different implementations of the same arithmetic, which this is not.</para>
/// </remarks>
public sealed class LayerCyclingPerplexityTests : IDisposable
{
    private readonly string _scratch;
    private readonly string _ggufPath;
    private readonly List<GgufFile> _openFiles = [];

    /// <summary>Trunk depth of the fixture: GDN at layers 0 and 2, attention at 1 and 3.</summary>
    private const int BlockCount = 4;

    /// <summary>
    /// Two windows of 8 tokens at stride 4. Two windows is the minimum that can observe
    /// cross-window state leakage at all — with one window the correct and the broken forms
    /// coincide, which is exactly the degenerate shape that has let a bug ship in this project
    /// before. <see cref="CycledScoring_WithoutPerWindowReset_IsIndistinguishableOnASingleWindow"/>
    /// pins that degeneracy so the choice cannot be quietly undone.
    /// </summary>
    private static readonly int[] Corpus = [1, 2, 3, 4, 5, 6, 7, 0, 3, 1, 4, 1, 5, 2, 6, 0];

    private static readonly PerplexityOptions SlidingOptions =
        new(PerplexityMode.SlidingWindow, ContextLength: 8, Stride: 4);

    public LayerCyclingPerplexityTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-ppl-cycle-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
        _ggufPath = SyntheticQwen35MoeGguf.Write(
            Path.Combine(_scratch, "qwen35moe-tiny-4l.gguf"), blockCount: BlockCount);
    }

    public void Dispose()
    {
        foreach (GgufFile f in _openFiles) f.Dispose();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (IModel Model, ModelConfig Config) LoadPristineModel()
    {
        var gguf = GgufFile.Open(_ggufPath);
        _openFiles.Add(gguf);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        Assert.Equal(BlockCount, config.NumLayers);
        IModel model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.True(model.RequiresPerSequenceState,
            "the fixture must be recurrent or these tests cannot discriminate the reset");
        return (model, config);
    }

    /// <summary>
    /// The fixture must place recurrent layers on both sides of the cut used below, or a cycled run
    /// would only ever exercise recurrence in one pass and the coverage would be illusory.
    /// </summary>
    [Fact]
    public void Fixture_HasRecurrentLayersOnBothSidesOfTheCut()
    {
        (IModel model, ModelConfig config) = LoadPristineModel();
        using (model)
        {
            HybridLayerKind[] kinds = config.HybridLayout!.LayerKind;
            const int cut = BlockCount / 2;
            Assert.Contains(HybridLayerKind.GatedDeltaNet, kinds[..cut]);
            Assert.Contains(HybridLayerKind.GatedDeltaNet, kinds[cut..]);
        }
    }

    /// <summary>
    /// THE acceptance test: a cycled run and a whole-model run must produce the same number.
    /// </summary>
    [Fact]
    public void CycledScoring_MatchesWholeModelScoring()
    {
        double expected = ScoreWholeModel();
        double actual = ScoreCycled(windowSize: BlockCount / 2, sabotageReset: false, Corpus, SlidingOptions);

        Assert.Equal(expected, actual, 12);
    }

    /// <summary>
    /// One layer per window — the maximum number of cuts — must also agree, so the equality is not
    /// an accident of cutting exactly once.
    /// </summary>
    [Fact]
    public void CycledScoring_MatchesWholeModelScoring_WithOneLayerPerWindow()
    {
        double expected = ScoreWholeModel();
        double actual = ScoreCycled(windowSize: 1, sabotageReset: false, Corpus, SlidingOptions);

        Assert.Equal(expected, actual, 12);
    }

    /// <summary>
    /// THE discriminating test: dropping the per-corpus-window recurrent reset inside a layer window
    /// must move the number.
    /// </summary>
    /// <remarks>
    /// This is the assertion that gives the two above their meaning. Without it they would pass just
    /// as happily against an implementation that leaks Gated-DeltaNet state from corpus window
    /// <c>n-1</c> into corpus window <c>n</c> on every pass — the exact failure of issue #261 — since
    /// the leak is silent and moves the perplexity rather than raising. Asserting inequality proves
    /// the test can tell the broken form from the fixed one.
    /// </remarks>
    [Fact]
    public void CycledScoring_WithoutPerWindowReset_ProducesADifferentNumber()
    {
        double correct = ScoreCycled(windowSize: BlockCount / 2, sabotageReset: false, Corpus, SlidingOptions);
        double leaked = ScoreCycled(windowSize: BlockCount / 2, sabotageReset: true, Corpus, SlidingOptions);

        Assert.NotEqual(correct, leaked, 6);
    }

    /// <summary>
    /// Pins the degenerate shape: over a single corpus window the leaked and the reset forms
    /// coincide, so a one-window corpus cannot discriminate them.
    /// </summary>
    /// <remarks>
    /// Recorded as a test rather than a comment because the corpus length is the kind of incidental
    /// constant a later edit shortens without noticing that it disarms the test above.
    /// </remarks>
    [Fact]
    public void CycledScoring_WithoutPerWindowReset_IsIndistinguishableOnASingleWindow()
    {
        int[] oneWindow = Corpus[..8];
        // Same window shape as SlidingOptions (context 8, unscored prefix 4) so that only the
        // number of windows differs from the discriminating test; the stride is irrelevant with a
        // corpus exactly one window long.
        var options = new PerplexityOptions(
            PerplexityMode.SlidingWindow, ContextLength: 8, Stride: 8, UnscoredPrefix: 4);

        double correct = ScoreCycled(BlockCount / 2, sabotageReset: false, oneWindow, options);
        double leaked = ScoreCycled(BlockCount / 2, sabotageReset: true, oneWindow, options);

        Assert.Equal(correct, leaked, 12);
    }

    /// <summary>A cycling partition that does not cover every layer must be rejected, not scored.</summary>
    [Fact]
    public void Cycling_RejectsAPartitionThatDoesNotCoverTheTrunk()
    {
        (IModel model, ModelConfig config) = LoadPristineModel();
        using (model)
        using (var windows = new CpuLayerWindowModel(model, config))
        {
            LayerWindow[] partial = [new LayerWindow(0, BlockCount - 1)];
            Assert.Throws<ArgumentException>(() =>
                CyclingPerplexityEvaluator.Evaluate(windows, Corpus, SlidingOptions, partial));

            LayerWindow[] gapped = [new LayerWindow(0, 1), new LayerWindow(2, BlockCount - 2)];
            Assert.Throws<ArgumentException>(() =>
                CyclingPerplexityEvaluator.Evaluate(windows, Corpus, SlidingOptions, gapped));
        }
    }

    /// <summary><see cref="CyclingPerplexityEvaluator.PartitionLayers"/> tiles the trunk exactly once.</summary>
    [Fact]
    public void PartitionLayers_TilesTheTrunkWithARaggedLastWindow()
    {
        IReadOnlyList<LayerWindow> windows = CyclingPerplexityEvaluator.PartitionLayers(numLayers: 7, windowSize: 3);

        Assert.Equal(3, windows.Count);
        Assert.Equal(new LayerWindow(0, 3), windows[0]);
        Assert.Equal(new LayerWindow(3, 3), windows[1]);
        Assert.Equal(new LayerWindow(6, 1), windows[2]);
    }

    private double ScoreWholeModel()
    {
        (IModel model, _) = LoadPristineModel();
        using (model)
        {
            var perplexityModel = new BackendPerplexityModel(
                model, deviceId: -1, BackendPerplexityModel.Probe(model, deviceId: -1));
            Assert.True(perplexityModel.ReturnsAllRows, "sliding-window mode needs all-rows logits");
            return PerplexityEvaluator.Evaluate(perplexityModel, Corpus, SlidingOptions)
                .MeanNegativeLogLikelihood;
        }
    }

    private double ScoreCycled(int windowSize, bool sabotageReset, int[] corpus, PerplexityOptions options)
    {
        (IModel model, ModelConfig config) = LoadPristineModel();
        using (model)
        {
            using ILayerWindowModel windows = new CpuLayerWindowModel(model, config);
            ILayerWindowModel target = sabotageReset ? new NoResetLayerWindowModel(windows) : windows;
            return CyclingPerplexityEvaluator.Evaluate(
                    target, corpus, options,
                    CyclingPerplexityEvaluator.PartitionLayers(config.NumLayers, windowSize))
                .MeanNegativeLogLikelihood;
        }
    }

    /// <summary>
    /// Decorator that swallows <see cref="ILayerWindowExecutor.ResetState"/>, reproducing the
    /// pre-#261 behaviour so the tests can assert the two forms are distinguishable.
    /// </summary>
    private sealed class NoResetLayerWindowModel(ILayerWindowModel inner) : ILayerWindowModel
    {
        public int NumLayers => inner.NumLayers;

        public int HiddenSize => inner.HiddenSize;

        public int VocabSize => inner.VocabSize;

        public int MaxContextLength => inner.MaxContextLength;

        public ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount)
            => new NoResetExecutor(inner.CreateWindow(firstLayer, layerCount));

        public ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen)
            => inner.ApplyOutputHead(hidden, seqLen);

        /// <summary>No-op: the decorator borrows <paramref name="inner"/>, which the test disposes.</summary>
        public void Dispose()
        {
        }

        private sealed class NoResetExecutor(ILayerWindowExecutor inner) : ILayerWindowExecutor
        {
            public int FirstLayer => inner.FirstLayer;

            public int LayerCount => inner.LayerCount;

            public void Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> hiddenIn,
                            ReadOnlySpan<int> positions, Span<float> hiddenOut)
                => inner.Run(tokenIds, hiddenIn, positions, hiddenOut);

            public void ResetState()
            {
                // Deliberately nothing — this is the bug under test.
            }

            public void Dispose() => inner.Dispose();
        }
    }
}
