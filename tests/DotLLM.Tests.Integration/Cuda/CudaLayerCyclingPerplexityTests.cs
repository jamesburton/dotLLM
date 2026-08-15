using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// The CUDA half of issue #395's acceptance: a layer-cycled run — the GPU holding only a slice of
/// the trunk at a time and resuming each slice from checkpointed boundary activations — must agree
/// with a whole-device run on a model that fits both ways.
/// </summary>
/// <remarks>
/// <para><b>Why teacher-forced and not sliding-window for the equivalence check.</b> The whole-device
/// CUDA model's <c>Forward</c> returns only the last logit row, so it cannot run sliding-window mode
/// at all (<c>PerplexityEvaluator</c> throws); its only scoring path is the growing-prefix
/// teacher-forced one. So the like-for-like comparison against a whole-device run has to be
/// teacher-forced, over a short corpus because that path is O(n^2) in forwards.</para>
/// <para><b>Why the tolerance is a bound rather than an equality.</b> Unlike the CPU cycled-vs-whole
/// test — which is an identity, because both sides run the same kernels in the same order — this
/// comparison changes two things at once. The residual stream makes an FP16 round trip at every layer
/// cut (the boundary's separate add + RMSNorm replaces an in-window fused add-RMSNorm), and the cycled
/// path applies the output head on the host in FP32 where the whole-device path applies it on the GPU
/// in FP16. Both are rounding, not logic, but they are real, and the FP16 error accumulates over the
/// reduction length rather than staying constant — so the tolerance is a relative bound on the mean
/// NLL and scales with the model, not an absolute constant borrowed from another test.</para>
/// <para><b>A bound is only worth something if it can be exceeded.</b>
/// <see cref="CycledScoring_DetectsAPerturbedBoundaryCheckpoint"/> injects a defect and asserts these
/// bounds catch it — required by <c>CLAUDE.md</c> (issue #418), and load-bearing here: the first
/// draft of this file scored random token ids, where the mean NLL sat at 12.18 against a uniform
/// floor of <c>ln(128256) ~ 11.76</c> and could not move far no matter what broke, so a 1% bound was
/// satisfiable by an almost arbitrarily damaged implementation. Real text drops the figure to ~3.0
/// and the bounds are correspondingly tighter.</para>
/// <para><b>The self-consistency check is the sharper instrument.</b> Cycling with one layer per
/// window versus a single whole-trunk window changes ONLY the number of boundary round trips, with
/// the same head and the same kernels on both sides. A logic error in the windowing — a mis-indexed
/// layer, a dropped residual, a KV or state leak across a cut — moves that comparison a long way,
/// while FP16 boundary rounding moves it very little.</para>
/// </remarks>
public sealed class CudaLayerCyclingPerplexityTests(ITestOutputHelper output)
{
    /// <summary>Short corpus: the whole-device reference path is O(n^2) in forward passes.</summary>
    private const int CorpusTokens = 48;

    /// <summary>
    /// Cycling the GPU through the trunk in windows must reproduce the whole-device figure.
    /// </summary>
    [SkippableFact]
    public void CycledScoring_MatchesWholeDeviceScoring()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null, "Small quantized GGUF fixture not found; see DOTLLM_CUDA_CYCLE_GGUF.");

        using GgufFile gguf = GgufFile.Open(ggufPath!);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tokens = RealTextTokens(gguf, CorpusTokens);
        var options = new PerplexityOptions(
            PerplexityMode.TeacherForced, ContextLength: CorpusTokens, Stride: CorpusTokens);

        double wholeDevice;
        using (GgufFile deviceGguf = GgufFile.Open(ggufPath!))
        {
            ModelConfig deviceConfig = GgufModelConfigExtractor.Extract(deviceGguf.Metadata);
            (IModel model, _) = CudaModelLoader.CreateFromGguf(deviceGguf, deviceConfig, deviceId: 0);
            using (model)
            {
                var adapter = new BackendPerplexityModel(
                    model, deviceId: 0, BackendPerplexityModel.Probe(model, deviceId: 0));
                wholeDevice = PerplexityEvaluator.Evaluate(adapter, tokens, options)
                    .MeanNegativeLogLikelihood;
            }
        }

        double cycled = ScoreCycled(gguf, config, tokens, options, windowSize: Math.Max(1, config.NumLayers / 4));

        output.WriteLine($"whole-device mean NLL = {wholeDevice:F6}");
        output.WriteLine($"cycled       mean NLL = {cycled:F6}");
        double relative = Math.Abs(cycled - wholeDevice) / Math.Abs(wholeDevice);
        output.WriteLine($"relative difference   = {relative:P4}");

        // 0.3% of the mean NLL. The two paths differ by an FP16 boundary round trip per layer cut
        // and by an FP32-host versus FP16-device output head; measured, that is 0.036%, so this is
        // ~8x headroom rather than the order of magnitude the first draft of this test allowed. The
        // looser bound was a hangover from scoring random token ids, where the figure sat near the
        // uniform floor and could not move far no matter what broke.
        Assert.True(relative < 0.003,
            $"cycled mean NLL {cycled:F6} differs from whole-device {wholeDevice:F6} by {relative:P4}, " +
            "which is more than FP16 boundary rounding accounts for.");
    }

    /// <summary>
    /// Cycling with the maximum number of layer cuts must agree with cycling with none.
    /// </summary>
    /// <remarks>
    /// Both sides use the same host output head and the same device kernels, so the only difference
    /// is how many times the residual stream round-trips through host FP32. That isolates the
    /// windowing logic from every other source of difference, which the whole-device comparison
    /// above cannot.
    /// </remarks>
    [SkippableFact]
    public void CycledScoring_IsInsensitiveToTheNumberOfLayerCuts()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null, "Small quantized GGUF fixture not found; see DOTLLM_CUDA_CYCLE_GGUF.");

        using GgufFile gguf = GgufFile.Open(ggufPath!);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tokens = RealTextTokens(gguf, CorpusTokens);
        var options = new PerplexityOptions(
            PerplexityMode.SlidingWindow, ContextLength: CorpusTokens / 2, Stride: CorpusTokens / 4);

        double oneWindow = ScoreCycled(gguf, config, tokens, options, windowSize: config.NumLayers);
        double perLayer = ScoreCycled(gguf, config, tokens, options, windowSize: 1);

        output.WriteLine($"single window ({config.NumLayers} layers) mean NLL = {oneWindow:F6}");
        output.WriteLine($"one layer per window                     mean NLL = {perLayer:F6}");
        double relative = Math.Abs(perLayer - oneWindow) / Math.Abs(oneWindow);
        output.WriteLine($"relative difference                      = {relative:P4}");

        // 0.2%; measured 0.054%. See the tightening note on the equivalence test above.
        Assert.True(relative < 0.002,
            $"cutting the trunk into {config.NumLayers} windows moved the mean NLL from {oneWindow:F6} " +
            $"to {perLayer:F6} ({relative:P4}); only FP16 boundary rounding should differ.");
    }

    /// <summary>
    /// Demonstrates that the bounds above can actually fail: corrupting the checkpointed boundary
    /// hidden state must push the figure outside them.
    /// </summary>
    /// <remarks>
    /// <para>A tolerance nobody has ever seen exceeded is not evidence of correctness — it may simply
    /// be looser than any defect the test can produce. Required by <c>CLAUDE.md</c>'s "demonstrate the
    /// test can fail" rule (issue #418).</para>
    /// <para><b>The calibration, stated plainly, because it bounds what these tests can claim.</b>
    /// Measured on Llama-3.2-1B-Q8_0: FP16 rounding alone moves the mean NLL 0.054%; a 1%
    /// element-wise boundary perturbation moves it 0.123%; a 5% one moves it 0.632%. So the 0.2%
    /// bound resolves a boundary corruption of roughly 2% or more and does <em>not</em> resolve a 1%
    /// one — a sub-percent boundary defect would hide inside FP16 rounding and these tests would pass.
    /// That is a real limit, not a reason to loosen the assertion: the logic load is carried by the
    /// CPU tests, which are bit-identical and therefore catch a corruption of any size, and by
    /// <see cref="CycledScoring_IsInsensitiveToTheNumberOfLayerCuts"/>, which multiplies whatever
    /// error a single cut introduces by the number of cuts.</para>
    /// <para>The earlier 1% magnitude is kept in the record deliberately: it is what showed that a
    /// <em>uniform</em> scale is invisible here (0.043%, less than plain rounding) because the next
    /// layer's RMSNorm is scale-invariant — see <see cref="PerturbedBoundaryModel"/>.</para>
    /// </remarks>
    [SkippableFact]
    public void CycledScoring_DetectsAPerturbedBoundaryCheckpoint()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ggufPath = ResolveModelPath();
        Skip.If(ggufPath is null, "Small quantized GGUF fixture not found; see DOTLLM_CUDA_CYCLE_GGUF.");

        using GgufFile gguf = GgufFile.Open(ggufPath!);
        ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] tokens = RealTextTokens(gguf, CorpusTokens);
        var options = new PerplexityOptions(
            PerplexityMode.SlidingWindow, ContextLength: CorpusTokens / 2, Stride: CorpusTokens / 4);

        int half = Math.Max(1, config.NumLayers / 2);
        double clean = ScoreCycled(gguf, config, tokens, options, half);
        double perturbed = ScoreCycled(gguf, config, tokens, options, half, boundaryPerturbation: 0.05f);

        output.WriteLine($"clean     mean NLL = {clean:F6}");
        output.WriteLine($"perturbed mean NLL = {perturbed:F6}");
        double relative = Math.Abs(perturbed - clean) / Math.Abs(clean);
        output.WriteLine($"relative difference = {relative:P4}");

        Assert.True(relative > 0.002,
            $"a 5% element-wise perturbation of the boundary checkpoint moved the mean NLL by only "
            + $"{relative:P4}, which is inside the tolerance the equivalence tests use — so those "
            + "tolerances are not actually discriminating and the corpus or the bounds need tightening.");
    }

    private static double ScoreCycled(
        GgufFile gguf, ModelConfig config, int[] tokens, PerplexityOptions options, int windowSize,
        float boundaryPerturbation = 0.0f)
    {
        // The CPU model supplies the output head only; every transformer layer runs on the GPU.
        using IModel cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var cpuWindows = new CpuLayerWindowModel(cpuModel, config);
        using var cudaWindows = DotLLM.Cuda.Evaluation.CudaLayerWindowModel.LoadFromGguf(
            gguf, config, deviceId: 0, cpuWindows);

        ILayerWindowModel deviceWindows = boundaryPerturbation == 0.0f
            ? cudaWindows
            : new PerturbedBoundaryModel(cudaWindows, boundaryPerturbation);

        var assignments = new List<CompositeLayerWindowModel.LayerAssignment>();
        foreach (LayerWindow w in CyclingPerplexityEvaluator.PartitionLayers(config.NumLayers, windowSize))
            assignments.Add(new CompositeLayerWindowModel.LayerAssignment(w, deviceWindows));

        using var composite = new CompositeLayerWindowModel(assignments, cpuWindows);
        return CyclingPerplexityEvaluator
            .Evaluate(composite, tokens, options, composite.Windows())
            .MeanNegativeLogLikelihood;
    }

    /// <summary>
    /// Perturbs every boundary checkpoint element-wise by <c>+/- magnitude</c>, standing in for "the
    /// next window resumed from something slightly different from what the previous window
    /// produced".
    /// </summary>
    /// <remarks>
    /// <para>The perturbation is applied to the window's OUTPUT, so it disturbs exactly the tensor
    /// that crosses a layer cut and nothing else — the failure a mis-sized copy, a stale buffer or a
    /// dropped residual at the boundary would produce.</para>
    /// <para><b>Element-wise with alternating sign, NOT a uniform scale.</b> A uniform scale of the
    /// residual stream is very nearly invisible here, and for a structural reason rather than a
    /// tolerance one: the next layer's first operation is RMSNorm, which is scale-invariant, so
    /// multiplying the whole checkpoint by 1.01 is close to a no-op by construction. Measured, it
    /// moved the mean NLL by 0.04% — less than the difference between one layer cut and sixteen.
    /// A fault model the architecture cancels proves nothing about the tolerances, so the sign
    /// alternates per element and the perturbation changes the checkpoint's <em>direction</em>,
    /// which RMSNorm cannot undo.</para>
    /// </remarks>
    private sealed class PerturbedBoundaryModel(ILayerWindowModel inner, float magnitude) : ILayerWindowModel
    {
        public int NumLayers => inner.NumLayers;

        public int HiddenSize => inner.HiddenSize;

        public int VocabSize => inner.VocabSize;

        public int MaxContextLength => inner.MaxContextLength;

        public ILayerWindowExecutor CreateWindow(int firstLayer, int layerCount)
            => new PerturbedWindow(inner.CreateWindow(firstLayer, layerCount), magnitude);

        public ITensor ApplyOutputHead(ReadOnlySpan<float> hidden, int seqLen)
            => inner.ApplyOutputHead(hidden, seqLen);

        /// <summary>No-op: the decorator borrows the inner model, which the test disposes.</summary>
        public void Dispose()
        {
        }

        private sealed class PerturbedWindow(ILayerWindowExecutor inner, float magnitude) : ILayerWindowExecutor
        {
            public int FirstLayer => inner.FirstLayer;

            public int LayerCount => inner.LayerCount;

            public void Run(ReadOnlySpan<int> tokenIds, ReadOnlySpan<float> hiddenIn,
                            ReadOnlySpan<int> positions, Span<float> hiddenOut)
            {
                inner.Run(tokenIds, hiddenIn, positions, hiddenOut);

                // Alternating sign, deterministic in the element index: keeps the checkpoint's
                // magnitude essentially unchanged while rotating its direction, which is the part
                // RMSNorm cannot absorb.
                for (int i = 0; i < hiddenOut.Length; i++)
                    hiddenOut[i] *= 1.0f + ((i & 1) == 0 ? magnitude : -magnitude);
            }

            public void ResetState() => inner.ResetState();

            public void Dispose() => inner.Dispose();
        }
    }

    /// <summary>
    /// Ordinary English, tokenized by the model's own tokenizer.
    /// </summary>
    /// <remarks>
    /// <b>Not random ids, deliberately.</b> Random ids score near the uniform floor
    /// (<c>ln(128256) ~ 11.76</c> against a measured 12.18), and a figure already pinned to chance
    /// cannot move far when the hidden state is partially corrupted — a defect would have to scramble
    /// the state almost completely before it escaped a sub-1% bound. That is precisely the
    /// weak-discriminator hazard issue #395 warns about. Real text puts the model well below the
    /// floor, so a partial corruption has somewhere to move the number to.
    /// </remarks>
    private static int[] RealTextTokens(GgufFile gguf, int count)
    {
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        const string corpus =
            "The history of natural language processing began in the nineteen fifties, when researchers "
            + "first attempted to translate text between human languages using simple rule based systems. "
            + "Over the following decades, statistical methods gradually replaced handwritten rules, and the "
            + "introduction of neural networks transformed the field once more. Today, large language models "
            + "are trained on vast collections of text drawn from books, articles, and conversations, learning "
            + "to predict the next word from everything that came before it. This deceptively simple objective "
            + "turns out to capture a surprising amount of structure about grammar, facts, and reasoning.";
        int[] ids = tokenizer.Encode(corpus);
        Assert.True(ids.Length >= count,
            $"corpus tokenized to {ids.Length} ids but {count} are needed.");
        return ids[..count];
    }

    private static string? ResolveModelPath()
    {
        string? envPath = Environment.GetEnvironmentVariable("DOTLLM_CUDA_CYCLE_GGUF");
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string relative = Path.Combine(
            "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        string[] roots =
        [
            Path.Combine(home, ".dotllm", "models"),
            Path.Combine(home, ".dotllm", "test-cache"),
        ];

        foreach (string root in roots)
        {
            string candidate = Path.Combine(root, relative);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }
}
