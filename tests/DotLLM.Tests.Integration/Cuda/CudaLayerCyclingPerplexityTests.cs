using DotLLM.Core.Evaluation;
using DotLLM.Core.Models;
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
/// <para><b>Why the tolerance is not tight.</b> Unlike the CPU cycled-vs-whole test — which is an
/// identity, because both sides run the same kernels in the same order — this comparison changes two
/// things at once. The residual stream makes an FP16 round trip at every layer cut (the boundary's
/// separate add + RMSNorm replaces an in-window fused add-RMSNorm), and the cycled path applies the
/// output head on the host in FP32 where the whole-device path applies it on the GPU in FP16. Both
/// are rounding, not logic, but they are real, and the FP16 error accumulates over the reduction
/// length rather than staying constant — so the tolerance is stated as a relative bound on the mean
/// NLL and scales with the model, not as an absolute constant borrowed from another test.</para>
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
        int[] tokens = SyntheticTokens(config.VocabSize, CorpusTokens);
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

        // 1% of the mean NLL. The two paths differ by an FP16 boundary round trip per layer cut and
        // by an FP32-host versus FP16-device output head; a logic error moves this by far more.
        Assert.True(relative < 0.01,
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
        int[] tokens = SyntheticTokens(config.VocabSize, CorpusTokens);
        var options = new PerplexityOptions(
            PerplexityMode.SlidingWindow, ContextLength: CorpusTokens / 2, Stride: CorpusTokens / 4);

        double oneWindow = ScoreCycled(gguf, config, tokens, options, windowSize: config.NumLayers);
        double perLayer = ScoreCycled(gguf, config, tokens, options, windowSize: 1);

        output.WriteLine($"single window ({config.NumLayers} layers) mean NLL = {oneWindow:F6}");
        output.WriteLine($"one layer per window                     mean NLL = {perLayer:F6}");
        double relative = Math.Abs(perLayer - oneWindow) / Math.Abs(oneWindow);
        output.WriteLine($"relative difference                      = {relative:P4}");

        Assert.True(relative < 0.005,
            $"cutting the trunk into {config.NumLayers} windows moved the mean NLL from {oneWindow:F6} " +
            $"to {perLayer:F6} ({relative:P4}); only FP16 boundary rounding should differ.");
    }

    private static double ScoreCycled(
        GgufFile gguf, ModelConfig config, int[] tokens, PerplexityOptions options, int windowSize)
    {
        // The CPU model supplies the output head only; every transformer layer runs on the GPU.
        using IModel cpuModel = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var cpuWindows = new CpuLayerWindowModel(cpuModel, config);
        using var cudaWindows = DotLLM.Cuda.Evaluation.CudaLayerWindowModel.LoadFromGguf(
            gguf, config, deviceId: 0, cpuWindows);

        var assignments = new List<CompositeLayerWindowModel.LayerAssignment>();
        foreach (LayerWindow w in CyclingPerplexityEvaluator.PartitionLayers(config.NumLayers, windowSize))
            assignments.Add(new CompositeLayerWindowModel.LayerAssignment(w, cudaWindows));

        using var composite = new CompositeLayerWindowModel(assignments, cpuWindows);
        return CyclingPerplexityEvaluator
            .Evaluate(composite, tokens, options, composite.Windows())
            .MeanNegativeLogLikelihood;
    }

    /// <summary>
    /// Deterministic pseudo-random ids. The comparison is between two execution paths over identical
    /// inputs, so the corpus only has to be in-vocabulary and reproducible, not meaningful text.
    /// </summary>
    private static int[] SyntheticTokens(int vocabSize, int count)
    {
        var tokens = new int[count];
        uint state = 0x5EEDu;
        for (int i = 0; i < count; i++)
        {
            state = (state * 1664525u) + 1013904223u;
            tokens[i] = (int)(state % (uint)Math.Min(vocabSize, 30000));
        }
        return tokens;
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
