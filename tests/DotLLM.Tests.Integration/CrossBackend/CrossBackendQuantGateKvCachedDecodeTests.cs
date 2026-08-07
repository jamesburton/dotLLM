using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.CrossBackend;

/// <summary>
/// Issue #279: <see cref="CrossBackendQuantGateTests"/>'s decode leg
/// (<c>ScoreDecodeAgreementAsync</c>) advances a growing token list through the CACHELESS
/// 3-arg <c>IModel.Forward</c> overload (<c>kvCache: null</c>). Because the prompt itself is
/// already multiple tokens, every mid-layer Q/K/V/O/Gate/Up/Down projection in that leg stays
/// on the batched HGEMM path (<c>seqLen &gt; 1</c>) for CUDA — it never reaches
/// <see cref="CudaTransformerModel.ForwardDecodeGraph"/> /
/// <see cref="CudaTransformerModel"/>'s single-token <c>Project()</c> branch that issue #279
/// actually reports on (the "cached <c>seqLen == 1</c> (GEMV)" leg). Confirmed empirically:
/// <c>Backend_AgreesWithCpu(IQ3_S, Cuda)</c> passes identically before and after the #279
/// candidate fix (mean 1-cos ~1.25e-4 both times) — that harness cannot discriminate this bug.
///
/// This class instead does REAL incremental single-token decode: one prefill call to populate
/// a <see cref="CudaTransformerModel.CreateKvCache"/>-allocated cache, then repeated
/// single-token <c>Forward(tokBuf, posBuf, deviceId, kvCache)</c> calls (the same 4-arg
/// overload / KV-cache pattern <c>CudaGraphCaptureEquivalenceTest</c> uses), which is what
/// actually engages <c>ForwardDecodeGraph</c>/<c>CaptureDecodeGraph</c> and therefore the
/// real per-type decode GEMV dispatch (the generic dequant+cuBLAS-GEMV fallback for IQ3_S/
/// IQ1_S — types with no dedicated decode kernel, per <c>CudaKernels.HasMmq</c>/
/// <c>HasQuantizedGemvKernel</c>). CPU has no <see cref="DotLLM.Core.Attention.IKvCache"/>
/// implementation (confirmed: zero real hits for IKvCache under <c>src/DotLLM.Cpu/</c>), so the
/// CPU oracle side re-runs the ordinary cacheless full-context forward each step — the
/// asymmetry is inherent to how CPU decode works today, not a shortcut taken here.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CrossBackendQuantGateKvCachedDecodeTests
{
    private readonly ITestOutputHelper _output;

    public CrossBackendQuantGateKvCachedDecodeTests(ITestOutputHelper output) => _output = output;

    private const string Prompt = "The capital of France is";
    private const int DecodeSteps = 8;

    /// <summary>
    /// Bound on <c>(1 - cosine_similarity)</c> per real single-token KV-cached decode step,
    /// CPU vs CUDA. Set at the same order of magnitude as the healthy baseline documented in
    /// <c>CrossBackendQuantGateTests.OneMinusCosineTolerance</c>'s remarks (1e-3-scale "inside
    /// the healthy continuum" for #276's IQ1_S), well below the issue's reported defect
    /// magnitude (1 - 0.982404 = 0.0176 for CUDA's cached step 1).
    /// </summary>
    private const double PerStepOneMinusCosineTolerance = 0.01;

    public static IEnumerable<object[]> SpotCheckTypes()
    {
        // IQ3_S: the issue's own type. IQ1_S: same "no dedicated decode GEMV/MMQ kernel, falls
        // to the generic dequant+cuBLAS-GEMV fallback" bucket per CudaKernels.HasMmq /
        // HasQuantizedGemvKernel - spot-checking whether the #279 hypothesis (G1's FP16-accumulate
        // wrongly applied to that fallback's m=1 shape) is IQ3_S-specific or systemic.
        yield return new object[] { QuantizationType.IQ3_S };
        yield return new object[] { QuantizationType.IQ1_S };
    }

    /// <summary>
    /// Four distinct prompts, matching the issue's own framing ("CUDA's other three prompts are
    /// 0.99983-0.99989") that only ONE specific prompt showed the reported 0.982404 divergence -
    /// a single fixed prompt could plausibly miss the defect entirely.
    /// </summary>
    public static IEnumerable<object[]> MultiPromptCases()
    {
        yield return new object[] { "The capital of France is" };
        yield return new object[] { "In 1969, the first humans landed on the Moon during the Apollo 11 mission, which was commanded by" };
        yield return new object[] { "def fibonacci(n):\n    if n <= 1:\n        return n\n    return" };
        yield return new object[] { "The quick brown fox jumps over the lazy dog. Water boils at a temperature of" };
    }

    [SkippableTheory]
    [MemberData(nameof(MultiPromptCases))]
    public async Task KvCachedDecode_CudaAgreesWithCpu_Iq3S_MultiPrompt(string prompt)
        => await RunCudaKvCachedDecode(QuantizationType.IQ3_S, prompt);

    [SkippableTheory]
    [MemberData(nameof(SpotCheckTypes))]
    public async Task KvCachedDecode_CudaAgreesWithCpu(QuantizationType quantType)
        => await RunCudaKvCachedDecode(quantType, Prompt);

    private async Task RunCudaKvCachedDecode(QuantizationType quantType, string prompt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string? path = CrossBackendQuantGateTests.ResolveFixturePath(quantType);
        Skip.If(path is null, $"{quantType}: fixture not found (see CrossBackendQuantGateTests.FixtureHint).");

        using var cpuGguf = GgufFile.Open(path!);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(cpuGguf, cpuConfig, ThreadingConfig.Auto);

        using var gpuGguf = GgufFile.Open(path!);
        var gpuConfig = GgufModelConfigExtractor.Extract(gpuGguf.Metadata);
        using var cudaModel = CudaTransformerModel.LoadFromGguf(gpuGguf, gpuConfig);

        int[] promptIds = tokenizer.Encode(prompt);
        _output.WriteLine($"[{quantType}] prompt tokens: {promptIds.Length}");

        int kvCap = promptIds.Length + DecodeSteps + 8;
        using var kvCache = cudaModel.CreateKvCache(kvCap);

        // Prefill: single multi-token CUDA forward populates the KV cache (HGEMM path, already
        // known-good per the issue). Not scored here - #279 is specifically about what happens
        // AFTER this, on the first real single-token decode call.
        int[] prefillPositions = new int[promptIds.Length];
        for (int i = 0; i < promptIds.Length; i++) prefillPositions[i] = i;
        using (var _ = cudaModel.Forward(promptIds, prefillPositions, 0, kvCache)) { }

        // CPU oracle: teacher-forced full recompute at each step (CPU has no IKvCache - see
        // class remarks). Ground truth continuation is CPU's own greedy argmax, so both sides
        // decode the identical token stream and never fork onto unrelated positions.
        var tokens = new List<int>(promptIds);
        int[] cpuPositions0 = new int[tokens.Count];
        for (int i = 0; i < cpuPositions0.Length; i++) cpuPositions0[i] = i;
        float[] cpuLogits0 = LastRowLogits(cpuModel, tokens.ToArray(), cpuPositions0, cpuConfig.VocabSize);
        int curTok = Argmax(cpuLogits0);
        tokens.Add(curTok);

        double sumOneMinusCos = 0;
        int argmaxMismatches = 0;
        double worstOneMinusCos = 0;
        int worstStep = -1;

        for (int step = 0; step < DecodeSteps; step++)
        {
            // === Real single-token KV-cached CUDA decode - the #279 leg. ===
            int[] tokBuf = { curTok };
            int[] posBuf = { promptIds.Length + step };
            float[] cudaLogits;
            using (var t = cudaModel.Forward(tokBuf, posBuf, 0, kvCache))
            {
                unsafe
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, cpuConfig.VocabSize);
                    cudaLogits = span.ToArray();
                }
            }

            // === CPU oracle: cacheless full recompute over the identical token stream so far. ===
            int[] cpuPositions = new int[tokens.Count];
            for (int i = 0; i < cpuPositions.Length; i++) cpuPositions[i] = i;
            float[] cpuLogits = LastRowLogits(cpuModel, tokens.ToArray(), cpuPositions, cpuConfig.VocabSize);

            double cos = CosineSimilarity(cpuLogits, cudaLogits);
            double oneMinusCos = 1.0 - cos;
            sumOneMinusCos += oneMinusCos;
            if (oneMinusCos > worstOneMinusCos) { worstOneMinusCos = oneMinusCos; worstStep = step; }

            int cpuArgmax = Argmax(cpuLogits);
            int cudaArgmax = Argmax(cudaLogits);
            bool top1Match = cpuArgmax == cudaArgmax;
            if (!top1Match) argmaxMismatches++;

            _output.WriteLine(
                $"[{quantType}] KV-decode step {step}: cos={cos:F6} (1-cos={oneMinusCos:E4}) "
                + $"top1 cpu={cpuArgmax} cuda={cudaArgmax} match={top1Match}");

            int nextTok = cpuArgmax;
            tokens.Add(nextTok);
            curTok = nextTok;
            await Task.Yield();
        }

        double meanOneMinusCos = sumOneMinusCos / DecodeSteps;
        _output.WriteLine(
            $"[{quantType}] KV-decode summary: mean(1-cos)={meanOneMinusCos:E4}, "
            + $"worst(1-cos)={worstOneMinusCos:E4} at step {worstStep}, "
            + $"top1 mismatches={argmaxMismatches}/{DecodeSteps} (bound {PerStepOneMinusCosineTolerance})");

        Assert.True(worstOneMinusCos <= PerStepOneMinusCosineTolerance,
            $"[{quantType}] real KV-cached decode step {worstStep} diverged from CPU: "
            + $"1-cos={worstOneMinusCos:E4} exceeds bound {PerStepOneMinusCosineTolerance}. "
            + "This is the real single-token seqLen==1 GEMV decode path (issue #279) - "
            + "CrossBackendQuantGateTests.Backend_AgreesWithCpu's cacheless growing-prefix leg "
            + "cannot catch this because the prompt is never length 1.");
    }

    /// <summary>
    /// Vulkan counterpart of <see cref="KvCachedDecode_CudaAgreesWithCpu"/>, same real
    /// single-token KV-cached decode methodology (prefill once via
    /// <c>VulkanModelLoader.CreateFromGguf</c>'s <c>KvCacheFactory</c>, then repeated
    /// single-token <c>Forward(tokBuf, posBuf, deviceId, kvCache)</c> calls) — for a fair
    /// comparison against the CUDA numbers above, since the previous Vulkan pass
    /// (<c>CrossBackendQuantGateTests.Backend_AgreesWithCpu(IQ3_S, Vulkan)</c> at 1.31e-4) used
    /// the same non-discriminating cacheless growing-prefix leg the CUDA class remarks describe.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SpotCheckTypes))]
    public void KvCachedDecode_VulkanAgreesWithCpu(QuantizationType quantType)
    {
        bool vulkanOk;
        try { using var probe = DotLLM.Vulkan.VulkanDevice.Create(); vulkanOk = true; }
        catch { vulkanOk = false; }
        Skip.IfNot(vulkanOk, "Vulkan runtime not available on this host.");

        string? spvDir = ResolveSpvDir();
        Skip.If(spvDir is null, "Vulkan SPIR-V directory not found.");

        string? path = CrossBackendQuantGateTests.ResolveFixturePath(quantType);
        Skip.If(path is null, $"{quantType}: fixture not found (see CrossBackendQuantGateTests.FixtureHint).");

        using var cpuGguf = GgufFile.Open(path!);
        var cpuConfig = GgufModelConfigExtractor.Extract(cpuGguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(cpuGguf.Metadata);
        using var cpuModel = ModelLoader.CreateCpuModelFromGguf(cpuGguf, cpuConfig, ThreadingConfig.Auto);

        using var device = DotLLM.Vulkan.VulkanDevice.Create();
        using var gpuGguf = GgufFile.Open(path!);
        var gpuConfig = GgufModelConfigExtractor.Extract(gpuGguf.Metadata);
        var (vulkanModel, kvCacheFactory) = DotLLM.Vulkan.VulkanModelLoader.CreateFromGguf(
            device, gpuGguf, gpuConfig, spvDir!);
        using var vulkanModelDisposable = vulkanModel;

        int[] promptIds = tokenizer.Encode(Prompt);
        _output.WriteLine($"[{quantType}/Vulkan] prompt tokens: {promptIds.Length}");

        int kvCap = promptIds.Length + DecodeSteps + 8;
        var kvCache = kvCacheFactory(kvCap);
        using var kvCacheDisposable = kvCache as IDisposable;

        int[] prefillPositions = new int[promptIds.Length];
        for (int i = 0; i < promptIds.Length; i++) prefillPositions[i] = i;
        using (var _ = vulkanModel.Forward(promptIds, prefillPositions, 0, kvCache)) { }

        var tokens = new List<int>(promptIds);
        int[] cpuPositions0 = new int[tokens.Count];
        for (int i = 0; i < cpuPositions0.Length; i++) cpuPositions0[i] = i;
        float[] cpuLogits0 = LastRowLogits(cpuModel, tokens.ToArray(), cpuPositions0, cpuConfig.VocabSize);
        int curTok = Argmax(cpuLogits0);
        tokens.Add(curTok);

        double sumOneMinusCos = 0;
        int argmaxMismatches = 0;
        double worstOneMinusCos = 0;
        int worstStep = -1;

        for (int step = 0; step < DecodeSteps; step++)
        {
            int[] tokBuf = { curTok };
            int[] posBuf = { promptIds.Length + step };
            float[] vulkanLogits;
            using (var t = vulkanModel.Forward(tokBuf, posBuf, 0, kvCache))
            {
                unsafe
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, cpuConfig.VocabSize);
                    vulkanLogits = span.ToArray();
                }
            }

            int[] cpuPositions = new int[tokens.Count];
            for (int i = 0; i < cpuPositions.Length; i++) cpuPositions[i] = i;
            float[] cpuLogits = LastRowLogits(cpuModel, tokens.ToArray(), cpuPositions, cpuConfig.VocabSize);

            double cos = CosineSimilarity(cpuLogits, vulkanLogits);
            double oneMinusCos = 1.0 - cos;
            sumOneMinusCos += oneMinusCos;
            if (oneMinusCos > worstOneMinusCos) { worstOneMinusCos = oneMinusCos; worstStep = step; }

            int cpuArgmax = Argmax(cpuLogits);
            int vulkanArgmax = Argmax(vulkanLogits);
            bool top1Match = cpuArgmax == vulkanArgmax;
            if (!top1Match) argmaxMismatches++;

            _output.WriteLine(
                $"[{quantType}/Vulkan] KV-decode step {step}: cos={cos:F6} (1-cos={oneMinusCos:E4}) "
                + $"top1 cpu={cpuArgmax} vulkan={vulkanArgmax} match={top1Match}");

            int nextTok = cpuArgmax;
            tokens.Add(nextTok);
            curTok = nextTok;
        }

        double meanOneMinusCos = sumOneMinusCos / DecodeSteps;
        _output.WriteLine(
            $"[{quantType}/Vulkan] KV-decode summary: mean(1-cos)={meanOneMinusCos:E4}, "
            + $"worst(1-cos)={worstOneMinusCos:E4} at step {worstStep}, "
            + $"top1 mismatches={argmaxMismatches}/{DecodeSteps} (bound {PerStepOneMinusCosineTolerance})");

        Assert.True(worstOneMinusCos <= PerStepOneMinusCosineTolerance,
            $"[{quantType}/Vulkan] real KV-cached decode step {worstStep} diverged from CPU: "
            + $"1-cos={worstOneMinusCos:E4} exceeds bound {PerStepOneMinusCosineTolerance}.");
    }

    private static string? ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    private static unsafe float[] LastRowLogits(
        IModel model, int[] tokens, int[] positions, int vocab)
    {
        using var logits = model.Forward(tokens, positions, -1);
        int seqLen = logits.Shape.Rank == 2 ? logits.Shape[0] : 1;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        var result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private static double CosineSimilarity(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        if (na == 0 || nb == 0) return 0;
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static int Argmax(float[] xs)
    {
        int best = 0; float bestV = xs[0];
        for (int i = 1; i < xs.Length; i++) if (xs[i] > bestV) { bestV = xs[i]; best = i; }
        return best;
    }
}
