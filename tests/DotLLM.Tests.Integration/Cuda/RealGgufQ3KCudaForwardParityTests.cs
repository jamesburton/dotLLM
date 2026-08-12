using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// End-to-end CPU↔CUDA forward parity on a real Q3_K GGUF: prefill plus teacher-forced
/// decode steps, comparing full logit rows.
/// </summary>
/// <remarks>
/// <para>
/// Kernel-level parity does not establish that the right kernel is wired into the forward
/// pass, and a kernel-level suite cannot see a stale build at all. Both failure modes have
/// bitten this project: Q3_K shipped scrambled for months behind green kernel tests (#311),
/// and a Q3_K parity sweep was then read as a live 14.586 L∞ regression when it was a
/// half-updated Release build — fixed shaders scored against a stale transposed CPU oracle,
/// because `.spv` files load from the repo tree while CPU kernels live in a compiled DLL
/// (#341). An end-to-end test on real bytes is what actually exercises the shipping path.
/// </para>
/// <para>
/// CUDA has no packed Q3_K matmul (pinned by <c>CudaQ3KKernelSurfaceTests</c>): the entire
/// committed PTX tree exposes one Q3_K entry point, <c>dequant_q3_k_f16</c>, and
/// <c>CudaKernels.HasQuantizedGemv</c> excludes Q3_K, so <c>CudaWeights.SkipFp16</c> is
/// false and every Q3_K tensor is decoded once at load into a persistent FP16 copy that
/// type-agnostic cuBLAS then consumes. This test is what turns that reading of the code
/// into a measured fact — it exercises whatever the model actually does.
/// </para>
/// <para>
/// <b>Bounds</b> match the Vulkan real-GGUF parity harness (L∞ ≤ 3.0, top-10 Jaccard ≥ 0.5)
/// so results are directly comparable across backends. They are loose enough for the CPU's
/// FP32 accumulation versus CUDA's FP16 weights, and were the bounds a real Q3_K layout
/// error blew straight past on Vulkan.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class RealGgufQ3KCudaForwardParityTests
{
    private const float LogitsAbsTol = 3.0f;
    private const int TopKForJaccard = 10;
    private const float TopKJaccardFloor = 0.5f;
    private const int DecodeSteps = 3;

    private readonly ITestOutputHelper _output;

    public RealGgufQ3KCudaForwardParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Q3K_RealGguf_CudaForward_MatchesCpuForward()
    {
        string? path = ResolveQ3KFixture();
        Skip.If(path is null,
            "No Q3_K GGUF fixture. Set DOTLLM_QUANT_FIXTURE_Q3_K, or generate the quant ladder "
            + "per docs/QUANT_FIXTURES.md into ~/.dotllm/quant-ladder/Llama-3.2-1B-pure/.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using GgufFile gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        _output.WriteLine(
            $"{Path.GetFileName(path)}: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H "
            + $"vocab={config.VocabSize}");

        // A Q3_K file is a *mixture*; assert the type under test is actually present rather
        // than trusting the filename (mradermacher's SmolLM "i1-Q3_K_M" contains no Q3_K at all).
        int q3kTensors = gguf.Tensors.Count(t =>
            t.QuantizationType == DotLLM.Core.Configuration.QuantizationType.Q3_K);
        Assert.True(q3kTensors > 0,
            $"'{Path.GetFileName(path)}' contains no Q3_K tensor — this fixture cannot establish "
            + "anything about the Q3_K path.");
        _output.WriteLine($"Q3_K tensors: {q3kTensors}");

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] promptTokens = tokenizer.Encode("The capital of France is");
        Assert.True(promptTokens.Length >= 2, "Prompt tokenised to fewer than 2 tokens.");

        int kvCapacity = promptTokens.Length + DecodeSteps + 4;

        var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        using var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, kvCapacity);
        var cudaModel = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!);
        using var cudaKv = cudaModel.CreateKvCache(kvCapacity);

        try
        {
            int[] positions = new int[promptTokens.Length];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;

            float[] cpuLogits = ForwardLastRow(cpuModel, promptTokens, positions, config.VocabSize, -1, cpuKv);
            float[] cudaLogits = ForwardLastRow(cudaModel, promptTokens, positions, config.VocabSize, 0, cudaKv);
            AssertLogitsMatch(cpuLogits, cudaLogits, step: 0, "prefill");

            // Teacher-forced decode: feed BOTH models the CPU's token so a single
            // divergence cannot cascade into an incomparable continuation.
            int nextPos = promptTokens.Length;
            for (int step = 1; step <= DecodeSteps; step++)
            {
                int forced = ArgMax(cpuLogits);
                cpuLogits = ForwardLastRow(cpuModel, [forced], [nextPos], config.VocabSize, -1, cpuKv);
                cudaLogits = ForwardLastRow(cudaModel, [forced], [nextPos], config.VocabSize, 0, cudaKv);
                AssertLogitsMatch(cpuLogits, cudaLogits, step, "decode");
                nextPos++;
            }
        }
        finally
        {
            cudaModel.Dispose();
            cpuModel.Dispose();
        }
    }

    private static unsafe float[] ForwardLastRow(
        IModel model, int[] tokenIds, int[] positions, int vocab, int deviceId, IKvCache kvCache)
    {
        using ITensor logits = model.Forward(tokenIds, positions, deviceId, kvCache);
        Assert.Equal(2, logits.Shape.Rank);
        int seqLen = logits.Shape[0];
        Assert.Equal(vocab, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, checked(seqLen * vocab));
        float[] result = new float[vocab];
        span.Slice((seqLen - 1) * vocab, vocab).CopyTo(result);
        return result;
    }

    private void AssertLogitsMatch(float[] cpu, float[] cuda, int step, string label)
    {
        Assert.Equal(cpu.Length, cuda.Length);

        float maxAbs = 0f;
        int worstIdx = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            float diff = Math.Abs(cpu[i] - cuda[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }

        float jaccard = TopKJaccard(cpu, cuda, TopKForJaccard);
        int cpuTop1 = ArgMax(cpu), cudaTop1 = ArgMax(cuda);

        _output.WriteLine(
            $"[{label}] step {step}: L∞={maxAbs:F4} (idx {worstIdx}); "
            + $"top-{TopKForJaccard} jaccard={jaccard:F2}; argmax cpu={cpuTop1} cuda={cudaTop1}");

        Assert.True(maxAbs <= LogitsAbsTol,
            $"[{label}] step {step}: L∞ {maxAbs:F4} exceeds {LogitsAbsTol:F2}. A Q3_K weight-layout "
            + "disagreement lands orders of magnitude above this bound, not just outside it.");
        Assert.True(jaccard >= TopKJaccardFloor,
            $"[{label}] step {step}: top-{TopKForJaccard} jaccard {jaccard:F2} below floor {TopKJaccardFloor:F2}.");
        Assert.True(cpuTop1 == cudaTop1,
            $"[{label}] step {step}: argmax disagrees (cpu={cpuTop1}, cuda={cudaTop1}).");
    }

    private static float TopKJaccard(float[] a, float[] b, int k)
    {
        HashSet<int> aTop = [.. TopKIndices(a, k)];
        HashSet<int> bTop = [.. TopKIndices(b, k)];
        int intersection = aTop.Count(bTop.Contains);
        int union = aTop.Count + bTop.Count - intersection;
        return union == 0 ? 1f : (float)intersection / union;
    }

    private static int[] TopKIndices(float[] values, int k)
        => Enumerable.Range(0, values.Length)
            .OrderByDescending(i => values[i])
            .Take(k)
            .ToArray();

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>
    /// Env override <c>DOTLLM_QUANT_FIXTURE_Q3_K</c> first, else the conventional quant-ladder
    /// path from <c>docs/QUANT_FIXTURES.md</c>. Returns null so the caller self-skips.
    /// </summary>
    private static string? ResolveQ3KFixture()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_QUANT_FIXTURE_Q3_K");
        if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string conventional = Path.Combine(
            home, ".dotllm", "quant-ladder", "Llama-3.2-1B-pure", "Llama-3.2-1B-pure-Q3_K.gguf");
        return File.Exists(conventional) ? conventional : null;
    }

    private static string? FindPtxDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        ];
        foreach (string dir in candidates)
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
