using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1149: the hybrid-dense Vulkan host's prefill projections take the dense host's dp4a MMQ (Q8_1 activation rows) where
/// <see cref="VulkanDensePrefillGemmPolicy"/> says MMQ: K-quants below the coopmat crossover, IQ4_NL at every row count. Real Tev1-4B
/// tensors (Q4_K_M from the model store; the IQ4_NL requant from the legacy test cache), CPU host as the oracle.
/// </summary>
/// <remarks>
/// The "path ran" proof is by perturbation, not by <c>IsSupported</c>: the same prefill is run with <c>DOTLLM_VK_HYBRID_PREFILL_MMQ=0</c>
/// (the pre-#1149 dispatch) and the two logit vectors must differ, the MMQ arm's dispatch counter must be non-zero and the control
/// arm's zero. The parity bound is comparative: the MMQ arm's KL from the CPU oracle must stay within a small factor of the control arm's,
/// not under a fixed number that a weak fixture could never exceed. Skips without the GGUFs or without integer dot products.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDensePrefillMmqTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen3HybridDensePrefillMmqTests(ITestOutputHelper output) => _out = output;

    public const string Q4KM = "q4_k_m";
    public const string Iq4Nl = "iq4_nl";

    private static string? FindGguf(string which) => which switch
    {
        Q4KM => TestFixtureResolver.ResolveFile("DOTLLM_TEV1_4B_GGUF", "bartowski", "togethercomputer_Tev1-4B-experimental-GGUF", "*Q4_K_M.gguf").Path,
        Iq4Nl => TestFixtureResolver.ResolveFile(
            ["DOTLLM_TEV1_4B_IQ4_NL_GGUF"], "bartowski", "togethercomputer_Tev1-4B-experimental-GGUF",
            ["Tev1-4B-IQ4_NL-requant.gguf", "*IQ4_NL*.gguf"],
            [Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "tev1-iq")]).Path,
        _ => throw new ArgumentOutOfRangeException(nameof(which)),
    };

    /// <summary>A deterministic prompt of <paramref name="n"/> real token ids (the forward-batch tests' prompt pattern).</summary>
    private static int[] Ids(int n) => [.. Enumerable.Range(0, n).Select(i => 1000 + i * 37 % 5000)];

    private static T WithEnv<T>(string name, string? value, Func<T> body)
    {
        string? saved = Environment.GetEnvironmentVariable(name);
        Environment.SetEnvironmentVariable(name, value);
        try { return body(); }
        finally { Environment.SetEnvironmentVariable(name, saved); }
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        using (logits)
        {
            int rows = logits.Shape[0];
            return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
        }
    }

    private static int Argmax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>KL(p || q) over softmaxed logits, in nats.</summary>
    private static double Kl(float[] p, float[] q)
    {
        double lsp = LogSumExp(p), lsq = LogSumExp(q), kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lsp;
            kl += Math.Exp(lp) * (lp - (q[i] - lsq));
        }
        return kl;
    }

    private static double LogSumExp(float[] x)
    {
        double m = double.NegativeInfinity;
        foreach (float v in x) m = Math.Max(m, v);
        double s = 0;
        foreach (float v in x) s += Math.Exp(v - m);
        return m + Math.Log(s);
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double w = 0;
        for (int i = 0; i < a.Length; i++) w = Math.Max(w, Math.Abs(a[i] - b[i]));
        return w;
    }

    private sealed record VkRun(float[] Logits, long MmqDispatches, DensePrefillGemmPath Q4K, DensePrefillGemmPath Iq4NlPath, bool IntegerDot, bool Q4KCoopmat);

    private static VkRun RunVulkan(string path, ModelConfig config, int[] ids, string spvDir)
    {
        using var gguf = GgufFile.Open(path);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        float[] row = LastRow(model.Forward(ids, pos, deviceId: -1), config.VocabSize);
        return new VkRun(row, model.MmqPrefillDispatchCount,
            model.PrefillGemmPathFor(QuantizationType.Q4_K, config.HiddenSize, ids.Length),
            model.PrefillGemmPathFor(QuantizationType.IQ4_NL, config.HiddenSize, ids.Length),
            device.HasIntegerDotProduct,
            model.HasCoopmatPrefillFor(QuantizationType.Q4_K, config.HiddenSize));
    }

    private static float[] RunCpu(string path, ModelConfig config, int[] ids)
    {
        using var gguf = GgufFile.Open(path);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config);
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        return LastRow(model.Forward(ids, pos, -1), config.VocabSize);
    }

    /// <summary>
    /// Below the K-quant crossover (48 rows): Q4_K_M's projections and IQ4_NL's both run on the MMQ; the MMQ arm's logits track the CPU
    /// oracle as closely as the control arm's (same greedy token, KL within 3x + 1e-3), and the control arm proves the MMQ arm ran.
    /// </summary>
    [SkippableTheory]
    [InlineData(Q4KM)]
    [InlineData(Iq4Nl)]
    public void Prefill48_TakesMmq_MatchesCpu(string which)
    {
        string? path = FindGguf(which);
        Skip.If(path is null, $"Tev1-4B {which} GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        ModelConfig config;
        using (var gguf = GgufFile.Open(path!)) config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] ids = Ids(48);
        Assert.True(ids.Length < VulkanDensePrefillGemmPolicy.KQuantCoopmatMinTokens, "test rows must sit below the K-quant crossover");

        VkRun mmq = WithEnv(VulkanQwen3MoeHybridKernels.HybridPrefillMmqEnvVar, null, () => RunVulkan(path!, config, ids, spvDir));
        Skip.If(!mmq.IntegerDot, "device has no integer dot products: no MMQ to test.");
        VkRun control = WithEnv(VulkanQwen3MoeHybridKernels.HybridPrefillMmqEnvVar, "0", () => RunVulkan(path!, config, ids, spvDir));
        float[] cpu = RunCpu(path!, config, ids);

        double klMmq = Kl(cpu, mmq.Logits), klControl = Kl(cpu, control.Logits), delta = MaxAbsDiff(mmq.Logits, control.Logits);
        _out.WriteLine($"{which} n={ids.Length}: mmq dispatches={mmq.MmqDispatches} (control {control.MmqDispatches}); " +
                       $"KL(cpu||mmq)={klMmq:E3} KL(cpu||control)={klControl:E3}; max|mmq-control|={delta:E3}; " +
                       $"argmax cpu={Argmax(cpu)} mmq={Argmax(mmq.Logits)} control={Argmax(control.Logits)}; paths q4_k={mmq.Q4K} iq4_nl={mmq.Iq4NlPath}");

        // The path ran: counted, and observable in the output (Q8_1 activations are not bit-identical to F32 ones).
        Assert.True(mmq.MmqDispatches > 0, "no projection took the MMQ prefill below the crossover");
        Assert.Equal(0, control.MmqDispatches);
        Assert.True(delta > 0, "MMQ arm is bit-identical to the control arm: the MMQ path did not run");
        Assert.Equal(DensePrefillGemmPath.Mmq, mmq.Q4K);
        Assert.Equal(DensePrefillGemmPath.Mmq, mmq.Iq4NlPath);

        // Parity: same greedy token as the CPU oracle, and no worse than the control arm beyond the Q8_1 activation floor.
        Assert.Equal(Argmax(cpu), Argmax(mmq.Logits));
        Assert.True(klMmq <= 3 * klControl + 1e-3, $"KL(cpu||mmq)={klMmq:E3} exceeds 3x the control arm's {klControl:E3} + 1e-3");
    }

    /// <summary>
    /// Above the crossover (160 rows) the K-quants return to the coopmat GEMM where the device has one (Q4_K_M: zero MMQ dispatches on
    /// such a device), while IQ4_NL (no coopmat GEMM) stays on the MMQ: the selection is the dense host's table, not "MMQ everywhere".
    /// </summary>
    [SkippableTheory]
    [InlineData(Q4KM)]
    [InlineData(Iq4Nl)]
    public void Prefill160_KQuantsCoopmatWhereAvailable_Iq4NlStaysMmq(string which)
    {
        string? path = FindGguf(which);
        Skip.If(path is null, $"Tev1-4B {which} GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        ModelConfig config;
        using (var gguf = GgufFile.Open(path!)) config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] ids = Ids(160);
        Assert.True(ids.Length >= VulkanDensePrefillGemmPolicy.KQuantCoopmatMinTokens);

        VkRun run = WithEnv(VulkanQwen3MoeHybridKernels.HybridPrefillMmqEnvVar, null, () => RunVulkan(path!, config, ids, spvDir));
        Skip.If(!run.IntegerDot, "device has no integer dot products: no MMQ to test.");
        float[] cpu = RunCpu(path!, config, ids);
        _out.WriteLine($"{which} n={ids.Length}: mmq dispatches={run.MmqDispatches}; KL(cpu||vk)={Kl(cpu, run.Logits):E3}; " +
                       $"paths q4_k={run.Q4K} iq4_nl={run.Iq4NlPath}; q4_k coopmat on device={run.Q4KCoopmat}");

        Assert.Equal(Argmax(cpu), Argmax(run.Logits));
        Assert.Equal(DensePrefillGemmPath.Mmq, run.Iq4NlPath);
        // Q4_K at this row count: coopmat where the device has the kernel, else still MMQ (never the scalar GEMM while an MMQ exists).
        Assert.Equal(run.Q4KCoopmat ? DensePrefillGemmPath.Coopmat : DensePrefillGemmPath.Mmq, run.Q4K);
        if (which == Iq4Nl)
            Assert.True(run.MmqDispatches > 0, "IQ4_NL projections did not take the MMQ above the K-quant crossover");
        else if (run.Q4KCoopmat)
            Assert.Equal(0, run.MmqDispatches);   // every K-quant projection is back on coopmat above the crossover
        else
            Assert.True(run.MmqDispatches > 0, "no coopmat on this device: the K-quants must take the MMQ, not the scalar GEMM");
    }
}
