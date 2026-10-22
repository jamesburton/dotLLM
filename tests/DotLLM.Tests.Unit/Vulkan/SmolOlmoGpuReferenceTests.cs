using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// GPU <c>smollm3</c> / <c>olmo2</c> / <c>olmoe</c> (GGUF, #765) against the independent double-precision
/// <see cref="GraniteGgufReference"/>. The reference (not the CPU backend) is the oracle for OLMoE: the CPU routed-MoE
/// path uses Q8-quantized activations. Each backend must (a) match the reference and (b) be farther than the ablation
/// threshold from the reference with the arch-specific feature switched OFF — i.e. NoPE layers, the post-norm-only layout,
/// the whole-projection Q/K norm and OLMoE's non-renormalised gating are really implemented, not silently dropped.
/// CUDA arms skip on hosts without an NVIDIA device (run on the T5500).
/// </summary>
[Trait("Category", "GPU")]
public sealed class SmolOlmoGpuReferenceTests
{
    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3];

    public static IEnumerable<object[]> Fixtures()
    {
        yield return new object[] { "smollm3", new SyntheticGraniteConfig { Arch = "smollm3", Layers = 8, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0 } };
        yield return new object[] { "olmo2", new SyntheticGraniteConfig { Arch = "olmo2", Layers = 4, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0 } };
        yield return new object[] { "olmoe", new SyntheticGraniteConfig { Arch = "olmoe", Layers = 4, ExpertsQ8_0 = true, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0 } };
    }

    private static unsafe float[] Last(ITensor t)
    {
        int rows = t.Shape[0], v = t.Shape[1];
        var r = new float[v];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * v, v).CopyTo(r);
        return r;
    }

    private static double Max(float[] a, double[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    private static List<(string Name, GraniteGgufReference.Options Opt)> Ablations(string arch)
    {
        var a = new List<(string, GraniteGgufReference.Options)>();
        if (arch == "smollm3") a.Add(("NoPE layers", new() { NoPe = false }));
        if (arch == "olmo2") a.Add(("post-norm-only layout", new() { Olmo2Layout = false }));
        if (arch is "olmo2" or "olmoe")
        {
            a.Add(("Q/K norm", new() { QkNorm = false }));
            a.Add(("whole-projection Q/K norm", new() { QkNormWhole = false }));
        }
        if (arch == "olmoe") a.Add(("no top-k renormalisation", new() { OlmoeNoRenorm = false }));
        return a;
    }

    private static void CheckAgainstReference(string who, string arch, SyntheticGraniteGguf.Weights wts, IModel model,
        Func<int, IKvCache> makeCache, double tol)
    {
        double[] reference = GraniteGgufReference.LastTokenLogits(wts, Prompt);
        float[] gpu;
        using (ITensor l = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), -1, kvCache: null))
            gpu = Last(l);
        // abs + 1% of the logit range: 8 layers of F32 reduction-order drift is ~0.7% on this fixture (the ablated references
        // sit 5+ logits away, see the sensitivity checks below).
        tol += 0.01 * reference.Max(Math.Abs);
        double d = Max(gpu, reference);
        Assert.True(d < tol, $"{who} {arch}: cacheless vs reference = {d:E3} (tol {tol:E3})");

        int split = Prompt.Length - 1;
        using (var cache = makeCache(Prompt.Length + 1))
        {
            using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, cache)) { }
            using ITensor l = model.Forward([Prompt[split]], [split], -1, cache);
            double dd = Max(Last(l), reference);
            Assert.True(dd < tol, $"{who} {arch}: cached decode vs reference = {dd:E3}");
        }

        foreach (var (name, opt) in Ablations(arch))
        {
            double far = Max(gpu, GraniteGgufReference.LastTokenLogits(wts, Prompt, opt));
            Assert.True(far > 0.05, $"{who} {arch}: result is within {far:E3} of the reference WITHOUT {name} (not sensitive enough).");
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Fixtures))]
    public void Vulkan_MatchesIndependentReference(string arch, SyntheticGraniteConfig cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        var wts = SyntheticGraniteGguf.BuildWeights(cfg);
        string path = Path.Combine(Path.GetTempPath(), $"smololmo_vk_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, SpvDir());
            CheckAgainstReference("Vulkan", arch, wts, model, n => model.CreateKvCache(n), 0.03);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Fixtures))]
    public void Cuda_MatchesIndependentReference(string arch, SyntheticGraniteConfig cfg)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA device/driver on this host.");

        var wts = SyntheticGraniteGguf.BuildWeights(cfg);
        string path = Path.Combine(Path.GetTempPath(), $"smololmo_cuda_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        try
        {
            var (model, gguf, _) = CudaModelLoader.LoadFromGguf(path, deviceId: 0);
            using (gguf)
            using (model)
                CheckAgainstReference("CUDA", arch, wts, model, n => model.CreateKvCache(n), 0.06);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }
}
