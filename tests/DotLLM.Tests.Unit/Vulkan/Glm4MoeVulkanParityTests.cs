using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1600: the Vulkan forward pass for <c>glm4moe</c> against the CPU reference (#1599) on the hermetic synthetic GGUF. Skipped when no
/// Vulkan device is present.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Glm4MoeVulkanParityTests : IDisposable
{
    private static readonly int[] Ids = [3, 14, 8, 1, 5, 9, 0, 11, 2, 15, 7, 6];
    private const float Tol = 3e-2f;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-glm4moe-vk-{Guid.NewGuid():N}");

    public Glm4MoeVulkanParityTests() => Directory.CreateDirectory(_scratch);
    public void Dispose() { try { Directory.Delete(_scratch, true); } catch { } }

    private string Write(int experts, bool nextn, int headDim, bool asF32 = false)
    {
        string p = Path.Combine(_scratch, $"m{experts}_{headDim}_{nextn}_{asF32}.gguf");
        File.WriteAllBytes(p, SyntheticGlm4MoeGguf.Build(numExperts: experts, withNextn: nextn, headDim: headDim, q8Weights: true, q8AsF32: asF32));
        return p;
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private static float[] Cpu(GgufFile gguf, ModelConfig cfg)
    {
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        using ITensor t = model.Forward(Ids, Enumerable.Range(0, Ids.Length).ToArray(), deviceId: -1);
        return Copy(t);
    }

    private static float[] Vk(GgufFile gguf, ModelConfig cfg, string spvDir, int[] chunks)
    {
        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.LoadFromGguf(device, gguf, cfg, spvDir);
        using var cache = model.CreateKvCache(64);
        var all = new List<float>();
        int at = 0;
        foreach (int n in chunks)
        {
            using ITensor t = model.Forward(Ids.AsSpan(at, n), Enumerable.Range(at, n).ToArray(), deviceId: -1, cache);
            all.AddRange(Copy(t));
            at += n;
        }
        Assert.Equal(Ids.Length, at);
        return all.ToArray();
    }

    [SkippableTheory]
    [InlineData(new[] { 12 }, 8, 16)]
    [InlineData(new[] { 5, 7 }, 8, 16)]
    [InlineData(new[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 8, 16)]
    [InlineData(new[] { 4, 4, 4 }, 8, 16)]
    [InlineData(new[] { 12 }, 6, 16)]    // pruned expert count (REAP-like, not a power of two)
    [InlineData(new[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 6, 16)]
    [InlineData(new[] { 12 }, 8, 128)]   // Air head geometry (hd128, rope 64)
    [InlineData(new[] { 3, 1, 8 }, 8, 128)]
    public void Vulkan_Matches_Cpu(int[] chunks, int experts, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var gguf = GgufFile.Open(Write(experts, nextn: true, headDim: headDim));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] cpu = Cpu(gguf, cfg);
        float[] vk = Vk(gguf, cfg, spvDir, chunks);
        // The cached Vulkan forward returns the LAST row of each chunk; pick the matching CPU rows.
        int v = SyntheticGlm4MoeGguf.VocabSize;
        var want = new List<float>();
        int end = 0;
        foreach (int n in chunks) { end += n; want.AddRange(cpu.AsSpan((end - 1) * v, v).ToArray()); }
        float d = MaxAbs(want.ToArray(), vk);
        // Exact-weight oracle: the same Q8_0 values as F32 on the CPU. Both backends quantise activations, so both sit within activation noise of it.
        using var gF32 = GgufFile.Open(Write(experts, nextn: true, headDim: headDim, asF32: true));
        float[] oracle = Cpu(gF32, GgufModelConfigExtractor.Extract(gF32.Metadata));
        var oracleRows = new List<float>();
        end = 0;
        foreach (int n in chunks) { end += n; oracleRows.AddRange(oracle.AsSpan((end - 1) * v, v).ToArray()); }
        Console.WriteLine($"  vk-vs-F32oracle {MaxAbs(oracleRows.ToArray(), vk)}  cpu-vs-F32oracle {MaxAbs(oracleRows.ToArray(), want.ToArray())}");
        float scale = cpu.Max(MathF.Abs);
        Console.WriteLine($"glm4moe vk-vs-cpu max diff {d} (logit scale {scale}) chunks [{string.Join(',', chunks)}] experts {experts} hd {headDim}");
        Assert.True(d < Tol, $"max |vulkan - cpu| = {d} (chunks [{string.Join(',', chunks)}], experts {experts}, hd {headDim})");
    }
}
