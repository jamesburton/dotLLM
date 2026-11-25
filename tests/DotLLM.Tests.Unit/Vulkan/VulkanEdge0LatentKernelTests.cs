using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 G3 (#1981): the three glue kernels of the Vulkan latent MoE against plain C# references written from the semantics in
/// <c>docs/EDGE0_PACK_FORMAT.md</c> section 1. Shapes are non-degenerate on purpose: uneven group labels (an expert-to-group map that is neither the
/// identity nor <c>e % G</c>), a non-uniform column scale, topK > 1 with repeated groups inside one token, groups with no expert for some tokens.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanEdge0LatentKernelTests
{
    private const int E = 37, G = 8, K = 5;

    private static int[] Labels(Random rng) { var l = new int[E]; for (int e = 0; e < E; e++) l[e] = (e * 5 + rng.Next(3)) % G; return l; }

    private static VulkanDevice.Buffer Up<T>(VulkanDevice d, T[] data, out long bytes) where T : unmanaged
    {
        bytes = (long)data.Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var b = d.Allocate((bytes + 3) & ~3L);
        d.Upload(MemoryMarshal.AsBytes<T>(data), b);
        return b;
    }

    [SkippableFact]
    public void Gather_PicksTheGroupRowOfEachExpertAndAppliesInverseScale()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int L = 300, T = 3, rows = T * K;
        var rng = new Random(0xE0);
        int[] labels = Labels(rng);
        float[] xl = new float[T * G * L]; for (int i = 0; i < xl.Length; i++) xl[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] inv = new float[L]; for (int i = 0; i < L; i++) inv[i] = 0.5f + (float)rng.NextDouble();
        int[] idx = new int[rows]; for (int i = 0; i < rows; i++) idx[i] = rng.Next(E);
        using var bXl = Up(device, xl, out _); using var bIdx = Up(device, idx, out _); using var bLab = Up(device, labels, out _); using var bInv = Up(device, inv, out _);
        using var bOut = device.Allocate((long)rows * L * 4);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            k.RecordGather(ctx.CommandBuffer, bXl, bIdx, bLab, bInv, bOut, L, G, K, rows, E);
            ctx.SubmitAndWait();
        }
        float[] got = new float[rows * L]; device.Download(bOut, got);
        for (int v = 0; v < rows; v++)
            for (int c = 0; c < L; c++)
            {
                float want = xl[((v / K) * G + labels[idx[v]]) * L + c] * inv[c];
                Assert.True(want == got[v * L + c], $"v {v} c {c}: want {want:G9} got {got[v * L + c]:G9}");
            }
        // the reference must be able to tell the arms apart: a different group, no scale, or the wrong token row each change the value
        int vv = 0; float a = xl[((vv / K) * G + labels[idx[vv]]) * L] * inv[0];
        Assert.NotEqual(a, xl[((vv / K) * G + (labels[idx[vv]] + 1) % G) * L] * inv[0]);
        Assert.NotEqual(a, xl[((vv / K) * G + labels[idx[vv]]) * L]);
    }

    [SkippableFact]
    public void GroupCombine_SumsWeightedRowsOfThatGroupOnly()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int L = 300, T = 4, rows = T * K;
        var rng = new Random(0xE1);
        int[] labels = Labels(rng);
        float[] down = new float[rows * L]; for (int i = 0; i < down.Length; i++) down[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] w = new float[rows]; for (int i = 0; i < rows; i++) w[i] = (float)rng.NextDouble();
        int[] idx = new int[rows]; for (int i = 0; i < rows; i++) idx[i] = rng.Next(E);
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        int emptyCells = 0;
        for (int g = 0; g < G; g++)
        {
            using var bZ = device.Allocate((long)T * L * 4);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                k.RecordGroupCombine(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bZ, L, K, g, T, E);
                ctx.SubmitAndWait();
            }
            float[] z = new float[T * L]; device.Download(bZ, z);
            for (int t = 0; t < T; t++)
            {
                bool any = false;
                for (int c = 0; c < L; c++)
                {
                    float acc = 0;
                    for (int j = 0; j < K; j++) { int v = t * K + j; if (labels[idx[v]] == g) { acc += w[v] * down[v * L + c]; any = true; } }
                    Assert.True(MathF.Abs(acc - z[t * L + c]) <= 1e-6f * (1 + MathF.Abs(acc)), $"g {g} t {t} c {c}: want {acc:G9} got {z[t * L + c]:G9}");
                }
                if (!any) emptyCells++;
            }
        }
        Assert.True(emptyCells > 0, "fixture never produced a (token, group) pair without an expert: the zeroing path is untested");
    }

    [SkippableFact]
    public void USum_AddsTheEightGroupOutputs()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int n = 1000;
        var rng = new Random(0xE2);
        var us = new float[G][]; var bufs = new VulkanDevice.Buffer[G];
        for (int g = 0; g < G; g++) { us[g] = new float[n]; for (int i = 0; i < n; i++) us[g][i] = (float)(rng.NextDouble() * 2 - 1) * (g + 1); bufs[g] = Up(device, us[g], out _); }
        using var bOut = device.Allocate(n * 4);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordSum(ctx.CommandBuffer, bufs, bOut, n); ctx.SubmitAndWait(); }
        float[] got = new float[n]; device.Download(bOut, got);
        for (int i = 0; i < n; i++)
        {
            float want = 0; for (int g = 0; g < G; g++) want += us[g][i];
            Assert.True(want == got[i], $"i {i}: want {want:G9} got {got[i]:G9}");
        }
        foreach (var b in bufs) b.Dispose();
    }
}
