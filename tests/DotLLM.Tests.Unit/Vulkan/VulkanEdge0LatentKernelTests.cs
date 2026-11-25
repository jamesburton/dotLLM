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

    // ───────────────────────── decode group skipping (G3 step 2a) ─────────────────────────

    private static byte[] RandomQ8Rows(Random rng, int rows, int k)
    {
        int rowBytes = (k / 32) * 34;
        var b = new byte[(long)rows * rowBytes];
        rng.NextBytes(b);
        for (long o = 0; o + 34 <= b.Length; o += 34)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.02f));
            b[o] = (byte)bits; b[o + 1] = (byte)(bits >> 8);
        }
        return b;
    }

    /// <summary>K experts spread over exactly the given groups (so the other groups are unused).</summary>
    private static int[] ExpertsInFewGroups(int[] labels, int[] groups)
    {
        var idx = new List<int>();
        foreach (int g in groups) for (int e = 0; e < E && idx.Count < K; e++) if (labels[e] == g && !idx.Contains(e)) { idx.Add(e); break; }
        for (int e = 0; idx.Count < K && e < E; e++) if (groups.Contains(labels[e]) && !idx.Contains(e)) idx.Add(e);
        Assert.Equal(K, idx.Count);
        return idx.ToArray();
    }

    [SkippableFact]
    public void CombineAll_IsBitIdenticalToThePerGroupCombine()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasDecodeKernels, "decode kernels unavailable");
        const int L = 300;
        var rng = new Random(0xE3);
        int[] labels = Labels(rng);
        int[] idx = ExpertsInFewGroups(labels, [1, 4, 6]);
        float[] down = new float[K * L]; for (int i = 0; i < down.Length; i++) down[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] w = new float[K]; for (int i = 0; i < K; i++) w[i] = (float)rng.NextDouble();
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        using var bAll = device.Allocate((long)G * L * 4);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordCombineAll(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bAll, L, K, E); ctx.SubmitAndWait(); }
        float[] all = new float[G * L]; device.Download(bAll, all);
        for (int g = 0; g < G; g++)
        {
            using var bZ = device.Allocate((long)L * 4);
            using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordGroupCombine(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bZ, L, K, g, 1, E); ctx.SubmitAndWait(); }
            float[] z = new float[L]; device.Download(bZ, z);
            for (int c = 0; c < L; c++) Assert.True(z[c].Equals(all[g * L + c]), $"g {g} c {c}: per-group {z[c]:G9} vs all-groups {all[g * L + c]:G9}");
            if (g is not (1 or 4 or 6)) Assert.All(z, v => Assert.Equal(0f, v));   // an unused group is exact zeros
        }
    }

    [SkippableTheory]
    [InlineData(1792, 100)]
    [InlineData(224, 37)]      // 7 blocks: shorter than one 8-block window
    public void DecodeU_SkippingUnusedGroups_IsBitIdenticalToTheGenericAllGroupsPath(int L, int H)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasDecodeKernels, "decode kernels unavailable");
        using var q1 = QuantizeQ8_1Kernel.TryCreate(device, spvDir)!; using var qRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)!;
        var rng = new Random(0xE4 + L);
        int[] labels = Labels(rng);
        int[] used = [0, 2, 5, 7];
        int[] idx = ExpertsInFewGroups(labels, used);
        var uBufs = new VulkanDevice.Buffer[G]; var zs = new float[G][];
        for (int g = 0; g < G; g++)
        {
            byte[] u = RandomQ8Rows(rng, H, L);
            uBufs[g] = device.Allocate((u.Length + 3L) & ~3L); device.Upload(u, uBufs[g]);
            zs[g] = new float[L];
            if (used.Contains(g)) for (int c = 0; c < L; c++) zs[g][c] = (float)(rng.NextDouble() * 2 - 1);   // unused groups stay exact zeros (what edge0_group_combine writes)
        }
        float[] zAll = zs.SelectMany(z => z).ToArray();
        using var bIdx = Up(device, idx, out _); using var bLab = Up(device, labels, out _); using var bZAll = Up(device, zAll, out _);

        // generic: per group quantise (single-row kernel) + dense Q8_0 MMVQ, then the eight-way sum
        var parts = new VulkanDevice.Buffer[G];
        for (int g = 0; g < G; g++)
        {
            using var bz = Up(device, zs[g], out _);
            using var bxq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(L)); using var bxds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(L));
            parts[g] = device.Allocate((long)H * 4);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                q1.Record(ctx.CommandBuffer, bz, bxq, bxds, L);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                mmvq.Record(ctx.CommandBuffer, uBufs[g], bxq, bxds, parts[g], H, L);
                ctx.SubmitAndWait();
            }
            float[] part = new float[H]; device.Download(parts[g], part);
            // a skipped group would have contributed zero: its partial is exactly 0 for every row
            if (!used.Contains(g)) Assert.All(part, v => Assert.Equal(0f, v));
        }
        using var bGeneric = device.Allocate((long)H * 4);
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordSum(ctx.CommandBuffer, parts, bGeneric, H); ctx.SubmitAndWait(); }

        // fast: eight rows quantised together + one fused dispatch that skips the unused groups
        using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(G, L)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(G, L));
        using var bFast = device.Allocate((long)H * 4);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            qRows.Record(ctx.CommandBuffer, bZAll, bXq, bXds, G, L);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            k.RecordDecodeU(ctx.CommandBuffer, uBufs, bXq, bXds, bFast, bIdx, bLab, H, L, K, E);
            ctx.SubmitAndWait();
        }
        float[] a = new float[H], b = new float[H]; device.Download(bGeneric, a); device.Download(bFast, b);
        Assert.True(a.Max(MathF.Abs) > 1e-3f, "reference degenerate");
        for (int i = 0; i < H; i++) Assert.True(a[i].Equals(b[i]), $"row {i}: generic {a[i]:G9} vs skipping {b[i]:G9}");
        // the fixture must make the skip observable: the used groups really are a strict subset
        Assert.True(Enumerable.Range(0, G).Count(g => idx.Any(e => labels[e] == g)) < G);
        foreach (var u in uBufs) u.Dispose();
        foreach (var p in parts) p.Dispose();
    }

    [SkippableFact]
    public void DecodeD_ComputesTheUsedGroupsBitIdenticallyAndLeavesTheOthersUntouched()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasDecodeKernels, "decode kernels unavailable");
        using var q1 = QuantizeQ8_1Kernel.TryCreate(device, spvDir)!; using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)!;
        const int L = 160, H = 512;
        var rng = new Random(0xE5);
        int[] labels = Labels(rng);
        int[] used = [1, 3, 4];
        int[] idx = ExpertsInFewGroups(labels, used);
        byte[] d = RandomQ8Rows(rng, G * L, H);
        float[] x = new float[H]; for (int i = 0; i < H; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using var bD = device.Allocate((d.Length + 3L) & ~3L); device.Upload(d, bD);
        using var bx = Up(device, x, out _); using var bIdx = Up(device, idx, out _); using var bLab = Up(device, labels, out _);
        using var bxq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(H)); using var bxds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(H));
        using var bRef = device.Allocate((long)G * L * 4); using var bFast = Up(device, Enumerable.Repeat(float.NaN, G * L).ToArray(), out _);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q1.Record(ctx.CommandBuffer, bx, bxq, bxds, H);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bD, bxq, bxds, bRef, G * L, H);
            k.RecordDecodeD(ctx.CommandBuffer, bD, bxq, bxds, bFast, bIdx, bLab, G, L, H, K, E);
            ctx.SubmitAndWait();
        }
        float[] r = new float[G * L], f = new float[G * L]; device.Download(bRef, r); device.Download(bFast, f);
        int skipped = 0;
        for (int row = 0; row < G * L; row++)
        {
            int g = row / L;
            bool isUsed = idx.Any(e => labels[e] == g);
            if (isUsed) Assert.True(r[row].Equals(f[row]), $"row {row} (group {g}): dense {r[row]:G9} vs skipping {f[row]:G9}");
            else { Assert.True(float.IsNaN(f[row]), $"row {row} (unused group {g}) was written"); skipped++; }
        }
        Assert.True(skipped > 0, "no group was skipped: the fixture proves nothing");
    }
}
