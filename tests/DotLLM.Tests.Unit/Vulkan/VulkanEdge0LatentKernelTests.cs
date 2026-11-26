using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 G3 (#1981): the glue and F16-output kernels of the Vulkan latent MoE against plain C# references written from the semantics in
/// <c>docs/EDGE0_PACK_FORMAT.md</c> section 1. Shapes are non-degenerate on purpose: uneven group labels (an expert-to-group map that is neither the
/// identity nor <c>e % G</c>), a non-uniform column scale, topK > 1 with repeated groups inside one token, groups with no expert for some tokens.
/// The F16-output variants are checked against the F32-output kernels they derive from (rounded once to F16: the result must be bit-identical, the
/// accumulation and the order are untouched), and the fused combine-to-Q8_1 against the CPU Q8_1 quantiser contract.
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

    private static uint[] DownloadU32(VulkanDevice d, VulkanDevice.Buffer b, int words)
    {
        var raw = new uint[words];
        d.Download(b, MemoryMarshal.Cast<uint, float>(raw.AsSpan()));
        return raw;
    }

    private static ushort[] DownloadHalfBits(VulkanDevice d, VulkanDevice.Buffer b, int count)
    {
        var w = DownloadU32(d, b, (count + 1) / 2);
        return MemoryMarshal.Cast<uint, ushort>(w.AsSpan()).Slice(0, count).ToArray();
    }

    private static ushort HalfBits(float v) => BitConverter.HalfToUInt16Bits((Half)v);

    /// <summary>The CPU Q8_1 contract (== quantize_q8_1_rows.comp): d = amax / 127, id = 1 / d, q = roundEven(x * id), s = d * sum(q).</summary>
    private static (uint[] Xq, float[] D, float[] S) QuantRef(float[] z, int rows, int L)
    {
        var xq = new uint[rows * L / 4]; var d = new float[rows * L / 32]; var s = new float[rows * L / 32];
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < L / 32; b++)
            {
                float amax = 0; for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(z[r * L + b * 32 + i]));
                float dd = amax / 127f; float id = dd != 0f ? 1f / dd : 0f;
                int sum = 0;
                for (int w = 0; w < 8; w++)
                {
                    uint packed = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        int q = (int)MathF.Round(z[r * L + b * 32 + w * 4 + j] * id, MidpointRounding.ToEven);
                        sum += q; packed |= (uint)(q & 0xFF) << (8 * j);
                    }
                    xq[r * (L / 4) + b * 8 + w] = packed;
                }
                d[r * (L / 32) + b] = dd; s[r * (L / 32) + b] = dd * sum;
            }
        return (xq, d, s);
    }

    private static (uint[] Xq, float[] D, float[] S) DownloadQ81(VulkanDevice device, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int rows, int L)
    {
        var q = DownloadU32(device, xq, rows * L / 4);
        var ds = DownloadU32(device, xds, rows * L / 32 * 2);
        var d = new float[rows * L / 32]; var s = new float[rows * L / 32];
        for (int i = 0; i < d.Length; i++) { d[i] = BitConverter.UInt32BitsToSingle(ds[2 * i]); s[i] = BitConverter.UInt32BitsToSingle(ds[2 * i + 1]); }
        return (q, d, s);
    }

    [SkippableFact]
    public void Gather_PicksTheGroupRowOfEachExpertAndAppliesInverseScale_FromF16()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int L = 300, T = 3, rows = T * K;
        var rng = new Random(0xE0);
        int[] labels = Labels(rng);
        var xl = new Half[T * G * L]; for (int i = 0; i < xl.Length; i++) xl[i] = (Half)(float)(rng.NextDouble() * 2 - 1);
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
                float want = (float)xl[((v / K) * G + labels[idx[v]]) * L + c] * inv[c];
                Assert.True(want == got[v * L + c], $"v {v} c {c}: want {want:G9} got {got[v * L + c]:G9}");
            }
        // the reference must be able to tell the arms apart: a different group, no scale, or the wrong token row each change the value
        int vv = 0; float a = (float)xl[((vv / K) * G + labels[idx[vv]]) * L] * inv[0];
        Assert.NotEqual(a, (float)xl[((vv / K) * G + (labels[idx[vv]] + 1) % G) * L] * inv[0]);
        Assert.NotEqual(a, (float)xl[((vv / K) * G + labels[idx[vv]]) * L]);
    }

    /// <summary>Dyadic data so every product and sum is exact in F32 (no FMA ambiguity) but NOT representable in F16: z is rounded to F16 first, then the combine must equal the CPU Q8_1 quantiser contract bit for bit.</summary>
    [SkippableTheory]
    [InlineData(288, 4)]
    [InlineData(1792, 3)]
    public void GroupCombine_WritesQ8_1_BitIdenticalToTheCpuQuantiserContract(int L, int T)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        int rows = T * K;
        var rng = new Random(0xE1 + L);
        int[] labels = Labels(rng);
        float[] down = new float[rows * L]; for (int i = 0; i < down.Length; i++) down[i] = rng.Next(-4096, 4097) / 1024f;
        float[] w = new float[rows]; for (int i = 0; i < rows; i++) w[i] = rng.Next(0, 17) / 16f;
        int[] idx = new int[rows]; for (int i = 0; i < rows; i++) idx[i] = rng.Next(E);
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        int emptyCells = 0, notF16Exact = 0;
        for (int g = 0; g < G; g++)
        {
            using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(T, L)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(T, L));
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                k.RecordGroupCombine(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bXq, bXds, L, K, g, T, E);
                ctx.SubmitAndWait();
            }
            var z = new float[T * L];
            for (int t = 0; t < T; t++)
            {
                bool any = false;
                for (int c = 0; c < L; c++)
                {
                    float acc = 0;
                    for (int j = 0; j < K; j++) { int v = t * K + j; if (labels[idx[v]] == g) { acc += w[v] * down[v * L + c]; any = true; } }
                    if ((float)(Half)acc != acc) notF16Exact++;
                    z[t * L + c] = (float)(Half)acc;   // the combine rounds z to F16 before quantising
                }
                if (!any) emptyCells++;
            }
            var want = QuantRef(z, T, L);
            var got = DownloadQ81(device, bXq, bXds, T, L);
            for (int i = 0; i < want.Xq.Length; i++) Assert.True(want.Xq[i] == got.Xq[i], $"g {g}: xq word {i}: want {want.Xq[i]:X8} got {got.Xq[i]:X8}");
            for (int i = 0; i < want.D.Length; i++)
            {
                Assert.True(BitConverter.SingleToUInt32Bits(want.D[i]) == BitConverter.SingleToUInt32Bits(got.D[i]), $"g {g}: d {i}: want {want.D[i]:G9} got {got.D[i]:G9}");
                Assert.True(BitConverter.SingleToUInt32Bits(want.S[i]) == BitConverter.SingleToUInt32Bits(got.S[i]), $"g {g}: s {i}: want {want.S[i]:G9} got {got.S[i]:G9}");
            }
            if (g == 0) Assert.Contains(want.D, v => v > 0);   // the reference is not all zeros
        }
        Assert.True(emptyCells > 0, "fixture never produced a (token, group) pair without an expert: the zeroing path (d = 0, q = 0) is untested");
        Assert.True(notF16Exact > 100, "fixture z values are all F16-exact: the F16 rounding is unobservable");
    }

    /// <summary>Random data: the dequantised blocks must reproduce the C# group sums to within half a quantisation step (FMA contraction may move a sum by an ULP) plus the F16 rounding of z.</summary>
    [SkippableFact]
    public void GroupCombine_RandomData_DequantisesToTheWeightedGroupSum()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int L = 416, T = 4, rows = T * K;
        var rng = new Random(0xE9);
        int[] labels = Labels(rng);
        float[] down = new float[rows * L]; for (int i = 0; i < down.Length; i++) down[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] w = new float[rows]; for (int i = 0; i < rows; i++) w[i] = (float)rng.NextDouble();
        int[] idx = new int[rows]; for (int i = 0; i < rows; i++) idx[i] = rng.Next(E);
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        for (int g = 0; g < G; g++)
        {
            using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(T, L)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(T, L));
            using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordGroupCombine(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bXq, bXds, L, K, g, T, E); ctx.SubmitAndWait(); }
            var got = DownloadQ81(device, bXq, bXds, T, L);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < L; c++)
                {
                    float acc = 0;
                    for (int j = 0; j < K; j++) { int v = t * K + j; if (labels[idx[v]] == g) acc += w[v] * down[v * L + c]; }
                    uint word = got.Xq[t * (L / 4) + c / 4];
                    int q = (sbyte)((word >> (8 * (c % 4))) & 0xFF);
                    float d = got.D[t * (L / 32) + c / 32];
                    Assert.True(MathF.Abs(q * d - acc) <= 0.5001f * d + 6e-4f * MathF.Abs(acc) + 1e-6f, $"g {g} t {t} c {c}: z {acc:G9} dequantised {q * d:G9} (d {d:G9})");
                }
        }
    }

    [SkippableFact]
    public void UAcc_SumsTheGroupsInAscendingOrderInF32_FromF16()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        const int n = 1000;
        var rng = new Random(0xE2);
        var us = new Half[G][];
        for (int g = 0; g < G; g++) { us[g] = new Half[n]; for (int i = 0; i < n; i++) us[g][i] = (Half)((float)(rng.NextDouble() * 2 - 1) * (g + 1)); }
        using var bOut = Up(device, Enumerable.Repeat(float.NaN, n).ToArray(), out _);   // group 0 must OVERWRITE, not add to, what was there
        for (int g = 0; g < G; g++)
        {
            using var bu = Up(device, us[g], out _);
            using var ctx = device.CreateSubmitContext();
            ctx.Begin(); k.RecordUAcc(ctx.CommandBuffer, bu, bOut, n, first: g == 0); ctx.SubmitAndWait();
        }
        float[] got = new float[n]; device.Download(bOut, got);
        for (int i = 0; i < n; i++)
        {
            float want = (float)us[0][i]; for (int g = 1; g < G; g++) want += (float)us[g][i];
            Assert.True(want == got[i], $"i {i}: want {want:G9} got {got[i]:G9}");
        }
    }

    // ───────────────────────── decode group skipping (G3 step 2a), F16 D output ─────────────────────────

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
        const int L = 320;
        var rng = new Random(0xE3);
        int[] labels = Labels(rng);
        int[] idx = ExpertsInFewGroups(labels, [1, 4, 6]);
        float[] down = new float[K * L]; for (int i = 0; i < down.Length; i++) down[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] w = new float[K]; for (int i = 0; i < K; i++) w[i] = (float)rng.NextDouble();
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        using var bAq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(G, L)); using var bAds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(G, L));
        using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordCombineAll(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bAq, bAds, L, K, E); ctx.SubmitAndWait(); }
        var all = DownloadQ81(device, bAq, bAds, G, L);
        for (int g = 0; g < G; g++)
        {
            using var bq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, L)); using var bds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, L));
            using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordGroupCombine(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bq, bds, L, K, g, 1, E); ctx.SubmitAndWait(); }
            var one = DownloadQ81(device, bq, bds, 1, L);
            for (int i = 0; i < L / 4; i++) Assert.True(one.Xq[i] == all.Xq[g * (L / 4) + i], $"g {g} word {i}: per-group {one.Xq[i]:X8} vs all-groups {all.Xq[g * (L / 4) + i]:X8}");
            for (int i = 0; i < L / 32; i++)
            {
                Assert.True(one.D[i].Equals(all.D[g * (L / 32) + i]) && one.S[i].Equals(all.S[g * (L / 32) + i]), $"g {g} block {i}");
                if (g is not (1 or 4 or 6)) { Assert.Equal(0f, one.D[i]); Assert.Equal(0f, one.S[i]); }   // an unused group is exact zeros
            }
            if (g is not (1 or 4 or 6)) Assert.All(one.Xq, v => Assert.Equal(0u, v));
        }
    }

    [SkippableTheory]
    [InlineData(1792, 100)]
    [InlineData(224, 37)]      // 7 blocks: shorter than one 8-block window
    public void DecodeU_SkippingUnusedGroups_IsBitIdenticalToVisitingEveryGroupAndToTheDensePath(int L, int H)
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

        // dense: per group quantise (single-row kernel) + dense Q8_0 MMVQ, then the ascending eight-way sum on the host
        var sum = new float[H];
        for (int g = 0; g < G; g++)
        {
            using var bz = Up(device, zs[g], out _);
            using var bxq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(L)); using var bxds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(L));
            using var part = device.Allocate((long)H * 4);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                q1.Record(ctx.CommandBuffer, bz, bxq, bxds, L);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                mmvq.Record(ctx.CommandBuffer, uBufs[g], bxq, bxds, part, H, L);
                ctx.SubmitAndWait();
            }
            float[] p = new float[H]; device.Download(part, p);
            if (!used.Contains(g)) Assert.All(p, v => Assert.Equal(0f, v));   // a skipped group would have contributed exactly zero
            for (int i = 0; i < H; i++) sum[i] = g == 0 ? p[i] : sum[i] + p[i];
        }

        // fast: eight rows quantised together + one fused dispatch, skipping and not skipping the unused groups
        using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(G, L)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(G, L));
        using var bSkip = device.Allocate((long)H * 4); using var bAll = device.Allocate((long)H * 4);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            qRows.Record(ctx.CommandBuffer, bZAll, bXq, bXds, G, L);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            k.RecordDecodeU(ctx.CommandBuffer, uBufs, bXq, bXds, bSkip, bIdx, bLab, H, L, K, E, skipUnused: true);
            k.RecordDecodeU(ctx.CommandBuffer, uBufs, bXq, bXds, bAll, bIdx, bLab, H, L, K, E, skipUnused: false);
            ctx.SubmitAndWait();
        }
        float[] a = new float[H], b = new float[H]; device.Download(bSkip, a); device.Download(bAll, b);
        Assert.True(sum.Max(MathF.Abs) > 1e-3f, "reference degenerate");
        for (int i = 0; i < H; i++)
        {
            Assert.True(sum[i].Equals(a[i]), $"row {i}: dense {sum[i]:G9} vs skipping {a[i]:G9}");
            Assert.True(a[i].Equals(b[i]), $"row {i}: skipping {a[i]:G9} vs all groups {b[i]:G9}");
        }
        // the fixture must make the skip observable: the used groups really are a strict subset
        Assert.True(Enumerable.Range(0, G).Count(g => idx.Any(e => labels[e] == g)) < G);
        foreach (var u in uBufs) u.Dispose();
    }

    [SkippableFact]
    public void DecodeD_F16Output_IsTheDenseKernelRoundedOnce_AndSkipsUnusedGroups()
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
        using var bRef = device.Allocate((long)G * L * 4);
        using var bSkip = Up(device, Enumerable.Repeat((ushort)0x7E00, G * L).ToArray(), out _);      // Half NaN sentinel
        using var bAll = Up(device, Enumerable.Repeat((ushort)0x7E00, G * L).ToArray(), out _);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q1.Record(ctx.CommandBuffer, bx, bxq, bxds, H);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bD, bxq, bxds, bRef, G * L, H);
            k.RecordDecodeD(ctx.CommandBuffer, bD, bxq, bxds, bSkip, bIdx, bLab, G, L, H, K, E, skipUnused: true);
            k.RecordDecodeD(ctx.CommandBuffer, bD, bxq, bxds, bAll, bIdx, bLab, G, L, H, K, E, skipUnused: false);
            ctx.SubmitAndWait();
        }
        float[] r = new float[G * L]; device.Download(bRef, r);
        ushort[] f = DownloadHalfBits(device, bSkip, G * L), a = DownloadHalfBits(device, bAll, G * L);
        int skipped = 0;
        for (int row = 0; row < G * L; row++)
        {
            int g = row / L;
            Assert.True(a[row] == HalfBits(r[row]), $"row {row}: all-groups {a[row]:X4} vs dense rounded {HalfBits(r[row]):X4} ({r[row]:G9})");
            if (idx.Any(e => labels[e] == g)) Assert.True(f[row] == HalfBits(r[row]), $"row {row} (group {g}): skipping {f[row]:X4} vs dense rounded {HalfBits(r[row]):X4}");
            else { Assert.True(f[row] == 0x7E00, $"row {row} (unused group {g}) was written"); skipped++; }
        }
        Assert.True(skipped > 0, "no group was skipped: the fixture proves nothing");
        Assert.Contains(r, v => MathF.Abs(v) > 1e-3f);
        // F16 really rounds: at least one dense value is not representable in half precision
        Assert.Contains(Enumerable.Range(0, G * L), i => (float)(Half)r[i] != r[i]);
    }

    // ───────────────────────── F16-output Q8_0 matmuls (D / U for more than one row) ─────────────────────────

    [SkippableTheory]
    [InlineData(200, 256, 150)]   // partial 128-tiles in M and N
    [InlineData(300, 192, 9)]
    public void GemmD_F32In_F16Out_IsTheBlockedF32GemmRoundedOnce(int m, int kk, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix && device.SubgroupSize == 64, "needs wave64 cooperative matrices");
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasNarrowPath, "narrow path unavailable");
        var variant = Q8_0GemmCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(variant.IsSupportedOn(device, spvDir), "blocked Q8_0 GEMM unavailable");
        using var refK = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, variant);
        var rng = new Random(0xE6 + m);
        byte[] wq = RandomQ8Rows(rng, m, kk);
        float[] b = new float[n * kk]; for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        using var bW = device.Allocate((wq.Length + 3L) & ~3L); device.Upload(wq, bW);
        using var bB = Up(device, b, out _);
        using var bRef = device.Allocate((long)n * m * 4); using var bC = device.Allocate(((long)n * m * 2 + 3) & ~3L);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            refK.Record(ctx.CommandBuffer, bW, bB, bRef, m, kk, n);
            k.RecordGemmD(ctx.CommandBuffer, bW, bB, bC, m, kk, n);
            ctx.SubmitAndWait();
        }
        float[] r = new float[n * m]; device.Download(bRef, r);
        ushort[] c = DownloadHalfBits(device, bC, n * m);
        for (int i = 0; i < r.Length; i++) Assert.True(c[i] == HalfBits(r[i]), $"i {i}: f16 {c[i]:X4} vs f32 rounded {HalfBits(r[i]):X4} ({r[i]:G9})");
        Assert.Contains(Enumerable.Range(0, r.Length), i => (float)(Half)r[i] != r[i]);
    }

    /// <summary>Same exact-arithmetic data as the Q8_1 combine test: the F16 combine must equal the F16-rounded group sum bit for bit, with exact zeros for a group no expert of the token is in.</summary>
    [SkippableFact]
    public void GroupCombineF16_IsTheGroupSumRoundedOnceToF16()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix && device.SubgroupSize == 64, "needs wave64 cooperative matrices");
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasNarrowPath, "narrow path unavailable");
        const int L = 300, T = 5, rows = T * K;
        var rng = new Random(0xEA);
        int[] labels = Labels(rng);
        float[] down = new float[rows * L]; for (int i = 0; i < down.Length; i++) down[i] = rng.Next(-4096, 4097) / 1024f;
        float[] w = new float[rows]; for (int i = 0; i < rows; i++) w[i] = rng.Next(0, 17) / 16f;
        int[] idx = new int[rows]; for (int i = 0; i < rows; i++) idx[i] = rng.Next(E);
        using var bDown = Up(device, down, out _); using var bIdx = Up(device, idx, out _); using var bW = Up(device, w, out _); using var bLab = Up(device, labels, out _);
        int empty = 0, notExact = 0;
        for (int g = 0; g < G; g++)
        {
            using var bZ = device.Allocate(((long)T * L * 2 + 3) & ~3L);
            using (var ctx = device.CreateSubmitContext()) { ctx.Begin(); k.RecordGroupCombineF16(ctx.CommandBuffer, bDown, bIdx, bW, bLab, bZ, L, K, g, T, E); ctx.SubmitAndWait(); }
            ushort[] z = DownloadHalfBits(device, bZ, T * L);
            for (int t = 0; t < T; t++)
            {
                bool any = false;
                for (int c = 0; c < L; c++)
                {
                    float acc = 0;
                    for (int j = 0; j < K; j++) { int v = t * K + j; if (labels[idx[v]] == g) { acc += w[v] * down[v * L + c]; any = true; } }
                    if ((float)(Half)acc != acc) notExact++;
                    Assert.True(z[t * L + c] == HalfBits(acc), $"g {g} t {t} c {c}: want {HalfBits(acc):X4} got {z[t * L + c]:X4} (z {acc:G9})");
                }
                if (!any) empty++;
            }
        }
        Assert.True(empty > 0 && notExact > 100, "fixture does not exercise empty groups and F16 rounding");
    }

    [SkippableTheory]
    [InlineData(200, 256, 150)]
    [InlineData(300, 192, 9)]
    public void GemmU_F16In_F16Out_IsTheBlockedF32GemmRoundedOnce(int m, int kk, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix && device.SubgroupSize == 64, "needs wave64 cooperative matrices");
        using var k = Edge0LatentKernels.Create(device, spvDir);
        Skip.IfNot(k.HasNarrowPath, "narrow path unavailable");
        var variant = Q8_0GemmCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(variant.IsSupportedOn(device, spvDir), "blocked Q8_0 GEMM unavailable");
        using var refK = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, variant);
        var rng = new Random(0xE7 + m);
        byte[] wq = RandomQ8Rows(rng, m, kk);
        var z = new Half[n * kk]; for (int i = 0; i < z.Length; i++) z[i] = (Half)(float)(rng.NextDouble() * 2 - 1);
        float[] zf = z.Select(h => (float)h).ToArray();
        using var bW = device.Allocate((wq.Length + 3L) & ~3L); device.Upload(wq, bW);
        using var bZ = Up(device, z, out _); using var bZf = Up(device, zf, out _);
        using var bRef = device.Allocate((long)n * m * 4); using var bC = device.Allocate(((long)n * m * 2 + 3) & ~3L);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            refK.Record(ctx.CommandBuffer, bW, bZf, bRef, m, kk, n);
            k.RecordGemmU(ctx.CommandBuffer, bW, bZ, bC, m, kk, n);
            ctx.SubmitAndWait();
        }
        float[] r = new float[n * m]; device.Download(bRef, r);
        ushort[] c = DownloadHalfBits(device, bC, n * m);
        for (int i = 0; i < r.Length; i++) Assert.True(c[i] == HalfBits(r[i]), $"i {i}: f16 {c[i]:X4} vs f32 rounded {HalfBits(r[i]):X4} ({r[i]:G9})");
        Assert.Contains(Enumerable.Range(0, r.Length), i => (float)(Half)r[i] != r[i]);
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(8)]
    public void MmvqMulti_F16Output_IsTheF32KernelRoundedOnce(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var c16 = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir, f16Output: true);
        using var f32 = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir);
        Skip.If(c16 is null || f32 is null, "multi kernels unavailable");
        using var qRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)!;
        const int m = 190, kk = 224;
        var rng = new Random(0xE8 + n);
        byte[] wq = RandomQ8Rows(rng, m, kk);
        float[] x = new float[n * kk]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using var bW = device.Allocate((wq.Length + 3L) & ~3L); device.Upload(wq, bW);
        using var bX = Up(device, x, out _);
        using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, kk)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, kk));
        using var bRef = device.Allocate((long)n * m * 4); using var bC = device.Allocate(((long)n * m * 2 + 3) & ~3L);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            qRows.Record(ctx.CommandBuffer, bX, bXq, bXds, n, kk);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            f32!.Record(ctx.CommandBuffer, bW, bXq, bXds, bRef, m, kk, n);
            c16!.Record(ctx.CommandBuffer, bW, bXq, bXds, bC, m, kk, n);
            ctx.SubmitAndWait();
        }
        float[] r = new float[n * m]; device.Download(bRef, r);
        ushort[] c = DownloadHalfBits(device, bC, n * m);
        for (int i = 0; i < r.Length; i++) Assert.True(c[i] == HalfBits(r[i]), $"i {i}: f16 {c[i]:X4} vs f32 rounded {HalfBits(r[i]):X4} ({r[i]:G9})");
        Assert.Contains(Enumerable.Range(0, r.Length), i => (float)(Half)r[i] != r[i]);
    }
    // ----------------------- F16 wide GEMV (D / U at 9..16 rows, #2004) -----------------------

    /// <summary>Scalar double reference of a Q8_0 matmul (block = fp16 scale + 32 int8).</summary>
    private static double[] ScalarQ8Matmul(byte[] wq, float[] x, int m, int kk, int n)
    {
        var y = new double[n * m];
        int bpr = kk / 32;
        for (int c = 0; c < n; c++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int b = 0; b < bpr; b++)
                {
                    long o = ((long)r * bpr + b) * 34;
                    double d = (double)BitConverter.UInt16BitsToHalf((ushort)(wq[o] | (wq[o + 1] << 8)));
                    double dot = 0;
                    for (int i = 0; i < 32; i++) dot += (sbyte)wq[o + 2 + i] * (double)x[c * kk + b * 32 + i];
                    acc += d * dot;
                }
                y[c * m + r] = acc;
            }
        return y;
    }

    /// <summary>D (F32 in) and U (F16 in) wide GEMVs: bit-identical to the F32 wide GEMV they derive from, rounded once to F16, and within F32-accumulation error of a scalar double reference.</summary>
    [SkippableTheory]
    [InlineData(false, 9)]
    [InlineData(false, 12)]
    [InlineData(false, 16)]
    [InlineData(true, 9)]
    [InlineData(true, 12)]
    [InlineData(true, 16)]
    public void WideGemv_F16Output_IsTheF32WideKernelRoundedOnce_AndMatchesAScalarReference(bool f16Input, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var wide = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input);
        using var f32 = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir);
        Skip.If(wide is null || f32 is null, "wide GEMV unavailable");
        const int m = 203, kk = 224;   // M not a multiple of NR (partial last workgroup), K not a multiple of the 64-lane window
        var rng = new Random(0xE9 + n + (f16Input ? 100 : 0));
        byte[] wq = RandomQ8Rows(rng, m, kk);
        float[] x = new float[n * kk]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        if (f16Input) for (int i = 0; i < x.Length; i++) x[i] = (float)(Half)x[i];   // z is F16-valued, so the F32 kernel sees the same numbers
        using var bW = device.Allocate((wq.Length + 3L) & ~3L); device.Upload(wq, bW);
        using var bXf = Up(device, x, out _);
        using var bXh = Up(device, x.Select(v => (Half)v).ToArray(), out _);
        using var bRef = device.Allocate((long)n * m * 4); using var bC = device.Allocate(((long)n * m * 2 + 3) & ~3L);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            f32!.Record(ctx.CommandBuffer, bW, bXf, bRef, m, kk, n);
            wide!.Record(ctx.CommandBuffer, bW, f16Input ? bXh : bXf, bC, m, kk, n);
            ctx.SubmitAndWait();
        }
        float[] r = new float[n * m]; device.Download(bRef, r);
        ushort[] c = DownloadHalfBits(device, bC, n * m);
        for (int i = 0; i < r.Length; i++) Assert.True(c[i] == HalfBits(r[i]), $"i {i} (col {i / m}, row {i % m}): f16 {c[i]:X4} vs f32 rounded {HalfBits(r[i]):X4} ({r[i]:G9})");
        Assert.Contains(Enumerable.Range(0, r.Length), i => (float)(Half)r[i] != r[i]);
        double[] sc = ScalarQ8Matmul(wq, x, m, kk, n);
        for (int i = 0; i < sc.Length; i++)
        {
            float g = (float)BitConverter.UInt16BitsToHalf(c[i]);
            Assert.True(Math.Abs(g - sc[i]) <= 2e-3 * Math.Max(1.0, Math.Abs(sc[i])) + 1e-3, $"i {i}: gpu {g} scalar {sc[i]}");
        }
    }
}
