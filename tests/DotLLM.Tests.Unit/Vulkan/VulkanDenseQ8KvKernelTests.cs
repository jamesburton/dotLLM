using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using static DotLLM.Tests.Unit.Vulkan.VulkanQ8KvAttentionKernelTests;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1884: the dense chain's Q8_0-KV twins read the 36-byte blocks of a <see cref="VulkanKvCache"/> in place (no F32 dequant
/// scratch), the Vulkan counterpart of CUDA's #1850.
/// </summary>
/// <remarks>
/// <para>Oracles, as <see cref="VulkanQ8KvAttentionKernelTests"/>:</para>
/// <list type="bullet">
/// <item>The kernels that compute in F32 (per-token, scalar flash, scalar GQA flash, small-row split-KV) must be bit-identical, to a few
/// ULP, to their own F32 kernel run on rows dequantised by the C# reference. Any difference is a block, word, byte or scale indexing bug.</item>
/// <item>The coopmat kernels round K / V to F16 in LDS, so they are compared with their F16 twin on the F16-rounded dequantised rows,
/// at coopmat tolerance.</item>
/// <item>Every case also asserts that the output differs from the attention over the unquantised rows, and only by the Q8_0 budget.</item>
/// </list>
/// <para>Shapes: hd64 (Qwen2.5-0.5B: 14 heads, 2 KV heads) and hd128 (32 / 8), prefill at an offset and from zero.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanDenseQ8KvKernelTests
{
    private delegate void Rec(nint cmd, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer o);

    private static VulkanKvStorage _storage = VulkanKvStorage.Q8_0;   // set per test from the theory row

    /// <summary>
    /// The Q4_0 KV layout contract (kv_load.glsl KV_Q4, #1884): 20-byte blocks [F32 scale][16 bytes], scale = amax / 7 (IEEE), nibble =
    /// clamp(roundEven(x * (1 / scale)) + 8, 0, 15), byte j = element 2j (low) | element 2j + 1 (high): the CPU / CUDA KV rule.
    /// </summary>
    internal static byte[] QuantizeQ4Ref(ReadOnlySpan<float> x)
    {
        int blocks = x.Length / 32;
        var o = new byte[blocks * 20];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(x[b * 32 + i]));
            float d = amax / 7f, id = d != 0 ? 1f / d : 0f;
            BitConverter.TryWriteBytes(o.AsSpan(b * 20, 4), d);
            for (int j = 0; j < 16; j++)
            {
                int lo = d != 0 ? Math.Clamp((int)MathF.Round(x[b * 32 + 2 * j] * id, MidpointRounding.ToEven) + 8, 0, 15) : 8;
                int hi = d != 0 ? Math.Clamp((int)MathF.Round(x[b * 32 + 2 * j + 1] * id, MidpointRounding.ToEven) + 8, 0, 15) : 8;
                o[b * 20 + 4 + j] = (byte)((hi << 4) | lo);
            }
        }
        return o;
    }

    internal static float[] DequantizeQ4Ref(ReadOnlySpan<byte> blocks)
    {
        int n = blocks.Length / 20;
        var r = new float[n * 32];
        for (int b = 0; b < n; b++)
        {
            float d = BitConverter.ToSingle(blocks.Slice(b * 20, 4));
            for (int j = 0; j < 16; j++)
            {
                byte by = blocks[b * 20 + 4 + j];
                r[b * 32 + 2 * j] = ((by & 15) - 8) * d;
                r[b * 32 + 2 * j + 1] = ((by >> 4) - 8) * d;
            }
        }
        return r;
    }

    private static (float[] Q, float[] K, float[] V, byte[] Kq, byte[] Vq, float[] Kd, float[] Vd) Data(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int seed)
    {
        var rng = new Random(seed);
        int kvElems = seqKv * numKvHeads * headDim;
        float[] q = Rand(rng, seqQ * numHeads * headDim, 0.5f), k = Rand(rng, kvElems, 0.5f), v = Rand(rng, kvElems, 0.5f);
        // Each side in its own block format (mixed K / V, #1884).
        static (byte[] Q, float[] D) Side(VulkanKvStorage f, float[] x)
        {
            if (f == VulkanKvStorage.Q4_0) { var b = QuantizeQ4Ref(x); return (b, DequantizeQ4Ref(b)); }
            var b8 = QuantizeRef(x); return (b8, DequantizeRef(b8));
        }
        var (kq, kd) = Side(_storage.KeyFormat(), k);
        var (vq, vd) = Side(_storage.ValueFormat(), v);
        return (q, k, v, kq, vq, kd, vd);
    }

    /// <summary>F32-computing twin vs the F32 kernel on the dequantised rows: bit-identical up to FMA contraction.</summary>
    private static void ExactCase(VulkanDevice device, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, Rec q8, Rec f32)
    {
        var (q, k, v, kq, vq, kd, vd) = Data(seqQ, seqKv, numHeads, numKvHeads, headDim, 0x1884 + seqQ * 7 + seqKv);
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK8 = device.Allocate(kq.Length); using var bV8 = device.Allocate(vq.Length);
        using var bKd = device.Allocate(kd.Length * 4L); using var bVd = device.Allocate(vd.Length * 4L);
        using var bKr = device.Allocate(k.Length * 4L); using var bVr = device.Allocate(v.Length * 4L);
        using var oQ8 = device.Allocate(q.Length * 4L); using var oRef = device.Allocate(q.Length * 4L); using var oRaw = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kq, bK8); device.Upload(vq, bV8); device.Upload(kd, bKd); device.Upload(vd, bVd);
        device.Upload(k, bKr); device.Upload(v, bVr);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q8(ctx.CommandBuffer, bQ, bK8, bV8, oQ8);
            f32(ctx.CommandBuffer, bQ, bKd, bVd, oRef);
            f32(ctx.CommandBuffer, bQ, bKr, bVr, oRaw);
            ctx.SubmitAndWait();
        }
        int n = q.Length;
        float[] a = new float[n], r = new float[n], raw = new float[n];
        device.Download(oQ8, a); device.Download(oRef, r); device.Download(oRaw, raw);
        // Same operands (int8 * scale is exact in F32); only FMA contraction of that product into the dot / accumulate differs, so a few
        // ULP of the output amplitude. An indexing bug (block, word, byte, scale) moves results by percent.
        float refMax = r.Max(MathF.Abs);
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(a[i] - r[i]) <= 1e-5f * refMax, $"[{i}] Q8 kernel {a[i]:R} vs F32 kernel on dequantised rows {r[i]:R}");
        AssertQ8Budget(a, raw, r);
    }

    /// <summary>
    /// Controls: the Q8 output is NOT the attention over the unquantised rows (a kernel that read stale F32 or ignored the quantisation
    /// cannot pass), and is within the Q8_0 budget as a relative L2 error. A per-element bound is not used: early prefill rows attend to a
    /// handful of keys, so one outlier-heavy V row sets their output, and its quantisation error is larger than the average.
    /// </summary>
    private static void AssertQ8Budget(float[] a, float[] raw, float[] reference)
    {
        float maxAbs = raw.Max(MathF.Abs), worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - raw[i]));
        Assert.True(worst > 1e-5f * maxAbs, $"output is indistinguishable from the unquantised attention (worst {worst:G3})");
        // The twin's error vs the unquantised attention is the quantisation's own error: the reference kernel on the same dequantised rows
        // shows exactly that error, so the twin may not add to it. (A fixed budget would depend on the data: these blocks carry outliers that
        // set their scales, so Q4_0's 16 levels lose ~30% here, as CPU and CUDA would with the same rule.)
        double twin = RelL2(a, raw), own = RelL2(reference, raw);
        Assert.True(twin <= own * 1.02 + 1e-4, $"relative L2 error {twin:G4} exceeds the quantisation's own {own:G4}");
        Assert.True(twin < Budget, $"relative L2 error {twin:G3} exceeds the {_storage} sanity cap {Budget}");
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / den);
    }

    /// <summary>Coopmat twin vs its F16 twin on the F16-rounded dequantised rows (coopmat tolerance), with the Q8 budget controls.</summary>
    private static void CoopmatCase(VulkanDevice device, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, Rec q8, Rec f16)
    {
        var (q, k, v, kq, vq, kd, vd) = Data(seqQ, seqKv, numHeads, numKvHeads, headDim, 0x1885 + seqQ * 7 + seqKv);
        ushort[] kh = kd.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray(), vh = vd.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray();
        ushort[] krh = k.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray(), vrh = v.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray();
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK8 = device.Allocate(kq.Length); using var bV8 = device.Allocate(vq.Length);
        using var bK16 = device.Allocate(kh.Length * 2L); using var bV16 = device.Allocate(vh.Length * 2L);
        using var bKr = device.Allocate(krh.Length * 2L); using var bVr = device.Allocate(vrh.Length * 2L);
        using var oQ8 = device.Allocate(q.Length * 4L); using var oF16 = device.Allocate(q.Length * 4L); using var oRaw = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kq, bK8); device.Upload(vq, bV8);
        device.Upload(MemoryMarshal.AsBytes(kh.AsSpan()), bK16); device.Upload(MemoryMarshal.AsBytes(vh.AsSpan()), bV16);
        device.Upload(MemoryMarshal.AsBytes(krh.AsSpan()), bKr); device.Upload(MemoryMarshal.AsBytes(vrh.AsSpan()), bVr);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q8(ctx.CommandBuffer, bQ, bK8, bV8, oQ8);
            f16(ctx.CommandBuffer, bQ, bK16, bV16, oF16);
            f16(ctx.CommandBuffer, bQ, bKr, bVr, oRaw);
            ctx.SubmitAndWait();
        }
        int n = q.Length;
        float[] a = new float[n], r = new float[n], raw = new float[n];
        device.Download(oQ8, a); device.Download(oF16, r); device.Download(oRaw, raw);
        float maxAbs = r.Max(MathF.Abs), worst = 0, worstRaw = 0;
        for (int i = 0; i < n; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - r[i]));
            worstRaw = MathF.Max(worstRaw, MathF.Abs(a[i] - raw[i]));
        }
        Assert.True(maxAbs > 1e-3f, "degenerate reference");
        Assert.True(worst <= 2e-3f * maxAbs + 1e-5f, $"Q8 twin differs from the F16 twin on the dequantised rows by {worst:G3} (amplitude {maxAbs:G3})");
        Assert.True(worstRaw > 1e-5f * maxAbs, "control: the Q8 output must differ from attention over the unquantised rows");
        double twin = RelL2(a, raw), own = RelL2(r, raw);
        Assert.True(twin <= own * 1.02 + 1e-4, $"relative L2 error {twin:G4} exceeds the quantisation's own {own:G4}");
        Assert.True(twin < Budget, $"relative L2 error {twin:G3} exceeds the {_storage} sanity cap {Budget}");
    }

    // Sanity cap on the relative L2 error vs the unquantised attention (the sharp check is against the reference's own error).
    private static double Budget => _storage == VulkanKvStorage.Q8_0 ? 0.03 : 1.0;   // any side in Q4_0: loose cap

    /// <summary>
    /// The Q4_0 writer stores exactly the C# reference's bytes (scale bits and nibbles), at a non-zero destination block, leaving the
    /// rest of the buffer untouched. Hand-made blocks: all zero (scale 0, nibbles 8) and a lone spike. Values within 1e-4 of a rounding
    /// tie may land on either neighbour (x * (1 / scale) is not correctly rounded on every driver); the test data avoids them.
    /// </summary>
    [SkippableTheory]
    [InlineData(512, 7, 0)]
    [InlineData(512, 5, 3)]
    [InlineData(128, 1, 0)]
    public void Q4Writer_MatchesReferenceQuantiser_ByteForByte(int stride, int positions, int dstRow)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var writer = VulkanKvQ8WriteKernel.Create(device, spvDir, VulkanKvStorage.Q4_0);
        Assert.Equal(VulkanKvStorage.Q4_0, writer.Storage);
        var rng = new Random(0xA4 + stride + positions);
        float[] src = Rand(rng, positions * stride, 0.5f);
        Array.Clear(src, 0, 32);
        Array.Clear(src, 32, 32); src[41] = -12.5f;
        // Nudge values off rounding ties of x * (7 / amax) so a driver's approximate multiply cannot flip a nibble.
        for (int b = 0; b < src.Length / 32; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(src[b * 32 + i]));
            if (amax == 0) continue;
            float id = 1f / (amax / 7f);
            for (int i = 0; i < 32; i++)
            {
                double t = src[b * 32 + i] * (double)id, frac = Math.Abs(t - Math.Truncate(t));
                if (Math.Abs(frac - 0.5) < 1e-3) src[b * 32 + i] *= 0.999f;
            }
        }

        long blockOff = (long)dstRow * stride / 32, blocks = (long)positions * stride / 32;
        long dstBytes = (blockOff + blocks) * 20;
        using var bSrc = device.Allocate((long)src.Length * 4);
        using var bDst = device.Allocate(dstBytes);
        device.Upload(Enumerable.Repeat((byte)0xEE, (int)dstBytes).ToArray(), bDst);   // sentinel bytes outside the written range
        device.Upload(src, bSrc);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            writer.RecordF32ToQ8(ctx.CommandBuffer, bSrc, bDst, (long)dstRow * stride, (long)positions * stride);
            ctx.SubmitAndWait();
        }
        var gotF = new float[dstBytes / 4];
        device.Download(bDst, gotF.AsSpan());
        byte[] got = MemoryMarshal.AsBytes(gotF.AsSpan()).ToArray();
        byte[] expected = QuantizeQ4Ref(src);
        for (long i = 0; i < blockOff * 20; i++) Assert.Equal(0xEE, got[i]);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(expected[i] == got[blockOff * 20 + i], $"byte {i} (block {i / 20}): writer {got[blockOff * 20 + i]:X2} vs reference {expected[i]:X2}");
        Assert.Equal(0f, BitConverter.ToSingle(got, (int)(blockOff * 20)));               // the all-zero block: scale 0 ...
        Assert.Equal(0x88, got[blockOff * 20 + 4]);                                        // ... nibbles 8
    }

    public static TheoryData<VulkanKvStorage, int, int, int, int, int> Shapes()
    {
        var d = new TheoryData<VulkanKvStorage, int, int, int, int, int>();
        foreach (var st in new[] { VulkanKvStorage.Q8_0, VulkanKvStorage.Q4_0, VulkanKvStorage.Q8_0K_Q4_0V, VulkanKvStorage.Q4_0K_Q8_0V })
        {
            d.Add(st, 37, 290, 14, 2, 64);    // hd64 prefill at an offset (253), partial tiles
            d.Add(st, 64, 64, 14, 2, 64);     // hd64 fresh prefill
            d.Add(st, 45, 333, 32, 8, 128);   // hd128 at an offset, group of 4
            d.Add(st, 1, 200, 32, 8, 128);    // one decode row
        }
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void PerTokenKernel_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var q8 = AttentionF32Kernel.Create(device, spvDir, storage);
        using var f32 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void ScalarFlash_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        Skip.If(seqQ < 2, "flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var q8 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None, kvStorage: storage);
        using var f32 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None, kvStorage: VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void ScalarGqaFlash_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        Skip.If(seqQ < 2, "GQA flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, storage);
        var f32 = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.F32);
        Skip.If(q8 is null || f32 is null || !q8.CanDispatch(headDim, numHeads, numKvHeads), "no scalar GQA flash for this shape / device");
        using var _q8 = q8!; using var _f32 = f32!;
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.Q8_0, 3, 300, 14, 2, 64)]
    [InlineData(VulkanKvStorage.Q8_0, 4, 900, 32, 8, 128)]
    [InlineData(VulkanKvStorage.Q4_0, 3, 300, 14, 2, 64)]
    [InlineData(VulkanKvStorage.Q4_0, 4, 900, 32, 8, 128)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 3, 300, 14, 2, 64)]
    [InlineData(VulkanKvStorage.Q4_0K_Q8_0V, 4, 900, 32, 8, 128)]
    public void SmallRowSplitKv_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir, storage);
        var f32 = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir, VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        Skip.If(q8 is null || f32 is null || !VulkanSplitKvRowsAttentionKernel.CanServe(seqQ, off, numHeads), "small-row split-KV unavailable");
        using var _q8 = q8!; using var _f32 = f32!;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, numHeads, numKvHeads, headDim, off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, numHeads, numKvHeads, headDim, off));
    }

    /// <summary>Split-KV decode (the dense chain's decode at depth): the GQA-grouped v2 pass (its own block loader) and the per-head pass.</summary>
    [SkippableTheory]
    [InlineData(VulkanKvStorage.Q8_0, 900, 14, 2, 64, true)]
    [InlineData(VulkanKvStorage.Q4_0, 900, 14, 2, 64, true)]
    [InlineData(VulkanKvStorage.Q4_0, 2600, 32, 8, 128, true)]
    [InlineData(VulkanKvStorage.Q4_0, 777, 32, 8, 128, false)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 900, 14, 2, 64, true)]
    [InlineData(VulkanKvStorage.Q4_0K_Q8_0V, 2600, 32, 8, 128, true)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 777, 32, 8, 128, false)]
    public void SplitKvDecode_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqKv, int numHeads, int numKvHeads, int headDim, bool gqa)
    {
        _storage = storage;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var f32 = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var blk = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage);
        f32.PreferGqa = gqa; blk.PreferGqa = gqa;
        Skip.If(gqa && !blk.UsesGqa(numHeads, numKvHeads, headDim), "GQA-grouped split pass unavailable");
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads));
        int pos = seqKv - 1;
        ExactCase(device, 1, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => blk.Record(c, q, k, v, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void CoopmatFlash_ReadsBlockKvInPlace_MatchesF16TwinOnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        Skip.If(seqQ < 2, "flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, storage);
        var f16 = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, VulkanKvStorage.F16);
        Skip.If(q8 is null || f16 is null || headDim > q8.SupportedMaxHeadDim, "no coopmat flash on this device / shape");
        using var _q8 = q8!; using var _f16 = f16!;
        int off = seqKv - seqQ;
        CoopmatCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f16.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void CoopmatGqaFlash_ReadsBlockKvInPlace_MatchesF16TwinOnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        _storage = storage;
        Skip.If(seqQ < 2, "GQA flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, storage);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16);
        Skip.If(q8 is null || f16 is null || !q8.CanDispatch(headDim, numHeads, numKvHeads), "no coopmat GQA flash on this device / shape");
        using var _q8 = q8!; using var _f16 = f16!;
        int off = seqKv - seqQ;
        CoopmatCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f16.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    // ── Flint sink attention (#1884: the Q4_0 and mixed twins of the sink kernels; Flint selects them with DOTLLM_VK_KV) ──

    private static VulkanDevice.Buffer Sinks(VulkanDevice device, int numHeads)
    {
        var b = device.Allocate(numHeads * 4L);
        device.Upload(Enumerable.Range(0, numHeads).Select(h => 0.5f + 0.01f * h).ToArray(), b);
        return b;
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.Q4_0, 2600, 513, true)]
    [InlineData(VulkanKvStorage.Q4_0, 777, 0, false)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 2600, 513, true)]
    [InlineData(VulkanKvStorage.Q4_0K_Q8_0V, 4000, 0, true)]
    public void SinkSplitKvDecode_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqKv, int window, bool gqa)
    {
        _storage = storage;
        const int numHeads = 32, numKvHeads = 4, headDim = 128;   // a Flint layer
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var f32 = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var blk = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage);
        f32.PreferGqa = gqa; blk.PreferGqa = gqa;
        Skip.If(gqa && !blk.UsesGqa(numHeads, numKvHeads, headDim), "GQA-grouped split pass unavailable");
        using var bS = Sinks(device, numHeads);
        int pos = seqKv - 1, kvStart = window > 0 ? Math.Max(0, pos - window + 1) : 0;
        ExactCase(device, 1, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => blk.Record(c, q, k, v, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart));
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.Q4_0, 100, 612, 512, 0)]
    [InlineData(VulkanKvStorage.Q4_0, 77, 1205, 1128, 513)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 100, 612, 512, 0)]
    [InlineData(VulkanKvStorage.Q4_0K_Q8_0V, 77, 1205, 1128, 513)]
    public void SinkFlashPrefill_ReadsBlockKvInPlace_MatchesF16TwinOnDequantisedRows(VulkanKvStorage storage, int seqQ, int seqKv, int posOffset, int window)
    {
        _storage = storage;
        const int numHeads = 32, numKvHeads = 4, headDim = 128;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var blk = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, storage, sinks: true, splitQ: null);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F16, sinks: true, splitQ: null);
        Skip.If(blk is null || f16 is null || !blk.CanDispatch(headDim, numHeads, numKvHeads), "no coopmat GQA sink flash on this device");
        using var _b = blk!; using var _f = f16!;
        using var bS = Sinks(device, numHeads);
        CoopmatCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => blk.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOffset, slidingWindow: window, sinks: bS),
            (c, q, k, v, o) => f16.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOffset, slidingWindow: window, sinks: bS));
    }

    [SkippableTheory]
    [InlineData(VulkanKvStorage.Q4_0, 20, 0)]
    [InlineData(VulkanKvStorage.Q8_0K_Q4_0V, 40, 16)]
    [InlineData(VulkanKvStorage.Q4_0K_Q8_0V, 33, 0)]
    public void ReferenceSinkKernel_ReadsBlockKvInPlace_MatchesF32OnDequantisedRows(VulkanKvStorage storage, int seqKv, int window)
    {
        _storage = storage;
        const int numHeads = 32, numKvHeads = 4, headDim = 128;
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var bS = Sinks(device, numHeads);
        int pos = seqKv - 1;
        ExactCase(device, 1, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => gpt.RecordAttentionSinks(c, q, k, v, o, bS, 1, seqKv, numHeads, numKvHeads, headDim, pos, window, kvStorage: storage),
            (c, q, k, v, o) => gpt.RecordAttentionSinks(c, q, k, v, o, bS, 1, seqKv, numHeads, numKvHeads, headDim, pos, window, kvStorage: VulkanKvStorage.F32));
    }
}
