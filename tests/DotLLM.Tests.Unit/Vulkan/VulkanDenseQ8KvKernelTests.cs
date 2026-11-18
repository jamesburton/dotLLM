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

    private static (float[] Q, float[] K, float[] V, byte[] Kq, byte[] Vq, float[] Kd, float[] Vd) Data(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int seed)
    {
        var rng = new Random(seed);
        int kvElems = seqKv * numKvHeads * headDim;
        float[] q = Rand(rng, seqQ * numHeads * headDim, 0.5f), k = Rand(rng, kvElems, 0.5f), v = Rand(rng, kvElems, 0.5f);
        byte[] kq = QuantizeRef(k), vq = QuantizeRef(v);
        return (q, k, v, kq, vq, DequantizeRef(kq), DequantizeRef(vq));
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
        AssertQ8Budget(a, raw);
    }

    /// <summary>
    /// Controls: the Q8 output is NOT the attention over the unquantised rows (a kernel that read stale F32 or ignored the quantisation
    /// cannot pass), and is within the Q8_0 budget as a relative L2 error. A per-element bound is not used: early prefill rows attend to a
    /// handful of keys, so one outlier-heavy V row sets their output, and its quantisation error is larger than the average.
    /// </summary>
    private static void AssertQ8Budget(float[] a, float[] raw)
    {
        float maxAbs = raw.Max(MathF.Abs), worst = 0;
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - raw[i]));
            num += (double)(a[i] - raw[i]) * (a[i] - raw[i]);
            den += (double)raw[i] * raw[i];
        }
        Assert.True(worst > 1e-5f * maxAbs, $"Q8 output is indistinguishable from the unquantised attention (worst {worst:G3})");
        Assert.True(Math.Sqrt(num / den) < 0.03, $"Q8 relative L2 error {Math.Sqrt(num / den):G3} exceeds the 3% Q8_0 budget");
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
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { num += (double)(a[i] - raw[i]) * (a[i] - raw[i]); den += (double)raw[i] * raw[i]; }
        Assert.True(Math.Sqrt(num / den) < 0.03, $"Q8 relative L2 error {Math.Sqrt(num / den):G3} exceeds the 3% Q8_0 budget");
    }

    public static TheoryData<int, int, int, int, int> Shapes => new()
    {
        { 37, 290, 14, 2, 64 },    // hd64 prefill at an offset (253), partial tiles
        { 64, 64, 14, 2, 64 },     // hd64 fresh prefill
        { 45, 333, 32, 8, 128 },   // hd128 at an offset, group of 4
        { 1, 200, 32, 8, 128 },    // one decode row
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void PerTokenKernel_ReadsQ8InPlace_MatchesF32OnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var q8 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.Q8_0);
        using var f32 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void ScalarFlash_ReadsQ8InPlace_MatchesF32OnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Skip.If(seqQ < 2, "flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var q8 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None, kvStorage: VulkanKvStorage.Q8_0);
        using var f32 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None, kvStorage: VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void ScalarGqaFlash_ReadsQ8InPlace_MatchesF32OnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Skip.If(seqQ < 2, "GQA flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.Q8_0);
        var f32 = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.F32);
        Skip.If(q8 is null || f32 is null || !q8.CanDispatch(headDim, numHeads, numKvHeads), "no scalar GQA flash for this shape / device");
        using var _q8 = q8!; using var _f32 = f32!;
        int off = seqKv - seqQ;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }

    [SkippableTheory]
    [InlineData(3, 300, 14, 2, 64)]
    [InlineData(4, 900, 32, 8, 128)]
    public void SmallRowSplitKv_ReadsQ8InPlace_MatchesF32OnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir, VulkanKvStorage.Q8_0);
        var f32 = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir, VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        Skip.If(q8 is null || f32 is null || !VulkanSplitKvRowsAttentionKernel.CanServe(seqQ, off, numHeads), "small-row split-KV unavailable");
        using var _q8 = q8!; using var _f32 = f32!;
        ExactCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, numHeads, numKvHeads, headDim, off),
            (c, q, k, v, o) => f32.Record(c, q, k, v, o, seqQ, numHeads, numKvHeads, headDim, off));
    }

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void CoopmatFlash_ReadsQ8InPlace_MatchesF16TwinOnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Skip.If(seqQ < 2, "flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, VulkanKvStorage.Q8_0);
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
    public void CoopmatGqaFlash_ReadsQ8InPlace_MatchesF16TwinOnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim)
    {
        Skip.If(seqQ < 2, "GQA flash serves prefill rows");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.Q8_0);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16);
        Skip.If(q8 is null || f16 is null || !q8.CanDispatch(headDim, numHeads, numKvHeads), "no coopmat GQA flash on this device / shape");
        using var _q8 = q8!; using var _f16 = f16!;
        int off = seqKv - seqQ;
        CoopmatCase(device, seqQ, seqKv, numHeads, numKvHeads, headDim,
            (c, q, k, v, o) => q8.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off),
            (c, q, k, v, o) => f16.Record(c, q, k, v, o, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: off));
    }
}
