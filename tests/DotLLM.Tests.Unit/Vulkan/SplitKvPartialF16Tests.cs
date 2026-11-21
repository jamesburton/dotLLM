using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;
using static DotLLM.Tests.Unit.Vulkan.VulkanQ8KvAttentionKernelTests;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1926 E4: the v2 grouped split-KV pass storing its O partials as F16 (normalized by the split's running sum; the merge
/// multiplies the sum back in) against the F32 control on the same kernel and buffers.
/// </summary>
/// <remarks>
/// <para>
/// Tolerance, stated up front: the F16 arm may differ from the F32 control by at most <b>1e-3 of the output amplitude</b>. A
/// normalized partial is a convex combination of V rows, so |o_s| &lt;= max|v|; F16 rounds it to 2^-11 relative (4.9e-4), and the
/// merge is a weighted average of those, so the output error is bounded by about 4.9e-4 x max|o|. Anything larger is a layout,
/// stride or scale bug.
/// </para>
/// <para>
/// The fixtures are built to fail the two ways an F16 partial can go wrong: scores of magnitude ~20 whose per-split maxima differ by
/// more than 1 (an F16 running max would move exp() weights by ~1%), and a long equal-weight split over |v| ~ 3e4 (an un-normalized
/// F16 accumulator would overflow 65504).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class SplitKvPartialF16Tests
{
    private readonly ITestOutputHelper _out;
    public SplitKvPartialF16Tests(ITestOutputHelper output) => _out = output;

    private const float Tolerance = 1e-3f;

    public static TheoryData<VulkanKvStorage, int, int, int, int> Cases() => new()
    {
        { VulkanKvStorage.F16, 32, 8, 128, 1000 },
        { VulkanKvStorage.F16, 32, 8, 128, 8192 },
        { VulkanKvStorage.Q4_0, 32, 8, 128, 4097 },
        { VulkanKvStorage.Q8_0, 32, 4, 128, 2600 },
        { VulkanKvStorage.F16, 14, 2, 64, 1000 },
        { VulkanKvStorage.Q4_0, 96, 8, 128, 777 },
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void F16Partials_MatchF32Control_WithinStatedTolerance(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        var rng = new Random(0x19264 + seqKv + numHeads);
        int stride = numKvHeads * headDim;
        // Large scores (Rand's per-block amplitudes reach 13x): per-split running maxima of order 10-30 that differ between splits,
        // where an F16 running max would be off by up to 2^-6 and move the exp() weights by ~1%.
        float[] q = Rand(rng, numHeads * headDim, 1.5f);
        float[] k = Rand(rng, seqKv * stride, 0.5f), v = Rand(rng, seqKv * stride, 0.5f);
        var (f16, f32) = RunBoth(storage, numHeads, numKvHeads, headDim, seqKv, q, k, v);
        AssertClose(f16, f32, $"{storage} {numHeads}/{numKvHeads}/{headDim} seqKv {seqKv}");
    }

    /// <summary>A long equal-weight split over |v| ~ 3e4: the un-normalized split sum (~4e6) would overflow F16; the normalized one cannot.</summary>
    [SkippableFact]
    public void F16Partials_LargeValuesEqualWeights_NoOverflow()
    {
        const int numHeads = 32, numKvHeads = 8, headDim = 128, seqKv = 8192;
        var rng = new Random(0x19265);
        float[] q = new float[numHeads * headDim];   // all scores 0: every weight in a split is exactly 1
        float[] k = Rand(rng, seqKv * numKvHeads * headDim, 0.5f);
        float[] v = new float[seqKv * numKvHeads * headDim];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(2e4 + 1e4 * rng.NextDouble()) * (i % 3 == 0 ? -1f : 1f);
        var (f16, f32) = RunBoth(VulkanKvStorage.F16, numHeads, numKvHeads, headDim, seqKv, q, k, v);
        Assert.All(f16, x => Assert.True(float.IsFinite(x), "F16 partial overflowed"));
        AssertClose(f16, f32, "large-V equal-weight");
    }

    private void AssertClose(float[] f16, float[] f32, string label)
    {
        float amp = f32.Max(MathF.Abs), worst = 0;
        int differ = 0;
        for (int i = 0; i < f16.Length; i++)
        {
            float d = MathF.Abs(f16[i] - f32[i]);
            if (d > 0) differ++;
            worst = MathF.Max(worst, d);
        }
        _out.WriteLine($"{label}: max |F16 - F32| = {worst / amp:E2} of amplitude {amp:G4}; {differ}/{f16.Length} outputs differ");
        Assert.True(amp > 1e-3f, "degenerate reference");
        Assert.True(differ > 0, "F16 arm is bit-identical to F32: the F16 partials did not engage");
        Assert.True(worst <= Tolerance * amp, $"{label}: F16 partials differ from the F32 control by {worst / amp:E2} of the amplitude (limit {Tolerance:E0})");
    }

    private static (float[] F16, float[] F32) RunBoth(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int seqKv,
        float[] q, float[] k, float[] v)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kern = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage);
        kern.PreferGqa = true;
        Skip.IfNot(kern.UsesGqa2(numHeads, numKvHeads, headDim), "v2 pass unavailable on this device");
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads));
        byte[] kb = SplitKvV2LoaderBitIdentityTests.Encode(storage.KeyFormat(), k), vb = SplitKvV2LoaderBitIdentityTests.Encode(storage.ValueFormat(), v);
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK = device.Allocate(kb.Length);
        using var bV = device.Allocate(vb.Length);
        using var o16 = device.Allocate(q.Length * 4L);
        using var o32 = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kb, bK); device.Upload(vb, bV);
        kern.PreferPartF16 = true;
        kern.Launch(bQ, bK, bV, o16, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);
        Assert.True(kern.LastPartF16, "the F16 partial pipeline was not bound");
        kern.PreferPartF16 = false;
        kern.Launch(bQ, bK, bV, o32, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);
        Assert.False(kern.LastPartF16);
        float[] a = new float[q.Length], b = new float[q.Length];
        device.Download(o16, a); device.Download(o32, b);
        return (a, b);
    }
}
