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

    /// <summary>
    /// The default-on F16 partials on the paths the shipped models take beyond plain causal decode (final review of #1926, I2): TurboQuant
    /// (tq3 / tq4, codes read in place), gpt-oss attention sinks (the merge adds exp(sink - M) to the denominator), and a sliding window
    /// (kvStart &gt; 0, so the splits cover a clipped span). Each runs the same kernel with F16 and F32 partials on the same buffers; the F32
    /// arm is held to its oracle by VulkanTurboQuantInPlaceKernelTests / VulkanSinkSplitKvAttentionTests, so 1e-3 of the amplitude here
    /// bounds the F16 arm against that oracle too.
    /// </summary>
    public static TheoryData<string, int, int, int, int, int> PathCases() => new()
    {
        { "tq4", 32, 8, 128, 4096, 0 },
        { "tq3", 32, 8, 128, 2600, 0 },
        { "tq4", 14, 2, 64, 1000, 0 },
        { "sinks", 64, 8, 64, 700, 128 },     // gpt-oss geometry: sinks + sliding window
        { "sinks", 32, 4, 128, 4000, 0 },     // Flint full layer with sinks
        { "window", 32, 8, 128, 4097, 513 },  // sliding window only: kvStart = 3584
    };

    [SkippableTheory]
    [MemberData(nameof(PathCases))]
    public void F16Partials_TqSinksAndWindow_MatchF32Control(string path, int numHeads, int numKvHeads, int headDim, int seqKv, int window)
    {
        var rng = new Random(0x19266 + seqKv + numHeads + path.Length);
        float[] q = Rand(rng, numHeads * headDim, 1.5f);
        float[] k = Rand(rng, seqKv * numKvHeads * headDim, 0.5f), v = Rand(rng, seqKv * numKvHeads * headDim, 0.5f);
        float[]? sinks = null;
        if (path == "sinks")
        {
            sinks = new float[numHeads];
            for (int h = 0; h < numHeads; h++) sinks[h] = 2f + (float)(rng.NextDouble() * 2 - 1);   // a sink that holds real softmax mass
        }
        int? tqBits = path switch { "tq4" => 4, "tq3" => 3, _ => null };
        var (f16, f32) = RunBoth(tqBits is null ? VulkanKvStorage.F16 : VulkanKvStorage.Tq, numHeads, numKvHeads, headDim, seqKv, q, k, v,
            window, sinks, tqBits, rng);
        AssertClose(f16, f32, $"{path} {numHeads}/{numKvHeads}/{headDim} seqKv {seqKv} window {window}");
    }

    private static (float[] F16, float[] F32) RunBoth(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int seqKv,
        float[] q, float[] k, float[] v, int window = 0, float[]? sinks = null, int? tqBits = null, Random? rng = null)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        DotLLM.Core.Attention.TurboQuantLayout? l = tqBits is int bits
            ? new DotLLM.Core.Attention.TurboQuantLayout(headDim, numKvHeads, bits, DotLLM.Core.Attention.TurboQuantLayout.DefaultNorm) : null;
        using var kern = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage, l, "");
        kern.PreferGqa = true;
        Skip.IfNot(kern.UsesGqa2(numHeads, numKvHeads, headDim), "v2 pass unavailable on this device");
        int pos = seqKv - 1, kvStart = window > 0 ? Math.Max(0, pos - window + 1) : 0;
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv - kvStart, numHeads));
        using var tq = l is { } tl ? TqFixture.Create(device, rng!, tl, seqKv) : null;
        byte[] kb = tq is null ? SplitKvV2LoaderBitIdentityTests.Encode(storage.KeyFormat(), k) : [];
        byte[] vb = tq is null ? SplitKvV2LoaderBitIdentityTests.Encode(storage.ValueFormat(), v) : [];
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK = tq is null ? device.Allocate(kb.Length) : null;
        using var bV = tq is null ? device.Allocate(vb.Length) : null;
        using var bS = sinks is null ? null : device.Allocate(sinks.Length * 4L);
        using var o16 = device.Allocate(q.Length * 4L);
        using var o32 = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ);
        if (bK is not null) { device.Upload(kb, bK); device.Upload(vb, bV!); }
        if (bS is not null) device.Upload(sinks!, bS);
        void Run(VulkanDevice.Buffer o)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            kern.Record(ctx.CommandBuffer, bQ, tq?.K ?? bK!, tq?.V ?? bV!, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos,
                slidingWindow: window, sinks: bS, kvStart: kvStart, tq: tq?.Operands);
            ctx.SubmitAndWait();
        }
        kern.PreferPartF16 = true;
        Run(o16);
        Assert.True(kern.LastPartF16, "the F16 partial pipeline was not bound");
        kern.PreferPartF16 = false;
        Run(o32);
        Assert.False(kern.LastPartF16);
        float[] a = new float[q.Length], b = new float[q.Length];
        device.Download(o16, a); device.Download(o32, b);
        return (a, b);
    }
}
