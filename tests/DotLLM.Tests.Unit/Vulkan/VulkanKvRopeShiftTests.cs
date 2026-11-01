using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// RoPE position-delta shift of cached K rows on the device (#1531 step 2). Oracle: the CPU model's own K at the target
/// position (<c>RoPE.Execute</c> on the same raw K), plus an exact double-precision rotation. The shifted rows are accepted
/// when they are about as close to the exact target as the model's own K is (the model rounds its angle in float at large
/// positions, so no shift can beat that), measured per head as max-abs error / head norm.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvRopeShiftTests
{
    private const int Rows = 24, StartRow = 7, MaxLen = 64;
    private readonly ITestOutputHelper _out;
    public VulkanKvRopeShiftTests(ITestOutputHelper output) => _out = output;

    public static IEnumerable<object[]> Cases()
    {
        foreach (var storage in new[] { VulkanKvStorage.F32, VulkanKvStorage.F16 })
        {
            // (storage, type, kvHeads, headDim, ropeDim, theta, srcPos0, delta)
            yield return new object[] { storage, RoPEType.NeoX, 8, 128, 128, 5_000_000f, 40, 1500 };    // Qwen3-4B shape, forward shift
            yield return new object[] { storage, RoPEType.NeoX, 8, 128, 128, 5_000_000f, 1600, -1500 }; // backwards shift
            yield return new object[] { storage, RoPEType.Norm, 2, 64, 32, 10_000f, 100, 700 };         // interleaved + partial rotary
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void ShiftK_MatchesModelKeyAtTargetPosition(VulkanKvStorage storage, RoPEType type, int kvHeads, int headDim, int ropeDim,
        float theta, int srcPos0, int delta)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int stride = kvHeads * headDim;
        var rng = new Random(1531);
        var raw = new float[Rows * stride];
        for (int i = 0; i < raw.Length; i++) raw[i] = (float)(rng.NextDouble() * 2 - 1);

        var src = RotateCpu(raw, srcPos0, kvHeads, headDim, ropeDim, theta, type);
        var model = RotateCpu(raw, srcPos0 + delta, kvHeads, headDim, ropeDim, theta, type);
        var exact = RotateExact(raw, srcPos0 + delta, kvHeads, headDim, ropeDim, theta, type);

        using var dev = VulkanDevice.Create();
        using var cache = new VulkanKvCache(dev, new[] { stride }, MaxLen, storage);
        using var kernel = RopeShiftKvKernel.Create(dev, spvDir);
        // Sentinel everywhere, then the source rows at StartRow: rows outside the shifted range must come back unchanged.
        var sentinel = new float[MaxLen * stride];
        for (int i = 0; i < sentinel.Length; i++) sentinel[i] = 0.25f + (i % 13) * 0.01f;
        cache.WriteRows(0, 0, MaxLen, sentinel, sentinel);
        cache.WriteRows(0, StartRow, Rows, src, src);
        var kBefore = new float[MaxLen * stride];
        var vBefore = new float[MaxLen * stride];
        cache.ReadRows(0, 0, MaxLen, kBefore, vBefore);

        kernel.Shift(cache, 0, StartRow, Rows, delta, kvHeads, headDim, ropeDim, theta,
            type == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm);

        var kAfter = new float[MaxLen * stride];
        var vAfter = new float[MaxLen * stride];
        cache.ReadRows(0, 0, MaxLen, kAfter, vAfter);

        for (int i = 0; i < MaxLen * stride; i++)
        {
            Assert.Equal(vBefore[i], vAfter[i]);
            if (i / stride < StartRow || i / stride >= StartRow + Rows)
                Assert.Equal(kBefore[i], kAfter[i]);
        }

        double worstShiftVsExact = 0, worstModelVsExact = 0, worstShiftVsModel = 0;
        for (int r = 0; r < Rows; r++)
            for (int h = 0; h < kvHeads; h++)
            {
                int o = r * stride + h * headDim;
                double norm = 0;
                for (int d = 0; d < headDim; d++) norm += (double)exact[o + d] * exact[o + d];
                norm = Math.Sqrt(norm);
                double eS = 0, eM = 0, eSM = 0;
                for (int d = 0; d < headDim; d++)
                {
                    double got = kAfter[(StartRow + r) * stride + h * headDim + d];
                    eS = Math.Max(eS, Math.Abs(got - exact[o + d]));
                    eM = Math.Max(eM, Math.Abs(model[o + d] - exact[o + d]));
                    eSM = Math.Max(eSM, Math.Abs(got - model[o + d]));
                }
                worstShiftVsExact = Math.Max(worstShiftVsExact, eS / norm);
                worstModelVsExact = Math.Max(worstModelVsExact, eM / norm);
                worstShiftVsModel = Math.Max(worstShiftVsModel, eSM / norm);
            }
        _out.WriteLine($"{storage} {type} delta={delta}: shift-vs-exact {worstShiftVsExact:E2}, model-vs-exact {worstModelVsExact:E2}, shift-vs-model {worstShiftVsModel:E2} (of head norm)");

        // F16 storage rounds the stored row (relative eps 2^-11 per element, twice: source + result); the F32 path only floats.
        double floor = storage == VulkanKvStorage.F16 ? 2.5e-3 : 1e-6;
        Assert.True(worstShiftVsExact <= 2 * worstModelVsExact + floor,
            $"shifted K is {worstShiftVsExact:E2} from exact vs the model's own {worstModelVsExact:E2} (floor {floor:E1})");
    }

    // Row r is rotated as position pos0 + r by the CPU model's float table.
    private static float[] RotateCpu(float[] raw, int pos0, int kvHeads, int headDim, int ropeDim, float theta, RoPEType type)
    {
        var k = (float[])raw.Clone();
        int rows = k.Length / (kvHeads * headDim);
        int maxPos = pos0 + rows + 1;
        var cos = new float[maxPos * ropeDim / 2];
        var sin = new float[maxPos * ropeDim / 2];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cos, sin);
        var pos = new int[rows];
        for (int i = 0; i < rows; i++) pos[i] = pos0 + i;
        RoPE.Execute(Array.Empty<float>(), k, pos, 0, kvHeads, headDim, ropeDim, cos, sin, type);
        return k;
    }

    // Row r sits at position pos0 + r and is rotated by the exact angle position * theta^(-2i/ropeDim) in double.
    private static float[] RotateExact(float[] raw, int pos0, int kvHeads, int headDim, int ropeDim, float theta, RoPEType type)
    {
        var o = (float[])raw.Clone();
        int rows = raw.Length / (kvHeads * headDim);
        int half = ropeDim / 2;
        for (int r = 0; r < rows; r++)
            for (int h = 0; h < kvHeads; h++)
            {
                int b = r * kvHeads * headDim + h * headDim;
                for (int i = 0; i < half; i++)
                {
                    double ang = (pos0 + r) * Math.Pow((double)theta, -2.0 * i / ropeDim);
                    double c = Math.Cos(ang), s = Math.Sin(ang);
                    int i0 = type == RoPEType.NeoX ? b + i : b + 2 * i;
                    int i1 = type == RoPEType.NeoX ? b + i + half : b + 2 * i + 1;
                    double v0 = raw[i0], v1 = raw[i1];
                    o[i0] = (float)(v0 * c - v1 * s);
                    o[i1] = (float)(v0 * s + v1 * c);
                }
            }
        return o;
    }
}
