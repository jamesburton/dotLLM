using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <see cref="VulkanQwen4ExpTransformerModel.RequantBank"/> (#900): the MTP head's Q8_0 routed banks re-quantised to Q4_K (gate/up) and Q5_1 (down)
/// so the head fits beside the 79 GiB UD-Q4_K_XL trunk and runs the fused single-token MoE chain. Pure host code, no device.
/// </summary>
public sealed unsafe class Qwen4ExpMtpHeadRequantTests
{
    private static float[] RandomBank(int experts, int m, int k, int seed)
    {
        var rng = new Random(seed);
        var v = new float[(long)experts * m * k];
        for (long i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return v;
    }

    private static (nint Ptr, long Bytes) ToQuant(float[] bank, int experts, int m, int k, QuantizationType qt, List<nint> owned)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        nint mem = (nint)NativeMemory.AlignedAlloc((nuint)(rowBytes * m * experts), 64);
        owned.Add(mem);
        for (long r = 0; r < (long)experts * m; r++)
            Quantize.FromFloat32(bank.AsSpan((int)(r * k), k), k, qt, new Span<byte>((void*)(mem + (nint)(r * rowBytes)), (int)rowBytes));
        return (mem, rowBytes * m * experts);
    }

    private static double Rms(float[] a, float[] b)
    {
        double e = 0, n = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; e += d * d; n += (double)a[i] * a[i]; }
        return Math.Sqrt(e / n);
    }

    private static float[] Dequant(nint ptr, QuantizationType qt, int experts, int m, int k)
    {
        long rowBytes = Dequantize.RowByteSize(k, qt);
        var outv = new float[(long)experts * m * k];
        for (long r = 0; r < (long)experts * m; r++)
            Dequantize.ToFloat32(ptr + (nint)(r * rowBytes), k, qt, outv.AsSpan((int)(r * k), k));
        return outv;
    }

    [Theory]
    [InlineData(QuantizationType.Q4_K, 512)]
    [InlineData(QuantizationType.Q5_1, 640)]
    public void Requant_FromQ8_0_KeepsTheBankWithinTheTargetFormatsError(QuantizationType target, int k)
    {
        const int Experts = 5, M = 7;
        var owned = new List<nint>();
        try
        {
            var original = RandomBank(Experts, M, k, 11);
            var (q8, _) = ToQuant(original, Experts, M, k, QuantizationType.Q8_0, owned);
            var (ptr, qt) = VulkanQwen4ExpTransformerModel.RequantBank(q8, QuantizationType.Q8_0, Experts, M, k, target, owned);

            Assert.Equal(target, qt);
            Assert.NotEqual(q8, ptr);
            double err = Rms(original, Dequant(ptr, qt, Experts, M, k));
            // Q4_K on uniform noise is ~6-9 % RMS, Q5_1 ~2-3 %; the Q8_0 source adds well under 1 %. A layout slip (rows swapped across experts) is ~100 %.
            Assert.True(err < (target == QuantizationType.Q4_K ? 0.12 : 0.05), $"{target}: relative RMS error {err:F4}");
        }
        finally { foreach (nint p in owned) NativeMemory.AlignedFree((void*)p); }
    }

    [Fact]
    public void Requant_LeavesABankAlone_WhenItAlreadyHasTheTargetOrTheWidthDoesNotFit()
    {
        var owned = new List<nint>();
        try
        {
            var bank = RandomBank(2, 3, 640, 5);
            var (q8, _) = ToQuant(bank, 2, 3, 640, QuantizationType.Q8_0, owned);
            // 640 is not a multiple of the Q4_K super-block (256): the bank must come back untouched, not truncated.
            var (p1, t1) = VulkanQwen4ExpTransformerModel.RequantBank(q8, QuantizationType.Q8_0, 2, 3, 640, QuantizationType.Q4_K, owned);
            Assert.Equal(q8, p1);
            Assert.Equal(QuantizationType.Q8_0, t1);
            var (p2, t2) = VulkanQwen4ExpTransformerModel.RequantBank(q8, QuantizationType.Q8_0, 2, 3, 640, QuantizationType.Q8_0, owned);
            Assert.Equal(q8, p2);
            Assert.Equal(QuantizationType.Q8_0, t2);
        }
        finally { foreach (nint p in owned) NativeMemory.AlignedFree((void*)p); }
    }
}
