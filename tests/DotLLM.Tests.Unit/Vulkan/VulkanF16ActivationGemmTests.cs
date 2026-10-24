using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// F16-activation GEMM variants (Q4_K / Q5_K / Q6_K / Q8_0) and the F16-output SwiGLU. The GEMM template already stages B into
/// LDS as F16, so for activations that were rounded to F16 once the F16-B kernel and the F32-B kernel run the SAME arithmetic:
/// results must match EXACTLY (no tolerance), including on the split-K path. A wrong B stride, half-pair order or element
/// count would show immediately.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanF16ActivationGemmTests
{
    public enum Quant { Q4K, Q5K, Q6K, Q8_0 }

    private static (int BlockBytes, int Group, int DOffset) Layout(Quant q) => q switch
    {
        Quant.Q4K => (QuantFormat.Q4_KBlockBytes, 256, 0),     // d at 0, dmin at 2
        Quant.Q5K => (QuantFormat.Q5_KBlockBytes, 256, 0),
        Quant.Q6K => (QuantFormat.Q6_KBlockBytes, 256, 208),   // d at the end of the block
        Quant.Q8_0 => (QuantFormat.Q8_0BlockBytes, 32, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(q)),
    };

    [SkippableTheory]
    [InlineData(Quant.Q4K, 128, 2560, 9216)]    // 20 tiles -> split-K x4 on the F16 twin
    [InlineData(Quant.Q4K, 512, 2560, 9216)]    // plain F16 path (80 tiles)
    [InlineData(Quant.Q4K, 100, 1000, 2560)]    // ragged n and m, small grid
    [InlineData(Quant.Q5K, 512, 2560, 9216)]
    [InlineData(Quant.Q5K, 128, 2560, 4608)]
    [InlineData(Quant.Q6K, 512, 2560, 9216)]
    [InlineData(Quant.Q6K, 128, 2560, 4608)]
    [InlineData(Quant.Q8_0, 512, 2560, 9216)]
    [InlineData(Quant.Q8_0, 100, 1000, 2560)]
    public void F16Activation_MatchesF32KernelOnF16RoundedInput(Quant quant, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4KGemmCoopmatKernel.IsSupportedOn(device, spvDir), "coopmat kernels unsupported here.");

        var (blockBytes, group, dOffset) = Layout(quant);
        var rng = new Random(0xF16 + (int)quant * 7919 + n + m + k);
        long blocks = (long)m * (k / group);
        byte[] w = new byte[blocks * blockBytes];
        rng.NextBytes(w);
        for (long b = 0; b < blocks; b++)
        {
            int o = (int)(b * blockBytes) + dOffset;
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)(0.0001f + rng.NextSingle() * 0.0003f));
            if (quant is Quant.Q4K or Quant.Q5K)
                BitConverter.TryWriteBytes(w.AsSpan(o + 2, 2), (Half)(0.0001f + rng.NextSingle() * 0.0003f));
        }

        float[] b32 = new float[(long)n * k];
        ushort[] b16 = new ushort[(long)n * k];
        for (int i = 0; i < b32.Length; i++)
        {
            float raw = rng.NextSingle() * 2f - 1f;
            var h = (Half)raw;
            // F32 buffer holds the UNROUNDED value on purpose: the F32-B kernel then rounds it with the shader's own float16_t
            // cast, so this also proves that cast matches the producer's (round-to-nearest-even) for ordinary values.
            b32[i] = raw;
            b16[i] = BitConverter.HalfToUInt16Bits(h);
        }

        using var bufW = device.AllocateDeviceLocal(((long)w.Length + 3) & ~3L);
        using var bufB32 = device.AllocateDeviceLocal((long)b32.Length * 4);
        using var bufB16 = device.AllocateDeviceLocal(((long)b16.Length * 2 + 3) & ~3L);
        using var c32 = device.AllocateDeviceLocal((long)n * m * 4);
        using var c16 = device.AllocateDeviceLocal((long)n * m * 4);
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b32, bufB32);
        device.Upload(new ReadOnlySpan<byte>(System.Runtime.InteropServices.MemoryMarshal.AsBytes(b16.AsSpan()).ToArray()), bufB16);

        CoopmatF16Gemm? f16;
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            switch (quant)
            {
                case Quant.Q4K:
                {
                    using var k32 = MatMulQ4KGemmCoopmatKernel.Create(device, spvDir);
                    f16 = k32.F16Activation; Skip.If(f16 is null, "F16 path unavailable");
                    k32.Record(ctx.CommandBuffer, bufW, bufB32, c32, m, k, n);
                    f16!.Record(ctx.CommandBuffer, bufW, bufB16, c16, m, k, n);
                    ctx.SubmitAndWait();
                    break;
                }
                case Quant.Q5K:
                {
                    using var k32 = MatMulQ5KGemmCoopmatKernel.Create(device, spvDir);
                    f16 = k32.F16Activation; Skip.If(f16 is null, "F16 path unavailable");
                    k32.Record(ctx.CommandBuffer, bufW, bufB32, c32, m, k, n);
                    f16!.Record(ctx.CommandBuffer, bufW, bufB16, c16, m, k, n);
                    ctx.SubmitAndWait();
                    break;
                }
                case Quant.Q6K:
                {
                    using var k32 = MatMulQ6KGemmCoopmatKernel.Create(device, spvDir);
                    f16 = k32.F16Activation; Skip.If(f16 is null, "F16 path unavailable");
                    k32.Record(ctx.CommandBuffer, bufW, bufB32, c32, m, k, n);
                    f16!.Record(ctx.CommandBuffer, bufW, bufB16, c16, m, k, n);
                    ctx.SubmitAndWait();
                    break;
                }
                default:
                {
                    using var k32 = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir);
                    f16 = k32.F16Activation; Skip.If(f16 is null, "F16 path unavailable (blocked variant not selected)");
                    k32.Record(ctx.CommandBuffer, bufW, bufB32, c32, m, k, n);
                    f16!.Record(ctx.CommandBuffer, bufW, bufB16, c16, m, k, n);
                    ctx.SubmitAndWait();
                    break;
                }
            }
        }

        float[] want = new float[(long)n * m];
        float[] got = new float[(long)n * m];
        device.Download(c32, want);
        device.Download(c16, got);
        double maxDiff = 0;
        int bad = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = Math.Abs((double)want[i] - got[i]);
            if (d > maxDiff) maxDiff = d;
            if (d != 0) bad++;
        }
        Assert.True(bad == 0, $"{quant} n={n} m={m} k={k}: {bad}/{want.Length} outputs differ from the F32-B kernel (max |diff| {maxDiff:E3}); expected bit-identical.");
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(1026)]
    [InlineData(9216 * 3)]
    public void SwiGluF16Out_RoundsLikeTheGemmStaging(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        Skip.IfNot(swiglu.HasF16Out, "F16-output SwiGLU unavailable.");

        var rng = new Random(n);
        float[] gate = new float[n], up = new float[n];
        for (int i = 0; i < n; i++) { gate[i] = (rng.NextSingle() - 0.5f) * 12f; up[i] = (rng.NextSingle() - 0.5f) * 4f; }

        using var bg = device.AllocateDeviceLocal((long)n * 4);
        using var bu = device.AllocateDeviceLocal((long)n * 4);
        using var bo32 = device.AllocateDeviceLocal((long)n * 4);
        using var bo16 = device.AllocateDeviceLocal((long)n * 2);
        device.Upload(gate, bg);
        device.Upload(up, bu);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            swiglu.Record(ctx.CommandBuffer, bg, bu, bo32, n);
            swiglu.RecordF16Out(ctx.CommandBuffer, bg, bu, bo16, n);
            ctx.SubmitAndWait();
        }

        float[] f32 = new float[n];
        float[] packed = new float[(n + 1) / 2];   // n halves = n/2 words; read back as float bits
        device.Download(bo32, f32);
        device.Download(bo16, packed);
        byte[] raw = new byte[n * 2];
        Buffer.BlockCopy(packed, 0, raw, 0, raw.Length);   // n even in the cases below
        for (int i = 0; i < n; i++)
        {
            ushort bits = BitConverter.ToUInt16(raw, i * 2);
            Half got = BitConverter.UInt16BitsToHalf(bits);
            Half want = (Half)f32[i];
            Assert.True(got == want || (Half.IsNaN(got) && Half.IsNaN(want)),
                $"element {i}: F16 out {(float)got} != half({f32[i]}) = {(float)want}");
        }
    }
}
