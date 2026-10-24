using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The six IQ dequant-to-F16 shaders (issue #621: IQ prefill dequantises each weight matrix once into F16 scratch and runs the
/// F16 coopmat GEMM) against the CPU dequantisers. Blocks are fully random bytes (every grid index, sign index, scale, delta bit);
/// the shaders do the same fp32 arithmetic as the CPU and then round to half, so the result must equal <c>(Half)cpuValue</c> exactly.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanIqDequantF16KernelTests
{
    public static TheoryData<QuantizationType, int> Cases()
    {
        var d = new TheoryData<QuantizationType, int>();
        foreach (var qt in new[]
                 {
                     QuantizationType.IQ1_S, QuantizationType.IQ2_XXS, QuantizationType.IQ2_XS,
                     QuantizationType.IQ2_S, QuantizationType.IQ3_XXS, QuantizationType.IQ3_S,
                 })
            foreach (int blocks in new[] { 1, 7, 64 })
                d.Add(qt, blocks);
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public unsafe void Launch_MatchesCpuDequantiser_RoundedToHalf(QuantizationType qt, int blocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int blockBytes = (int)QuantFormat.TryGetInfo(qt)!.Value.BlockBytes;
        int elements = blocks * 256;
        var rng = new Random(0x621 + (int)qt * 131 + blocks);
        byte[] bytes = new byte[blocks * blockBytes];
        rng.NextBytes(bytes);
        for (int b = 0; b < blocks; b++)   // finite, small d (fp16 at offset 0)
            BitConverter.TryWriteBytes(new Span<byte>(bytes, b * blockBytes, 2), (Half)((rng.NextSingle() * 2f - 1f) * 0.01f));

        float[] expected = new float[elements];
        fixed (byte* p = bytes)
            Dequantize.ToFloat32((nint)p, elements, qt, expected);

        using var device = VulkanDevice.Create();
        using var src = device.Allocate(((long)bytes.Length + 3) & ~3L);
        using var dst = device.Allocate((long)elements * sizeof(ushort));
        device.Upload(new ReadOnlySpan<byte>(bytes), src);
        Launch(qt, device, spvDir, src, dst, blocks);

        float[] raw = new float[elements / 2];
        device.Download(dst, raw);
        ReadOnlySpan<ushort> actual = MemoryMarshal.Cast<float, ushort>(raw);

        int mismatches = 0;
        string first = "";
        for (int i = 0; i < elements; i++)
        {
            ushort want = BitConverter.HalfToUInt16Bits((Half)expected[i]);
            if (actual[i] != want && mismatches++ == 0)
                first = $"element {i}: gpu 0x{actual[i]:X4} ({(float)BitConverter.UInt16BitsToHalf(actual[i])}) vs cpu {expected[i]} (0x{want:X4})";
        }
        Assert.True(mismatches == 0, $"{qt} blocks={blocks}: {mismatches}/{elements} elements differ; {first}");
    }

    private static void Launch(QuantizationType qt, VulkanDevice device, string spvDir,
                               VulkanDevice.Buffer src, VulkanDevice.Buffer dst, int blocks)
    {
        switch (qt)
        {
            case QuantizationType.IQ1_S: { using var k = Iq1SDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            case QuantizationType.IQ2_XXS: { using var k = Iq2XxsDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            case QuantizationType.IQ2_XS: { using var k = Iq2XsDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            case QuantizationType.IQ2_S: { using var k = Iq2SDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            case QuantizationType.IQ3_XXS: { using var k = Iq3XxsDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            case QuantizationType.IQ3_S: { using var k = Iq3SDequantF16Kernel.Create(device, spvDir); k.Launch(src, dst, blocks); break; }
            default: throw new ArgumentOutOfRangeException(nameof(qt));
        }
    }
}
