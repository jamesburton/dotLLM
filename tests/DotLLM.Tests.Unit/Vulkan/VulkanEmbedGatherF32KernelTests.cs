using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1174: <see cref="EmbedGatherF32Kernel"/> gathers token rows straight from a packed Q8_0 / Q4_K / Q6_K table. The bar is bit-exact
/// against <see cref="Dequantize.ToFloat32"/>, which is what the widened F32 table held, so keeping the table packed cannot move a logit.
/// Ids are out of order, repeated and include the first and last rows, so a dropped per-token or per-row offset cannot pass.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanEmbedGatherF32KernelTests
{
    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, 41, 256)]
    [InlineData(QuantizationType.Q4_K, 97, 5120)] // Qwen3.8-27B's hidden size: 20 super-blocks per row
    [InlineData(QuantizationType.Q6_K, 41, 256)]
    [InlineData(QuantizationType.Q6_K, 97, 2560)]
    [InlineData(QuantizationType.Q8_0, 64, 288)]
    public void Gather_MatchesCpuDequant_BitExact(QuantizationType format, int vocab, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(EmbedGatherF32Kernel.IsSupported(format, hidden, spvDir), $"{format} gather shader not compiled");

        var rng = new Random(1174 + hidden);
        float[] values = new float[vocab * hidden];
        for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 7);
        byte[] blob = format switch
        {
            QuantizationType.Q4_K => Q4KFixture.QuantizeRows(values, vocab, hidden),
            QuantizationType.Q6_K => Q6KFixture.QuantizeRows(values, vocab, hidden),
            _ => QuantizeRows(values, vocab, hidden, format),
        };
        long rowBytes = Dequantize.RowByteSize(hidden, format);
        Assert.Equal(rowBytes * vocab, blob.Length);

        int[] ids = [vocab - 1, 0, 7 % vocab, vocab - 1, 1, vocab / 2, 0, 3 % vocab];
        float[] expected = new float[ids.Length * hidden];
        unsafe
        {
            fixed (byte* p = blob)
                for (int t = 0; t < ids.Length; t++)
                    Dequantize.ToFloat32((nint)(p + ids[t] * rowBytes), hidden, format, expected.AsSpan(t * hidden, hidden));
        }

        using var device = VulkanDevice.Create();
        using var kernel = EmbedGatherF32Kernel.Create(device, spvDir, format);
        using var table = device.Allocate(blob.Length);
        using var idBuf = device.Allocate((long)ids.Length * sizeof(int));
        using var dst = device.Allocate((long)ids.Length * hidden * sizeof(float));
        device.Upload(blob.AsSpan(), table);
        device.Upload(MemoryMarshal.AsBytes(ids.AsSpan()), idBuf);
        kernel.Launch(table, idBuf, dst, ids.Length, hidden, vocab);
        float[] actual = new float[ids.Length * hidden];
        device.Download(dst, actual);

        for (int i = 0; i < actual.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{format} gather: token {i / hidden} (id {ids[i / hidden]}) col {i % hidden}: cpu {expected[i]:G9} gpu {actual[i]:G9}");
    }

    private static byte[] QuantizeRows(float[] values, int rows, int cols, QuantizationType qt)
    {
        long rowBytes = Dequantize.RowByteSize(cols, qt);
        byte[] bytes = new byte[rowBytes * rows];
        for (int r = 0; r < rows; r++)
            Quantize.FromFloat32(values.AsSpan(r * cols, cols), cols, qt, bytes.AsSpan((int)(r * rowBytes), (int)rowBytes));
        return bytes;
    }
}
