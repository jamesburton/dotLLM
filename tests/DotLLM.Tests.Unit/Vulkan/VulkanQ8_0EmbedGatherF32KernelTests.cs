using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>q8_0_embed_gather_f32</c> — the device-resident Q8_0
/// token-embedding gather (issue #352) that replaces the F32 row
/// <c>vkCmdCopyBuffer</c> when the embedding table is left in its quantized
/// layout on the device.
/// </summary>
/// <remarks>
/// <para>
/// The bar is <b>bit-exact</b> against <c>Dequantize.DequantizeQ8_0Scalar</c>:
/// dequant is a single <c>scale * qs</c> multiply, exactly rounded on both
/// sides, and the F32-upload path this replaces produced exactly those bytes.
/// Any drift means the shader is not reproducing the host dequant.
/// </para>
/// <para>
/// The id sequences below are deliberately out-of-order, repeated and include
/// the last row: an identity sequence <c>0,1,2,…</c> would make a dropped
/// per-token row offset indistinguishable from a correct gather, which is the
/// exact bug class that makes a mis-specified embedding path silently corrupt
/// every token.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ8_0EmbedGatherF32KernelTests
{
    private const int BlockBytes = 34;
    private const int GroupSize = 32;

    /// <summary>Builds a deterministic pseudo-random Q8_0 table of [vocab, hidden].</summary>
    private static byte[] BuildTable(int vocab, int hidden, int seed)
    {
        int blocksPerRow = hidden / GroupSize;
        byte[] blob = new byte[(long)vocab * blocksPerRow * BlockBytes];
        var rng = new Random(seed);
        int off = 0;
        for (int r = 0; r < vocab; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                // Scales span several orders of magnitude (and sign) so a wrong
                // fp16 read cannot pass by coincidence.
                var d = (Half)(float)((rng.NextDouble() * 2.0 - 1.0) * Math.Pow(10, rng.Next(-3, 2)));
                MemoryMarshal.Write(blob.AsSpan(off, 2), in d);
                for (int i = 0; i < GroupSize; i++)
                    blob[off + 2 + i] = (byte)(sbyte)rng.Next(-128, 128);
                off += BlockBytes;
            }
        }
        return blob;
    }

    private static unsafe float[] CpuGather(byte[] blob, int hidden, ReadOnlySpan<int> ids)
    {
        int blocksPerRow = hidden / GroupSize;
        long rowBytes = (long)blocksPerRow * BlockBytes;
        float[] expected = new float[ids.Length * hidden];
        fixed (byte* p = blob)
        {
            for (int t = 0; t < ids.Length; t++)
                Dequantize.ToFloat32(
                    (nint)(p + ids[t] * rowBytes), hidden,
                    DotLLM.Core.Configuration.QuantizationType.Q8_0,
                    expected.AsSpan(t * hidden, hidden));
        }
        return expected;
    }

    [SkippableTheory]
    // hidden = 32 (one block/row), 256 (exact workgroup), 288 (workgroup + tail),
    // 2048 (Llama-3.2-1B), 3072 (multi-workgroup).
    [InlineData(64, 32)]
    [InlineData(64, 256)]
    [InlineData(64, 288)]
    [InlineData(512, 2048)]
    [InlineData(97, 3072)]
    public void Gather_MatchesCpuDequant_BitExact(int vocab, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q8_0_embed_gather_f32.spv")),
            "q8_0_embed_gather_f32.spv not compiled (glslc / Vulkan SDK required).");

        byte[] blob = BuildTable(vocab, hidden, seed: 0x352 + hidden);

        // Out-of-order, repeated, first and last row — see remarks.
        int[] ids = [vocab - 1, 0, 7 % vocab, vocab - 1, 1, vocab / 2, 0, 3 % vocab];

        float[] expected = CpuGather(blob, hidden, ids);

        using var device = VulkanDevice.Create();
        using var kernel = Q8_0EmbedGatherF32Kernel.Create(device, spvDir);
        using var table = device.Allocate(blob.Length);
        using var idBuf = device.Allocate((long)ids.Length * sizeof(int));
        using var dst = device.Allocate((long)ids.Length * hidden * sizeof(float));
        device.Upload(blob.AsSpan(), table);
        device.Upload(MemoryMarshal.AsBytes(ids.AsSpan()), idBuf);

        kernel.Launch(table, idBuf, dst, ids.Length, hidden, vocab);

        float[] actual = new float[ids.Length * hidden];
        device.Download(dst, actual);

        for (int i = 0; i < actual.Length; i++)
            Assert.True(
                BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"q8_0 embed gather mismatch at token {i / hidden} (id={ids[i / hidden]}), col {i % hidden} " +
                $"(vocab={vocab}, hidden={hidden}): cpu={expected[i]:G9} gpu={actual[i]:G9}.");
    }

    /// <summary>
    /// Single-token decode — the hot path. Every row in the table is gathered
    /// on its own, so a row-stride error shows up regardless of which row
    /// happens to be first.
    /// </summary>
    [SkippableFact]
    public void Gather_SingleToken_EveryRow_MatchesCpuDequant()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q8_0_embed_gather_f32.spv")),
            "q8_0_embed_gather_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int Vocab = 33, Hidden = 128;
        byte[] blob = BuildTable(Vocab, Hidden, seed: 0xBEEF);

        using var device = VulkanDevice.Create();
        using var kernel = Q8_0EmbedGatherF32Kernel.Create(device, spvDir);
        using var table = device.Allocate(blob.Length);
        using var idBuf = device.Allocate(sizeof(int));
        using var dst = device.Allocate((long)Hidden * sizeof(float));
        device.Upload(blob.AsSpan(), table);

        float[] actual = new float[Hidden];
        for (int row = 0; row < Vocab; row++)
        {
            int[] ids = [row];
            device.Upload(MemoryMarshal.AsBytes(ids.AsSpan()), idBuf);
            kernel.Launch(table, idBuf, dst, 1, Hidden, Vocab);
            device.Download(dst, actual);

            float[] expected = CpuGather(blob, Hidden, ids);
            for (int i = 0; i < Hidden; i++)
                Assert.True(
                    BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"row {row} col {i}: cpu={expected[i]:G9} gpu={actual[i]:G9}.");
        }
    }

    /// <summary>
    /// Rejects a hidden size that is not a multiple of the 32-element Q8_0
    /// group — the row stride would be undefined.
    /// </summary>
    [SkippableFact]
    public void Record_RejectsNonMultipleOf32Hidden()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q8_0_embed_gather_f32.spv")),
            "q8_0_embed_gather_f32.spv not compiled (glslc / Vulkan SDK required).");

        using var device = VulkanDevice.Create();
        using var kernel = Q8_0EmbedGatherF32Kernel.Create(device, spvDir);
        using var table = device.Allocate(4096);
        using var idBuf = device.Allocate(sizeof(int));
        using var dst = device.Allocate(4096);

        Assert.Throws<ArgumentException>(
            () => kernel.Launch(table, idBuf, dst, 1, hidden: 33, vocabSize: 4));
    }
}
