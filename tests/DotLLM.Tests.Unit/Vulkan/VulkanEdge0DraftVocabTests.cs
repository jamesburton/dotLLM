using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The Edge0 draft-vocab head (#2013): the row-subset Q8_0 GEMV against the full GEMV (every subset row bit-identical to the same row of the full output, with an UNSORTED
/// id table like <c>draft_vocab.bin</c>), and the compact-argmax-to-token-id mapping against mutants of it.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanEdge0DraftVocabTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    [SkippableTheory]
    [InlineData(4096, 640, 1500)]
    [InlineData(777, 224, 300)]     // odd row count, short K
    public void RowSubsetGemv_IsBitIdenticalToTheFullGemvRows(int rows, int k, int subset)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device does not advertise VK_KHR_shader_integer_dot_product.");
        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var full = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing.");
        using var sub = MatMulQ8_0MmvqRowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq_rows.spv missing (glslc native/vulkan/shaders).");

        var rng = new Random(0x2012 + rows);
        float[] w = new float[rows * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        byte[] wq = QuantizeRows(w, rows, k);
        // unsorted, unique ids (a shuffled subset), the shape of draft_vocab.bin
        int[] ids = Enumerable.Range(0, rows).OrderBy(_ => rng.Next()).Take(subset).ToArray();
        float[] idBits = ids.Select(BitConverter.Int32BitsToSingle).ToArray();

        using var bufW = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * 4);
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var yFull = device.Allocate((long)rows * 4);
        using var ySub = device.Allocate((long)subset * 4);
        using var bufIds = device.Allocate((long)subset * 4);
        device.Upload(new ReadOnlySpan<byte>(wq), bufW);
        device.Upload(x, bufX);
        device.Upload(idBits, bufIds);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            full.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, yFull, rows, k);
            sub.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, ySub, bufIds, subset, k, rows);
            ctx.SubmitAndWait();
        }
        float[] a = new float[rows], b = new float[subset];
        device.Download(yFull, a);
        device.Download(ySub, b);
        int bad = 0, sortedMismatch = 0;
        int[] sorted = ids.OrderBy(i => i).ToArray();
        for (int i = 0; i < subset; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[ids[i]]) != BitConverter.SingleToInt32Bits(b[i])) bad++;
            if (BitConverter.SingleToInt32Bits(a[sorted[i]]) != BitConverter.SingleToInt32Bits(b[i])) sortedMismatch++;   // mutant: the table read as if sorted
        }
        Assert.Equal(0, bad);
        Assert.True(a.Max(MathF.Abs) > 0, "oracle output is all zero - the comparison is vacuous");
        Assert.True(sortedMismatch > subset / 2, "the fixture cannot tell an unsorted id table from a sorted one");
    }

    [Fact]
    public void ArgMaxMapping_ReturnsTheTokenIdOfTheCompactWinner_AndMutantsDie()
    {
        // unsorted ids with a base-order prefix and appended ids, as in draft_vocab.bin
        int[] ids = [90, 3, 41, 7, 255, 12, 1000, 0];
        float[] logits = [0.1f, 0.5f, 0.2f, 0.9f, 0.3f, 0.95f, 0.4f, 0.0f];   // winner at compact row 5 -> token 12
        Assert.Equal(12, Edge0DraftVocabHead.ArgMax(logits, ids));

        // mutants of the mapping must give a different token on this fixture
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        Assert.NotEqual(12, best);                                              // M1: forgot the map (returns the compact row)
        Assert.NotEqual(12, ids[(best + 1) % ids.Length]);                      // M2: off by one
        Assert.NotEqual(12, ids.OrderBy(i => i).ToArray()[best]);               // M3: id table sorted
        Assert.NotEqual(12, ids[ids.Length - 1 - best]);                        // M4: reversed table
        // M5: the winner is in the appended tail -> a map that only covers the first half misses it
        float[] tail = [0.1f, 0.2f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.99f];
        Assert.Equal(0, Edge0DraftVocabHead.ArgMax(tail, ids));
        // exact tie: first in file order
        Assert.Equal(90, Edge0DraftVocabHead.ArgMax([1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f], ids));
        Assert.Throws<ArgumentException>(() => Edge0DraftVocabHead.ArgMax([1f], ids));
    }

    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int rowBytes = k / Q8_0GroupSize * Q8_0BlockBytes;
        var dst = new byte[m * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
            for (int row = 0; row < m; row++)
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
        return dst;
    }
}
