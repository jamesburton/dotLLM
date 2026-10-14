using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// <see cref="VulkanSplitKvRowsAttentionKernel"/> (the speculative-verify rows variant of the split-KV decode attention) must reproduce, for every row,
/// the 1-row <see cref="VulkanSplitKvAttentionKernel"/> decode step at that row's own length BIT FOR BIT: same split count, same chunk boundaries, same arithmetic.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSplitKvRowsAttentionKernelTests
{
    // Shapes chosen to discriminate: GQA; batches whose rows have DIFFERENT split counts (16 heads: cap 16 splits, ceil(kv / 16) below it for
    // kv < 256, so 109 + 1..5 keys = 110..114 gives 7,7,7,8,8); a ragged last split; a context past one 256-key tile per split.
    [SkippableTheory]
    [InlineData(20, 5, 16, 4, 256)]
    [InlineData(100, 5, 16, 4, 256)]
    [InlineData(109, 5, 16, 4, 256)]
    [InlineData(1000, 3, 32, 8, 256)]
    [InlineData(2040, 5, 32, 8, 256)]   // qwen4exp-like head dim, near the dense limit
    [InlineData(777, 2, 4, 2, 64)]
    public void EveryRow_IsBitIdentical_ToTheOneRowDecodeKernelAtThatLength(int positionOffset, int rows, int numHeads, int numKvHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Assert.True(VulkanSplitKvRowsAttentionKernel.CanServe(rows, positionOffset, numHeads));

        int seqKv = positionOffset + rows;
        var rng = new Random(0x909 + positionOffset * 31 + rows);
        float[] q = Rand(rng, rows * numHeads * headDim), k = Rand(rng, seqKv * numKvHeads * headDim), v = Rand(rng, seqKv * numKvHeads * headDim);

        using var device = VulkanDevice.Create();
        using var one = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var rowsKernel = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("rows SPIR-V missing");
        using var bq = device.Allocate((long)q.Length * 4);
        using var bk = device.Allocate((long)k.Length * 4);
        using var bv = device.Allocate((long)v.Length * 4);
        using var bOut = device.Allocate((long)q.Length * 4);
        device.Upload(q.AsSpan(), bq); device.Upload(k.AsSpan(), bk); device.Upload(v.AsSpan(), bv);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            rowsKernel.Record(ctx.CommandBuffer, bq, bk, bv, bOut, rows, numHeads, numKvHeads, headDim, positionOffset);
            ctx.SubmitAndWait();
        }
        Assert.Equal(1, rowsKernel.RecordCount);   // the path under test ran
        var got = new float[q.Length];
        device.Download(bOut, got);

        // Reference: the shipping 1-row kernel, once per row, over exactly that row's keys (seqKv = pos + 1).
        int rowLen = numHeads * headDim;
        for (int r = 0; r < rows; r++)
        {
            int pos = positionOffset + r;
            using var bqr = device.Allocate((long)rowLen * 4);
            using var bor = device.Allocate((long)rowLen * 4);
            device.Upload(q.AsSpan(r * rowLen, rowLen), bqr);
            one.Launch(bqr, bk, bv, bor, 1, pos + 1, numHeads, numKvHeads, headDim, pos);
            var want = new float[rowLen];
            device.Download(bor, want);
            for (int i = 0; i < rowLen; i++)
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[r * rowLen + i]),
                    $"row {r} (pos {pos}) element {i}: 1-row {want[i]:R} vs rows kernel {got[r * rowLen + i]:R}");
        }
    }


    // #996: sink layers (MiMo-V2 / gpt-oss). Row r must equal the 1-row sink decode the model runs (kvStart = max(0, pos - W + 1), splits of the clipped
    // span, the sink joining the merge denominator) bit for bit. Positions straddle the window edge (pos - W + 1 crossing 0), include a window
    // wider than the context, and 64 heads (4-split cap, the MiMo shape) next to a 16-head shape.
    [SkippableTheory]
    [InlineData(126, 4, 16, 4, 64, 128)]    // rows 126..129: the window starts clipping inside the batch
    [InlineData(200, 4, 64, 8, 192, 128)]   // MiMo SWA geometry, window fully inside the context
    [InlineData(127, 3, 64, 8, 192, 128)]
    [InlineData(300, 2, 16, 4, 64, 4096)]   // window wider than the context: no clipping, sink still joins
    [InlineData(1000, 4, 32, 8, 128, 128)]
    public void SinkRows_AreBitIdentical_ToTheOneRowSinkDecodeAtThatLength(int positionOffset, int rows, int numHeads, int numKvHeads, int headDim, int window)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Assert.True(VulkanSplitKvRowsAttentionKernel.CanServe(rows, positionOffset, numHeads, window, hasSinks: true));

        int seqKv = positionOffset + rows;
        var rng = new Random(0x996 + positionOffset * 31 + rows);
        float[] q = Rand(rng, rows * numHeads * headDim), k = Rand(rng, seqKv * numKvHeads * headDim), v = Rand(rng, seqKv * numKvHeads * headDim);
        var sinkVals = new float[numHeads];
        for (int h = 0; h < numHeads; h++) sinkVals[h] = (float)(rng.NextDouble() * 6 - 3);   // comparable to the scores so the sink moves the output

        using var device = VulkanDevice.Create();
        using var one = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var rowsKernel = VulkanSplitKvRowsAttentionKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("rows SPIR-V missing");
        using var bq = device.Allocate((long)q.Length * 4);
        using var bk = device.Allocate((long)k.Length * 4);
        using var bv = device.Allocate((long)v.Length * 4);
        using var bOut = device.Allocate((long)q.Length * 4);
        using var bSink = device.Allocate((long)numHeads * 4);
        device.Upload(q.AsSpan(), bq); device.Upload(k.AsSpan(), bk); device.Upload(v.AsSpan(), bv); device.Upload(sinkVals.AsSpan(), bSink);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            rowsKernel.Record(ctx.CommandBuffer, bq, bk, bv, bOut, rows, numHeads, numKvHeads, headDim, positionOffset, window, bSink);
            ctx.SubmitAndWait();
        }
        var got = new float[q.Length];
        device.Download(bOut, got);

        int rowLen = numHeads * headDim;
        int distinctFromNoSink = 0;
        for (int r = 0; r < rows; r++)
        {
            int pos = positionOffset + r;
            using var bqr = device.Allocate((long)rowLen * 4);
            using var bor = device.Allocate((long)rowLen * 4);
            using var bor2 = device.Allocate((long)rowLen * 4);
            device.Upload(q.AsSpan(r * rowLen, rowLen), bqr);
            int kvStart = Math.Max(0, pos - window + 1);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                one.Record(ctx.CommandBuffer, bqr, bk, bv, bor, 1, pos + 1, numHeads, numKvHeads, headDim,
                    positionOffset: pos, slidingWindow: window, sinks: bSink, kvStart: kvStart);
                ctx.SubmitAndWait();
            }
            one.Launch(bqr, bk, bv, bor2, 1, pos + 1, numHeads, numKvHeads, headDim, pos, window);   // same window, NO sink: the control
            var want = new float[rowLen]; var noSink = new float[rowLen];
            device.Download(bor, want); device.Download(bor2, noSink);
            for (int i = 0; i < rowLen; i++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[r * rowLen + i]),
                    $"row {r} (pos {pos}) element {i}: 1-row sink decode {want[i]:R} vs rows kernel {got[r * rowLen + i]:R}");
                if (want[i] != noSink[i]) distinctFromNoSink++;
            }
        }
        Assert.True(distinctFromNoSink > rows * rowLen / 2, "the sink must change the output (a control that could not disagree proves nothing)");
    }

    [SkippableFact]
    public void CanServe_RefusesRowsThatWouldNotSplitAndOutOfRangeCounts()
    {
        Assert.False(VulkanSplitKvRowsAttentionKernel.CanServe(1, 500, 16));   // one row is the decode kernel's job
        Assert.False(VulkanSplitKvRowsAttentionKernel.CanServe(VulkanSplitKvRowsAttentionKernel.MaxRows + 1, 500, 16));
        Assert.False(VulkanSplitKvRowsAttentionKernel.CanServe(3, 0, 16));     // first row has 1 key: one split, the decode path would use the per-token kernel
        Assert.True(VulkanSplitKvRowsAttentionKernel.CanServe(3, 500, 16));
    }

    private static float[] Rand(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
