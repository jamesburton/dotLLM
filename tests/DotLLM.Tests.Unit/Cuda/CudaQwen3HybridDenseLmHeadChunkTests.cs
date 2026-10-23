using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1099: an all-logits forward over more than <see cref="CudaSmallSGemvDispatch.MaxColumns"/>
/// rows must not dequantize the whole LM head to F16. On
/// <see cref="CudaQwen3HybridDenseTransformerModel"/> a PQ2_0 head runs in row chunks through the packed
/// multi-row GEMV instead (the dp4a one by default, #482's F16-activation one with dp4a off).
/// </summary>
/// <remarks>
/// <para>
/// <b>Numerics (synthetic PQ2_0 fixture).</b> The trunk is identical in both arms — at 20 rows every layer
/// projection takes the same dequant + cuBLAS path either way — so the arms differ only in the head's
/// kernel. The reference arm is the pre-#1099 whole-head dequant
/// (<see cref="CudaWideProjection.BoundedLmHeadOverride"/> = false). With dp4a off, the chunked F16-activation
/// GEMV must match it at the existing multi-column bar (the #482 model tests' 2e-3 + 1e-2·|ref|). With dp4a on,
/// the bar is the existing dp4a-vs-F16-activation one (<c>CudaPQ2_0GemvDp4aTests</c>: relative RMS ≤ 2e-2 per
/// row, argmax wherever the reference's top-2 gap is clear). No tolerance is new or wider.
/// </para>
/// <para>
/// <b>Memory (real Bonsai 2 27B).</b> After a 20-row and a 9-row all-logits forward the F16 dequant scratch
/// is no larger than the model's widest layer projection, where it used to hold the whole 2.5 GiB head.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseLmHeadChunkTests : IDisposable
{
    private const int WideRows = 20;
    private const int VerifyRows = 9;

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseLmHeadChunkTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-lmhead-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        CudaWideProjection.BoundedLmHeadOverride = null;
        CudaSmallSGemvDispatch.Dp4aOverride = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Synthetic PQ2_0 fixture: a 20-row all-logits forward with the chunked head matches the whole-head
    /// dequant reference, for both packed small-row tiers, and the chunked route provably ran.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChunkedHead_MatchesWholeHeadDequant_OnPq2_0Fixture(bool dp4a)
    {
        string ptxDir = dp4a ? CudaPQ2_0GemvDp4aTests.SkipUnlessDp4aKernel() : CudaPQ2_0GemvMultiTests.SkipUnlessMultiKernel();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-pq2_0-head.gguf"), withMtp: false, contextLength: 64, pq2_0Projections: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        int[] tokens = Enumerable.Range(0, WideRows).Select(i => (i * 5 + 1) % config.VocabSize).ToArray();

        int[] poison = tokens.Select(t => (t + 7) % config.VocabSize).ToArray();

        // Chunked arm first, each arm after a forward over different tokens: a chunk that writes the wrong
        // rows must not be masked by the previous arm's logits still sitting in the device buffer.
        CudaSmallSGemvDispatch.Dp4aOverride = dp4a;
        CudaWideProjection.BoundedLmHeadOverride = false;
        RunAllLogits(model, config.VocabSize, poison);
        Assert.Equal((0, 0), model.LmHeadLaunchCounts);
        CudaWideProjection.BoundedLmHeadOverride = null;
        float[][] chunked = RunAllLogits(model, config.VocabSize, tokens);
        int chunks = model.LmHeadLaunchCounts.RowChunks;

        CudaWideProjection.BoundedLmHeadOverride = false;
        RunAllLogits(model, config.VocabSize, poison);
        float[][] reference = RunAllLogits(model, config.VocabSize, tokens);
        Assert.Equal(chunks, model.LmHeadLaunchCounts.RowChunks);
        _out.WriteLine($"dp4a={dp4a}: LM-head row chunks {chunks}");
        Assert.True(chunks == (WideRows + CudaSmallSGemvDispatch.MaxColumns - 1) / CudaSmallSGemvDispatch.MaxColumns,
            $"expected the {WideRows}-row head in {CudaSmallSGemvDispatch.MaxColumns}-row chunks, got {chunks} chunks");
        Assert.False(BitwiseEqual(reference, chunked), "chunked and whole-head logits are bit-identical — the chunked kernel did not run");

        if (dp4a)
            AssertDp4aBar(reference, chunked);
        else
            AssertMultiColumnBar(reference, chunked);
    }

    /// <summary>
    /// Real Bonsai 2 27B: after a 20-row and a 9-row all-logits forward the F16 weight-dequant scratch stays at or
    /// below the widest layer projection (it held the 2.5 GiB head before #1099), and the logits are finite.
    /// </summary>
    [SkippableFact]
    public void AllLogitsForwards_AllocateNoHeadSizedScratch_OnRealBonsai2()
    {
        string ptxDir = CudaPQ2_0GemvMultiTests.SkipUnlessMultiKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "hadamard_fwht.ptx")), "hadamard_fwht.ptx not generated (Bonsai 2 is Hadamard-folded)");
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BONSAI_PQ2_0_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "Bonsai 2 PQ2_0 GGUF not found (set DOTLLM_BONSAI_PQ2_0_GGUF)");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        long widestLayerBytes = WidestLayerProjectionElems(gguf, config.NumLayers) * sizeof(ushort);
        GgufTensorDescriptor head = gguf.TensorsByName["output.weight"];
        Assert.Equal(QuantizationType.PQ2_0, head.QuantizationType);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        long free0 = FreeBytes();
        int[] wide = Enumerable.Range(0, WideRows).Select(i => 760 + 37 * i).ToArray();
        int[] verify = Enumerable.Range(0, VerifyRows).Select(i => 1052 + 11 * i).ToArray();

        foreach ((int[] ids, int start, string label) in new[] { (wide, 0, "20-row prefill"), (verify, WideRows, "9-row verify") })
        {
            using var logits = model.Forward(ids, Enumerable.Range(start, ids.Length).ToArray(), deviceId: -1, kv);
            Assert.Equal(ids.Length, logits.Shape[0]);
            AssertFinite(logits, ids.Length, config.VocabSize, label);
            long scratch = model.DequantScratchF16WeightBytes;
            _out.WriteLine($"{label}: dequant scratch {scratch >> 20} MiB (widest layer projection {widestLayerBytes >> 20} MiB, " +
                           $"whole head {head.Shape.ElementCount * sizeof(ushort) >> 20} MiB); free VRAM {free0 >> 20} -> {FreeBytes() >> 20} MiB; " +
                           $"head chunks so far {model.LmHeadLaunchCounts.RowChunks}");
            Assert.True(scratch <= widestLayerBytes,
                $"{label}: the F16 dequant scratch is {scratch:N0} B, above the widest layer projection {widestLayerBytes:N0} B — the LM head was expanded");
        }
        Assert.True(model.LmHeadLaunchCounts.RowChunks >= 3 + 2, "the LM head did not take the chunked route on both forwards");
    }

    private static long WidestLayerProjectionElems(GgufFile gguf, int numLayers)
    {
        long widest = 0;
        foreach (GgufTensorDescriptor t in gguf.Tensors)
        {
            if (t.Shape.Rank != 2 || !t.Name.StartsWith("blk.", StringComparison.Ordinal)) continue;
            int dot = t.Name.IndexOf('.', 4);
            if (int.Parse(t.Name.AsSpan(4, dot - 4)) >= numLayers) continue;   // the MTP block is not a trunk layer
            widest = Math.Max(widest, t.Shape.ElementCount);
        }
        return widest;
    }

    private static float[][] RunAllLogits(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] tokens)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        using var logits = model.Forward(tokens, Enumerable.Range(0, tokens.Length).ToArray(), deviceId: -1, kv);
        Assert.Equal(tokens.Length, logits.Shape[0]);
        float[][] rows = new float[tokens.Length][];
        for (int r = 0; r < rows.Length; r++) rows[r] = Row(logits, r, vocab);
        return rows;
    }

    // The #482 model tests' bar for the multi-column GEMV against the dequant + cuBLAS path.
    private void AssertMultiColumnBar(float[][] expected, float[][] actual)
    {
        for (int r = 0; r < expected.Length; r++)
        {
            float maxDiff = 0;
            for (int c = 0; c < expected[r].Length; c++)
            {
                float a = expected[r][c], b = actual[r][c], d = MathF.Abs(a - b);
                maxDiff = MathF.Max(maxDiff, d);
                Assert.True(d <= 2e-3f + 1e-2f * MathF.Abs(a), $"row {r} logit {c}: reference {a}, got {b}, |diff| {d:E3}");
            }
            _out.WriteLine($"multi-column chunks vs whole-head dequant: row {r} max|diff| {maxDiff:E3}");
            Assert.Equal(ArgMax(expected[r]), ArgMax(actual[r]));
        }
    }

    // CudaPQ2_0GemvDp4aTests.Dp4a_AgreesWithF16ActivationKernel_OnRealisticData's bar.
    private void AssertDp4aBar(float[][] expected, float[][] actual)
    {
        for (int r = 0; r < expected.Length; r++)
        {
            double se = 0, sr = 0, maxDiff = 0;
            for (int c = 0; c < expected[r].Length; c++)
            {
                double f = expected[r][c], d = actual[r][c];
                Assert.True(float.IsFinite(actual[r][c]), $"row {r} logit {c} is not finite");
                se += (f - d) * (f - d);
                sr += f * f;
                maxDiff = Math.Max(maxDiff, Math.Abs(f - d));
            }
            double relRms = Math.Sqrt(se / sr);
            (int best, float gap) = ArgMaxWithGap(expected[r]);
            _out.WriteLine($"dp4a chunks vs whole-head dequant: row {r} rel RMS {relRms:E3}, max|diff| {maxDiff:E3}, top-2 gap {gap:E3}");
            Assert.True(relRms <= 2e-2, $"row {r}: relative RMS error {relRms:E3} vs the whole-head dequant exceeds 2e-2");
            if (gap > 2 * maxDiff)
                Assert.Equal(best, ArgMax(actual[r]));
        }
    }

    private static unsafe void AssertFinite(DotLLM.Core.Tensors.ITensor logits, int rows, int vocab, string label)
    {
        float* p = (float*)logits.DataPointer;
        for (long i = 0; i < (long)rows * vocab; i++)
            Assert.True(float.IsFinite(p[i]), $"{label}: logit {i} is not finite");
    }

    private static long FreeBytes()
    {
        CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
        return (long)free;
    }

    private static (int Index, float Gap) ArgMaxWithGap(float[] row)
    {
        int best = 0;
        float second = float.NegativeInfinity;
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > row[best]) { second = row[best]; best = i; }
            else if (row[i] > second) second = row[i];
        }
        return (best, row[best] - second);
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    private static bool BitwiseEqual(float[][] a, float[][] b)
    {
        for (int r = 0; r < a.Length; r++)
            for (int c = 0; c < a[r].Length; c++)
                if (BitConverter.SingleToInt32Bits(a[r][c]) != BitConverter.SingleToInt32Bits(b[r][c]))
                    return false;
        return true;
    }

    private static unsafe float[] Row(DotLLM.Core.Tensors.ITensor logits, int row, int vocab)
    {
        float* p = (float*)logits.DataPointer + (long)row * vocab;
        return new ReadOnlySpan<float>(p, vocab).ToArray();
    }
}
