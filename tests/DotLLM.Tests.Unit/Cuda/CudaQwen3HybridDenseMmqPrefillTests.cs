using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Model-level coverage for issue #490's packed PQ2_0 prefill GEMM on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>: a wide prefill forward (12 tokens — past
/// #485's 8-column GEMV, so every PQ2_0 projection in the forward takes the new kernel, lm_head
/// included) must land where the established paths do.
/// </summary>
/// <remarks>
/// <para>
/// <b>Three arms, one prefix.</b> (1) the prefill in one forward with MMQ on, (2) the same forward
/// with MMQ off — the dequant-to-F16 + cuBLAS path this replaces, (3) the same tokens fed one at a
/// time through the #485 dp4a decode GEMV, which is the W2A8 reference the CPU greedy oracle already
/// pins. (1) is W2A8 like (3) and must not drift from it further than the quantization floor (2)
/// vs (1) measures, and (1) must differ bitwise from (2) — the control proving the new kernel ran
/// rather than silently falling back.
/// </para>
/// <para>
/// <b>Why the drift bound is a multiple of the quantization floor, not a fixed epsilon.</b> Same
/// reasoning as the #485 tests: int8-per-32 rounding is a step function ~20x coarser than half, so
/// the tiny trunk-kernel differences between a batched and a per-token forward flip rounding
/// decisions and the two runs end up with independent quantization-noise realizations whose
/// difference is ~√2 x the floor. The structural checks below are the exact ones.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMmqPrefillTests : IDisposable
{
    // 12 tokens: > 8, so the whole prefill (and the lm_head over 12 rows) takes the MMQ path.
    // Every id is inside the synthetic fixture's 12-token vocabulary.
    private static readonly int[] Prompt = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 11, 0];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseMmqPrefillTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-qwen35-mmq-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        CudaSmallSGemvDispatch.MmqOverride = null;
        CudaSmallSGemvDispatch.MmqTileOverride = null;
        CudaSmallSGemvDispatch.Dp4aOverride = null;
        CudaSmallSGemvDispatch.MaxColumnsOverride = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Mmq_Prefill_MatchesDp4aDecode_OnPq2_0Fixture()
    {
        string ptxDir = CudaPQ2_0MmqDp4aTests.SkipUnlessMmqKernel();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-pq2_0-mmq.gguf"), withMtp: false, pq2_0Projections: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.PQ2_0MmqAvailable, "pq2_0_mmq_dp4a.ptx present but the MMQ module did not load (stale PTX?)");

        RunAndAssert(model, config.VocabSize, "fixture", Prompt);
    }

    /// <summary>
    /// The same arms on the real Bonsai 2 27B checkpoint — the shapes the issue's numbers come from
    /// (hidden 5120, ffn 17408, a 248k-row lm_head), skipped unless the GGUF is present.
    /// </summary>
    [SkippableFact]
    public void Mmq_Prefill_MatchesDp4aDecode_OnRealBonsai2()
    {
        string ptxDir = CudaPQ2_0MmqDp4aTests.SkipUnlessMmqKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "hadamard_fwht.ptx")),
            "hadamard_fwht.ptx not generated (Bonsai 2 is Hadamard-folded)");
        string? path = FindBonsai2Checkpoint();
        Skip.If(path is null, "Bonsai 2 MTP GGUF not found (set DOTLLM_BONSAI2_MTP_GGUF)");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.PQ2_0MmqAvailable, "pq2_0_mmq_dp4a.ptx present but the MMQ module did not load (stale PTX?)");

        int[] prompt = [760, 6511, 314, 9338, 369, 11, 1052, 3070, 262, 1938, 290, 1517];
        RunAndAssert(model, config.VocabSize, "Bonsai 2", prompt);
    }

    private void RunAndAssert(CudaQwen3HybridDenseTransformerModel model, int vocab, string label, int[] prompt)
    {
        float[][] mmq, mmqNoShare, old, singles;
        int mmqLaunches, mmqNoShareLaunches, oldLaunches;
        (int Gemv, int Quantize) counts, countsNoShare;
        try
        {
            CudaSmallSGemvDispatch.Dp4aOverride = true;      // the S = 1 reference is #485's W2A8 GEMV
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
            CudaSmallSGemvDispatch.MmqOverride = true;
            mmq = RunBatched(model, vocab, prompt, out mmqLaunches, out counts);
            // #490 widened SharesDp4aInput to prefill widths: with sharing off every projection
            // re-quantizes its own input, and the logits must be bit-identical — a stale shared int8
            // scratch cannot hide behind the tolerance below.
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = false;
            mmqNoShare = RunBatched(model, vocab, prompt, out mmqNoShareLaunches, out countsNoShare);
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
            CudaSmallSGemvDispatch.MmqOverride = false;
            old = RunBatched(model, vocab, prompt, out oldLaunches, out _);
            CudaSmallSGemvDispatch.MmqOverride = null;
            singles = RunSingles(model, vocab, prompt);
        }
        finally
        {
            CudaSmallSGemvDispatch.MmqOverride = null;
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = null;
        }

        _out.WriteLine($"{label}: prefill launches (mmq, gemv, quantize): sharing on ({mmqLaunches}, {counts.Gemv}, " +
                       $"{counts.Quantize}), off ({countsNoShare.Gemv}, {countsNoShare.Quantize})");
        Assert.True(counts.Quantize <= mmqLaunches + counts.Gemv,
            $"{label}: more quantize launches than projections with input sharing on");
        Assert.Equal(mmqNoShareLaunches + countsNoShare.Gemv, countsNoShare.Quantize);   // sharing off: one quantize each
        Assert.True(BitwiseEqual(mmq, mmqNoShare),
            $"{label}: prefill logits change when dp4a input sharing is turned off (max|diff| " +
            $"{MaxAbsDiff(mmq[prompt.Length - 1], mmqNoShare[prompt.Length - 1]):E3}) — a shared quantized scratch is stale.");

        // Structural: the new kernel ran, the old arm did not use it, and the two arms differ.
        _out.WriteLine($"{label}: MMQ launches per prefill forward: on {mmqLaunches}, off {oldLaunches}");
        Assert.True(mmqLaunches > 0, $"{label}: no MMQ launch in an S={prompt.Length} forward — the path did not engage");
        Assert.Equal(0, oldLaunches);
        Assert.True(!BitwiseEqual(mmq, old),
            $"{label}: prefill logits are bit-identical with the MMQ path on and off — the kernel did not run.");

        int last = prompt.Length - 1;
        float floor = MaxAbsDiff(old[last], mmq[last]);          // W2A8-vs-F16 quantization floor at this S
        float mmqDrift = MaxAbsDiff(singles[last], mmq[last]);
        float oldDrift = MaxAbsDiff(singles[last], old[last]);
        _out.WriteLine($"{label}: last row max|diff| — MMQ vs F16 prefill {floor:E3}, " +
                       $"MMQ vs dp4a singles {mmqDrift:E3}, F16 prefill vs dp4a singles {oldDrift:E3}");
        for (int r = 0; r < prompt.Length; r++)
            Assert.All(mmq[r], v => Assert.True(float.IsFinite(v), $"{label}: row {r} has a non-finite logit"));

        Assert.True(mmqDrift <= 2f * MathF.Max(floor, oldDrift) + 1e-2f,
            $"{label}: MMQ prefill drifts {mmqDrift:E3} from the dp4a per-token reference — beyond 2x the " +
            $"quantization floor (MMQ vs F16 {floor:E3}, F16 vs singles {oldDrift:E3})");

        (int best, float gap) = ArgMaxWithGap(singles[last]);
        if (gap > 2 * mmqDrift)
            Assert.Equal(best, ArgMax(mmq[last]));
        else
            _out.WriteLine($"{label}: last-row top-2 gap {gap:E3} within 2x the observed drift — argmax not asserted");
    }

    private static float[][] RunBatched(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt,
        out int mmqLaunches, out (int Gemv, int Quantize) dp4aCounts)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        int[] positions = new int[prompt.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        model.ResetDp4aLaunchCounts();
        using var logits = model.Forward(prompt, positions, deviceId: -1, kv);
        mmqLaunches = model.Pq2_0MmqLaunchCount;
        dp4aCounts = model.Dp4aLaunchCounts;
        Assert.Equal(prompt.Length, logits.Shape[0]);
        float[][] rows = new float[prompt.Length][];
        for (int r = 0; r < rows.Length; r++) rows[r] = Row(logits, r, vocab);
        return rows;
    }

    private static float[][] RunSingles(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        float[][] rows = new float[prompt.Length][];
        for (int i = 0; i < prompt.Length; i++)
        {
            using var logits = model.Forward([prompt[i]], [i], deviceId: -1, kv);
            rows[i] = Row(logits, 0, vocab);
        }
        return rows;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int c = 0; c < a.Length; c++) m = MathF.Max(m, MathF.Abs(a[c] - b[c]));
        return m;
    }

    private static bool BitwiseEqual(float[][] a, float[][] b)
    {
        for (int r = 0; r < a.Length; r++)
            for (int c = 0; c < a[r].Length; c++)
                if (BitConverter.SingleToInt32Bits(a[r][c]) != BitConverter.SingleToInt32Bits(b[r][c]))
                    return false;
        return true;
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

    private static unsafe float[] Row(DotLLM.Core.Tensors.ITensor logits, int row, int vocab)
    {
        float* p = (float*)logits.DataPointer + (long)row * vocab;
        return new ReadOnlySpan<float>(p, vocab).ToArray();
    }

    private static string? FindBonsai2Checkpoint()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_MTP_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string repo = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--ProCreations--Ternary-Bonsai-2-27B-MTP", "snapshots");
        if (!Directory.Exists(repo)) return null;
        foreach (string snapshot in Directory.EnumerateDirectories(repo))
        {
            string[] hits = Directory.GetFiles(snapshot, "*MTP*.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }
}
