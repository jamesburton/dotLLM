using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Model-level coverage for issue #482's small-S multi-column PQ2_0 GEMV on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>: an S = 3 forward (what an MTP verify runs)
/// must match three S = 1 decode forwards over the same prefix.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a PQ2_0 fixture.</b> The default <see cref="SyntheticQwen35HybridDenseMtpGguf"/> is
/// all-F32 with every width below 128, so every projection goes through cuBLAS F32 at every
/// seqLen and a test on it would pass with the new kernel deleted. These tests build the
/// <c>pq2_0Projections</c> variant (trunk projections, token_embd and output in PQ2_0, input widths
/// 128/256/512), whose seqLen 2..8 projections reach the multi-column kernel.
/// </para>
/// <para>
/// <b>Three arms, one prefix.</b> The prompt is fed one token at a time in every arm, so the
/// prefix state is identical and independent of the dispatch under test. Then: (1) one S = 3
/// forward on the default dispatch (multi-column), (2) the same S = 3 forward with the multi-column
/// path switched off (<see cref="CudaSmallSGemvDispatch.MaxColumnsOverride"/> = 0, the pre-#482
/// dequant + cuBLAS path), (3) three S = 1 forwards. (1) and (2) must both match (3) within the
/// usual trunk-drift tolerance (the S = 3 and S = 1 forwards also differ in their GDN / attention
/// kernels, not only in the projections), and (1) must differ bitwise from (2) somewhere — proof the
/// multi-column path actually ran rather than silently falling back.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseSmallSGemvTests : IDisposable
{
    private static readonly int[] Prompt = [1, 3, 5, 7];
    private static readonly int[] Continuation = [9, 4, 6];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseSmallSGemvTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-qwen35-smalls-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        CudaSmallSGemvDispatch.MaxColumnsOverride = null;
        CudaSmallSGemvDispatch.Dp4aOverride = null;
        CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Issue #485, dp4a on, synthetic PQ2_0 fixture without a Hadamard fold: an S = 3 forward must
    /// match three S = 1 forwards at the usual trunk-drift bar, plus the structural checks in
    /// <see cref="AssertDp4aStructure"/> (same dp4a coverage at S = 1 and S = 3, input sharing
    /// bit-identical to per-projection quantization, and proof the dp4a kernel ran).
    /// </summary>
    [SkippableFact]
    public void Dp4a_Forward_S3_MatchesThreeS1Forwards_OnPq2_0Fixture() => RunDp4aFixture(folded: false);

    /// <summary>
    /// The same on the PQ2_0 fixture WITH a Hadamard fold declared — Bonsai 2's configuration, where
    /// GDN qkv/gate and attention Q/K/V read the rotated input while alpha/beta read the unrotated
    /// one, so the dp4a input-sharing decisions differ from the unfolded fixture.
    /// </summary>
    [SkippableFact]
    public void Dp4a_Forward_S3_MatchesThreeS1Forwards_OnFoldedPq2_0Fixture() => RunDp4aFixture(folded: true);

    private void RunDp4aFixture(bool folded)
    {
        string ptxDir = CudaPQ2_0GemvDp4aTests.SkipUnlessDp4aKernel();
        if (folded)
            Skip.IfNot(File.Exists(Path.Combine(ptxDir, "hadamard_fwht.ptx")), "hadamard_fwht.ptx not generated");
        string path = folded
            ? SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "qwen35-pq2_0-fold-dp4a.gguf"),
                withMtp: false, pq2_0Projections: true)
            : SyntheticQwen35HybridDenseMtpGguf.Write(
                Path.Combine(_scratch, "qwen35-pq2_0-dp4a.gguf"), withMtp: false, pq2_0Projections: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (folded) config = config with { HadamardFold = SyntheticHadamardFold.For(config) };
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.PQ2_0Dp4aAvailable, "pq2_0_gemv_dp4a.ptx present but the dp4a module did not load (stale PTX?)");

        string label = folded ? "folded fixture" : "fixture";
        Dp4aArms arms = RunDp4aArms(model, config.VocabSize, Prompt, Continuation);
        AssertDp4aStructure(arms, label);

        AssertRowsMatch(arms.Dp4aSingles, arms.Dp4aBatched, absTol: 2e-3f, relTol: 1e-2f, $"{label}: dp4a S=3 vs dp4a 3x S=1");
        float dp4aVsF16 = MaxAbsDiff(arms.F16Singles, arms.Dp4aSingles);
        _out.WriteLine($"{label}: dp4a vs F16 (S=1 arms): max|diff| {dp4aVsF16:E3}");
        AssertArgMaxWhereClear(arms.F16Singles, arms.Dp4aSingles, dp4aVsF16, $"{label}: dp4a vs F16 S=1");
    }

    /// <summary>
    /// The dp4a arms on the real Bonsai 2 27B MTP checkpoint (skipped unless present).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>What is exact and what is not.</b> The structural checks are bit-exact / count-exact (see
    /// <see cref="AssertDp4aStructure"/>): if the shared int8 scratch were ever stale, or a projection
    /// took a different path at S = 3 than at S = 1, they fail and name it.
    /// </para>
    /// <para>
    /// <b>Why the S = 3 vs S = 1 drift bound is relative to the dp4a quantization floor, not the F16
    /// path's drift.</b> The per-column dp4a arithmetic is S-independent, but the trunk is not (GDN /
    /// attention kernels differ between S = 3 and S = 1 by ~1e-5 relative). Int8-per-32 rounding is a
    /// step function ~20x coarser than half: that tiny trunk drift flips some rounding decisions, the
    /// flips perturb the next layer's input, and over ~60 layers the two runs end up with essentially
    /// independent quantization-noise realizations. Their difference is then ~√2 x the dp4a-vs-F16
    /// error (measured on an RTX 3060: 0.091 vs √2 x 0.070 = 0.098), while two half-rounded paths
    /// (F16 drift 0.0046) never decorrelate. So the bound is 2 x the dp4a quantization floor, with
    /// argmax agreement on every row.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void Dp4a_Forward_S3_MatchesThreeS1Forwards_OnRealBonsai2()
    {
        string ptxDir = CudaPQ2_0GemvDp4aTests.SkipUnlessDp4aKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "hadamard_fwht.ptx")),
            "hadamard_fwht.ptx not generated (Bonsai 2 is Hadamard-folded)");
        string? path = FindBonsai2Checkpoint();
        Skip.If(path is null, "Bonsai 2 MTP GGUF not found (set DOTLLM_BONSAI2_MTP_GGUF)");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.PQ2_0Dp4aAvailable, "pq2_0_gemv_dp4a.ptx present but the dp4a module did not load (stale PTX?)");

        int[] prompt = [760, 6511, 314, 9338, 369];
        int[] continuation = [11, 1052, 3070];
        Dp4aArms arms = RunDp4aArms(model, config.VocabSize, prompt, continuation);
        AssertDp4aStructure(arms, "Bonsai 2");

        float dp4aDrift = MaxAbsDiff(arms.Dp4aSingles, arms.Dp4aBatched);
        float f16Drift = MaxAbsDiff(arms.F16Singles, arms.F16Batched);
        float floorS1 = MaxAbsDiff(arms.F16Singles, arms.Dp4aSingles);
        float floorS3 = MaxAbsDiff(arms.F16Batched, arms.Dp4aBatched);
        _out.WriteLine($"Bonsai 2: max|S3-S1| dp4a {dp4aDrift:E3}, F16 {f16Drift:E3}; " +
                       $"max|dp4a-F16| at S=1 {floorS1:E3}, at S=3 {floorS3:E3}");
        for (int r = 0; r < continuation.Length; r++)
        {
            _out.WriteLine($"  row {r}: max|S3-S1| dp4a {MaxAbsDiff(arms.Dp4aSingles[r], arms.Dp4aBatched[r]):E3}, " +
                           $"F16 {MaxAbsDiff(arms.F16Singles[r], arms.F16Batched[r]):E3}; " +
                           $"max|dp4a-F16| S=1 {MaxAbsDiff(arms.F16Singles[r], arms.Dp4aSingles[r]):E3}");
            Assert.Equal(ArgMax(arms.Dp4aSingles[r]), ArgMax(arms.Dp4aBatched[r]));
        }
        float floor = MathF.Max(floorS1, floorS3);
        Assert.True(dp4aDrift <= 2f * floor + 1e-2f,
            $"dp4a S=3 drifts {dp4aDrift:E3} from dp4a S=1 — beyond 2x the dp4a quantization floor {floor:E3} " +
            "(two independent quantization-noise realizations give ~1.4x)");
        AssertArgMaxWhereClear(arms.F16Singles, arms.Dp4aSingles, floorS1, "Bonsai 2 dp4a vs F16 S=1");
    }

    private sealed record Dp4aArms(
        float[][] Dp4aBatched, float[][] Dp4aSingles, float[][] F16Batched, float[][] F16Singles,
        float[][] Dp4aBatchedNoShare, float[][] Dp4aSinglesNoShare,
        (int Gemv, int Quantize) CountsS3, (int Gemv, int Quantize) CountsS1,
        (int Gemv, int Quantize) CountsS3NoShare, (int Gemv, int Quantize) CountsS1NoShare);

    private static Dp4aArms RunDp4aArms(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation)
    {
        try
        {
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = true;
            float[][] dBatched = RunBatched(model, vocab, prompt, continuation, out var c3);
            float[][] dSingles = RunSingles(model, vocab, prompt, continuation, out var c1);
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = false;
            float[][] dBatchedNs = RunBatched(model, vocab, prompt, continuation, out var c3Ns);
            float[][] dSinglesNs = RunSingles(model, vocab, prompt, continuation, out var c1Ns);
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            float[][] fBatched = RunBatched(model, vocab, prompt, continuation, out _);
            float[][] fSingles = RunSingles(model, vocab, prompt, continuation, out _);
            return new Dp4aArms(dBatched, dSingles, fBatched, fSingles, dBatchedNs, dSinglesNs, c3, c1, c3Ns, c1Ns);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = null;
            CudaSmallSGemvDispatch.ShareDp4aInputOverride = null;
        }
    }

    /// <summary>
    /// Exact checks that separate a dp4a dispatch bug from numerics: (1) the same number of dp4a GEMV
    /// and quantizer launches per forward at S = 1 and S = 3 (identical coverage); (2) with input
    /// sharing off every GEMV quantizes its own input, and the logits are bit-identical to sharing on
    /// at both S (a stale shared scratch cannot hide); (3) the dp4a logits differ from the F16 path's.
    /// </summary>
    private void AssertDp4aStructure(Dp4aArms arms, string label)
    {
        _out.WriteLine($"{label}: dp4a launches per forward (gemv, quantize): S=1 {arms.CountsS1}, S=3 {arms.CountsS3}; " +
                       $"sharing off: S=1 {arms.CountsS1NoShare}, S=3 {arms.CountsS3NoShare}");
        Assert.True(arms.CountsS1.Gemv > 0, $"{label}: no dp4a GEMV launched at S=1 — the path did not engage");
        Assert.True(arms.CountsS1 == arms.CountsS3,
            $"{label}: dp4a coverage differs between S=1 {arms.CountsS1} and S=3 {arms.CountsS3}");
        Assert.True(arms.CountsS1NoShare == arms.CountsS3NoShare,
            $"{label}: dp4a coverage (sharing off) differs between S=1 {arms.CountsS1NoShare} and S=3 {arms.CountsS3NoShare}");
        Assert.Equal(arms.CountsS1.Gemv, arms.CountsS1NoShare.Gemv);
        Assert.Equal(arms.CountsS1NoShare.Gemv, arms.CountsS1NoShare.Quantize);
        Assert.True(arms.CountsS1.Quantize <= arms.CountsS1.Gemv, $"{label}: more quantize than GEMV launches with sharing on");

        Assert.True(BitwiseEqual(arms.Dp4aSingles, arms.Dp4aSinglesNoShare),
            $"{label}: S=1 logits change when dp4a input sharing is turned off (max|diff| " +
            $"{MaxAbsDiff(arms.Dp4aSingles, arms.Dp4aSinglesNoShare):E3}) — a shared quantized scratch is stale.");
        Assert.True(BitwiseEqual(arms.Dp4aBatched, arms.Dp4aBatchedNoShare),
            $"{label}: S=3 logits change when dp4a input sharing is turned off (max|diff| " +
            $"{MaxAbsDiff(arms.Dp4aBatched, arms.Dp4aBatchedNoShare):E3}) — a shared quantized scratch is stale.");

        Assert.True(!BitwiseEqual(arms.Dp4aSingles, arms.F16Singles),
            $"{label}: S=1 logits are bit-identical with dp4a on and off — the dp4a kernel did not run.");
        Assert.True(!BitwiseEqual(arms.Dp4aBatched, arms.F16Batched),
            $"{label}: S=3 logits are bit-identical with dp4a on and off — the dp4a kernel did not run.");
    }

    private void AssertArgMaxWhereClear(float[][] reference, float[][] actual, float observedDiff, string label)
    {
        for (int r = 0; r < reference.Length; r++)
        {
            (int best, float gap) = ArgMaxWithGap(reference[r]);
            if (gap > 2 * observedDiff)
                Assert.True(best == ArgMax(actual[r]), $"{label}: row {r} argmax {ArgMax(actual[r])} != reference {best} (gap {gap:E3})");
            else
                _out.WriteLine($"{label}: row {r} top-2 gap {gap:E3} within 2x max|diff| — argmax not asserted");
        }
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

    [SkippableFact]
    public void Forward_S3_MatchesThreeS1Forwards_OnPq2_0Fixture()
    {
        string ptxDir = CudaPQ2_0GemvMultiTests.SkipUnlessMultiKernel();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-pq2_0.gguf"), withMtp: false, pq2_0Projections: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        Arms arms = RunArms(model, config.VocabSize, Prompt, Continuation);

        // Same bar as the synthetic CUDA<->CPU hybrid parity tests (trunk-path drift, not kernel error).
        AssertRowsMatch(arms.Singles, arms.Multi, absTol: 2e-3f, relTol: 1e-2f, "multi-column S=3 vs 3x S=1");
        AssertRowsMatch(arms.Singles, arms.Old, absTol: 2e-3f, relTol: 1e-2f, "control: dequant+cuBLAS S=3 vs 3x S=1");
        Assert.True(!BitwiseEqual(arms.Multi, arms.Old),
            "S=3 logits are bit-identical with the multi-column path on and off — the multi-column kernel did not run.");
    }

    /// <summary>
    /// The same three arms on the real Bonsai 2 27B MTP checkpoint (skipped unless it is present —
    /// <c>DOTLLM_BONSAI2_MTP_GGUF</c> or the HF hub cache — and <c>hadamard_fwht.ptx</c> exists).
    /// One model instance, <see cref="CudaQwen3HybridDenseTransformerModel.ResetSequenceState"/>
    /// between arms (the weights do not fit twice in 12 GB). Self-calibrating: the multi-column
    /// arm's drift from the S = 1 reference must not exceed the old path's by more than 50% (plus a
    /// small absolute slack), and every row's argmax must agree with the reference.
    /// </summary>
    [SkippableFact]
    public void Forward_S3_MatchesThreeS1Forwards_OnRealBonsai2()
    {
        string ptxDir = CudaPQ2_0GemvMultiTests.SkipUnlessMultiKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "hadamard_fwht.ptx")),
            "hadamard_fwht.ptx not generated (Bonsai 2 is Hadamard-folded)");
        string? path = FindBonsai2Checkpoint();
        Skip.If(path is null, "Bonsai 2 MTP GGUF not found (set DOTLLM_BONSAI2_MTP_GGUF)");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        int[] prompt = [760, 6511, 314, 9338, 369];   // arbitrary in-vocab ids
        int[] continuation = [11, 1052, 3070];
        Arms arms = RunArms(model, config.VocabSize, prompt, continuation);

        float multiDrift = MaxAbsDiff(arms.Singles, arms.Multi);
        float oldDrift = MaxAbsDiff(arms.Singles, arms.Old);
        _out.WriteLine($"Bonsai 2: max|S3-S1| multi-column {multiDrift:E3}, dequant+cuBLAS {oldDrift:E3}");
        for (int r = 0; r < continuation.Length; r++)
        {
            Assert.Equal(ArgMax(arms.Singles[r]), ArgMax(arms.Multi[r]));
        }
        Assert.True(multiDrift <= 1.5f * oldDrift + 1e-2f,
            $"multi-column S=3 drifts {multiDrift:E3} from the S=1 reference, vs {oldDrift:E3} for the old path");
        Assert.True(!BitwiseEqual(arms.Multi, arms.Old),
            "S=3 logits are bit-identical with the multi-column path on and off — the multi-column kernel did not run.");
    }

    private sealed record Arms(float[][] Multi, float[][] Old, float[][] Singles);

    private Arms RunArms(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation)
    {
        float[][] multi, old, singles;
        try
        {
            // #482 arms: pin the dp4a path off so a process-wide DOTLLM_CUDA_PQ2_0_DP4A=1 cannot route
            // the S = 1 reference (or either S = 3 arm) through it.
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            multi = RunBatched(model, vocab, prompt, continuation, out _);
            CudaSmallSGemvDispatch.MaxColumnsOverride = 0;
            old = RunBatched(model, vocab, prompt, continuation, out _);
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            singles = RunSingles(model, vocab, prompt, continuation, out _);
        }
        finally
        {
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = null;
        }
        return new Arms(multi, old, singles);
    }

    /// <param name="dp4aCounts">dp4a (GEMV, quantize) launches of the S = continuation.Length forward alone.</param>
    private static float[][] RunBatched(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation,
        out (int Gemv, int Quantize) dp4aCounts)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        Prefill(model, kv, prompt);
        int[] positions = new int[continuation.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = prompt.Length + i;
        model.ResetDp4aLaunchCounts();
        using var logits = model.Forward(continuation, positions, deviceId: -1, kv);
        dp4aCounts = model.Dp4aLaunchCounts;
        Assert.Equal(continuation.Length, logits.Shape[0]);
        float[][] rows = new float[continuation.Length][];
        for (int r = 0; r < rows.Length; r++) rows[r] = Row(logits, r, vocab);
        return rows;
    }

    /// <param name="dp4aCounts">dp4a (GEMV, quantize) launches of the first S = 1 continuation forward alone.</param>
    private static float[][] RunSingles(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation,
        out (int Gemv, int Quantize) dp4aCounts)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        Prefill(model, kv, prompt);
        float[][] rows = new float[continuation.Length][];
        dp4aCounts = default;
        for (int i = 0; i < continuation.Length; i++)
        {
            model.ResetDp4aLaunchCounts();
            using var logits = model.Forward([continuation[i]], [prompt.Length + i], deviceId: -1, kv);
            if (i == 0) dp4aCounts = model.Dp4aLaunchCounts;
            Assert.Equal(1, logits.Shape[0]);
            rows[i] = Row(logits, 0, vocab);
        }
        return rows;
    }

    // One token at a time: the prefix never touches the seqLen 2..8 dispatch under test.
    private static void Prefill(CudaQwen3HybridDenseTransformerModel model, DotLLM.Core.Attention.IKvCache kv, int[] prompt)
    {
        for (int i = 0; i < prompt.Length; i++)
            using (model.Forward([prompt[i]], [i], deviceId: -1, kv)) { }
    }

    private void AssertRowsMatch(float[][] expected, float[][] actual, float absTol, float relTol, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int r = 0; r < expected.Length; r++)
        {
            float maxDiff = 0;
            for (int c = 0; c < expected[r].Length; c++)
            {
                float a = expected[r][c], b = actual[r][c];
                float d = MathF.Abs(a - b);
                maxDiff = MathF.Max(maxDiff, d);
                Assert.True(float.IsFinite(b), $"{label}: row {r} logit {c} is not finite");
                Assert.True(d <= absTol + relTol * MathF.Abs(a),
                    $"{label}: row {r} logit {c}: reference {a}, got {b}, |diff| {d:E3}");
            }
            _out.WriteLine($"{label}: row {r} max|diff| {maxDiff:E3}");
            Assert.Equal(ArgMax(expected[r]), ArgMax(actual[r]));
        }
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int c = 0; c < a.Length; c++) m = MathF.Max(m, MathF.Abs(a[c] - b[c]));
        return m;
    }

    private static float MaxAbsDiff(float[][] a, float[][] b)
    {
        float m = 0;
        for (int r = 0; r < a.Length; r++)
            for (int c = 0; c < a[r].Length; c++)
                m = MathF.Max(m, MathF.Abs(a[r][c] - b[r][c]));
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
