using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
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
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Issue #485, dp4a on: an S = 3 forward must match three S = 1 forwards (both on the dp4a GEMV —
    /// its per-column arithmetic does not depend on S, so only the trunk's S-dependent kernels
    /// differ), and the dp4a logits must be close to the F16-activation path's without being
    /// bit-identical to it (proof the dp4a kernel ran).
    /// </summary>
    /// <remarks>
    /// The dp4a-vs-F16 bound is self-calibrating: W2A8 on a random synthetic model has no fixed
    /// expected error, so it is held to its own S = 3 vs S = 1 trunk drift scale plus the argmax,
    /// and the argmax only where the F16 top-2 gap is clear of the observed difference.
    /// </remarks>
    [SkippableFact]
    public void Dp4a_Forward_S3_MatchesThreeS1Forwards_OnPq2_0Fixture()
    {
        string ptxDir = CudaPQ2_0GemvDp4aTests.SkipUnlessDp4aKernel();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-pq2_0-dp4a.gguf"), withMtp: false, pq2_0Projections: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.PQ2_0Dp4aAvailable, "pq2_0_gemv_dp4a.ptx present but the dp4a module did not load (stale PTX?)");

        Dp4aArms arms = RunDp4aArms(model, config.VocabSize, Prompt, Continuation);

        AssertRowsMatch(arms.Dp4aSingles, arms.Dp4aBatched, absTol: 2e-3f, relTol: 1e-2f, "dp4a S=3 vs dp4a 3x S=1");
        float dp4aVsF16 = MaxAbsDiff(arms.F16Singles, arms.Dp4aSingles);
        _out.WriteLine($"dp4a vs F16 (S=1 arms): max|diff| {dp4aVsF16:E3}");
        AssertArgMaxWhereClear(arms.F16Singles, arms.Dp4aSingles, dp4aVsF16, "dp4a vs F16 S=1");
        Assert.True(!BitwiseEqual(arms.Dp4aSingles, arms.F16Singles),
            "S=1 logits are bit-identical with dp4a on and off — the dp4a kernel did not run.");
        Assert.True(!BitwiseEqual(arms.Dp4aBatched, arms.F16Batched),
            "S=3 logits are bit-identical with dp4a on and off — the dp4a kernel did not run.");
    }

    /// <summary>
    /// The dp4a arms on the real Bonsai 2 27B MTP checkpoint (skipped unless present). dp4a S = 3 vs
    /// dp4a 3x S = 1 must agree in argmax and drift no more than the F16 multi-column path does (×1.5
    /// + slack); dp4a S = 1 must agree with F16 S = 1 in argmax wherever the gap is clear.
    /// </summary>
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

        float dp4aDrift = MaxAbsDiff(arms.Dp4aSingles, arms.Dp4aBatched);
        float f16Drift = MaxAbsDiff(arms.F16Singles, arms.F16Batched);
        float dp4aVsF16 = MaxAbsDiff(arms.F16Singles, arms.Dp4aSingles);
        _out.WriteLine($"Bonsai 2: max|S3-S1| dp4a {dp4aDrift:E3}, F16 {f16Drift:E3}; max|dp4a-F16| at S=1 {dp4aVsF16:E3}");
        for (int r = 0; r < continuation.Length; r++)
            Assert.Equal(ArgMax(arms.Dp4aSingles[r]), ArgMax(arms.Dp4aBatched[r]));
        Assert.True(dp4aDrift <= 1.5f * f16Drift + 1e-2f,
            $"dp4a S=3 drifts {dp4aDrift:E3} from dp4a S=1, vs {f16Drift:E3} for the F16 path");
        AssertArgMaxWhereClear(arms.F16Singles, arms.Dp4aSingles, dp4aVsF16, "Bonsai 2 dp4a vs F16 S=1");
        Assert.True(!BitwiseEqual(arms.Dp4aBatched, arms.F16Batched),
            "S=3 logits are bit-identical with dp4a on and off — the dp4a kernel did not run.");
    }

    private sealed record Dp4aArms(float[][] Dp4aBatched, float[][] Dp4aSingles, float[][] F16Batched, float[][] F16Singles);

    private static Dp4aArms RunDp4aArms(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation)
    {
        try
        {
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = true;
            float[][] dBatched = RunBatched(model, vocab, prompt, continuation);
            float[][] dSingles = RunSingles(model, vocab, prompt, continuation);
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            float[][] fBatched = RunBatched(model, vocab, prompt, continuation);
            float[][] fSingles = RunSingles(model, vocab, prompt, continuation);
            return new Dp4aArms(dBatched, dSingles, fBatched, fSingles);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = null;
        }
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
        float[][] multi, old;
        try
        {
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
            multi = RunBatched(model, vocab, prompt, continuation);
            CudaSmallSGemvDispatch.MaxColumnsOverride = 0;
            old = RunBatched(model, vocab, prompt, continuation);
        }
        finally
        {
            CudaSmallSGemvDispatch.MaxColumnsOverride = null;
        }
        float[][] singles = RunSingles(model, vocab, prompt, continuation);
        return new Arms(multi, old, singles);
    }

    private static float[][] RunBatched(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        Prefill(model, kv, prompt);
        int[] positions = new int[continuation.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = prompt.Length + i;
        using var logits = model.Forward(continuation, positions, deviceId: -1, kv);
        Assert.Equal(continuation.Length, logits.Shape[0]);
        float[][] rows = new float[continuation.Length][];
        for (int r = 0; r < rows.Length; r++) rows[r] = Row(logits, r, vocab);
        return rows;
    }

    private static float[][] RunSingles(CudaQwen3HybridDenseTransformerModel model, int vocab, int[] prompt, int[] continuation)
    {
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(maxSeqLen: 64);
        Prefill(model, kv, prompt);
        float[][] rows = new float[continuation.Length][];
        for (int i = 0; i < continuation.Length; i++)
        {
            using var logits = model.Forward([continuation[i]], [prompt.Length + i], deviceId: -1, kv);
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
