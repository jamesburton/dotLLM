using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #472: the batched, KV-only MTP absorb must leave the head's KV-cache exactly where the
/// #469 per-token absorb loop leaves it.
/// </summary>
/// <remarks>
/// The sequence is split across TWO trunk forwards on purpose. Row 0 of the second batch pairs
/// with the carried row (the last hidden of the first batch), rows 1.. with that batch's own
/// captured rows 0.. — the two places an i-vs-(i-1) pairing slip or a carry mix-up can hide.
/// The fixture's random weights make every hidden row distinct, so any mispairing moves the K/V
/// rows by O(1), not by rounding.
/// </remarks>
public sealed class Qwen3HybridDenseMtpBatchedAbsorbTests : IDisposable
{
    // F32 fixture: GEMM(n=S) vs GEMV(n=1) differ only in reduction order.
    private const float Tol = 1e-4f;

    private readonly string _scratch;

    public Qwen3HybridDenseMtpBatchedAbsorbTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-mtp-absorb-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Theory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]    // issue #486: the Q8_0-head fixture variant loads and absorbs on CPU too
    public void BatchedAbsorb_KvCacheMatchesPerTokenAbsorb(bool mtpHasOwnHeadTensors, bool q8_0MtpHead)
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"mtp-own{mtpHasOwnHeadTensors}-q8{q8_0MtpHead}.gguf"),
            withMtp: true, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors, q8_0MtpHead: q8_0MtpHead);

        var perToken = RunSequence(path, perToken: true);
        var batched = RunSequence(path, perToken: false);

        Assert.Equal(perToken.Length, batched.Length);
        AssertClose(perToken.Keys, batched.Keys, "K");
        AssertClose(perToken.Values, batched.Values, "V");
        AssertClose(perToken.Pending, batched.Pending, "pending hidden");
        AssertClose(perToken.DraftLogits, batched.DraftLogits, "first draft step logits");
    }

    /// <summary>
    /// A verify-shaped absorb over speculative slots: the head drafted past the batch start, and the
    /// batched absorb must overwrite those slots (rolling back) rather than append after them.
    /// </summary>
    [Fact]
    public void BatchedAbsorb_OverwritesSpeculativeDraftSlots()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "mtp-verify.gguf"), withMtp: true);

        var perToken = RunVerifyShaped(path, perToken: true);
        var batched = RunVerifyShaped(path, perToken: false);

        Assert.Equal(perToken.Length, batched.Length);
        AssertClose(perToken.Keys, batched.Keys, "K");
        AssertClose(perToken.Values, batched.Values, "V");
    }

    private sealed record Snapshot(int Length, float[] Keys, float[] Values, float[] Pending, float[] DraftLogits);

    private static Snapshot RunSequence(string path, bool perToken)
    {
        MtpAbsorbDispatch.PerTokenOverride = perToken;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var state = (CpuMtpState)model.CreateMtpState()!;

            using (ITensor _ = model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv, adapter: null, state)) { }
            using (ITensor _ = model.Forward([4, 5, 6, 7], [3, 4, 5, 6], deviceId: -1, kv, adapter: null, state)) { }

            var snap = Capture(state, config.HiddenSize);
            using ITensor draft = model.ForwardMtp(state, 8, 7);
            return snap with { DraftLogits = Copy(draft, config.VocabSize) };
        }
        finally
        {
            MtpAbsorbDispatch.PerTokenOverride = null;
        }
    }

    private static Snapshot RunVerifyShaped(string path, bool perToken)
    {
        MtpAbsorbDispatch.PerTokenOverride = perToken;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var state = (CpuMtpState)model.CreateMtpState()!;

            using (ITensor _ = model.Forward([1, 2, 3], [0, 1, 2], deviceId: -1, kv, adapter: null, state)) { }
            // Draft three speculative slots 3..5 — the verify batch below starts at 3 again.
            for (int i = 0; i < 3; i++)
                using (ITensor _ = model.ForwardMtp(state, 9 - i, 3 + i)) { }
            Assert.Equal(6, state.CurrentLength);
            using (ITensor _ = model.Forward([5, 10, 11], [3, 4, 5], deviceId: -1, kv, adapter: null, state)) { }

            return Capture(state, config.HiddenSize);
        }
        finally
        {
            MtpAbsorbDispatch.PerTokenOverride = null;
        }
    }

    private static unsafe Snapshot Capture(CpuMtpState state, int hiddenSize)
    {
        int n = state.CurrentLength * (SyntheticQwen35HybridDenseMtpGguf.NumKvHeads * SyntheticQwen35HybridDenseMtpGguf.HeadDim);
        var keys = new ReadOnlySpan<float>(state.KeyCachePtr, n).ToArray();
        var values = new ReadOnlySpan<float>(state.ValueCachePtr, n).ToArray();
        Assert.Equal(hiddenSize, state.PendingHidden.Length);
        return new Snapshot(state.CurrentLength, keys, values, state.PendingHidden.ToArray(), []);
    }

    private static unsafe float[] Copy(ITensor t, int count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}[{i}] is not finite");
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tol,
                $"{what}[{i}]: per-token {expected[i]} vs batched {actual[i]}");
        }
    }
}
