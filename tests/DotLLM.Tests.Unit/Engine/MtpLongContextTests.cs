using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #966: the MTP head's position-indexed KV-cache beyond the old 4096-position default.
/// Host-only (CPU backend + pure bookkeeping), so these run everywhere; the CUDA twin is
/// <c>CudaQwen3HybridDenseMtpLongContextTests</c>.
/// </summary>
public sealed class MtpLongContextTests : IDisposable
{
    private const int ContextLength = 5200;   // > 4096 + a K=3 round, with room for a 1024-step grow past it
    private const int NewTokens = 8;
    private const int K = 3;

    private readonly string _scratch;

    public MtpLongContextTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-mtp-longctx-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ── Pure bookkeeping ─────────────────────────────────────────────────────

    [Theory]
    [InlineData(100, 100)]        // short ceiling: exactly the eager size, never more
    [InlineData(1024, 1024)]
    [InlineData(1025, 1024)]
    [InlineData(65536, 1024)]
    public void Initial_NeverExceedsTheEagerSize_AndCapsAtOneStep(int ceiling, int expected)
        => Assert.Equal(expected, MtpKvCapacity.Initial(ceiling));

    [Theory]
    [InlineData(1024, 1024, 65536, 1024)]    // fits: unchanged
    [InlineData(1024, 1025, 65536, 2048)]    // first growth, rounded up to a step
    [InlineData(2048, 4096, 65536, 4096)]
    [InlineData(4096, 4097, 65536, 5120)]    // past the old cap
    [InlineData(1024, 1500, 1500, 1500)]     // clamped to the ceiling, not rounded past it
    [InlineData(1024, 1025, 1030, 1030)]
    [InlineData(1500, 1500, 1500, 1500)]
    public void Next_RoundsUpToAStep_AndNeverPassesTheCeiling(int current, int needed, int ceiling, int expected)
        => Assert.Equal(expected, MtpKvCapacity.Next(current, needed, ceiling));

    [Fact]
    public void Next_AboveTheCeiling_ThrowsInsteadOfClamping()
    {
        var ex = Assert.Throws<InvalidOperationException>(() => MtpKvCapacity.Next(1024, 1501, 1500));
        Assert.Contains("exhausted", ex.Message);
    }

    [Fact]
    public void Next_IsMonotoneAndAlwaysCoversTheRequest()
    {
        const int ceiling = 70000;
        int cap = MtpKvCapacity.Initial(ceiling);
        for (int need = 1; need <= ceiling; need += 37)
        {
            int next = MtpKvCapacity.Next(cap, need, ceiling);
            Assert.True(next >= cap);
            Assert.True(next >= need);
            Assert.True(next <= ceiling);
            cap = next;
        }
    }

    [Fact]
    public void CpuMtpState_GrowsLazily_PreservesCommittedRows_AndRefusesPastTheCeiling()
    {
        const int hidden = 4, kvHeads = 2, headDim = 3, ceiling = 5000;
        const int stride = kvHeads * headDim;
        using var st = new CpuMtpState(hidden, kvHeads, headDim, ceiling);

        Assert.Equal(ceiling, st.MaxSteps);
        Assert.Equal(1024, st.Capacity);
        long initialBytes = st.AllocatedBytes;
        Assert.Equal(2L * 1024 * stride * sizeof(float), initialBytes);

        // Commit 1500 rows (absorb path), each stamped with its own index.
        st.BeginAbsorb(0, 1500);
        Assert.Equal(2048, st.Capacity);
        for (int p = 0; p < 1500; p++)
        {
            st.GetKeyRow(p).Fill(p + 0.25f);
            st.GetValueRow(p).Fill(-(p + 0.5f));
        }
        st.EndAbsorb(1500);

        // Grow again through a draft-shaped single-step advance region.
        st.EnsureCapacity(3000);
        Assert.Equal(3072, st.Capacity);
        for (int p = 0; p < 1500; p++)
        {
            Assert.Equal(p + 0.25f, st.GetKeyRow(p)[stride - 1]);
            Assert.Equal(-(p + 0.5f), st.GetValueRow(p)[0]);
        }
        Assert.True(st.GetKeyRow(2999).ToArray().All(v => v == 0f), "grown rows start zeroed");

        // The ceiling is a hard stop, and a refused request changes nothing.
        int before = st.Capacity;
        Assert.Throws<InvalidOperationException>(() => st.EnsureCapacity(ceiling + 1));
        Assert.Throws<InvalidOperationException>(() => st.BeginAbsorb(1500, ceiling));
        Assert.Equal(before, st.Capacity);

        st.EnsureCapacity(ceiling);
        Assert.Equal(ceiling, st.Capacity);   // clamped, not rounded to 5120
        Assert.Throws<ArgumentOutOfRangeException>(() => st.GetKeyRow(ceiling));
    }

    // ── End to end on the CPU backend ────────────────────────────────────────

    /// <summary>
    /// Prompt lengths straddling the first growth step (1024) and the old 4096 cap: the batched
    /// prefill absorb ends at L, then draft steps and verify absorbs write L .. L+K. MTP greedy must
    /// reproduce plain greedy exactly and the head must have grown to cover the sequence.
    /// </summary>
    [Theory]
    [InlineData(1023)]
    [InlineData(1024)]
    [InlineData(1025)]
    [InlineData(4095)]
    [InlineData(4096)]
    [InlineData(4097)]
    public void MtpGreedy_AcrossGrowthAndTheOldCap_MatchesPlainGreedy(int promptLength)
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"mtp-longctx-{promptLength}.gguf"), withMtp: true, contextLength: ContextLength);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = MtpLongContextHarness.Prompt(promptLength, config.VocabSize);

        int[] expected;
        using (var plain = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config))
        using (var kv = new SimpleKvCache(plain.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength))
            expected = MtpLongContextHarness.PlainGreedy(plain, kv, prompt, NewTokens, config.VocabSize);

        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        // The engine's shape: ceiling = the trunk KV-cache's MaxLength.
        using var mtp = (CpuMtpState)model.CreateMtpState(kvCache.MaxLength)!;
        Assert.Equal(1024, mtp.Capacity);   // lazy: nothing beyond the first step before the prefill

        int[] actual = MtpLongContextHarness.MtpGreedy(model, kvCache, mtp, prompt, NewTokens, K, config.VocabSize);

        Assert.Equal(expected, actual);
        Assert.True(mtp.Capacity >= promptLength + 1, $"capacity {mtp.Capacity} does not cover the sequence");
        Assert.True(mtp.Capacity <= mtp.MaxSteps);
        Assert.True(mtp.Capacity % MtpKvCapacity.GrowthStep == 0 || mtp.Capacity == mtp.MaxSteps);
        Assert.True(mtp.CurrentLength >= promptLength);
    }

    /// <summary>
    /// The head KV itself, not just verified output (which equals plain greedy however bad the drafts
    /// are): a lazily grown state must produce BIT-IDENTICAL draft logits to one that was fully sized up
    /// front. A grow that dropped, shifted or zeroed committed rows changes attention and shows here.
    /// </summary>
    [Theory]
    [InlineData(1023)]   // drafts cross the first growth step inside ForwardMtp
    [InlineData(2500)]   // several growths inside the prefill absorb
    [InlineData(4095)]   // drafts cross the old 4096 cap inside ForwardMtp
    public void DraftLogits_AfterLazyGrowth_AreBitIdenticalToAFullySizedState(int promptLength)
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"mtp-longctx-draft-{promptLength}.gguf"), withMtp: true, contextLength: ContextLength);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int[] prompt = MtpLongContextHarness.Prompt(promptLength, config.VocabSize);

        float[][] eager;
        using (var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength))
        using (var mtp = (CpuMtpState)model.CreateMtpState(kv.MaxLength)!)
        {
            mtp.EnsureCapacity(kv.MaxLength);   // the pre-#966 eager allocation
            Assert.Equal(kv.MaxLength, mtp.Capacity);
            eager = MtpLongContextHarness.DraftLogits(model, kv, mtp, prompt, steps: 5, config.VocabSize);
            Assert.Equal(kv.MaxLength, mtp.Capacity);
        }

        float[][] lazy;
        using (var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength))
        using (var mtp = (CpuMtpState)model.CreateMtpState(kv.MaxLength)!)
        {
            lazy = MtpLongContextHarness.DraftLogits(model, kv, mtp, prompt, steps: 5, config.VocabSize);
            Assert.True(mtp.Capacity < kv.MaxLength, "the lazy state must not have been sized up front");
        }

        for (int s = 0; s < eager.Length; s++)
            Assert.Equal(eager[s], lazy[s]);   // exact float equality
    }

    /// <summary>
    /// The parameterless overload used to cap at 4096 positions; it now uses the model's context window
    /// as its ceiling, so a prompt past 4096 absorbs instead of throwing.
    /// </summary>
    [Fact]
    public void ParameterlessCreateMtpState_NoLongerCapsAt4096()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "mtp-longctx-default.gguf"), withMtp: true, contextLength: ContextLength);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtp = (CpuMtpState)model.CreateMtpState()!;

        Assert.Equal(ContextLength, mtp.MaxSteps);
        int len = 4100;
        int[] prompt = MtpLongContextHarness.Prompt(len, config.VocabSize);
        using (model.Forward(prompt, Enumerable.Range(0, len).ToArray(), deviceId: -1, kvCache, adapter: null, mtp)) { }
        Assert.Equal(len, mtp.CurrentLength);
    }

    /// <summary>
    /// A state created with a ceiling below the sequence refuses with a clear exception, before any
    /// write: no silent clamp, no wrong attention.
    /// </summary>
    [Fact]
    public void ExceedingTheCeiling_ThrowsInvalidOperation_AndNeverGrowsPastIt()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "mtp-longctx-ceiling.gguf"), withMtp: true, contextLength: ContextLength);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        const int ceiling = 1030;
        using var mtp = (CpuMtpState)model.CreateMtpState(ceiling)!;
        int len = ceiling + 1;
        int[] prompt = MtpLongContextHarness.Prompt(len, config.VocabSize);

        var ex = Assert.Throws<InvalidOperationException>(() =>
        {
            using DotLLM.Core.Tensors.ITensor _ = model.Forward(prompt, Enumerable.Range(0, len).ToArray(), deviceId: -1, kvCache, adapter: null, mtp);
        });
        Assert.Contains("exhausted", ex.Message);
        Assert.True(mtp.Capacity <= ceiling);
        Assert.Equal(0, mtp.CurrentLength);   // nothing was committed
    }
}
