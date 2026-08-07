using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Coverage for issue #253 (Multi-Token Prediction self-speculative decoding): GGUF detection
/// and loading of the trailing MTP/"NextN" head on <see cref="Qwen3HybridDenseTransformerModel"/>,
/// and the MTP head's own forward pass. Uses <see cref="SyntheticQwen35HybridDenseMtpGguf"/> — a
/// tiny fixture built from confirmed llama.cpp PR ggml-org/llama.cpp#22673 tensor naming/layout
/// (no real Qwen3.6-MTP-GGUF fixture is cached locally; see the issue's fixture-availability note).
/// </summary>
public sealed class Qwen3HybridDenseMtpTests : IDisposable
{
    private readonly string _scratch;

    public Qwen3HybridDenseMtpTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string WriteFixture(bool withMtp, bool mtpHasOwnHeadTensors = true, string name = "qwen35-mtp.gguf") =>
        SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, name), withMtp: withMtp, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors);

    // ── GGUF detection / zero-behavior-change ──────────────────────────────────

    [Fact]
    public void LoadFromGguf_WithMtp_DetectsHeadAndSubtractsTrunkLayerCount()
    {
        string path = WriteFixture(withMtp: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        Assert.Equal(1, config.NextnPredictLayers);
        // Raw GGUF block_count is BlockCount+1 (trunk+MTP); NumLayers must be trunk-only.
        Assert.Equal(SyntheticQwen35HybridDenseMtpGguf.BlockCount, config.NumLayers);
        Assert.NotNull(config.HybridLayout);
        Assert.Equal(config.NumLayers, config.HybridLayout!.LayerKind.Length);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3HybridDenseTransformerModel>(model);
        Assert.True(hybrid.SupportsMtp);
        using IMtpState? state = hybrid.CreateMtpState();
        Assert.NotNull(state);
    }

    [Fact]
    public void LoadFromGguf_WithoutMtp_ZeroBehaviorChange()
    {
        string path = WriteFixture(withMtp: false);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(0, config.NextnPredictLayers);
        Assert.Equal(SyntheticQwen35HybridDenseMtpGguf.BlockCount, config.NumLayers);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3HybridDenseTransformerModel>(model);
        Assert.False(hybrid.SupportsMtp);
        Assert.Null(hybrid.CreateMtpState());

        Assert.Throws<NotSupportedException>(() =>
        {
            using var state = new CpuMtpState(hiddenSize: 1, numKvHeads: 1, headDim: 1, maxSteps: 1);
            hybrid.ForwardMtp(state, tokenId: 0, position: 0);
        });
    }

    [Fact]
    public void Forward_WithMtpStateCapture_ProducesIdenticalLogitsToPlainForward()
    {
        // The hidden-state capture is documented as a pure side effect — verify it byte-for-byte.
        string path = WriteFixture(withMtp: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using var kvCache1 = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using ITensor plainLogits = model.Forward(tokenIds, positions, deviceId: -1, kvCache1);

        // The model owns a default GDN recurrent-state cache when no per-sequence state is
        // threaded explicitly (see IModel.ResetSequenceState docs / issue #261) — reset it so the
        // second call starts a genuinely fresh sequence, matching the first call's starting state.
        model.ResetSequenceState();

        using var kvCache2 = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;
        using ITensor capturedLogits = model.Forward(tokenIds, positions, deviceId: -1, kvCache2, adapter: null, mtpState);

        Assert.Equal(plainLogits.Shape[0], capturedLogits.Shape[0]);
        Assert.Equal(plainLogits.Shape[1], capturedLogits.Shape[1]);
        unsafe
        {
            int n = tokenIds.Length * config.VocabSize;
            var a = new ReadOnlySpan<float>((void*)plainLogits.DataPointer, n);
            var b = new ReadOnlySpan<float>((void*)capturedLogits.DataPointer, n);
            for (int i = 0; i < n; i++)
                Assert.Equal(a[i], b[i]); // byte-identical float compare — pure side effect claim
        }

        Assert.Equal(tokenIds.Length, mtpState.CapturedRowCount);
        Assert.Equal(config.HiddenSize, mtpState.HiddenSize);
    }

    // ── MTP head forward math ───────────────────────────────────────────────────

    [Fact]
    public void ForwardMtp_ProducesFiniteLogitsAndAdvancesState()
    {
        string path = WriteFixture(withMtp: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        // Prefill + capture the trunk's hidden state to seed the MTP head.
        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState))
        {
        }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        Assert.Equal(0, mtpState.CurrentLength);

        using ITensor draft0 = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 2);
        Assert.Equal(1, draft0.Shape[0]);
        Assert.Equal(config.VocabSize, draft0.Shape[1]);
        Assert.Equal(1, mtpState.CurrentLength);
        AssertAllFinite(draft0, config.VocabSize);

        // Second autoregressive MTP step, seeded from the head's own output (not the trunk's).
        int argmax0 = ArgMax(draft0, config.VocabSize);
        using ITensor draft1 = model.ForwardMtp(mtpState, tokenId: argmax0, position: 3);
        Assert.Equal(2, mtpState.CurrentLength);
        AssertAllFinite(draft1, config.VocabSize);
    }

    [Fact]
    public void ForwardMtp_Deterministic_SameInputsSameOutputs()
    {
        string path = WriteFixture(withMtp: true);

        float[] logitsA = RunSingleMtpStep(path);
        float[] logitsB = RunSingleMtpStep(path);

        Assert.Equal(logitsA.Length, logitsB.Length);
        for (int i = 0; i < logitsA.Length; i++)
            Assert.Equal(logitsA[i], logitsB[i]);
    }

    [Fact]
    public void ForwardMtp_WithoutOwnHeadTensors_FallsBackToTrunkHeadAndEmbedding()
    {
        // mtpHasOwnHeadTensors:false exercises the "nextn.shared_head_head / shared_head_norm /
        // embed_tokens absent" fallback path — the loader must fall back to the trunk's own
        // token_embd/output/output_norm rather than throwing or producing garbage.
        string path = WriteFixture(withMtp: true, mtpHasOwnHeadTensors: false, name: "qwen35-mtp-noheadtensors.gguf");
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.True(model.SupportsMtp);

        int[] tokenIds = [0, 1];
        int[] positions = [0, 1];
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        using ITensor draft = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 1);
        AssertAllFinite(draft, config.VocabSize);
    }

    private static float[] RunSingleMtpStep(string path)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        using ITensor draft = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 2);
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize);
            return span.ToArray();
        }
    }

    private static void AssertAllFinite(ITensor tensor, int vocabSize)
    {
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, vocabSize);
            foreach (float v in span)
                Assert.True(float.IsFinite(v), $"non-finite logit: {v}");
        }
    }

    private static int ArgMax(ITensor tensor, int vocabSize)
    {
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, vocabSize);
            int best = 0;
            for (int i = 1; i < span.Length; i++)
                if (span[i] > span[best]) best = i;
            return best;
        }
    }
}
