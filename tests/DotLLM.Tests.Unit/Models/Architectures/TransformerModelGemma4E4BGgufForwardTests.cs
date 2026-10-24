using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Coverage for the Gemma-4 <b>dense-PLE</b> GGUF variant (the <c>gemma-4-E4B-it</c>
/// shape; llama.cpp <c>gemma4.cpp</c> without experts): Per-Layer Embeddings fed as a
/// gated residual into every layer, trailing shared-KV layers that attend over an
/// earlier donor layer's KV, full-head-dim dual RoPE with a <c>rope_freqs</c>
/// proportional-factor tensor, per-layer <c>layer_output_scale</c>, and the dense-only
/// FFN branch. Uses the deterministic <see cref="SyntheticGemma4Gguf.E4BLike"/>
/// fixture through the full GGUF load path so loader + forward are validated together.
/// AltUp / Laurel / activation sparsity are Gemma-3n-era components that Gemma 4
/// dropped — no released gemma4 GGUF carries their tensors, so no coverage exists here.
/// </summary>
public sealed class TransformerModelGemma4E4BGgufForwardTests : IDisposable
{
    private readonly string _scratch;

    public TransformerModelGemma4E4BGgufForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4e4b-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // Stays within the sliding window (8) so cacheless vs cached attention windows
    // coincide; token ids < VocabSize (256).
    private static readonly int[] Ids = [2, 7, 8, 9, 5, 6, 3];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5, 6];

    [Fact]
    public void ConfigExtraction_E4BLike_MapsPleSharedKvAndFullRotation()
    {
        string path = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Gemma4, cfg.Architecture);
        Assert.True(cfg.Gemma4DualFfn);
        Assert.Null(cfg.Moe);                       // dense tower — no experts
        Assert.NotNull(cfg.PerLayerEmbedding);
        Assert.Equal(16, cfg.PerLayerEmbedding!.PerLayerDim);
        Assert.Equal(cfg.VocabSize, cfg.PerLayerEmbedding.VocabSize);
        Assert.Equal(4, cfg.NumSharedKvLayers);
        // Full-dim rotation on the PLE variant (rope_freqs supplies the factors).
        Assert.Null(cfg.PartialRotaryFactor);
        Assert.Equal(32, cfg.GlobalHeadDim);
        Assert.Equal(16, cfg.HeadDim);

        // llama.cpp donor rule: kvFromStart = 8 - 4 = 4 → sliding shared layers
        // borrow layer 2 (sliding), full shared layers borrow layer 3 (global).
        Assert.True(cfg.LayerHasOwnKv(3));
        Assert.False(cfg.LayerHasOwnKv(4));
        Assert.False(cfg.LayerHasOwnKv(7));
        Assert.Equal(2, cfg.SharedKvDonorLayer(4));  // layer 4 sliding → donor 2
        Assert.Equal(3, cfg.SharedKvDonorLayer(7));  // layer 7 global → donor 3
    }

    [Fact]
    public void Forward_E4BLike_FiniteNonDegenerate_SoftcapClamped()
    {
        string path = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        float[] logits = RunLogits(path, cfg => cfg);

        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        float min = logits.Min(), max = logits.Max();
        Assert.True(max > min, "logits degenerate (constant).");
        Assert.True(min > -30.0f && max < 30.0f,
            $"final-logit soft-cap did not clamp: min={min}, max={max}");
    }

    [Fact]
    public void Ple_ChangesOutput_VsPleDisabled()
    {
        // Same file loaded twice — once with PerLayerEmbedding (tables loaded +
        // injected per layer) and once with it null (PLE tensors ignored).
        string path = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        float[] withPle = RunLogits(path, cfg => cfg);
        float[] noPle = RunLogits(path, cfg => cfg with { PerLayerEmbedding = null });

        Assert.True(MaxAbsDiff(withPle, noPle) > 1e-4f,
            "PLE injection had no measurable effect on the forward output.");
    }

    [Fact]
    public void SharedKv_IgnoresSharedLayersOwnKvWeights()
    {
        // Two fixtures identical except the shared layers' (dead) K/V weights:
        // random in A, all-zero in B. Shared layers must attend over the DONOR's
        // KV, so the logits must be bit-identical. A forward that wrongly computed
        // the shared layers' own K/V would diverge hard on B.
        string a = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        string b = WriteFixture(SyntheticGemma4Gguf.E4BLike with { ZeroSharedKvWeights = true });

        float[] la = RunLogits(a, cfg => cfg);
        float[] lb = RunLogits(b, cfg => cfg);
        Assert.Equal(la, lb);
    }

    [Fact]
    public void SharedKv_ChangesOutput_VsNoSharing()
    {
        // Same tensor bytes; config with sharing disabled uses the shared layers'
        // own (random) K/V weights instead of the donor KV → logits must differ.
        string path = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        float[] shared = RunLogits(path, cfg => cfg);
        float[] unshared = RunLogits(path, cfg => cfg with { NumSharedKvLayers = 0 });

        Assert.True(MaxAbsDiff(shared, unshared) > 1e-4f,
            "shared-KV reuse had no measurable effect vs per-layer KV.");
    }

    [Fact]
    public void RopeFreqFactors_ChangeOutput_ForNonZeroPositions_OnlyGlobalLayers()
    {
        // rope_freqs modulates the GLOBAL layers' rotation angle (angle/ff). At
        // position 0 every angle is 0, so a single BOS-position forward must be
        // IDENTICAL with and without the tensor; a multi-position forward must
        // differ. This pins both the wiring and the "full layers only" scope.
        string with = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        string without = WriteFixture(SyntheticGemma4Gguf.E4BLike with { EmitRopeFreqs = false });

        float[] p0With = RunLogits(with, cfg => cfg, [2], [0]);
        float[] p0Without = RunLogits(without, cfg => cfg, [2], [0]);
        Assert.Equal(p0With, p0Without);

        float[] seqWith = RunLogits(with, cfg => cfg);
        float[] seqWithout = RunLogits(without, cfg => cfg);
        Assert.True(MaxAbsDiff(seqWith, seqWithout) > 1e-4f,
            "rope_freqs factors had no measurable effect on multi-position logits.");
    }

    [Fact]
    public void KvCacheDecode_MatchesCachelessForward_AcrossSharedLayers()
    {
        // Prefill-then-decode through SimpleKvCache must reproduce the cacheless
        // oracle. This exercises the donor-KV read from the CACHE (shared layers
        // never Update; they read the donor layer's lines) plus the per-layer
        // strided dual-head-dim geometry.
        string path = WriteFixture(SyntheticGemma4Gguf.E4BLike);
        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, cfg);
        int vocab = cfg.VocabSize;
        int last = Ids.Length - 1;

        float[] cachelessLast;
        using (ITensor logits = model.Forward(Ids, Pos, deviceId: -1, kvCache: null))
            cachelessLast = LastRow(logits, vocab);

        float[] decodeLast;
        using (var kv = new SimpleKvCache(KvGeometry.FromConfig(cfg), maxSeqLen: 16))
        {
            using (model.Forward(Ids.AsSpan(0, last), Pos.AsSpan(0, last), deviceId: -1, kvCache: kv)) { }
            using ITensor logits = model.Forward(Ids.AsSpan(last, 1), Pos.AsSpan(last, 1), deviceId: -1, kvCache: kv);
            decodeLast = LastRow(logits, vocab);
        }

        Assert.Equal(ArgMax(cachelessLast), ArgMax(decodeLast));
        const float absTol = 6.0e-2f, relTol = 5.0e-3f;
        for (int c = 0; c < vocab; c++)
        {
            float a = cachelessLast[c], b = decodeLast[c];
            Assert.True(MathF.Abs(a - b) <= absTol + relTol * MathF.Abs(a),
                $"col {c}: cacheless={a:F6} vs decode={b:F6}");
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    private string WriteFixture(SyntheticGemma4Config cfg)
    {
        string path = Path.Combine(_scratch, $"e4b-{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, cfg);
        return path;
    }

    private float[] RunLogits(string path, Func<ModelConfig, ModelConfig> mutate,
                              int[]? ids = null, int[]? pos = null)
    {
        ids ??= Ids;
        pos ??= Pos;
        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = mutate(GgufModelConfigExtractor.Extract(gguf.Metadata));
        // Single-threaded → deterministic reduction order, so the bit-exact
        // discriminator assertions (Assert.Equal on float[]) are stable.
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, DotLLM.Core.Configuration.ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(ids, pos, deviceId: -1);
        return CopyAll(logits);
    }

    private static unsafe float[] CopyAll(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        float[] row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }
}
