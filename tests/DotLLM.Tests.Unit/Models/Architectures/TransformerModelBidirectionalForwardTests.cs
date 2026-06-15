using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end coverage for the bidirectional / hybrid attention-mask seam (issue #26 PR-3) wired
/// into the cacheless CPU <see cref="TransformerModel"/> forward path. A tiny synthetic dense
/// (Llama-style) safetensors model is loaded and run under each mask mode.
/// <list type="bullet">
///   <item><b>Golden causal:</b> the default <c>Forward</c> and the mask-aware overload with the
///         Causal spec produce bit-identical logits.</item>
///   <item><b>Bidirectional:</b> mutating a future token changes an earlier position's logits
///         (proving <c>j&gt;i</c> attention) — and does NOT change them under Causal.</item>
///   <item><b>Hybrid:</b> mutating a canvas token leaves a prefix position's logits unchanged but
///         changes another canvas position's logits.</item>
///   <item><b>Sliding window under bidirectional:</b> a per-layer window measurably changes the
///         bidirectional output.</item>
/// </list>
/// </summary>
public sealed class TransformerModelBidirectionalForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 8
    private const int IntermediateSize = 24;

    private readonly string _scratch;

    public TransformerModelBidirectionalForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-bidi-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ───────────────────────── Golden causal identity ─────────────────────────

    [Fact]
    public void Causal_DefaultForward_ByteIdenticalToExplicitCausalSpec()
    {
        string path = Path.Combine(_scratch, "causal.safetensors");
        WriteFixture(path, seed: 42, tokenIds: [0, 1, 2, 3, 4]);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfg = BuildConfig();
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        float[] viaDefault;
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
            viaDefault = CopyLogits(l);

        float[] viaExplicit;
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1,
                                         kvCache: null, adapter: null, AttentionMaskSpec.Causal))
            viaExplicit = CopyLogits(l);

        Assert.Equal(viaDefault, viaExplicit); // exact bitwise equality
    }

    // ───────────────────────── Bidirectional ─────────────────────────

    [Fact]
    public void Bidirectional_FutureTokenChange_ChangesEarlierPositionLogits()
    {
        // Two token sequences identical except for the LAST token. Under Causal, position 0's
        // logits are invariant to the last token; under Bidirectional they must change.
        int[] tokensA = [1, 2, 3, 4, 5];
        int[] tokensB = [1, 2, 3, 4, 6]; // differs only at the final position
        int[] positions = [0, 1, 2, 3, 4];

        string path = Path.Combine(_scratch, "bidi.safetensors");
        WriteFixture(path, seed: 17, tokenIds: tokensA);

        ModelConfig cfg = BuildConfig();
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        // Causal: position 0 row identical between A and B.
        float[] causalA = Forward(model, tokensA, positions, AttentionMaskSpec.Causal);
        float[] causalB = Forward(model, tokensB, positions, AttentionMaskSpec.Causal);
        for (int j = 0; j < VocabSize; j++)
            Assert.Equal(causalA[j], causalB[j], 1e-5f);

        // Bidirectional: position 0 row must differ between A and B.
        float[] bidiA = Forward(model, tokensA, positions, AttentionMaskSpec.Bidirectional);
        float[] bidiB = Forward(model, tokensB, positions, AttentionMaskSpec.Bidirectional);
        float maxDiff = 0f;
        for (int j = 0; j < VocabSize; j++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(bidiA[j] - bidiB[j]));
        Assert.True(maxDiff > 1e-4f,
            $"Bidirectional position 0 logits did not react to a future-token change (maxDiff={maxDiff}).");
    }

    // ───────────────────────── Hybrid ─────────────────────────

    [Fact]
    public void Hybrid_CanvasTokenChange_LeavesPrefixLogitsUnchanged_ButChangesCanvasLogits()
    {
        // prefixLen = 3. Positions 0..2 are the causal prefix; 3..5 the canvas. Two sequences
        // differing only at a CANVAS position (index 5). Prefix logits (position 1) must be
        // identical; a different canvas position (3) must change.
        const int prefixLen = 3;
        int[] tokensA = [1, 2, 3, 4, 5, 6];
        int[] tokensB = [1, 2, 3, 4, 5, 7]; // differs only at canvas index 5
        int[] positions = [0, 1, 2, 3, 4, 5];

        string path = Path.Combine(_scratch, "hybrid.safetensors");
        WriteFixture(path, seed: 23, tokenIds: tokensA);

        ModelConfig cfg = BuildConfig();
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        AttentionMaskSpec hybrid = AttentionMaskSpec.Hybrid(prefixLen);
        float[] a = Forward(model, tokensA, positions, hybrid);
        float[] b = Forward(model, tokensB, positions, hybrid);

        // Prefix position 1 (< prefixLen): causal among prefix, cannot see canvas key 5 → unchanged.
        for (int j = 0; j < VocabSize; j++)
            Assert.Equal(a[1 * VocabSize + j], b[1 * VocabSize + j], 1e-5f);

        // Canvas position 3 (>= prefixLen): attends to canvas key 5 → must change.
        float maxDiff = 0f;
        for (int j = 0; j < VocabSize; j++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a[3 * VocabSize + j] - b[3 * VocabSize + j]));
        Assert.True(maxDiff > 1e-4f,
            $"Hybrid canvas position 3 logits did not react to a canvas-token change (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Hybrid_DiffersFromBothCausalAndBidirectional()
    {
        // Sanity that hybrid is a genuine third mode: with a partial prefix its logits differ
        // from both pure-causal and pure-bidirectional on the same input.
        const int prefixLen = 2;
        int[] tokenIds = [1, 2, 3, 4, 5];
        int[] positions = [0, 1, 2, 3, 4];

        string path = Path.Combine(_scratch, "hybrid3.safetensors");
        WriteFixture(path, seed: 71, tokenIds: tokenIds);

        ModelConfig cfg = BuildConfig();
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        float[] causal = Forward(model, tokenIds, positions, AttentionMaskSpec.Causal);
        float[] bidi = Forward(model, tokenIds, positions, AttentionMaskSpec.Bidirectional);
        float[] hybrid = Forward(model, tokenIds, positions, AttentionMaskSpec.Hybrid(prefixLen));

        Assert.True(MaxAbsDiff(hybrid, causal) > 1e-4f, "Hybrid collapsed onto Causal.");
        Assert.True(MaxAbsDiff(hybrid, bidi) > 1e-4f, "Hybrid collapsed onto Bidirectional.");
    }

    // ───────────────────────── Sliding window under bidirectional ─────────────────────────

    [Fact]
    public void Bidirectional_PerLayerSlidingWindow_ChangesOutput()
    {
        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        string path = Path.Combine(_scratch, "bidi-window.safetensors");
        WriteFixture(path, seed: 88, tokenIds: tokenIds);

        // Baseline: no sliding window, bidirectional.
        ModelConfig cfgFull = BuildConfig();
        float[] full;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgFull))
            full = Forward(model, tokenIds, positions, AttentionMaskSpec.Bidirectional);

        // With a window=2 sliding limit on every layer, bidirectional must change.
        ModelConfig cfgWin = cfgFull with
        {
            SlidingWindowSize = 2,
            PerLayerSlidingWindow = new int?[NumLayers] { 2, 2 },
        };
        float[] windowed;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgWin))
            windowed = Forward(model, tokenIds, positions, AttentionMaskSpec.Bidirectional);

        Assert.True(MaxAbsDiff(full, windowed) > 1e-4f,
            "Per-layer sliding window had no measurable effect under Bidirectional.");
    }

    [Fact]
    public void NonCausal_WithKvCache_Throws()
    {
        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        string path = Path.Combine(_scratch, "kvcache-guard.safetensors");
        WriteFixture(path, seed: 5, tokenIds: tokenIds);

        ModelConfig cfg = BuildConfig();
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, cfg);

        using var kv = new DotLLM.Engine.KvCache.SimpleKvCache(
            cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, cfg.MaxSequenceLength);

        Assert.Throws<NotSupportedException>(() =>
            model.Forward(tokenIds, positions, deviceId: -1, kv, adapter: null,
                          AttentionMaskSpec.Bidirectional));
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float[] Forward(IModel model, int[] tokenIds, int[] positions, AttentionMaskSpec spec)
    {
        using ITensor l = model.Forward(tokenIds, positions, deviceId: -1,
                                        kvCache: null, adapter: null, spec);
        return CopyLogits(l);
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float maxDiff = 0f;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a[i] - b[i]));
        return maxDiff;
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(
            Theta: 10000.0f,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        return new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = null,
            PerLayerSlidingWindow = null,
            MlaConfig = null,
            Moe = null,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic dense (Llama-style) safetensors fixture: embed_tokens, lm_head,
    /// model.norm, and per-layer input_layernorm, self_attn.{q,k,v,o}_proj,
    /// post_attention_layernorm (the standard two-norm layout), and mlp.{gate,up,down}_proj —
    /// all HF row-major F32. <paramref name="tokenIds"/> is accepted for signature symmetry with
    /// the per-test inputs but is not used for the weights (weights are seed-driven).
    /// </summary>
    private static void WriteFixture(string path, int seed, int[] tokenIds)
    {
        _ = tokenIds;
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim; // = HiddenSize
        int kvStride = NumHeads * HeadDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.2f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], amplitude: 1.0f, seed: seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.2f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize], amplitude: 1.0f, seed: s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], amplitude: 1.0f, seed: s + 1);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.2f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.2f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.2f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.2f, s + 5);

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.2f, s + 6);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.2f, s + 7);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.2f, s + 8);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        b.AddFloat32(name, shape, values);
    }
}
