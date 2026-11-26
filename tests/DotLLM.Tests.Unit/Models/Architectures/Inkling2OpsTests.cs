using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Op-level pins for the Inkling 2 CPU reference (#2026). Each test restates one llama.cpp graph fragment (<c>src/models/inkling2.cpp</c>,
/// <c>build_moe_ffn</c> LLM_FFN_SITU, <c>llama_kv_cache::set_input_relative_pos</c>, <c>ggml_ssm_conv</c>) as a naive double-precision definition
/// and compares. Cases are chosen so the plausible wrong readings give different numbers (stated per test).
/// </summary>
public sealed class Inkling2OpsTests
{
    private static float[] Rand(int n, int seed, float scale = 1f)
    {
        var r = new Random(seed); var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    [Fact]
    public void RmsNorm_MatchesDefinition_WithAndWithoutGamma()
    {
        var x = Rand(48, 1, 3f); var g = Rand(48, 2);
        var y = new float[48]; var y2 = new float[48];
        Inkling2Ops.RmsNorm(x, g, 1e-6f, y);
        Inkling2Ops.RmsNorm(x, default, 1e-6f, y2);
        double ms = x.Sum(v => (double)v * v) / 48;
        for (int i = 0; i < 48; i++)
        {
            Assert.Equal(x[i] / Math.Sqrt(ms + 1e-6) * g[i], y[i], 1e-5);
            Assert.Equal(x[i] / Math.Sqrt(ms + 1e-6), y2[i], 1e-5);
        }
    }

    [Fact]
    public void ClipL2_AppliesScaleBeforeClipping_AndOnlyShrinks()
    {
        // |x| = 5 (3,4): threshold 10 leaves it alone, threshold 2.5 halves it; a channel scale of 2 makes |x| = 10 BEFORE the clip.
        var small = new float[] { 3f, 4f };
        Inkling2Ops.ClipL2(small, default, 10f);
        Assert.Equal(3f, small[0], 1e-6); Assert.Equal(4f, small[1], 1e-6);

        var shrink = new float[] { 3f, 4f };
        Inkling2Ops.ClipL2(shrink, default, 2.5f);
        Assert.Equal(1.5f, shrink[0], 1e-5); Assert.Equal(2f, shrink[1], 1e-5);

        // scale first: (3,4)*2 = (6,8), |.| = 10 > 8 -> clipped to norm 8 = (4.8, 6.4). Clipping first (norm 5 < 8) then scaling would give (6, 8).
        var scaled = new float[] { 3f, 4f };
        Inkling2Ops.ClipL2(scaled, new float[] { 2f, 2f }, 8f);
        Assert.Equal(4.8f, scaled[0], 1e-5); Assert.Equal(6.4f, scaled[1], 1e-5);
        Assert.Equal(8.0, Math.Sqrt(scaled[0] * scaled[0] + scaled[1] * scaled[1]), 1e-5);
    }

    [Theory]
    [InlineData(0.5f, 2f)]
    [InlineData(-3f, 10f)]
    [InlineData(40f, -60f)]       // far into both softcaps
    public void Situ_MatchesDefinition(float gate, float up)
    {
        double beta = 4, lb = 25;
        double expect = beta * Math.Tanh(gate / beta) * (1.0 / (1.0 + Math.Exp(-gate))) * (lb * Math.Tanh(up / lb));
        Assert.Equal(expect, Inkling2Ops.Situ(gate, up, 4f, 25f), 1e-4);
    }

    [Fact]
    public void Situ_UpSoftcapIsOptional_AndSaturates()
    {
        Assert.Equal(4.0 * Math.Tanh(1.0 / 4) * (1 / (1 + Math.Exp(-1.0))) * 100.0, Inkling2Ops.Situ(1f, 100f, 4f, 0f), 1e-3);   // no up cap: linear in up
        Assert.True(Inkling2Ops.Situ(1f, 1e6f, 4f, 25f) < 25f * 4f);                                                              // capped: |up part| <= lb
    }

    [Fact]
    public void DepthMix_IsSoftmaxWeightedSumOfTheStates_PerQuery()
    {
        int n = 8, count = 3, nq = 2;
        var states = Rand(count * n, 3); var q = Rand(nq * n, 4, 2f);
        var o = new float[nq * n];
        Inkling2Ops.DepthMixToken(states, count, q, nq, n, o);
        for (int k = 0; k < nq; k++)
        {
            var s = new double[count];
            for (int i = 0; i < count; i++) { double d = 0; for (int j = 0; j < n; j++) d += states[i * n + j] * q[k * n + j]; s[i] = d / Math.Sqrt(n); }
            double m = s.Max(), z = s.Sum(v => Math.Exp(v - m));
            for (int j = 0; j < n; j++)
            {
                double e = 0; for (int i = 0; i < count; i++) e += Math.Exp(s[i] - m) / z * states[i * n + j];
                Assert.Equal(e, o[k * n + j], 1e-5);
            }
        }
        // the state order does not matter (llama.cpp stacks them last-first)
        var rev = new float[count * n];
        for (int i = 0; i < count; i++) Array.Copy(states, i * n, rev, (count - 1 - i) * n, n);
        var o2 = new float[nq * n];
        Inkling2Ops.DepthMixToken(rev, count, q, nq, n, o2);
        for (int i = 0; i < o.Length; i++) Assert.Equal(o[i], o2[i], 1e-5);
    }

    [Fact]
    public void ControlCombine_IsForgetThenGain_WithTanh()
    {
        var cur = new float[] { 1f, -2f }; var msg = new float[] { 10f, 20f }; var gain = new float[] { 0.5f, -1f }; var forget = new float[] { 2f, 0f };
        var o = new float[2];
        Inkling2Ops.ControlCombine(cur, msg, gain, forget, o);
        Assert.Equal((1 - Math.Tanh(2.0) * 1) + Math.Tanh(0.5) * 10, o[0], 1e-5);
        Assert.Equal((-2 - 0.0) + Math.Tanh(-1.0) * 20, o[1], 1e-5);     // forget = tanh(0) = 0 leaves cur untouched
    }

    [Fact]
    public void RelativeBias_LooksUpByDistance_AndIsZeroBeyondTheExtent()
    {
        // table rows: d=0 -> (1,0), d=1 -> (0,1), d=2 -> (1,1); r = (2, 3)
        var table = new float[] { 1, 0, 0, 1, 1, 1 };
        var r = new float[] { 2f, 3f };
        Assert.Equal(2f, Inkling2Ops.RelativeBias(table, 3, 2, r, 0));
        Assert.Equal(3f, Inkling2Ops.RelativeBias(table, 3, 2, r, 1));
        Assert.Equal(5f, Inkling2Ops.RelativeBias(table, 3, 2, r, 2));      // last row, distance extent-1
        Assert.Equal(0f, Inkling2Ops.RelativeBias(table, 3, 2, r, 3));      // distance == extent: the zero pad row, NOT a wrap or a clamp to the last row
        Assert.Equal(0f, Inkling2Ops.RelativeBias(table, 3, 2, r, 5000));
    }

    [Fact]
    public void LogScale_IsOneBelowTheFloor_AndGrowsLogarithmicallyAfter()
    {
        Assert.Equal(1f, Inkling2Ops.LogScale(0, 8192, 0.1f));
        Assert.Equal(1f, Inkling2Ops.LogScale(8190, 8192, 0.1f));           // (pos+1)/floor < 1 is clamped to 1
        Assert.Equal(1f, Inkling2Ops.LogScale(8191, 8192, 0.1f), 1e-6);
        Assert.Equal(1.0 + 0.1 * Math.Log(2.0), Inkling2Ops.LogScale(16383, 8192, 0.1f), 1e-5);
        Assert.Equal(1.0 + 0.1 * Math.Log(8.0), Inkling2Ops.LogScale(65535, 8192, 0.1f), 1e-5);
    }

    [Fact]
    public void SelectTopK_TiesResolveToTheLowerIndex()
    {
        var idx = new int[3];
        Inkling2TransformerModel.SelectTopK(new float[] { 1f, 5f, 5f, 3f, 5f, 2f }, 3, idx);
        Assert.Equal(new[] { 1, 2, 4 }, idx);
    }

    // ───────────── short convolution ─────────────

    /// <summary>Naive reference: <c>(x_t + sum_i x_{t-(K-1)+i} w[i] + [dyn] sum_lag x_{t-lag} d[lag]) / sqrt((dyn?2:1)*K+1)</c> over a zero-padded history.</summary>
    private static float[] NaiveConv(float[] x, int T, int C, float[] w, int K, float[]? dyn, int heads)
    {
        var o = new float[T * C];
        double scale = 1.0 / Math.Sqrt((dyn is null ? 1 : 2) * K + 1);
        double X(int t, int c) => t < 0 ? 0 : x[t * C + c];
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++)
            {
                double acc = 0;
                for (int i = 0; i < K; i++) acc += X(t - (K - 1) + i, c) * w[c * K + i];
                if (dyn is not null)
                    for (int lag = 0; lag < K; lag++) acc += X(t - lag, c) * dyn[t * heads * K + (c / (C / heads)) * K + lag];
                o[t * C + c] = (float)((X(t, c) + acc) * scale);
            }
        return o;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ShortConv_MatchesNaive_AndStateCarriesAcrossCalls(bool dynamic)
    {
        int C = 8, K = 4, T = 7, heads = 2;
        var x = Rand(T * C, 5); var w = Rand(C * K, 6); var d = dynamic ? Rand(T * heads * K, 7) : null;
        var want = NaiveConv(x, T, C, w, K, d, heads);

        using var st = new Inkling2ConvState(C, K, Inkling2ConvStateStorage.F32);
        var got = new float[T * C];
        // 3 tokens, then 1, then 3: chunking must not change the result
        int[] cuts = [3, 1, 3]; int at = 0;
        foreach (int n in cuts)
        {
            st.Run(n, x.AsSpan(at * C, n * C), w, d is null ? default : d.AsSpan(at * heads * K, n * heads * K), heads, got.AsSpan(at * C, n * C));
            at += n;
        }
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], got[i], 1e-5);
    }

    [Fact]
    public void ShortConv_Tap0IsTheOldestInput_AndDynamicLag0IsTheCurrentOne()
    {
        // one channel, K=4; a unit impulse at t=0 shows where each tap lands. Static w=[1,0,0,0] reads x_{t-3}: the impulse reappears at t=3.
        int K = 4;
        var x = new float[] { 1f, 0f, 0f, 0f, 0f };
        using (var st = new Inkling2ConvState(1, K, Inkling2ConvStateStorage.F32))
        {
            var o = new float[5];
            st.Run(5, x, new float[] { 1f, 0f, 0f, 0f }, default, 1, o);
            double s = 1 / Math.Sqrt(K + 1);
            Assert.Equal(1 * s, o[0], 1e-6);                       // residual term only
            Assert.Equal(0, o[1], 1e-6); Assert.Equal(0, o[2], 1e-6);
            Assert.Equal(1 * s, o[3], 1e-6);                       // tap 0 -> three steps later
        }
        using (var st = new Inkling2ConvState(1, K, Inkling2ConvStateStorage.F32))
        {
            var o = new float[5];
            st.Run(5, x, new float[4], new float[] { 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f }, 1, o);
            // dynamic taps only on lag 3 (index 3 of every token's tap vector): the impulse reappears 3 tokens later; lag 0 would hit at t=0
            double s = 1.0 / 3.0;
            Assert.Equal(1 * s, o[0], 1e-6);
            Assert.Equal(1 * s, o[3], 1e-6);
            Assert.Equal(0, o[1], 1e-6);
        }
    }

    [Theory]
    [InlineData(Inkling2ConvStateStorage.F16, 2e-3)]
    [InlineData(Inkling2ConvStateStorage.BF16, 1.6e-2)]
    public void ShortConv_NarrowState_StaysWithinStorageRounding(Inkling2ConvStateStorage storage, double tol)
    {
        int C = 16, K = 4, T = 6;
        var x = Rand(T * C, 8); var w = Rand(C * K, 9, 0.5f);
        var want = NaiveConv(x, T, C, w, K, null, 1);
        using var st = new Inkling2ConvState(C, K, storage);
        var got = new float[T * C];
        st.Run(3, x.AsSpan(0, 3 * C), w, default, 1, got.AsSpan(0, 3 * C));
        st.Run(3, x.AsSpan(3 * C, 3 * C), w, default, 1, got.AsSpan(3 * C, 3 * C));
        double worst = 0;
        for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(want[i] - got[i]));
        Assert.True(worst < tol, $"worst abs error {worst}");
        Assert.True(worst > 0, "a narrow state must round something (otherwise the storage is not actually narrow)");
        Assert.True(st.Bytes == C * (K - 1) * 2);
    }

    // ───────────── config ─────────────

    private static Dictionary<string, GgufMetadataValue> ValidMeta(bool loop = true)
    {
        GgufMetadataValue U(uint v) => new(GgufValueType.UInt32, v);
        GgufMetadataValue F(float v) => new(GgufValueType.Float32, v);
        string a = "inkling2_loop";
        var d = new Dictionary<string, GgufMetadataValue>
        {
            ["general.architecture"] = new(GgufValueType.String, a),
            [$"{a}.block_count"] = U(12), [$"{a}.embedding_length"] = U(64),
            [$"{a}.attention.head_count"] = new(GgufValueType.Array, Enumerable.Repeat(8, 12).ToArray()),
            [$"{a}.attention.head_count_kv"] = new(GgufValueType.Array, Enumerable.Repeat(2, 12).ToArray()),
            [$"{a}.attention.key_length"] = U(16), [$"{a}.attention.value_length"] = U(16),
            [$"{a}.attention.layer_norm_rms_epsilon"] = F(1e-6f),
            [$"{a}.vocab_size"] = U(104), [$"{a}.vocab_size_unpadded"] = U(100),
            [$"{a}.expert_count"] = U(8), [$"{a}.expert_used_count"] = U(2), [$"{a}.expert_feed_forward_length"] = U(32),
            [$"{a}.attention.sliding_window"] = U(16),
            [$"{a}.attention.sliding_window_pattern"] = new(GgufValueType.Array, Enumerable.Range(0, 12).Select(i => i % 6 != 5).ToArray()),
            [$"{a}.shortconv.l_cache"] = U(4), [$"{a}.expert_weights_scale"] = F(8f),
            [$"{a}.activation.situ_beta"] = F(4f), [$"{a}.activation.situ_linear_beta"] = F(25f),
            [$"{a}.attn_res.block_size"] = U(6), [$"{a}.final_logit_softcapping"] = F(30f),
            [$"{a}.attention.relative_length"] = U(4), [$"{a}.attention.relative_extent"] = U(16),
            [$"{a}.attention.qk_clip"] = F(22.6f), [$"{a}.attention.relative_clip"] = F(8f),
            [$"{a}.attention.log_scale_floor"] = U(8192), [$"{a}.attention.log_scale_alpha"] = F(0.1f),
        };
        if (loop)
        {
            d[$"{a}.loop.start"] = U(6); d[$"{a}.loop.end"] = U(12); d[$"{a}.loop.count"] = U(3); d[$"{a}.loop.gate_rank"] = U(8);
            d[$"{a}.loop.varying"] = new(GgufValueType.Bool, false);
        }
        return d;
    }

    [Fact]
    public void Config_Extracts_AndKeepsPaddedAndUnpaddedVocabApart()
    {
        var cfg = GgufModelConfigExtractor.Extract(new GgufMetadata(ValidMeta()));
        Assert.Equal(Architecture.Inkling2, cfg.Architecture);
        Assert.Equal(100, cfg.VocabSize);                    // logits width = tokenizer vocabulary
        Assert.Equal(104, cfg.Inkling2!.PaddedVocabSize);    // embedding / output rows
        Assert.Equal(3, cfg.Inkling2.LoopPasses);
        Assert.Equal(6 + 2 * 6 + 0, cfg.Inkling2.CacheSlots(12) - 0 - 0 + 0 - 12 + 6 - 6 + 6 - 6 + 0);  // 12 + 2*6 = 24 cache slots
        Assert.Equal(24, cfg.Inkling2.CacheSlots(12));
        Assert.True(cfg.Inkling2.IsLocalLayer[0]); Assert.False(cfg.Inkling2.IsLocalLayer[5]);
    }

    [Fact]
    public void Config_WithoutLoopKeys_IsASinglePass()
    {
        var cfg = GgufModelConfigExtractor.Extract(new GgufMetadata(ValidMeta(loop: false)));
        Assert.Equal(1, cfg.Inkling2!.LoopPasses);
        Assert.Equal(12, cfg.Inkling2.CacheSlots(12));
    }

    [Fact]
    public void Config_RefusesWhatItDoesNotImplement()
    {
        var d = ValidMeta(); d["inkling2_loop.loop.varying"] = new(GgufValueType.Bool, true);
        Assert.Throws<NotSupportedException>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));

        d = ValidMeta(); d["inkling2_loop.rope.dimension_count"] = new(GgufValueType.UInt32, 16u);
        Assert.Throws<NotSupportedException>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));

        d = ValidMeta(); d["inkling2_loop.loop.start"] = new(GgufValueType.UInt32, 5u);            // not on a block boundary
        Assert.Throws<InvalidDataException>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));

        d = ValidMeta(); var heads = Enumerable.Repeat(8, 12).ToArray(); heads[3] = 4;              // per-layer head counts that differ
        d["inkling2_loop.attention.head_count"] = new(GgufValueType.Array, heads);
        Assert.Throws<NotSupportedException>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));

        d = ValidMeta(); d.Remove("inkling2_loop.attention.qk_clip");
        Assert.ThrowsAny<Exception>(() => GgufModelConfigExtractor.Extract(new GgufMetadata(d)));
    }

    [Fact]
    public void HfInkling2Config_IsRecognised_AndRefusedExplicitly()
    {
        var ex = Assert.Throws<NotSupportedException>(() => HfConfigExtractor.Extract("""{"architectures":["Inkling2LoopForConditionalGeneration"],"model_type":"inkling2_loop"}"""));
        Assert.Contains("inkling2", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("GGUF", ex.Message, StringComparison.Ordinal);
    }
}
