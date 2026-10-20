using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1037 — Nemotron 3.5 Bolt (a Puzzle/NAS derivative of nemotron_h_moe). Two features
/// Lightning does not have, both exercised end to end through a synthetic GGUF, the real
/// config extractor and the real CPU loader:
/// <list type="bullet">
///   <item><b>Heterogeneous per-layer MoE geometry</b>: <c>expert_used_count</c> and
///   <c>expert_feed_forward_length</c> are per-layer arrays, and each layer's expert count comes
///   from its bank's <c>ne[2]</c> (<c>expert_count</c> is only the maximum).</item>
///   <item><b>LatentMoE</b>: the routed experts run on <c>ffn_latent_down(x)</c> and their weighted
///   sum goes through <c>ffn_latent_up</c>; the router and the shared expert read the full x.</item>
/// </list>
/// The reference is transcribed from llama.cpp b11739 <c>src/models/nemotron-h.cpp</c>
/// <c>build_ffn_layer</c>. The two layers differ in expert count (4 vs 6), top-k (2 vs 3) and
/// width (8 vs 6), so using the global maximum for any of them, or skipping either latent
/// projection, changes the logits or fails the load.
/// </summary>
public sealed class NemotronHBoltLatentMoeTests : IDisposable
{
    private const string Arch = "nemotron_h_moe";
    private const int Hidden = 8;
    private const int Latent = 4;
    private const int Vocab = 12;
    private const int SharedInter = 5;
    private const float Eps = 1e-5f;
    private const float WScale = 2.5f;

    // Per-layer geometry: layer 0 and layer 1 differ in every dimension.
    private static readonly int[] Experts = [4, 6];
    private static readonly int[] TopK = [2, 3];
    private static readonly int[] Inter = [8, 6];

    private static readonly int[] Ids = [3, 7, 1];
    private static readonly int[] Pos = [0, 1, 2];

    private readonly string _scratch;

    public NemotronHBoltLatentMoeTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-bolt-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Extract_PerLayerArraysAndLatent_PopulateMoeConfig()
    {
        var (path, _) = WriteFixture(seed: 1);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
        var moe = Assert.IsType<MoeConfig>(config.Moe);
        Assert.Equal(6, moe.NumExperts);
        Assert.Equal(3, moe.NumExpertsPerTok);          // maximum of the per-layer array
        Assert.Equal(8, moe.MoeIntermediateSize);       // maximum of the per-layer array
        Assert.Equal(Latent, moe.LatentSize);
        Assert.Equal(2, moe.ExpertsPerTokForLayer(0));
        Assert.Equal(3, moe.ExpertsPerTokForLayer(1));
        Assert.Equal(8, moe.IntermediateSizeForLayer(0));
        Assert.Equal(6, moe.IntermediateSizeForLayer(1));
    }

    [Fact]
    public void Extract_ScalarKeys_LeavePerLayerArraysNull()
    {
        // Lightning-shaped scalars must keep working, with the per-layer accessors falling back.
        var (path, _) = WriteFixture(seed: 1, scalarMoeKeys: true);
        using var gguf = GgufFile.Open(path);
        var moe = GgufModelConfigExtractor.Extract(gguf.Metadata).Moe!;

        Assert.Null(moe.PerLayerNumExpertsPerTok);
        Assert.Null(moe.PerLayerMoeIntermediateSize);
        Assert.Equal(3, moe.ExpertsPerTokForLayer(0));
        Assert.Equal(8, moe.IntermediateSizeForLayer(1));
    }

    [Fact]
    public unsafe void Forward_LatentHeterogeneousMoe_MatchesLlamaCppReference()
    {
        var (path, w) = WriteFixture(seed: 1037);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(Ids, Pos, deviceId: -1);
        float* lp = (float*)logits.DataPointer;

        float[][] expected = Reference(w);
        for (int t = 0; t < Ids.Length; t++)
            for (int i = 0; i < Vocab; i++)
            {
                float actual = lp[t * Vocab + i];
                Assert.True(Math.Abs(expected[t][i] - actual) <= 2e-4f,
                    $"pos {t} logit {i}: expected {expected[t][i]:G9}, got {actual:G9} — the Bolt "
                    + "LatentMoE / per-layer routing must match llama.cpp build_ffn_layer (#1037)");
            }
    }

    [Fact]
    public void Load_LatentTensorsWithoutLatentSize_Throws()
    {
        // ffn_latent_* present but no moe_latent_size: the banks' row width would be misread.
        var (path, _) = WriteFixture(seed: 1, omitLatentSizeKey: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        var ex = Assert.Throws<InvalidDataException>(
            () => NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        Assert.Contains("moe_latent_size", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void VulkanGuard_RefusesLatentAndHeterogeneousLayers()
    {
        Assert.Throws<NotSupportedException>(() =>
            DotLLM.Vulkan.VulkanNemotronHTransformerModel.ThrowIfMoeLayerUnsupported(0, hasLatent: true, null, (4, 2, 8)));
        Assert.Throws<NotSupportedException>(() =>
            DotLLM.Vulkan.VulkanNemotronHTransformerModel.ThrowIfMoeLayerUnsupported(1, hasLatent: false, (4, 2, 8), (6, 3, 6)));

        // Homogeneous, non-latent layers (Lightning) still pass.
        DotLLM.Vulkan.VulkanNemotronHTransformerModel.ThrowIfMoeLayerUnsupported(1, hasLatent: false, (4, 2, 8), (4, 2, 8));
    }

    // ───────────────────────── fixture ─────────────────────────

    /// <summary>Weights of the synthetic model, kept for the in-test reference.</summary>
    private sealed class Weights
    {
        public required float[] Embed, OutNorm, Output;
        public required LayerW[] Layers;
    }

    private sealed class LayerW
    {
        public required float[] Norm, Router, Bias, Up, Down, UpSh, DownSh, LatDown, LatUp;
    }

    private (string Path, Weights W) WriteFixture(
        int seed, bool scalarMoeKeys = false, bool omitLatentSizeKey = false)
    {
        var rng = new Random(seed);
        var g = new GgufWriter();
        g.AddString("general.architecture", Arch);
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32($"{Arch}.context_length", 64);
        g.AddUInt32($"{Arch}.embedding_length", Hidden);
        g.AddUInt32($"{Arch}.block_count", 2);
        g.AddUInt32($"{Arch}.attention.head_count", 2);
        g.AddInt32Array($"{Arch}.attention.head_count_kv", [0, 0]);
        g.AddInt32Array($"{Arch}.feed_forward_length", Inter);
        g.AddUInt32($"{Arch}.attention.key_length", 4);
        g.AddUInt32($"{Arch}.vocab_size", Vocab);
        g.AddFloat32($"{Arch}.attention.layer_norm_rms_epsilon", Eps);
        g.AddUInt32($"{Arch}.ssm.conv_kernel", 4);
        g.AddUInt32($"{Arch}.ssm.state_size", 4);
        g.AddUInt32($"{Arch}.ssm.group_count", 1);
        g.AddUInt32($"{Arch}.ssm.time_step_rank", 2);
        g.AddUInt32($"{Arch}.ssm.inner_size", 8);
        g.AddUInt32($"{Arch}.expert_count", 6);
        if (scalarMoeKeys)
        {
            g.AddUInt32($"{Arch}.expert_used_count", 3);
            g.AddUInt32($"{Arch}.expert_feed_forward_length", 8);
        }
        else
        {
            g.AddInt32Array($"{Arch}.expert_used_count", TopK);
            g.AddInt32Array($"{Arch}.expert_feed_forward_length", Inter);
        }
        g.AddUInt32($"{Arch}.expert_shared_feed_forward_length", SharedInter);
        g.AddUInt32($"{Arch}.expert_shared_count", 1);
        g.AddUInt32($"{Arch}.expert_gating_func", 2);
        g.AddBool($"{Arch}.expert_weights_norm", true);
        g.AddFloat32($"{Arch}.expert_weights_scale", WScale);
        g.AddUInt32($"{Arch}.expert_group_count", 1);
        g.AddUInt32($"{Arch}.expert_group_used_count", 1);
        if (!omitLatentSizeKey)
            g.AddUInt32($"{Arch}.moe_latent_size", Latent);

        float[] embed = Mat(g, rng, "token_embd.weight", Hidden, Vocab, 0.5f);
        var layers = new LayerW[2];
        for (int l = 0; l < 2; l++)
        {
            string p = $"blk.{l}";
            int n = Experts[l], ff = Inter[l];
            layers[l] = new LayerW
            {
                Norm = Norm(g, rng, $"{p}.attn_norm.weight", Hidden),
                Router = Mat(g, rng, $"{p}.ffn_gate_inp.weight", Hidden, n, 0.8f),
                Bias = Vec(g, rng, $"{p}.exp_probs_b.bias", n, 0.3f),
                Up = Bank(g, rng, $"{p}.ffn_up_exps.weight", Latent, ff, n),
                Down = Bank(g, rng, $"{p}.ffn_down_exps.weight", ff, Latent, n),
                UpSh = Mat(g, rng, $"{p}.ffn_up_shexp.weight", Hidden, SharedInter, 0.4f),
                DownSh = Mat(g, rng, $"{p}.ffn_down_shexp.weight", SharedInter, Hidden, 0.4f),
                LatDown = Mat(g, rng, $"{p}.ffn_latent_down.weight", Hidden, Latent, 0.6f),
                LatUp = Mat(g, rng, $"{p}.ffn_latent_up.weight", Latent, Hidden, 0.6f),
            };
        }
        float[] outNorm = Norm(g, rng, "output_norm.weight", Hidden);
        float[] output = Mat(g, rng, "output.weight", Hidden, Vocab, 0.5f);

        string path = Path.Combine(_scratch, $"bolt-{seed}-{scalarMoeKeys}-{omitLatentSizeKey}.gguf");
        File.WriteAllBytes(path, g.Build());
        return (path, new Weights { Embed = embed, OutNorm = outNorm, Output = output, Layers = layers });
    }

    // ───────────────────────── reference ─────────────────────────

    /// <summary>
    /// llama.cpp <c>build_ffn_layer</c> MoE branch per layer, with residual add, then the final
    /// norm and LM head. Matrices are row-major [out][in] (GGUF ne0 = in).
    /// </summary>
    private static float[][] Reference(Weights w)
    {
        var result = new float[Ids.Length][];
        bool flipSeen = false;
        for (int t = 0; t < Ids.Length; t++)
        {
            float[] x = w.Embed.AsSpan(Ids[t] * Hidden, Hidden).ToArray();
            for (int l = 0; l < 2; l++)
            {
                var L = w.Layers[l];
                int n = Experts[l], k = TopK[l], ff = Inter[l];
                float[] h = RmsNorm(x, L.Norm);

                float[] probs = MatVec(L.Router, h, n, Hidden).Select(Sigmoid).ToArray();
                int[] top = TopKIndices(probs.Select((p, e) => p + L.Bias[e]).ToArray(), k);
                if (!top.SequenceEqual(TopKIndices(probs, k))) flipSeen = true;
                float[] gate = top.Select(e => probs[e]).ToArray();
                float sum = gate.Sum();
                for (int j = 0; j < k; j++) gate[j] = gate[j] / MathF.Max(sum, 6.103515625e-5f) * WScale;

                float[] xl = MatVec(L.LatDown, h, Latent, Hidden);
                float[] accL = new float[Latent];
                for (int j = 0; j < k; j++)
                {
                    int e = top[j];
                    float[] mid = MatVec(L.Up.AsSpan(e * ff * Latent, ff * Latent).ToArray(), xl, ff, Latent)
                        .Select(Relu2).ToArray();
                    float[] o = MatVec(L.Down.AsSpan(e * Latent * ff, Latent * ff).ToArray(), mid, Latent, ff);
                    for (int i = 0; i < Latent; i++) accL[i] += gate[j] * o[i];
                }
                float[] moeOut = MatVec(L.LatUp, accL, Hidden, Latent);
                float[] sh = MatVec(L.DownSh, MatVec(L.UpSh, h, SharedInter, Hidden).Select(Relu2).ToArray(),
                    Hidden, SharedInter);
                for (int i = 0; i < Hidden; i++) x[i] += moeOut[i] + sh[i];
            }
            result[t] = MatVec(w.Output, RmsNorm(x, w.OutNorm), Vocab, Hidden);
        }
        // The selection bias must actually change selection somewhere, or "bias for selection
        // only" is not exercised on this geometry.
        Assert.True(flipSeen, "fixture produced no biased-selection flip — adjust the seed/bias scale");
        return result;
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));

    private static float Relu2(float v) { float r = MathF.Max(0f, v); return r * r; }

    private static int[] TopKIndices(float[] scores, int k)
    {
        var s = (float[])scores.Clone();
        int[] idx = new int[k];
        for (int j = 0; j < k; j++)
        {
            int best = 0;
            for (int e = 1; e < s.Length; e++) if (s[e] > s[best]) best = e;
            idx[j] = best;
            s[best] = float.NegativeInfinity;
        }
        return idx;
    }

    private static float[] RmsNorm(float[] x, float[] w)
    {
        double ss = 0;
        foreach (float v in x) ss += (double)v * v;
        float inv = (float)(1.0 / Math.Sqrt(ss / x.Length + Eps));
        return x.Select((v, i) => v * inv * w[i]).ToArray();
    }

    private static float[] MatVec(float[] m, float[] x, int rows, int cols)
    {
        float[] y = new float[rows];
        for (int r = 0; r < rows; r++)
        {
            double acc = 0;
            for (int c = 0; c < cols; c++) acc += (double)m[r * cols + c] * x[c];
            y[r] = (float)acc;
        }
        return y;
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return a;
    }

    /// <summary>2D F32 tensor [ne0 = inK, ne1 = outM], i.e. row-major [outM][inK].</summary>
    private static float[] Mat(GgufWriter g, Random rng, string name, int inK, int outM, float amp)
    {
        float[] a = Rand(rng, inK * outM, amp);
        g.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, Bytes(a));
        return a;
    }

    /// <summary>3D F32 expert bank [ne0 = inK, ne1 = outM, ne2 = experts].</summary>
    private static float[] Bank(GgufWriter g, Random rng, string name, int inK, int outM, int experts)
    {
        float[] a = Rand(rng, inK * outM * experts, 0.7f);
        g.AddTensor(name, [inK, outM, experts], (uint)QuantizationType.F32, Bytes(a));
        return a;
    }

    private static float[] Vec(GgufWriter g, Random rng, string name, int n, float amp)
    {
        float[] a = Rand(rng, n, amp);
        g.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(a));
        return a;
    }

    private static float[] Norm(GgufWriter g, Random rng, string name, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        g.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(a));
        return a;
    }

    private static byte[] Bytes(float[] a)
    {
        byte[] b = new byte[a.Length * sizeof(float)];
        Buffer.BlockCopy(a, 0, b, 0, b.Length);
        return b;
    }
}
