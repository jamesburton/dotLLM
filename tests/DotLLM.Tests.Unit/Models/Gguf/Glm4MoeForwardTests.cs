using System.Security.Cryptography;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #1599 (S1 of the GLM-4.5-Air plan): hermetic CPU forward tests for <c>glm4moe</c>. The random-weight GGUF written by
/// <see cref="SyntheticGlm4MoeGguf"/> (no fixture files) is loaded into HuggingFace transformers <c>Glm4MoeForCausalLM</c> (F32, eager) by
/// <c>tests/scripts/glm4moe_synthetic_hf_reference.py</c>, and the resulting logits are the golden below: HF is the independent oracle for
/// the layer algebra (GQA + QKV bias, partial NeoX RoPE, dense layer 0, sigmoid + e_score_correction_bias routing with unbiased gathered
/// weights, norm-topk, the always-on shared expert). Every mechanism is observable: each mutant must move the logits far past the parity
/// tolerance (see <see cref="Mutant_Moves_Logits"/>).
/// </summary>
/// <remarks>
/// Regenerating the golden: <c>DOTLLM_GLM4MOE_SYNTH_WRITE=&lt;dir&gt; dotnet test --filter WriteSyntheticFixture</c>, then
/// <c>python tests/scripts/glm4moe_synthetic_hf_reference.py &lt;dir&gt;/synthetic.gguf</c> (needs torch + transformers &gt;= 4.56) and paste
/// the printed sha256 and base64 below. The sha256 pins the fixture bytes: changing <see cref="SyntheticGlm4MoeGguf"/> fails
/// <see cref="Fixture_Bytes_Match_The_Golden_Pin"/> instead of silently comparing against logits of different weights.
/// </remarks>
public sealed class Glm4MoeForwardTests : IDisposable
{
    private readonly Xunit.Abstractions.ITestOutputHelper _out;
    /// <summary>12 tokens (vocab 16): enough positions for RoPE phase to matter and for the causal mask to be exercised.</summary>
    internal static readonly int[] Ids = [3, 14, 8, 1, 5, 9, 0, 11, 2, 15, 7, 6];

    /// <summary>Parity bound vs HF F32 (measured ~1e-6; two orders under the weakest mutant). Tighter than DriftTolerances.Tight on purpose.</summary>
    private const float Tolerance = 2e-4f;

    /// <summary>Every mutant must move the logits by more than this (>= 20x the parity bound).</summary>
    private const float MutantFloor = 4e-3f;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-glm4moe-fwd-{Guid.NewGuid():N}");

    public Glm4MoeForwardTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, true); } catch { /* best effort */ }
    }

    private string Write(bool nextn = false, string name = "m")
    {
        string p = Path.Combine(_scratch, name + ".gguf");
        SyntheticGlm4MoeGguf.Write(p, withNextn: nextn);
        return p;
    }

    private static unsafe float[] Run(GgufFile gguf, ModelConfig cfg, bool decode = false, ThreadingConfig? threading = null)
    {
        int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, threading ?? ThreadingConfig.SingleThreaded);
        if (!decode)
        {
            using ITensor t = model.Forward(Ids, pos, deviceId: -1);
            return Copy(t);
        }
        var cache = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), 64);
        var all = new List<float>();
        for (int i = 0; i < Ids.Length; i++)
        {
            using ITensor t = model.Forward(Ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, cache);
            all.AddRange(Copy(t));
        }
        return all.ToArray();
    }

    /// <summary>Prefill through a KV cache in the given chunk sizes (the serving path: one prefill call, then decode).</summary>
    private static float[] RunChunked(GgufFile gguf, ModelConfig cfg, params int[] chunks)
    {
        int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        var cache = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), 64);
        var all = new List<float>();
        int at = 0;
        foreach (int n in chunks)
        {
            using ITensor t = model.Forward(Ids.AsSpan(at, n), pos.AsSpan(at, n), deviceId: -1, cache);
            all.AddRange(Copy(t));
            at += n;
        }
        Assert.Equal(Ids.Length, at);
        return all.ToArray();
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Fixture_Bytes_Match_The_Golden_Pin()
    {
        string hex = Convert.ToHexString(SHA256.HashData(SyntheticGlm4MoeGguf.Build(withNextn: false))).ToLowerInvariant();
        Assert.True(hex == FixtureSha256,
            $"SyntheticGlm4MoeGguf output changed (sha256 {hex}); regenerate the HF golden (see the class remarks) and update FixtureSha256 + Golden.");
    }

    [Fact]
    public void Forward_Matches_HuggingFace_Golden()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] ours = Run(gguf, cfg);
        Assert.Equal(Ids.Length * SyntheticGlm4MoeGguf.VocabSize, ours.Length);
        float diff = MaxAbs(ours, Golden);
        _out.WriteLine($"max |dotLLM - HF| = {diff}");
        Assert.True(diff < Tolerance, $"max |dotLLM - HF| = {diff}");
    }

    [Fact]
    public void NextnBlock_Is_Ignored_By_The_Trunk()
    {
        // The trunk file carries a trailing NextN block (blk.3.*) that is counted in block_count; the S1 forward must not run it.
        using var gguf = GgufFile.Open(Write(nextn: true, name: "n"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(1, cfg.NextnPredictLayers);
        Assert.Equal(3, cfg.NumLayers);
        // Same trunk tensors as the no-nextn file for the layers that run? The PRNG stream differs once the NextN block is appended,
        // but the trunk tensors are generated first, so the trunk logits are identical.
        float diff = MaxAbs(Run(gguf, cfg), Golden);
        Assert.True(diff < Tolerance, $"max |dotLLM(nextn file) - HF| = {diff}");
    }

    [Fact]
    public void KvCacheDecode_Equals_BatchedForward()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float diff = MaxAbs(Run(gguf, cfg, decode: true), Run(gguf, cfg));
        Assert.True(diff < Tolerance, $"decode vs batched {diff}");
    }

    [Theory]
    [InlineData(new[] { 12 })]
    [InlineData(new[] { 5, 7 })]
    [InlineData(new[] { 1, 11 })]
    [InlineData(new[] { 4, 4, 3, 1 })]
    public void CachedPrefill_Equals_HuggingFace_Golden(int[] chunks)
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float diff = MaxAbs(RunChunked(gguf, cfg, chunks), Golden);
        Assert.True(diff < Tolerance, $"cached prefill chunks [{string.Join(',', chunks)}] vs HF {diff}");
    }

    [Fact]
    public void Threaded_Equals_Serial()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float diff = MaxAbs(Run(gguf, cfg, threading: new ThreadingConfig(4)), Run(gguf, cfg));
        Assert.True(diff < Tolerance, $"threaded vs serial {diff}");
    }

    [Fact]
    public void Experts_And_SharedExpert_And_DenseLayer0_Stay_In_The_Raw_View()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, cfg, allowGlm4MoeCpu: true);
        Assert.Null(weights.Layers[0].Moe);                      // layer 0 is the dense SwiGLU FFN
        Assert.NotEqual(0, weights.Layers[0].GateWeight);
        for (int l = 1; l < cfg.NumLayers; l++)
        {
            var moe = weights.Layers[l].Moe!;
            Assert.True(moe.HasRawQuantView);
            Assert.All(moe.W1, p => Assert.Equal(0, p));         // no widened routed copy (CLAUDE.md rule 2)
            Assert.All(moe.W2, p => Assert.Equal(0, p));
            Assert.All(moe.W3, p => Assert.Equal(0, p));
            Assert.All(moe.SharedGateProj, p => Assert.Equal(0, p));   // no widened shared copy
            Assert.True(moe.SigmoidGating);
            Assert.NotNull(moe.SelectionBias);
            Assert.Equal(cfg.Moe!.SharedExpertIntermediateSize, moe.SharedIntermediateSize);
            Assert.Null(moe.SharedExpertGate);                   // ungated
        }
    }

    // ---- each glm4moe mechanism must be observable: a config mutant of it has to move the logits far beyond the parity tolerance ----

    [Theory]
    [InlineData("rope-all-dims")]        // rotate all 16 dims instead of the leading 8
    [InlineData("rope-norm-pairing")]    // interleaved (Norm) pairs instead of NeoX halves
    [InlineData("softmax-gating")]       // softmax instead of sigmoid scoring
    [InlineData("no-norm-topk")]         // skip the sum renormalisation of the selected weights
    [InlineData("no-shared-expert")]     // routed experts only
    [InlineData("topk-minus-one")]
    [InlineData("topk-plus-one")]
    [InlineData("weights-scale-2")]
    [InlineData("no-selection-bias")]    // exp_probs_b ignored
    public void Mutant_Moves_Logits(string mutant)
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var rope = cfg.RoPEConfig!.Value;
        ModelConfig m = mutant switch
        {
            "rope-all-dims" => cfg with { RoPEConfig = rope with { DimensionCount = cfg.HeadDim } },
            "rope-norm-pairing" => cfg with { RoPEConfig = rope with { Type = RoPEType.Norm } },
            "softmax-gating" => cfg with { Moe = cfg.Moe! with { SigmoidGating = false } },
            "no-norm-topk" => cfg with { Moe = cfg.Moe! with { NormalizeExpertWeights = false } },
            "no-shared-expert" => cfg with { Moe = cfg.Moe! with { SharedExpertIntermediateSize = null, NumSharedExperts = 0 } },
            "topk-minus-one" => cfg with { Moe = cfg.Moe! with { NumExpertsPerTok = cfg.Moe!.NumExpertsPerTok - 1 } },
            "topk-plus-one" => cfg with { Moe = cfg.Moe! with { NumExpertsPerTok = cfg.Moe!.NumExpertsPerTok + 1 } },
            "weights-scale-2" => cfg with { Moe = cfg.Moe! with { ExpertWeightsScale = 2f } },
            _ => cfg with { Moe = cfg.Moe! with { HasSelectionBias = false } },
        };
        float diff = MaxAbs(Run(gguf, m), Golden);
        _out.WriteLine($"mutant {mutant}: max diff {diff}");
        Assert.True(diff > MutantFloor, $"mutant '{mutant}' is invisible to the golden (max diff {diff})");
    }

    [SkippableFact]
    public void WriteSyntheticFixture()
    {
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_SYNTH_WRITE");
        Skip.If(string.IsNullOrEmpty(dir), "generator for the HF golden - run by hand with DOTLLM_GLM4MOE_SYNTH_WRITE=<dir>");
        Directory.CreateDirectory(dir!);
        File.WriteAllBytes(Path.Combine(dir!, "synthetic.gguf"), SyntheticGlm4MoeGguf.Build(withNextn: false));
        File.WriteAllText(Path.Combine(dir!, "ids.txt"), string.Join(' ', Ids));

        // Bisection aid: dotLLM logits after the first N trunk layers (compare with the HF model truncated to N layers).
        using var gguf = GgufFile.Open(Path.Combine(dir!, "synthetic.gguf"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        for (int n = 1; n <= cfg.NumLayers; n++)
        {
            float[] ours = Run(gguf, cfg with { NumLayers = n });
            var bytes = new byte[ours.Length * 4];
            Buffer.BlockCopy(ours, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir!, $"ours-{n}layers.f32"), bytes);
        }
    }

    private const string FixtureSha256 = "ed5975b2415b79872375e1d9fef64ca91af22bc2b83266ef0c69e69b23409d4f";

    /// <summary>HF transformers F32 logits (12 x 16) for <see cref="Ids"/> on <c>SyntheticGlm4MoeGguf.Build(withNextn: false)</c>.</summary>
    private static readonly float[] Golden = DecodeGolden(
        "nu+XvaxT1L1gRyq+pWIZvcqnKz7SmZS+72T8vYycUr1QFZi+l6kLvulTT74fQyU+D6ugvt4zaL52R2M9dEpoPo72oD7FIY2+HYOLvNOtJbx5ZHq7vU3jvYgM" +
        "Pr65QyE+/TksvhegSb1JRCU+IqxQvU9Ijb6Pw4K+91MJvtSVP71cBoQ85aVtvruOT77+oWC83dE4Pq3rqr5A9ka+q4GzvZNcnb5vVKq90T9avjE3W7xdGZm9" +
        "5O9mvtSvDr5XCW0+3eGRPnvzRb52Ww6+FS3iPQsO3b172cm+unOWvt9FgD05vZe8vu+fvZAURr6APjm+Q2YnvmXc/b3PoCi+N4sZPREnwrydROK94zkhvvQY" +
        "rj0CPwC9E2PcvmGxtr2tBIO8mf/SvUCZPL5nUNQ9xjYVPpKIHr5cTxG+0moLvpJe/L0huy++FP+DPTxu3L11jFo9XFEQPHJZh77xi4G+gnAdvZpsojxrjRA+" +
        "vtGTvjNPjL7ykyC+mdF+vW0EOr4YWhY7Z6aePdL9ITwOY4k9IDRxPX3BUj285W++kXWgvjwAMr2ClQO+mIQ5vVGsLr6q8zK+hTUMvmFtiL06UR88JUhuvXn2" +
        "4z0MI8A8puf4PW19ND3orxG7JRiRvsUfwb41KjW+/UO9veNy3L26LUq+4CYvvkMGQ72k5zq+qOWJvVOdGz0Bv2q+pbR1vQk+sb4ocRI+oLOFPb2kwr5GGEu7" +
        "c6ikvWTFFz7n8X28/asmvTyfPT5rPrG8mVt2vq7nnb3aNww9V1z1O/cAhr73mu+9/56ZvakZbb33b9G9UAeCvUE6K75AGom+XAApvjUtZ722hWE+q0m5vcD8" +
        "Eb8dY0+9JY2WPlhQUb3abwU+Lgu9O+LEmr2Dfpy+GmAXvlEftL6H9QI8dS0iPqVkrT5I9dO++3dsvv+yUj3NOQm9G8yxvcfcNz6jzIU9tw/auzVWXz2HjWg9" +
        "W3+mvP8qBL4hC8C+i6pKvjxhsr1YQsm9NitJvtUZob6pNPK8j+sXvvZJprx3crc9");

    private static float[] DecodeGolden(string b64)
    {
        byte[] bytes = Convert.FromBase64String(b64);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }
}
