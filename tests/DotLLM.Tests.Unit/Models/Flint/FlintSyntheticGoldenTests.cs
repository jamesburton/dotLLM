using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Flint;

/// <summary>
/// Hermetic Flint CPU tests (#1158): a random-weight <c>flint</c> GGUF is written by <see cref="SyntheticFlintGguf"/> (no fixture
/// files) and checked against golden logits produced by llama.cpp CPU (F32 KV, no flash attention) on the SAME bytes, with a build
/// that includes ggml-org/llama.cpp#30928 (the Flint arch). HF transformers has no Flint modeling code, so llama.cpp is the oracle.
/// </summary>
/// <remarks>
/// Regenerating the goldens: <c>DOTLLM_FLINT_SYNTH_WRITE=&lt;dir&gt; dotnet test --filter WriteSyntheticFixture</c>, then
/// <c>tools/mimo2-tiny-oracle/dumplogits.exe &lt;dir&gt;/synthetic{,-q}.gguf ids.txt out.f32</c> (built against that llama.cpp) and
/// paste the base64 of each output.
/// </remarks>
public sealed class FlintSyntheticGoldenTests : IDisposable
{
    /// <summary>16 tokens: longer than the sliding window (4) so the window clips on the 3 sliding layers.</summary>
    internal static readonly int[] Ids = [3, 17, 8, 22, 5, 39, 1, 14, 30, 9, 26, 11, 2, 35, 20, 6];

    /// <summary>Quantised variant: Q4_1 routed banks (the official Flint file type) and a Q8_0 shared expert.</summary>
    internal static readonly SyntheticFlintGguf.Options Quantized = new()
    {
        RoutedQuant = QuantizationType.Q4_1,
        SharedQuant = QuantizationType.Q8_0,
    };

    /// <summary>Attention and dense-FFN projections Q4_1 (as the official file); experts F32 so nothing runs on Q8 activations.</summary>
    internal static readonly SyntheticFlintGguf.Options DenseQ4_1 = new() { DenseQuant = QuantizationType.Q4_1 };

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-flint-{Guid.NewGuid():N}");

    public FlintSyntheticGoldenTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, true); } catch { /* best effort */ }
    }

    private string Write(SyntheticFlintGguf.Options? o = null, string name = "m")
    {
        string p = Path.Combine(_scratch, name + ".gguf");
        File.WriteAllBytes(p, SyntheticFlintGguf.Build(o));
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

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Config_Maps_Flint_Keys()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig c = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Flint, c.Architecture);
        Assert.Equal(SyntheticFlintGguf.Layers, c.NumLayers);
        Assert.Equal(SyntheticFlintGguf.HeadDim, c.HeadDim);
        Assert.Equal(SyntheticFlintGguf.Kv, c.NumKvHeads);
        Assert.Null(c.NumGlobalKvHeads);
        Assert.Equal([null, null, 4, 4, null, 4], c.PerLayerSlidingWindow!.ToArray());
        Assert.Equal([0, 1, 4], c.NoRopeLayers!.ToArray());
        Assert.Null(c.GlobalRoPEConfig);   // every full layer is NoPE
        Assert.Equal(SyntheticFlintGguf.HeadDim, c.RoPEConfig!.Value.DimensionCount);
        Assert.Equal(RoPEType.NeoX, c.RoPEConfig.Value.Type);
        Assert.Equal(1f / MathF.Sqrt(SyntheticFlintGguf.HeadDim), c.AttentionScoreScale(c.HeadDim));
        var moe = c.Moe!;
        Assert.True(moe.SigmoidGating);
        Assert.True(moe.NormalizeExpertWeights);
        Assert.Equal(SyntheticFlintGguf.SharedFf, moe.SharedExpertIntermediateSize);
        Assert.Equal(1, moe.NumSharedExperts);   // expert_shared_count = 2 is a width ratio, not two experts
        Assert.False(moe.HasSharedExpertGate);
        Assert.Equal(SyntheticFlintGguf.Experts, moe.NumExperts);
        Assert.Equal(SyntheticFlintGguf.TopK, moe.NumExpertsPerTok);
        Assert.False(moe.IsMoeLayer(0));
        Assert.False(moe.IsMoeLayer(1));
        Assert.True(moe.IsMoeLayer(2));
        Assert.True(moe.IsMoeLayer(5));
    }

    [Fact]
    public void Forward_Matches_LlamaCpp_Golden()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] ours = Run(gguf, cfg);
        Assert.Equal(Golden.Length, ours.Length);
        float diff = MaxAbs(ours, Golden);
        Assert.True(diff < 1e-4f, $"max |dotLLM - llama.cpp| = {diff}");
    }

    [Fact]
    public void Quantized_Forward_Matches_LlamaCpp_Golden()
    {
        using var gguf = GgufFile.Open(Write(Quantized, "q"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] ours = Run(gguf, cfg);
        Assert.Equal(GoldenQuantized.Length, ours.Length);
        float diff = MaxAbs(ours, GoldenQuantized);
        Assert.True(diff < QuantizedTolerance, $"max |dotLLM - llama.cpp| = {diff}");
    }

    /// <summary>
    /// Attention and dense FFN in Q4_1 (the official file's layout). Prefill must equal llama.cpp run on the SAME weights dequantised
    /// to F32 (<c>llama-quantize --allow-requantize ... F32</c>): dotLLM's batched Q4_1 GEMM multiplies the exact dequantised weights
    /// by F32 activations. llama.cpp's own Q4_1 path quantises the activations to Q8_1 and lands far from both (KL 6e-3 at 512
    /// tokens, head_dim 128), which is why the real-model KL against llama.cpp is not a parity metric for Flint (#1158). The routed
    /// and shared experts run on Q8_1 activations in both engines (the packed dot), so they are covered by the quantised golden.
    /// </summary>
    [Fact]
    public void DenseQ4_1_Prefill_Matches_Exact_Dequantised_Reference()
    {
        using var gguf = GgufFile.Open(Write(DenseQ4_1, "denseq"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] ours = Run(gguf, cfg);
        Assert.Equal(GoldenDenseQ4_1Exact.Length, ours.Length);
        float diff = MaxAbs(ours, GoldenDenseQ4_1Exact);
        Assert.True(diff < 1e-4f, $"max |dotLLM - exact| = {diff}");
    }

    [Fact]
    public void Quantized_Experts_Stay_Raw_And_Threaded_Matches_Serial()
    {
        using var gguf = GgufFile.Open(Write(Quantized, "q"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using (var weights = TransformerWeights.LoadFromGguf(gguf, cfg))
        {
            for (int layer = SyntheticFlintGguf.LeadingDense; layer < SyntheticFlintGguf.Layers; layer++)
            {
                var moe = weights.Layers[layer].Moe!;
                Assert.All(moe.W1, p => Assert.Equal(0, p));                // no F32 routed copy (rule 2)
                Assert.All(moe.SharedGateProj, p => Assert.Equal(0, p));    // no F32 shared copy either
                Assert.Equal(QuantizationType.Q4_1, moe.GateExpsRawQt);
                Assert.Equal(QuantizationType.Q8_0, moe.SharedGateRawQt);
            }
        }
        float diff = MaxAbs(Run(gguf, cfg, threading: new ThreadingConfig(4)), Run(gguf, cfg));
        Assert.True(diff < 1e-4f, $"threaded vs serial {diff}");
    }

    [Fact]
    public void KvCacheDecode_Equals_BatchedForward()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float diff = MaxAbs(Run(gguf, cfg, decode: true), Run(gguf, cfg));
        Assert.True(diff < 1e-4f, $"decode vs batched {diff}");
    }

    // ---- each Flint mechanism must be observable: a mutant of it has to move the logits far beyond the parity tolerance ----

    [Theory]
    [InlineData("nope-off")]
    [InlineData("window")]
    [InlineData("no-shared-expert")]
    [InlineData("no-renorm")]
    public void Mutant_Moves_Logits(string mutant)
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ModelConfig m = mutant switch
        {
            "nope-off" => cfg with { NoRopeLayers = null },
            "window" => cfg with { PerLayerSlidingWindow = cfg.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : 4096).ToArray() },
            "no-shared-expert" => cfg with { Moe = cfg.Moe! with { SharedExpertIntermediateSize = null, NumSharedExperts = 0 } },
            _ => cfg with { Moe = cfg.Moe! with { NormalizeExpertWeights = false } },
        };
        float diff = MaxAbs(Run(gguf, m), Golden);
        Assert.True(diff > 0.02f, $"mutant '{mutant}' is invisible to the golden (max diff {diff})");
    }

    [Fact]
    public void Sinks_Are_Read_And_Change_The_Output()
    {
        using var withSinks = GgufFile.Open(Write(name: "sinks"));
        using var noSinks = GgufFile.Open(Write(new SyntheticFlintGguf.Options { Sinks = false }, "nosinks"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(withSinks.Metadata);
        float diff = MaxAbs(Run(withSinks, cfg), Run(noSinks, cfg));
        Assert.True(diff > 0.02f, $"attention sinks have no effect on the CPU forward (max diff {diff})");
    }

    [Fact]
    public void Gpu_Backends_Refuse_Loudly()
    {
        using var gguf = GgufFile.Open(Write());
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var vk = Assert.Throws<NotSupportedException>(() => DotLLM.Vulkan.VulkanTransformerModel.RejectUnsupportedArchitecture(cfg));
        Assert.Contains("Flint", vk.Message);
        Assert.Contains("issues/1157", vk.Message);
        var cu = Assert.Throws<NotSupportedException>(() => DotLLM.Cuda.CudaTransformerModel.RejectUnsupportedArchitecture(cfg));
        Assert.Contains("Flint", cu.Message);
    }

    [Fact]
    public void Tokenizer_Loads_Flint_PreType()
    {
        using var gguf = GgufFile.Open(Write());
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Assert.NotEmpty(tok.Encode("abc"));
    }

    [SkippableFact]
    public void WriteSyntheticFixture()
    {
        string? dir = Environment.GetEnvironmentVariable("DOTLLM_FLINT_SYNTH_WRITE");
        Skip.If(string.IsNullOrEmpty(dir), "generator for the goldens - run by hand with DOTLLM_FLINT_SYNTH_WRITE=<dir>");
        Directory.CreateDirectory(dir!);
        File.WriteAllBytes(Path.Combine(dir!, "synthetic.gguf"), SyntheticFlintGguf.Build());
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-q.gguf"), SyntheticFlintGguf.Build(Quantized));
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-denseq.gguf"), SyntheticFlintGguf.Build(DenseQ4_1));
        File.WriteAllText(Path.Combine(dir!, "ids.txt"), string.Join(' ', Ids));

        // Real-scale attention probe (not a golden): head_dim 128 over a 512-token prompt, as the real model runs.
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-hd128.gguf"), SyntheticFlintGguf.Build(new SyntheticFlintGguf.Options { HeadDim = 128, Window = 513 }));
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-hd128-q.gguf"), SyntheticFlintGguf.Build(new SyntheticFlintGguf.Options
        {
            HeadDim = 128, Window = 513,
            DenseQuant = QuantizationType.Q4_1, RoutedQuant = QuantizationType.Q4_1, SharedQuant = QuantizationType.Q4_1,
        }));
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-hd128-dense.gguf"), SyntheticFlintGguf.Build(new SyntheticFlintGguf.Options
        { HeadDim = 128, Window = 513, DenseQuant = QuantizationType.Q4_1 }));
        File.WriteAllBytes(Path.Combine(dir!, "synthetic-hd128-shared.gguf"), SyntheticFlintGguf.Build(new SyntheticFlintGguf.Options
        { HeadDim = 128, Window = 513, SharedQuant = QuantizationType.Q4_1 }));
        var rng = new Random(7);
        File.WriteAllText(Path.Combine(dir!, "ids512.txt"), string.Join(' ', Enumerable.Range(0, 512).Select(_ => rng.Next(SyntheticFlintGguf.Vocab))));
    }

    /// <summary>
    /// Both engines quantise the activations to Q8_1 / Q8_0 for the quantised banks the same way, so the packed Q4_1 x Q8_1 and
    /// Q8_0 x Q8_0 paths land within 8e-6 of llama.cpp (measured); the bound matches the F32 golden's.
    /// </summary>
    private const float QuantizedTolerance = 1e-4f;

    /// <summary>llama.cpp CPU F32 logits (16 x 40) for <see cref="Ids"/> on <c>SyntheticFlintGguf.Build()</c>.</summary>
    private static readonly float[] Golden = DecodeGolden(
        "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");

    /// <summary>llama.cpp CPU logits (16 x 40) for <see cref="Ids"/> on <c>SyntheticFlintGguf.Build(Quantized)</c>.</summary>
    private static readonly float[] GoldenQuantized = DecodeGolden(
        "QB0kPjJKqcBsN+A/+Og4wFjewD+WdbC+POsWQLEge0A6ygS/XEdYvvHXIb/XIgjA/MHSviPVK8BER5w/ioz6P0+JaL8CTcy+Vu5sQJ62fMCeuBJAnnGNvzyH" +
        "BMDBaXe/VDKhvhPGCkB/HKC/rQa0v9Tumj8QuJG97yEvwJaeRsBSRy6/CvWcwG2S3L87DIG/5uE6wJHkoL4C1gE/fv1ivweCSkDwEQtAUqaZPoDLrD0E4do+" +
        "AGoOwMxbsr+q5w+/HM7AP8RKSb9sSRS+t+A2v13m3L6i/hLA7jHLP2LyTL/gFeq/mjj+v2tVTkBAl7e84+PYP+C+HD9qeQTAfHwmPwASR776nM2/ZMhFQHA+" +
        "eD2EEaXAkN7svYygvb+H+a+/ViIAQMh/9j6g75HATkE4QCPQtEC8ZFTAcBtvQHp3xD6G8zC/Gue+P8amQsCtV6RA8G2yP7Cmwr2g0So9oOjzvyGPnL8hXeI/" +
        "Hu+kvwikkT8ASnK8Ylf5Pyj8ZkBE0wjAMFj5vwi0vj4VvuE/gqHOv8DKGL5LWhdAIUmVvzlhpj9RXyzAkwUQv4DVhb78vaG/UBBHQLtZqj4CQ6o/tjRmwI9d" +
        "oT6mdV2/mocEPrJdgr8w4lg/NFUuv6e+7D48gTHA/Ce6vhW6db9GKHc/pBssQCZSt78CGqi+hDK1P8zL2b+SoZK/yBe4P71P2L+wT7E/gLTKv+gzkT6+Jh8/" +
        "eBdywHxNR79noUm/g7Z2QHwNxb8W98Y/RELWP5ZPib9oj4k/KMi2voC0xD3oa64/yp4YPnwg8T7gGcg/5goKQJeklr9pdNc+QW8LwPQrZD+kDpk/QCVHPkzx" +
        "/L2FG7g/XwoYwI5Ig79ye4xAuLPTvzPvn0Bh3ZA+/VvHv4a4vT60sq+95teEPykzYkCoGuc/OPEjQHsmPL9MthzAznVJQBY7e8DIFQLAa9nFv3KIfkClkV2/" +
        "FhlLP+LqckCou1G9mDm/vdp/zL+3JOG/dg1MwIhbrT5wD9O8QtAOv4EReb/DkVbADbJJwKHWdr8S9rA+nn54P+L7HECCYpu/kllRQJYAiL+qql5AdN13v7HK" +
        "tb++CS2/TZpQQLCRScAiKPG/0+wVP9hkSz4CBpG+iu7nv0g/IcAk5mfA6v6ov5ZAH8A9uABA0dozwH5PD7+qQKY/9hn6Po4eM7+N0R1AdhSBv1SdgD63Esy/" +
        "tjSYP7dXWj8FNEA/rgGaP8A3w74oMb+/vI6cv9mTF7+YR0q/lO+MwEPW6z+UtcA+yOlQv8Zhtz9+PVM/yP0cwM1o3L9FS8O/DU3bPywMI0DYcKu/s3Xgvyz4" +
        "kb+Uy3+/kzKdv8z+zb9o86o/JBfbvwzQQj8ocM8/eDxLP08XB0AUE6S+hHsdPnTNjb/sb9g+xoe8vqhOD8Aw6Xu+r8cswGD2Lr0QxLa/hOA5QDBriED4cfs9" +
        "eoYLwNjQC8AKRqQ/hhVFv0awmr+t7Yw+ZYa5P7sC9z/kLivAjtYXwAxxRD/tLUhAYuhHwAcoCkCymCJAm8TwvxINUMA8nzQ/kPGQP6AQrTzASOU8OJThPY7q" +
        "ib/m+wJAxqFZQDRpaMDkBbM+8NGsPpsAAcA1Gt+/c3zpP6Ra9D+OFdS/Ok/cP87SB8CA9Tu94JADvyYm8D8Ut3xAHOOMPwyjIr8jrmC/qzmov2jTzr965TLA" +
        "NHHbPxgOf0A7hg1AXaFQwJhK/r44+SI/dbAOQJ/ZTMBsSzzAow6eQAA7pTxSwwZAj+UxQCz5Kr5w/QBAKKNbQGLUuL+Fe70/++yXP9KFlj+Me1fA2zFGwFiZ" +
        "ZcAnF/4/pqyZv0gFgT/L74tAXEJ9v6NOyj8uj76/AnQYQJbtV75koj0/tnsqQJt0J786GxG/8Pz7vfHIpL8aUJS/1zcjv60JHEBf7TPAXKULQFcWc8Ag/gy9" +
        "AEwBOw8yiUAsgWLA9hKWPwKs8D/Ch5g+zP2Qvn+qyj/ADsC/GOwOQCp+MD+4+Do/sMwjvq6FfkAAXTs/tXV7wCQUjL/iqE0/aqAPwOjgxT4cHqq/vG1HQMik" +
        "GL/A44c87nCqv0ah/D72OSTAlv0MQKx1XD7cq4q/zq9vveqfKD+uoZ6/jsGcP8y81T4jLyQ/GDybPzshbD/ZWmbA9qYQwHtc1z+AfPm9BKMAP+72JT5Iguu+" +
        "fAaRwIgHtD+tVJK/CQW+P3OkGkCgVx4/9AzJPqZ4KEDbwKK/Mb7NP7CSZL9AQIw+KNgHvnGpcT9PAiU/jiovP/yJD0CAk9i9vOfZvxwe77/bylm/iCWyvszs" +
        "jz4S044/fNxwPzho0j8ber0/RqifP/R3Dz/YNk4+K5yFQKATuz7sIQJA/N+KwNpUyD8Qow6/byO7P5vfYMD20KS/Ku6SP7Q+OMDnZZm/QKufvdudGUAyrTdA" +
        "KGCtvlj7fj5/jbM/PBynP4w7AkC0Z3LATBoUwJTu67/u/Im/Kik0QCpKn76wXrhAXq+UP6Ksvz+AGnW/RtQYQJOe5T9VNQ5AR4cUQJGNBkB2OTzAUGvbPsy5" +
        "KEBSMH4/PmBowB55jUCUazZAcAzOPjf9AcCveb0+J8qeP48tx0Aq41i/9BiZPRg15j0Ea/y/0GP2vlQ4Fr8VYOA/8DPHPy4mLj9/U1Y/0MhGPx5T0z+Isvk/" +
        "JptFwJcMjb84Qyu+GYeovwDQyD+8PUQ/fCqUQBdBXECx1/I/mA0lP6RCnb4JYfe/Fp8wvwJMzb9YlA6/HzJ0vlYUsD6dKDG/Qu/MPo8zrr+OixNAXA9eQDIp" +
        "5b4wDQA/WmOaPyq7Or+YsjQ+KcAowG6RS8D+kYdAvbhmP3PHzD8YTqC/2ZbgPzkbbT+SXzE/ppDAP/4v7T+AFss/Cc7hv4+KiD/uIYU/QJ6JP1gWAD63aINA" +
        "LPaWvwUO7D/EPZ9A5icCQLh7Tj/UQ/S+ZKIJwIi/PcAv0vI/1l93v5o8j768sGNAp4w2v7MZor/kioNAbGxuP5g+1z83zSpAxwNHwOQgFj9gNBg+JN6VPwH5" +
        "PsBm/QLArlyOP4id5j0YqT6/UBZUvlxzBMAt/Ne/kC/xvlyD6r9WjxjApKTsvvyZCcBUCrK/2001QETgC0DvPsK/bF8nPyc38L8AzKY57JseQGAQYb4Acvc7" +
        "Wz+IP9SnmT96mFlAo0E0QPQOLUCJJzW/9K8GQCnMg792bwLAHPOWQOpm27816cU/hNZ6QNEd6787K5m/gMbIvjAc+76MMYTAJv9KwA5ICEELfae/iK5qPfny" +
        "MkBWBDNAAHK6PzLqQ0AUavO+AOgePMX2JL8O+xfApjUuv53joL4A6Hg6zqZ/v37rHD8+lyJA6MWSviAvJT9rWUu/fNrHvuGphL9BYPm/zlMDP3/FTkAaE1rA" +
        "rcFjwE7tK0DsbAa/SeRRwP+EBsC6DP8/y6mVv6TmHUAuAYDAPBnzvg==");

    /// <summary>llama.cpp CPU logits (16 x 40) for <see cref="Ids"/> on <c>Build(DenseQ4_1)</c> requantised to F32 (exact dequantised weights).</summary>
    private static readonly float[] GoldenDenseQ4_1Exact = DecodeGolden(
        "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");

    private static float[] DecodeGolden(string b64)
    {
        byte[] b = Convert.FromBase64String(b64);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
}
