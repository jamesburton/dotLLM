using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// Three-way parity on a tiny random MiMo-V2 checkpoint: HF torch (remote <c>modeling_mimo_v2.py</c>, eager attention),
/// llama.cpp CPU (F32 KV, no flash attention) and dotLLM CPU. The fixture is produced by
/// <c>tools/mimo2-tiny-oracle/make_tiny.py</c> (HF model + HF logits) → llama.cpp <c>convert_hf_to_gguf.py</c> → GGUF, and
/// <c>tools/mimo2-tiny-oracle/dumplogits.cpp</c> (llama.cpp logits). It is NOT in the repository (the vocabulary is the real
/// 152576-token one, 88 MB): it lives in <c>~/.dotllm/test-cache/mimo2-tiny/</c> (override with <c>DOTLLM_MIMO2_TINY_DIR</c>).
/// The hermetic equivalent that needs no fixture is <see cref="MiMoV2SyntheticGoldenTests"/>.
/// </summary>
/// <remarks>
/// Geometry (every ratio distinct on purpose): hidden 64, 8 Q heads, K head_dim 48 vs V head_dim 32, rotary 16 of 48,
/// global layers 2 KV heads (GQA 4) / sliding layers 4 KV heads (GQA 2), layers 0 4 global + 1 2 3 5 sliding (window 4,
/// non-zero sinks), 8 experts top-2 with a non-zero selection bias, layer 0 dense, V scale 0.707, 23 tokens (> window).
/// </remarks>
public sealed class MiMoV2TinyOracleParityTests
{
    private static string? Dir()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MIMO2_TINY_DIR");
        string d = !string.IsNullOrEmpty(env) ? env
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "mimo2-tiny");
        return File.Exists(Path.Combine(d, "tiny_mimo_f32.gguf")) ? d : null;
    }

    internal static float[] ReadF32(string path)
    {
        byte[] b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    internal static (float[] Logits, int Vocab, int N) RunDotLlm(string dir, Func<ModelConfig, ModelConfig>? mutate = null, bool decode = false, string file = "tiny_mimo_f32.gguf", ThreadingConfig? threading = null)
    {
        int[] ids = File.ReadAllText(Path.Combine(dir, "ids.txt")).Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int[] pos = Enumerable.Range(0, ids.Length).ToArray();
        using var gguf = GgufFile.Open(Path.Combine(dir, file));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (mutate is not null) cfg = mutate(cfg);
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, threading ?? ThreadingConfig.SingleThreaded);
        if (!decode)
        {
            using ITensor logits = model.Forward(ids, pos, deviceId: -1);
            return (Copy(logits), cfg.VocabSize, ids.Length);
        }
        // KV-cache decode path: prefill 1 token then feed one token at a time through a per-layer-strided cache.
        var cache = new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), 64);
        var all = new float[ids.Length * cfg.VocabSize];
        for (int i = 0; i < ids.Length; i++)
        {
            using ITensor logits = model.Forward(ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, cache);
            Copy(logits).CopyTo(all, i * cfg.VocabSize);
        }
        return (all, cfg.VocabSize, ids.Length);
    }

    private static unsafe float[] Copy(ITensor t)
    {
        int n = checked((int)t.Shape.ElementCount);
        var f = new float[n];
        new ReadOnlySpan<float>((void*)t.DataPointer, n).CopyTo(f);
        return f;
    }

    internal static (float MaxAbs, double MaxKl, double Top1) Compare(float[] a, float[] reference, int vocab)
    {
        float maxAbs = 0f; double maxKl = 0; int agree = 0, rows = a.Length / vocab;
        for (int r = 0; r < rows; r++)
        {
            var la = a.AsSpan(r * vocab, vocab); var lr = reference.AsSpan(r * vocab, vocab);
            for (int i = 0; i < vocab; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(la[i] - lr[i]));
            double ma = double.NegativeInfinity, mr = double.NegativeInfinity;
            int ia = 0, ir = 0;
            for (int i = 0; i < vocab; i++)
            {
                if (la[i] > ma) { ma = la[i]; ia = i; }
                if (lr[i] > mr) { mr = lr[i]; ir = i; }
            }
            double za = 0, zr = 0;
            for (int i = 0; i < vocab; i++) { za += Math.Exp(la[i] - ma); zr += Math.Exp(lr[i] - mr); }
            double lza = Math.Log(za) + ma, lzr = Math.Log(zr) + mr, kl = 0;
            for (int i = 0; i < vocab; i++)
            {
                double pr = Math.Exp(lr[i] - lzr);
                kl += pr * ((lr[i] - lzr) - (la[i] - lza));
            }
            maxKl = Math.Max(maxKl, kl);
            if (ia == ir) agree++;
        }
        return (maxAbs, maxKl, (double)agree / rows);
    }

    [SkippableFact]
    public void Config_Extracted_From_Llama_Converted_Gguf()
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny fixture not present (see class remarks)");
        using var gguf = GgufFile.Open(Path.Combine(dir!, "tiny_mimo_f32.gguf"));
        ModelConfig c = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.MiMoV2, c.Architecture);
        Assert.Equal(6, c.NumLayers);                  // 9 blocks - 3 NextN
        Assert.Equal(3, c.NextnPredictLayers);
        Assert.Equal(48, c.HeadDim);
        Assert.Equal(32, c.ValueHeadDim);
        Assert.Equal(4, c.NumKvHeads);                 // sliding layers
        Assert.Equal(2, c.NumGlobalKvHeads);           // global layers
        Assert.Equal(16, c.RoPEConfig!.Value.DimensionCount);
        Assert.Equal(10000f, c.RoPEConfig.Value.Theta);
        Assert.Equal(1e7f, c.GlobalRoPEConfig!.Value.Theta);
        Assert.Equal(0.707f, c.AttentionValueScale!.Value, 6);
        Assert.Equal([null, 4, 4, 4, null, 4], c.PerLayerSlidingWindow!.Select(x => x).ToArray());
        Assert.True(c.Moe!.SigmoidGating);
        Assert.True(c.Moe.HasSelectionBias);
        Assert.True(c.Moe.NormalizeExpertWeights);
        Assert.False(c.Moe.IsMoeLayer(0));
        Assert.True(c.Moe.IsMoeLayer(1));
    }

    [SkippableFact]
    public void DotLlm_Matches_Hf_And_LlamaCpp_Batched()
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny fixture not present (see class remarks)");
        var (ours, vocab, n) = RunDotLlm(dir!);
        float[] hf = ReadF32(Path.Combine(dir!, "hf_logits.f32"));
        float[] ll = ReadF32(Path.Combine(dir!, "ll_batch.f32"));
        var vsHf = Compare(ours, hf, vocab);
        var vsLl = Compare(ours, ll, vocab);
        var llVsHf = Compare(ll, hf, vocab);
        Console.WriteLine($"[mimo2-tiny] n={n} dotLLM-vs-HF maxAbs={vsHf.MaxAbs:E2} maxKL={vsHf.MaxKl:E2} top1={vsHf.Top1:P0}; " +
                          $"dotLLM-vs-llama maxAbs={vsLl.MaxAbs:E2} maxKL={vsLl.MaxKl:E2} top1={vsLl.Top1:P0}; llama-vs-HF maxAbs={llVsHf.MaxAbs:E2}");
        Assert.True(vsHf.MaxAbs < 5e-3f, $"dotLLM vs HF maxAbs {vsHf.MaxAbs}");
        Assert.True(vsHf.MaxKl < 1e-6, $"dotLLM vs HF maxKL {vsHf.MaxKl}");
        Assert.Equal(1.0, vsHf.Top1);
        Assert.True(vsLl.MaxAbs < 5e-3f, $"dotLLM vs llama.cpp maxAbs {vsLl.MaxAbs}");
    }

    [SkippableFact]
    public void DotLlm_KvCacheDecode_Matches_Hf()
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny fixture not present (see class remarks)");
        var (ours, vocab, _) = RunDotLlm(dir!, decode: true);
        float[] hf = ReadF32(Path.Combine(dir!, "hf_logits.f32"));
        var vsHf = Compare(ours, hf, vocab);
        Console.WriteLine($"[mimo2-tiny decode] dotLLM-vs-HF maxAbs={vsHf.MaxAbs:E2} maxKL={vsHf.MaxKl:E2} top1={vsHf.Top1:P0}");
        Assert.True(vsHf.MaxAbs < 5e-3f, $"decode maxAbs {vsHf.MaxAbs}");
        Assert.Equal(1.0, vsHf.Top1);
    }

    // ---- discriminating mutants: the oracle must be able to tell each MiMo-specific mechanism apart ----

    public static TheoryData<string> Mutants => new()
    {
        "no-value-scale", "no-window", "no-global-kv-heads", "global-theta-for-all", "full-rotation", "wrong-value-dim-ignored",
    };

    [SkippableTheory]
    [MemberData(nameof(Mutants))]
    public void Mutant_Of_Config_Is_Detected(string mutant)
    {
        string? dir = Dir();
        Skip.If(dir is null, "mimo2-tiny fixture not present (see class remarks)");
        Func<ModelConfig, ModelConfig> m = mutant switch
        {
            "no-value-scale" => c => c with { AttentionValueScale = null },
            // keep the sliding/global classification (it drives the fused-QKV split) but make the window effectively infinite
            "no-window" => c => c with { PerLayerSlidingWindow = c.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : 4096).ToArray() },
            "global-theta-for-all" => c => c with { RoPEConfig = c.GlobalRoPEConfig },
            "full-rotation" => c => c with
            {
                RoPEConfig = c.RoPEConfig!.Value with { DimensionCount = c.HeadDim },
                GlobalRoPEConfig = c.GlobalRoPEConfig!.Value with { DimensionCount = c.HeadDim },
            },
            _ => c => c,
        };
        if (mutant is "no-global-kv-heads" or "wrong-value-dim-ignored")
        {
            // These break the tensor geometry itself: loading must fail loudly rather than silently compute garbage.
            Func<ModelConfig, ModelConfig> bad = mutant == "no-global-kv-heads"
                ? c => c with { NumGlobalKvHeads = null }
                : c => c with { ValueHeadDim = null };
            Assert.ThrowsAny<Exception>(() => RunDotLlm(dir!, bad));
            return;
        }
        float[] hf = ReadF32(Path.Combine(dir!, "hf_logits.f32"));
        var (ours, vocab, _) = RunDotLlm(dir!, m);
        var vsHf = Compare(ours, hf, vocab);
        Console.WriteLine($"[mimo2-tiny mutant {mutant}] maxAbs={vsHf.MaxAbs:E2} maxKL={vsHf.MaxKl:E2}");
        Assert.True(vsHf.MaxAbs > 0.05f, $"mutant '{mutant}' is not detected by the oracle (maxAbs {vsHf.MaxAbs})");
    }
}
