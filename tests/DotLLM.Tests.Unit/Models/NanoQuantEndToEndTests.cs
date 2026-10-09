using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Issue #869: the community Qwen3-0.6B NanoQuant GGUF loaded end to end as a Qwen3 model on CPU and compared with the
/// llama.cpp-fork oracle (committed top-1/top-5/greedy in <c>NanoQuantData/oracle_pN.json</c>; full per-position logits
/// from <c>DOTLLM_NANOQUANT_ORACLE_DIR</c> or <c>~/.dotllm/test-cache/nanoquant-oracle</c>). Cleanly skips when the GGUF is absent.
/// Not parallelised with other env-mutating tests (the F32-decoded control switches on an environment variable).
/// </summary>
[Collection("NanoQuantEnv")]
public sealed unsafe class NanoQuantEndToEndTests(ITestOutputHelper output)
{
    private static JsonDocument Oracle(int i) => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Cpu", "Kernels", "NanoQuantData", $"oracle_p{i}.json")));

    private static string? LogitsDir()
    {
        string? d = Environment.GetEnvironmentVariable("DOTLLM_NANOQUANT_ORACLE_DIR");
        if (string.IsNullOrEmpty(d))
            d = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "nanoquant-oracle");
        return Directory.Exists(d) ? d : null;
    }

    private static float[][] Forward(TransformerModel model, int[] tokens, int vocab)
    {
        var pos = Enumerable.Range(0, tokens.Length).ToArray();
        using var t = model.Forward(tokens, pos, 0, null);
        var all = new float[tokens.Length][];
        for (int i = 0; i < tokens.Length; i++)
        {
            all[i] = new float[vocab];
            new ReadOnlySpan<float>((float*)t.DataPointer + (long)i * vocab, vocab).CopyTo(all[i]);
        }
        return all;
    }

    private static int Argmax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    /// <summary>KL(p_ref || p_ours) in nats over the full vocabulary, float64.</summary>
    private static double Kl(float[] refLogits, float[] ours)
    {
        static double Lse(float[] x) { double m = x.Max(); return m + Math.Log(x.Sum(v => Math.Exp(v - m))); }
        double lr = Lse(refLogits), lo = Lse(ours), kl = 0;
        for (int i = 0; i < refLogits.Length; i++)
        {
            double lp = refLogits[i] - lr;
            kl += Math.Exp(lp) * (lp - (ours[i] - lo));
        }
        return kl;
    }

    private (TransformerModel model, GgufFile file, int vocab)? Open(string path)
    {
        var file = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        var model = TransformerModel.LoadFromGguf(file, cfg, ThreadingConfig.Auto);
        return (model, file, cfg.VocabSize);
    }

    [Fact]
    public void LoadsAsQwen3_AndMatchesLlamaCppOracle()
    {
        string? path = NanoQuantTests.RealGguf();
        if (path is null) { output.WriteLine("SKIP: NanoQuant GGUF absent (set DOTLLM_NANOQUANT_GGUF)."); return; }
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        string? dir = LogitsDir();
        var (model, file, vocab) = Open(path)!.Value;
        using (file) using (model)
        {
            Assert.True(FactorizedWeights.LiveBytes > 0);
            int agree = 0, total = 0;
            for (int p = 1; p <= 3; p++)
            {
                using var doc = Oracle(p);
                var root = doc.RootElement;
                int[] toks = root.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var ours = Forward(model, toks, vocab);
                var pos = root.GetProperty("positions").EnumerateArray().ToArray();
                float[]? refAll = null;
                if (dir is not null && File.Exists(Path.Combine(dir, $"p{p}.logits.f32")))
                {
                    var bytes = File.ReadAllBytes(Path.Combine(dir, $"p{p}.logits.f32"));
                    refAll = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, refAll, 0, bytes.Length);
                }
                for (int i = 0; i < toks.Length; i++)
                {
                    int refTop1 = pos[i].GetProperty("top5")[0][0].GetInt32();
                    bool ok = Argmax(ours[i]) == refTop1;
                    total++; if (ok) agree++;
                    if (refAll is not null)
                    {
                        var r = refAll.AsSpan(i * vocab, vocab).ToArray();
                        double maxd = 0; for (int j = 0; j < vocab; j++) maxd = Math.Max(maxd, Math.Abs(r[j] - ours[i][j]));
                        double kl = Kl(r, ours[i]);
                        output.WriteLine($"p{p} pos{i}: top1 ref={refTop1} ours={Argmax(ours[i])} max|dlogit|={maxd:F4} KL={kl:E2}");
                        Assert.True(kl < 5e-3, $"p{p} pos{i} KL {kl}");
                    }
                    else output.WriteLine($"p{p} pos{i}: top1 ref={refTop1} ours={Argmax(ours[i])}");
                }

                // greedy continuation identity (uncached re-forward of the growing sequence)
                var seq = toks.ToList();
                var want = root.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var got = new List<int>();
                for (int g = 0; g < want.Length; g++)
                {
                    var l = Forward(model, seq.ToArray(), vocab);
                    int nxt = Argmax(l[^1]);
                    got.Add(nxt); seq.Add(nxt);
                }
                int same = want.TakeWhile((w, i) => w == got[i]).Count();
                output.WriteLine($"p{p} greedy identical prefix {same}/{want.Length}: ref=[{string.Join(',', want)}] ours=[{string.Join(',', got)}]");
                Assert.True(same >= 8, $"p{p}: greedy diverged after {same} tokens");
            }
            output.WriteLine($"top-1 agreement {agree}/{total}");
            Assert.True(agree >= total - 2, $"top-1 agreement {agree}/{total}");
        }
    }

    // Note: the logits come out (near-)bit-identical because the head is Q8_0 and its int8 input quantisation absorbs the
    // ~3e-6 relative difference between the factorised kernel and the dense decode (layer-level test below shows the raw gap).
    [Fact]
    public void NonCpuLoaders_AndOtherArchitectures_RejectClearly()
    {
        string? path = NanoQuantTests.RealGguf();
        if (path is null) { output.WriteLine("SKIP: NanoQuant GGUF absent."); return; }
        using var file = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        // Vulkan / CUDA / HIP / hybrid loaders all call LoadFromGguf without allowNanoQuant.
        var ex = Assert.Throws<NotSupportedException>(() => TransformerWeights.LoadFromGguf(file, cfg));
        Assert.Contains("CPU backend only", ex.Message);
        var ex2 = Assert.Throws<NotSupportedException>(() =>
            TransformerWeights.LoadFromGguf(file, cfg with { Architecture = Architecture.Llama }, allowNanoQuant: true));
        Assert.Contains("dense Qwen3", ex2.Message);
    }

    [Fact]
    public void F32DecodedControl_MatchesFactorisedKernels()
    {
        string? path = NanoQuantTests.RealGguf();
        if (path is null) { output.WriteLine("SKIP: NanoQuant GGUF absent."); return; }
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        using var doc = Oracle(3);
        int[] toks = doc.RootElement.GetProperty("tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        float[][] fact, dense;
        {
            var (m, f, v) = Open(path)!.Value;
            Assert.True(FactorizedWeights.LiveBytes > 0, "factorised run must register layers");
            using (f) using (m) fact = Forward(m, toks, v);
            output.WriteLine($"factorised logits[0][0..3] = {fact[3][0]:R} {fact[3][1]:R} {fact[3][2]:R}");
        }
        string? old = Environment.GetEnvironmentVariable("DOTLLM_LITTLEBIT_DENSE_CONTROL");
        Environment.SetEnvironmentVariable("DOTLLM_LITTLEBIT_DENSE_CONTROL", "1");
        try
        {
            long before = FactorizedWeights.LiveBytes;
            var (m, f, v) = Open(path)!.Value;
            Assert.Equal(before, FactorizedWeights.LiveBytes);   // the control registers NO factorised layers: dense F32 only
            using (f) using (m) dense = Forward(m, toks, v);
            output.WriteLine($"dense logits[0][0..3] = {dense[3][0]:R} {dense[3][1]:R} {dense[3][2]:R}");
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_LITTLEBIT_DENSE_CONTROL", old); }
        for (int i = 0; i < toks.Length; i++)
        {
            double maxd = 0; for (int j = 0; j < fact[i].Length; j++) maxd = Math.Max(maxd, Math.Abs(fact[i][j] - dense[i][j]));
            double kl = Kl(dense[i], fact[i]);
            output.WriteLine($"pos{i}: factorised vs dense-F32 max|dlogit|={maxd:F5} KL={kl:E2} top1 {Argmax(fact[i])}/{Argmax(dense[i])}");
            Assert.Equal(Argmax(dense[i]), Argmax(fact[i]));
            Assert.True(kl < 1e-4, $"pos{i} KL {kl}");
        }
    }
}

public sealed unsafe class NanoQuantLayerControlTests(ITestOutputHelper output)
{
    [Fact]
    public void LayerKernel_VsDenseDecode_DifferInLowBits_ButAgree()
    {
        string? path = NanoQuantTests.RealGguf();
        if (path is null || !System.Runtime.Intrinsics.X86.Avx2.IsSupported) return;
        using var file = GgufFile.Open(path);
        using var layer = DotLLM.Models.Quantization.NanoQuantLoader.Load(file, "blk.0.ffn_gate");
        var x = Enumerable.Range(0, layer.DIn).Select(i => (float)(1.5 * Math.Sin(0.37 * i + 0.1))).ToArray();
        var y1 = new float[layer.DOut]; var y2 = new float[layer.DOut];
        float* w = layer.DecodeDense();
        try
        {
            using var sc = layer.CreateScratch();
            fixed (float* xp = x, yp = y1) layer.Gemv(xp, yp, sc, null);
            for (int o = 0; o < layer.DOut; o++)
                y2[o] = System.Numerics.Tensors.TensorPrimitives.Dot(new ReadOnlySpan<float>(w + (long)o * layer.DIn, layer.DIn), x);
        }
        finally { System.Runtime.InteropServices.NativeMemory.AlignedFree(w); }
        double maxd = 0, rms = Math.Sqrt(y1.Sum(v => (double)v * v) / y1.Length);
        int ident = 0;
        for (int o = 0; o < y1.Length; o++) { maxd = Math.Max(maxd, Math.Abs(y1[o] - y2[o])); if (y1[o] == y2[o]) ident++; }
        output.WriteLine($"kernel vs dense: max|d|/rms={maxd / rms:E2}, bit-identical rows {ident}/{y1.Length}");
        Assert.True(maxd / rms < 1e-4);
    }
}
