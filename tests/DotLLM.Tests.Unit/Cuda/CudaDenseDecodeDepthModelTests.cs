using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>Restores a static test toggle on dispose.</summary>
internal sealed class StaticToggle : IDisposable
{
    private readonly Action _restore;
    private StaticToggle(Action restore) => _restore = restore;
    public void Dispose() => _restore();

    internal static StaticToggle StepGraph(bool on)
    {
        bool prev = CudaTransformerModel.StepGraphEnabled;
        CudaTransformerModel.StepGraphEnabled = on;
        return new StaticToggle(() => CudaTransformerModel.StepGraphEnabled = prev);
    }

    internal static StaticToggle DenseV2(bool on, int minSeqKv)
    {
        bool prevOn = CudaDenseDecodeV2.Enabled;
        int prevMin = CudaDenseDecodeV2.MinSeqKv;
        CudaDenseDecodeV2.Enabled = on;
        CudaDenseDecodeV2.MinSeqKv = minSeqKv;
        return new StaticToggle(() => { CudaDenseDecodeV2.Enabled = prevOn; CudaDenseDecodeV2.MinSeqKv = prevMin; });
    }

    internal static StaticToggle F32Partials(bool on)
    {
        bool prev = CudaKernels.DenseDecodeV2F32Partials;
        CudaKernels.DenseDecodeV2F32Partials = on;
        return new StaticToggle(() => CudaKernels.DenseDecodeV2F32Partials = prev);
    }

    internal static StaticToggle GraphMaxDepth(int depth)
    {
        int prev = CudaTransformerModel.GraphCaptureMaxDepth;
        CudaTransformerModel.GraphCaptureMaxDepth = depth;
        return new StaticToggle(() => CudaTransformerModel.GraphCaptureMaxDepth = prev);
    }
}

/// <summary>
/// Issue #1922, model level, SmolLM-135M (hd64, 9 query / 3 KV heads, GQA group 3) through <see cref="CudaTransformerModel"/>:
/// <list type="bullet">
/// <item>E1 dispatch proof: the dense v2 decode kernel runs on every layer of every decode step for f16, q8_0, q4_0 and in-place
/// tq4 KV when the gate allows, the scalar kernel does not; the kill switch and the depth threshold send every step back to the
/// scalar kernel, bit-identical to a run without the feature.</item>
/// <item>E1 numerics: teacher-forced logits track the scalar kernel (same tokens fed to both arms).</item>
/// <item>E2: the step graph (eager body captured per step) is bit-identical to the eager body for every KV type, and it engaged
/// on every step.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaDenseDecodeDepthModelTests
{
    private readonly ITestOutputHelper _out;
    public CudaDenseDecodeDepthModelTests(ITestOutputHelper output) => _out = output;

    private const int DecodeSteps = 24;

    private static string? FindPtxDir()
    {
        foreach (var dir in new[] { Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx") })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static KvCacheConfig Kv(string kind) => kind switch
    {
        "f16" => new KvCacheConfig(),
        "q8_0" => new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0),
        "q4_0" => new KvCacheConfig(KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        "q8q4" => new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q4_0),
        "tq4" => new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, 4),
        "tq3" => new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, 3),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    private sealed record Run(List<float[]> Rows, long DenseLaunches, long ScalarKvqLaunches, long ScalarKvTqLaunches,
        int StepGraphLaunches, int EagerDecodes);

    private (GgufFile gguf, DotLLM.Core.Models.ModelConfig config, int[] prompt) Load()
    {
        string path = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(path), $"SmolLM-135M Q8_0 GGUF not found at {path}");
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tok.Encode("The quick brown fox jumps over the lazy dog. The capital of France is Paris, the capital of " +
                                  "Germany is Berlin, and the capital of Italy is");
        return (gguf, config, prompt);
    }

    /// <summary>Prefill + teacher-forced decode over <paramref name="fed"/> (or greedy when null). One fresh model per run.</summary>
    private Run Decode(GgufFile gguf, DotLLM.Core.Models.ModelConfig config, int[] prompt, string kv, bool graph,
        int[]? fed, out int[] tokens)
    {
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, FindPtxDir()!);
        model.UseGraphCapture = graph;
        using var cache = model.CreateKvCache(prompt.Length + DecodeSteps + 4, Kv(kv));
        int vocab = config.VocabSize;
        var rows = new List<float[]>();
        var positions = Enumerable.Range(0, prompt.Length).ToArray();
        float[] last;
        using (var l = model.Forward(prompt, positions, 0, cache)) last = Row(l, vocab);
        long d0 = CudaKernels.AttentionMmaDecodeDenseLaunches, q0 = CudaKernels.AttentionKvqLaunches, t0 = CudaKernels.AttentionKvTqLaunches;
        int sg0 = model.StepGraphLaunches, e0 = model.EagerDecodeCount;
        tokens = new int[DecodeSteps];
        for (int s = 0; s < DecodeSteps; s++)
        {
            int token = fed?[s] ?? Argmax(last);
            tokens[s] = token;
            using var l = model.Forward([token], [prompt.Length + s], 0, cache);
            last = Row(l, vocab);
            rows.Add(last);
        }
        return new Run(rows, CudaKernels.AttentionMmaDecodeDenseLaunches - d0, CudaKernels.AttentionKvqLaunches - q0,
            CudaKernels.AttentionKvTqLaunches - t0, model.StepGraphLaunches - sg0, model.EagerDecodeCount - e0);
    }

    private static float[] Row(DotLLM.Core.Tensors.ITensor t, int vocab)
    {
        var r = new float[vocab];
        new ReadOnlySpan<float>((void*)t.DataPointer, vocab).CopyTo(r);
        return r;
    }

    private static int Argmax(float[] a)
    {
        int b = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[b]) b = i;
        return b;
    }

    private static int BitDiffs(List<float[]> a, List<float[]> b)
    {
        int n = 0;
        for (int s = 0; s < a.Count; s++)
            for (int i = 0; i < a[s].Length; i++)
                if (BitConverter.SingleToInt32Bits(a[s][i]) != BitConverter.SingleToInt32Bits(b[s][i])) n++;
        return n;
    }

    /// <summary>E1: v2 runs on every layer of every step and tracks the scalar kernel (teacher-forced on the scalar arm's tokens).</summary>
    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    [InlineData("q4_0")]
    [InlineData("q8q4")]
    [InlineData("tq4")]
    public void DenseV2_Dispatches_AndTracksScalar(string kv)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (gguf, config, prompt) = Load();
        using var _g = gguf;
        using var off = StaticToggle.StepGraph(false);
        Run scalar, v2;
        int[] fed;
        using (StaticToggle.DenseV2(false, 1)) scalar = Decode(gguf, config, prompt, kv, graph: false, null, out fed);
        using (StaticToggle.DenseV2(true, 1)) v2 = Decode(gguf, config, prompt, kv, graph: false, fed, out _);
        int layers = config.NumLayers;
        _out.WriteLine($"{kv}: dense launches scalar-arm={scalar.DenseLaunches} v2-arm={v2.DenseLaunches} " +
                       $"kvq {scalar.ScalarKvqLaunches}/{v2.ScalarKvqLaunches} kvtq {scalar.ScalarKvTqLaunches}/{v2.ScalarKvTqLaunches}");
        Assert.Equal(0, scalar.DenseLaunches);
        Assert.Equal((long)layers * DecodeSteps, v2.DenseLaunches);
        Assert.Equal(0, v2.ScalarKvqLaunches);
        Assert.Equal(0, v2.ScalarKvTqLaunches);
        if (kv is "q8_0" or "q4_0" or "q8q4") Assert.Equal((long)layers * DecodeSteps, scalar.ScalarKvqLaunches);
        if (kv == "tq4") Assert.Equal((long)layers * DecodeSteps, scalar.ScalarKvTqLaunches);

        double worstCos = 1, worstAbs = 0;
        int argmaxAgree = 0;
        for (int s = 0; s < DecodeSteps; s++)
        {
            float[] a = scalar.Rows[s], b = v2.Rows[s];
            double dot = 0, na = 0, nb = 0, mx = 0;
            for (int i = 0; i < a.Length; i++)
            {
                Assert.True(float.IsFinite(b[i]), $"step {s}: non-finite logit {i}");
                dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
                mx = Math.Max(mx, Math.Abs(a[i] - b[i]));
            }
            double cos = dot / Math.Sqrt(na * nb);
            worstCos = Math.Min(worstCos, cos); worstAbs = Math.Max(worstAbs, mx);
            if (Argmax(a) == Argmax(b)) argmaxAgree++;
        }
        _out.WriteLine($"{kv}: worst cos(v2, scalar)={worstCos:F7} worst max|dlogit|={worstAbs:F4} argmax agree {argmaxAgree}/{DecodeSteps}");
        // Sanity bound only. The two arms differ by F16 rounding of P / of decoded elements inside attention; over 30 layers of a
        // 135M model that drift is amplified, and on quantized KV it is digitized into flipped codes in later rows (first run,
        // framework: worst cos f16 0.99915, q8_0 0.99980, q4_0 0.976, q8q4 0.980, tq4 0.949, argmax 23-24/24). The quality gate
        // is CudaDenseDecodeQualityGateTests (Qwen3-4B, teacher-forced PPL and KL); correctness is gated at kernel level.
        double floor = kv == "f16" ? 0.998 : 0.9;
        Assert.True(worstCos > floor, $"{kv}: v2 logits diverge from the scalar kernel (cos {worstCos:F7} <= {floor})");
        Assert.True(argmaxAgree >= DecodeSteps - 2, $"{kv}: argmax agreement {argmaxAgree}/{DecodeSteps}");
    }

    /// <summary>E1: the kill switch and the depth threshold leave decode bit-identical to the scalar kernel, and v2 never runs.</summary>
    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    [InlineData("tq4")]
    public void DenseV2_BelowThresholdOrOff_IsBitIdenticalToScalar(string kv)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (gguf, config, prompt) = Load();
        using var _g = gguf;
        using var off = StaticToggle.StepGraph(false);
        Run reference, killed, shallow;
        int[] fed;
        using (StaticToggle.DenseV2(false, 1)) reference = Decode(gguf, config, prompt, kv, false, null, out fed);
        using (StaticToggle.DenseV2(false, 1)) killed = Decode(gguf, config, prompt, kv, false, fed, out _);
        using (StaticToggle.DenseV2(true, 100_000)) shallow = Decode(gguf, config, prompt, kv, false, fed, out _);
        Assert.Equal(0, killed.DenseLaunches + shallow.DenseLaunches);
        Assert.Equal(0, BitDiffs(reference.Rows, killed.Rows));
        Assert.Equal(0, BitDiffs(reference.Rows, shallow.Rows));
    }

    /// <summary>
    /// E2: the step graph replays exactly the eager launch sequence, so its logits are bit-identical to the eager body's on every
    /// step (v2 engaged in both); it engaged on every step and the eager body never ran. F16 needs the shallow graph's depth cap
    /// lowered below the prompt so the step graph, not the device-scalar graph, takes it.
    /// </summary>
    [SkippableTheory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    [InlineData("q4_0")]
    [InlineData("tq4")]
    [InlineData("tq3")]
    public void StepGraph_IsBitIdenticalToEager_AndEngages(string kv)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (gguf, config, prompt) = Load();
        using var _g = gguf;
        using var v2 = StaticToggle.DenseV2(true, 1);
        using var cap = StaticToggle.GraphMaxDepth(1);
        Run eager, graphed;
        int[] fed;
        using (StaticToggle.StepGraph(false)) eager = Decode(gguf, config, prompt, kv, graph: true, null, out fed);
        using (StaticToggle.StepGraph(true)) graphed = Decode(gguf, config, prompt, kv, graph: true, fed, out _);
        _out.WriteLine($"{kv}: eager-arm stepGraph={eager.StepGraphLaunches} eagerDecodes={eager.EagerDecodes}; " +
                       $"graph-arm stepGraph={graphed.StepGraphLaunches} eagerDecodes={graphed.EagerDecodes} dense={graphed.DenseLaunches}");
        Assert.Equal(0, eager.StepGraphLaunches);
        Assert.Equal(DecodeSteps, eager.EagerDecodes);
        Assert.Equal(DecodeSteps, graphed.StepGraphLaunches);
        Assert.Equal(0, graphed.EagerDecodes);
        Assert.Equal((long)config.NumLayers * DecodeSteps, graphed.DenseLaunches);
        Assert.Equal(0, BitDiffs(eager.Rows, graphed.Rows));
    }
}
