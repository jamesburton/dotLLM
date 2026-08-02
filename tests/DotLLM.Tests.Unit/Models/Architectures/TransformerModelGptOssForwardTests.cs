using System.Runtime.InteropServices;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end gpt-oss (OpenAI gpt-oss-20b/120b, llama.cpp
/// <c>LLM_ARCH_OPENAI_MOE</c>) CPU forward-pass coverage through the full
/// GGUF load path (<see cref="GgufModelConfigExtractor.Extract"/> +
/// <see cref="TransformerModel.LoadFromGguf(GgufFile, ModelConfig, ThreadingConfig)"/>).
/// Complements the kernel-level tests in <c>GptOssKernelTests.cs</c>
/// (attention sinks, swiglu_oai, quantized-expert MoE routing, each in
/// isolation) and the config-detection tests in <c>GptOssConfigTests.cs</c>
/// (metadata → <see cref="ModelConfig"/> mapping only) by exercising all
/// three gpt-oss deltas TOGETHER on a real (tiny) GGUF file:
/// <list type="bullet">
///   <item>Per-head attention sinks (<c>attn_sinks.weight</c>) loaded from
///     GGUF and wired into every layer's softmax denominator.</item>
///   <item>Alternating sliding-window / dense attention
///     (<see cref="ModelConfig.SlidingWindowPattern"/> = 2: even layers
///     windowed, odd layers dense — llama.cpp <c>set_swa_pattern(2)</c>).</item>
///   <item>Every layer a routed MoE FFN with MXFP4-quantized expert weights
///     read straight from the GGUF mmap (no F32 inflation) via
///     <c>MoeQuantSwiGluMlp</c>: softmax-after-top-k gating, clamped
///     swiglu_oai activation, per-expert biases.</item>
/// </list>
/// <para>
/// No real <c>gpt-oss-20b-mxfp4.gguf</c> checkpoint is available locally
/// (checked the HF cache and every local model directory — see issue #138)
/// so this synthetic fixture is the acceptable minimum bar; real-checkpoint
/// numerical validation against llama.cpp remains pending until the file is
/// available.
/// </para>
/// </summary>
public sealed class TransformerModelGptOssForwardTests : IDisposable
{
    // MXFP4 blocks are 32 elements — every quantized row's K dimension must
    // be a multiple of 32, so hidden size and MoE intermediate size are both
    // pinned to exactly one block per row.
    private const int HiddenSize = 32;
    private const int NumLayers = 4;     // pattern 2 → layers 0,2 sliding; 1,3 dense.
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = HiddenSize / NumHeads; // 8
    private const int VocabSize = 6;
    private const int NumExperts = 4;
    private const int TopK = 2;
    private const int MoeIntermediateSize = 32;
    private const int SlidingWindow = 3;
    private const int ContextLength = 64;

    private static readonly int[] Ids = [0, 1, 2, 3, 4, 5];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5];

    private readonly string _scratch;

    public TransformerModelGptOssForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gptoss-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_GptOss_MxfpExpertsSinksAndSwa_FiniteNonDegenerate()
    {
        string path = WriteFixture("gptoss-main", seed: 42);

        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.GptOss, cfg.Architecture);
        Assert.NotNull(cfg.Moe);
        Assert.Equal(NumExperts, cfg.Moe!.NumExperts);
        Assert.Equal(TopK, cfg.Moe.NumExpertsPerTok);
        Assert.True(cfg.Moe.SoftmaxAfterTopK);
        Assert.True(cfg.Moe.UseSwiGluOai);
        Assert.True(cfg.Moe.HasExpertBiases);
        Assert.Equal(2, cfg.SlidingWindowPattern);
        Assert.Equal(SlidingWindow, cfg.SlidingWindowSize);

        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(Ids, Pos, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(Ids.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f, $"Logits degenerate: std={stats.StdDev}");
    }

    [Fact]
    public void Forward_GptOss_AttentionSinks_StrongVsNegligible_DiffersMeasurably()
    {
        // Two fixtures with IDENTICAL weights, differing ONLY in attn_sinks
        // magnitude: "strong" (moderate positive — materially competes in the
        // softmax denominator) vs "negligible" (very negative — a hugely
        // negative sink contributes ~0 to the denominator, i.e. ~equivalent
        // to no sink at all, per GptOssKernelTests' discriminative finding).
        // Sinks are generated from a closed-form per-head formula (no RNG
        // draw), so switching mode never perturbs the shared RNG sequence
        // used for every other tensor — the two fixtures are bit-identical
        // apart from attn_sinks.weight. Proves the GGUF loader actually reads
        // attn_sinks.weight and TransformerModel wires it into the real
        // forward path end-to-end (not just the isolated attention kernel).
        string strong = WriteFixture("gptoss-sink-strong", seed: 7, sinkMode: SinkMode.Strong);
        string negligible = WriteFixture("gptoss-sink-weak", seed: 7, sinkMode: SinkMode.Negligible);

        float[] strongLogits = RunLogits(strong);
        float[] weakLogits = RunLogits(negligible);

        Assert.All(strongLogits, v => Assert.True(float.IsFinite(v)));
        Assert.All(weakLogits, v => Assert.True(float.IsFinite(v)));

        float maxDiff = MaxAbsDiff(strongLogits, weakLogits);
        Assert.True(maxDiff > 1e-4f,
            $"Attention sinks had no measurable effect through the full GGUF forward path (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_GptOss_SlidingWindowPattern_DiffersFromAllDense()
    {
        // Discriminative: the fixture's extracted config alternates sliding
        // (layers 0,2; window=3) and dense (layers 1,3) attention. Disabling
        // the window entirely (SlidingWindowSize = null -> every layer full
        // dense) over a 6-token sequence (> window) must change the logits,
        // proving ModelConfig.SlidingWindowPattern actually drives the
        // per-layer window choice inside TransformerModel's forward dispatch
        // end-to-end — not just at config-extraction time (already covered
        // in isolation by GptOssConfigTests).
        string path = WriteFixture("gptoss-swa", seed: 99);

        using var gguf = GgufFile.Open(path);
        ModelConfig windowed = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ModelConfig dense = windowed with { SlidingWindowSize = null };

        float[] windowedLogits = RunLogits(gguf, windowed);
        float[] denseLogits = RunLogits(gguf, dense);

        Assert.All(windowedLogits, v => Assert.True(float.IsFinite(v)));
        Assert.All(denseLogits, v => Assert.True(float.IsFinite(v)));

        float maxDiff = MaxAbsDiff(windowedLogits, denseLogits);
        Assert.True(maxDiff > 1e-4f,
            $"SlidingWindowPattern had no measurable effect vs all-dense attention (maxDiff={maxDiff}).");
    }

    // ───────────────────────── helpers ─────────────────────────

    private enum SinkMode { Strong, Negligible }

    private float[] RunLogits(string path)
    {
        using var gguf = GgufFile.Open(path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        return RunLogits(gguf, cfg);
    }

    private static float[] RunLogits(GgufFile gguf, ModelConfig cfg)
    {
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(Ids, Pos, deviceId: -1);
        return CopyAll(logits);
    }

    /// <summary>
    /// Writes a synthetic gpt-oss GGUF: token_embd/output/output_norm, and per
    /// layer attn_norm + Q/K/V/O (F32, with biases) + attn_sinks + the
    /// gpt-oss-named post_attention_norm (no ffn_norm.weight — exercises the
    /// loader's fallback) + a routed-MoE block (ffn_gate_inp router + MXFP4
    /// ffn_{gate,up,down}_exps 3D expert banks + per-expert biases).
    /// </summary>
    private string WriteFixture(string name, int seed, SinkMode sinkMode = SinkMode.Strong)
    {
        string path = Path.Combine(_scratch, $"{name}.gguf");
        var rng = new Random(seed);
        var w = new GgufWriter();
        const string arch = "gpt-oss";

        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", (uint)ContextLength);
        w.AddUInt32($"{arch}.embedding_length", (uint)HiddenSize);
        w.AddUInt32($"{arch}.block_count", (uint)NumLayers);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)MoeIntermediateSize);
        w.AddUInt32($"{arch}.attention.head_count", (uint)NumHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", (uint)NumKvHeads);
        w.AddUInt32($"{arch}.attention.key_length", (uint)HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)HeadDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.attention.sliding_window", (uint)SlidingWindow);
        // sliding_window_pattern intentionally omitted -> extractor defaults to 2.
        w.AddUInt32($"{arch}.expert_count", (uint)NumExperts);
        w.AddUInt32($"{arch}.expert_used_count", (uint)TopK);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)MoeIntermediateSize);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        w.AddString($"{arch}.rope.scaling.type", "yarn");
        w.AddFloat32($"{arch}.rope.scaling.factor", 2.0f);
        w.AddUInt32($"{arch}.rope.scaling.original_context_length", 32);
        w.AddUInt32($"{arch}.vocab_size", (uint)VocabSize);

        AddMatrixF32(w, rng, "token_embd.weight", inK: HiddenSize, outM: VocabSize, 0.05f);

        for (int i = 0; i < NumLayers; i++)
            AddLayer(w, rng, i, sinkMode);

        AddNormF32(w, rng, "output_norm.weight", HiddenSize);
        AddMatrixF32(w, rng, "output.weight", inK: HiddenSize, outM: VocabSize, 0.05f);

        File.WriteAllBytes(path, w.Build());
        return path;
    }

    private static void AddLayer(GgufWriter w, Random rng, int layer, SinkMode sinkMode)
    {
        string p = $"blk.{layer}";
        const int qOut = NumHeads * HeadDim;   // = HiddenSize
        const int kvOut = NumKvHeads * HeadDim;

        AddNormF32(w, rng, $"{p}.attn_norm.weight", HiddenSize);
        AddMatrixF32(w, rng, $"{p}.attn_q.weight", inK: HiddenSize, outM: qOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_k.weight", inK: HiddenSize, outM: kvOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_v.weight", inK: HiddenSize, outM: kvOut, 0.1f);
        AddMatrixF32(w, rng, $"{p}.attn_output.weight", inK: qOut, outM: HiddenSize, 0.1f);
        AddVectorF32(w, rng, $"{p}.attn_q.bias", qOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_k.bias", kvOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_v.bias", kvOut, 0.02f);
        AddVectorF32(w, rng, $"{p}.attn_output.bias", HiddenSize, 0.02f);

        // Attention sinks — formula-driven (NOT an RNG draw), so the "strong"
        // vs "negligible" fixtures stay bit-identical on every other tensor.
        float[] sinks = new float[NumHeads];
        for (int h = 0; h < NumHeads; h++)
            sinks[h] = sinkMode == SinkMode.Strong ? 1.0f + 0.5f * h : -30.0f;
        w.AddTensor($"{p}.attn_sinks.weight", [NumHeads], (uint)QuantizationType.F32, ToBytes(sinks));

        // gpt-oss names its pre-FFN norm "post_attention_norm" (llama.cpp
        // LLM_TENSOR_ATTN_POST_NORM) — no ffn_norm.weight tensor at all, so
        // this exercises the loader's fallback lookup.
        AddNormF32(w, rng, $"{p}.post_attention_norm.weight", HiddenSize);

        // Router + quantized-expert MoE (every gpt-oss layer is routed-MoE).
        AddMatrixF32(w, rng, $"{p}.ffn_gate_inp.weight", inK: HiddenSize, outM: NumExperts, 0.1f);
        AddVectorF32(w, rng, $"{p}.ffn_gate_inp.bias", NumExperts, 0.05f);

        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_gate_exps.weight",
            inK: HiddenSize, midOut: MoeIntermediateSize, experts: NumExperts);
        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_up_exps.weight",
            inK: HiddenSize, midOut: MoeIntermediateSize, experts: NumExperts);
        AddMxfp4ExpertBank(w, rng, $"{p}.ffn_down_exps.weight",
            inK: MoeIntermediateSize, midOut: HiddenSize, experts: NumExperts);

        // Per-expert biases, flat [E × inner] expert-major (matches
        // TransformerWeights.LoadQuantExpertMoeLayer / MoeQuantSwiGluMlp).
        AddVectorF32(w, rng, $"{p}.ffn_gate_exps.bias", NumExperts * MoeIntermediateSize, 0.05f);
        AddVectorF32(w, rng, $"{p}.ffn_up_exps.bias", NumExperts * MoeIntermediateSize, 0.05f);
        AddVectorF32(w, rng, $"{p}.ffn_down_exps.bias", NumExperts * HiddenSize, 0.05f);
    }

    /// <summary>Emits a 2D F32 matrix [ne0=K, ne1=M] of small deterministic weights.</summary>
    private static void AddMatrixF32(GgufWriter w, Random rng, string name, int inK, int outM, float amplitude)
    {
        long count = (long)inK * outM;
        float[] f = new float[count];
        for (long i = 0; i < count; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>Emits a 1D F32 RMSNorm weight vector [n], centered at 1.0.</summary>
    private static void AddNormF32(GgufWriter w, Random rng, string name, int n)
    {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        w.AddTensor(name, [n], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>Emits a 1D F32 bias/vector [n] of small deterministic values.</summary>
    private static void AddVectorF32(GgufWriter w, Random rng, string name, int n, float amplitude)
    {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        w.AddTensor(name, [n], (uint)QuantizationType.F32, ToBytes(f));
    }

    /// <summary>
    /// Emits a 3D MXFP4 expert bank [ne0=K, ne1=midOut, ne2=experts]: for
    /// each of the <c>midOut × experts</c> rows, one 17-byte block (K == 32,
    /// so exactly one block per row) — a random E8M0 scale byte near unit
    /// magnitude (llama.cpp's own dequant reference range) plus 16 bytes of
    /// nibble pairs, matching <c>GptOssKernelTests.RandomMxfp4</c>. Values
    /// don't need to reconstruct any particular float (that arithmetic is
    /// already pinned by Mxfp4Tests/GptOssKernelTests) — only to be
    /// well-formed MXFP4 bytes flowing through the real GGUF loader.
    /// </summary>
    private static void AddMxfp4ExpertBank(GgufWriter w, Random rng, string name, int inK, int midOut, int experts)
    {
        if (inK % 32 != 0)
            throw new ArgumentException("MXFP4 K dimension must be a multiple of 32.", nameof(inK));

        long rows = (long)midOut * experts;
        long blocksPerRow = inK / 32;
        long totalBytes = rows * blocksPerRow * 17;
        byte[] data = new byte[totalBytes];
        long o = 0;
        for (long r = 0; r < rows; r++)
        {
            for (long b = 0; b < blocksPerRow; b++)
            {
                data[o++] = (byte)rng.Next(122, 133); // E8M0 scale byte, near unit magnitude.
                for (int j = 0; j < 16; j++) data[o++] = (byte)rng.Next(256);
            }
        }
        w.AddTensor(name, [inK, midOut, experts], (uint)QuantizationType.MXFP4, data);
    }

    private static byte[] ToBytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float maxDiff = 0f;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxDiff) maxDiff = d;
        }
        return maxDiff;
    }

    private static unsafe float[] CopyAll(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)stddev);
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly record struct LogitStats(int TotalCount, int FiniteCount, float StdDev);
}
