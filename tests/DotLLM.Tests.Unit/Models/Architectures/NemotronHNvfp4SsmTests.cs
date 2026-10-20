using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1038 — NVIDIA's Nemotron 3.5 Bolt GGUF stores every Mamba-2 <c>ssm_in</c>/<c>ssm_out</c> as NVFP4
/// with a per-tensor <c>X.scale</c> that llama.cpp multiplies into the matmul output
/// (<c>build_lora_mm(w, cur, w_s)</c>, <c>mamba-base.cpp</c>). These tests run a two-SSM-layer
/// <c>nemotron_h</c> GGUF through the real extractor, loader and CPU forward, and compare it with the same
/// model whose SSM projections are stored as F32 <c>scale · dequant(W)</c>, dequantized by this test's own
/// transcription of <c>dequantize_row_nvfp4</c>. The scales (0.05 and 3) are far from 1 and the two layers
/// differ, so dropping a scale, swapping them, or misreading the nibble order changes the logits by far
/// more than the Q8_0 activation rounding the NVFP4 path carries.
/// </summary>
public sealed class NemotronHNvfp4SsmTests : IDisposable
{
    private const string Arch = "nemotron_h";
    private const int Hidden = 64;    // ssm_in contraction: a multiple of 64
    private const int DInner = 64;    // ssm_out contraction: a multiple of 64
    private const int NHead = 2;
    private const int DState = 8;
    private const int DConv = 4;
    internal const int Vocab = 16;
    private const float Eps = 1e-5f;
    private static readonly float[] InScale = [0.05f, 0.08f];
    private static readonly float[] OutScale = [3.0f, 2.0f];
    internal static readonly int[] Ids = [3, 7, 1, 11, 0];
    internal static readonly int[] Pos = [0, 1, 2, 3, 4];
    private static readonly sbyte[] KValues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    public enum Variant
    {
        /// <summary>SSM projections NVFP4 with their X.scale tensors (the Bolt layout).</summary>
        Nvfp4Scaled,

        /// <summary>The same model with SSM projections stored as F32 scale · dequant(W).</summary>
        F32Reference,

        /// <summary>Nvfp4Scaled plus an NVFP4 output head with an output.scale this model does not apply.</summary>
        ScaledNvfp4OutputHead,

        /// <summary>
        /// The same model as <see cref="Nvfp4Scaled"/> at Bolt's magnitudes: UE4M3 sub-block scales 256x larger
        /// (Bolt's are around 0x62), SSM input norms 16x larger, and X.scale smaller by the same 4096x, so the
        /// unscaled GEMM output (before X.scale) is far beyond FP16 range while the scaled output is unchanged.
        /// </summary>
        Nvfp4ScaledBoltMagnitude,
    }

    /// <summary>Factor by which <see cref="Variant.Nvfp4ScaledBoltMagnitude"/> raises the UE4M3 sub-block scales.</summary>
    private const int BoltUe4m3Boost = 256;     // +8 on the UE4M3 exponent

    /// <summary>Factor by which <see cref="Variant.Nvfp4ScaledBoltMagnitude"/> raises the SSM input norm weights.</summary>
    private const float BoltNormBoost = 16f;

    private readonly string _scratch;

    public NemotronHNvfp4SsmTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-nvfp4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Theory]
    [InlineData(1, Variant.Nvfp4Scaled)]
    [InlineData(4, Variant.Nvfp4Scaled)]
    [InlineData(1, Variant.Nvfp4ScaledBoltMagnitude)] // the same model at Bolt magnitudes (pins the variant the CUDA test uses)
    public void Forward_Nvfp4ScaledSsm_MatchesF32ScaledDequantReference(int threads, Variant variant)
    {
        float[] nv = Logits(WriteFixture(variant), threads);
        float[] f32 = Logits(WriteFixture(Variant.F32Reference), threads);

        float maxAbs = f32.Max(MathF.Abs);
        float maxDiff = nv.Zip(f32, (a, b) => MathF.Abs(a - b)).Max();
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        // Measured: 0.035 x max|logit| here, all of it the Q8_0 activation rounding llama.cpp also applies (the same
        // model run through the f32 dequantize-rows GEMM matches to ~1e-6). A dropped or swapped scale, or a wrong
        // nibble order, moves the logits by a multiple of max|logit|.
        Assert.True(maxDiff <= 0.06f * maxAbs,
            $"NVFP4 SSM logits differ from the F32 scale*dequant reference by {maxDiff} (max |logit| {maxAbs}); "
            + "ssm_in/ssm_out must be NVFP4 x Q8_0 times their per-tensor X.scale (#1038)");
    }

    [Fact]
    public void Load_ReadsPerTensorScalesIntoSsmWeights()
    {
        using var gguf = GgufFile.Open(WriteFixture(Variant.Nvfp4Scaled));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        Assert.Equal(Architecture.NemotronH, config.Architecture);

        var weights = Assert.IsType<NemotronHSsmWeights>(SsmWeightsOf(model, 1));
        Assert.Equal(QuantizationType.NVFP4, weights.InQuantType);
        Assert.Equal(QuantizationType.NVFP4, weights.OutQuantType);
        Assert.Equal(InScale[1], weights.InScale);
        Assert.Equal(OutScale[1], weights.OutScale);
    }

    [Fact]
    public void Load_ScaledNvfp4TensorTheModelDoesNotApply_Throws()
    {
        using var gguf = GgufFile.Open(WriteFixture(Variant.ScaledNvfp4OutputHead));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var ex = Assert.Throws<NotSupportedException>(
            () => NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded));
        Assert.Contains("output.scale", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void GpuGuard_RefusesNvfp4OrScaledSsmProjections()
    {
        using var gguf = GgufFile.Open(WriteFixture(Variant.Nvfp4Scaled));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        var ssm = SsmWeightsOf(model, 0);

        var ex = Assert.Throws<NotSupportedException>(() => ssm.ThrowIfNvfp4OrScaled("CUDA"));
        Assert.Contains("#1038", ex.Message, StringComparison.Ordinal);

        // F32 projections that carry a scale are refused too: the GPU paths do not multiply X.scale either.
        using var refGguf = GgufFile.Open(WriteFixture(Variant.F32Reference));
        using var refModel = NemotronHTransformerModel.LoadFromGguf(refGguf, config, ThreadingConfig.SingleThreaded);
        SsmWeightsOf(refModel, 0).ThrowIfNvfp4OrScaled("CUDA"); // unscaled F32: allowed
        var scaledF32 = new NemotronHSsmWeights
        {
            InWeight = 1, InQuantType = QuantizationType.F32, InInputDim = 1, InOutputDim = 1,
            Conv1dWeight = [], Conv1dBias = [], A = [], D = [], DtBias = [], NormWeight = [],
            OutWeight = 1, OutQuantType = QuantizationType.F32, OutInputDim = 1, OutOutputDim = 1,
            OutScale = 0.5f,
        };
        Assert.Throws<NotSupportedException>(() => scaledF32.ThrowIfNvfp4OrScaled("CUDA"));
    }

    [Fact]
    public void Nvfp4Support_ScalesAndRejections()
    {
        static GgufTensorDescriptor D(string n, QuantizationType qt) => new(n, new TensorShape(64), qt, 0);
        var tensors = new Dictionary<string, GgufTensorDescriptor>
        {
            ["blk.0.ssm_in.weight"] = D("blk.0.ssm_in.weight", QuantizationType.NVFP4),
            ["blk.0.ssm_in.scale"] = D("blk.0.ssm_in.scale", QuantizationType.F32),
            ["blk.0.ssm_out.weight"] = D("blk.0.ssm_out.weight", QuantizationType.NVFP4), // NVFP4, no scale
            // An architectural .scale next to a non-NVFP4 weight (Gemma-4's ffn_gate_inp.scale) is not ours.
            ["blk.0.ffn_gate_inp.weight"] = D("blk.0.ffn_gate_inp.weight", QuantizationType.F32),
            ["blk.0.ffn_gate_inp.scale"] = D("blk.0.ffn_gate_inp.scale", QuantizationType.F32),
        };

        Assert.Equal(["blk.0.ssm_in.scale"], Nvfp4Support.ScalesOfNvfp4Weights(tensors));
        Assert.Equal("blk.3.ssm_out.scale", Nvfp4Support.ScaleNameFor("blk.3.ssm_out.weight"));

        Assert.Throws<NotSupportedException>(() => Nvfp4Support.RejectUnappliedScales(tensors, null, "test"));
        Nvfp4Support.RejectUnappliedScales(tensors, new HashSet<string> { "blk.0.ssm_in.scale" }, "test");

        var ex = Assert.Throws<NotSupportedException>(() => Nvfp4Support.RejectAny(tensors, "Vulkan"));
        Assert.Contains("blk.0.ssm_in.weight", ex.Message, StringComparison.Ordinal);
        tensors.Remove("blk.0.ssm_in.weight");
        tensors.Remove("blk.0.ssm_out.weight");
        Nvfp4Support.RejectAny(tensors, "Vulkan");
    }

    /// <summary>
    /// CUDA runs NVFP4 on the Nemotron-H path only, so <c>CudaModelLoader.CreateFromGguf</c>'s refusal is now
    /// conditional on the architecture (the source scan below only counts the call). Any other architecture
    /// must still be refused, before any CUDA call, so this needs no GPU.
    /// </summary>
    [Fact]
    public void CudaCreateFromGguf_RefusesNvfp4OutsideNemotronH()
    {
        using var gguf = GgufFile.Open(WriteFixture(Variant.Nvfp4Scaled));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata) with { Architecture = Architecture.Llama };
        var ex = Assert.Throws<NotSupportedException>(
            () => DotLLM.Cuda.CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0));
        // The loader's own refusal, not the shared weight loader's unapplied-scale one further down.
        Assert.Contains("The CUDA backend has no NVFP4", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// Every GGUF loader in <c>DotLLM.Models/Architectures</c> that reads the tensor table must name the
    /// NVFP4 scales it applies (or none) — a new loader that forgets would run Bolt-style files unscaled.
    /// The match is the call statement itself, not a mention in a comment.
    /// </summary>
    [Fact]
    public void EveryGgufLoaderThatReadsTensors_GuardsNvfp4Scales()
    {
        string root = FindRepoRoot();
        string dir = Path.Combine(root, "src", "DotLLM.Models", "Architectures");
        var call = new Regex(@"^\s*Nvfp4Support\.RejectUnappliedScales\(", RegexOptions.Multiline, TimeSpan.FromSeconds(5));
        var loader = new Regex(@"static\s+\w+\s+LoadFromGguf\(\s*GgufFile\s+gguf", RegexOptions.Multiline, TimeSpan.FromSeconds(5));
        var missing = new List<string>();
        int loaders = 0;
        foreach (string file in Directory.EnumerateFiles(dir, "*.cs"))
        {
            string src = File.ReadAllText(file);
            if (!loader.IsMatch(src) || !src.Contains("gguf.TensorsByName", StringComparison.Ordinal))
                continue;
            loaders++;
            if (!call.IsMatch(src))
                missing.Add(Path.GetFileName(file));
        }

        Assert.True(loaders >= 7, $"scan found only {loaders} GGUF loaders; the scan itself is broken");
        Assert.True(missing.Count == 0, "GGUF loaders without an NVFP4 scale guard: " + string.Join(", ", missing));

        // Every public GGUF entry point of a backend without NVFP4 kernels: CudaModelLoader has three (CreateFromGguf, its Partial
        // branch and the path-based LoadFromGguf), the Vulkan|CUDA hybrid one.
        foreach (var (file, backend, calls) in new[]
        {
            ("src/DotLLM.Vulkan/VulkanModelLoader.cs", "Vulkan", 1),
            ("src/DotLLM.Cuda/CudaModelLoader.cs", "CUDA", 3),
            ("src/DotLLM.Cuda/HybridVulkanCudaTransformerModel.cs", "Vulkan/CUDA hybrid", 1),
        })
        {
            string src = File.ReadAllText(Path.Combine(root, file));
            var guard = new Regex($@"^\s*Nvfp4Support\.RejectAny\(gguf\.TensorsByName, ""{Regex.Escape(backend)}""\);", RegexOptions.Multiline, TimeSpan.FromSeconds(5));
            Assert.True(guard.Matches(src).Count == calls, $"{file}: expected {calls} NVFP4 refusal call(s), found {guard.Matches(src).Count}");
        }
    }

    // ───────────────────────── model helpers ─────────────────────────

    private static unsafe float[] Logits(string path, int threads)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, new ThreadingConfig(threads, threads));
        using var logits = model.Forward(Ids, Pos, deviceId: -1);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, Ids.Length * Vocab).ToArray();
    }

    private static NemotronHSsmWeights SsmWeightsOf(NemotronHTransformerModel model, int layer)
        => model.LayerWeights[layer].Ssm ?? throw new InvalidOperationException($"layer {layer} is not SSM");

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "dotLLM.slnx")))
            dir = dir.Parent;
        return dir?.FullName ?? throw new InvalidOperationException("repo root not found");
    }

    // ───────────────────────── fixture ─────────────────────────

    private string WriteFixture(Variant variant) => WriteFixture(_scratch, variant);

    /// <summary>Writes the fixture GGUF for <paramref name="variant"/> into <paramref name="dir"/> (shared with the GPU tests).</summary>
    internal static string WriteFixture(string dir, Variant variant)
    {
        var rng = new Random(1038);
        var g = new GgufWriter();
        g.AddString("general.architecture", Arch);
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32($"{Arch}.context_length", 64);
        g.AddUInt32($"{Arch}.embedding_length", Hidden);
        g.AddUInt32($"{Arch}.block_count", 2);
        g.AddUInt32($"{Arch}.attention.head_count", 2);
        g.AddInt32Array($"{Arch}.attention.head_count_kv", [0, 0]);
        g.AddInt32Array($"{Arch}.feed_forward_length", [0, 0]);
        g.AddUInt32($"{Arch}.attention.key_length", 32);
        g.AddUInt32($"{Arch}.vocab_size", Vocab);
        g.AddFloat32($"{Arch}.attention.layer_norm_rms_epsilon", Eps);
        g.AddUInt32($"{Arch}.ssm.conv_kernel", DConv);
        g.AddUInt32($"{Arch}.ssm.state_size", DState);
        g.AddUInt32($"{Arch}.ssm.group_count", 1);
        g.AddUInt32($"{Arch}.ssm.time_step_rank", NHead);
        g.AddUInt32($"{Arch}.ssm.inner_size", DInner);

        int convDim = DInner + 2 * DState;
        int inProj = 2 * DInner + 2 * DState + NHead;

        F32(g, "token_embd.weight", [Hidden, Vocab], Rand(rng, Hidden * Vocab, 1.0f));
        for (int l = 0; l < 2; l++)
        {
            string p = $"blk.{l}";
            F32(g, $"{p}.attn_norm.weight", [Hidden], Boost(variant, Ones(rng, Hidden)));
            Projection(g, rng, variant, $"{p}.ssm_in", Hidden, inProj, InScale[l]);
            F32(g, $"{p}.ssm_conv1d.weight", [DConv, convDim], Rand(rng, DConv * convDim, 0.5f));
            F32(g, $"{p}.ssm_conv1d.bias", [convDim], Rand(rng, convDim, 0.1f));
            F32(g, $"{p}.ssm_a", [1, NHead], Enumerable.Range(0, NHead).Select(_ => -(float)(rng.NextDouble() * 0.5 + 0.2)).ToArray());
            F32(g, $"{p}.ssm_d", [1, NHead], Rand(rng, NHead, 0.5f));
            F32(g, $"{p}.ssm_dt.bias", [NHead], Rand(rng, NHead, 0.5f));
            F32(g, $"{p}.ssm_norm.weight", [DInner, 1], Boost(variant, Ones(rng, DInner)));
            Projection(g, rng, variant, $"{p}.ssm_out", DInner, Hidden, OutScale[l]);
        }

        F32(g, "output_norm.weight", [Hidden], Ones(rng, Hidden));
        if (variant == Variant.ScaledNvfp4OutputHead)
        {
            g.AddTensor("output.weight", [Hidden, Vocab], (uint)QuantizationType.NVFP4, RandomNvfp4(rng, Hidden * Vocab));
            F32(g, "output.scale", [1], [0.5f]);
        }
        else
        {
            F32(g, "output.weight", [Hidden, Vocab], Rand(rng, Hidden * Vocab, 0.5f));
        }

        string path = Path.Combine(dir, $"nvfp4-{variant}.gguf");
        File.WriteAllBytes(path, g.Build());
        return path;
    }

    /// <summary>
    /// Writes <c>{name}.weight</c> [inK, outM]: NVFP4 plus <c>{name}.scale</c>, or (reference) F32
    /// <c>scale · dequant(W)</c> from the same random bytes. Both variants consume the RNG identically.
    /// </summary>
    private static void Projection(GgufWriter g, Random rng, Variant variant, string name, int inK, int outM, float scale)
    {
        byte[] nvfp4 = RandomNvfp4(rng, inK * outM);
        if (variant == Variant.F32Reference)
        {
            float[] w = DequantNvfp4(nvfp4);
            for (int i = 0; i < w.Length; i++) w[i] *= scale;
            F32(g, $"{name}.weight", [inK, outM], w);
        }
        else if (variant == Variant.Nvfp4ScaledBoltMagnitude)
        {
            // Raise every UE4M3 exponent by 8 (x256; the bytes stay below 0x7F) and fold that and the 16x input
            // norm into X.scale, so scale . dequant(W) . x is the same model as the other variants.
            for (int blk = 0; blk < nvfp4.Length; blk += 36)
                for (int s = 0; s < 4; s++)
                    nvfp4[blk + s] = checked((byte)(nvfp4[blk + s] + 0x40));
            g.AddTensor($"{name}.weight", [inK, outM], (uint)QuantizationType.NVFP4, nvfp4);
            F32(g, $"{name}.scale", [1], [scale / (BoltUe4m3Boost * BoltNormBoost)]);
        }
        else
        {
            g.AddTensor($"{name}.weight", [inK, outM], (uint)QuantizationType.NVFP4, nvfp4);
            F32(g, $"{name}.scale", [1], [scale]);
        }
    }

    /// <summary>Multiplies the SSM input norm weights by <see cref="BoltNormBoost"/> for the Bolt-magnitude variant.</summary>
    private static float[] Boost(Variant variant, float[] norm)
    {
        if (variant == Variant.Nvfp4ScaledBoltMagnitude)
            for (int i = 0; i < norm.Length; i++) norm[i] *= BoltNormBoost;
        return norm;
    }

    /// <summary>Random NVFP4 blocks with UE4M3 scale bytes in 0x28..0x37 (0.0625 .. 0.47 after halving).</summary>
    private static byte[] RandomNvfp4(Random rng, int elements)
    {
        byte[] b = new byte[elements / 64 * 36];
        rng.NextBytes(b);
        for (int blk = 0; blk < b.Length; blk += 36)
            for (int s = 0; s < 4; s++)
                b[blk + s] = (byte)rng.Next(0x28, 0x38);
        return b;
    }

    /// <summary>This test's own transcription of llama.cpp <c>dequantize_row_nvfp4</c> + <c>ggml_ue4m3_to_fp32</c>.</summary>
    private static float[] DequantNvfp4(byte[] b)
    {
        float[] y = new float[b.Length / 36 * 64];
        for (int blk = 0; blk < b.Length / 36; blk++)
            for (int s = 0; s < 4; s++)
            {
                byte e = b[blk * 36 + s];
                int exp = (e >> 3) & 0xF, man = e & 7;
                float d = (e == 0 || e == 0x7F) ? 0f
                    : (exp == 0 ? man * MathF.Pow(2, -9) : (1 + man / 8f) * MathF.Pow(2, exp - 7)) * 0.5f;
                for (int j = 0; j < 8; j++)
                {
                    byte q = b[blk * 36 + 4 + 8 * s + j];
                    y[blk * 64 + 16 * s + j] = KValues[q & 0xF] * d;
                    y[blk * 64 + 16 * s + 8 + j] = KValues[q >> 4] * d;
                }
            }
        return y;
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return a;
    }

    private static float[] Ones(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return a;
    }

    private static void F32(GgufWriter g, string name, int[] dims, float[] data)
        => g.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
}
