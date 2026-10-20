using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Architecture = DotLLM.Core.Configuration.Architecture;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #402: nemotron_h_moe on Vulkan. The CPU forward (<c>NemotronHTransformerModel</c>, itself pinned to llama.cpp
/// <c>build_moe_ffn</c> by <c>NemotronHMoeForwardTests</c> and to real llama.cpp logits in #926) is the oracle.
/// The fixture uses the quantizations of the published file: Q5_0 up banks, Q5_0 / Q8_0 down banks (a different one per layer),
/// Q5_0 shared experts, an F32 router with a selection bias crafted so biased and unbiased top-k pick different experts.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHMoeParityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const int Vocab = 16, Hidden = 64, NExpert = 8, TopK = 3, MoeInter = 64, SharedInter = 32;
    private const float Eps = 1e-5f, WScale = 2.5f;

    // ── Q5_0 -> Q5_1 repack (host, no GPU) ──────────────────────────────────────────────────────────────────────

    [Fact]
    public unsafe void RepackQ5_0ToQ5_1_DequantizesIdentically()
    {
        var rng = new Random(402);
        const int k = 256;                               // 8 blocks per row
        float[] row = new float[k];
        for (int trial = 0; trial < 64; trial++)
        {
            // Mixed scales, including tiny and large blocks and sign-heavy rows, so d and -16*d span the half range.
            double scale = Math.Pow(10, rng.NextDouble() * 6 - 4);
            for (int i = 0; i < k; i++) row[i] = (float)((rng.NextDouble() * 2 - 1) * scale * (i % 32 == 0 ? 8 : 1));
            byte[] q50 = Quantize.FromFloat32(row, k, QuantizationType.Q5_0);
            byte[] q51 = new byte[Dequantize.RowByteSize(k, QuantizationType.Q5_1)];
            fixed (byte* s = q50) fixed (byte* d = q51)
                VulkanWeights.ConvertQ5_0BlocksToQ5_1((nint)s, (nint)d, k / 32);

            float[] a = new float[k], b = new float[k];
            fixed (byte* s = q50) Dequantize.ToFloat32((nint)s, k, QuantizationType.Q5_0, a);
            fixed (byte* d = q51) Dequantize.ToFloat32((nint)d, k, QuantizationType.Q5_1, b);
            for (int i = 0; i < k; i++)
                Assert.True(a[i] == b[i], $"trial {trial} elem {i}: Q5_0 {a[i]:G9} vs repacked Q5_1 {b[i]:G9}");
        }
    }

    [Fact]
    public unsafe void RepackNegativeControl_BrokenMinTermIsDetected()
    {
        // The fixture above would pass a repack that left m = 0 only if no weight had q != 16; prove it would not.
        var rng = new Random(7);
        float[] row = new float[32];
        for (int i = 0; i < 32; i++) row[i] = (float)(rng.NextDouble() * 2 - 1);
        byte[] q50 = Quantize.FromFloat32(row, 32, QuantizationType.Q5_0);
        byte[] q51 = new byte[24];
        fixed (byte* s = q50) fixed (byte* d = q51)
            VulkanWeights.ConvertQ5_0BlocksToQ5_1((nint)s, (nint)d, 1);
        q51[2] = 0; q51[3] = 0;                          // sabotage: m = 0
        float[] a = new float[32], b = new float[32];
        fixed (byte* s = q50) Dequantize.ToFloat32((nint)s, 32, QuantizationType.Q5_0, a);
        fixed (byte* d = q51) Dequantize.ToFloat32((nint)d, 32, QuantizationType.Q5_1, b);
        Assert.Contains(Enumerable.Range(0, 32), i => a[i] != b[i]);
    }

    // ── Vulkan vs CPU forward ───────────────────────────────────────────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(true)]    // dp4a MMVQ arm (the default; int8-quantized activations)
    [InlineData(false)]   // F32-activation indexed arm (tight)
    public unsafe void Forward_MatchesCpu_BiasMatters_AndTheIntendedKernelRan(bool mmvq)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.If(mmvq && !device.HasIntegerDotProduct, "device lacks integer dot product");

        var fx = new Fixture(biasMode: BiasMode.Real);
        int[] tokens = [3, 9, 1, 14];
        int[] positions = [0, 1, 2, 3];
        float[] cpu = fx.Cpu(tokens, positions);
        float[] exact = fx.Reference(tokens);
        float[] cpuNoBias = new Fixture(BiasMode.Zero).Cpu(tokens, positions);

        // Control: the fixture must be sensitive to the selection bias, or "matches CPU" proves nothing about it.
        float control = 0;
        for (int i = 0; i < cpu.Length; i++) control = MathF.Max(control, MathF.Abs(cpu[i] - cpuNoBias[i]));
        Assert.True(control > 0.05f, $"fixture insensitive to the selection bias (max logit delta {control:E3})");

        bool saved = VulkanNemotronHTransformerModel.MoeMmvqEnabled;
        VulkanNemotronHTransformerModel.MoeMmvqEnabled = mmvq;
        VulkanNemotronHTransformerModel.MoePathCounters.Reset();
        try
        {
            float[] vk = fx.Vulkan(device, spvDir, tokens, positions);
            long viaMmvq = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.Mmvq);
            long viaF32 = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.F32Indexed);
            if (mmvq) { Assert.True(viaMmvq > 0 && viaF32 == 0, $"MMVQ arm did not run (mmvq={viaMmvq}, f32={viaF32})"); }
            else { Assert.True(viaF32 > 0 && viaMmvq == 0, $"F32 indexed arm did not run (mmvq={viaMmvq}, f32={viaF32})"); }

            float vsCpu = MaxDiff(cpu, vk), vsExact = MaxDiff(exact, vk), cpuVsExact = MaxDiff(exact, cpu);
            output.WriteLine($"mmvq={mmvq}: |vk-exact|={vsExact:E3} |vk-cpu|={vsCpu:E3} |cpu-exact|={cpuVsExact:E3} biasControl={control:E3}");
            // The CPU forward quantizes its activations (int8) like llama.cpp; the F32-activation Vulkan arm does not, so it is held to the
            // double-precision reference instead. The dp4a arm quantizes its activations too and is held to the CPU model.
            if (mmvq)
                Assert.True(vsCpu <= 0.06f, $"max |cpu - vulkan(mmvq)| = {vsCpu:E3}");
            else
                Assert.True(vsExact <= 2e-3f, $"max |reference - vulkan(f32 indexed)| = {vsExact:E3}");
            // The agreement must be far inside the effect being tested.
            Assert.True(MathF.Min(vsCpu, vsExact) * 4 < control, $"agreement is not small against the bias effect {control:E3}");
        }
        finally { VulkanNemotronHTransformerModel.MoeMmvqEnabled = saved; }
    }

    [SkippableFact]
    public void Forward_UnsupportedExpertQuant_FailsLoudInsteadOfWideningToF32()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        // Q4_K banks are served since #1045, but only whole 256-element super-blocks: K = 64 here cannot be, and must not widen to F32.
        var fx = new Fixture(BiasMode.Real, upQt: QuantizationType.Q4_K);
        var ex = Assert.ThrowsAny<NotSupportedException>(() => fx.Vulkan(device, spvDir, [1], [0]));
        Assert.Contains("ffn_up_exps", ex.Message, StringComparison.Ordinal);
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    private enum BiasMode { Real, Zero }

    /// <summary>Two nemotron_h_moe layers built from random weights, shared by the CPU oracle and the Vulkan model.</summary>
    private sealed unsafe class Fixture
    {
        private readonly float[] _embed, _outNorm, _wOut, _norm0, _norm1;
        private readonly List<GCHandle> _pins = [];
        private readonly NemotronHLayerWeights[] _layers;
        private readonly ModelConfig _config;

        public Fixture(BiasMode biasMode, QuantizationType upQt = QuantizationType.Q5_0)
        {
            var rng = new Random(402);
            _embed = Rand(rng, Vocab * Hidden, 0.5f);
            _outNorm = RandPos(rng, Hidden);
            _wOut = Rand(rng, Vocab * Hidden, 0.3f);
            _norm0 = RandPos(rng, Hidden);
            _norm1 = RandPos(rng, Hidden);
            // Bias in the order of the sigmoid probabilities' spread, so it flips top-k for a good share of tokens.
            float[] realBias = [0.9f, -0.9f, 0.3f, -0.3f, 0.6f, -0.6f, 0.05f, -0.05f];
            float[] bias = biasMode == BiasMode.Real ? realBias : new float[NExpert];

            _layers =
            [
                BuildLayer(rng, _norm0, bias, upQt, QuantizationType.Q8_0),
                BuildLayer(rng, _norm1, bias, upQt, QuantizationType.Q5_0),
            ];
            _config = new ModelConfig
            {
                Architecture = Architecture.NemotronHMoe,
                VocabSize = Vocab,
                HiddenSize = Hidden,
                IntermediateSize = MoeInter,
                NumLayers = 2,
                NumAttentionHeads = 1,
                NumKvHeads = 1,
                HeadDim = Hidden,
                MaxSequenceLength = 16,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.None,
                ActivationFunction = ActivationFunction.ReluSquared,
                NormType = NormType.RMSNorm,
                NormEpsilon = Eps,
                TiedEmbeddings = false,
                HybridLayout = new HybridLayerLayout
                {
                    LayerKind = [HybridLayerKind.Ffn, HybridLayerKind.Ffn],
                    HeadCountKv = [0, 0],
                    FeedForwardLength = [MoeInter, MoeInter],
                },
                SsmConfig = new MambaSsmConfig(DConv: 4, DInner: 8, DState: 4, NGroup: 1, NHead: 1),
                ChatTemplate = null,
            };
        }

        public float[] Cpu(int[] tokens, int[] positions)
        {
            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                _config, _layers, _outNorm, Pin(_embed), QuantizationType.F32, Pin(_wOut), QuantizationType.F32, Vocab, Hidden);
            using var logits = model.Forward(tokens, positions, deviceId: 0);
            return Copy((float*)logits.DataPointer, logits.ElementCount);
        }

        public float[] Vulkan(VulkanDevice device, string spvDir, int[] tokens, int[] positions)
        {
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, _config, _layers, _outNorm, Pin(_wOut), QuantizationType.F32, Vocab, Hidden,
                Pin(_embed), QuantizationType.F32, spvDir);
            // The Vulkan forward returns the last row unless all-row logits are requested.
            model.TrySetAllRowLogitsLimit(tokens.Length + 1);
            using var logits = model.Forward(tokens, positions, deviceId: 0);
            return Copy((float*)logits.DataPointer, logits.ElementCount);
        }

        /// <summary>Double-precision transcription of llama.cpp <c>build_moe_ffn</c> over the DEQUANTIZED banks (no activation quantization).</summary>
        public float[] Reference(int[] tokens)
        {
            var result = new List<float>();
            foreach (int tok in tokens)
            {
                double[] x = new double[Hidden];
                for (int i = 0; i < Hidden; i++) x[i] = _embed[tok * Hidden + i];
                foreach (var layer in _layers)
                {
                    var moe = layer.Moe!;
                    double[] h = RmsNorm(x, layer.AttnNormWeight);
                    float[] router = Dequant(moe.GateInpWeight, moe.GateInpQuantType, NExpert, Hidden);
                    double[] probs = new double[NExpert];
                    for (int e = 0; e < NExpert; e++)
                    {
                        double s = 0;
                        for (int i = 0; i < Hidden; i++) s += router[e * Hidden + i] * h[i];
                        probs[e] = 1.0 / (1.0 + Math.Exp(-s));
                    }
                    int[] top = Enumerable.Range(0, NExpert).OrderByDescending(e => probs[e] + moe.SelectionBias[e]).ThenBy(e => e).Take(TopK).ToArray();
                    double sum = top.Sum(e => probs[e]);
                    double[] w = top.Select(e => probs[e] / Math.Max(sum, 6.103515625e-5) * moe.WeightsScale).ToArray();
                    double[] acc = new double[Hidden];
                    float[] upBank = Dequant(moe.UpExpsWeight, moe.UpExpsQuantType, NExpert * MoeInter, Hidden);
                    float[] downBank = Dequant(moe.DownExpsWeight, moe.DownExpsQuantType, NExpert * Hidden, MoeInter);
                    for (int k = 0; k < TopK; k++)
                    {
                        int e = top[k];
                        double[] mid = Relu2(MatVec(upBank, e * MoeInter, MoeInter, Hidden, h));
                        double[] o = MatVec(downBank, e * Hidden, Hidden, MoeInter, mid);
                        for (int i = 0; i < Hidden; i++) acc[i] += w[k] * o[i];
                    }
                    double[] shMid = Relu2(MatVec(Dequant(moe.UpShexpWeight, moe.UpShexpQuantType, SharedInter, Hidden), 0, SharedInter, Hidden, h));
                    double[] sh = MatVec(Dequant(moe.DownShexpWeight, moe.DownShexpQuantType, Hidden, SharedInter), 0, Hidden, SharedInter, shMid);
                    for (int i = 0; i < Hidden; i++) x[i] += acc[i] + sh[i];
                }
                double[] y = RmsNorm(x, _outNorm);
                double[] logits = MatVec(_wOut, 0, Vocab, Hidden, y);
                result.AddRange(logits.Select(v => (float)v));
            }
            return [.. result];
        }

        private static float[] Dequant(nint ptr, QuantizationType qt, int rows, int cols)
        {
            float[] f = new float[(long)rows * cols];
            long rowBytes = Dequantize.RowByteSize(cols, qt);
            for (int r = 0; r < rows; r++)
                Dequantize.ToFloat32(ptr + (nint)(r * rowBytes), cols, qt, f.AsSpan(r * cols, cols));
            return f;
        }

        private static double[] RmsNorm(double[] x, float[] w)
        {
            double ss = 0;
            foreach (double v in x) ss += v * v;
            double inv = 1.0 / Math.Sqrt(ss / x.Length + Eps);
            return x.Select((v, i) => v * inv * w[i]).ToArray();
        }

        private static double[] MatVec(float[] m, int firstRow, int rows, int cols, double[] x)
        {
            double[] y = new double[rows];
            for (int r = 0; r < rows; r++)
            {
                double acc = 0;
                for (int c = 0; c < cols; c++) acc += (double)m[(firstRow + r) * cols + c] * x[c];
                y[r] = acc;
            }
            return y;
        }

        private static double[] Relu2(double[] v) => v.Select(a => a > 0 ? a * a : 0.0).ToArray();

        private static float[] Copy(float* p, long n)
        {
            var a = new float[n];
            for (long i = 0; i < n; i++) a[i] = p[i];
            return a;
        }

        private NemotronHLayerWeights BuildLayer(Random rng, float[] norm, float[] bias, QuantizationType upQt, QuantizationType downQt)
        {
            float[] router = Rand(rng, NExpert * Hidden, 0.9f);
            nint up = PinBytes(QuantBank(Rand(rng, NExpert * MoeInter * Hidden, 0.35f), NExpert * MoeInter, Hidden, upQt), out long upBytes);
            nint down = PinBytes(QuantBank(Rand(rng, NExpert * Hidden * MoeInter, 0.35f), NExpert * Hidden, MoeInter, downQt), out long downBytes);
            nint shUp = PinBytes(QuantBank(Rand(rng, SharedInter * Hidden, 0.35f), SharedInter, Hidden, QuantizationType.Q5_0), out _);
            nint shDown = PinBytes(QuantBank(Rand(rng, Hidden * SharedInter, 0.35f), Hidden, SharedInter, QuantizationType.Q5_0), out _);
            return new NemotronHLayerWeights
            {
                AttnNormWeight = norm,
                Moe = new NemotronHMoeWeights
                {
                    GateInpWeight = Pin(router), GateInpQuantType = QuantizationType.F32,
                    SelectionBias = bias,
                    UpExpsWeight = up, UpExpsQuantType = upQt, UpPerExpertBytes = upBytes / NExpert,
                    DownExpsWeight = down, DownExpsQuantType = downQt, DownPerExpertBytes = downBytes / NExpert,
                    UpShexpWeight = shUp, UpShexpQuantType = QuantizationType.Q5_0,
                    DownShexpWeight = shDown, DownShexpQuantType = QuantizationType.Q5_0,
                    NumExperts = NExpert,
                    NumExpertsPerTok = TopK,
                    MoeIntermediateSize = MoeInter,
                    SharedIntermediateSize = SharedInter,
                    NormalizeWeights = true,
                    WeightsScale = WScale,
                },
            };
        }

        private static byte[] QuantBank(float[] values, int rows, int cols, QuantizationType qt)
        {
            long rowBytes = Dequantize.RowByteSize(cols, qt);
            byte[] bytes = new byte[rowBytes * rows];
            for (int r = 0; r < rows; r++)
                Quantize.FromFloat32(values.AsSpan(r * cols, cols), cols, qt, bytes.AsSpan((int)(r * rowBytes), (int)rowBytes));
            return bytes;
        }

        private nint Pin(float[] a)
        {
            var h = GCHandle.Alloc(a, GCHandleType.Pinned);
            _pins.Add(h);
            return h.AddrOfPinnedObject();
        }

        private nint PinBytes(byte[] a, out long length)
        {
            length = a.Length;
            var h = GCHandle.Alloc(a, GCHandleType.Pinned);
            _pins.Add(h);
            return h.AddrOfPinnedObject();
        }

        private static float[] Rand(Random rng, int n, float scale)
        {
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
            return a;
        }

        private static float[] RandPos(Random rng, int n)
        {
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            return a;
        }
    }
}
