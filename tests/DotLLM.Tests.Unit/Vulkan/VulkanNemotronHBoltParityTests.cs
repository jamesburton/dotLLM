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
/// #1045: Nemotron 3.5 Bolt's routed-MoE features on Vulkan, held to the CPU forward (pinned to llama.cpp b11739
/// <c>build_ffn_layer</c> by <c>NemotronHBoltLatentMoeTests</c> and to real llama.cpp logits in #1035) and to a double-precision
/// transcription of the same graph.
/// <list type="bullet">
///   <item><b>LatentMoE</b>: layers 0 and 1 run their experts on <c>ffn_latent_down(x)</c> and send the weighted sum through
///   <c>ffn_latent_up</c>; the router and the shared expert read the full x.</item>
///   <item><b>Per-layer geometry</b>: the three MoE layers differ in expert count (4 / 6 / 5), top-k (2 / 3 / 1) and width
///   (512 / 256 / 256), and layer 2 is NOT latent (its experts read the hidden size), so a scratch or dispatch sized from any one
///   layer, or a latent switch applied model-wide, is wrong somewhere.</item>
///   <item><b>K-quant expert banks</b>: Q4_K / Q5_K / Q6_K / Q8_0, the AD-Q4_K_M and AD-Q6_K mix.</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHBoltParityTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const int Vocab = 16, Hidden = 512, Latent = 256, SharedInter = 64;
    private const float Eps = 1e-5f, WScale = 5.0f;

    private static readonly LayerSpec[] Layers =
    [
        new(Experts: 4, TopK: 2, Inter: 512, Latent: Latent, UpQt: QuantizationType.Q4_K, DownQt: QuantizationType.Q5_K),
        new(Experts: 6, TopK: 3, Inter: 256, Latent: Latent, UpQt: QuantizationType.Q6_K, DownQt: QuantizationType.Q6_K),
        new(Experts: 5, TopK: 1, Inter: 256, Latent: 0, UpQt: QuantizationType.Q5_K, DownQt: QuantizationType.Q8_0),
    ];

    // 20 rows: at the grouped-prefill threshold (16) and enough that experts receive zero, one and many routed rows.
    private static readonly int[] Tokens = [3, 9, 1, 14, 6, 0, 11, 5, 15, 2, 8, 12, 7, 4, 13, 10, 3, 9, 6, 1];
    private static readonly int[] Positions = [.. Enumerable.Range(0, 20)];

    private sealed record LayerSpec(int Experts, int TopK, int Inter, int Latent, QuantizationType UpQt, QuantizationType DownQt)
    {
        public int ExpertDim => Latent > 0 ? Latent : Hidden;
    }

    /// <param name="mmvq">dp4a MMVQ indexed arm (int8 activations) vs the F32-activation indexed arm.</param>
    /// <param name="grouped">Expert-grouped prefill (#1046): layer 0 (Q4_K/Q5_K) takes the dp4a MMQ kernels, layer 1 (Q6_K/Q6_K) the coopmat
    /// kernels, layer 2 (Q8_0 down) stays indexed. Single-token forwards are below the threshold, so they check the indexed decode arm.</param>
    /// <param name="xdiv">#1049: K-quant up projections on the MMVQ arm read the per-token rows (xDiv = topK, multi-row Q4_K kernel) instead
    /// of a topK-fold broadcast.</param>
    [SkippableTheory]
    [InlineData(true, false, true)]
    [InlineData(true, false, false)]
    [InlineData(false, false, true)]
    [InlineData(true, true, true)]
    public void Forward_BoltLatentHeterogeneousKQuant_MatchesCpuAndReference(bool mmvq, bool grouped, bool xdiv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.If(mmvq && !device.HasIntegerDotProduct, "device lacks integer dot product");

        using var fx = new Fixture(zeroBias: false);
        float[] cpu = fx.Cpu(Tokens, Positions);
        float[] exact = fx.Reference(Tokens);
        float[] cpuNoBias;
        using (var noBias = new Fixture(zeroBias: true)) cpuNoBias = noBias.Cpu(Tokens, Positions);

        // Control: the fixture must be sensitive to the selection bias, or "matches CPU" proves nothing about the routing.
        float control = MaxDiff(cpu, cpuNoBias);
        Assert.True(control > 0.05f, $"fixture insensitive to the selection bias (max logit delta {control:E3})");
        float cpuVsExact = MaxDiff(exact, cpu);
        Assert.True(cpuVsExact * 4 < control, $"CPU oracle vs reference {cpuVsExact:E3} is not small against the bias effect {control:E3}");

        bool saved = VulkanNemotronHTransformerModel.MoeMmvqEnabled, savedGrouped = VulkanNemotronHTransformerModel.MoeGroupedEnabled;
        VulkanNemotronHTransformerModel.MoeMmvqEnabled = mmvq;
        VulkanNemotronHTransformerModel.MoeGroupedEnabled = grouped;
        bool savedXDiv = VulkanNemotronHTransformerModel.MoeXDivEnabled;
        VulkanNemotronHTransformerModel.MoeXDivEnabled = xdiv;
        VulkanNemotronHTransformerModel.MoePathCounters.Reset();
        try
        {
            float[] vk = fx.Vulkan(device, spvDir, Tokens, Positions);
            long viaMmvq = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.Mmvq);
            long viaF32 = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.F32Indexed);
            long viaMmq = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.GroupedMmq);
            long viaCoopmat = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.GroupedCoopmat);
            long viaXDiv = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.XDivUp);
            long viaMr = Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.MultiRowUp);
            output.WriteLine($"paths: mmvq={viaMmvq} f32={viaF32} groupedMmq={viaMmq} groupedCoopmat={viaCoopmat} xdivUp={viaXDiv} multiRowUp={viaMr}");
            if (!grouped && mmvq && xdiv && device.HasIntegerDotProduct)
            {
                // Every layer's up bank is a K-quant: all three take the xDiv MMVQ; layers 0 (Q4_K) and 2 (Q5_K) the multi-row kernel,
                // and layer 0's Q5_K down bank the multi-row Q5_K kernel.
                Assert.Equal(3, viaXDiv);
                if (VulkanNemotronHTransformerModel.MoeUpMultiRow > 1)
                {
                    Assert.Equal(2, viaMr);
                    Assert.Equal(1, Interlocked.Read(ref VulkanNemotronHTransformerModel.MoePathCounters.MultiRowDown));
                }
            }
            if (!xdiv || !mmvq) Assert.Equal(0, viaXDiv);
            if (grouped)
            {
                // One forward: layer 0 on MMQ, layer 1 on coopmat (where the device has them), layer 2 (Q8_0 down) indexed.
                if (device.HasIntegerDotProduct) Assert.Equal(1, viaMmq);
                if (device.HasCooperativeMatrix) Assert.Equal(1, viaCoopmat);
                Assert.True(viaMmq + viaCoopmat > 0, "no layer took the grouped prefill");
            }
            else
            {
                Assert.True(viaMmq + viaCoopmat == 0, $"grouped prefill ran while disabled (mmq={viaMmq}, coopmat={viaCoopmat})");
                if (mmvq) Assert.True(viaMmvq > 0 && viaF32 == 0, $"MMVQ arm did not run for every bank (mmvq={viaMmvq}, f32={viaF32})");
                else Assert.True(viaF32 > 0 && viaMmvq == 0, $"F32 indexed arm did not run for every bank (mmvq={viaMmvq}, f32={viaF32})");
            }

            // Single-token forwards (the decode dispatch shapes) must agree with the batched forward row for row: the model has no
            // attention or SSM layer, so every row is independent.
            float[] vkRows = new float[vk.Length];
            for (int t = 0; t < Tokens.Length; t++)
                Array.Copy(fx.Vulkan(device, spvDir, [Tokens[t]], [Positions[t]]), 0, vkRows, t * Vocab, Vocab);

            float vsCpu = MaxDiff(cpu, vk), vsExact = MaxDiff(exact, vk), rowsVsExact = MaxDiff(exact, vkRows);
            output.WriteLine($"mmvq={mmvq}: |vk-exact|={vsExact:E3} |vkRows-exact|={rowsVsExact:E3} |vk-cpu|={vsCpu:E3} "
                + $"|cpu-exact|={cpuVsExact:E3} biasControl={control:E3}");
            // Both arms are held to the double-precision reference. The F32-activation arm is exact up to F32 / F16-GEMM rounding; the dp4a
            // arm rounds every expert activation to int8 per 32-block, which relu^2 and the x5 routing scale amplify (measured ~0.05 on
            // logits of this fixture; the CPU oracle, which quantizes its K-quant dot activations to Q8_K, sits ~0.5 from the reference).
            // The grouped kernels round activations to Q8_1 (MMQ) or F16 (coopmat): the dp4a tolerance.
            float tol = mmvq || grouped ? 0.1f : 5e-3f;
            Assert.True(tol * 10 < control, $"tolerance {tol} is not small against the bias effect {control:E3}");
            Assert.True(vsExact <= tol, $"max |reference - vulkan(batched, mmvq={mmvq})| = {vsExact:E3} > {tol}");
            Assert.True(rowsVsExact <= tol, $"max |reference - vulkan(single-token, mmvq={mmvq})| = {rowsVsExact:E3} > {tol}");
            Assert.True(vsCpu * 4 < control, $"max |cpu - vulkan| = {vsCpu:E3} is not small against the bias effect {control:E3}");
        }
        finally
        {
            VulkanNemotronHTransformerModel.MoeMmvqEnabled = saved;
            VulkanNemotronHTransformerModel.MoeGroupedEnabled = savedGrouped;
            VulkanNemotronHTransformerModel.MoeXDivEnabled = savedXDiv;
        }
    }

    /// <summary>#1048: the per-layer-submit profiling mode must attribute the MoE buckets and must not change a single logit.</summary>
    [SkippableFact]
    public void Profile_AttributesMoeBuckets_AndLeavesLogitsBitIdentical()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var fx = new Fixture(zeroBias: false);
        float[] plain = fx.Vulkan(device, spvDir, Tokens, Positions, profile: false, out var none);
        float[] profiled = fx.Vulkan(device, spvDir, Tokens, Positions, profile: true, out var report);
        Assert.Null(none);
        Assert.NotNull(report);
        output.WriteLine(report!.Format("nemotron-profile"));
        for (int i = 0; i < plain.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(plain[i]) == BitConverter.SingleToInt32Bits(profiled[i]),
                $"logit {i}: {plain[i]:G9} unprofiled vs {profiled[i]:G9} profiled");
        Assert.True(report.ByCategory.GetValueOrDefault("moe_routed") > 0, "routed experts not attributed");
        Assert.True(report.ByCategory.GetValueOrDefault("moe_dense") > 0, "dense MoE projections not attributed");
        Assert.True(report.ByCategory.GetValueOrDefault("lm_head") > 0, "LM head not attributed");
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>Three Bolt-shaped nemotron_h_moe layers from random weights, shared by the CPU oracle, the reference and the Vulkan model.</summary>
    private sealed unsafe class Fixture : IDisposable
    {
        private readonly float[] _embed, _outNorm, _wOut;
        private readonly List<GCHandle> _pins = [];
        private readonly NemotronHLayerWeights[] _layers;
        private readonly ModelConfig _config;

        public Fixture(bool zeroBias)
        {
            var rng = new Random(1045);
            _embed = Rand(rng, Vocab * Hidden, 0.5f);
            _outNorm = RandPos(rng, Hidden);
            _wOut = Rand(rng, Vocab * Hidden, 0.08f);
            _layers = new NemotronHLayerWeights[Layers.Length];
            for (int l = 0; l < Layers.Length; l++)
                _layers[l] = BuildLayer(rng, Layers[l], zeroBias);

            _config = new ModelConfig
            {
                Architecture = Architecture.NemotronHMoe,
                VocabSize = Vocab,
                HiddenSize = Hidden,
                IntermediateSize = 512,
                NumLayers = Layers.Length,
                NumAttentionHeads = 1,
                NumKvHeads = 1,
                HeadDim = Hidden,
                MaxSequenceLength = 64,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.None,
                ActivationFunction = ActivationFunction.ReluSquared,
                NormType = NormType.RMSNorm,
                NormEpsilon = Eps,
                TiedEmbeddings = false,
                HybridLayout = new HybridLayerLayout
                {
                    LayerKind = [.. Layers.Select(_ => HybridLayerKind.Ffn)],
                    HeadCountKv = [.. Layers.Select(_ => 0)],
                    FeedForwardLength = [.. Layers.Select(s => s.Inter)],
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
            => Vulkan(device, spvDir, tokens, positions, profile: false, out _);

        public float[] Vulkan(VulkanDevice device, string spvDir, int[] tokens, int[] positions, bool profile,
            out DotLLM.Vulkan.VulkanOpProfileReport? report)
        {
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, _config, _layers, _outNorm, Pin(_wOut), QuantizationType.F32, Vocab, Hidden,
                Pin(_embed), QuantizationType.F32, spvDir);
            model.ProfileOverride = profile;
            model.TrySetAllRowLogitsLimit(tokens.Length + 1);
            using var logits = model.Forward(tokens, positions, deviceId: 0);
            report = model.LastProfile;
            return Copy((float*)logits.DataPointer, logits.ElementCount);
        }

        /// <summary>
        /// Double-precision llama.cpp <c>build_ffn_layer</c> (MoE branch) over the DEQUANTIZED weights, no activation quantization:
        /// router and shared expert on the full normed x; experts on <c>latent_down(x)</c>, weighted sum through <c>latent_up</c>.
        /// </summary>
        public float[] Reference(int[] tokens)
        {
            var result = new List<float>();
            foreach (int tok in tokens)
            {
                double[] x = new double[Hidden];
                for (int i = 0; i < Hidden; i++) x[i] = _embed[tok * Hidden + i];
                for (int l = 0; l < _layers.Length; l++)
                {
                    var spec = Layers[l];
                    var moe = _layers[l].Moe!;
                    int n = spec.Experts, topK = spec.TopK, ff = spec.Inter, ed = spec.ExpertDim;
                    double[] h = RmsNorm(x, _layers[l].AttnNormWeight);
                    float[] router = Dequant(moe.GateInpWeight, moe.GateInpQuantType, n, Hidden);
                    double[] probs = new double[n];
                    for (int e = 0; e < n; e++)
                    {
                        double s = 0;
                        for (int i = 0; i < Hidden; i++) s += router[e * Hidden + i] * h[i];
                        probs[e] = 1.0 / (1.0 + Math.Exp(-s));
                    }
                    int[] top = Enumerable.Range(0, n).OrderByDescending(e => probs[e] + moe.SelectionBias[e]).ThenBy(e => e).Take(topK).ToArray();
                    double sum = top.Sum(e => probs[e]);
                    double[] w = top.Select(e => probs[e] / Math.Max(sum, 6.103515625e-5) * moe.WeightsScale).ToArray();

                    double[] xe = spec.Latent > 0
                        ? MatVec(Dequant(moe.LatentDownWeight, moe.LatentDownQuantType, spec.Latent, Hidden), 0, spec.Latent, Hidden, h)
                        : h;
                    float[] upBank = Dequant(moe.UpExpsWeight, moe.UpExpsQuantType, n * ff, ed);
                    float[] downBank = Dequant(moe.DownExpsWeight, moe.DownExpsQuantType, n * ed, ff);
                    double[] acc = new double[ed];
                    for (int k = 0; k < topK; k++)
                    {
                        int e = top[k];
                        double[] mid = Relu2(MatVec(upBank, e * ff, ff, ed, xe));
                        double[] o = MatVec(downBank, e * ed, ed, ff, mid);
                        for (int i = 0; i < ed; i++) acc[i] += w[k] * o[i];
                    }
                    double[] routed = spec.Latent > 0
                        ? MatVec(Dequant(moe.LatentUpWeight, moe.LatentUpQuantType, Hidden, spec.Latent), 0, Hidden, spec.Latent, acc)
                        : acc;
                    double[] shMid = Relu2(MatVec(Dequant(moe.UpShexpWeight, moe.UpShexpQuantType, SharedInter, Hidden), 0, SharedInter, Hidden, h));
                    double[] sh = MatVec(Dequant(moe.DownShexpWeight, moe.DownShexpQuantType, Hidden, SharedInter), 0, Hidden, SharedInter, shMid);
                    for (int i = 0; i < Hidden; i++) x[i] += routed[i] + sh[i];
                }
                double[] y = RmsNorm(x, _outNorm);
                result.AddRange(MatVec(_wOut, 0, Vocab, Hidden, y).Select(v => (float)v));
            }
            return [.. result];
        }

        private NemotronHLayerWeights BuildLayer(Random rng, LayerSpec spec, bool zeroBias)
        {
            int n = spec.Experts, ff = spec.Inter, ed = spec.ExpertDim;
            float[] norm = RandPos(rng, Hidden);
            float[] router = Rand(rng, n * Hidden, 0.12f);
            // Bias of the order of the sigmoid probabilities' spread, so it flips top-k for a good share of tokens.
            float[] bias = new float[n];
            for (int e = 0; e < n; e++) bias[e] = zeroBias ? 0f : (e % 2 == 0 ? 1 : -1) * (0.05f + 0.15f * e);
            nint up = PinBytes(QuantBank(Rand(rng, n * ff * ed, 0.12f), n * ff, ed, spec.UpQt), out long upBytes);
            nint down = PinBytes(QuantBank(Rand(rng, n * ed * ff, 0.06f), n * ed, ff, spec.DownQt), out long downBytes);
            nint shUp = PinBytes(QuantBank(Rand(rng, SharedInter * Hidden, 0.08f), SharedInter, Hidden, QuantizationType.Q8_0), out _);
            nint shDown = PinBytes(QuantBank(Rand(rng, Hidden * SharedInter, 0.2f), Hidden, SharedInter, QuantizationType.Q8_0), out _);
            nint latDown = 0, latUp = 0;
            if (spec.Latent > 0)
            {
                latDown = PinBytes(QuantBank(Rand(rng, spec.Latent * Hidden, 0.1f), spec.Latent, Hidden, QuantizationType.Q8_0), out _);
                latUp = PinBytes(QuantBank(Rand(rng, Hidden * spec.Latent, 0.1f), Hidden, spec.Latent, QuantizationType.Q8_0), out _);
            }
            return new NemotronHLayerWeights
            {
                AttnNormWeight = norm,
                Moe = new NemotronHMoeWeights
                {
                    GateInpWeight = Pin(router), GateInpQuantType = QuantizationType.F32,
                    SelectionBias = bias,
                    UpExpsWeight = up, UpExpsQuantType = spec.UpQt, UpPerExpertBytes = upBytes / n,
                    DownExpsWeight = down, DownExpsQuantType = spec.DownQt, DownPerExpertBytes = downBytes / n,
                    UpShexpWeight = shUp, UpShexpQuantType = QuantizationType.Q8_0,
                    DownShexpWeight = shDown, DownShexpQuantType = QuantizationType.Q8_0,
                    NumExperts = n,
                    NumExpertsPerTok = spec.TopK,
                    MoeIntermediateSize = ff,
                    SharedIntermediateSize = SharedInter,
                    NormalizeWeights = true,
                    WeightsScale = WScale,
                    LatentSize = spec.Latent,
                    LatentDownWeight = latDown, LatentDownQuantType = QuantizationType.Q8_0,
                    LatentUpWeight = latUp, LatentUpQuantType = QuantizationType.Q8_0,
                },
            };
        }

        private static byte[] QuantBank(float[] values, int rows, int cols, QuantizationType qt) => qt switch
        {
            QuantizationType.Q4_K => Q4KFixture.QuantizeRows(values, rows, cols),
            QuantizationType.Q5_K => Q5KFixture.QuantizeRows(values, rows, cols),
            QuantizationType.Q6_K => Q6KFixture.QuantizeRows(values, rows, cols),
            _ => QuantRows(values, rows, cols, qt),
        };

        private static byte[] QuantRows(float[] values, int rows, int cols, QuantizationType qt)
        {
            long rowBytes = Dequantize.RowByteSize(cols, qt);
            byte[] bytes = new byte[rowBytes * rows];
            for (int r = 0; r < rows; r++)
                Quantize.FromFloat32(values.AsSpan(r * cols, cols), cols, qt, bytes.AsSpan((int)(r * rowBytes), (int)rowBytes));
            return bytes;
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
                for (int c = 0; c < cols; c++) acc += (double)m[(long)(firstRow + r) * cols + c] * x[c];
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

        public void Dispose()
        {
            foreach (var h in _pins) h.Free();
            _pins.Clear();
        }
    }
}
