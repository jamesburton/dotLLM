using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Cuda;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1083: the MoE expert-compute service on an <b>integrated</b> Vulkan GPU (<see cref="VulkanNemotronHMoeExpertService"/>), the
/// iGPU cold leg of the CUDA expert offload, held to PR C's CPU per-expert API (<see cref="NemotronHMoeExpertCompute"/>). Needs no CUDA
/// device: runs on Strix Halo's Radeon 8060S or the dev box's Arc. Skips when there is no integrated Vulkan GPU.
/// <list type="bullet">
///   <item>Random banks of every type the Vulkan Nemotron path reads for Bolt (Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS), rows of n ∈ {1, 3, 17}
///   tokens routed to two experts each: 2 and 6 rows take the indexed MMVQ / F32 arm, 34 rows the expert-grouped prefill where the
///   device has it.</item>
///   <item>The banks of PR A's synthetic Bolt fixture (four MoE layers, per-layer geometry, latent and non-latent).</item>
/// </list>
/// Both the service and the CPU oracle quantise their activations (Q8_1 / F16 on the device, Q8_K on the CPU), so each is held to a
/// double-precision reference over the dequantised weights within a bound built per output element: the activations' int8 rounding
/// (from each block's actual maximum) plus F32 accumulation (√K-scaled), propagated through relu² and the down projection. The CPU oracle must meet the same bound (so the bound is honest) and the bound must be small against the outputs (so it
/// can catch a wrong expert, row or bank).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanNemotronHMoeExpertServiceTests(ITestOutputHelper output) : IDisposable
{
    private const int ExpertDim = 256, Inter = 512, Experts = 6;

    /// <summary>Activation block the rounding bound assumes: Q8_K's 256 (the device's Q8_1 32-blocks have a smaller maximum).</summary>
    private const int RoundingBlock = 256;

    /// <summary>
    /// Standard deviations of the root-sum-square rounding error the bound allows: thousands of elements are checked per case, so
    /// three would fail by chance; six essentially never does, while a wrong expert / row / bank moves an element by O(the output).
    /// </summary>
    private const double RoundingSigmas = 6;

    private readonly List<nint> _allocs = [];

    public void Dispose()
    {
        foreach (nint p in _allocs) NativeMemory.Free((void*)p);
        foreach (nint p in _alignedAllocs) NativeMemory.AlignedFree((void*)p);
    }

    public static TheoryData<QuantizationType, QuantizationType, int> Cases()
    {
        var d = new TheoryData<QuantizationType, QuantizationType, int>();
        (QuantizationType, QuantizationType)[] banks =
        [
            (QuantizationType.Q4_K, QuantizationType.Q5_K),
            (QuantizationType.Q6_K, QuantizationType.Q6_K),
            (QuantizationType.Q5_K, QuantizationType.Q8_0),
            (QuantizationType.Q8_0, QuantizationType.Q4_K),
            (QuantizationType.IQ2_XXS, QuantizationType.IQ2_XXS),
        ];
        foreach (var (up, down) in banks)
            foreach (int n in new[] { 1, 3, 17 })
                d.Add(up, down, n);
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Service_MatchesTheCpuPerExpertApi_OnRandomBanks(QuantizationType upQt, QuantizationType downQt, int n)
    {
        string spvDir = RequireIntegratedGpu(out var device);
        using (device)
        {
            var moe = Layer(upQt, downQt, seed: 1083 + (int)upQt * 7 + (int)downQt);
            const int LayerIndex = 3, TopK = 2;
            using var service = VulkanNemotronHMoeExpertService.Create(device, ownsDevice: false, spvDir, [(LayerIndex, moe)], hiddenSize: 1024,
                maxRows: 17 * TopK, log: output.WriteLine);
            Skip.IfNot(service.Supports(LayerIndex), $"the iGPU refused {upQt}/{downQt} banks: {service.RefusalReason(LayerIndex)}");
            output.WriteLine($"{service.DeviceDescription}: {service.ImportedBanks} bank(s) imported, {service.StagedBanks} staged");

            // n tokens, each routed to two distinct experts; rows in the caller's (arbitrary) order, tokens repeated.
            var rng = new Random(n * 31 + 7);
            float* x = Floats(n * ExpertDim, seed: n);
            int rows = n * TopK;
            int[] rowExperts = new int[rows], rowTokens = new int[rows];
            for (int t = 0; t < n; t++)
            {
                int a = rng.Next(Experts), b = (a + 1 + rng.Next(Experts - 1)) % Experts;
                rowExperts[2 * t] = a;
                rowExperts[2 * t + 1] = b;
                rowTokens[2 * t] = t;
                rowTokens[2 * t + 1] = t;
            }

            VulkanNemotronHTransformerModel.MoePathCounters.Reset();
            float* y = Floats(rows * ExpertDim);
            service.Compute(LayerIndex, rowExperts, rowTokens, x, y);
            output.WriteLine($"rows={rows}: mmvq={VulkanNemotronHTransformerModel.MoePathCounters.Mmvq} " +
                $"f32={VulkanNemotronHTransformerModel.MoePathCounters.F32Indexed} iqMmvq={VulkanNemotronHTransformerModel.MoePathCounters.IqMmvq} " +
                $"groupedMmq={VulkanNemotronHTransformerModel.MoePathCounters.GroupedMmq} groupedCoopmat={VulkanNemotronHTransformerModel.MoePathCounters.GroupedCoopmat}; " +
                $"compute {service.LastComputeUs:F0} us, wait {service.LastWaitUs:F0} us");
            if (rows < VulkanNemotronHTransformerModel.MoeGroupedMinTokens)
                Assert.Equal(0, VulkanNemotronHTransformerModel.MoePathCounters.GroupedMmq + VulkanNemotronHTransformerModel.MoePathCounters.GroupedCoopmat);

            AssertRowsMatch(moe, ExpertDim, x, rowExperts, rowTokens, y, $"{upQt}/{downQt} n={n}");

            // The same rows again: identical (no state leaks between submissions).
            float* y2 = Floats(rows * ExpertDim);
            service.Compute(LayerIndex, rowExperts, rowTokens, x, y2);
            for (int i = 0; i < rows * ExpertDim; i++)
                Assert.True(BitConverter.SingleToInt32Bits(y[i]) == BitConverter.SingleToInt32Bits(y2[i]), $"element {i} differs between two identical calls");
        }
    }

    [SkippableFact]
    public void Service_OnTheSyntheticBoltFixtureBanks_MatchesTheCpuApi_AndBeginEndOverlapsHostWork()
    {
        string spvDir = RequireIntegratedGpu(out var device);
        using (device)
        {
            string scratch = Path.Combine(Path.GetTempPath(), $"dotllm-igpu-svc-{Guid.NewGuid():N}");
            Directory.CreateDirectory(scratch);
            try
            {
                foreach (bool latent in new[] { true, false })
                {
                    string path = CudaNemotronHMoeParityTests.WriteFixture(scratch, latent, zeroBias: false, f32Banks: false);
                    using var gguf = GgufFile.Open(path);
                    var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                    using var cpu = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
                    var layers = new List<(int, NemotronHMoeWeights)>();
                    for (int l = 0; l < cpu.LayerWeights.Count; l++)
                        if (cpu.LayerWeights[l].Moe is { } m) layers.Add((l, m));
                    Assert.Equal(4, layers.Count);

                    using var service = VulkanNemotronHMoeExpertService.Create(device, ownsDevice: false, spvDir, layers, config.HiddenSize,
                        maxRows: 64, log: output.WriteLine);
                    foreach (var (layer, moe) in layers)
                    {
                        if (!service.Supports(layer))
                        {
                            // Only the IQ2_XXS layer may be refused, and only on a device without integer dot product.
                            Assert.Equal(QuantizationType.IQ2_XXS, moe.UpExpsQuantType);
                            Assert.False(device.HasIntegerDotProduct);
                            continue;
                        }
                        int ed = NemotronHMoeExpertCompute.ExpertDim(moe, config.HiddenSize);
                        Assert.Equal(ed, service.ExpertDim(layer));
                        foreach (int tokens in new[] { 1, 3, 17 })
                        {
                            float* x = Floats(tokens * ed, seed: layer * 100 + tokens);
                            int rows = tokens * moe.NumExpertsPerTok;
                            int[] rowExperts = new int[rows], rowTokens = new int[rows];
                            for (int r = 0; r < rows; r++)
                            {
                                rowTokens[r] = r / moe.NumExpertsPerTok;
                                rowExperts[r] = (r * 5 + layer) % moe.NumExperts;
                            }
                            float* y = Floats(rows * ed);

                            // Begin, then host work, then End: the submission runs while the host computes the oracle.
                            service.Begin(layer, rowExperts, rowTokens, x);
                            Assert.True(service.InFlight);
                            Assert.Throws<InvalidOperationException>(() => service.Begin(layer, rowExperts, rowTokens, x));
                            float* want = Oracle(moe, ed, x, rowExperts, rowTokens);
                            Assert.Equal(rows, service.End(y));
                            Assert.False(service.InFlight);
                            AssertRowsMatch(moe, ed, x, rowExperts, rowTokens, y, $"latent={latent} blk.{layer} tokens={tokens}", want);
                        }
                    }
                }
            }
            finally
            {
                try { Directory.Delete(scratch, recursive: true); } catch { /* best-effort */ }
            }
        }
    }

    [SkippableFact]
    public void Calibration_SubmitRoundTripAndExpertTimes_ArePositive_AndBadCallsAreRefused()
    {
        string spvDir = RequireIntegratedGpu(out var device);
        using (device)
        {
            var moe = Layer(QuantizationType.Q4_K, QuantizationType.Q5_K, seed: 9);
            using var service = VulkanNemotronHMoeExpertService.Create(device, ownsDevice: false, spvDir, [(5, moe)], 1024, maxRows: 8,
                log: output.WriteLine);
            double rtt = service.MeasureSubmitRoundTripUs(reps: 10);
            double us1 = service.MeasureExpertUs(5, expert: 0, n: 1, reps: 5);
            double us8 = service.MeasureExpertUs(5, expert: 0, n: 8, reps: 5);
            output.WriteLine($"{service.DeviceDescription}: submit+fence {rtt:F0} us; one expert {us1:F0} us (n=1), {us8:F0} us (n=8)");
            Assert.True(rtt > 0 && us1 > 0 && us8 > 0);

            float* x = Floats(ExpertDim), y = Floats(9 * ExpertDim);
            Assert.Throws<InvalidOperationException>(() => service.Begin(4, [0], [0], x));                 // not a service layer
            Assert.Throws<ArgumentOutOfRangeException>(() => service.Begin(5, [Experts], [0], x));         // no such expert
            Assert.Throws<ArgumentOutOfRangeException>(() => service.Begin(5, new int[9], new int[9], x)); // over maxRows
            Assert.Throws<InvalidOperationException>(() => service.End(y));                               // nothing in flight
            Assert.False(service.InFlight);
        }
    }

    // ───────────────────────── oracle and bound ─────────────────────────

    /// <summary>PR C's CPU per-expert API, one call per row.</summary>
    private float* Oracle(NemotronHMoeWeights moe, int ed, float* x, int[] rowExperts, int[] rowTokens)
    {
        int rows = rowExperts.Length;
        float* want = Floats(rows * ed);
        for (int r = 0; r < rows; r++)
            NemotronHMoeExpertCompute.Compute(moe, rowExperts[r], ed, x + (long)rowTokens[r] * ed, 1, Floats(moe.MoeIntermediateSize),
                want + (long)r * ed);
        return want;
    }

    /// <summary>
    /// Holds the service rows <paramref name="y"/> and the CPU oracle to a double-precision reference within the per-element bound
    /// (see the type remarks), and to each other within twice it; checks the bound is not vacuous.
    /// </summary>
    private void AssertRowsMatch(NemotronHMoeWeights moe, int ed, float* x, int[] rowExperts, int[] rowTokens, float* y, string what,
        float* want = null)
    {
        int rows = rowExperts.Length, inter = moe.MoeIntermediateSize;
        if (want == null) want = Oracle(moe, ed, x, rowExperts, rowTokens);
        double worstVk = 0, worstCpu = 0, boundSum = 0, refSum = 0;
        var upCache = new Dictionary<int, float[]>();
        var downCache = new Dictionary<int, float[]>();
        for (int r = 0; r < rows; r++)
        {
            int e = rowExperts[r];
            float[] wu = upCache.TryGetValue(e, out var u) ? u : upCache[e] = Dequant(moe.UpExpsWeight + (nint)(e * moe.UpPerExpertBytes), moe.UpExpsQuantType, inter * ed);
            float[] wd = downCache.TryGetValue(e, out var dd) ? dd : downCache[e] = Dequant(moe.DownExpsWeight + (nint)(e * moe.DownPerExpertBytes), moe.DownExpsQuantType, ed * inter);
            float* xr = x + (long)rowTokens[r] * ed;

            // Up: pre-activation and its error bound: the input's int8 rounding (each element off by at most its 256-block maximum /
            // 254) as a root-sum-square of independent uniform errors at RoundingSigmas, plus sqrt(K)-scaled F32 accumulation.
            double[] dx = BlockRounding(xr, ed);
            var mid = new double[inter];
            var midErr = new double[inter];
            for (int j = 0; j < inter; j++)
            {
                double acc = 0, sq = 0, q = 0;
                for (int c = 0; c < ed; c++)
                {
                    double w = wu[(long)j * ed + c];
                    double t = w * xr[c];
                    acc += t;
                    sq += t * t;
                    q += w * w * dx[c] * dx[c];
                }
                double eps = RoundingSigmas * Math.Sqrt(q / 3) + 1e-6 * Math.Sqrt(ed) * Math.Sqrt(sq) + 1e-7;
                double pre = Math.Max(acc, 0);
                mid[j] = pre * pre;
                midErr[j] = 2 * (Math.Abs(acc) + eps) * eps;   // relu^2 of a value off by eps
            }

            // Down: the rounding of its own quantised input (relu^2 output, block maxima grown by the carried error), the up error
            // carried through (root-sum-square of independent errors, at RoundingSigmas / 2 since midErr is already a bound), and
            // sqrt(K)-scaled accumulation.
            double[] dmid = BlockRounding(mid, midErr, inter);
            for (int i = 0; i < ed; i++)
            {
                double acc = 0, sq = 0, carried = 0, q = 0;
                for (int j = 0; j < inter; j++)
                {
                    double w = wd[(long)i * inter + j];
                    double t = w * mid[j];
                    acc += t;
                    sq += t * t;
                    carried += w * w * midErr[j] * midErr[j];
                    q += w * w * dmid[j] * dmid[j];
                }
                double bound = RoundingSigmas * Math.Sqrt(q / 3) + RoundingSigmas / 2 * Math.Sqrt(carried)
                    + 1e-6 * Math.Sqrt(inter) * Math.Sqrt(sq) + 1e-6;
                double vk = y[(long)r * ed + i], cpu = want[(long)r * ed + i];
                Assert.False(float.IsNaN((float)vk), $"{what}: row {r} element {i} is NaN");
                worstVk = Math.Max(worstVk, Math.Abs(vk - acc) / bound);
                worstCpu = Math.Max(worstCpu, Math.Abs(cpu - acc) / bound);
                Assert.True(Math.Abs(vk - cpu) <= 2 * bound,
                    $"{what}: row {r} (expert {e}, token {rowTokens[r]}) element {i}: iGPU {vk:G6} vs CPU {cpu:G6}, reference {acc:G6}, bound {bound:E2}");
                boundSum += bound;
                refSum += Math.Abs(acc);
            }
        }
        output.WriteLine($"{what}: worst |iGPU - ref| / bound {worstVk:F3}, worst |CPU - ref| / bound {worstCpu:F3}, " +
            $"mean bound / mean |ref| {boundSum / Math.Max(refSum, 1e-30):F3}");
        Assert.True(worstCpu <= 1, $"{what}: the CPU oracle exceeds the bound ({worstCpu:F3}): the bound is too tight to judge the iGPU");
        Assert.True(worstVk <= 1, $"{what}: the iGPU exceeds the bound ({worstVk:F3})");
        Assert.True(boundSum < 0.5 * refSum, $"{what}: the bound ({boundSum:E2}) is not small against the outputs ({refSum:E2})");
    }

    /// <summary>Per element, the largest int8 rounding error of its <see cref="RoundingBlock"/>-element block: block max / 254.</summary>
    private static double[] BlockRounding(float* a, int n)
    {
        var d = new double[n];
        for (int b0 = 0; b0 < n; b0 += RoundingBlock)
        {
            int b1 = Math.Min(n, b0 + RoundingBlock);
            double max = 0;
            for (int i = b0; i < b1; i++) max = Math.Max(max, Math.Abs(a[i]));
            for (int i = b0; i < b1; i++) d[i] = max / 254;
        }
        return d;
    }

    /// <summary>As above for values known only to within <paramref name="err"/>: the block maximum grows by the error.</summary>
    private static double[] BlockRounding(double[] a, double[] err, int n)
    {
        var d = new double[n];
        for (int b0 = 0; b0 < n; b0 += RoundingBlock)
        {
            int b1 = Math.Min(n, b0 + RoundingBlock);
            double max = 0;
            for (int i = b0; i < b1; i++) max = Math.Max(max, Math.Abs(a[i]) + err[i]);
            for (int i = b0; i < b1; i++) d[i] = max / 254;
        }
        return d;
    }

    private static float[] Dequant(nint p, QuantizationType qt, long elements)
    {
        var f = new float[elements];
        Dequantize.ToFloat32(p, elements, qt, f);
        return f;
    }

    // ───────────────────────── fixture ─────────────────────────

    /// <summary>
    /// Skips unless an integrated Vulkan GPU exists (never the NVIDIA device), and opens it; returns the SPIR-V directory.
    /// </summary>
    private static string RequireIntegratedGpu(out VulkanDevice device)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var pick = VulkanIgpuDeviceSelector.Select(VulkanDevice.EnumeratePhysicalDevices(),
            Environment.GetEnvironmentVariable("DOTLLM_MOE_IGPU_DEVICE"), out string reason);
        Skip.If(pick is null, $"no integrated Vulkan GPU: {reason}");
        device = VulkanDevice.Create(pick!.Value.Index);
        return spvDir;
    }

    private NemotronHMoeWeights Layer(QuantizationType upQt, QuantizationType downQt, int seed)
    {
        var rng = new Random(seed);
        long upPer = Dequantize.RowByteSize(ExpertDim, upQt) * Inter;
        long downPer = Dequantize.RowByteSize(Inter, downQt) * ExpertDim;
        return new NemotronHMoeWeights
        {
            GateInpWeight = 0, GateInpQuantType = QuantizationType.F32,
            SelectionBias = new float[Experts],
            UpExpsWeight = Bank(rng, upQt, Experts * Inter, ExpertDim, upPer * Experts, 0.12f), UpExpsQuantType = upQt, UpPerExpertBytes = upPer,
            DownExpsWeight = Bank(rng, downQt, Experts * ExpertDim, Inter, downPer * Experts, 0.06f), DownExpsQuantType = downQt,
            DownPerExpertBytes = downPer,
            UpShexpWeight = 0, UpShexpQuantType = QuantizationType.F32,
            DownShexpWeight = 0, DownShexpQuantType = QuantizationType.F32,
            NumExperts = Experts, NumExpertsPerTok = 2, MoeIntermediateSize = Inter, SharedIntermediateSize = 1,
            NormalizeWeights = true, WeightsScale = 1f, LatentSize = ExpertDim,
        };
    }

    private nint Bank(Random rng, QuantizationType qt, int rows, int k, long bytes, float amp)
    {
        byte[] b;
        if (qt == QuantizationType.IQ2_XXS)
        {
            // Random codebook indices and signs with small positive block scales (the format has no float quantiser here).
            b = new byte[bytes];
            rng.NextBytes(b);
            for (long blk = 0; blk < bytes; blk += 66)
            {
                ushort h = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.002f));
                b[blk] = (byte)(h & 0xFF);
                b[blk + 1] = (byte)(h >> 8);
            }
        }
        else
        {
            float[] w = new float[(long)rows * k];
            for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
            b = qt switch
            {
                QuantizationType.Q4_K => Q4KFixture.QuantizeRows(w, rows, k),
                QuantizationType.Q5_K => Q5KFixture.QuantizeRows(w, rows, k),
                QuantizationType.Q6_K => Q6KFixture.QuantizeRows(w, rows, k),
                QuantizationType.Q8_0 => Quantize.FromFloat32(w, w.Length, QuantizationType.Q8_0),
                _ => throw new NotSupportedException(qt.ToString()),
            };
        }
        Assert.Equal(bytes, b.LongLength);
        // Page-aligned, like a mapped GGUF slice, so the zero-copy import is exercised rather than refused for alignment.
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)bytes, 65536);
        _alignedAllocs.Add((nint)p);
        b.CopyTo(new Span<byte>(p, b.Length));
        return (nint)p;
    }

    private readonly List<nint> _alignedAllocs = [];

    private float* Floats(int count, int seed = -1)
    {
        float* p = (float*)NativeMemory.AllocZeroed((nuint)Math.Max(count, 1), sizeof(float));
        _allocs.Add((nint)p);
        if (seed >= 0)
        {
            var rng = new Random(seed);
            for (int i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        return p;
    }
}
