using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in A/B harness for the #445 GDN-scan factorial arms at Bonsai 2's real prefill shape.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a kernel bench and not four <c>bench --device vulkan</c> launches.</b> Absolute
/// throughput on this UMA box swings ~40 % with CPU memory-bandwidth contention, and a
/// cold-vs-warm process launch can show a 2–3x phantom delta purely from GPU clock ramp. Both
/// confounds are process-scoped, so the only defence is to measure every arm inside ONE process,
/// interleaved, with the arm order reversed on alternate rounds — which is what this does. A
/// per-arm process launch cannot do that at all.
/// </para>
/// <para>
/// <b>The factorial.</b> Two hypotheses explain a kernel that runs at ~2 % of VALU peak while
/// implying 471 GB/s of state traffic: (A) it does its memory work redundantly — six passes over
/// the state column per token where four suffice; (B) it does that work at the wrong level of the
/// hierarchy — a global SSBO where a 16 KiB LDS slab would do. A ladder cannot separate
/// "A works", "B works", "only A+B works" and "neither"; the 2x2 can, in four cells.
/// </para>
/// <para>
/// <b>Inputs.</b> <c>g</c> is drawn from [0.98, 1.0] rather than the parity test's [0.5, 1.0]:
/// over 512 tokens a decay of 0.75 would drive the state into denormals, and denormal handling is
/// a timing artefact that has nothing to do with either hypothesis. The state is re-uploaded
/// before every timed dispatch so each arm sees identical inputs and no arm inherits another's
/// drifted state.
/// </para>
/// <para>
/// <b>Metric.</b> Minimum dispatch time over the rounds, per arm — the same "trust
/// <c>decode_min_ms</c>, not the mean" discipline the campaign adopted after UMA contention was
/// shown to move means by 40 %. Ratios only; the absolute milliseconds are not a claim.
/// </para>
/// <para>Enable with <c>DOTLLM_GDN_SCAN_AB=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnScanVariantBench
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the bench with the xUnit output sink.</summary>
    /// <param name="output">Sink for the results table.</param>
    public VulkanGdnScanVariantBench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Times all four #445 arms at Bonsai 2's GDN shape, interleaved and order-reversed.
    /// </summary>
    [SkippableFact]
    public void AB_GdnScanArms_Bonsai2Shape()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_GDN_SCAN_AB"), "1", StringComparison.Ordinal),
            "DOTLLM_GDN_SCAN_AB=1 to enable this A/B.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // Bonsai 2: ssm.inner_size 6144 / ssm.state_size 128 = 48 value heads,
        // ssm.group_count 16 key heads, and the pp512 prefill width.
        int seqLen = EnvInt("DOTLLM_GDN_SCAN_AB_SEQ", 512);
        const int nVHead = 48, nKHead = 16, dState = 128;
        int rounds = EnvInt("DOTLLM_GDN_SCAN_AB_ROUNDS", 5);
        int warmups = EnvInt("DOTLLM_GDN_SCAN_AB_WARMUPS", 2);

        var arms = new[]
        {
            GdnScanMultiTokenF32Kernel.Variant.Baseline,
            GdnScanMultiTokenF32Kernel.Variant.Fused,
            GdnScanMultiTokenF32Kernel.Variant.Lds,
            GdnScanMultiTokenF32Kernel.Variant.LdsFused,
        };

        var rng = new Random(445);
        float[] state0 = RandomFloats(rng, nVHead * dState * dState, 0.1f);
        float[] q = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] k = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] v = RandomFloats(rng, seqLen * nVHead * dState, 1.0f);
        float[] g = new float[seqLen * nVHead];
        float[] beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = 0.98f + 0.02f * (float)rng.NextDouble();
            beta[i] = (float)rng.NextDouble();
        }

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device: {device.DeviceName}  subgroup={device.SubgroupSize}");
        _output.WriteLine($"shape seqLen={seqLen} nVHead={nVHead} nKHead={nKHead} dState={dState}  " +
            $"rounds={rounds} warmups={warmups}");
        _output.WriteLine($"state per head = {dState * dState * sizeof(float) / 1024} KiB; " +
            $"total state = {(long)nVHead * dState * dState * sizeof(float) / 1024 / 1024} MiB");

        // Buffers allocated ONCE and reused by every arm. Per-iteration allocate/free would
        // recycle handles into the handle-keyed DescriptorSetCache and can silently serve a
        // stale descriptor set — correct-then-zeros output that mimics a kernel bug.
        using var stateBuf = device.Allocate((long)state0.Length * sizeof(float));
        using var qBuf = device.Allocate((long)q.Length * sizeof(float));
        using var kBuf = device.Allocate((long)k.Length * sizeof(float));
        using var vBuf = device.Allocate((long)v.Length * sizeof(float));
        using var gBuf = device.Allocate((long)g.Length * sizeof(float));
        using var betaBuf = device.Allocate((long)beta.Length * sizeof(float));
        using var outBuf = device.Allocate((long)seqLen * nVHead * dState * sizeof(float));
        device.Upload(q.AsSpan(), qBuf);
        device.Upload(k.AsSpan(), kBuf);
        device.Upload(v.AsSpan(), vBuf);
        device.Upload(g.AsSpan(), gBuf);
        device.Upload(beta.AsSpan(), betaBuf);

        var kernels = new Dictionary<GdnScanMultiTokenF32Kernel.Variant, GdnScanMultiTokenF32Kernel>();
        try
        {
            foreach (var arm in arms)
                kernels[arm] = GdnScanMultiTokenF32Kernel.Create(device, spvDir, arm);

            var best = new Dictionary<GdnScanMultiTokenF32Kernel.Variant, double>();
            var all = new Dictionary<GdnScanMultiTokenF32Kernel.Variant, List<double>>();
            foreach (var arm in arms) { best[arm] = double.MaxValue; all[arm] = new List<double>(); }

            for (int w = 0; w < warmups; w++)
                foreach (var arm in arms)
                    TimeOne(device, kernels[arm], stateBuf, qBuf, kBuf, vBuf, gBuf, betaBuf, outBuf,
                        state0, seqLen, nVHead, nKHead, dState);

            for (int round = 0; round < rounds; round++)
            {
                // Order reversed on alternate rounds: if the GPU is ramping or another process
                // is loading the memory system, a fixed order silently credits the change to
                // whichever arm happens to run in the favourable slot.
                var order = (round % 2 == 0) ? arms : arms.Reverse().ToArray();
                foreach (var arm in order)
                {
                    double ms = TimeOne(device, kernels[arm], stateBuf, qBuf, kBuf, vBuf, gBuf, betaBuf, outBuf,
                        state0, seqLen, nVHead, nKHead, dState);
                    all[arm].Add(ms);
                    if (ms < best[arm]) best[arm] = ms;
                }
                _output.WriteLine($"round {round} ({(round % 2 == 0 ? "fwd" : "rev")}): " +
                    string.Join("  ", order.Select(a => $"{a}={all[a][^1]:F2}ms")));
            }

            double baseMs = best[GdnScanMultiTokenF32Kernel.Variant.Baseline];
            _output.WriteLine("");
            _output.WriteLine("arm                min_ms    speedup_vs_baseline   all_ms");
            foreach (var arm in arms)
            {
                _output.WriteLine($"{arm,-16} {best[arm],8:F2}   {baseMs / best[arm],8:F3}x            " +
                    string.Join(",", all[arm].Select(x => x.ToString("F2"))));
            }
        }
        finally
        {
            foreach (var kv in kernels) kv.Value.Dispose();
        }
    }

    private static double TimeOne(
        VulkanDevice device, GdnScanMultiTokenF32Kernel kernel,
        VulkanDevice.Buffer state, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v,
        VulkanDevice.Buffer g, VulkanDevice.Buffer beta, VulkanDevice.Buffer output,
        float[] state0, int seqLen, int nVHead, int nKHead, int dState)
    {
        // Reset the recurrent state OUTSIDE the timed region: the scan mutates it in place, so
        // without this each arm would start from whatever the previous arm left behind.
        device.Upload(state0.AsSpan(), state);

        var sw = Stopwatch.StartNew();
        kernel.Launch(state, q, k, v, g, beta, output, seqLen, nVHead, nKHead, dState);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static int EnvInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;
}
