using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in A/B harness for the issue #441 wide-head flash-attention variants at Bonsai 2's real
/// full-attention prefill shape.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a kernel bench and not three <c>bench --device vulkan</c> launches.</b> Absolute
/// throughput on this UMA box swings ~40 % with CPU memory-bandwidth contention, and a
/// cold-vs-warm process launch can show a 2–3x phantom delta purely from GPU clock ramp. Both
/// confounds are process-scoped, so every arm is measured inside ONE process, interleaved, with
/// the arm order reversed on alternate rounds. The end-to-end pp512 numbers are corroboration;
/// this is the claim.
/// </para>
/// <para>
/// <b>The arms.</b> <c>Naive</c> is the per-token <see cref="AttentionF32Kernel"/> that Bonsai 2
/// actually ran before #441 — the honest baseline, since the fast path was never engaging.
/// <c>Hd256Br16</c> and <c>Hd256Br8</c> are the two wide flash variants. They are not a ladder:
/// they trade the same two resources in opposite directions (BR=16 → 36.2 KB LDS, 1 workgroup
/// resident per CU, each KV row read once per 16 query rows; BR=8 → 18.1 KB LDS, 3 workgroups
/// resident, each KV row read once per 8). KV-traffic amortisation and latency hiding are
/// separately plausible binding constraints here, which is exactly the situation where reading a
/// counter profile ("the memory unit is busy") names the unit and not the fix.
/// </para>
/// <para>
/// <b>The <c>Naive</c> arm's absolute number does not reconcile with the model and must not be
/// quoted.</b> It measures 84.11 ms at seq 512 for the same shape the model's own GPU timestamps
/// put at ~38 ms per layer (<c>attn_core</c> 604-621 ms over 16 full-attention layers). The flash
/// arms DO agree across the two harnesses - bench 3.32 ms vs ~2.6 ms in the pass, the gap being
/// this harness's per-<c>Launch</c> submit-and-wait - so the discrepancy is specific to the naive
/// arm and is unexplained. It is kept here as a correctness oracle and a rough sanity floor; size
/// the win against the fallback from the end-to-end <c>attn_core</c> bucket
/// (<c>scripts/441-e2e-ab.sh</c>) instead.
/// </para>
/// <para>
/// <b>Correctness first.</b> Every arm's output is compared against the naive arm before any
/// timing is reported. A faster kernel that computes something else is not a result.
/// </para>
/// <para>
/// <b>Metric.</b> Minimum wall time over the rounds, per arm — the campaign's "trust the min, not
/// the mean" discipline, adopted after UMA contention was shown to move means by 40 %. Ratios
/// only; the absolute milliseconds are not a claim.
/// </para>
/// <para>Enable with <c>DOTLLM_FLASH_HD256_AB=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionHd256VariantBench
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the bench with the xUnit output sink.</summary>
    /// <param name="output">Sink for the results table.</param>
    public VulkanFlashAttentionHd256VariantBench(ITestOutputHelper output) => _output = output;

    private enum Arm
    {
        Naive,
        Hd256Br16,
        Hd256Br8,
        Hd256Br4,
    }

    /// <summary>
    /// Times the per-token kernel against both wide flash variants at Bonsai 2's attention shape.
    /// </summary>
    [SkippableFact]
    public void AB_Hd256Arms_Bonsai2Shape()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_FLASH_HD256_AB"), "1", StringComparison.Ordinal),
            "DOTLLM_FLASH_HD256_AB=1 to enable this A/B.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // Bonsai 2 (qwen35): attention.head_count 24, head_count_kv 4 (GQA-6),
        // key_length = value_length = 256, full_attention_interval 4 -> 16 of 64 layers.
        int seq = EnvInt("DOTLLM_FLASH_HD256_AB_SEQ", 512);
        const int numHeads = 24, numKvHeads = 4, headDim = 256;
        int rounds = EnvInt("DOTLLM_FLASH_HD256_AB_ROUNDS", 5);
        int warmups = EnvInt("DOTLLM_FLASH_HD256_AB_WARMUPS", 2);

        var arms = new[] { Arm.Naive, Arm.Hd256Br16, Arm.Hd256Br8, Arm.Hd256Br4 };

        var rng = new Random(441);
        float[] q = RandomFloats(rng, seq * numHeads * headDim);
        float[] k = RandomFloats(rng, seq * numKvHeads * headDim);
        float[] v = RandomFloats(rng, seq * numKvHeads * headDim);

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device: {device.DeviceName}  subgroup={device.SubgroupSize}");
        _output.WriteLine($"shape seq={seq} heads={numHeads}/{numKvHeads} headDim={headDim}  " +
            $"rounds={rounds} warmups={warmups}");

        // Allocated ONCE and reused: per-iteration allocate/free recycles handles into the
        // handle-keyed DescriptorSetCache and can silently serve a stale set (correct-then-zeros
        // output that mimics a kernel bug).
        using var bufQ = device.Allocate((long)q.Length * sizeof(float));
        using var bufK = device.Allocate((long)k.Length * sizeof(float));
        using var bufV = device.Allocate((long)v.Length * sizeof(float));
        using var bufOut = device.Allocate((long)q.Length * sizeof(float));
        device.Upload(q.AsSpan(), bufQ);
        device.Upload(k.AsSpan(), bufK);
        device.Upload(v.AsSpan(), bufV);

        using var naive = AttentionF32Kernel.Create(device, spvDir);
        using var br16 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.Hd256Br16);
        using var br8 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.Hd256Br8);
        using var br4 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.Hd256Br4);
        Skip.If(br16.SupportedMaxHeadDim < headDim || br8.SupportedMaxHeadDim < headDim
                || br4.SupportedMaxHeadDim < headDim,
            "wide hd256 SPVs are absent from this build.");

        void Launch(Arm arm)
        {
            switch (arm)
            {
                case Arm.Naive:
                    naive.Launch(bufQ, bufK, bufV, bufOut, seq, seq, numHeads, numKvHeads, headDim);
                    break;
                case Arm.Hd256Br16:
                    br16.Launch(bufQ, bufK, bufV, bufOut, seq, seq, numHeads, numKvHeads, headDim);
                    break;
                case Arm.Hd256Br8:
                    br8.Launch(bufQ, bufK, bufV, bufOut, seq, seq, numHeads, numKvHeads, headDim);
                    break;
                default:
                    br4.Launch(bufQ, bufK, bufV, bufOut, seq, seq, numHeads, numKvHeads, headDim);
                    break;
            }
        }

        // ── correctness gate, before any timing ──
        float[] reference = new float[q.Length];
        Launch(Arm.Naive);
        device.Download(bufOut, reference);
        foreach (Arm arm in arms.Where(a => a != Arm.Naive))
        {
            float[] actual = new float[q.Length];
            Launch(arm);
            device.Download(bufOut, actual);
            double maxAbs = 0, maxRel = 0;
            for (int i = 0; i < reference.Length; i++)
            {
                double d = Math.Abs(reference[i] - actual[i]);
                maxAbs = Math.Max(maxAbs, d);
                maxRel = Math.Max(maxRel, d / Math.Max(1e-6, Math.Abs(reference[i])));
            }
            _output.WriteLine($"{arm} vs Naive: maxAbs={maxAbs:E3} maxRel={maxRel:E3}");
            Assert.True(maxAbs < 1e-3, $"{arm} diverges from the naive kernel (maxAbs={maxAbs:E3}).");
        }

        var best = arms.ToDictionary(a => a, _ => double.MaxValue);
        var all = arms.ToDictionary(a => a, _ => new List<double>());

        for (int w = 0; w < warmups; w++)
            foreach (Arm arm in arms)
                Time(Launch, arm);

        for (int round = 0; round < rounds; round++)
        {
            // Order reversed on alternate rounds: with a fixed order, a ramping GPU or another
            // process loading the memory system silently credits the change to whichever arm
            // happens to occupy the favourable slot.
            Arm[] order = (round % 2 == 0) ? arms : arms.Reverse().ToArray();
            foreach (Arm arm in order)
            {
                double ms = Time(Launch, arm);
                all[arm].Add(ms);
                if (ms < best[arm]) best[arm] = ms;
            }
            _output.WriteLine($"round {round} ({(round % 2 == 0 ? "fwd" : "rev")}): " +
                string.Join("  ", order.Select(a => $"{a}={all[a][^1]:F2}ms")));
        }

        double baseMs = best[Arm.Naive];
        _output.WriteLine("");
        _output.WriteLine("arm              min_ms   speedup_vs_naive   all_ms");
        foreach (Arm arm in arms)
        {
            _output.WriteLine($"{arm,-14} {best[arm],8:F2}   {baseMs / best[arm],8:F3}x          " +
                string.Join(",", all[arm].Select(x => x.ToString("F2"))));
        }
    }

    private static double Time(Action<Arm> launch, Arm arm)
    {
        var sw = Stopwatch.StartNew();
        launch(arm);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        float[] a = new float[count];
        for (int i = 0; i < count; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;
}
