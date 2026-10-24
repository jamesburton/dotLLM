using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Flint;

/// <summary>
/// Compact F16 sliding-window KV ring on Vulkan (#1176) against the full-length F16 cache on the same device and against the CPU's compact
/// cache (#1169), on the synthetic Flint (GQA 8:2, sinks on every layer, NoPE full layers, window 20 &lt; context). The schedule forces every
/// ring path: an in-ring compaction whose source and destination overlap (staged through the scratch), a prefill chunk larger than the
/// ring (grown before recording), and decode compactions with a direct copy.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class FlintVulkanCompactKvTests
{
    // Ring 2 * (19 + 64) + 255 = 421 rows (bases are multiples of the 256-row prefill tile); a compaction keeps 83..338 rows.
    private const int Window = 20;
    private const int MaxSeq = 1024;

    // 40, 100: fit. Decode 140..439: compaction at 421, base 0 -> 256, shift 256 >= keep 165 (direct copy).
    // 340 (440..779): ring grows 421 -> 524 rows (before recording). 60 (780..839): base 256 -> 512, shift 256 < keep 268 (overlap, staged
    // through the scratch). Decode 840..939.
    private static readonly int[] Schedule = [40, 100, .. Enumerable.Repeat(1, 300), 340, 60, .. Enumerable.Repeat(1, 100)];

    private readonly ITestOutputHelper _output;

    public FlintVulkanCompactKvTests(ITestOutputHelper output) => _output = output;

    private static int[] Positions(int n) => Enumerable.Range(0, n).ToArray();

    /// <summary>Last-row logits of each forward in <paramref name="schedule"/>, starting at position <paramref name="start"/>.</summary>
    private static List<float[]> Run(Func<int[], int[], ITensor> forward, int[] ids, IEnumerable<int> schedule, int start = 0)
    {
        var rows = new List<float[]>();
        int pos = start;
        foreach (int n in schedule)
        {
            int[] p = Enumerable.Range(pos, n).ToArray();
            using (ITensor t = forward(ids[pos..(pos + n)], p))
                rows.Add(FlintVulkanParityTests.Copy(t)[^SyntheticFlintGguf.Vocab..]);
            pos += n;
        }

        return rows;
    }

    private static List<float[]> RunVk(VulkanTransformerModel vk, VulkanKvCache cache, int[] ids, IEnumerable<int> schedule, int start = 0)
        => Run((t, p) => vk.Forward(t, p, deviceId: 0, cache), ids, schedule, start);

    /// <summary>Max |a - b| over every row, and the count of rows whose argmax differs.</summary>
    private (float MaxAbs, int ArgmaxDiffs) Diff(string tag, List<float[]> a, List<float[]> b)
    {
        Assert.Equal(a.Count, b.Count);
        float maxAbs = 0;
        int flips = 0;
        for (int r = 0; r < a.Count; r++)
        {
            for (int i = 0; i < a[r].Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(a[r][i] - b[r][i]));
            if (Array.IndexOf(a[r], a[r].Max()) != Array.IndexOf(b[r], b[r].Max())) flips++;
        }

        string line = $"[flint vk compact {tag}] rows={a.Count} maxAbs={maxAbs:E2} argmaxDiffs={flips}";
        _output.WriteLine(line);
        Console.WriteLine(line);
        return (maxAbs, flips);
    }

    private static (int Windowed, int Full) Layers(FlintVulkanParityTests.Fixture fx)
    {
        var g = KvGeometry.FromConfig(fx.Config, compactSlidingWindows: true);
        int windowed = Enumerable.Range(0, g.LayerCount).First(l => g.SlidingWindowOf(l) > 0);
        int full = Enumerable.Range(0, g.LayerCount).First(l => g.SlidingWindowOf(l) == 0);
        return (windowed, full);
    }

    /// <summary>
    /// The ring matches the full-length F16 cache bit for bit over 403 forwards and three kinds of compaction, matches the CPU
    /// compact cache like the existing F16-KV parity tests, and allocates less.
    /// </summary>
    [SkippableFact]
    public void Ring_MatchesFullLengthAndCpu_AcrossCompactions()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new FlintVulkanParityTests.Fixture(FlintVulkanParityTests.Q41(window: Window));
        var (windowed, full) = Layers(fx);
        int[] ids = FlintVulkanParityTests.Ids(Schedule.Sum());

        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        Assert.True(vk.SupportsCompactSlidingKv);
        using var fullCache = vk.CreateKvCache(MaxSeq, compactSlidingWindows: false);
        using var ring = vk.CreateKvCache(MaxSeq);
        Assert.True(ring.HasCompactWindows);
        var vkFull = RunVk(vk, fullCache, ids, Schedule);
        var vkRing = RunVk(vk, ring, ids, Schedule);

        Assert.Equal(512, ring.RowBaseOf(windowed));
        Assert.Equal(0, ring.RowBaseOf(full));
        // F16 rows of Kv * HeadDim = 128 halves; windowed layers settle at the grown ring (524 rows).
        var g = KvGeometry.FromConfig(fx.Config, compactSlidingWindows: true);
        long ringRows = Enumerable.Range(0, g.LayerCount).Sum(l => g.SlidingWindowOf(l) > 0 ? 524L : MaxSeq);
        Assert.Equal(ringRows * 2 * SyntheticFlintGguf.Kv * 64 * sizeof(ushort), ring.AllocatedBytes);
        Assert.True(ring.AllocatedBytes < fullCache.AllocatedBytes);

        var (maxAbs, flips) = Diff("ring vs full", vkRing, vkFull);
        Assert.Equal(0f, maxAbs);
        Assert.Equal(0, flips);

        // CPU compact cache, same schedule: the Vulkan ring must agree like every other F16-KV parity check.
        using var cpu = TransformerModel.LoadFromGguf(fx.Gguf, fx.Config, ThreadingConfig.SingleThreaded);
        using var cpuCache = new SimpleKvCache(KvGeometry.FromConfig(fx.Config, cpu.SupportsCompactSlidingKv), MaxSeq);
        var cpuRows = Run((t, p) => cpu.Forward(t, p, deviceId: -1, cpuCache), ids, Schedule);
        Assert.True(cpuCache.RowBaseOf(windowed) > 0);
        var kl = FlintVulkanParityTests.RowKl(vkRing.SelectMany(r => r).ToArray(), cpuRows.SelectMany(r => r).ToArray(), SyntheticFlintGguf.Vocab);
        var sorted = kl.OrderBy(x => x).ToArray();
        double median = sorted[sorted.Length / 2], good = kl.Count(x => x < 0.05) / (double)kl.Length;
        _output.WriteLine($"[flint vk compact vs cpu] rows={kl.Length} medianKL={median:E2} maxKL={sorted[^1]:E2} rows<0.05={good:P0}");
        Assert.True(median < 0.01, $"median per-row KL {median}");
        Assert.True(good >= 0.84, $"only {good:P0} of rows agree");
    }

    /// <summary>
    /// Rollback within the 64-position margin keeps matching the full cache; a cut past it is refused (the prompt cache's truncation miss);
    /// emptying resets the base and the (grown) ring serves a fresh sequence.
    /// </summary>
    [SkippableFact]
    public void Ring_RollbackWithinMargin_RefusedBeyond_ResetReuses()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fx = new FlintVulkanParityTests.Fixture(FlintVulkanParityTests.Q41(window: Window));
        var (windowed, _) = Layers(fx);
        int[] ids = FlintVulkanParityTests.Ids(Schedule.Sum() + 40);

        using var vk = VulkanTransformerModel.LoadFromGguf(fx.Gguf, fx.Config, spvDir);
        using var fullCache = vk.CreateKvCache(MaxSeq, compactSlidingWindows: false);
        using var ring = vk.CreateKvCache(MaxSeq);
        int[] head = Schedule[..^30]; // stop at 910 (base 512)
        RunVk(vk, fullCache, ids, head);
        RunVk(vk, ring, ids, head);
        int pos = head.Sum();

        // Speculative-decode style rollback of 10 tokens (base 512 <= 900 - window + 1): truncation hit.
        int back = pos - 10;
        Assert.True(ring.CanTruncateTo(back));
        fullCache.Rollback(back);
        ring.Rollback(back);
        var tail = Enumerable.Repeat(1, 40).ToArray();
        var (maxAbs, flips) = Diff("after rollback", RunVk(vk, ring, ids, tail, back), RunVk(vk, fullCache, ids, tail, back));
        Assert.Equal(0f, maxAbs);
        Assert.Equal(0, flips);

        // Past the margin: the windowed layer already dropped the rows (truncation miss).
        Assert.False(ring.CanTruncateTo(Window + 1));
        Assert.Throws<InvalidOperationException>(() => ring.Rollback(Window + 1));
        Assert.Throws<InvalidOperationException>(() => ring.SetCurrentLength(Window + 1));

        ring.Reset();
        fullCache.Reset();
        Assert.Equal(0, ring.RowBaseOf(windowed));
        int[] again = [30, 200, .. Enumerable.Repeat(1, 20)];
        var (maxAbs2, flips2) = Diff("after reset", RunVk(vk, ring, ids, again), RunVk(vk, fullCache, ids, again));
        Assert.Equal(0f, maxAbs2);
        Assert.Equal(0, flips2);
    }

    /// <summary>The planner alone (no GPU): the schedule above hits the three move kinds the GPU tests rely on.</summary>
    [Fact]
    public void Planner_ScheduleHitsOverlapGrowAndDirectMoves()
    {
        var cfg = new FlintVulkanParityTests.Fixture(FlintVulkanParityTests.Q41(window: Window));
        using (cfg)
        {
            var rows = new SlidingWindowRows(KvGeometry.FromConfig(cfg.Config, compactSlidingWindows: true), MaxSeq, growOnly: true, baseAlignment: 256);
            var (windowed, _) = Layers(cfg);
            var kinds = new List<string>();
            int pos = 0;
            foreach (int n in Schedule)
            {
                if (rows.TryPlan(windowed, pos, pos + n - 1, pos, out var m))
                {
                    kinds.Add(m.Resize ? "grow" : m.Shift >= m.Keep ? "direct" : "overlap");
                    rows.Commit(windowed, m);
                }

                pos += n;
            }

            Assert.Equal(["direct", "grow", "overlap"], kinds);
        }
    }
}
