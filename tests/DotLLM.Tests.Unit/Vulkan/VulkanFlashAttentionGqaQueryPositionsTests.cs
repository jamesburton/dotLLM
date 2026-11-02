using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1540: the tiled per-query-token-position attention (<c>VulkanFlashAttentionGqaKernel</c> with <c>queryPositions</c>), both
/// implementations and both KV formats, against a double-precision reference and against the one-workgroup-per-row oracle
/// (<see cref="AttentionQueryPositionsKernel"/>, #1534).
/// </summary>
/// <remarks>
/// Shapes are the ones a contiguous-row mask or a "first token's position bounds the block" shortcut gets wrong: positions are
/// SHUFFLED (row order != position order, so a workgroup's first/last token is neither its smallest nor largest position), scattered
/// over a KV range far longer than the last one, and the GQA groups are 4, 6 and 1 with more than one KV head. Tolerances are the
/// family's: scalar abs 1e-4 / rel 1e-3, cooperative-matrix abs 1e-3 / rel 1e-2.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionGqaQueryPositionsTests
{
    private readonly ITestOutputHelper _out;

    /// <summary>Initializes the test with the xUnit output sink.</summary>
    /// <param name="output">Sink.</param>
    public VulkanFlashAttentionGqaQueryPositionsTests(ITestOutputHelper output) => _out = output;

    /// <summary>Test cases: implementation x KV format x shape.</summary>
    /// <returns>Theory data.</returns>
    public static IEnumerable<object[]> Cases()
    {
        foreach (bool coopmat in new[] { false, true })
            foreach (var storage in new[] { VulkanKvStorage.F32, VulkanKvStorage.F16 })
            {
                //                       cm       storage  H   Hkv hd   seqKv rows window mode
                yield return new object[] { coopmat, storage, 32, 8, 128, 300, 70, 0, 0 };    // Qwen3-4B heads, shuffled scattered rows
                yield return new object[] { coopmat, storage, 24, 4, 256, 520, 45, 0, 0 };    // group of 6, hd256
                yield return new object[] { coopmat, storage, 32, 4, 128, 400, 50, 100, 0 };  // sliding window
                yield return new object[] { coopmat, storage, 4, 4, 64, 150, 40, 0, 0 };      // MHA (group of 1)
                yield return new object[] { coopmat, storage, 32, 8, 128, 300, 33, 0, 1 };    // all rows end far before seqKv
                yield return new object[] { coopmat, storage, 32, 8, 128, 300, 100, 0, 2 };   // dense run in the middle of the cache
            }
    }

    /// <summary>Each row attends exactly keys [0, its position] (and its window).</summary>
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void AttendsEachTokenUpToItsOwnPosition(bool coopmat, VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim,
        int seqKv, int seqQ, int window, int posMode)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = coopmat
            ? VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, storage, queryPositions: true)
            : VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, storage, queryPositions: true);
        Skip.If(kernel is null, $"the {(coopmat ? "coopmat" : "scalar")} qpos kernel is unavailable on {device.DeviceName}");

        var rng = new Random(1540 + headDim + seqKv + posMode);
        int[] positions = MakePositions(rng, seqKv, seqQ, posMode);
        float[] q = Rand(rng, seqQ * numHeads * headDim, 1f);
        float[] k = Rand(rng, seqKv * numKvHeads * headDim, 3f);
        float[] v = Rand(rng, seqKv * numKvHeads * headDim, 3f);
        if (storage == VulkanKvStorage.F16)   // the reference reads exactly what the kernel reads
            for (int i = 0; i < k.Length; i++) { k[i] = (float)(Half)k[i]; v[i] = (float)(Half)v[i]; }

        using var bq = Up(device, q);
        using var bk = UpKv(device, k, storage);
        using var bv = UpKv(device, v, storage);
        using var bo = device.Allocate(q.Length * 4L);
        using var bp = device.Allocate(seqQ * 4L);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bp);

        kernel!.Launch(bq, bk, bv, bo, seqQ, seqKv, numHeads, numKvHeads, headDim, slidingWindow: window, queryPositions: bp);
        var got = new float[q.Length];
        device.Download(bo, got);

        float absTol = coopmat ? 1e-3f : 1e-4f, relTol = coopmat ? 1e-2f : 1e-3f;
        double worst = 0;
        int bad = 0;
        int group = numHeads / numKvHeads;
        double scale = 1.0 / Math.Sqrt(headDim);
        var w = new double[seqKv];
        for (int t = 0; t < seqQ; t++)
            for (int h = 0; h < numHeads; h++)
            {
                int hk = h / group, pq = positions[t];
                double mx = double.NegativeInfinity;
                for (int j = 0; j <= pq; j++)
                {
                    if (window > 0 && pq - j >= window) { w[j] = double.NaN; continue; }
                    double s = 0;
                    for (int d = 0; d < headDim; d++) s += (double)q[(t * numHeads + h) * headDim + d] * k[(j * numKvHeads + hk) * headDim + d];
                    w[j] = s * scale;
                    mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = 0; j <= pq; j++) if (!double.IsNaN(w[j])) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int d = 0; d < headDim; d++)
                {
                    double acc = 0;
                    for (int j = 0; j <= pq; j++) if (!double.IsNaN(w[j])) acc += w[j] / sum * v[(j * numKvHeads + hk) * headDim + d];
                    double err = Math.Abs(acc - got[(t * numHeads + h) * headDim + d]);
                    worst = Math.Max(worst, err);
                    if (err > absTol + relTol * Math.Abs(acc)) bad++;
                }
            }

        _out.WriteLine($"coopmat={coopmat} {storage} H={numHeads}/{numKvHeads} hd={headDim} kv={seqKv} rows={seqQ} win={window} mode={posMode}: worst abs {worst:E2}, {bad} over tolerance");
        Assert.True(bad == 0, $"{bad} outputs exceed abs {absTol} + rel {relTol}; worst abs {worst:E2}");
    }

    /// <summary>The tiled kernel agrees with the #1534 oracle on a Qwen3-sized shape (sanity that both read the same positions buffer).</summary>
    [SkippableFact]
    public void MatchesReferenceQueryPositionsKernel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var tiled = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32, queryPositions: true);
        Skip.If(tiled is null, "tiled qpos kernel unavailable");
        using var oracle = AttentionQueryPositionsKernel.Create(device, spvDir);
        const int H = 32, Hkv = 8, hd = 128, seqKv = 700, seqQ = 90;
        var rng = new Random(15401);
        int[] positions = MakePositions(rng, seqKv, seqQ, 0);
        float[] q = Rand(rng, seqQ * H * hd, 1f), k = Rand(rng, seqKv * Hkv * hd, 3f), v = Rand(rng, seqKv * Hkv * hd, 3f);
        using var bq = Up(device, q); using var bk = Up(device, k); using var bv = Up(device, v);
        using var bo1 = device.Allocate(q.Length * 4L); using var bo2 = device.Allocate(q.Length * 4L);
        using var bp = device.Allocate(seqQ * 4L);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bp);
        tiled!.Launch(bq, bk, bv, bo1, seqQ, seqKv, H, Hkv, hd, queryPositions: bp);
        oracle.Launch(bq, bk, bv, bo2, bp, seqQ, seqKv, H, Hkv, hd);
        var a = new float[q.Length]; var b = new float[q.Length];
        device.Download(bo1, a); device.Download(bo2, b);
        double d2 = 0, n2 = 0;
        for (int i = 0; i < a.Length; i++) { d2 += (a[i] - b[i]) * (double)(a[i] - b[i]); n2 += (double)b[i] * b[i]; }
        double rel = Math.Sqrt(d2 / n2);
        _out.WriteLine($"tiled(coopmat={tiled.IsCoopmat}) vs oracle: rel L2 {rel:E2}");
        Assert.True(rel < 3e-3, $"rel L2 {rel:E2}");
    }

    /// <summary>
    /// Opt-in kernel timing, old (#1534) vs tiled, Qwen3-4B heads (32 / 8 / 128), F32 KV: <c>DOTLLM_QPOS_BENCH=1</c>. Row counts are kept small (64 / 16 / 8) so the old kernel never runs a long submit (unclean reboots correlated with 30-68 s ATTN_QPOS submits). Rows are a sorted
    /// random subset of the cache. Take the GPU lock first.
    /// </summary>
    [SkippableFact]
    public void Bench_OldVsTiled()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_QPOS_BENCH") == "1", "DOTLLM_QPOS_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var oracle = AttentionQueryPositionsKernel.Create(device, spvDir);
        using var tiled = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32, queryPositions: true);
        using var ordinary = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32);
        const int H = 32, Hkv = 8, hd = 128;
        (int kv, int rows)[] pts = { (2048, 64), (2048, 410), (8192, 128), (16384, 64) };
        string? only = Environment.GetEnvironmentVariable("DOTLLM_QPOS_BENCH_POINTS");
        int reps = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_QPOS_BENCH_REPS") ?? "5");
        _out.WriteLine($"device {device.DeviceName}; tiled coopmat={tiled?.IsCoopmat}; ms per submit-and-wait (min of {reps}), per-row in us");
        foreach (var (kv, rows) in pts)
        {
            if (only is not null && !only.Split(',').Contains(kv.ToString())) continue;
            var rng = new Random(kv);
            int[] positions = MakePositions(rng, kv, rows, 3);
            Array.Sort(positions);
            float[] q = Rand(rng, rows * H * hd, 1f), k = Rand(rng, kv * Hkv * hd, 3f), v = Rand(rng, kv * Hkv * hd, 3f);
            using var bq = Up(device, q); using var bk = Up(device, k); using var bv = Up(device, v);
            using var bo = device.Allocate(q.Length * 4L); using var bp = device.Allocate(rows * 4L);
            device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bp);
            double Time(Action a)
            {
                a();   // warm
                double best = double.MaxValue;
                for (int r = 0; r < reps; r++) { long t0 = Stopwatch.GetTimestamp(); a(); best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
                return best;
            }
            // The old kernel runs at most 64 rows at 2K and 8 at 8K+ (long submits correlated with unclean reboots); its per-row cost is
            // flat in rows, so rows beyond the cap use the capped run's per-row figure.
            int oldRows = Math.Min(rows, kv <= 2048 ? 64 : 8);
            // A strided sample of the sorted positions keeps the visible-key distribution representative.
            int[] oldPos = Enumerable.Range(0, oldRows).Select(i => positions[i * rows / oldRows]).ToArray();
            using var bpOld = device.Allocate(oldRows * 4L);
            device.Upload(MemoryMarshal.AsBytes(oldPos.AsSpan()), bpOld);
            double tOld = Time(() => oracle.Launch(bq, bk, bv, bo, bpOld, oldRows, kv, H, Hkv, hd)) * rows / oldRows;
            double tNew = tiled is null ? double.NaN : Time(() => tiled.Launch(bq, bk, bv, bo, rows, kv, H, Hkv, hd, queryPositions: bp));
            // The ordinary prefill kernel on the same row count as a contiguous tail run (the "prefill-class" control).
            double tOrd = ordinary is null ? double.NaN : Time(() => ordinary.Launch(bq, bk, bv, bo, rows, kv, H, Hkv, hd, positionOffset: kv - rows));
            _out.WriteLine($"kv={kv,6} rows={rows,4}: old {tOld,9:F2} ms ({tOld * 1000 / rows,8:F0} us/row)  tiled {tNew,8:F2} ms ({tNew * 1000 / rows,7:F0} us/row)  ordinary-tail {tOrd,8:F2} ms ({tOrd * 1000 / rows,7:F0} us/row)  speedup {tOld / tNew:F1}x");
        }
    }

    // mode 0: shuffled sorted-distinct subset incl. 0 and seqKv-1; 1: all < seqKv/3; 2: dense run in the middle, shuffled; 3: random subset
    private static int[] MakePositions(Random rng, int seqKv, int n, int mode)
    {
        int lo = 0, hi = seqKv;
        if (mode == 1) hi = seqKv / 3;
        if (mode == 2) { lo = seqKv / 3; hi = lo + n; }
        var set = new SortedSet<int>();
        if (mode == 0) { set.Add(0); set.Add(seqKv - 1); }
        while (set.Count < n) set.Add(lo + rng.Next(hi - lo));
        var arr = set.ToArray();
        if (mode != 3)
            for (int i = arr.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (arr[i], arr[j]) = (arr[j], arr[i]); }
        return arr;
    }

    private static float[] Rand(Random rng, int n, float scale)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return a;
    }

    private static VulkanDevice.Buffer Up(VulkanDevice device, float[] data)
    {
        var buf = device.Allocate((long)data.Length * sizeof(float));
        device.Upload(data.AsSpan(), buf);
        return buf;
    }

    private static VulkanDevice.Buffer UpKv(VulkanDevice device, float[] data, VulkanKvStorage storage)
    {
        if (storage == VulkanKvStorage.F32) return Up(device, data);
        ushort[] halves = new ushort[data.Length];
        for (int i = 0; i < data.Length; i++) halves[i] = BitConverter.HalfToUInt16Bits((Half)data[i]);
        var buf = device.Allocate((long)halves.Length * sizeof(ushort));
        device.Upload(MemoryMarshal.AsBytes(halves.AsSpan()), buf);
        return buf;
    }
}
