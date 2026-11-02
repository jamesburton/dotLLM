using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1543: the tiled per-query-position attention (#1540) on F16 KV at long contexts (2K..32K), with the row count per submit
/// capped at 128 (the shape the chunk-KV recompute uses). Checked against a double-precision reference on sampled (row, head) pairs
/// and against the one-workgroup-per-row oracle (#1534, <c>DOTLLM_VULKAN_ATTN_QPOS=ref</c>) on a few rows.
/// </summary>
/// <remarks>
/// Qwen3-4B heads (32 / 8 / 128). Positions are a shuffled random subset (incl. 0 and seqKv-1) so a workgroup's first/last token is
/// neither its smallest nor largest position, and the sampled rows always include the largest and smallest positions. K is drawn
/// at a scale that keeps the softmax spread over many keys, so a dropped / mis-indexed tail of the range moves the output.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanFlashAttentionGqaQueryPositionsLongContextTests
{
    private const int H = 32, Hkv = 8, Hd = 128, Rows = 128;
    private readonly ITestOutputHelper _out;

    /// <summary>Initializes the test with the xUnit output sink.</summary>
    /// <param name="output">Sink.</param>
    public VulkanFlashAttentionGqaQueryPositionsLongContextTests(ITestOutputHelper output) => _out = output;

    /// <summary>Test cases: implementation x KV length.</summary>
    /// <returns>Theory data.</returns>
    public static IEnumerable<object[]> Cases()
    {
        foreach (bool coopmat in new[] { false, true })
            foreach (int kv in new[] { 2048, 8192, 16384, 32768 })
                yield return new object[] { coopmat, kv };
    }

    /// <summary>F16 KV, 128 shuffled rows: sampled rows match a double reference and the #1534 oracle.</summary>
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void F16Kv_LongContext_MatchesReferenceAndOracle(bool coopmat, int seqKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = coopmat
            ? VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16, queryPositions: true)
            : VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.F16, queryPositions: true);
        Skip.If(kernel is null, $"the {(coopmat ? "coopmat" : "scalar")} qpos kernel is unavailable on {device.DeviceName}");
        using var oracle = AttentionQueryPositionsKernel.Create(device, spvDir, VulkanKvStorage.F16);

        var rng = new Random(1543 + seqKv);
        int[] positions = MakePositions(rng, seqKv, Rows);
        float[] q = Rand(rng, Rows * H * Hd, 1f);
        float[] k = Rand(rng, seqKv * Hkv * Hd, 1.5f);
        float[] v = Rand(rng, seqKv * Hkv * Hd, 3f);
        ushort[] kh = new ushort[k.Length], vh = new ushort[v.Length];
        for (int i = 0; i < k.Length; i++)
        {
            var hk = (Half)k[i]; var hv = (Half)v[i];
            kh[i] = BitConverter.HalfToUInt16Bits(hk); vh[i] = BitConverter.HalfToUInt16Bits(hv);
            k[i] = (float)hk; v[i] = (float)hv;   // the reference reads exactly what the kernel reads
        }

        using var bq = Up(device, q);
        using var bk = UpHalf(device, kh);
        using var bv = UpHalf(device, vh);
        using var bo = device.Allocate(q.Length * 4L);
        using var bp = device.Allocate(Rows * 4L);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bp);

        var sw = Stopwatch.StartNew();
        kernel!.Launch(bq, bk, bv, bo, Rows, seqKv, H, Hkv, Hd, queryPositions: bp);
        double ms = sw.Elapsed.TotalMilliseconds;
        var got = new float[q.Length];
        device.Download(bo, got);

        // Sampled rows: largest and smallest position, plus 14 others; heads span groups 0, 1, 3, 7 (first and last head of a group).
        int iMax = Array.IndexOf(positions, positions.Max()), iMin = Array.IndexOf(positions, positions.Min());
        var rowSet = new SortedSet<int> { iMax, iMin };
        while (rowSet.Count < 16) rowSet.Add(rng.Next(Rows));
        int[] heads = { 0, 3, 5, 6, 13, 12, 28, 31 };
        float absTol = coopmat ? 1e-3f : 1e-4f, relTol = coopmat ? 1e-2f : 1e-3f;
        double worst = 0; int bad = 0;
        var w = new double[seqKv];
        double scale = 1.0 / Math.Sqrt(Hd);
        foreach (int t in rowSet)
            foreach (int h in heads)
            {
                int hk = h / (H / Hkv), pq = positions[t];
                double mx = double.NegativeInfinity;
                for (int j = 0; j <= pq; j++)
                {
                    double s = 0;
                    for (int d = 0; d < Hd; d++) s += (double)q[(t * H + h) * Hd + d] * k[(j * Hkv + hk) * Hd + d];
                    w[j] = s * scale; mx = Math.Max(mx, w[j]);
                }
                double sum = 0;
                for (int j = 0; j <= pq; j++) { w[j] = Math.Exp(w[j] - mx); sum += w[j]; }
                for (int d = 0; d < Hd; d++)
                {
                    double acc = 0;
                    for (int j = 0; j <= pq; j++) acc += w[j] / sum * v[(j * Hkv + hk) * Hd + d];
                    double err = Math.Abs(acc - got[(t * H + h) * Hd + d]);
                    worst = Math.Max(worst, err);
                    if (err > absTol + relTol * Math.Abs(acc)) bad++;
                }
            }
        _out.WriteLine($"coopmat={coopmat} F16 kv={seqKv} rows={Rows}: submit {ms:F1} ms; vs f64 worst abs {worst:E2}, {bad} over tolerance");
        Assert.True(bad == 0, $"{bad} outputs exceed abs {absTol} + rel {relTol}; worst abs {worst:E2}");

        // Oracle (#1534): 4 rows incl. the largest position; the tiled kernel is relaunched on the same 4 so the row count differs too.
        int[] pick = { iMax, iMin, rowSet.ElementAt(5), rowSet.ElementAt(9) };
        int[] pos4 = pick.Select(i => positions[i]).ToArray();
        var q4 = new float[4 * H * Hd];
        for (int i = 0; i < 4; i++) Array.Copy(q, pick[i] * H * Hd, q4, i * H * Hd, H * Hd);
        using var bq4 = Up(device, q4);
        using var bp4 = device.Allocate(16);
        device.Upload(MemoryMarshal.AsBytes(pos4.AsSpan()), bp4);
        using var bo1 = device.Allocate(q4.Length * 4L); using var bo2 = device.Allocate(q4.Length * 4L);
        kernel.Launch(bq4, bk, bv, bo1, 4, seqKv, H, Hkv, Hd, queryPositions: bp4);
        oracle.Launch(bq4, bk, bv, bo2, bp4, 4, seqKv, H, Hkv, Hd);
        var a = new float[q4.Length]; var b = new float[q4.Length];
        device.Download(bo1, a); device.Download(bo2, b);
        double d2 = 0, n2 = 0;
        for (int i = 0; i < a.Length; i++) { d2 += (a[i] - b[i]) * (double)(a[i] - b[i]); n2 += (double)b[i] * b[i]; }
        double rel = Math.Sqrt(d2 / n2);
        _out.WriteLine($"  vs oracle (4 rows): rel L2 {rel:E2}");
        Assert.True(rel < (coopmat ? 3e-3 : 1e-4), $"rel L2 vs oracle {rel:E2}");
    }

    /// <summary>
    /// Opt-in timing, old (#1534 oracle) vs tiled, F16 KV, Qwen3-4B heads: <c>DOTLLM_QPOS_BENCH=1</c>. The old kernel runs 8 rows only
    /// (per-row cost is flat in rows). Hold the GPU locks first.
    /// </summary>
    [SkippableFact]
    public void Bench_F16_OldVsTiled()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_QPOS_BENCH") == "1", "DOTLLM_QPOS_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var oracle = AttentionQueryPositionsKernel.Create(device, spvDir, VulkanKvStorage.F16);
        using var tiled = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F16, queryPositions: true);
        Skip.If(tiled is null, "tiled qpos kernel unavailable");
        int reps = int.Parse(Environment.GetEnvironmentVariable("DOTLLM_QPOS_BENCH_REPS") ?? "5");
        _out.WriteLine($"device {device.DeviceName}; tiled coopmat={tiled!.IsCoopmat}; F16 KV; one layer, ms per submit-and-wait (min of {reps})");
        foreach (int kv in new[] { 2048, 8192, 16384, 32768 })
        {
            var rng = new Random(kv);
            int[] positions = MakePositions(rng, kv, Rows); Array.Sort(positions);
            float[] q = Rand(rng, Rows * H * Hd, 1f);
            ushort[] kh = RandHalf(rng, kv * Hkv * Hd, 1.5f), vh = RandHalf(rng, kv * Hkv * Hd, 3f);
            using var bq = Up(device, q); using var bk = UpHalf(device, kh); using var bv = UpHalf(device, vh);
            using var bo = device.Allocate(q.Length * 4L); using var bp = device.Allocate(Rows * 4L);
            device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), bp);
            double Time(Action a)
            {
                a();
                double best = double.MaxValue;
                for (int r = 0; r < reps; r++) { long t0 = Stopwatch.GetTimestamp(); a(); best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
                return best;
            }
            const int oldRows = 8;
            int[] oldPos = Enumerable.Range(0, oldRows).Select(i => positions[i * Rows / oldRows]).ToArray();
            using var bpOld = device.Allocate(oldRows * 4L);
            device.Upload(MemoryMarshal.AsBytes(oldPos.AsSpan()), bpOld);
            double tOld = Time(() => oracle.Launch(bq, bk, bv, bo, bpOld, oldRows, kv, H, Hkv, Hd));
            double tNew = Time(() => tiled.Launch(bq, bk, bv, bo, Rows, kv, H, Hkv, Hd, queryPositions: bp));
            _out.WriteLine($"kv={kv,6}: old {tOld,8:F2} ms / {oldRows} rows = {tOld * 1000 / oldRows,8:F0} us/row   tiled {tNew,8:F2} ms / {Rows} rows = {tNew * 1000 / Rows,7:F0} us/row   speedup/row {tOld / oldRows / (tNew / Rows):F1}x");
        }
    }

    private static int[] MakePositions(Random rng, int seqKv, int n)
    {
        var set = new SortedSet<int> { 0, seqKv - 1 };
        while (set.Count < n) set.Add(rng.Next(seqKv));
        var arr = set.ToArray();
        for (int i = arr.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (arr[i], arr[j]) = (arr[j], arr[i]); }
        return arr;
    }

    private static float[] Rand(Random rng, int n, float scale)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return a;
    }

    private static ushort[] RandHalf(Random rng, int n, float scale)
    {
        var a = new ushort[n];
        for (int i = 0; i < n; i++) a[i] = BitConverter.HalfToUInt16Bits((Half)((float)(rng.NextDouble() * 2.0 - 1.0) * scale));
        return a;
    }

    private static VulkanDevice.Buffer Up(VulkanDevice device, float[] data)
    {
        var buf = device.Allocate((long)data.Length * sizeof(float));
        device.Upload(data.AsSpan(), buf);
        return buf;
    }

    private static VulkanDevice.Buffer UpHalf(VulkanDevice device, ushort[] halves)
    {
        var buf = device.Allocate((long)halves.Length * sizeof(ushort));
        device.Upload(MemoryMarshal.AsBytes(halves.AsSpan()), buf);
        return buf;
    }
}
