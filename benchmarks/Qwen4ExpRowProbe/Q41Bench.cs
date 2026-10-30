using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// Kernel-only Q4_1 vs Q4_0 timing at Recursive Flint shapes (#1171): dense dp4a MMVQ (q_proj 4096x2048, o_proj 2048x4096), dense
/// 128x128x4 coopmat prefill GEMM (n = 512), and the indexed MoE MMVQ over 256 experts (top-6 decode rows: gate/up 512x2048 with the
/// activation broadcast, down 2048x512). Banks / weights are random bytes with a sane fp16 scale (timing only). The two formats run
/// interleaved in one process; best-of-reps per dispatch, effective GB/s over the weight bytes read. <c>q41bench [reps]</c>; GPU lock first.
/// </summary>
internal static class Q41Bench
{
    private const int Iters = 40;
    private const int WeightCopies = 16;

    public static void Run(string[] args)
    {
        int reps = args.Length > 1 ? int.Parse(args[1]) : 7;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        Console.WriteLine($"device: {device.DeviceName}  dp4a={device.HasIntegerDotProduct} coopmat={device.HasCooperativeMatrix} subgroup={device.SubgroupSize}");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spv)!;

        foreach (var (m, k) in new[] { (4096, 2048), (2048, 4096) })
        {
            var res = new Dictionary<QuantizationType, (double Us, double Gbs)>();
            foreach (var qt in new[] { QuantizationType.Q4_0, QuantizationType.Q4_1 })
            {
                using var kern = MatMulQ4_0MmvqKernel.TryCreate(device, spv, quant: qt);
                if (kern is null) { Console.WriteLine($"{qt} mmvq unavailable"); continue; }
                long bytes = (long)m * (k / 32) * BlockBytes(qt);
                // Rotate over enough copies (> 64 MiB) that the weights stream from DRAM, not the last-level cache.
                var ws = Enumerable.Range(0, WeightCopies).Select(_ => RandomWeights(device, bytes, BlockBytes(qt))).ToArray();
                using var xq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
                using var xds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
                using var y = device.Allocate((long)m * 4);
                int it = 0;
                double best = Time(device, reps, cb => kern.Record(cb, ws[it++ % ws.Length], xq, xds, y, m, k));
                foreach (var b in ws) b.Dispose();
                res[qt] = (best * 1000, bytes / 1e9 / (best / 1000));
            }
            Report($"dense MMVQ M={m} K={k}", res);
        }

        if (MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spv) && MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spv, QuantizationType.Q4_1))
        {
            const int n = 512, m = 4096, k = 2048;
            var res = new Dictionary<QuantizationType, (double Us, double Gbs)>();
            foreach (var qt in new[] { QuantizationType.Q4_0, QuantizationType.Q4_1 })
            {
                using var kern = MatMulQ4_0GemmCoopmatKernel.Create(device, spv, qt);
                long bytes = (long)m * (k / 32) * BlockBytes(qt);
                using var w = RandomWeights(device, bytes, BlockBytes(qt));
                using var b = device.Allocate((long)n * k * 4);
                using var c = device.Allocate((long)n * m * 4);
                double best = Time(device, reps, cb => kern.Record(cb, w, b, c, m, k, n), iters: 8);
                res[qt] = (best * 1000, 2.0 * n * m * k / 1e12 / (best / 1000));   // TFLOP/s in the "GB/s" slot
            }
            Report($"dense coopmat GEMM n=512 M=4096 K=2048 (TFLOP/s)", res);
        }

        const int E = 256, TopK = 6;
        foreach (var (m, k, xDiv) in new[] { (512, 2048, TopK), (2048, 512, 1) })
        {
            var res = new Dictionary<QuantizationType, (double Us, double Gbs)>();
            foreach (var qt in new[] { QuantizationType.Q4_0, QuantizationType.Q4_1 })
            {
                var f = MoeIqFormats.FromQuantizationType(qt)!.Value;
                using var kern = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, f);
                if (kern is null) { Console.WriteLine($"{qt} MoE mmvq unavailable"); continue; }
                long bankBytes = (long)E * m * (k / 32) * BlockBytes(qt);
                using var bank = RandomWeights(device, bankBytes, BlockBytes(qt));
                int xRows = TopK / xDiv;
                using var xq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(TopK, k));
                using var xds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(TopK, k));
                // A fresh random expert set per dispatch (as consecutive layers / tokens would hit), so the bank streams from DRAM.
                var rnd = new Random(11);
                var idxs = Enumerable.Range(0, Iters).Select(_ =>
                {
                    var buf = device.Allocate(TopK * 4L);
                    device.Upload(MemoryMarshal.AsBytes<int>(Enumerable.Range(0, E).OrderBy(_ => rnd.Next()).Take(TopK).ToArray()), buf);
                    return buf;
                }).ToArray();
                using var y = device.Allocate((long)TopK * m * 4);
                int it = 0;
                double best = Time(device, reps, cb => kern.Record(cb, bank, xq, xds, idxs[it++ % idxs.Length], y, m, k, TopK, E, xDiv));
                foreach (var b in idxs) b.Dispose();
                long read = (long)TopK * m * (k / 32) * BlockBytes(qt);
                res[qt] = (best * 1000, read / 1e9 / (best / 1000));
            }
            Report($"MoE indexed MMVQ E=256 top-6 M={m} K={k} xDiv={xDiv}", res);
        }
    }

    private static int BlockBytes(QuantizationType qt) => qt == QuantizationType.Q4_1 ? QuantFormat.Q4_1BlockBytes : QuantFormat.Q4_0BlockBytes;

    private static VulkanDevice.Buffer RandomWeights(VulkanDevice device, long bytes, int blockBytes)
    {
        var host = new byte[(bytes + 3) & ~3L];
        new Random(5).NextBytes(host);
        ushort d = BitConverter.HalfToUInt16Bits((Half)0.01f), mn = BitConverter.HalfToUInt16Bits((Half)(-0.08f));
        for (long b = 0; b + blockBytes <= bytes; b += blockBytes)
        {
            host[b] = (byte)d; host[b + 1] = (byte)(d >> 8);
            if (blockBytes == 20) { host[b + 2] = (byte)mn; host[b + 3] = (byte)(mn >> 8); }
        }
        var buf = device.Allocate(host.Length);
        device.Upload(host, buf);
        return buf;
    }

    /// <summary>Best per-dispatch wall time (ms) of <paramref name="iters"/> back-to-back dispatches in one submit, cold rep discarded.</summary>
    private static double Time(VulkanDevice device, int reps, Action<nint> record, int iters = Iters)
    {
        double best = double.MaxValue;
        for (int rep = 0; rep <= reps; rep++)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            for (int it = 0; it < iters; it++)
            {
                record(ctx.CommandBuffer);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            }
            long t0 = Stopwatch.GetTimestamp();
            ctx.SubmitAndWait();
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / iters;
            if (rep > 0) best = Math.Min(best, ms);
        }
        return best;
    }

    private static void Report(string title, Dictionary<QuantizationType, (double Us, double Gbs)> r)
    {
        string Cell(QuantizationType q) => r.TryGetValue(q, out var v) ? $"{v.Us,8:F1} us {v.Gbs,7:F2}" : "     n/a";
        string ratio = r.ContainsKey(QuantizationType.Q4_0) && r.ContainsKey(QuantizationType.Q4_1)
            ? $"  Q4_1/Q4_0 time {r[QuantizationType.Q4_1].Us / r[QuantizationType.Q4_0].Us:F2}x" : "";
        Console.WriteLine($"{title,-56} Q4_0 {Cell(QuantizationType.Q4_0)} | Q4_1 {Cell(QuantizationType.Q4_1)}{ratio}");
    }
}
