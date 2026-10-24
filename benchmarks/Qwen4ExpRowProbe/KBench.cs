using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>Kernel-only microbench for the routed-MoE indexed MMVQ at qwen4exp shapes (no model load). Banks are random bytes (timing only).</summary>
internal static class KBench
{
    public static void Run(string[] args)
    {
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        const int E = 384, Iters = 12;
        int[] nsList = { 10, 20, 50, 80 };
        // gate/up shape and down shape
        foreach (var (name, m, k, q51) in new[] { ("Q4K gate/up", 640, 2560, false), ("Q5_1 down", 2560, 640, true) })
        {
            long rowBytes = q51 ? (long)k / 32 * 24 : (long)k / 256 * 144;
            long bankBytes = E * (long)m * rowBytes;
            var rnd = new Random(1);
            using var bank = device.Allocate(bankBytes);
            var host = new byte[bankBytes]; rnd.NextBytes(host);
            device.Upload(host, bank);
            host = null!;
            int maxN = 80;
            using var xq = device.Allocate(maxN * (long)k);
            using var xds = device.Allocate(maxN * (long)k / 32 * 8);
            using var y = device.Allocate(maxN * (long)m * 4);
            using var scale = device.Allocate(E * 4L);
            device.Upload(Enumerable.Repeat(1f, E).ToArray(), scale);
            var xqInit = new byte[maxN * k]; rnd.NextBytes(xqInit); device.Upload(xqInit, xq);
            var dsInit = new float[maxN * k / 32 * 2]; for (int i = 0; i < dsInit.Length; i++) dsInit[i] = 0.01f; device.Upload(dsInit, xds);
            var idxBufs = new VulkanDevice.Buffer[Iters];
            foreach (int n in nsList)
            {
                for (int it = 0; it < Iters; it++)
                {
                    idxBufs[it]?.Dispose();
                    idxBufs[it] = device.Allocate(n * 4L);
                    var idx = new int[n]; for (int i = 0; i < n; i++) idx[i] = rnd.Next(E);
                    device.Upload(MemoryMarshal.AsBytes<int>(idx), idxBufs[it]);
                }
                var line = $"{name} n={n,3}: ";
                foreach (int nr in new[] { 1, 2, 4, 8 })
                {
                    Action<nint, VulkanDevice.Buffer> rec;
                    IDisposable kern;
                    if (!q51)
                    {
                        var kk = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spv, MoeGroupedKQuant.Q4_K, nr)!;
                        kern = kk; rec = (cmd, idx) => kk.Record(cmd, bank, xq, xds, idx, y, m, k, n, E);
                    }
                    else
                    {
                        var kk = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spv, nr)!;
                        kern = kk; rec = (cmd, idx) => kk.Record(cmd, bank, xq, xds, idx, y, scale, m, k, n, E);
                    }
                    using (kern)
                    {
                        double best = double.MaxValue;
                        for (int rep = 0; rep < 5; rep++)
                        {
                            using var ctx = device.CreateSubmitContext();
                            ctx.Begin();
                            for (int it = 0; it < Iters; it++)
                            {
                                rec(ctx.CommandBuffer, idxBufs[it]);
                                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                            }
                            var t0 = Stopwatch.GetTimestamp();
                            ctx.SubmitAndWait();
                            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Iters);
                        }
                        double gb = n * (double)m * rowBytes / 1e9;
                        line += $"NR={nr}: {best * 1000,6:F0}us ({gb / (best / 1000),5:F0}GB/s)  ";
                    }
                }
                Console.WriteLine(line);
            }
        }
    }
}
