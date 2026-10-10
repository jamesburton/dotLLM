using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// Kernel-only microbench for the IQ-family routed-MoE kernels (#823) at qwen4exp shapes, against the Q4_K baseline (no model load; banks are
/// random bytes, so only timing is meaningful). <c>iqbench [mmvq|grouped|all]</c>. Run under the GPU lock.
/// </summary>
internal static class IqKBench
{
    private sealed record Fmt(string Name, MoeIqQuant? Iq, int BlockBytes, int Group, int M, int K);

    public static void Run(string[] args)
    {
        string mode = args.Length > 1 ? args[1] : "all";
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        const int E = 384, Iters = 12;
        using var iq3 = Iq3Codebooks.Create(device);
        using var iq2 = Iq2Codebooks.Create(device);

        var fmts = new List<Fmt>();
        foreach (var q in new[] { MoeIqQuant.IQ3_S, MoeIqQuant.IQ4_XS, MoeIqQuant.IQ2_XXS, MoeIqQuant.IQ2_XS, MoeIqQuant.IQ2_S, MoeIqQuant.IQ3_XXS })
        {
            var (bb, g) = MoeIqFormats.Describe(q);
            fmts.Add(new Fmt(q + " gate/up", q, bb, g, 640, 2560));
        }
        foreach (var q in new[] { MoeIqQuant.IQ4_NL, MoeIqQuant.Q2_0 })
        {
            var (bb, g) = MoeIqFormats.Describe(q);
            fmts.Add(new Fmt(q + " down", q, bb, g, 2560, 640));
        }
        fmts.Add(new Fmt("Q4_K gate/up (baseline)", null, 144, 256, 640, 2560));
        fmts.Add(new Fmt("Q5_1 down (baseline)", null, 24, 32, 2560, 640));

        foreach (var f in fmts)
        {
            long rowBytes = (long)f.K / f.Group * f.BlockBytes;
            long bankBytes = E * (long)f.M * rowBytes;
            var rnd = new Random(1);
            using var bank = device.Allocate(bankBytes);
            var host = new byte[bankBytes];
            rnd.NextBytes(host);
            for (long b = 0; b + f.BlockBytes <= bankBytes; b += f.BlockBytes)
            {
                ushort d = BitConverter.HalfToUInt16Bits((Half)0.01f);
                host[b] = (byte)d; host[b + 1] = (byte)(d >> 8);
            }
            device.Upload(host, bank);
            host = null!;

            if (mode is "all" or "mmvq")
            {
                int maxN = 80;
                using var xq = device.Allocate(maxN * (long)f.K);
                using var xds = device.Allocate(maxN * (long)f.K / 32 * 8);
                using var y = device.Allocate(maxN * (long)f.M * 4);
                using var scale = device.Allocate(E * 4L);
                device.Upload(Enumerable.Repeat(1f, E).ToArray(), scale);
                var xqInit = new byte[maxN * f.K]; rnd.NextBytes(xqInit); device.Upload(xqInit, xq);
                var dsInit = new float[maxN * f.K / 32 * 2]; for (int i = 0; i < dsInit.Length; i++) dsInit[i] = 0.01f; device.Upload(dsInit, xds);
                var idxBufs = new VulkanDevice.Buffer[Iters];
                foreach (int n in new[] { 10, 80 })
                {
                    for (int it = 0; it < Iters; it++)
                    {
                        idxBufs[it]?.Dispose();
                        idxBufs[it] = device.Allocate(n * 4L);
                        var idx = new int[n]; for (int i = 0; i < n; i++) idx[i] = rnd.Next(E);
                        device.Upload(MemoryMarshal.AsBytes<int>(idx), idxBufs[it]);
                    }
                    Action<nint, VulkanDevice.Buffer> rec;
                    IDisposable kern;
                    if (f.Iq is { } q)
                    {
                        var kk = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, q, iq3, iq2) ?? throw new InvalidOperationException("no kernel " + q);
                        kern = kk; rec = (cmd, idx) => kk.Record(cmd, bank, xq, xds, idx, y, f.M, f.K, n, E);
                    }
                    else if (f.Group == 256)
                    {
                        var kk = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spv, MoeGroupedKQuant.Q4_K)!;
                        kern = kk; rec = (cmd, idx) => kk.Record(cmd, bank, xq, xds, idx, y, f.M, f.K, n, E);
                    }
                    else
                    {
                        var kk = MoeIndexedMatmulQ5_1MmvqKernel.TryCreate(device, spv)!;
                        kern = kk; rec = (cmd, idx) => kk.Record(cmd, bank, xq, xds, idx, y, scale, f.M, f.K, n, E);
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
                        double gb = n * (double)f.M * rowBytes / 1e9;
                        Console.WriteLine($"MMVQ    {f.Name,-26} n={n,3}: {best * 1000,7:F0} us  ({gb / (best / 1000),5:F0} GB/s of expert bytes)");
                    }
                }
            }

            if ((mode is "all" or "grouped") && f.Iq is { } gq)
            {
                // 1K-token prefill step: 10240 expanded rows over 384 experts (~27 each).
                int rows = 10240;
                var counts = new int[E];
                for (int i = 0; i < rows; i++) counts[rnd.Next(E)]++;
                uint[] offs = new uint[E + 1];
                for (int e = 0; e < E; e++) offs[e + 1] = offs[e] + (uint)counts[e];
                using var x = device.Allocate((long)rows * f.K * 4);
                using var yy = device.Allocate((long)rows * f.M * 4);
                using var bufOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
                using var bufArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
                device.Upload(MemoryMarshal.AsBytes<uint>(offs), bufOff);
                var xh = new float[(long)rows * f.K];
                for (int i = 0; i < xh.Length; i++) xh[i] = (float)(rnd.NextDouble() - 0.5);
                device.Upload(xh, x);
                if (!MoeGroupedMatmulIqCoopmatKernel.IsSupportedOn(device, spv, gq)) { Console.WriteLine("grouped not supported"); continue; }
                using var kern = MoeGroupedMatmulIqCoopmatKernel.Create(device, spv, gq, iq3, iq2);
                using var build = MoeBuildTileListKernel.Create(device, spv);
                double best = double.MaxValue;
                for (int rep = 0; rep < 5; rep++)
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int it = 0; it < 4; it++)
                    {
                        build.Record(ctx.CommandBuffer, bufOff, bufArgs, E, kern.MTiles(f.M), kern.MTiles(f.M), kern.RowTile);
                        KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
                        kern.RecordIndirect(ctx.CommandBuffer, bank, x, bufOff, yy, bufArgs, 0, f.M, f.K, rows, E);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    }
                    var t0 = Stopwatch.GetTimestamp();
                    ctx.SubmitAndWait();
                    best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds / 4);
                }
                double tflops = 2.0 * rows * f.M * f.K / (best / 1000) / 1e12;
                Console.WriteLine($"GROUPED {f.Name,-26} rows={rows}: {best,7:F2} ms  ({tflops:F1} TFLOPS)");
            }
        }
    }
}
