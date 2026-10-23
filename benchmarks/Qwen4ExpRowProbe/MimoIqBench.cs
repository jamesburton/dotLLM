using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// Kernel-only A/B of the indexed IQ MMVQ decode kernels at MiMo-V2.6-Flash shapes (#991): gate/up IQ1_M (M 2048, K 4096), down IQ2_XXS / IQ2_XS
/// (M 4096, K 2048), 256 experts, n = 8 routed rows. V1 and V2 kernels are built in one process and interleaved; banks are random bytes (timing only,
/// plus a V1-vs-V2 output comparison). <c>mimobench [reps]</c>. Run under the GPU lock.
/// </summary>
internal static class MimoIqBench
{
    public static void Run(string[] args)
    {
        int reps = args.Length > 1 ? int.Parse(args[1]) : 7;
        string? exp = null;
        int nBanks = int.TryParse(Environment.GetEnvironmentVariable("BANKS"), out var nb) ? nb : 1;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        using var iq1 = Iq1Codebooks.Create(device);
        using var iq2 = Iq2Codebooks.Create(device);
        const int E = 256, N = 8, Iters = 46;
        var shapes = new (MoeIqQuant Q, int M, int K)[] { (MoeIqQuant.IQ1_M, 2048, 4096), (MoeIqQuant.IQ2_XXS, 4096, 2048), (MoeIqQuant.IQ2_XS, 4096, 2048) };
        if (exp is not null) shapes = shapes[..1];
        if (Environment.GetEnvironmentVariable("TOKENSIM") == "1") { TokenSim(device, spv, iq1, iq2, reps, nBanks); return; }
        foreach (var (q, m, k) in shapes)
        {
            var (bb, g) = MoeIqFormats.Describe(q);
            long rowBytes = (long)k / g * bb;
            long bankBytes = E * (long)m * rowBytes;
            var rnd = new Random(7);
            var banks = new VulkanDevice.Buffer[nBanks];
            var host = new byte[bankBytes];
            rnd.NextBytes(host);
            for (long b = 0; b + bb <= bankBytes; b += bb)
            {
                if (q == MoeIqQuant.IQ1_M)
                {
                    ushort dh = BitConverter.HalfToUInt16Bits((Half)0.01f);
                    for (int w = 0; w < 4; w++) { int o = (int)b + 48 + 2 * w; ushort sc = (ushort)(host[o] | host[o + 1] << 8); sc = (ushort)((sc & 0x0FFF) | (((dh >> (4 * w)) & 0xF) << 12)); host[o] = (byte)sc; host[o + 1] = (byte)(sc >> 8); }
                    continue;
                }
                ushort d = BitConverter.HalfToUInt16Bits((Half)0.01f);
                host[b] = (byte)d; host[b + 1] = (byte)(d >> 8);
            }
            for (int bi = 0; bi < nBanks; bi++) { banks[bi] = device.Allocate(bankBytes); device.Upload(host, banks[bi]); }
            host = null!;
            using var xq = device.Allocate(N * (long)k);
            using var xds = device.Allocate(N * (long)k / 32 * 8);
            using var y1 = device.Allocate(N * (long)m * 4);
            using var y2 = device.Allocate(N * (long)m * 4);
            var xqInit = new byte[N * k]; rnd.NextBytes(xqInit); device.Upload(xqInit, xq);
            var dsInit = new float[N * k / 32 * 2]; for (int i = 0; i < dsInit.Length; i++) dsInit[i] = 0.01f; device.Upload(dsInit, xds);
            var idxBufs = new VulkanDevice.Buffer[Iters];
            for (int it = 0; it < Iters; it++)
            {
                idxBufs[it] = device.Allocate(N * 4L);
                var idx = new int[N]; for (int i = 0; i < N; i++) idx[i] = rnd.Next(E);
                device.Upload(MemoryMarshal.AsBytes<int>(idx), idxBufs[it]);
            }
            using var k1 = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, q, null, iq2, iq1, v2: false)!;
            using var k2 = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, q, null, iq2, iq1, v2: true)!;
            Console.WriteLine($"{q} M={m} K={k} n={N}: v2 shader active = {k2.IsV2}");
            double best1 = double.MaxValue, best2 = double.MaxValue;
            for (int rep = 0; rep < reps; rep++)
            {
                foreach (var (kern, isB) in new[] { (k1, false), (k2, true) })
                {
                    using var ctx = device.CreateSubmitContext();
                    ctx.Begin();
                    for (int it = 0; it < Iters; it++)
                    {
                        kern.Record(ctx.CommandBuffer, banks[it % nBanks], xq, xds, idxBufs[it], isB ? y2 : y1, m, k, N, E);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    }
                    var t0 = Stopwatch.GetTimestamp();
                    ctx.SubmitAndWait();
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Iters;
                    if (rep == 0) continue; // discard cold
                    if (isB) best2 = Math.Min(best2, ms); else best1 = Math.Min(best1, ms);
                }
            }
            double gb = N * (double)m * rowBytes / 1e9;
            // Last iteration's outputs (same index buffer) must agree.
            var a = new float[N * m]; var b2 = new float[N * m];
            device.Download(y1, a); device.Download(y2, b2);
            double maxRel = 0; for (int i = 0; i < a.Length; i++) maxRel = Math.Max(maxRel, Math.Abs(a[i] - b2[i]) / (Math.Abs(a[i]) + 1e-3));
            Console.WriteLine($"  v1 {best1 * 1000,7:F0} us ({gb / (best1 / 1000),5:F0} GB/s)   v2 {best2 * 1000,7:F0} us ({gb / (best2 / 1000),5:F0} GB/s)   speedup {best1 / best2:F2}x   max rel diff {maxRel:E2}");
            foreach (var bf in idxBufs) bf.Dispose();
            foreach (var bk in banks) bk.Dispose();
        }
    }

    // Mimics one decode token's expert chain: 46 layers x (gate, up | down) with xDiv = 8, banks cycling over nBanks sets.
    private static void TokenSim(VulkanDevice device, string spv, Iq1Codebooks iq1, Iq2Codebooks iq2, int reps, int nBanks)
    {
        const int E = 256, N = 8, L = 46;
        var rnd = new Random(3);
        VulkanDevice.Buffer Bank(MoeIqQuant q, int m, int k)
        {
            var (bb, g) = MoeIqFormats.Describe(q);
            long bytes = E * (long)m * (k / g) * bb;
            var h = new byte[bytes]; rnd.NextBytes(h);
            if (q == MoeIqQuant.IQ1_M)
                for (long b = 0; b + bb <= bytes; b += bb)
                {
                    ushort dh = BitConverter.HalfToUInt16Bits((Half)0.01f);
                    for (int w = 0; w < 4; w++) { int o = (int)b + 48 + 2 * w; ushort sc = (ushort)(h[o] | h[o + 1] << 8); sc = (ushort)((sc & 0x0FFF) | (((dh >> (4 * w)) & 0xF) << 12)); h[o] = (byte)sc; h[o + 1] = (byte)(sc >> 8); }
                }
            else
                for (long b = 0; b + bb <= bytes; b += bb) { ushort d = BitConverter.HalfToUInt16Bits((Half)0.01f); h[b] = (byte)d; h[b + 1] = (byte)(d >> 8); }
            var buf = device.Allocate(bytes); device.Upload(h, buf); return buf;
        }
        var gate = new VulkanDevice.Buffer[nBanks]; var up = new VulkanDevice.Buffer[nBanks]; var down = new VulkanDevice.Buffer[nBanks];
        for (int i = 0; i < nBanks; i++) { gate[i] = Bank(MoeIqQuant.IQ1_M, 2048, 4096); up[i] = Bank(MoeIqQuant.IQ1_M, 2048, 4096); down[i] = Bank(MoeIqQuant.IQ2_XXS, 4096, 2048); }
        using var xq = device.Allocate(N * 4096L); using var xds = device.Allocate(N * 4096L / 32 * 8);
        using var yg = device.Allocate(N * 2048L * 4); using var yu = device.Allocate(N * 2048L * 4); using var yd = device.Allocate(N * 4096L * 4);
        var xqInit = new byte[N * 4096]; rnd.NextBytes(xqInit); device.Upload(xqInit, xq);
        var dsInit = new float[N * 4096 / 32 * 2]; for (int i = 0; i < dsInit.Length; i++) dsInit[i] = 0.01f; device.Upload(dsInit, xds);
        var idx = new VulkanDevice.Buffer[L];
        for (int it = 0; it < L; it++) { idx[it] = device.Allocate(N * 4L); var a = new int[N]; for (int i = 0; i < N; i++) a[i] = rnd.Next(E); device.Upload(MemoryMarshal.AsBytes<int>(a), idx[it]); }
        foreach (bool v2 in new[] { false, true })
        {
            using var kg = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, MoeIqQuant.IQ1_M, null, iq2, iq1, v2: v2)!;
            using var kd = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spv, MoeIqQuant.IQ2_XXS, null, iq2, iq1, v2: v2)!;
            double best = double.MaxValue;
            for (int rep = 0; rep < reps; rep++)
            {
                using var ctx = device.CreateSubmitContext(); ctx.Begin();
                for (int l = 0; l < L; l++)
                {
                    kg.Record(ctx.CommandBuffer, gate[l % nBanks], xq, xds, idx[l], yg, 2048, 4096, N, E, xDiv: N);
                    kg.Record(ctx.CommandBuffer, up[l % nBanks], xq, xds, idx[l], yu, 2048, 4096, N, E, xDiv: N);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    kd.Record(ctx.CommandBuffer, down[l % nBanks], xq, xds, idx[l], yd, 4096, 2048, N, E, xDiv: 1);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                }
                var t0 = Stopwatch.GetTimestamp(); ctx.SubmitAndWait();
                double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                if (rep > 0) best = Math.Min(best, ms);
            }
            Console.WriteLine($"TOKENSIM banks={nBanks} gate/up={(kg.IsV2 ? "v2" : "v1")}: {best:F2} ms per 46 layers");
        }
    }
}
