using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// Kernel-only microbench of the cooperative-matrix QSA gather attention (#1356 follow-up) at the released qwen4exp geometry
/// (24 heads / 2 KV heads, head_dim 256, 64-key blocks, budget 8 blocks), no model load.
/// <c>qsabench [variants=0,1] [tokens=4096] [queries=1024] [reps=30] [rounds=6] [splits=1]</c>: variant 0 = scalar gather, 1 = cooperative-matrix gather; per round every variant is timed, order alternating.
/// </summary>
internal static class QsaBench
{
    private static uint[] Pack(Random rng, long n, float scale)
    {
        var w = new uint[n / 2];
        for (int i = 0; i < w.Length; i++)
        {
            ushort a = BitConverter.HalfToUInt16Bits((Half)(scale * (float)(rng.NextDouble() * 2 - 1)));
            ushort b = BitConverter.HalfToUInt16Bits((Half)(scale * (float)(rng.NextDouble() * 2 - 1)));
            w[i] = a | ((uint)b << 16);
        }
        return w;
    }

    public static void Run(string[] args)
    {
        var variants = (args.Length > 1 ? args[1] : "0,1").Split(',').Select(int.Parse).ToArray();
        int tokens = args.Length > 2 ? int.Parse(args[2]) : 4096, Q = args.Length > 3 ? int.Parse(args[3]) : 1024;
        int reps = args.Length > 4 ? int.Parse(args[4]) : 30, rounds = args.Length > 5 ? int.Parse(args[5]) : 6;
        int splits = args.Length > 6 ? int.Parse(args[6]) : 1;
        const int heads = 24, kvHeads = 2, headDim = 256, block = 64, budget = 8;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spv);
        int first = tokens - Q, kvStride = kvHeads * headDim;
        var rng = new Random(77);
        var qh = new float[(long)Q * heads * headDim]; for (long i = 0; i < qh.Length; i++) qh[i] = (float)(rng.NextDouble() * 2 - 1);
        var sel = new int[Q * budget];
        for (int tq = 0; tq < Q; tq++)
        {
            int nb = (first + tq + 1) / block;
            var ids = Enumerable.Range(0, nb).OrderBy(_ => rng.Next()).Take(budget).OrderBy(x => x).ToArray();
            for (int i = 0; i < budget; i++) sel[tq * budget + i] = i < ids.Length ? ids[i] : 0;
        }
        long partRows = (long)Q * heads * splits;
        using var bQ = device.AllocateDeviceLocal(qh.Length * 4L);
        using var bK = device.AllocateDeviceLocal((long)tokens * kvStride * 2);
        using var bV = device.AllocateDeviceLocal((long)tokens * kvStride * 2);
        using var bSel = device.AllocateDeviceLocal(sel.Length * 4L);
        using var bPart = device.AllocateDeviceLocal(partRows * headDim * 4);
        using var bMs = device.AllocateDeviceLocal(partRows * 2 * 4);
        using var bOut = device.AllocateDeviceLocal((long)Q * heads * headDim * 4);
        device.Upload(qh, bQ);
        device.Upload(MemoryMarshal.AsBytes<uint>(Pack(rng, (long)tokens * kvStride, 1f)), bK);
        device.Upload(MemoryMarshal.AsBytes<uint>(Pack(rng, (long)tokens * kvStride, 2f)), bV);
        device.Upload(MemoryMarshal.AsBytes<int>(sel), bSel);

        void Rec(nint cmd) => k.RecordAttention(cmd, bQ, bK, bV, bSel, bPart, bMs, bOut, 0, Q, first, heads, kvHeads, headDim, block, budget, splits, true);
        float[] Download(int variant)
        {
            Qwen4ExpQsaKernels.CoopmatGather = variant != 0;
            using var ctx = device.CreateSubmitContext(); ctx.Begin();
            KernelSupport.HostToComputeBarrier(ctx.CommandBuffer); Rec(ctx.CommandBuffer); KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer); ctx.SubmitAndWait();
            var o = new float[(long)Q * heads * headDim]; device.Download(bOut, o); return o;
        }
        var refOut = Download(0);
        foreach (var v in variants.Where(v => v != 0))
        {
            long before = k.CoopmatAttentionDispatches;
            var o = Download(v);
            if (k.CoopmatAttentionDispatches != before + 1) throw new InvalidOperationException("cm kernel did not dispatch");
            double worst = 0, sumRel = 0; int cnt = 0;
            for (int tq = 0; tq < Q; tq++) for (int h = 0; h < heads; h++)
            {
                double n = 0, den = 0;
                for (int d = 0; d < headDim; d++) { int i = (tq * heads + h) * headDim + d; double e = refOut[i] - o[i]; n += e * e; den += (double)refOut[i] * refOut[i]; }
                double rel = Math.Sqrt(n / Math.Max(den, 1e-30)); worst = Math.Max(worst, rel); sumRel += rel; cnt++;
            }
            Console.WriteLine($"[qsabench] variant {v}: vs scalar worst relL2 {worst:E2} mean {sumRel / cnt:E2}");
        }
        var samples = variants.ToDictionary(v => v, _ => new List<double>());
        for (int round = 0; round < rounds + 1; round++)
        {
            var order = round % 2 == 0 ? variants : variants.Reverse().ToArray();
            foreach (var v in order)
            {
                Qwen4ExpQsaKernels.CoopmatGather = v != 0;
                using var ctx = device.CreateSubmitContext(); ctx.Begin();
                KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
                for (int i = 0; i < reps; i++) { Rec(ctx.CommandBuffer); KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer); }
                KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
                var t0 = Stopwatch.GetTimestamp();
                ctx.SubmitAndWait();
                if (round > 0) samples[v].Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds / reps);   // round 0 = clock ramp warm-up, discarded
            }
        }
        foreach (var v in variants)
        {
            var s = samples[v].OrderBy(x => x).ToList();
            Console.WriteLine($"[qsabench] variant {v}: tokens={tokens} Q={Q} splits={splits}  median {s[s.Count / 2]:F3} ms  range {s[0]:F3}-{s[^1]:F3} ms  (all: {string.Join(' ', samples[v].Select(x => x.ToString("F3")))})");
        }
    }
}
