using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>Kernel-only microbench for the multi-column K-quant / IQ GEMVs (#902) at qwen4exp dense-projection shapes (no model load). Weights are random bytes (timing only).</summary>
internal static class McBench
{
    public static void Run(string[] args)
    {
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        const int Iters = 8;
        var formats = args.Length > 1 ? args[1].Split(',').Select(Enum.Parse<QuantGemvMultiFormat>).ToArray()
                                      : Enum.GetValues<QuantGemvMultiFormat>();
        foreach (var fmt in formats)
        {
            int blockBytes = fmt switch { QuantGemvMultiFormat.Q4_K => 144, QuantGemvMultiFormat.Q5_K => 176, QuantGemvMultiFormat.Q6_K or QuantGemvMultiFormat.Q6_K_R2 => 210, QuantGemvMultiFormat.IQ4_XS => 136, _ => 110 };
            using var multi = MatMulQuantGemvMultiKernel.TryCreate(device, spv, fmt);
            if (multi is null) { Console.WriteLine($"{fmt}: kernel unavailable"); continue; }
            using var codebook = fmt == QuantGemvMultiFormat.IQ3_S ? MatMulIq3SGemvF32Kernel.Create(device, spv) : null;
            bool mimo = args.Length > 2 && args[2] == "mimo";   // #996: MiMo-V2.6-Flash Q6_K shapes, n = 1 (shipped coalesced GEMV) .. 4
            using var one = mimo ? MatMulQ6KGemvF32Kernel.Create(device, spv) : null;
            var shapes = mimo ? new[] { ("q 12288x4096", 12288, 4096), ("k/v 1536x4096", 1536, 4096), ("k/v 768x4096", 768, 4096), ("o 4096x8192", 4096, 8192) }
                              : new[] { ("gdn qkv  10240x2560", 10240, 2560), ("attn out 2560x6144", 2560, 6144), ("attn q  12288x2560", 12288, 2560) };
            foreach (var (name, m, k) in shapes)
            {
                long rowBytes = (long)(k / 256) * blockBytes;
                var rnd = new Random(7);
                var host = new byte[(long)m * rowBytes];
                rnd.NextBytes(host);
                // keep the fp16 scale fields finite so the timing is not NaN-propagation specific
                int[] halves = fmt switch { QuantGemvMultiFormat.Q4_K or QuantGemvMultiFormat.Q5_K => new[] { 0, 2 }, QuantGemvMultiFormat.Q6_K or QuantGemvMultiFormat.Q6_K_R2 => new[] { 208 }, _ => new[] { 0 } };
                for (long b = 0; b < (long)m * (k / 256); b++)
                    foreach (int off in halves) BitConverter.TryWriteBytes(host.AsSpan((int)(b * blockBytes + off), 2), (Half)0.003f);
                using var w = device.Allocate(((long)host.Length + 3) & ~3L);
                device.Upload(host, w);
                using var x = device.Allocate(8L * k * 4);
                device.Upload(Enumerable.Range(0, 8 * k).Select(i => (float)Math.Sin(i * 0.001)).ToArray(), x);
                using var y = device.Allocate(8L * m * 4);
                var line = $"{fmt,-7} {name}: ";
                for (int n = mimo ? 1 : 2; n <= (mimo ? 4 : 8); n++)
                {
                    double best = double.MaxValue;
                    for (int rep = 0; rep < 5; rep++)
                    {
                        using var ctx = device.CreateSubmitContext();
                        ctx.Begin();
                        for (int it = 0; it < Iters; it++)
                        {
                            if (n == 1) one!.Record(ctx.CommandBuffer, w, x, y, m, k);
                            else multi.Record(ctx.CommandBuffer, w, x, y, m, k, n, codebook?.GridBuffer);
                            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                        }
                        var t0 = Stopwatch.GetTimestamp();
                        ctx.SubmitAndWait();
                        best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Iters);
                    }
                    line += $"n={n}: {best * 1000,5:F0}us  ";
                }
                Console.WriteLine(line + $"(weights {host.Length / 1e6:F1} MB)");
            }
        }
    }
}
