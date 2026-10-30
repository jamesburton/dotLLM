using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>Kernel-only microbench of the out-of-place group RMSNorm at the qwen4exp gated-residual read shape (#1357): rows x 4 groups x 2560, F32 (no model load).</summary>
internal static class GrBench
{
    public static void Run(string[] args)
    {
        int rows = args.Length > 1 ? int.Parse(args[1]) : 1024;
        const int S = 4, H = 2560, Reps = 40;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        using var kern = GroupRmsNormOopF32Kernel.Create(device, spv);
        long n = (long)rows * S * H;
        using var src = device.AllocateDeviceLocal(n * 4);
        using var dst = device.AllocateDeviceLocal(n * 4);
        using var w = device.AllocateDeviceLocal(S * H * 4L);
        var rnd = new Random(5);
        var host = new float[n]; for (long i = 0; i < n; i++) host[i] = (float)(rnd.NextDouble() * 2 - 1);
        device.Upload(host, src);
        device.Upload(Enumerable.Repeat(1.1f, S * H).ToArray(), w);
        double[] best = new double[3];
        for (int round = 0; round < 6; round++)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
            for (int i = 0; i < Reps; i++)
            {
                kern.Record(ctx.CommandBuffer, src, w, dst, rows, S, H, 1e-6f);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            }
            KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
            var t0 = Stopwatch.GetTimestamp();
            ctx.SubmitAndWait();
            double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Reps;
            Console.WriteLine($"[grbench] rows={rows} oop group-rms: {ms:F3} ms/dispatch, {2.0 * n * 4 / ms / 1e6:F0} GB/s (read+write)");
        }
    }
}
