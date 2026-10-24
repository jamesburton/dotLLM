using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// #1171 debug: the sink attention kernel on F32 K/V vs its F16-KV twin fed by the F16 writer, at Flint's shape (32 heads, 4 KV heads,
/// head_dim 128). <c>sinkf16 [seq] [window] [vscale]</c>. Prints max |F32 - F16| per output.
/// </summary>
internal static class SinkF16Probe
{
    public static void Run(string[] args)
    {
        int seq = args.Length > 1 ? int.Parse(args[1]) : 15;
        int window = args.Length > 2 ? int.Parse(args[2]) : 0;
        float vscale = args.Length > 3 ? float.Parse(args[3]) : 1f;
        const int H = 32, Kv = 4, D = 128;
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        using var device = VulkanDevice.Create();
        using var gk = VulkanGptOssKernels.Create(device, spv);
        using var writer = VulkanKvF16WriteKernel.Create(device, spv);
        var rnd = new Random(3);
        float[] Rand(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rnd.NextDouble() * 2 - 1) * s; return a; }
        float[] q = Rand(seq * H * D, 1f), k = Rand(seq * Kv * D, 1f), v = Rand(seq * Kv * D, vscale), sinks = Rand(H, 2f);
        using var bq = device.Allocate(q.Length * 4L); device.Upload(q, bq);
        using var bk = device.Allocate(k.Length * 4L); device.Upload(k, bk);
        using var bv = device.Allocate(v.Length * 4L); device.Upload(v, bv);
        using var bs = device.Allocate(H * 4L); device.Upload(sinks, bs);
        using var k16 = device.Allocate(k.Length * 2L);
        using var v16 = device.Allocate(v.Length * 2L);
        using var o32 = device.Allocate(q.Length * 4L);
        using var o16 = device.Allocate(q.Length * 4L);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            writer.RecordF32ToF16(ctx.CommandBuffer, bk, k16, 0, k.Length);
            writer.RecordF32ToF16(ctx.CommandBuffer, bv, v16, 0, v.Length);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            gk.RecordAttentionSinks(ctx.CommandBuffer, bq, bk, bv, o32, bs, seq, seq, H, Kv, D, 0, window);
            gk.RecordAttentionSinks(ctx.CommandBuffer, bq, k16, v16, o16, bs, seq, seq, H, Kv, D, 0, window, 0f, VulkanKvStorage.F16);
            ctx.SubmitAndWait();
        }
        var a = new float[q.Length]; var b = new float[q.Length];
        device.Download(o32, a); device.Download(o16, b);
        double maxAbs = 0, maxRef = 0; int worst = -1;
        for (int i = 0; i < a.Length; i++) { double d = Math.Abs(a[i] - b[i]); if (d > maxAbs) { maxAbs = d; worst = i; } maxRef = Math.Max(maxRef, Math.Abs(a[i])); }
        Console.WriteLine($"{device.DeviceName}: seq={seq} window={window} max|f32-f16|={maxAbs:E3} (ref max {maxRef:E3}) worst idx {worst} (t={worst / (H * D)} h={worst / D % H} d={worst % D}) a={a[worst]} b={b[worst]}");
    }
}
