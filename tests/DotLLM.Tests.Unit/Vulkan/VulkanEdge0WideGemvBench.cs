using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark (<c>DOTLLM_EDGE0_WIDE_BENCH=1</c>; hold the Strix GPU lock) for Edge0 latent D and U at 9..16 rows (#2004), at the real
/// shapes (D: m = 8 * 1792, k = 2560; U: m = 2560, k = 1792). Arms per n: the pre-#1986 F32 wide GEMV (F32 out, the "old" target to beat), the
/// current blocked coopmat GEMM (F16 out) and the new F16 wide GEMV at NR 4 / 2. Each timed submit records <c>Reps</c> dispatches back to back, rotating
/// over several weight copies so the weights stream from DRAM like 48 different layers do (a single 27 MB D matrix would sit in the Infinity Cache).
/// Arms are timed in ABBA-interleaved passes after warm-up (the iGPU clock ramps; a cold-vs-warm comparison is a phantom delta); the table is
/// the per-arm median and min..max over passes, in microseconds per dispatch.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanEdge0WideGemvBench(ITestOutputHelper output)
{
    private const int H = 2560, L = 1792, G = 8, Copies = 6, Reps = 24, Passes = 9;

    private static byte[] RandomQ8Rows(Random rng, int rows, int k)
    {
        var b = new byte[(long)rows * (k / 32) * 34];
        rng.NextBytes(b);
        for (long o = 0; o + 34 <= b.Length; o += 34)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.02f));
            b[o] = (byte)bits; b[o + 1] = (byte)(bits >> 8);
        }
        return b;
    }

    private static VulkanDevice.Buffer Up<T>(VulkanDevice d, T[] data) where T : unmanaged
    {
        long bytes = (long)data.Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        var b = d.Allocate((bytes + 3) & ~3L);
        d.Upload(MemoryMarshal.AsBytes<T>(data), b);
        return b;
    }

    private sealed record Arm(string Name, Action<nint, int> Record);

    private void Run(VulkanDevice device, string title, int n, IReadOnlyList<Arm> arms)
    {
        var times = arms.Select(_ => new List<double>()).ToArray();
        void Time(int a)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            for (int r = 0; r < Reps; r++)
            {
                arms[a].Record(ctx.CommandBuffer, r % Copies);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            }
            var sw = Stopwatch.StartNew();
            ctx.SubmitAndWait();
            times[a].Add(sw.Elapsed.TotalMilliseconds * 1000.0 / Reps);
        }
        for (int w = 0; w < 4; w++) for (int a = 0; a < arms.Count; a++) Time(a);   // warm-up (clock ramp)
        foreach (var t in times) t.Clear();
        for (int p = 0; p < Passes; p++)
        {
            if ((p & 1) == 0) for (int a = 0; a < arms.Count; a++) Time(a);
            else for (int a = arms.Count - 1; a >= 0; a--) Time(a);
        }
        var sb = new System.Text.StringBuilder($"BENCH {title} n={n,2}:");
        for (int a = 0; a < arms.Count; a++)
        {
            var s = times[a].OrderBy(v => v).ToArray();
            sb.Append($"  {arms[a].Name} {s[s.Length / 2],7:F1} us [{s[0]:F1}..{s[^1]:F1}]");
        }
        output.WriteLine(sb.ToString());
    }

    [SkippableFact]
    public void DAndU_AtNineToSixteenRows()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_WIDE_BENCH") == "1", "DOTLLM_EDGE0_WIDE_BENCH=1 (GPU timing; hold the Strix GPU lock)");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Edge0LatentKernels.Create(device, spvDir);
        using var f32 = MatMulQ8_0GemvMultiKernel.TryCreate(device, spvDir)!;
        using var wideD4 = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: false, 4)!;
        using var wideD2 = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: false, 2)!;
        using var wideU4 = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: true, 4)!;
        using var wideU2 = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: true, 2)!;
        Skip.IfNot(k.HasNarrowPath, "coopmat GEMM unavailable");
        var rng = new Random(0xB2004);
        var dW = new VulkanDevice.Buffer[Copies]; var uW = new VulkanDevice.Buffer[Copies];
        for (int c = 0; c < Copies; c++)
        {
            byte[] d = RandomQ8Rows(rng, G * L, H); dW[c] = device.Allocate((d.Length + 3L) & ~3L); device.Upload(d, dW[c]);
            byte[] u = RandomQ8Rows(rng, H, L); uW[c] = device.Allocate((u.Length + 3L) & ~3L); device.Upload(u, uW[c]);
        }
        try
        {
            foreach (int n in new[] { 9, 10, 12, 14, 16 })
            {
                float[] x = new float[n * H]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
                float[] z = new float[n * L]; for (int i = 0; i < z.Length; i++) z[i] = (float)(Half)(float)(rng.NextDouble() * 2 - 1);
                using var bX = Up(device, x); using var bZf = Up(device, z); using var bZh = Up(device, z.Select(v => (Half)v).ToArray());
                using var bDf = device.Allocate((long)n * G * L * 4); using var bDh = device.Allocate((long)n * G * L * 2);
                using var bUf = device.Allocate((long)n * H * 4); using var bUh = device.Allocate((long)n * H * 2);
                Run(device, "D", n,
                [
                    new("oldF32", (cmd, c) => f32.Record(cmd, dW[c], bX, bDf, G * L, H, n)),
                    new("gemm", (cmd, c) => k.RecordGemmD(cmd, dW[c], bX, bDh, G * L, H, n)),
                    new("wideNR4", (cmd, c) => wideD4.Record(cmd, dW[c], bX, bDh, G * L, H, n)),
                    new("wideNR2", (cmd, c) => wideD2.Record(cmd, dW[c], bX, bDh, G * L, H, n)),
                ]);
                Run(device, "U", n,
                [
                    new("oldF32", (cmd, c) => f32.Record(cmd, uW[c], bZf, bUf, H, L, n)),
                    new("gemm", (cmd, c) => k.RecordGemmU(cmd, uW[c], bZh, bUh, H, L, n)),
                    new("wideNR4", (cmd, c) => wideU4.Record(cmd, uW[c], bZh, bUh, H, L, n)),
                    new("wideNR2", (cmd, c) => wideU2.Record(cmd, uW[c], bZh, bUh, H, L, n)),
                ]);
            }
        }
        finally { foreach (var b in dW) b.Dispose(); foreach (var b in uW) b.Dispose(); }
    }
}
