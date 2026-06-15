using System.Diagnostics;
using System.Text;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark: DP4a (integer dot product) Q8_0 GEMV — scalar vs
/// inline-x-quant vs shared-prequant — on decode-path (N=1) shapes. Enable with
/// <c>DOTLLM_VULKAN_DP4A_BENCH=1</c>. Emits a Markdown table to stdout.
/// </summary>
/// <remarks>
/// <para>
/// Methodology (fixed per advisor review): each side is timed with a
/// <b>batched fence</b> — N iterations recorded into one command buffer, a single
/// <c>vkQueueSubmit</c> + <c>vkQueueWaitIdle</c>, time divided by N. This removes
/// the per-Launch sync floor that pins small shapes at ~1.4 ms. A
/// <c>ComputeToComputeBarrier</c> after each iteration serializes them, so the
/// measurement is true per-matmul GPU time (including, for prequant, its internal
/// quantize→gemv barrier — an intrinsic per-matmul cost that does not amortize).
/// The three kernels are interleaved <b>within</b> each round and the reported
/// speedup is the <b>median of per-round ratios</b> (paired), so thermal/turbo
/// drift cancels instead of biasing independent per-side minima.
/// </para>
/// <para>
/// Q8_0 decode GEMV on Arc Xe-LPG is ALU/dequant-bound (not weight-read bound like
/// the discrete 3060), so DP4a wins on high-M / deep-K shapes once the activation
/// quantization is amortized. Small isolated matmuls retain the prequant pass's
/// extra-dispatch+barrier overhead; the real recovery there is sharing one
/// quantized activation across same-input projections (Q/K/V, gate/up) in the
/// forward pass — which this per-matmul bench structurally cannot show.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ8_0Dp4aMicroBench
{
    private const int Iterations = 100;
    private const int Rounds = 7;
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private readonly ITestOutputHelper _output;

    public VulkanQ8_0Dp4aMicroBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Bench_Dp4aVsScalarQ8_0Gemv()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DP4A_BENCH") == "1",
            "DOTLLM_VULKAN_DP4A_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' does not support VK_KHR_shader_integer_dot_product.");

        var sb = new StringBuilder();
        sb.AppendLine("| Shape (M×K) | Scalar ms | Inline ms | Prequant ms | inline× | prequant× |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|");

        (int m, int k)[] shapes =
        {
            (576, 576), (1536, 576), (576, 1536),
            (49152, 576), (4096, 4096), (4096, 14336),
        };
        foreach (var (m, k) in shapes)
            sb.AppendLine(BenchShape(device, spvDir, m, k));

        _output.WriteLine("Device: " + device.DeviceName);
        _output.WriteLine($"Batched fence; iterations/submit: {Iterations}, rounds: {Rounds} (median of per-round ratios)");
        _output.WriteLine(string.Empty);
        _output.WriteLine(sb.ToString());
    }

    private string BenchShape(VulkanDevice device, string spvDir, int m, int k)
    {
        var rng = new Random(0x51 + m * 7 + k);
        int blocksPerRow = k / Q8_0GroupSize;
        int totalBytes = m * blocksPerRow * Q8_0BlockBytes;
        byte[] weightsQ8 = new byte[totalBytes];
        unsafe
        {
            float[] rowF = new float[k];
            fixed (byte* dst = weightsQ8)
            fixed (float* rp = rowF)
            {
                for (int row = 0; row < m; row++)
                {
                    for (int i = 0; i < k; i++) rowF[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.1);
                    MatMul.QuantizeF32ToQ8_0(rp, dst + (long)row * blocksPerRow * Q8_0BlockBytes, k);
                }
            }
        }
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        using var bufXq = device.Allocate(MatMulQ8_0Dp4aPqKernel.XqScratchBytes(k));
        using var bufDx = device.Allocate(MatMulQ8_0Dp4aPqKernel.DxScratchBytes(k));
        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);

        using var scalar = MatMulQ8_0Kernel.Create(device, spvDir);
        using var inlineK = MatMulQ8_0Dp4aKernel.Create(device, spvDir);
        using var pq = MatMulQ8_0Dp4aPqKernel.Create(device, spvDir);

        Action<nint> recScalar = cb => { scalar.Record(cb, bufW, bufX, bufY, m, k); KernelSupport.ComputeToComputeBarrier(cb); };
        Action<nint> recInline = cb => { inlineK.Record(cb, bufW, bufX, bufY, m, k); KernelSupport.ComputeToComputeBarrier(cb); };
        Action<nint> recPq = cb => { pq.Record(cb, bufW, bufX, bufXq, bufDx, bufY, m, k); KernelSupport.ComputeToComputeBarrier(cb); };

        // Warm up each side once (pipeline ISA compile + first-submit cost).
        TimeBatched(device, recScalar, Iterations);
        TimeBatched(device, recInline, Iterations);
        TimeBatched(device, recPq, Iterations);

        var inlineRatios = new double[Rounds];
        var pqRatios = new double[Rounds];
        double scalarMed = 0, inlineMed = 0, pqMed = 0;
        var scalarMs = new double[Rounds];
        var inlineMs = new double[Rounds];
        var pqMs = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            scalarMs[r] = TimeBatched(device, recScalar, Iterations);
            inlineMs[r] = TimeBatched(device, recInline, Iterations);
            pqMs[r] = TimeBatched(device, recPq, Iterations);
            inlineRatios[r] = scalarMs[r] / inlineMs[r];
            pqRatios[r] = scalarMs[r] / pqMs[r];
        }
        scalarMed = Median(scalarMs);
        inlineMed = Median(inlineMs);
        pqMed = Median(pqMs);

        return $"| {m}×{k} | {scalarMed:F4} | {inlineMed:F4} | {pqMed:F4} | {Median(inlineRatios):F2}x | {Median(pqRatios):F2}x |";
    }

    private static double TimeBatched(VulkanDevice device, Action<nint> recordOne, int n)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        for (int i = 0; i < n; i++) recordOne(ctx.CommandBuffer);
        var sw = Stopwatch.StartNew();
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / n;
    }

    private static double Median(double[] vals)
    {
        var s = (double[])vals.Clone();
        Array.Sort(s);
        int n = s.Length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }
}
