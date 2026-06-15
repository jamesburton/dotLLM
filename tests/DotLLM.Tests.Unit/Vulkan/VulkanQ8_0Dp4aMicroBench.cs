using System.Diagnostics;
using System.Text;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark: DP4a (integer dot product) Q8_0 GEMV vs the scalar
/// Q8_0 GEMV, decode-path (N=1) shapes. Enable with
/// <c>DOTLLM_VULKAN_DP4A_BENCH=1</c>. Emits a Markdown table to stdout.
/// </summary>
/// <remarks>
/// Method: build a pool of kernel instances per side (descriptor pool is
/// maxSets=1, so one Launch per instance), warm up, then time the scalar batch
/// and the DP4a batch in alternating rounds, taking the per-side <b>min</b>
/// ms/iter across rounds to suppress the laptop's ~2× turbo/thermal drift. Sync
/// submission (<c>vkQueueWaitIdle</c>) per Launch includes per-dispatch overhead.
///
/// <para>
/// Q8_0 decode GEMV is largely <b>memory-bound</b> (≈34 weight bytes read per
/// 32-element block, ≈1 MAC/byte), so DP4a — which accelerates the MACs, not the
/// weight reads — is not expected to give the headline compute-bound INT8 speedup
/// here; the inline per-row activation re-quantization adds work. This bench
/// measures the real effect on the decode path rather than assuming one. The
/// compute-bound win belongs to a batched/GEMM (prefill) DP4a path (future work).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ8_0Dp4aMicroBench
{
    private const int Iterations = 100;
    private const int WarmupIterations = 10;
    private const int Rounds = 5;
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
        sb.AppendLine("| Shape (M×K) | Scalar ms/iter | DP4a inline ms/iter | DP4a+prequant ms/iter | inline× | prequant× |");
        sb.AppendLine("|---|---:|---:|---:|---:|---:|");

        (int m, int k)[] shapes =
        {
            (576, 576),     // SmolLM q/k/v projection
            (1536, 576),    // SmolLM gate/up
            (576, 1536),    // SmolLM down
            (49152, 576),   // SmolLM lm_head
            (4096, 4096),   // larger decode GEMV (deeper k)
            (4096, 14336),  // Llama-ish FFN down (deep k)
        };
        foreach (var (m, k) in shapes)
            sb.AppendLine(BenchShape(device, spvDir, m, k));

        _output.WriteLine("Device: " + device.DeviceName);
        _output.WriteLine($"Iterations/round: {Iterations}, rounds: {Rounds}, warmup: {WarmupIterations}");
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

        var scalar = new MatMulQ8_0Kernel[Iterations + WarmupIterations];
        var dp4a = new MatMulQ8_0Dp4aKernel[Iterations + WarmupIterations];
        var pq = new MatMulQ8_0Dp4aPqKernel[Iterations + WarmupIterations];
        for (int i = 0; i < scalar.Length; i++)
        {
            scalar[i] = MatMulQ8_0Kernel.Create(device, spvDir);
            dp4a[i] = MatMulQ8_0Dp4aKernel.Create(device, spvDir);
            pq[i] = MatMulQ8_0Dp4aPqKernel.Create(device, spvDir);
        }
        try
        {
            for (int w = 0; w < WarmupIterations; w++)
            {
                scalar[w].Launch(bufW, bufX, bufY, m, k);
                dp4a[w].Launch(bufW, bufX, bufY, m, k);
                pq[w].Launch(bufW, bufX, bufXq, bufDx, bufY, m, k);
            }

            double scalarMin = double.MaxValue, dp4aMin = double.MaxValue, pqMin = double.MaxValue;
            for (int r = 0; r < Rounds; r++)
            {
                scalarMin = Math.Min(scalarMin, TimeBatch(() =>
                {
                    for (int i = 0; i < Iterations; i++) scalar[WarmupIterations + i].Launch(bufW, bufX, bufY, m, k);
                }));
                dp4aMin = Math.Min(dp4aMin, TimeBatch(() =>
                {
                    for (int i = 0; i < Iterations; i++) dp4a[WarmupIterations + i].Launch(bufW, bufX, bufY, m, k);
                }));
                pqMin = Math.Min(pqMin, TimeBatch(() =>
                {
                    for (int i = 0; i < Iterations; i++) pq[WarmupIterations + i].Launch(bufW, bufX, bufXq, bufDx, bufY, m, k);
                }));
            }

            double scalarMs = scalarMin / Iterations;
            double dp4aMs = dp4aMin / Iterations;
            double pqMs = pqMin / Iterations;
            return $"| {m}×{k} | {scalarMs:F4} | {dp4aMs:F4} | {pqMs:F4} | {scalarMs / dp4aMs:F2}x | {scalarMs / pqMs:F2}x |";
        }
        finally
        {
            foreach (var s in scalar) s.Dispose();
            foreach (var d in dp4a) d.Dispose();
            foreach (var p in pq) p.Dispose();
        }
    }

    private static double TimeBatch(Action batch)
    {
        var sw = Stopwatch.StartNew();
        batch();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }
}
