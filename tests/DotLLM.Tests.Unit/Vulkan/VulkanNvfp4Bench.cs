using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// NVFP4 kernel timings at NVIDIA Bolt's Mamba projection shapes (<c>ssm_in</c> 18560 x 4096, <c>ssm_out</c> 4096 x 8192), for
/// #1064 (prefill GEMM) and #1065 (decode GEMV). Opt-in with <c>DOTLLM_NVFP4_BENCH=1</c>.
/// </summary>
/// <remarks>
/// Batched dispatches behind one fence (amortises submit latency), warm-up passes, median of several passes; the methodology of
/// <see cref="VulkanQ3KQ6KMmvqVsGemvBench"/>. Weight bytes are random: timing is data-independent.
/// </remarks>
[Trait("Category", "Benchmark")]
[Collection("VulkanKernels")]
public sealed class VulkanNvfp4Bench
{
    private const int WarmupPasses = 2;
    private const int Passes = 7;

    private static readonly (string Tag, int M, int K)[] BoltShapes =
    [
        ("ssm_in 18560x4096", 18560, 4096),
        ("ssm_out 4096x8192", 4096, 8192),
    ];

    private readonly ITestOutputHelper _output;

    /// <summary>Creates the bench.</summary>
    /// <param name="output">xUnit output sink for the result table.</param>
    public VulkanNvfp4Bench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// #1064: pp512-shaped NVFP4 GEMM, native coopmat (decode in shared memory) against #1060's F16 path (whole-matrix dequant to
    /// F16 scratch + F16 coopmat GEMM + scale pass), and that path's F16 GEMM alone on a pre-dequantised scratch.
    /// </summary>
    [SkippableFact]
    public void Bench_PrefillGemm_NativeVsF16Dequant()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_NVFP4_BENCH") == "1", "DOTLLM_NVFP4_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulNvfp4GemmCoopmatKernel.IsSupportedOn(device, spvDir), "no wave64 cooperative-matrix device");
        const int n = 512;
        const float scale = 1.3e-4f;

        using var native = MatMulNvfp4GemmCoopmatKernel.Create(device, spvDir);
        using var f16Gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        long maxElements = BoltShapes.Max(s => (long)s.M * s.K);
        using var f16Path = IqF16PrefillMatmul.TryCreate(device, spvDir, f16Gemm, maxElements)
            ?? throw new InvalidOperationException("F16 prefill helper unavailable");
        using var f16Scratch = device.Allocate(maxElements * sizeof(ushort));

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  n = {n}");
        _output.WriteLine("| shape | native ms | F16 path ms | F16 GEMM only ms | native TFLOPS | F16 path / native | F16 scratch MB |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|---:|");

        var rng = new Random(1064);
        foreach (var (tag, m, k) in BoltShapes)
        {
            using var bufW = UploadRandomWeights(device, rng, m, k);
            using var bufX = device.Allocate((long)n * k * sizeof(float));
            using var bufY = device.Allocate((long)n * m * sizeof(float));
            device.Upload(RandomActivations(rng, n * k), bufX);

            double nativeUs = MeasureMedian(device, 8, cb => native.Record(cb, bufW, bufX, bufY, m, k, n, scale));
            double f16Us = MeasureMedian(device, 8, cb =>
            {
                if (!f16Path.TryRecord(cb, QuantizationType.NVFP4, bufW, bufX, bufY, m, k, n, scale))
                    throw new InvalidOperationException("the F16 path declined");
            });
            double gemmOnlyUs = MeasureMedian(device, 8, cb =>
            {
                KernelSupport.ComputeToComputeBarrier(cb);
                f16Gemm.Record(cb, f16Scratch, bufX, bufY, m, k, n);
            });

            double tflops = 2.0 * m * k * n / (nativeUs * 1e-6) / 1e12;
            _output.WriteLine($"| {tag} | {nativeUs / 1e3:F2} | {f16Us / 1e3:F2} | {gemmOnlyUs / 1e3:F2} | {tflops:F1} | " +
                              $"{f16Us / nativeUs:F2}x | {(long)m * k * sizeof(ushort) / 1e6:F0} |");
        }
    }

    internal static VulkanDevice.Buffer UploadRandomWeights(VulkanDevice device, Random rng, int m, int k)
    {
        byte[] w = new byte[(long)m * (k / QuantFormat.Nvfp4GroupSize) * QuantFormat.Nvfp4BlockBytes];
        rng.NextBytes(w);
        for (int b = 0; b < w.Length; b += QuantFormat.Nvfp4BlockBytes)
            for (int s = 0; s < 4; s++)
                w[b + s] = (byte)rng.Next(0x20, 0x70);   // finite, non-zero UE4M3 scales
        var buf = device.Allocate(w.Length);
        device.Upload(new ReadOnlySpan<byte>(w), buf);
        return buf;
    }

    internal static float[] RandomActivations(Random rng, int count)
    {
        float[] x = new float[count];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;
        return x;
    }

    /// <summary>Median per-dispatch microseconds of <paramref name="record"/>, recorded <paramref name="batch"/> times per fence.</summary>
    internal static double MeasureMedian(VulkanDevice device, int batch, Action<nint> record)
    {
        double Pass()
        {
            using var ctx = device.CreateSubmitContext();
            var sw = Stopwatch.StartNew();
            ctx.Begin();
            for (int i = 0; i < batch; i++)
            {
                record(ctx.CommandBuffer);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            }
            ctx.SubmitAndWait();
            return sw.Elapsed.TotalMicroseconds / batch;
        }

        for (int i = 0; i < WarmupPasses; i++) Pass();
        double[] us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = Pass();
        Array.Sort(us);
        return us[Passes / 2];
    }
}
