using System.Diagnostics;
using System.Text;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark that times N dispatches of <c>rmsnorm_f32</c> and
/// <c>attention_f32</c> on both the shared-memory and subgroup-arithmetic
/// paths. Enable with <c>DOTLLM_VULKAN_SUBGROUP_BENCH=1</c>. Emits a small
/// Markdown table to stdout (captured by xUnit and copied into the perf-run
/// directory by the operator).
/// </summary>
/// <remarks>
/// This is deliberately not a BenchmarkDotNet harness — the task mandate
/// said "no BDN needed" and this cross-path comparison is a single-sample
/// walltime measurement, not a statistical study. Each sample runs
/// <see cref="Iterations"/> dispatches and divides by the count.
///
/// Sync submission (<c>vkQueueWaitIdle</c>) dominates at small shapes; the
/// numbers here are worst-case (per-dispatch overhead included). Real
/// end-to-end inference batches dispatches behind a single fence and will
/// see bigger wins from the subgroup path.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSubgroupMicroBench
{
    // Note: the wave-2 descriptor pool has maxSets=1 (documented deferred work
    // in docs/VULKAN.md §Deferred Work), so each kernel instance supports
    // exactly one Launch. To measure per-Launch time cleanly we build a batch
    // of kernel instances up front and time a single Launch on each.
    private const int Iterations = 200;
    private const int WarmupIterations = 10;

    private readonly ITestOutputHelper _output;

    public VulkanSubgroupMicroBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Bench_SubgroupVsSharedReduce()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_SUBGROUP_BENCH") == "1",
            "DOTLLM_VULKAN_SUBGROUP_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasSubgroupArithmetic,
            "Device does not advertise subgroup arithmetic; nothing to compare.");

        var sb = new StringBuilder();
        sb.AppendLine("| Kernel | Shape | Shared-mem ms/iter | Subgroup ms/iter | Speedup |");
        sb.AppendLine("|---|---|---:|---:|---:|");

        sb.AppendLine(BenchRmsNorm(device, spvDir, rowCount: 4, n: 1536));

        sb.AppendLine(BenchAttention(
            device, spvDir,
            seqQ: 1, seqKv: 128, numHeads: 9, numKvHeads: 3, headDim: 64, posOffset: 127,
            shape: "seqQ=1,seqKv=128,nh=9,nkv=3,hd=64"));

        sb.AppendLine(BenchAttention(
            device, spvDir,
            seqQ: 1, seqKv: 512, numHeads: 4, numKvHeads: 2, headDim: 128, posOffset: 511,
            shape: "seqQ=1,seqKv=512,nh=4,nkv=2,hd=128 (2-tile)"));

        _output.WriteLine("Device: " + device.DeviceName);
        _output.WriteLine($"SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"Iterations per sample: {Iterations} (warmup {WarmupIterations})");
        _output.WriteLine(string.Empty);
        _output.WriteLine(sb.ToString());
    }

    private static string BenchRmsNorm(VulkanDevice device, string spvDir, int rowCount, int n)
    {
        var rng = new Random(0x42);
        float[] input = new float[rowCount * n];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) weight[i] = 1.0f;

        double sharedMs = TimeRmsNorm(device, spvDir, input, weight, rowCount, n, forceShared: true);
        double subMs = TimeRmsNorm(device, spvDir, input, weight, rowCount, n, forceShared: false);
        return $"| rmsnorm_f32 | rowCount={rowCount},n={n} | {sharedMs:F3} | {subMs:F3} | {sharedMs / subMs:F2}x |";
    }

    private static double TimeRmsNorm(
        VulkanDevice device, string spvDir, float[] input, float[] weight,
        int rowCount, int n, bool forceShared)
    {
        string? original = Environment.GetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, forceShared ? "1" : null);

            using var bufIn  = device.Allocate((long)rowCount * n * sizeof(float));
            using var bufW   = device.Allocate((long)n * sizeof(float));
            using var bufOut = device.Allocate((long)rowCount * n * sizeof(float));
            device.Upload(input, bufIn);
            device.Upload(weight, bufW);

            // Build the kernel pool first (amortizes SPV-to-ISA compilation on
            // first use — the driver caches compiled pipelines, so the 2nd+
            // Create is much cheaper than the 1st).
            var kernels = new RmsNormF32Kernel[Iterations + WarmupIterations];
            for (int i = 0; i < kernels.Length; i++)
                kernels[i] = RmsNormF32Kernel.Create(device, spvDir);

            try
            {
                for (int w = 0; w < WarmupIterations; w++)
                    kernels[w].Launch(bufIn, bufW, bufOut, rowCount, n, 1e-5f);

                var sw = Stopwatch.StartNew();
                for (int i = 0; i < Iterations; i++)
                    kernels[WarmupIterations + i].Launch(bufIn, bufW, bufOut, rowCount, n, 1e-5f);
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / Iterations;
            }
            finally
            {
                foreach (var k in kernels) k.Dispose();
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, original);
        }
    }

    private static string BenchAttention(
        VulkanDevice device, string spvDir,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int posOffset, string shape)
    {
        var rng = new Random(0x42);
        float[] qh = new float[seqQ * numHeads * headDim];
        float[] kh = new float[seqKv * numKvHeads * headDim];
        float[] vh = new float[seqKv * numKvHeads * headDim];
        for (int i = 0; i < qh.Length; i++) qh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < kh.Length; i++) kh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < vh.Length; i++) vh[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        double sharedMs = TimeAttention(device, spvDir, qh, kh, vh,
            seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset, forceShared: true);
        double subMs = TimeAttention(device, spvDir, qh, kh, vh,
            seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset, forceShared: false);
        return $"| attention_f32 | {shape} | {sharedMs:F3} | {subMs:F3} | {sharedMs / subMs:F2}x |";
    }

    private static double TimeAttention(
        VulkanDevice device, string spvDir,
        float[] qh, float[] kh, float[] vh,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int posOffset,
        bool forceShared)
    {
        string? original = Environment.GetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, forceShared ? "1" : null);

            using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
            using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
            using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
            using var bufOut = device.Allocate((long)seqQ * numHeads * headDim * sizeof(float));
            device.Upload(qh, bufQ);
            device.Upload(kh, bufK);
            device.Upload(vh, bufV);

            var kernels = new AttentionF32Kernel[Iterations + WarmupIterations];
            for (int i = 0; i < kernels.Length; i++)
                kernels[i] = AttentionF32Kernel.Create(device, spvDir);

            try
            {
                for (int w = 0; w < WarmupIterations; w++)
                    kernels[w].Launch(bufQ, bufK, bufV, bufOut,
                        seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset);

                var sw = Stopwatch.StartNew();
                for (int i = 0; i < Iterations; i++)
                    kernels[WarmupIterations + i].Launch(bufQ, bufK, bufV, bufOut,
                        seqQ, seqKv, numHeads, numKvHeads, headDim, posOffset);
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / Iterations;
            }
            finally
            {
                foreach (var k in kernels) k.Dispose();
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, original);
        }
    }
}
