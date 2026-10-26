using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// A/B bench: int8 dp4a MMVQ decode GEMV vs the fp16/F32 dequant-then-GEMV
/// fallback, for Q3_K and Q6_K weights, at realistic decode shapes.
/// </summary>
/// <remarks>
/// <para>
/// Context (issue investigation, no source changed by this file): llama.cpp's
/// Vulkan backend disables MMVQ entirely for <c>GGML_TYPE_Q3_K</c> and
/// <c>GGML_TYPE_Q6_K</c> — see <c>ggml_vk_should_use_mmvq</c> in
/// <c>ggml/src/ggml-vulkan/ggml-vulkan.cpp</c> (~line 8064): "General
/// performance issue with q3_k and q6_k due to 2-byte alignment" — both
/// formats' 110-/210-byte super-blocks are not 4-byte aligned per row in
/// general, which the comment implies costs the int8-unpack path more than it
/// saves vs a straightforward fp16 dequant GEMV. dotLLM's
/// <c>HasMmvqDecodeKernel</c> / <c>RecordMatmul</c>
/// (<c>src/DotLLM.Vulkan/VulkanTransformerModel.cs</c>) has no such gating —
/// MMVQ is used unconditionally for every quant type whenever the kernel is
/// loaded and the shape is block-aligned, including Q3_K and Q6_K. This bench
/// times both kernels back-to-back on real hardware so a coordinator can
/// decide whether dotLLM should add equivalent gating for gfx1151.
/// </para>
/// <para>
/// Methodology mirrors <see cref="VulkanI2SGemvBench"/>: batched dispatches
/// behind one fence (amortizes submit/fence latency so the region is
/// compute-bound) and interleaved paired A/B per pass (cancels iGPU
/// clock/thermal drift), reporting the median ratio over several passes.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ3KQ6KMmvqVsGemvBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;

    private readonly ITestOutputHelper _output;
    public VulkanQ3KQ6KMmvqVsGemvBench(ITestOutputHelper output) => _output = output;

    // Realistic decode shapes: M in {4096, 8192}, K=4096 (7-8B-class ffn_down /
    // attn_v row widths where Q3_K/Q6_K commonly live in K-quant mixes).
    private static readonly (string Tag, int M, int K)[] Shapes =
    [
        ("M=4096 K=4096", 4096, 4096),
        ("M=8192 K=4096", 8192, 4096),
    ];

    [SkippableFact]
    public void Bench_Q3K_MmvqVsGemv()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_Q3K_Q6K_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_Q3K_Q6K_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quantize = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ3KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q3_k_mmvq.spv missing or unsupported.");
        using var gemv = MatMulQ3KGemvF32Kernel.Create(device, spvDir);

        RunQ3K(device, quantize, mmvq, gemv);
    }

    [SkippableFact]
    public void Bench_Q6K_MmvqVsGemv()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_Q3K_Q6K_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_Q3K_Q6K_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quantize = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ6KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q6_k_mmvq.spv missing or unsupported.");
        using var gemv = MatMulQ6KGemvF32Kernel.Create(device, spvDir);

        RunQ6K(device, quantize, mmvq, gemv);
    }

    // ─────────────────────────────────────────────────────────────
    // Q3_K
    // ─────────────────────────────────────────────────────────────

    private void RunQ3K(
        VulkanDevice device, QuantizeQ8_1Kernel quantize,
        MatMulQ3KMmvqKernel mmvq, MatMulQ3KGemvF32Kernel gemv)
    {
        const int groupSize = MatMulQ3KMmvqKernel.Q3KGroupSize;
        const int blockBytes = MatMulQ3KMmvqKernel.Q3KBlockBytes;

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine("Q3_K decode GEMV: MMVQ (int8 dp4a) vs GEMV (F32 dequant)");
        _output.WriteLine("| shape | mmvq µs | gemv µs | mmvq/gemv | mmvq GB/s | gemv GB/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x33_4B);
        foreach (var (tag, m, k) in Shapes)
        {
            int blocksPerRow = k / groupSize;
            long rowBytes = (long)blocksPerRow * blockBytes;
            long wBytes = m * rowBytes;

            using var bufW = device.Allocate(wBytes);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
            using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
            using var bufY = device.Allocate((long)m * sizeof(float));

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);                  // random packed codes; timing is data-independent
            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(x, bufX);

            // One-time quantize of x into the shared Q8_1 activation scratch —
            // the MMVQ pass reuses it every dispatch (matches the production
            // decode path, which quantizes once per token per shared group).
            QuantizeOnce(device, quantize, bufX, bufXq, bufXds, k);

            double mmvqUs = MeasureMedian(device, batch =>
            {
                using var ctx = device.CreateSubmitContext();
                var sw = Stopwatch.StartNew();
                ctx.Begin();
                for (int i = 0; i < batch; i++)
                    mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMicroseconds / batch;
            });

            double gemvUs = MeasureMedian(device, batch =>
            {
                using var ctx = device.CreateSubmitContext();
                var sw = Stopwatch.StartNew();
                ctx.Begin();
                for (int i = 0; i < batch; i++)
                    gemv.Record(ctx.CommandBuffer, bufW, bufX, bufY, m, k);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMicroseconds / batch;
            });

            double mmvqGbps = wBytes / (mmvqUs * 1e-6) / 1e9;
            double gemvGbps = wBytes / (gemvUs * 1e-6) / 1e9;
            _output.WriteLine(
                $"| {tag} | {mmvqUs:F2} | {gemvUs:F2} | {(gemvUs > 0 ? mmvqUs / gemvUs : 0):F2}x | {mmvqGbps:F1} | {gemvGbps:F1} |");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Q6_K
    // ─────────────────────────────────────────────────────────────

    private void RunQ6K(
        VulkanDevice device, QuantizeQ8_1Kernel quantize,
        MatMulQ6KMmvqKernel mmvq, MatMulQ6KGemvF32Kernel gemv)
    {
        const int groupSize = MatMulQ6KMmvqKernel.Q6KGroupSize;
        const int blockBytes = MatMulQ6KMmvqKernel.Q6KBlockBytes;

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine("Q6_K decode GEMV: MMVQ (int8 dp4a) vs GEMV (F32 dequant)");
        _output.WriteLine("| shape | mmvq µs | gemv µs | mmvq/gemv | mmvq GB/s | gemv GB/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x66_4B);
        foreach (var (tag, m, k) in Shapes)
        {
            int blocksPerRow = k / groupSize;
            long rowBytes = (long)blocksPerRow * blockBytes;
            long wBytes = m * rowBytes;

            using var bufW = device.Allocate(wBytes);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
            using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
            using var bufY = device.Allocate((long)m * sizeof(float));

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(x, bufX);

            QuantizeOnce(device, quantize, bufX, bufXq, bufXds, k);

            double mmvqUs = MeasureMedian(device, batch =>
            {
                using var ctx = device.CreateSubmitContext();
                var sw = Stopwatch.StartNew();
                ctx.Begin();
                for (int i = 0; i < batch; i++)
                    mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMicroseconds / batch;
            });

            double gemvUs = MeasureMedian(device, batch =>
            {
                using var ctx = device.CreateSubmitContext();
                var sw = Stopwatch.StartNew();
                ctx.Begin();
                for (int i = 0; i < batch; i++)
                    gemv.Record(ctx.CommandBuffer, bufW, bufX, bufY, m, k);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMicroseconds / batch;
            });

            double mmvqGbps = wBytes / (mmvqUs * 1e-6) / 1e9;
            double gemvGbps = wBytes / (gemvUs * 1e-6) / 1e9;
            _output.WriteLine(
                $"| {tag} | {mmvqUs:F2} | {gemvUs:F2} | {(gemvUs > 0 ? mmvqUs / gemvUs : 0):F2}x | {mmvqGbps:F1} | {gemvGbps:F1} |");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static void QuantizeOnce(
        VulkanDevice device, QuantizeQ8_1Kernel quantize,
        VulkanDevice.Buffer x, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int k)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        quantize.Record(ctx.CommandBuffer, x, xq, xds, k);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Runs <paramref name="pass"/> (which itself submits <c>batch</c>
    /// back-to-back dispatches behind one fence) <see cref="WarmupPasses"/> +
    /// <see cref="Passes"/> times and returns the median per-dispatch µs.
    /// </summary>
    private static double MeasureMedian(VulkanDevice device, Func<int, double> pass)
    {
        int batch = EnvInt("DOTLLM_Q3K_Q6K_BENCH_BATCH", 32);
        for (int i = 0; i < WarmupPasses; i++) pass(batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = pass(batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
}
