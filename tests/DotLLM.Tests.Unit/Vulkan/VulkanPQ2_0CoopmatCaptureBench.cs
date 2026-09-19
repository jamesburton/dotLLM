using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in capture target for hardware profiling of the shipping PQ2_0 coopmat GEMM (issue #440).
/// </summary>
/// <remarks>
/// <para>
/// <b>This measures nothing itself.</b> It exists so an external profiler — the Radeon Developer
/// Panel driving an RGP SQTT capture — has a process whose compute dispatch stream is
/// <i>exclusively</i> <c>matmul_pq2_0_f32_gemm_coopmat32</c> at one shape, so that
/// <c>--rgp-auto-capture dispatch:START:COUNT</c> maps to a known kernel without ambiguity.
/// <see cref="VulkanPQ2_0GemmBench"/> cannot serve that purpose: it interleaves three variants
/// and five shapes, so a dispatch index there does not identify a kernel.
/// </para>
/// <para>
/// <b>Shape.</b> <c>lm_head</c> (M=248320, K=5120), 337 MB packed — the only Bonsai 2 projection
/// too large for gfx1151's 32 MB MALL, and therefore the only one whose behaviour is not
/// flattered by cache residency across repeats (see <see cref="VulkanPQ2_0GemmBench"/> remarks).
/// </para>
/// <para>
/// <b>Schedule.</b> One dispatch per submit-and-wait, so dispatch index == submit index and the
/// profiler's dispatch counter is trivially predictable: <c>Warmups</c> untraced dispatches then
/// <c>Traced</c> more. Start the capture range inside the traced region.
/// </para>
/// <para>
/// Enable with <c>DOTLLM_PQ2_0_CAPTURE=1</c>. Tokens default to 32
/// (<c>DOTLLM_PQ2_0_CAPTURE_TOKENS</c>); SQTT volume scales with the dispatch's work, so keep it
/// small. <c>DOTLLM_PQ2_0_CAPTURE_HOLD_MS</c> (default 0) sleeps before the first dispatch, giving
/// an externally launched profiler time to attach.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanPQ2_0CoopmatCaptureBench
{
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the capture target with the xUnit output sink.</summary>
    /// <param name="output">Sink for the dispatch log.</param>
    public VulkanPQ2_0CoopmatCaptureBench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Dispatches the shipping coopmat32 PQ2_0 GEMM on the <c>lm_head</c> shape, one dispatch per
    /// submit, so an attached RGP capture can target a single known dispatch.
    /// </summary>
    [SkippableFact]
    public void Capture_PQ2_0Coopmat32_LmHead()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_CAPTURE"), "1", StringComparison.Ordinal),
            "DOTLLM_PQ2_0_CAPTURE=1 to enable this capture target.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int tokens = EnvInt("DOTLLM_PQ2_0_CAPTURE_TOKENS", 32);
        int warmups = EnvInt("DOTLLM_PQ2_0_CAPTURE_WARMUPS", 4);
        int traced = EnvInt("DOTLLM_PQ2_0_CAPTURE_DISPATCHES", 16);
        int holdMs = EnvInt("DOTLLM_PQ2_0_CAPTURE_HOLD_MS", 0);
        int m = EnvInt("DOTLLM_PQ2_0_CAPTURE_M", 248320);
        int k = EnvInt("DOTLLM_PQ2_0_CAPTURE_K", 5120);

        using var device = VulkanDevice.Create();
        Skip.IfNot(PQ2_0GemmVariant.Coopmat32.IsSupportedOn(device), "coopmat32 variant unsupported on this device.");

        var variant = PQ2_0GemmVariant.Coopmat32;
        _output.WriteLine($"pid={Environment.ProcessId}  process={Process.GetCurrentProcess().ProcessName}");
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  Coopmat: {device.HasCooperativeMatrix}");
        _output.WriteLine($"variant={variant.SpvFileName}  (SelectFor => {PQ2_0GemmVariant.SelectFor(device).SpvFileName})");
        _output.WriteLine($"shape M={m} K={k} tokens={tokens}  warmups={warmups} traced={traced}  holdMs={holdMs}");

        using var kernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);

        long rowBytes = (long)(k / GroupSize) * GroupBytes;
        long wBytes = m * rowBytes;
        using var bufW = device.Allocate((wBytes + 3) & ~3L);
        using var bufB = device.Allocate((long)tokens * k * sizeof(float));
        using var bufC = device.Allocate((long)tokens * m * sizeof(float));

        var rng = new Random(0x2A_53);
        byte[] w = new byte[wBytes];
        rng.NextBytes(w);                       // random packed codes; timing is data-independent
        float[] b = new float[(long)tokens * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b, bufB);
        _output.WriteLine($"weights {wBytes / (1024.0 * 1024.0):F1} MiB packed uploaded");

        if (holdMs > 0) Thread.Sleep(holdMs);

        double flop = 2.0 * m * (double)k * tokens;
        int index = 0;
        for (int i = 0; i < warmups; i++, index++)
            Dispatch(device, kernel, bufW, bufB, bufC, m, k, tokens);

        _output.WriteLine($"--- traced region begins at dispatch index {index} ---");
        for (int i = 0; i < traced; i++, index++)
        {
            double us = Dispatch(device, kernel, bufW, bufB, bufC, m, k, tokens);
            _output.WriteLine($"dispatch {index}: {us:F1} us  {flop / (us * 1e-6) / 1e9:F1} GFLOP/s");
        }

        Assert.True(index == warmups + traced);
    }

    private static double Dispatch(
        VulkanDevice device, MatMulPQ2_0GemmF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        kernel.Record(ctx.CommandBuffer, w, b, c, m, k, n);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= 0 ? v : fallback;
}
