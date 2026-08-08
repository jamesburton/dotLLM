using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #240 audit — real, same-session, order-reversed A/B for the three coopmat GEMM-family
/// pipelines whose <c>Coopmat32</c> (32-thread workgroup, pinned <c>requiredSubgroupSize=32</c>)
/// variant was written blind in PR (bc7f6393) without a Vulkan SDK to compile it or hardware to
/// measure it. The <c>.spv</c> files landed later (commit e366fe1c); this is the first session
/// with both the compiled shaders AND a device that can pin subgroup size (this machine's RTX
/// 3060 — see <see cref="VulkanMmqSubgroupSizeBench"/> for the standalone confirmation that
/// <c>VK_EXT_subgroup_size_control</c> is supported here, min=max=32).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this measurement means something different on Ampere than on gfx1151.</b> The #236
/// PQ2_0 result (1.29-1.79x from the pin alone) came from correcting an actual wave-width
/// mismatch: RDNA3.5's WMMA is wave32 but the driver defaults compute dispatch to wave64, so an
/// unpinned 64-thread coopmat workgroup silently ran as two redundant wave32 passes over the same
/// tile. NVIDIA Ampere's subgroup size is fixed at 32 with no wave64 mode
/// (<c>VkPhysicalDeviceSubgroupSizeControlProperties.minSubgroupSize == maxSubgroupSize == 32</c>
/// here), so there is no such mismatch to correct — the 64-thread <c>Coopmat64</c> kernel already
/// runs as two wave32 subgroups doing the SAME 16x16x16 tile redundantly (correct, per the
/// shader's own coopmat scope, just 2x wasted ALU on this device specifically), while
/// <c>Coopmat32</c>'s 32-thread workgroup does the tile exactly once. So on THIS hardware the A/B
/// below isolates a workgroup-occupancy/tiling effect (fewer redundant lanes, smaller WG = more
/// concurrent workgroups per SM), not a wave-width correctness fix — the pin itself
/// (<c>requiredSubgroupSize=32</c>) is a documented no-op given the device already always runs
/// subgroup=32. That is itself the useful cross-hardware data point the issue asks for: confirm
/// the mechanism doesn't regress or misbehave where it isn't the bottleneck, and report whether
/// the accompanying 32-thread-workgroup retiling helps independently of the pin.
/// </para>
/// <para>Methodology mirrors <see cref="VulkanPQ2_0GemmBench"/> exactly (issues #233/#236):
/// batched dispatches behind one fence per pass, interleaved order-reversed passes, median of
/// per-pass ratios. Enable with <c>DOTLLM_COOPMAT32_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanCoopmat32SubgroupPinBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int Batch = 8;

    private readonly ITestOutputHelper _output;
    public VulkanCoopmat32SubgroupPinBench(ITestOutputHelper output) => _output = output;

    private static readonly (string Tag, int M, int K)[] GemmShapes =
    [
        ("attn_qkvo   (576x576)",     576,  576),
        ("ffn_gate/up (1536x576)",   1536,  576),
        ("llama_proj  (4096x4096)",  4096, 4096),
    ];

    [SkippableFact]
    public void Bench_Q8_0GemmCoopmat32()
    {
        Skip.IfNot(Enabled, EnableMsg);
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        ReportDevice(device);
        Skip.IfNot(Q8_0GemmCoopmatVariant.Coopmat32.IsSupportedOn(device, spvDir),
            "matmul_q8_0_gemm_coopmat32.spv unavailable or device cannot pin requiredSubgroupSize=32.");

        using var refKernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, Q8_0GemmCoopmatVariant.Coopmat64);
        using var challKernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, Q8_0GemmCoopmatVariant.Coopmat32);

        const int tokens = 64;
        const int groupSize = 32, blockBytes = 34;
        _output.WriteLine("");
        _output.WriteLine("### matmul_q8_0_gemm: coopmat64 (unpinned) -> coopmat32 (pinned wave32)");
        _output.WriteLine("| shape | coopmat64 us | coopmat32 us | speedup | c64 GFLOP/s | c32 GFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x2A_53);
        foreach (var (tag, m, k) in GemmShapes)
        {
            int blocksPerRow = k / groupSize;
            long wBytes = (long)m * blocksPerRow * blockBytes;
            using var bufW = device.Allocate((wBytes + 3) & ~3L);
            using var bufB = device.Allocate((long)tokens * k * sizeof(float));
            using var bufC = device.Allocate((long)tokens * m * sizeof(float));

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            float[] b = RandomFloats(rng, tokens * k, 1.0f);
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(b, bufB);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                (cb, batch) => { for (int i = 0; i < batch; i++) refKernel.Record(cb, bufW, bufB, bufC, m, k, tokens); },
                (cb, batch) => { for (int i = 0; i < batch; i++) challKernel.Record(cb, bufW, bufB, bufC, m, k, tokens); },
                device);

            double flop = 2.0 * m * (double)k * tokens;
            _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x | "
                + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
        }
    }

    [SkippableFact]
    public void Bench_F16GemmCoopmat32()
    {
        Skip.IfNot(Enabled, EnableMsg);
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        ReportDevice(device);
        Skip.IfNot(F16GemmCoopmatVariant.Coopmat32.IsSupportedOn(device, spvDir),
            "matmul_f16_gemm_coopmat32.spv unavailable or device cannot pin requiredSubgroupSize=32.");

        using var refKernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir, F16GemmCoopmatVariant.Coopmat64);
        using var challKernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir, F16GemmCoopmatVariant.Coopmat32);

        const int tokens = 64;
        _output.WriteLine("");
        _output.WriteLine("### matmul_f16_gemm: coopmat64 (unpinned) -> coopmat32 (pinned wave32)");
        _output.WriteLine("| shape | coopmat64 us | coopmat32 us | speedup | c64 GFLOP/s | c32 GFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x2A_53);
        foreach (var (tag, m, k) in GemmShapes)
        {
            long wBytes = (long)m * k * 2;
            using var bufW = device.Allocate(wBytes);
            using var bufB = device.Allocate((long)tokens * k * sizeof(float));
            using var bufC = device.Allocate((long)tokens * m * sizeof(float));

            float[] wF32 = RandomFloats(rng, m * k, 0.1f);
            byte[] wF16 = F16Bf16Fixture.QuantizeRowsF16(wF32, m, k);
            float[] b = RandomFloats(rng, tokens * k, 1.0f);
            device.Upload(wF16, bufW);
            device.Upload(b, bufB);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                (cb, batch) => { for (int i = 0; i < batch; i++) refKernel.Record(cb, bufW, bufB, bufC, m, k, tokens); },
                (cb, batch) => { for (int i = 0; i < batch; i++) challKernel.Record(cb, bufW, bufB, bufC, m, k, tokens); },
                device);

            double flop = 2.0 * m * (double)k * tokens;
            _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x | "
                + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
        }
    }

    [SkippableFact]
    public void Bench_MoeGroupedCoopmat32()
    {
        Skip.IfNot(Enabled, EnableMsg);
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        ReportDevice(device);
        Skip.IfNot(MoeGroupedCoopmatVariant.Coopmat32.IsSupportedOn(device, spvDir),
            "moe_grouped_matmul_f16_coopmat32.spv unavailable or device cannot pin requiredSubgroupSize=32.");

        using var refKernel = MoeGroupedMatmulF16CoopmatKernel.Create(device, spvDir, MoeGroupedCoopmatVariant.Coopmat64);
        using var challKernel = MoeGroupedMatmulF16CoopmatKernel.Create(device, spvDir, MoeGroupedCoopmatVariant.Coopmat32);

        const int numExperts = 8;
        const int rowsPerExpert = 8;
        const int rows = numExperts * rowsPerExpert; // 64 tokens, evenly routed — occupancy-favorable case
        var offsets = new uint[numExperts + 1];
        for (int i = 0; i <= numExperts; i++) offsets[i] = (uint)(i * rowsPerExpert);

        _output.WriteLine("");
        _output.WriteLine("### moe_grouped_matmul_f16: coopmat64 (unpinned) -> coopmat32 (pinned wave32)");
        _output.WriteLine($"### numExperts={numExperts}, rows={rows} (evenly routed)");
        _output.WriteLine("| shape (outDim x hidden) | coopmat64 us | coopmat32 us | speedup | c64 GFLOP/s | c32 GFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x2A_53);
        foreach (var (tag, m, k) in GemmShapes)
        {
            long wBytes = (long)numExperts * m * k * 2;
            using var bufW = device.Allocate(wBytes);
            using var bufX = device.Allocate((long)rows * k * sizeof(float));
            using var bufOff = device.Allocate((long)offsets.Length * sizeof(uint));
            using var bufY = device.Allocate((long)rows * m * sizeof(float));

            float[] wF32 = RandomFloats(rng, numExperts * m * k, 0.1f);
            byte[] wF16 = F16Bf16Fixture.QuantizeRowsF16(wF32, numExperts * m, k);
            float[] x = RandomFloats(rng, rows * k, 1.0f);
            device.Upload(wF16, bufW);
            device.Upload(x, bufX);
            device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                (cb, batch) => { for (int i = 0; i < batch; i++) refKernel.Record(cb, bufW, bufX, bufOff, bufY, m, k, rows, numExperts); },
                (cb, batch) => { for (int i = 0; i < batch; i++) challKernel.Record(cb, bufW, bufX, bufOff, bufY, m, k, rows, numExperts); },
                device);

            double flop = 2.0 * m * (double)k * rows;
            _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x | "
                + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Shared A/B harness — same schedule as VulkanPQ2_0GemmBench.
    // ─────────────────────────────────────────────────────────────

    private static (double reference, double challenger, double ratio) MeasurePaired(
        Action<nint, int> recordReference, Action<nint, int> recordChallenger, VulkanDevice device)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, recordReference, Batch);
            RunPass(device, recordChallenger, Batch);
        }

        var refUs = new double[Passes];
        var challUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tr, tc;
            if ((p & 1) == 0)
            {
                tr = RunPass(device, recordReference, Batch);
                tc = RunPass(device, recordChallenger, Batch);
            }
            else
            {
                tc = RunPass(device, recordChallenger, Batch);
                tr = RunPass(device, recordReference, Batch);
            }
            refUs[p] = tr;
            challUs[p] = tc;
            ratios[p] = tc > 0 ? tr / tc : 0;
        }

        Array.Sort(refUs); Array.Sort(challUs); Array.Sort(ratios);
        return (refUs[Passes / 2], challUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double RunPass(VulkanDevice device, Action<nint, int> record, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        record(ctx.CommandBuffer, batch);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private void ReportDevice(VulkanDevice device)
    {
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  " +
            $"subgroup-size-control: supported={device.HasSubgroupSizeControl} min={device.MinSubgroupSize} max={device.MaxSubgroupSize}");
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_COOPMAT32_BENCH"), "1", StringComparison.Ordinal);

    private const string EnableMsg = "DOTLLM_COOPMAT32_BENCH=1 to enable this benchmark.";
}
