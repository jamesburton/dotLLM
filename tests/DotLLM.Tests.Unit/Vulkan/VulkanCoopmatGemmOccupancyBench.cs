using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Family-wide occupancy scan over every cooperative-matrix GEMM variant, using the
/// AMD proprietary driver's own post-compile allocation report
/// (<c>VK_AMD_shader_info</c>).
/// </summary>
/// <remarks>
/// <para>
/// Companion to <see cref="VulkanMmqShaderInfoBench"/>, which asked the same driver the
/// same question about one MMQ kernel. This one exists to test the hypothesis in
/// <c>.docs/COOPMAT_GEMM_DIAGNOSIS.md</c>: that every coopmat GEMM in the tree emits a
/// minimum-size 16x16 tile from a single subgroup, and is therefore occupancy- and
/// arithmetic-intensity-starved rather than merely "untuned".
/// </para>
/// <para>
/// The number that matters is <b>waves resident per SIMD</b>, which the driver does not
/// report directly; it is derived here from the VGPR and LDS allocations it does report.
/// If the shipping kernel sits at 1-3 waves, that also explains why the refuted 32x32
/// single-subgroup warptile (<c>matmul_i2_s_f32_gemm_coopmat_wt.comp</c>, issues
/// #384-#386) bought nothing: it doubled arithmetic intensity while doubling LDS at a
/// constant thread count, so any intensity gain was cancelled by lost occupancy.
/// </para>
/// <para>Reports ground truth; asserts nothing. Enable with <c>DOTLLM_COOPMAT_OCCUPANCY_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class VulkanCoopmatGemmOccupancyBench
{
    private readonly ITestOutputHelper _output;

    public VulkanCoopmatGemmOccupancyBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void ReportCoopmatGemmOccupancy()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_COOPMAT_OCCUPANCY_BENCH") != "1",
            "Set DOTLLM_COOPMAT_OCCUPANCY_BENCH=1 to run this diagnostic.");

        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasShaderInfoAmd, "VK_AMD_shader_info unavailable (needs the AMD proprietary driver).");

        _output.WriteLine($"Device: {device.DeviceName}   SubgroupSize={device.SubgroupSize}");
        _output.WriteLine("");

        var variants = new (string Name, PQ2_0GemmVariant V)[]
        {
            ("pq2_0 gemm coopmat32 (SHIPS on gfx1151)", PQ2_0GemmVariant.Coopmat32),
            ("pq2_0 gemm coopmat (64-thread sibling)",  PQ2_0GemmVariant.Coopmat),
            ("pq2_0 gemm register-blocked (oracle)",    PQ2_0GemmVariant.RegisterBlocked),
        };

        foreach (var (name, variant) in variants)
        {
            if (!variant.IsSupportedOn(device)) { _output.WriteLine($"{name}: not supported on this device"); continue; }

            using var k = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);
            var s = device.GetShaderStatisticsAmd(k.PipelineHandle);

            uint vgpr = s.resourceUsage.numUsedVgprs;
            uint avail = s.numAvailableVgprs == 0 ? 256u : s.numAvailableVgprs;
            ulong lds = s.resourceUsage.ldsUsageSizeInBytes;

            // Waves per SIMD from the VGPR file, workgroups per WGP from LDS.
            // Both are ceilings; the binding one is whichever is smaller.
            uint wavesByVgpr = vgpr == 0 ? 0 : avail / vgpr;
            double wgByLds = lds == 0 ? double.PositiveInfinity : 65536.0 / lds;

            _output.WriteLine(name);
            _output.WriteLine($"   workgroup           : {s.computeWorkGroupSizeX} threads");
            _output.WriteLine($"   VGPR used/available : {vgpr} / {avail}  (physical {s.numPhysicalVgprs})  -> {wavesByVgpr} waves/SIMD IF the budget is {avail}");
            _output.WriteLine($"   SGPR used/available : {s.resourceUsage.numUsedSgprs} / {s.numAvailableSgprs} (physical {s.numPhysicalSgprs})");
            _output.WriteLine($"   LDS used            : {lds} B -> {wgByLds:F1} workgroups/WGP by LDS (64 KB)");
            _output.WriteLine($"   scratch (spill)     : {s.resourceUsage.scratchMemUsageInBytes} B");
            _output.WriteLine("");
        }

        _output.WriteLine("Interpretation: a 16x16 tile from ONE subgroup stages 8 KB for 32,768 MACs");
        _output.WriteLine("= 4.0 MAC/byte. llama.cpp's applicable KHR_coopmat tile (128x128, BK=32,");
        _output.WriteLine("128 threads) reaches 32.0 MAC/byte in 16 KB. See .docs/COOPMAT_GEMM_DIAGNOSIS.md.");
    }
}
