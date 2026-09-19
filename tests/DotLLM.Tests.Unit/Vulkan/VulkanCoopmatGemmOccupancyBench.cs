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

        var variants = new (string Name, PQ2_0GemmVariant V, double MacPerByte)[]
        {
            ("pq2_0 gemm coopmat32 (SHIPS on gfx1151)",        PQ2_0GemmVariant.Coopmat32,      4.0),
            ("pq2_0 gemm coopmat (64-thread sibling)",         PQ2_0GemmVariant.Coopmat,        4.0),
            ("pq2_0 gemm register-blocked (oracle)",           PQ2_0GemmVariant.RegisterBlocked, 1.0),
            ("#439 ladder (a) 16x16 / 1 sg  / BK=32 CONTROL",  PQ2_0GemmVariant.Ladder16x16x1,   4.0),
            ("#439 ladder (b) 64x64 / 4 sg  / BK=32",          PQ2_0GemmVariant.Ladder64x64x4,  16.0),
            ("#439 ladder (c) 128x128 / 4 sg / BK=32",         PQ2_0GemmVariant.Ladder128x128x4, 32.0),
        };

        foreach (var (name, variant, macPerByte) in variants)
        {
            if (!variant.IsSupportedOn(device)) { _output.WriteLine($"{name}: not supported on this device"); continue; }

            using var k = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);
            var s = device.GetShaderStatisticsAmd(k.PipelineHandle);

            uint vgpr = s.resourceUsage.numUsedVgprs;
            uint cap = s.numAvailableVgprs == 0 ? 256u : s.numAvailableVgprs;
            uint file = s.numPhysicalVgprs == 0 ? 1536u : s.numPhysicalVgprs;
            ulong lds = s.resourceUsage.ldsUsageSizeInBytes;

            // WAVES PER SIMD COMES FROM THE REGISTER FILE, NOT THE PER-WAVE CAP. An earlier
            // revision of this bench divided by numAvailableVgprs (256 — the driver's per-wave
            // ALLOCATION CAP) and reported "1 wave/SIMD" for a kernel that actually runs at ~9.
            // A whole diagnosis was built on that and then retracted; see the retraction block in
            // .docs/COOPMAT_GEMM_DIAGNOSIS.md. The file is numPhysicalVgprs (1536 on gfx1151) and
            // RDNA caps residency at 16 waves/SIMD regardless.
            uint wavesByVgpr = vgpr == 0 ? 0 : Math.Min(16u, file / vgpr);
            double wgByLds = lds == 0 ? double.PositiveInfinity : 65536.0 / lds;

            _output.WriteLine(name);
            _output.WriteLine($"   arithmetic intensity: {macPerByte:F1} MAC/byte staged");
            _output.WriteLine($"   workgroup           : {s.computeWorkGroupSizeX} threads");
            _output.WriteLine($"   VGPR used           : {vgpr}  (per-wave cap {cap}, file {file}) -> {wavesByVgpr} waves/SIMD");
            _output.WriteLine($"   SGPR used/available : {s.resourceUsage.numUsedSgprs} / {s.numAvailableSgprs} (physical {s.numPhysicalSgprs})");
            _output.WriteLine($"   LDS used            : {lds} B -> {wgByLds:F1} workgroups/WGP by LDS (64 KB)");
            _output.WriteLine($"   scratch (spill)     : {s.resourceUsage.scratchMemUsageInBytes} B");
            _output.WriteLine("");
        }

        _output.WriteLine("Interpretation: a 16x16 tile from ONE subgroup stages 2 KB for 8,192 MACs at");
        _output.WriteLine("BK=32 (8 KB / 32,768 at the shipping BK=128) = 4.0 MAC/byte either way.");
        _output.WriteLine("llama.cpp's applicable KHR_coopmat tile (l_warptile_mmq: 128x128, BK=32, 128");
        _output.WriteLine("threads) reaches 32.0 MAC/byte in 16 KB. See .docs/COOPMAT_GEMM_DIAGNOSIS.md.");
    }
}
