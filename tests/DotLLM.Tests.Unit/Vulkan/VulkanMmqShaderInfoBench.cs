using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// One-shot ground-truth diagnostic: queries the AMD proprietary driver
/// directly (<c>VK_AMD_shader_info</c> / <c>vkGetShaderInfoAMD</c>) for the
/// Q8_0 MMQ prefill kernel's actual post-compile register/LDS allocation.
/// </summary>
/// <remarks>
/// <para>
/// This is the cheap, no-install-required check flagged in
/// <c>.docs/HANDOFF.md</c> as worth trying before installing the AMD Radeon
/// GPU Profiler (RGP): a real, driver-reported number for the "register
/// spill" theory that the SPIR-V-IR-only comparison in #388 could only infer.
/// Unlike the #384-#390 black-box timing experiments, this reads the driver's
/// own compiled-shader metadata directly — no dispatch, no A/B, no noise.
/// </para>
/// <para>Enable with <c>DOTLLM_MMQ_SHADER_INFO_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmqShaderInfoBench
{
    private readonly ITestOutputHelper _output;
    public VulkanMmqShaderInfoBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void Bench_MmqShaderStatistics()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_MMQ_SHADER_INFO_BENCH") == "1",
            "DOTLLM_MMQ_SHADER_INFO_BENCH=1 to enable this diagnostic.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device lacks integer-dot-product support — MMQ unavailable.");
        Skip.IfNot(device.HasShaderInfoAmd, "Device/driver does not advertise VK_AMD_shader_info.");

        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");

        int reportedSize = device.GetShaderInfoAmdReportedSize(mmq.PipelineHandle);
        _output.WriteLine($"Driver-reported required buffer size for VkShaderStatisticsInfoAMD: {reportedSize} bytes (our struct sizeof: {sizeof(VkShaderStatisticsInfoAmd)})");

        var stats = device.GetShaderStatisticsAmd(mmq.PipelineHandle);

        _output.WriteLine($"Device: {device.DeviceName}");
        _output.WriteLine("matmul_q8_0_mmq.comp — driver-reported post-compile shader statistics:");
        _output.WriteLine($"  VGPRs used/available: {stats.resourceUsage.numUsedVgprs} / {stats.numAvailableVgprs} (physical: {stats.numPhysicalVgprs})");
        _output.WriteLine($"  SGPRs used/available: {stats.resourceUsage.numUsedSgprs} / {stats.numAvailableSgprs} (physical: {stats.numPhysicalSgprs})");
        _output.WriteLine($"  LDS allocation granularity: {stats.resourceUsage.ldsSizePerLocalWorkGroup} bytes");
        _output.WriteLine($"  LDS actually used: {stats.resourceUsage.ldsUsageSizeInBytes} bytes");
        _output.WriteLine($"  Scratch memory (register spill to VRAM): {stats.resourceUsage.scratchMemUsageInBytes} bytes");
        _output.WriteLine($"  Compute workgroup size: {stats.computeWorkGroupSizeX} x {stats.computeWorkGroupSizeY} x {stats.computeWorkGroupSizeZ}");

        // Non-zero scratch memory means the driver spilled registers to VRAM —
        // the concrete, measurable form of "register spill" #388 could only
        // infer from SPIR-V IR shape. Reported, not asserted on — this test's
        // job is to produce ground truth, not to pass/fail a threshold.
        if (stats.resourceUsage.scratchMemUsageInBytes > 0)
            _output.WriteLine("  => Driver IS spilling registers to scratch memory for this kernel.");
        else
            _output.WriteLine("  => No register spill reported — VGPR/SGPR allocation fits without scratch memory.");
    }
}
