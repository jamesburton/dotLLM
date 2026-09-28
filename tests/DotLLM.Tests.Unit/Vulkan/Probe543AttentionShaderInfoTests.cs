using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #543 — the driver's own post-compile register / LDS / scratch
/// numbers for every coopmat flash-attention arm, via
/// <c>VK_AMD_shader_info</c>.
/// </summary>
/// <remarks>
/// <para>
/// #543 arm A ("f32-input scalar tail", <c>_v4f32tail</c>) costs a settled
/// 3.5-4% at p512 and the user asked whether that can be recovered. Two
/// hypotheses: the tail's serialized global V reads replacing one coalesced
/// staging pass, or occupancy lost to the extra <c>precise float acc[O_CELLS]</c>
/// accumulator array. This answers the second one with a number instead of
/// inferring it from timings — the move that settled #545's residual, where
/// <c>VK_AMD_shader_info</c> showed occupancy was NOT the cause.
/// </para>
/// <para>
/// Note what this can and cannot see: the spec-constant gate is substituted
/// before the driver's backend compiler runs, so the <c>REQUIRE_INVARIANT_PV=0</c>
/// row is the real compiled cost of the NVIDIA-exempt path, not a runtime
/// branch — that is exactly why #533's gate is a specialization constant and not
/// a push constant.
/// </para>
/// <para>Enable with <c>DOTLLM_543_SHADER_INFO=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe543AttentionShaderInfoTests
{
    private readonly ITestOutputHelper _out;
    public Probe543AttentionShaderInfoTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] Arms =
    [
        "attention_flash_f32_coopmat",
    ];

    [SkippableFact]
    public void Arms_ShaderStatistics()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_543_SHADER_INFO"), "1", StringComparison.Ordinal),
            "DOTLLM_543_SHADER_INFO=1 to enable this diagnostic.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(VulkanFlashAttentionCoopmatKernel.SupportsDevice(device), "No 16x16x16 f16->f32 subgroup coopmat tile.");
        Skip.IfNot(device.HasShaderInfoAmd, "Device/driver does not advertise VK_AMD_shader_info.");

        _out.WriteLine($"Device: {device.DeviceName}");
        _out.WriteLine("| shader | gate | VGPR | SGPR | LDS used | LDS alloc | scratch |");
        _out.WriteLine("|---|---:|---:|---:|---:|---:|---:|");

        foreach (string name in Arms)
        {
            foreach (uint? gate in new uint?[] { 1u, 0u })
            {
                VulkanFlashAttentionCoopmatKernel kernel;
                try
                {
                    kernel = VulkanFlashAttentionCoopmatKernel.Create(
                        device, spvDir, FlashAttentionCoopmatVariant.Default, name, requireInvariantPv: gate);
                }
                catch (Exception ex)
                {
                    _out.WriteLine($"| {name} | {gate} | UNAVAILABLE ({ex.GetType().Name}) | | | | |");
                    continue;
                }

                using (kernel)
                {
                    Report(device, name + " (base)", gate!.Value, kernel.PipelineHandle);
                    if (kernel.Hd64PipelineHandle != 0)
                        Report(device, name + " (hd64)", gate!.Value, kernel.Hd64PipelineHandle);
                }
            }
        }
    }

    private void Report(VulkanDevice device, string label, uint gate, nint pipeline)
    {
        var s = device.GetShaderStatisticsAmd(pipeline);
        _out.WriteLine($"| {label} | {gate} | {s.resourceUsage.numUsedVgprs} / {s.numAvailableVgprs} | " +
                       $"{s.resourceUsage.numUsedSgprs} | {s.resourceUsage.ldsUsageSizeInBytes} | " +
                       $"{s.resourceUsage.ldsSizePerLocalWorkGroup} | {s.resourceUsage.scratchMemUsageInBytes} |");
    }
}
