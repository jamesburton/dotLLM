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
/// Written for #543 and kept because it answered two questions no amount of
/// A/B timing could, and will answer the next one:
/// </para>
/// <list type="bullet">
///   <item>Is the f32 scalar tail's cost occupancy? <b>No.</b> It takes the base
///   shader from 117 to 129 VGPRs with zero scratch, and the kernel is LDS-bound
///   at 31744 B (2 workgroups/WGP) either way, so the VGPRs are nowhere near
///   binding. That redirected the hunt to the V read path, where the cost was.</item>
///   <item>How much LDS can the hd64 variant afford? <b>341 bytes.</b> It sits at
///   21504 B and 65536 / 21504 = 3.05, so it runs <b>3</b> workgroups per WGP, not
///   the 2 its header had claimed by inheritance from the 128-dim shader. The
///   budget to stay at 3 is 65536 / 3 = 21845 B. That is what refuted extending
///   #543's split-f16 V arm into this shader: an 8 KB V_lo tile lands at 29696 B,
///   safe against 32768 B and not against 21845 B.</item>
/// </list>
/// <para>
/// Note what the <c>gate = 0</c> rows mean: #533's gate is a SPECIALIZATION
/// constant, substituted before the driver's backend compiler runs, so those rows
/// are the real compiled cost of the vendor-exempt path rather than a runtime
/// branch — which is the whole reason it is not a push constant.
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
        // The retained pre-#543 control, so the table shows what the f32 tail cost
        // in registers against the f16 one it replaced.
        "attention_flash_f32_coopmat_pre543",
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
