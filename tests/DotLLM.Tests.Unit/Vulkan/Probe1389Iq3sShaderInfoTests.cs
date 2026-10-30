using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for #1389: the driver's post-compile VGPR / LDS / scratch numbers for the IQ3_S (and IQ4_XS, the control that scales to 16 columns)
/// multi-column GEMV pipelines at every column count, via <c>VK_AMD_shader_info</c>. Enable with <c>DOTLLM_1389_SHADER_INFO=1</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe1389Iq3sShaderInfoTests
{
    private readonly ITestOutputHelper _out;
    public Probe1389Iq3sShaderInfoTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void MultiGemv_ShaderStatistics()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_1389_SHADER_INFO"), "1", StringComparison.Ordinal),
            "DOTLLM_1389_SHADER_INFO=1 to enable this diagnostic.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasShaderInfoAmd, "Device/driver does not advertise VK_AMD_shader_info.");
        Console.WriteLine($"Device: {device.DeviceName}");
        Console.WriteLine("| format | NC | VGPR | SGPR | LDS used | LDS alloc | scratch |");
        Console.WriteLine("|---|---:|---:|---:|---:|---:|---:|");
        foreach (var format in new[] { QuantGemvMultiFormat.IQ3_S, QuantGemvMultiFormat.IQ4_XS })
        {
            using var k = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, format);
            if (k is null) { Console.WriteLine($"| {format} | unavailable |"); continue; }
            foreach (int n in new[] { 2, 4, 8, 9, 12, 16 })
            {
                if (n > k.MaxColumnsSupported) continue;
                var s = device.GetShaderStatisticsAmd(k.PipelineHandleFor(n));
                Console.WriteLine($"| {format} | {n} | {s.resourceUsage.numUsedVgprs} / {s.numAvailableVgprs} | {s.resourceUsage.numUsedSgprs} | " +
                                  $"{s.resourceUsage.ldsUsageSizeInBytes} | {s.resourceUsage.ldsSizePerLocalWorkGroup} | {s.resourceUsage.scratchMemUsageInBytes} |");
            }
        }
    }
}
