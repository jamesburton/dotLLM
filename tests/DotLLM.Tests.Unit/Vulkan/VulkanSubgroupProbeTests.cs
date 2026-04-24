using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Reports the <see cref="VulkanDevice.SubgroupSize"/> and
/// <see cref="VulkanDevice.HasSubgroupArithmetic"/> probe results to test
/// output — used both as a sanity check that the Vulkan 1.1 property query
/// does not throw and as a human-readable record of the host GPU's subgroup
/// capabilities. Always passes; the values are informational.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSubgroupProbeTests
{
    private readonly ITestOutputHelper _output;

    public VulkanSubgroupProbeTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Probe_ReportsSubgroupCapability()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device           : {device.DeviceName}");
        _output.WriteLine($"VendorId         : 0x{device.VendorId:X4}");
        _output.WriteLine($"SubgroupSize     : {device.SubgroupSize}");
        _output.WriteLine($"HasSubgroupArith : {device.HasSubgroupArithmetic}");

        // A Vulkan 1.1+ driver must advertise SOME non-zero subgroup width.
        // If SubgroupSize is 0 we're on a Vulkan 1.0-only driver — which is
        // a valid configuration; the shared-memory path must cover it.
        if (device.HasSubgroupArithmetic)
            Assert.True(device.SubgroupSize > 0, "HasSubgroupArithmetic=true implies SubgroupSize>0");
    }

    /// <summary>
    /// Reports the <see cref="VulkanDevice.HasCooperativeMatrix"/> +
    /// <see cref="VulkanDevice.SupportedCooperativeMatrixProperties"/> probe
    /// output. On hosts that support VK_KHR_cooperative_matrix (e.g. the
    /// AMD Radeon 8060S iGPU, llama.cpp reports <c>matrix cores: KHR_coopmat</c>)
    /// the probe must succeed and list at least one usable tile shape. On
    /// hosts without the extension the test is informational — it just dumps
    /// what the driver reported and confirms the probe did not throw.
    /// </summary>
    [SkippableFact]
    public void Probe_ReportsCooperativeMatrixCapability()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device                : {device.DeviceName}");
        _output.WriteLine($"VendorId              : 0x{device.VendorId:X4}");
        _output.WriteLine($"HasCooperativeMatrix  : {device.HasCooperativeMatrix}");
        _output.WriteLine($"Reported tile shapes  : {device.SupportedCooperativeMatrixProperties.Count}");

        foreach (var s in device.SupportedCooperativeMatrixProperties)
        {
            _output.WriteLine(
                $"  {s.MSize,3} x {s.NSize,3} x {s.KSize,3}   " +
                $"A={CompTypeName(s.AType)} B={CompTypeName(s.BType)} " +
                $"C={CompTypeName(s.CType)} R={CompTypeName(s.ResultType)}  " +
                $"scope={ScopeName(s.Scope)}");
        }

        // When the flag is true the probe must actually have listed at least
        // one usable (≥16×16×16, subgroup-scope, F16/F32 or Sint8/Sint32) entry.
        if (device.HasCooperativeMatrix)
        {
            bool anyUsable = false;
            foreach (var s in device.SupportedCooperativeMatrixProperties)
            {
                if (s.Scope != 3) continue;               // Subgroup
                if (s.MSize < 16 || s.NSize < 16 || s.KSize < 16) continue;
                bool f16f32 = s.AType == 0 && s.BType == 0 && s.CType == 1 && s.ResultType == 1;
                bool i8i32  = s.AType == 3 && s.BType == 3 && s.CType == 5 && s.ResultType == 5;
                if (f16f32 || i8i32) { anyUsable = true; break; }
            }
            Assert.True(anyUsable,
                "HasCooperativeMatrix=true but no usable tile shape (≥16×16×16, subgroup, F16/F32 or Sint8/Sint32) was reported.");
        }
    }

    private static string CompTypeName(int t) => t switch
    {
        0 => "f16", 1 => "f32", 2 => "f64",
        3 => "s8",  4 => "s16", 5 => "s32",  6 => "s64",
        7 => "u8",  8 => "u16", 9 => "u32", 10 => "u64",
        _ => $"t{t}"
    };

    private static string ScopeName(int s) => s switch
    {
        1 => "Device", 2 => "Workgroup", 3 => "Subgroup", 5 => "QueueFamily",
        _ => $"s{s}"
    };
}
