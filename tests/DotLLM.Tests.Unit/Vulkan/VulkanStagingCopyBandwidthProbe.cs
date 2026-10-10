using System.Diagnostics;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #874 diagnostic (opt-in: <c>DOTLLM_PROBE_COPY_BW=1</c>): raw host-visible -&gt; device-local <c>vkCmdCopyBuffer</c> throughput on this
/// device, i.e. the ceiling of the staged-upload path once the CPU side is parallel. Reports numbers; asserts only that it ran.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanStagingCopyBandwidthProbe
{
    private readonly ITestOutputHelper _out;
    public VulkanStagingCopyBandwidthProbe(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void HostVisibleToDeviceLocalCopyBandwidth()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_PROBE_COPY_BW") == "1", "opt-in diagnostic");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        foreach (long mib in new long[] { 32, 256 })
        {
            long bytes = mib << 20;
            using var host = device.Allocate(bytes);
            using var dev = device.AllocateDeviceLocal(bytes);
            device.CopyBufferSynchronous(host, dev, (ulong)bytes);   // warm-up
            const int reps = 8;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < reps; i++) device.CopyBufferSynchronous(host, dev, (ulong)bytes);
            double s = sw.Elapsed.TotalSeconds;
            _out.WriteLine($"host-visible -> device-local, {mib} MiB x{reps}: {bytes * reps / s / 1e9:F2} GB/s ({s / reps * 1000:F1} ms per copy)");
        }
    }
}
