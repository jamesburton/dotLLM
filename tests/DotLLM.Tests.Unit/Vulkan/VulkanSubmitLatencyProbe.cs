using System.Diagnostics;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in (<c>DOTLLM_SUBMIT_LATENCY_PROBE=1</c>): fixed per-submit round-trip cost of an EMPTY command buffer
/// (<c>Begin</c> -> <c>SubmitAndWait</c>), i.e. the floor under every forward's wall time that no kernel work can remove.
/// Run with <c>DOTLLM_VK_SPIN_WAIT=0</c> and <c>=full</c> to see how much of it is host wake-up.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanSubmitLatencyProbe
{
    private readonly ITestOutputHelper _output;
    public VulkanSubmitLatencyProbe(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Probe_EmptySubmitRoundTrip()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_SUBMIT_LATENCY_PROBE"), "1", StringComparison.Ordinal), "DOTLLM_SUBMIT_LATENCY_PROBE=1");
        using var device = VulkanDevice.Create();
        using var ctx = device.CreateSubmitContext();
        const int n = 300;
        for (int i = 0; i < 20; i++) { ctx.Begin(); ctx.SubmitAndWait(); }
        var times = new double[n];
        for (int i = 0; i < n; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            ctx.Begin();
            ctx.SubmitAndWait();
            times[i] = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
        }
        Array.Sort(times);
        Console.WriteLine($"EMPTY submit+wait (spin={Environment.GetEnvironmentVariable("DOTLLM_VK_SPIN_WAIT") ?? "auto"}): p10 {times[n / 10]:F3} ms, median {times[n / 2]:F3} ms, p90 {times[n * 9 / 10]:F3} ms, max {times[^1]:F3} ms");

        // Same probe after a host-side idle gap (busy-wait, so the CPU stays awake and only the GPU idles): does an idle GPU add wake-up latency?
        foreach (double gapMs in new[] { 0.5, 2.0, 5.0 })
        {
            var t2 = new double[100];
            for (int i = 0; i < t2.Length; i++)
            {
                long g0 = Stopwatch.GetTimestamp();
                while ((Stopwatch.GetTimestamp() - g0) * 1000.0 / Stopwatch.Frequency < gapMs) { }
                long t0 = Stopwatch.GetTimestamp();
                ctx.Begin();
                ctx.SubmitAndWait();
                t2[i] = (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;
            }
            Array.Sort(t2);
            Console.WriteLine($"EMPTY submit+wait after {gapMs} ms idle (spin={Environment.GetEnvironmentVariable("DOTLLM_VK_SPIN_WAIT") ?? "auto"}): median {t2[50]:F3} ms, p90 {t2[90]:F3} ms, max {t2[^1]:F3} ms");
        }
    }
}
