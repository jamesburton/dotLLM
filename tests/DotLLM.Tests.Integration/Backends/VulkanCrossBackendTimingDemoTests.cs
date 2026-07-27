using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// Proves the <see cref="CrossBackendTimingHarness"/> times a REAL GPU forward end-to-end.
/// Loads a SUPPORTED-arch tiny model (SmolLM-135M, Llama family — the same fixture the
/// existing <see cref="DotLLM.Tests.Integration.Vulkan.VulkanForwardPerfHarness"/> and
/// parity tests use) on the Vulkan backend and runs it through the cross-backend harness,
/// emitting per-phase CSV (load / warmup / prefill / decode) with tokens/sec.
/// </summary>
/// <remarks>
/// <para>This is the harness's proof-of-life on the real AMD Radeon 8060S present on the
/// dev host: it does NOT skip when a Vulkan device is available — it executes the GPU
/// forward and prints actual numbers. gemma4 is intentionally NOT exercised here (it is
/// not wired into the Vulkan backend); the supported-arch run demonstrates the timing
/// plumbing works, and <see cref="Gemma4GpuGapProbeTests"/> captures the gemma4 gap.</para>
/// <para>For the CPU↔Vulkan comparison the same harness can be re-run with
/// <see cref="TimingBackend.Cpu"/> against the same GGUF; the CSV rows compose because both
/// backends emit the identical <c>phase,name,ms,tokens_per_sec</c> shape.</para>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public sealed class VulkanCrossBackendTimingDemoTests
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanCrossBackendTimingDemoTests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void Harness_TimesRealVulkanForward_OnSupportedArch()
    {
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            CrossBackendTimingHarness.IsAvailable(TimingBackend.Vulkan),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();
        var options = new CrossBackendTimingHarness.Options(
            PrefillForwards: 3, PrefillTokens: 16, DecodeForwards: 16, WarmupForwards: 2, MaxSeqLen: 256);

        var rows = CrossBackendTimingHarness.Run(
            TimingBackend.Vulkan, _fixture.FilePath, "smollm-vk", options, spvDir);

        // Emit the CSV so the harness output is visible / capturable from the test log.
        _output.WriteLine(CrossBackendTimingHarness.ToCsv(rows));

        // Sanity: we actually timed a load + warmup + the requested prefill/decode forwards,
        // and the steady-state decode produced a positive tokens/sec on the real GPU.
        Assert.Contains(rows, r => r.Phase == "load");
        Assert.Equal(options.WarmupForwards, rows.Count(r => r.Phase == "warmup"));
        Assert.Equal(options.PrefillForwards, rows.Count(r => r.Phase == "prefill"));
        Assert.Equal(options.DecodeForwards, rows.Count(r => r.Phase == "decode"));

        var decodeAvg = rows.Single(r => r.Phase == "decode_avg");
        Assert.True(decodeAvg.Milliseconds > 0, "decode_avg ms should be positive");
        Assert.True(decodeAvg.TokensPerSec > 0, "decode_avg tokens/sec should be positive");
        _output.WriteLine(
            $"VULKAN decode_avg: {decodeAvg.Milliseconds:F2} ms/tok, {decodeAvg.TokensPerSec:F1} tok/s");
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
