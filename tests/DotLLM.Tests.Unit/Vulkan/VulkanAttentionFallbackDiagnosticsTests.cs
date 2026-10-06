using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the issue #441 silent-fallback diagnostic. No GPU required — the reporting side
/// is pure string handling, deliberately separable from the kernel it describes.
/// </summary>
/// <remarks>
/// Shares the <c>VulkanKernels</c> collection despite needing no GPU: these tests Reset() and
/// assert on the process-wide de-duplication set that the GPU-side factory tests also drive, so
/// they must not run concurrently with them.
/// </remarks>
[Collection("VulkanKernels")]
public sealed class VulkanAttentionFallbackDiagnosticsTests
{
    /// <summary>The head-dim warning must name every fact needed to act on it.</summary>
    [Fact]
    public void HeadDimWarning_NamesModelBoundAndRemedy()
    {
        VulkanAttentionFallbackDiagnostics.Reset();
        VulkanAttentionFallbackDiagnostics.ReportHeadDimTooWide("Qwen3Hybrid", headDim: 256, supportedMaxHeadDim: 128);

        string message = Assert.Single(VulkanAttentionFallbackDiagnostics.Reported);
        Assert.Contains("Qwen3Hybrid", message, StringComparison.Ordinal);
        Assert.Contains("256", message, StringComparison.Ordinal);
        Assert.Contains("128", message, StringComparison.Ordinal);
        // The remedy has to be in the text: the whole point of #441 is that the next person
        // should not have to reverse-engineer the gate from a profile.
        Assert.Contains("attention_flash_f32_hd", message, StringComparison.Ordinal);
        Assert.Contains("#441", message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A 64-layer model must print one line, not 64 — a diagnostic that floods gets muted, and
    /// a muted diagnostic is the silent fallback again.
    /// </summary>
    [Fact]
    public void RepeatedReports_AreDeduplicated()
    {
        VulkanAttentionFallbackDiagnostics.Reset();
        for (int i = 0; i < 64; i++)
            VulkanAttentionFallbackDiagnostics.ReportHeadDimTooWide("Qwen3Hybrid", 256, 128);

        Assert.Single(VulkanAttentionFallbackDiagnostics.Reported);
    }

    /// <summary>Distinct causes stay distinct — one model's gate must not mask another's.</summary>
    [Fact]
    public void DistinctCauses_AreReportedSeparately()
    {
        VulkanAttentionFallbackDiagnostics.Reset();
        VulkanAttentionFallbackDiagnostics.ReportHeadDimTooWide("Qwen3Hybrid", 256, 128);
        VulkanAttentionFallbackDiagnostics.ReportHeadDimTooWide("NemotronH", 192, 128);
        VulkanAttentionFallbackDiagnostics.ReportUnavailable("NemotronH", "SPV missing");

        Assert.Equal(3, VulkanAttentionFallbackDiagnostics.Reported.Count);
    }
}
