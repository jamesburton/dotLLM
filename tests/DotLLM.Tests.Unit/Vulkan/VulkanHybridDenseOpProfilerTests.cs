using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Accounting cover for the hybrid-dense per-op profiler (issue #434).
/// </summary>
/// <remarks>
/// <para>
/// The attribution table these hooks produce is only worth acting on if it accounts for the
/// pass. Two things can break that and neither is visible by eye in a 14-row table: a region of
/// the command stream recorded after a submit's last category mark (which would simply be
/// absent), and a mode whose per-category numbers do not add up to the total it divides by.
/// </para>
/// <para>
/// The first is what the <c>other</c> bucket exists for. <see cref="VulkanOpProfiler"/> writes a
/// closing stamp tagged <c>other</c> immediately before every submit, so any untagged tail is
/// charged there rather than disappearing — which makes "<c>other</c> is ~0" a real assertion
/// and not a tautology. Mid-stream work needs no such check: a bucket is by construction
/// "everything recorded since the previous mark", so an unmarked op is charged to the next
/// category rather than lost.
/// </para>
/// <para>
/// What these tests deliberately do <b>not</b> assert is a floor on
/// <c>attributed / wall</c>. On this synthetic fixture the GPU work is microseconds while the
/// host pays fixture load, pipeline creation and 4 fence round-trips, so coverage is
/// legitimately poor and a threshold here would encode a property of the fixture rather than of
/// the profiler. Coverage is a measurement to report from the real model, not a test.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanHybridDenseOpProfilerTests
{
    /// <summary>
    /// Every <see cref="VulkanOpProfiler.Cat"/> must have a reporting name at its own index —
    /// otherwise a newly added category would silently report under its neighbour's label, or
    /// index past the end of the array. Needs no GPU.
    /// </summary>
    [Fact]
    public void CategoryNames_CoverEveryCategory()
    {
        var values = Enum.GetValues<VulkanOpProfiler.Cat>();
        Assert.Equal(values.Length, VulkanOpProfiler.CategoryNames.Length);
        foreach (var cat in values)
            Assert.False(string.IsNullOrWhiteSpace(VulkanOpProfiler.CategoryNames[(int)cat]));
    }

    /// <summary>
    /// Profiling is off unless asked for: a forward on an untouched model must leave
    /// <c>LastProfile</c> null. This is the observable half of "zero cost when disabled" — no
    /// report means no timestamps were written and no query pool was created.
    /// </summary>
    [SkippableFact]
    public void Forward_WithoutOverride_ProducesNoProfile()
    {
        var model = Load(out var device, out var gguf);
        using (device)
        using (gguf)
        using (model)
        {
            using ITensor _ = model.Forward([1, 2, 3, 4], [0, 1, 2, 3], deviceId: -1);
            Assert.Null(model.LastProfile);
        }
    }

    /// <summary>
    /// GPU-timestamp mode: the per-category times must sum to the total the shares divide by,
    /// nothing may be negative, the untagged-tail bucket must be ~0, and the attributed GPU time
    /// cannot exceed the wall clock that contains it.
    /// </summary>
    [SkippableFact]
    public void Forward_GpuTimestampMode_AccountsForThePass()
    {
        AssertAccounts(split: false);
    }

    /// <summary>
    /// Split-submit cross-check mode. Same accounting invariants, a completely different
    /// measurement mechanism (host wall time across a drain per boundary, no query pool) — so
    /// agreement here is evidence the boundaries themselves are placed where the model thinks
    /// they are, independent of timestamp tagging.
    /// </summary>
    [SkippableFact]
    public void Forward_SplitSubmitMode_AccountsForThePass()
    {
        AssertAccounts(split: true);
    }

    /// <summary>
    /// Decode-shaped forwards are excluded by default (min sequence length 2), so a
    /// single-token forward must not produce a report even with the override set. Prefill is
    /// what #434 is attributing, and a per-token report would bury it.
    /// </summary>
    [SkippableFact]
    public void Forward_SingleToken_IsNotProfiledByDefault()
    {
        var model = Load(out var device, out var gguf);
        using (device)
        using (gguf)
        using (model)
        {
            model.ProfileOverride = true;
            using ITensor _ = model.Forward([5], [0], deviceId: -1);
            Assert.Null(model.LastProfile);
        }
    }

    private static void AssertAccounts(bool split)
    {
        var model = Load(out var device, out var gguf);
        using (device)
        using (gguf)
        using (model)
        {
            model.ProfileOverride = true;
            model.ProfileSplitOverride = split;

            using (ITensor _ = model.Forward([1, 2, 3, 4], [0, 1, 2, 3], deviceId: -1)) { }

            var report = model.LastProfile;
            Assert.NotNull(report);
            Skip.If(report.Mode == "unavailable", "Device does not support timestamp queries.");
            Assert.Equal(split ? "split" : "gpu-timestamp", report.Mode);
            Assert.Equal(4, report.SeqLen);

            double sum = 0;
            foreach (var (name, ms) in report.ByCategory)
            {
                Assert.True(ms >= 0, $"category '{name}' is negative ({ms:F4} ms)");
                sum += ms;
            }

            // The denominator every share is computed against must be the sum of the rows.
            Assert.Equal(sum, report.AttributedMs, 6);
            Assert.True(report.AttributedMs > 0, "nothing was attributed at all");

            // Untagged tail. Some categories are absent on this fixture (no PQ2_0, so no
            // hadamard); 'other' being absent entirely is the pass condition.
            report.ByCategory.TryGetValue("other", out double other);
            Assert.True(other <= 0.02 * report.AttributedMs,
                $"'other' holds {other:F3} ms of {report.AttributedMs:F3} ms — a command-stream " +
                "region is recorded after a submit's last category mark and is not attributed.");

            // The buckets measure work that happened inside the forward, so they cannot outlast it.
            Assert.True(report.AttributedMs <= report.WallMs + 1e-6,
                $"attributed {report.AttributedMs:F3} ms exceeds wall {report.WallMs:F3} ms");
            Assert.True(report.RecordMs >= 0 && report.WaitMs >= 0);

            // The buckets that this fixture's graph must reach, whichever layer kind ran.
            Assert.Contains("proj_ffn", report.ByCategory.Keys);
            Assert.Contains("lm_head", report.ByCategory.Keys);

            // #445 split gdn_scan and attention into their constituent ops. The PARENT names
            // are therefore no longer marked directly — asserting the sub-buckets is what now
            // proves the graph reached the recurrent scan and the attention kernel. If a future
            // change collapses a sub-bucket back into its parent this fails, which is the point.
            Assert.Contains("gdn_scan_core", report.ByCategory.Keys);
            Assert.Contains("gdn_postgate", report.ByCategory.Keys);
            Assert.Contains("attn_core", report.ByCategory.Keys);
            Assert.Contains("attn_gate", report.ByCategory.Keys);
            Assert.DoesNotContain("gdn_scan", report.ByCategory.Keys);
            Assert.DoesNotContain("attention", report.ByCategory.Keys);

            // ...and the roll-up must restore the #434-comparable parents in the formatted
            // report, so the new numbers can be laid beside the old table.
            string formatted = report.Format("t");
            Assert.Contains("rolled up to #434 parent buckets", formatted);
            Assert.Contains("gdn_scan ", formatted);
            Assert.Contains("attention ", formatted);

            // The census must name a concrete kernel per projection shape, not a placeholder.
            Assert.NotEmpty(report.Dispatches);
            Assert.Contains(report.Dispatches.Keys, k => k.StartsWith("matmul_", StringComparison.Ordinal));
        }
    }

    private static VulkanQwen3HybridDenseTransformerModel Load(
        out VulkanDevice device, out GgufFile gguf)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(Path.GetTempPath(), $"qwen35-prof-{Guid.NewGuid():N}.gguf"), withMtp: false);
        try
        {
            gguf = GgufFile.Open(path);
        }
        finally
        {
            // The loader mmaps the file; on Windows the handle keeps it alive until Dispose,
            // so deleting the path now would fail. Left for the OS temp sweep instead.
        }

        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        device = VulkanDevice.Create();
        var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
        return Assert.IsType<VulkanQwen3HybridDenseTransformerModel>(model);
    }
}
