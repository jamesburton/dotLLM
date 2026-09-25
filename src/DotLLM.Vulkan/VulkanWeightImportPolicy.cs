namespace DotLLM.Vulkan;

/// <summary>
/// The single decision point for "does this weight tensor become a zero-copy
/// <c>VK_EXT_external_memory_host</c> import of its mmap'd source pages, or a staging
/// copy into device-local memory?" — and the per-load ledger of what that decided
/// (issue #508).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this is one type and not a per-model behaviour.</b> Before #508 only
/// <c>VulkanWeights</c> could import; Bonsai 2 27B, Qwen3-MoE-hybrid, Nemotron-H and
/// Mamba-3 all staged unconditionally, so on a 32 GiB UMA box Bonsai 2 held 13.9 GiB —
/// 6.86 GiB of still-resident mmap plus 7.01 GiB of device-local copy
/// (<c>.docs/agents/438-report.md</c>). But the import is not the only route to that
/// memory, and the other route is its exact opposite:
/// </para>
/// <list type="bullet">
///   <item><b>#508 (import)</b> — the mmap pages <i>become</i> the device memory. No
///   copy, no second resident copy. Refused on a discrete GPU (#507), where those pages
///   are system RAM read across PCIe for the model's lifetime.</item>
///   <item><b>#438 (release)</b> — stage as before, then unmap the GGUF, returning the
///   6.8 GiB of source pages. Works everywhere, including a dGPU.</item>
/// </list>
/// <para>
/// <b>You cannot unmap pages you have imported</b>, so the two are mutually exclusive
/// per tensor and complementary across devices. This type is what makes them one policy
/// instead of two behaviours that can both be half-on:
/// </para>
/// <list type="bullet">
///   <item><see cref="MayReleaseWholeHostMapping"/> — true iff NOTHING imported from
///   this load. That is the precondition for #438's whole-file <c>gguf.Dispose()</c>. On
///   a dGPU the import is always refused, so it is always true there and #438 gets the
///   full release; on UMA with the import live it is false and the mapping must stay.</item>
///   <item><see cref="StagedSourceRanges"/> — the per-tensor source ranges that took the
///   staging copy and are therefore dead after upload. Even on a UMA load where some
///   tensors imported, #438 may still release <i>these</i> ranges page-wise
///   (<c>OfferVirtualMemory</c> on Windows, <c>madvise(MADV_DONTNEED)</c> elsewhere)
///   without touching the imported ones.</item>
/// </list>
/// <para>
/// <b>State is per load.</b> <see cref="Reset"/> at the top of each weights loader;
/// weight loading is single-threaded, so plain statics suffice (same convention as
/// <c>VulkanWeights</c>'s existing upload counters).
/// </para>
/// <para>
/// <b>Alignment is not a fallback reason.</b> <c>HostVisibleBuffer.TryCreate</c> rounds
/// the source pointer DOWN to <c>minImportedHostPointerAlignment</c> and binds the
/// buffer at the resulting offset, so an arbitrarily-aligned tensor start inside the
/// mmap still imports. Fallbacks are policy fallbacks — the extension is absent, the
/// device is discrete (#507), the env var is set, or the bytes going to the device are
/// not the source bytes (any host dequant/widen/convert) — not alignment ones.
/// </para>
/// </remarks>
internal static class VulkanWeightImportPolicy
{
    /// <summary>Escape hatch: <c>DOTLLM_VULKAN_DISABLE_HOST_IMPORT=1</c> forces every tensor
    /// to stage. Read per call — tests toggle it at runtime.</summary>
    public static bool IsDisabled =>
        string.Equals(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT"),
            "1", StringComparison.Ordinal);

    private static readonly List<(nint Pointer, long Bytes)> s_stagedRanges = [];

    /// <summary>Tensors that took the zero-copy import on the current load.</summary>
    public static int ImportedTensorCount { get; private set; }

    /// <summary>Bytes aliased (not copied) by the zero-copy import on the current load.</summary>
    public static long ImportedBytes { get; private set; }

    /// <summary>Tensors that fell back to a staging copy on the current load.</summary>
    public static int StagedTensorCount { get; private set; }

    /// <summary>Source bytes copied through staging on the current load.</summary>
    public static long StagedBytes { get; private set; }

    /// <summary>
    /// Why the most recent tensor fell back. One of "feature_absent" (no
    /// <c>VK_EXT_external_memory_host</c>, or the device is not integrated — see #507),
    /// "env_disabled", "null_src", "import_rejected", "too_small" (smaller than one import page),
    /// "not_source_bytes" (the device
    /// image is a host dequant/convert of the source, so there is nothing to alias), or
    /// the empty string when the last decision was an import.
    /// </summary>
    public static string LastFallbackReason { get; private set; } = string.Empty;

    /// <summary>
    /// Source ranges whose pages are dead after upload because their tensor staged —
    /// the #438 release candidates. Empty when nothing staged.
    /// </summary>
    public static IReadOnlyList<(nint Pointer, long Bytes)> StagedSourceRanges => s_stagedRanges;

    /// <summary>
    /// True iff no tensor on this load imported, so the whole source mapping can be
    /// unmapped after upload (#438's <c>gguf.Dispose()</c>). False the moment one tensor
    /// aliases the mapping — then only <see cref="StagedSourceRanges"/> may be released.
    /// </summary>
    public static bool MayReleaseWholeHostMapping => ImportedTensorCount == 0;

    /// <summary>Clears the ledger. Call at the top of each weights loader.</summary>
    public static void Reset()
    {
        ImportedTensorCount = 0;
        ImportedBytes = 0;
        StagedTensorCount = 0;
        StagedBytes = 0;
        LastFallbackReason = string.Empty;
        s_stagedRanges.Clear();
    }

    /// <summary>
    /// Attempts to alias <paramref name="bytes"/> bytes at <paramref name="srcPtr"/> as a
    /// device buffer. Returns false — with <see cref="LastFallbackReason"/> set — when the
    /// caller must stage instead. Callers that stage must then call
    /// <see cref="NoteStaged"/> so the #438 ledger stays complete.
    /// </summary>
    public static bool TryImport(
        VulkanDevice device, nint srcPtr, long bytes, out VulkanDevice.Buffer? buf)
    {
        ArgumentNullException.ThrowIfNull(device);
        buf = null;

        // HasExternalMemoryHost also covers the #507 UMA gate indirectly: on a discrete
        // GPU the extension may be present but TrySelectHostImportMemoryType refuses, and
        // TryWrapHostVisible returns null ("import_rejected").
        if (!device.HasExternalMemoryHost)
        {
            LastFallbackReason = "feature_absent";
            return false;
        }
        if (IsDisabled)
        {
            LastFallbackReason = "env_disabled";
            return false;
        }
        if (srcPtr == 0 || bytes <= 0)
        {
            LastFallbackReason = "null_src";
            return false;
        }
        // An import aliases whole pages and costs a VkDeviceMemory object, so it is only
        // worth it for tensors that are at least a page. Without this the per-layer
        // scalars of a recurrent model (d_state-sized norms, n_head-sized dt_bias) would
        // each burn an allocation slot to save a few hundred bytes of copy — and
        // maxMemoryAllocationCount is finite.
        if ((ulong)bytes < device.MinImportedHostPointerAlignment)
        {
            LastFallbackReason = "too_small";
            return false;
        }

        var wrapped = device.TryWrapHostVisible(srcPtr, bytes);
        if (wrapped is null)
        {
            // "rejected" on its own is not a cause. Carry the driver's own verdict —
            // which Vulkan call refused and with what VkResult — because the whole
            // import was dead on the real load path for want of exactly this line:
            // synthetic-memory tests passed while every production tensor was refused.
            LastFallbackReason =
                $"import_rejected({Interop.HostVisibleBuffer.LastImportFailureStage}" +
                $":{Interop.HostVisibleBuffer.LastImportFailureCode})";
            return false;
        }

        NoteImported(bytes);
        buf = wrapped;
        return true;
    }

    /// <summary>
    /// Records that a tensor of <paramref name="bytes"/> bytes aliased its source pages.
    /// Called by <see cref="TryImport"/>; separate so the #438 composition rule can be
    /// exercised without a Vulkan device.
    /// </summary>
    public static void NoteImported(long bytes)
    {
        ImportedTensorCount++;
        ImportedBytes += bytes;
        LastFallbackReason = string.Empty;
    }

    /// <summary>
    /// Records that a tensor whose source occupies <paramref name="sourceBytes"/> bytes at
    /// <paramref name="srcPtr"/> took the staging path — so those source pages are dead
    /// after upload and are a #438 release candidate. <paramref name="reason"/> defaults
    /// to whatever <see cref="TryImport"/> last set; pass it explicitly for tensors that
    /// never attempted an import (host dequant/convert paths).
    /// </summary>
    public static void NoteStaged(nint srcPtr, long sourceBytes, string? reason = null)
    {
        StagedTensorCount++;
        if (sourceBytes > 0) StagedBytes += sourceBytes;
        if (reason is not null) LastFallbackReason = reason;
        if (srcPtr != 0 && sourceBytes > 0)
            s_stagedRanges.Add((srcPtr, sourceBytes));
    }

    /// <summary>One-line ledger for load-time diagnostics and the #508 A/B.</summary>
    public static string Summary()
        => $"host-import: {ImportedTensorCount} tensors / {ImportedBytes / (1024.0 * 1024.0):F1} MiB aliased; " +
           $"{StagedTensorCount} tensors / {StagedBytes / (1024.0 * 1024.0):F1} MiB staged" +
           (LastFallbackReason.Length > 0 ? $" (last fallback: {LastFallbackReason})" : string.Empty) +
           $"; whole-mapping release {(MayReleaseWholeHostMapping ? "permitted" : "blocked")}";
}
