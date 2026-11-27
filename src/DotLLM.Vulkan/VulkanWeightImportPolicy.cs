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

    /// <summary>
    /// Whether a refused read-only import is retried through a shared writable alias view of the same file range (#1190, #1991).
    /// <c>DOTLLM_VULKAN_HOST_IMPORT_WRITABLE</c>: <c>0</c> = never, <c>1</c> = on (a discrete GPU still refuses the import, #507), unset = <b>off (was proposed on for an
    /// integrated device, withdrawn after measurement)</b>. The driver refuses read-only GGUF pages (<c>vkAllocateMemory</c> -1000072003), so without
    /// the alias every tensor is copied through host memory (a CLAUDE.md rule-3 bounce). Read per call.
    /// </summary>
    public static bool WritableAliasEnabled
    {
      get
      {
        string? v = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_HOST_IMPORT_WRITABLE");
        if (string.Equals(v, "0", StringComparison.Ordinal)) return false;
        if (string.Equals(v, "1", StringComparison.Ordinal)) return true;
        return false;   // #1991: opt-in. Default-on measured -19% decode / -33% prefill on dense Llama (see docs/GPU.md); DirectStorage is the neutral zero-copy route.
      }
    }

    /// <summary>Tensors whose import was attempted and refused on the current load (they are then staged) (#1991).</summary>
    public static int RejectedTensorCount { get; private set; }

    /// <summary>Bytes of the tensors counted by <see cref="RejectedTensorCount"/>.</summary>
    public static long RejectedBytes { get; private set; }

    /// <summary>The first refusal's detail on the current load (empty when none).</summary>
    public static string FirstRejection { get; private set; } = string.Empty;

    private static int s_reportedRejectedCount;

    /// <summary>
    /// One warning line when imports were refused since the last report, so a refusal is never silent (#1991); empty
    /// otherwise. Consuming: it reports each count once, so the staging buffers that are created and disposed after the
    /// load (per-layer MoE streaming, MTP heads, DFlash) or twice per load (<c>VulkanWeights</c>) do not reprint it.
    /// </summary>
    public static string RejectionWarning()
    {
        if (RejectedTensorCount == s_reportedRejectedCount) return string.Empty;
        s_reportedRejectedCount = RejectedTensorCount;
        return $"[dotllm] vulkan host import: {RejectedTensorCount} tensors / {RejectedBytes / (1024.0 * 1024.0):F1} MiB refused " +
               $"(first: {FirstRejection}); staged through host memory instead. Set DOTLLM_VULKAN_DISABLE_HOST_IMPORT=1 to silence.";
    }

    private static readonly List<(nint Pointer, long Bytes)> s_stagedRanges = [];

    /// <summary>Tensors below this size are not worth a DirectStorage request and a sub-buffer of their own (#1304).</summary>
    private const long DirectMinTensorBytes = 64 * 1024;

    /// <summary>Tensors loaded straight from the GGUF into device-local memory with DirectStorage on the current load (#1304).</summary>
    public static int DirectTensorCount { get; private set; }

    /// <summary>Bytes loaded with DirectStorage on the current load (#1304).</summary>
    public static long DirectBytes { get; private set; }

    /// <summary>First reason a tensor was declined by the DirectStorage tier on the current load (empty when none was); for the load log.</summary>
    public static string FirstDirectDecline { get; private set; } = string.Empty;

    /// <summary>Tensors the DirectStorage tier declined on the current load.</summary>
    public static int DirectDeclinedCount { get; private set; }

    // Host-only ranges (#818): source tensors that must NEVER reach the device. The Qwen4-Exp n-gram table is one 28.8 GB tensor in
    // the same mapping as the weights; importing or staging it would flip a ~80 GB model past the resident capacity (the 122B
    // WDDM-thrash class). Survives Reset() (that clears the per-load ledger, not a model-lifetime policy); guarded in TryImport,
    // NoteStaged and VulkanStagingBuffer.UploadBytes so a future caller cannot route the table through any of them by accident.
    private static readonly List<(nint Pointer, long Bytes)> s_hostOnly = [];
    private static readonly object s_hostOnlyLock = new();

    /// <summary>Registers a source range that must never be imported, staged or uploaded. Undo with <see cref="UnregisterHostOnly"/>.</summary>
    public static void RegisterHostOnly(nint pointer, long bytes)
    {
        if (pointer == 0 || bytes <= 0) return;
        lock (s_hostOnlyLock) s_hostOnly.Add((pointer, bytes));
    }

    /// <summary>Removes a range registered with <see cref="RegisterHostOnly"/>.</summary>
    public static void UnregisterHostOnly(nint pointer)
    {
        lock (s_hostOnlyLock) s_hostOnly.RemoveAll(r => r.Pointer == pointer);
    }

    /// <summary>The currently registered host-only ranges.</summary>
    public static IReadOnlyList<(nint Pointer, long Bytes)> HostOnlyRanges
    {
        get { lock (s_hostOnlyLock) return s_hostOnly.ToArray(); }
    }

    /// <summary>Throws when <c>[srcPtr, srcPtr + bytes)</c> overlaps a registered host-only range.</summary>
    public static void ThrowIfHostOnly(nint srcPtr, long bytes)
    {
        if (srcPtr == 0 || bytes <= 0) return;
        lock (s_hostOnlyLock)
        {
            foreach (var (p, n) in s_hostOnly)
                if (srcPtr < p + (nint)n && p < srcPtr + (nint)bytes)
                    throw new InvalidOperationException(
                        $"Refusing to upload {bytes:N0} source bytes at 0x{srcPtr:X}: the range overlaps a host-only tensor " +
                        $"({n:N0} bytes at 0x{p:X}, e.g. the qwen4exp n-gram table), which must stay on the host (#818).");
        }
    }

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
        DirectTensorCount = 0;
        DirectBytes = 0;
        FirstDirectDecline = string.Empty;
        DirectDeclinedCount = 0;
        DirectDeclinedBytes = 0;
        BankStagedBytes = 0;
        BankStagedCount = 0;
        StagedTensorCount = 0;
        StagedBytes = 0;
        RejectedTensorCount = 0;
        RejectedBytes = 0;
        FirstRejection = string.Empty;
        s_reportedRejectedCount = 0;
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
        ThrowIfHostOnly(srcPtr, bytes);

        // #1304 / #1994: the fast tier (default on Windows UMA when dstorage.dll is present). Anything it declines falls through below.
        if (TryDirect(device, srcPtr, bytes, out buf))
            return true;

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

        // #1190: amdvlk and the Arc driver both refuse read-only pages, which is how every GGUF is mapped. Retry from a
        // shared writable view of the same file range (no private pages). A copy-on-write view is also accepted but the
        // driver's write-lock privatises every page, so it costs what staging costs and is not tried.
        // #1991: still opt-in (default-on measured -19% decode / -33% prefill, docs/GPU.md). When on, the file is opened
        // shared-writable but nothing writes to it: weight buffers are bound read-only (VulkanWeightImportWriteGuardTests).
        string aliasFailure = string.Empty;
        bool aliasRequested = wrapped is null && WritableAliasEnabled;
        if (aliasRequested)
        {
            // TryMapImportAlias returns false (never a zero alias) when the range is not in an open read-only mapping, the GGUF is
            // mapped copy-on-write, or the file cannot be opened for write.
            if (!Models.Gguf.GgufFile.TryMapImportAlias(srcPtr, bytes, writable: true, out nint alias))
                aliasFailure = "alias_unavailable(not_in_readonly_mapping_or_file_not_writable)";
            else
            {
                wrapped = device.TryWrapHostVisible(alias, bytes);
                if (wrapped is null)
                    Models.Gguf.GgufFile.ReleaseImportAlias(alias);
            }
        }

        if (wrapped is null)
        {
            // "rejected" on its own is not a cause. Carry the driver's own verdict —
            // which Vulkan call refused and with what VkResult — because the whole
            // import was dead on the real load path for want of exactly this line:
            // synthetic-memory tests passed while every production tensor was refused.
            LastFallbackReason = aliasFailure.Length > 0
                ? aliasFailure
                : $"import_rejected({Interop.HostVisibleBuffer.LastImportFailureStage}" +
                  $":{Interop.HostVisibleBuffer.LastImportFailureCode})";
            // Warn only when the alias was requested (DOTLLM_VULKAN_HOST_IMPORT_WRITABLE=1) and still refused: the default path
            // stays silent (#1995 reports DirectStorage fallbacks instead). A discrete GPU refuses by design (#507): policy, not a refusal.
            if (aliasRequested && device.PhysicalDeviceTypeValue is Interop.VkPhysicalDeviceType.IntegratedGpu or Interop.VkPhysicalDeviceType.Cpu)
            {
                RejectedTensorCount++;
                RejectedBytes += bytes;
                if (FirstRejection.Length == 0) FirstRejection = LastFallbackReason;
            }
            return false;
        }

        NoteImported(bytes);
        buf = wrapped;
        return true;
    }

    /// <summary>
    /// The DirectStorage tier (#1304, default on Windows UMA since #1994): reads the tensor from the file straight into an imported D3D12 heap, i.e.
    /// into device-local memory (no page-cache or process-private duplicate, device-local speed under weight reuse; file-backed host pages
    /// cost ~19% decode / ~33% prefill, #1991). Not an alias of the mmap, so it does not block releasing the whole host mapping (#438).
    /// </summary>
    private static bool TryDirect(VulkanDevice device, nint srcPtr, long bytes, out VulkanDevice.Buffer? buf)
    {
        buf = null;
        if (IsDisabled || bytes < DirectMinTensorBytes || !DirectStorage.DirectWeightStore.EnabledFor(device))
            return false;
        var direct = DirectStorage.DirectWeightStore.For(device);
        if (direct is not null && direct.TryLoad(srcPtr, bytes, out buf))
        {
            DirectTensorCount++;
            DirectBytes += bytes;
            LastFallbackReason = string.Empty;
            return true;
        }
        DirectDeclinedCount++;
        DirectDeclinedBytes += bytes;
        if (FirstDirectDecline.Length == 0)
            FirstDirectDecline = direct is null ? "unavailable:" + DirectStorage.DirectWeightStore.UnavailableReason(device) : direct.LastFallbackReason;
        NoteDirectDeclined(bytes, FirstDirectDecline);
        LastFallbackReason = "direct_storage:" + FirstDirectDecline;
        return false;
    }

    /// <summary>
    /// DirectStorage-only load for a contiguous source range that is not a single weight tensor, namely a fused MoE expert bank
    /// (<c>[E,M,K]</c>, byte-identical to the device layout) (#1994). Never aliases the mmap (that route is slow, #1991). False means the
    /// caller must stage and call <see cref="NoteBankStaged"/>.
    /// </summary>
    public static bool TryLoadDirect(VulkanDevice device, nint srcPtr, long bytes, out VulkanDevice.Buffer? buf)
    {
        ThrowIfHostOnly(srcPtr, bytes);
        return TryDirect(device, srcPtr, bytes, out buf);
    }

    /// <summary>Whether <see cref="TryLoadDirect"/> can run for <paramref name="device"/> (callers skip device-side pre-allocation when it can).</summary>
    public static bool DirectTierActive(VulkanDevice device)
        => !IsDisabled && DirectStorage.DirectWeightStore.EnabledFor(device) && DirectStorage.DirectWeightStore.For(device) is not null;

    /// <summary>Records a MoE expert bank that took the staging copy (its pages stay mapped: the CPU may still use them).</summary>
    public static void NoteBankStaged(long bytes) { BankStagedBytes += bytes; BankStagedCount++; s_modelBankStagedBytes += bytes; }

    /// <summary>Expert-bank bytes copied through staging on the current load (#1994).</summary>
    public static long BankStagedBytes { get; private set; }

    /// <summary>Expert banks copied through staging on the current load.</summary>
    public static int BankStagedCount { get; private set; }

    /// <summary>Bytes of the tensors the DirectStorage tier declined on the current load.</summary>
    public static long DirectDeclinedBytes { get; private set; }

    // Model-level ledger (#1994): unlike the per-loader ledger above, Reset() does not clear it, so a model whose weights are uploaded by
    // several loaders / staging buffers (VulkanWeights twice, per-layer MoE streaming, MTP heads) still reports one total, once.
    private static readonly bool s_traceStaged = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MEM_TRACE") == "1";
    private static long s_modelStagedBytes, s_modelBankStagedBytes, s_modelDeclinedBytes;
    private static int s_modelDeclinedCount;
    private static string s_modelFirstDecline = string.Empty;

    /// <summary>Bytes copied through host memory since <see cref="BeginModelLoad"/> (tensors + expert banks); survives <see cref="Reset"/>.</summary>
    internal static long ModelCopiedBytes => s_modelStagedBytes + s_modelBankStagedBytes;

    /// <summary>Tensors the DirectStorage tier declined during the most recent model load (#2044); a load-summary / bench field.</summary>
    public static int ModelDirectDeclinedTensors => s_modelDeclinedCount;

    /// <summary>Bytes copied through host memory during the most recent model load (#2044); a load-summary / bench field.</summary>
    public static long ModelHostStagedBytes => ModelCopiedBytes;

    /// <summary>Records one tensor the DirectStorage tier declined, in the model-level ledger (#2044).</summary>
    internal static void NoteDirectDeclined(long bytes, string reason)
    {
        s_modelDeclinedCount++; s_modelDeclinedBytes += bytes;
        if (s_modelFirstDecline.Length == 0) s_modelFirstDecline = reason.Length > 0 ? reason : "declined";
    }

    /// <summary>Starts the model-level ledger; call once at the top of a model load (<c>VulkanModelLoader.CreateFromGguf</c>).</summary>
    public static void BeginModelLoad()
    {
        s_modelStagedBytes = s_modelBankStagedBytes = s_modelDeclinedBytes = 0;
        s_modelDeclinedCount = 0;
        s_modelFirstDecline = string.Empty;
    }

    /// <summary>
    /// Ends the model load and returns the one warning line for it, when the Windows UMA DirectStorage default could not be used and weights
    /// were copied through host memory instead (stating the total bytes copied and how to fix it, #1994); empty otherwise.
    /// </summary>
    public static string EndModelLoad(VulkanDevice device)
        => BuildLoadWarning(IsDisabled, DirectStorage.DirectWeightStore.WouldBeDefaultButDllMissing(device),
            s_modelDeclinedCount > 0 && DirectStorage.DirectWeightStore.EnabledFor(device));

    /// <summary>The warning text for the current model ledger; pure over the three device/environment facts so it is testable without a GPU.</summary>
    internal static string BuildLoadWarning(bool disabled, bool dllMissing, bool declinedAndEnabled)
    {
        long total = s_modelStagedBytes + s_modelBankStagedBytes;
        if (disabled || total == 0) return string.Empty;
        if (dllMissing)
            return $"[dotllm] DirectStorage is not installed: {total / (1024.0 * 1024.0):F0} MiB of weights were copied through host memory " +
                   $"({s_modelStagedBytes / (1024.0 * 1024.0):F0} MiB tensors + {s_modelBankStagedBytes / (1024.0 * 1024.0):F0} MiB expert banks). " +
                   (File.Exists(DirectStorage.DirectStorageInstaller.DeclinedMarkerPath)
                       ? $"You declined the download offer; delete {DirectStorage.DirectStorageInstaller.DeclinedMarkerPath} to be asked again, "
                       : "Run `dotllm run` once in a terminal to be offered Microsoft's DirectStorage (fetched from NuGet after you accept its licence), ") +
                   "or place dstorage.dll + dstoragecore.dll next to dotllm / in DOTLLM_DIRECTSTORAGE_DIR (DOTLLM_VULKAN_DIRECT_STORAGE=0 silences this).";
        if (declinedAndEnabled)
            return $"[dotllm] WARNING: DirectStorage declined {s_modelDeclinedCount} tensors ({s_modelDeclinedBytes / (1024.0 * 1024.0):F0} MiB); " +
                   $"{total / (1024.0 * 1024.0):F0} MiB of weights were staged through host memory instead (an extra copy, CLAUDE.md rule 3). " +
                   $"Reason (first): {s_modelFirstDecline}." +
                   (s_modelFirstDecline.StartsWith("d3d12_budget", StringComparison.Ordinal)
                       ? " Raise DOTLLM_VULKAN_DS_BUDGET_PCT (default 75; the device is removed near 84%) to keep more weights on the direct path."
                       : string.Empty);
        return string.Empty;
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
        ThrowIfHostOnly(srcPtr, sourceBytes);
        StagedTensorCount++;
        if (sourceBytes > 0) { StagedBytes += sourceBytes; s_modelStagedBytes += sourceBytes; }
        if (s_traceStaged && sourceBytes >= (1L << 20))   // #1994: which tensors still stage, and why
            Console.Error.WriteLine($"[vulkan-mem] staged {sourceBytes / (1024.0 * 1024.0):F1} MiB reason={reason ?? LastFallbackReason} " +
                string.Join(" < ", Environment.StackTrace.Split(Environment.NewLine).Skip(2).Take(3).Select(l => l.Trim())));
        if (reason is not null) LastFallbackReason = reason;
        if (srcPtr != 0 && sourceBytes > 0)
            s_stagedRanges.Add((srcPtr, sourceBytes));
    }

    /// <summary>One-line ledger for load-time diagnostics and the #508 A/B.</summary>
    public static string Summary()
        => $"host-import: {ImportedTensorCount} tensors / {ImportedBytes / (1024.0 * 1024.0):F1} MiB aliased; " +
           $"{DirectTensorCount} tensors / {DirectBytes / (1024.0 * 1024.0):F1} MiB direct-storage" +
           (DirectDeclinedCount > 0 ? $" ({DirectDeclinedCount} declined, first: {FirstDirectDecline})" : string.Empty) + "; " +
           $"{StagedTensorCount} tensors / {StagedBytes / (1024.0 * 1024.0):F1} MiB staged" +
           (BankStagedCount > 0 ? $" + {BankStagedCount} expert banks / {BankStagedBytes / (1024.0 * 1024.0):F1} MiB staged" : string.Empty) +
           (LastFallbackReason.Length > 0 ? $" (last fallback: {LastFallbackReason})" : string.Empty) +
           $"; whole-mapping release {(MayReleaseWholeHostMapping ? "permitted" : "blocked")}";
}
