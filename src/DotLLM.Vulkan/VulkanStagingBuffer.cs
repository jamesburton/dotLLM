using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Bounded, persistently-mapped host-visible staging buffer for weight uploads.
/// </summary>
/// <remarks>
/// <para>
/// Replaces the issue-#146 pattern of one giant staging buffer (sized to the largest
/// single upload — the F32-dequantised token-embed table, vocab×hidden×4 ≈ 2.1 GB on an
/// 8B model) that was re-mapped with <c>vkMapMemory</c> for <i>every</i> upload, including
/// KB-scale norm vectors. On Windows/WDDM each map makes the <i>entire</i> allocation
/// host-resident (residency is allocation-granular), so every tiny upload re-charged the
/// full GB-scale commit — the direct trigger of the transient
/// <c>VK_ERROR_MEMORY_MAP_FAILED</c> load flake under memory pressure.
/// </para>
/// <para>
/// This type caps total staging host commit at <see cref="MaxChunkBytes"/>
/// (<c>DOTLLM_VULKAN_STAGING_MB</c>, default 64 MiB) and maps it <b>once</b> for its
/// lifetime — uploads larger than a slot stream through it in bounded chunks
/// (<see cref="UploadBytes"/> / <see cref="UploadRows"/>). Host commit attributable to
/// staging is therefore bounded by the cap for the whole load, and there are zero
/// re-maps after construction.
/// </para>
/// <para>
/// <b>Double buffering (issue #510).</b> The cap is split across
/// <see cref="SlotCount"/> = 2 slots, each with its own command buffer and fence
/// (the second allocated lazily on the first rotation, so single-chunk uploads are
/// unchanged).
/// <see cref="Flush"/> submits the current slot <i>without waiting</i> and rotates to
/// the other, so the host fills slot B while slot A's copy is in flight; a slot is
/// fence-waited only when it comes round again, and <see cref="WaitAll"/> /
/// <see cref="Dispose"/> drain the rest. Before #510 every chunk was submit +
/// <c>vkWaitForFences</c> with nothing in flight across the wait.
/// </para>
/// <para>
/// <b>Consequence for <see cref="Capacity"/>.</b> Because the cap is now split two
/// ways, the per-call writable window is half what it was — 32 MiB at the default cap
/// rather than 64 MiB. That doubles the chunk count for single tensors above 32 MiB
/// (each chunk is now overlapped, so this is not a stall), and moves the
/// <c>fpBytes &lt;= staging.Capacity</c> I2_S whole-tensor dequant branch in
/// <c>VulkanWeights.UploadMatrix</c> into its else arm slightly earlier. Raise
/// <c>DOTLLM_VULKAN_STAGING_MB</c> to restore the old window.
/// </para>
/// <para>
/// <b>Synchronization contract.</b> After <see cref="Flush"/> or
/// <see cref="UploadBytes"/> returns, the copy may still be executing on the GPU.
/// Separate <c>vkQueueSubmit</c> calls on one queue are NOT ordered against each other
/// without a barrier or fence, so any consumer of the destination buffer — a compute
/// dispatch, a <c>Download</c>, or disposing the destination — must be preceded by
/// <see cref="WaitAll"/>. <see cref="Dispose"/> calls it, so the normal
/// <c>using var staging = VulkanStagingBuffer.Create(...)</c> loader shape needs no
/// extra call; only a loader that runs a kernel over just-staged bytes before the
/// staging goes out of scope does (see <c>VulkanWeights.UploadTokenEmbedding</c>).
/// </para>
/// </remarks>
internal sealed unsafe class VulkanStagingBuffer : IDisposable
{
    /// <summary>Default staging cap: 64 MiB (chosen by the #147 sweep of 32/64/128/256 MiB).</summary>
    public const long DefaultMaxChunkBytes = 64L * 1024 * 1024;

    /// <summary>Number of rotating staging slots (issue #510 double buffering).</summary>
    public const int SlotCount = 2;

    /// <summary>
    /// Total staging cap in bytes, across all slots. Override with
    /// <c>DOTLLM_VULKAN_STAGING_MB</c> (1..4096).
    /// </summary>
    public static long MaxChunkBytes { get; } = ParseChunkBytes();

    private static long ParseChunkBytes()
    {
        string? v = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_STAGING_MB");
        return long.TryParse(v, out long mb) && mb >= 1 && mb <= 4096
            ? mb * 1024 * 1024
            : DefaultMaxChunkBytes;
    }

    /// <summary>One rotating slot: mapped host memory + the command buffer/fence that drains it.</summary>
    private sealed class Slot
    {
        public required VulkanDevice.Buffer Buffer { get; init; }
        public required nint Mapped { get; init; }
        public required nint CommandBuffer { get; init; }
        public required nint Fence { get; init; }
        public bool InFlight { get; set; }
    }

    private readonly VulkanDevice _device;
    private readonly Slot?[] _slots;
    private int _current;
    private bool _disposed;

    /// <summary>Persistent host pointer to the CURRENT slot's mapped memory. Re-read it after every
    /// <see cref="Flush"/> — the slot rotates.</summary>
    public nint Mapped => _slots[_current]!.Mapped;

    /// <summary>Usable staging bytes per call — <c>min(neededBytes, MaxChunkBytes / SlotCount)</c>.</summary>
    public long Capacity { get; }

    /// <summary>Diagnostic (issue #510): total <c>vkQueueSubmit</c> calls this staging buffer has made.</summary>
    public long Submits { get; private set; }

    /// <summary>Diagnostic (issue #510): total host <c>vkWaitForFences</c> stalls. Before #510 this
    /// equalled <see cref="Submits"/> by construction; after it, it is still roughly
    /// <c>Submits - 1</c> because the rotation waits on the slot it is about to reuse. The
    /// count is NOT the win — <see cref="FenceWaitMilliseconds"/> is.</summary>
    public long FenceWaits { get; private set; }

    private long _fenceWaitTicks;

    /// <summary>Diagnostic (issue #510): total host time spent blocked in
    /// <c>vkWaitForFences</c>. This is the number #510 moves: the wait COUNT barely
    /// changes, but each wait now finds a copy that has been running since the previous
    /// chunk's memcpy started, instead of one that was submitted moments ago.</summary>
    public double FenceWaitMilliseconds => _fenceWaitTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

    private VulkanStagingBuffer(VulkanDevice device, Slot?[] slots, long capacity)
    {
        _device = device;
        _slots = slots;
        Capacity = capacity;
    }

    /// <summary>
    /// Allocates a host-visible staging slot of
    /// <c>min(neededBytes, MaxChunkBytes / SlotCount)</c> bytes and maps it once.
    /// <paramref name="neededBytes"/> is the largest single upload the caller will push
    /// through — smaller models get smaller slots.
    /// </summary>
    /// <remarks>
    /// Only the FIRST slot is built here. The second is allocated on the first rotation,
    /// so a staging buffer that only ever does one <see cref="Flush"/> costs exactly what
    /// it did before #510 — which matters because <c>VulkanQwen3MoeMoeUpload</c> creates
    /// and destroys one of these per MoE layer per forward on the non-resident streaming
    /// decode path, where an extra unused allocation + map would be a per-token cost.
    /// </remarks>
    public static VulkanStagingBuffer Create(VulkanDevice device, long neededBytes)
    {
        long capacity = Math.Max(4096, Math.Min(neededBytes, MaxChunkBytes / SlotCount));
        var slots = new Slot?[SlotCount];
        slots[0] = CreateSlot(device, capacity);
        return new VulkanStagingBuffer(device, slots, capacity);
    }

    private static Slot CreateSlot(VulkanDevice device, long capacity)
    {
        var buffer = device.Allocate(capacity);
        nint mapped;
        nint cmd = 0;
        nint fence = 0;
        try
        {
            mapped = device.MapMemoryWithRetry(
                buffer.Memory, 0, (ulong)capacity, "vkMapMemory VulkanStagingBuffer (persistent)");
            cmd = device.AllocateTransferCommandBuffer();
            fence = device.CreateUnsignalledFence();
        }
        catch
        {
            device.FreeTransferCommandBuffer(cmd);
            device.DestroyOwnedFence(fence);
            buffer.Dispose();
            throw;
        }
        return new Slot
        {
            Buffer = buffer,
            Mapped = mapped,
            CommandBuffer = cmd,
            Fence = fence,
        };
    }

    private static void DestroySlot(VulkanDevice device, Slot slot)
    {
        VulkanApi.vkUnmapMemory(device.Handle, slot.Buffer.Memory);
        device.FreeTransferCommandBuffer(slot.CommandBuffer);
        device.DestroyOwnedFence(slot.Fence);
        slot.Buffer.Dispose();
    }

    /// <summary>
    /// Queues a copy of the first <paramref name="bytes"/> bytes of the CURRENT staging
    /// slot into <paramref name="dst"/> at <paramref name="dstOffset"/> and rotates to
    /// the next slot. <b>Does not wait</b> — see the synchronization contract on the
    /// type. On return, <see cref="Mapped"/> points at a slot that is safe to write.
    /// </summary>
    public void Flush(VulkanDevice.Buffer dst, long dstOffset, long bytes)
    {
        if (bytes <= 0) return;
        var slot = _slots[_current]!;
        _device.SubmitCopyDeferred(
            slot.CommandBuffer, slot.Fence, slot.Buffer, dst,
            srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);
        slot.InFlight = true;
        Submits++;

        _current = (_current + 1) % SlotCount;
        var next = _slots[_current] ??= CreateSlot(_device, Capacity);
        if (next.InFlight)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _device.WaitAndResetFence(next.Fence);
            _fenceWaitTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            next.InFlight = false;
            FenceWaits++;
        }
    }

    /// <summary>
    /// Host-waits for every queued copy to complete. Required before any consumer of a
    /// destination buffer (compute dispatch, download, disposal) and called by
    /// <see cref="Dispose"/>.
    /// </summary>
    public void WaitAll()
    {
        for (int i = 0; i < _slots.Length; i++)
        {
            if (_slots[i] is not { InFlight: true } slot) continue;
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _device.WaitAndResetFence(slot.Fence);
            _fenceWaitTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            slot.InFlight = false;
            FenceWaits++;
        }
    }

    /// <summary>
    /// Streams <paramref name="bytes"/> raw bytes from <paramref name="src"/> into
    /// <paramref name="dst"/> at <paramref name="dstOffset"/>, in chunks of at most
    /// <see cref="Capacity"/> bytes, optionally followed by
    /// <paramref name="zeroTailBytes"/> zero bytes.
    /// </summary>
    /// <remarks>
    /// The zero tail is written into the same staging slot directly after the final
    /// data chunk and copied in the SAME region (issue #510) — the packed-SSBO
    /// round-up pad of every Q8_0 / Q6_K tensor with an odd block count used to cost a
    /// separate command buffer, submit and full host stall to move ≤ 3 bytes.
    /// </remarks>
    public void UploadBytes(nint src, long bytes, VulkanDevice.Buffer dst, long dstOffset = 0, long zeroTailBytes = 0)
    {
        var plan = new VulkanStagingChunkPlan(bytes, zeroTailBytes, Capacity);
        while (plan.MoveNext())
        {
            nint mapped = Mapped;
            if (plan.Length > 0)
                System.Buffer.MemoryCopy((void*)(src + (nint)plan.SrcOffset), (void*)mapped, Capacity, plan.Length);
            if (plan.ZeroTail > 0)
                new Span<byte>((void*)(mapped + (nint)plan.Length), (int)plan.ZeroTail).Clear();
            Flush(dst, dstOffset + plan.SrcOffset, plan.Length + plan.ZeroTail);
        }
    }

    /// <summary>
    /// Uploads a managed float span (norm/bias vectors, small F32 tables) into
    /// <paramref name="dst"/> at <paramref name="dstOffset"/>, chunked when needed.
    /// </summary>
    public void UploadFloats(ReadOnlySpan<float> src, VulkanDevice.Buffer dst, long dstOffset = 0)
    {
        int elemsPerChunk = (int)Math.Min(src.Length, Capacity / sizeof(float));
        for (int e = 0; e < src.Length; e += elemsPerChunk)
        {
            int n = Math.Min(elemsPerChunk, src.Length - e);
            src.Slice(e, n).CopyTo(new Span<float>((void*)Mapped, n));
            Flush(dst, dstOffset + (long)e * sizeof(float), (long)n * sizeof(float));
        }
    }

    /// <summary>
    /// Writes destination rows <c>[firstRow, firstRow+rowCount)</c> at the given staging
    /// pointer. Each destination row is <c>dstRowBytes</c> long; the writer produces the
    /// rows contiguously starting at <c>chunkPtr</c>.
    /// </summary>
    public delegate void RowChunkWriter(nint chunkPtr, long firstRow, int rowCount);

    /// <summary>
    /// Streams a row-major tensor of <paramref name="rowCount"/> rows ×
    /// <paramref name="dstRowBytes"/> bytes into <paramref name="dst"/> at
    /// <paramref name="dstOffset"/>, invoking <paramref name="writer"/> to produce each
    /// bounded chunk of rows (transform/dequant happens directly into mapped staging —
    /// no intermediate host buffer).
    /// </summary>
    public void UploadRows(long rowCount, long dstRowBytes, VulkanDevice.Buffer dst, long dstOffset, RowChunkWriter writer)
    {
        if (dstRowBytes > Capacity)
            throw new ArgumentOutOfRangeException(nameof(dstRowBytes),
                $"Row of {dstRowBytes} bytes exceeds the staging capacity {Capacity} " +
                "(raise DOTLLM_VULKAN_STAGING_MB).");
        long rowsPerChunk = Math.Max(1, Capacity / dstRowBytes);
        for (long r = 0; r < rowCount; r += rowsPerChunk)
        {
            int n = (int)Math.Min(rowsPerChunk, rowCount - r);
            writer(Mapped, r, n);
            Flush(dst, dstOffset + r * dstRowBytes, n * dstRowBytes);
        }
    }

    /// <summary>
    /// <c>DOTLLM_VULKAN_MEM_TRACE=1</c> also prints the #508 import ledger and the #510
    /// submit/stall counters when the staging buffer is torn down — i.e. at the end of
    /// each weights load. Without this the numbers both issues are measured by are
    /// unobservable from outside the process.
    /// </summary>
    private static readonly bool s_trace =
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MEM_TRACE"), "1", StringComparison.Ordinal);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        WaitAll();
        if (s_trace)
        {
            Console.Error.WriteLine(
                $"[vulkan-mem] staging({Capacity / (1024 * 1024)} MiB x{SlotCount}) " +
                $"submits={Submits} fenceWaits={FenceWaits} stall={FenceWaitMilliseconds:F1} ms; " +
                VulkanWeightImportPolicy.Summary());
        }
        for (int i = 0; i < _slots.Length; i++)
            if (_slots[i] is { } slot) DestroySlot(_device, slot);
    }
}
