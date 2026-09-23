namespace DotLLM.Vulkan;

/// <summary>
/// Pure, allocation-free plan for streaming <c>bytes</c> source bytes (plus an
/// optional zero-filled tail) through a bounded staging slot of
/// <c>capacity</c> bytes.
/// </summary>
/// <remarks>
/// <para>
/// Extracted as a value-type cursor so the chunking arithmetic of
/// <see cref="VulkanStagingBuffer.UploadBytes"/> is testable without a Vulkan
/// device (issue #510). The property the tests exist to pin is the
/// <b>region count</b>: the zero tail is folded into the LAST data chunk, so a
/// padded upload costs exactly <c>ceil(bytes / limit)</c> submits — not one
/// more. Before #510 the 0–3 pad bytes that every Q8_0 (34 B/block) and Q6_K
/// (210 B/block) tensor of odd block count needs were a second
/// <see cref="VulkanStagingBuffer.UploadBytes"/> call, i.e. a whole extra
/// command buffer, submit and full host fence stall to move ≤ 3 bytes.
/// </para>
/// <para>
/// Folding is possible in a single <c>VkBufferCopy</c> region rather than the
/// region array the issue proposed, because the pad is written into the
/// staging slot immediately after the data: source and pad are contiguous in
/// the staging allocation, so one region covers both. The cost is that the
/// data chunk limit shrinks by the pad size, which is why
/// <see cref="Limit"/> is capacity minus the tail.
/// </para>
/// </remarks>
internal struct VulkanStagingChunkPlan
{
    private readonly long _bytes;
    private readonly long _zeroTailBytes;
    private long _offset;
    private bool _emittedEmpty;

    /// <summary>Maximum data bytes per chunk — the slot capacity less the zero tail.</summary>
    public long Limit { get; }

    /// <summary>Source offset of the chunk the cursor is on. Valid after <see cref="MoveNext"/> returned true.</summary>
    public long SrcOffset { get; private set; }

    /// <summary>Data-byte count of the current chunk.</summary>
    public long Length { get; private set; }

    /// <summary>Zero bytes appended to the current chunk (non-zero only on the last one).</summary>
    public long ZeroTail { get; private set; }

    /// <summary>
    /// Creates a cursor for <paramref name="bytes"/> source bytes followed by
    /// <paramref name="zeroTailBytes"/> zero bytes, through a slot of
    /// <paramref name="capacity"/> bytes.
    /// </summary>
    public VulkanStagingChunkPlan(long bytes, long zeroTailBytes, long capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        ArgumentOutOfRangeException.ThrowIfNegative(zeroTailBytes);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(capacity, zeroTailBytes);

        _bytes = bytes;
        _zeroTailBytes = zeroTailBytes;
        Limit = capacity - zeroTailBytes;
        _offset = 0;
        _emittedEmpty = false;
        SrcOffset = 0;
        Length = 0;
        ZeroTail = 0;
    }

    /// <summary>Total number of chunks (= submits) this plan will produce.</summary>
    public readonly long ChunkCount
        => _bytes > 0
            ? (_bytes + Limit - 1) / Limit
            : (_zeroTailBytes > 0 ? 1 : 0);

    /// <summary>Advances to the next chunk; false when the plan is exhausted.</summary>
    public bool MoveNext()
    {
        if (_offset >= _bytes)
        {
            // Degenerate: nothing but a zero tail (bytes == 0, pad > 0).
            if (_bytes == 0 && _zeroTailBytes > 0 && !_emittedEmpty)
            {
                _emittedEmpty = true;
                SrcOffset = 0;
                Length = 0;
                ZeroTail = _zeroTailBytes;
                return true;
            }
            return false;
        }

        SrcOffset = _offset;
        Length = Math.Min(Limit, _bytes - _offset);
        _offset += Length;
        ZeroTail = _offset >= _bytes ? _zeroTailBytes : 0;
        return true;
    }
}
