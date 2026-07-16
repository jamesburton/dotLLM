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
/// This type caps the staging allocation at <see cref="MaxChunkBytes"/>
/// (<c>DOTLLM_VULKAN_STAGING_MB</c>, default 64 MiB) and maps it <b>once</b> for its
/// lifetime — uploads larger than the capacity stream through it in bounded chunks
/// (<see cref="UploadBytes"/> / <see cref="UploadRows"/>), each chunk fence-waited by
/// <see cref="Flush"/> before the mapped region is reused. Host commit attributable to
/// staging is therefore bounded by the cap for the whole load, and there are zero
/// re-maps after construction.
/// </para>
/// </remarks>
internal sealed unsafe class VulkanStagingBuffer : IDisposable
{
    /// <summary>Default staging cap: 64 MiB (chosen by the #147 sweep of 32/64/128/256 MiB).</summary>
    public const long DefaultMaxChunkBytes = 64L * 1024 * 1024;

    /// <summary>
    /// Staging cap in bytes. Override with <c>DOTLLM_VULKAN_STAGING_MB</c> (1..4096).
    /// </summary>
    public static long MaxChunkBytes { get; } = ParseChunkBytes();

    private static long ParseChunkBytes()
    {
        string? v = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_STAGING_MB");
        return long.TryParse(v, out long mb) && mb >= 1 && mb <= 4096
            ? mb * 1024 * 1024
            : DefaultMaxChunkBytes;
    }

    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer _buffer;
    private bool _disposed;

    /// <summary>Persistent host pointer to the mapped staging memory (valid until <see cref="Dispose"/>).</summary>
    public nint Mapped { get; }

    /// <summary>Usable staging bytes — <c>min(neededBytes, MaxChunkBytes)</c> at creation.</summary>
    public long Capacity { get; }

    private VulkanStagingBuffer(VulkanDevice device, VulkanDevice.Buffer buffer, nint mapped, long capacity)
    {
        _device = device;
        _buffer = buffer;
        Mapped = mapped;
        Capacity = capacity;
    }

    /// <summary>
    /// Allocates a host-visible staging buffer of <c>min(neededBytes, MaxChunkBytes)</c>
    /// bytes and maps it once. <paramref name="neededBytes"/> is the largest single
    /// upload the caller will push through — smaller models get a smaller buffer.
    /// </summary>
    public static VulkanStagingBuffer Create(VulkanDevice device, long neededBytes)
    {
        long capacity = Math.Max(4096, Math.Min(neededBytes, MaxChunkBytes));
        var buffer = device.Allocate(capacity);
        nint mapped;
        try
        {
            mapped = device.MapMemoryWithRetry(
                buffer.Memory, 0, (ulong)capacity, "vkMapMemory VulkanStagingBuffer (persistent)");
        }
        catch
        {
            buffer.Dispose();
            throw;
        }
        return new VulkanStagingBuffer(device, buffer, mapped, capacity);
    }

    /// <summary>
    /// Synchronously copies the first <paramref name="bytes"/> bytes of the staging
    /// buffer into <paramref name="dst"/> at <paramref name="dstOffset"/> (fence-waited —
    /// the mapped region is reusable on return).
    /// </summary>
    public void Flush(VulkanDevice.Buffer dst, long dstOffset, long bytes)
        => _device.CopyBufferRangeSynchronous(_buffer, dst, srcOffset: 0, dstOffset: (ulong)dstOffset, size: (ulong)bytes);

    /// <summary>
    /// Streams <paramref name="bytes"/> raw bytes from <paramref name="src"/> into
    /// <paramref name="dst"/> at <paramref name="dstOffset"/>, in chunks of at most
    /// <see cref="Capacity"/> bytes.
    /// </summary>
    public void UploadBytes(nint src, long bytes, VulkanDevice.Buffer dst, long dstOffset = 0)
    {
        for (long off = 0; off < bytes;)
        {
            long chunk = Math.Min(Capacity, bytes - off);
            System.Buffer.MemoryCopy((void*)(src + off), (void*)Mapped, Capacity, chunk);
            Flush(dst, dstOffset + off, chunk);
            off += chunk;
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

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        VulkanApi.vkUnmapMemory(_device.Handle, _buffer.Memory);
        _buffer.Dispose();
    }
}
