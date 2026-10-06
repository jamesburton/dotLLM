using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// A row-major F32 table (token embedding) resident on the device as one or more row-chunk
/// buffers, each no larger than the device's <c>maxStorageBufferRange</c> (issue #772).
/// </summary>
/// <remarks>
/// <para>
/// The ordinary widened embedding table is a single buffer gathered with <c>vkCmdCopyBuffer</c>
/// byte offsets. Qwen3.8-27B's <c>token_embd</c> [5120, 248320] widens to 5,085,593,600 bytes,
/// which cannot be allocated as one buffer on a device whose <c>maxStorageBufferRange</c> is
/// 4 GiB - 1. Splitting on <b>row</b> boundaries keeps every row contiguous in one chunk, so the
/// gather stays a plain per-token buffer-to-buffer copy: no new shader, and the result is
/// byte-identical to the unchunked path. When the whole table fits there is exactly one chunk
/// and behaviour is unchanged.
/// </para>
/// <para>
/// The LM head is not routed through here: it stays in its packed quantized form (Q4_K/Q6_K),
/// which is well under the limit, and its GEMV/GEMM kernels bind it whole.
/// </para>
/// </remarks>
internal sealed class VulkanChunkedRowTable : IDisposable
{
    /// <summary>
    /// Test hook: when non-null, replaces the device's real <c>maxStorageBufferRange</c> when
    /// planning chunks, so a tiny synthetic table can exercise the multi-chunk path. Tests that
    /// set it run in the serialized <c>VulkanKernels</c> collection.
    /// </summary>
    internal static ulong? LimitOverrideBytes { get; set; }

    /// <summary>Test/diagnostic counter: row copies recorded from a chunk other than chunk 0.</summary>
    internal static long NonFirstChunkCopies;

    private readonly VulkanDevice.Buffer[] _chunks;

    /// <summary>Total number of rows.</summary>
    public long Rows { get; }

    /// <summary>Bytes per row (F32).</summary>
    public long RowBytes { get; }

    /// <summary>Rows held by every chunk but possibly the last.</summary>
    public long RowsPerChunk { get; }

    /// <summary>Number of chunk buffers.</summary>
    public int ChunkCount => _chunks.Length;

    /// <summary>Total device bytes across all chunks.</summary>
    public long TotalBytes => Rows * RowBytes;

    private VulkanChunkedRowTable(VulkanDevice.Buffer[] chunks, long rows, long rowBytes, long rowsPerChunk)
    {
        _chunks = chunks;
        Rows = rows;
        RowBytes = rowBytes;
        RowsPerChunk = rowsPerChunk;
    }

    /// <summary>The limit chunk planning uses for <paramref name="device"/> (override, else queried).</summary>
    internal static ulong EffectiveLimit(VulkanDevice device)
        => LimitOverrideBytes ?? device.MaxStorageBufferRange;

    /// <summary>
    /// Rows per chunk for a table of <paramref name="rows"/> rows of <paramref name="rowBytes"/>
    /// bytes under <paramref name="limitBytes"/>; a <c>0</c> limit means "unreadable, do not split".
    /// </summary>
    internal static long PlanRowsPerChunk(long rows, long rowBytes, ulong limitBytes)
    {
        if (limitBytes == 0 || (ulong)(rows * rowBytes) <= limitBytes) return rows;
        long perChunk = (long)(limitBytes / (ulong)rowBytes);
        if (perChunk < 1)
            throw new NotSupportedException(
                $"A single table row of {rowBytes} bytes exceeds the device's maxStorageBufferRange ({limitBytes}).");
        return perChunk;
    }

    /// <summary>
    /// Widens <paramref name="rows"/> x <paramref name="cols"/> source rows (any quantization the
    /// projection-upload helper understands) to F32 and uploads them as row chunks.
    /// </summary>
    internal static VulkanChunkedRowTable Create(
        VulkanDevice device, VulkanStagingBuffer staging,
        nint srcPtr, QuantizationType qt, int rows, int cols)
    {
        long rowBytes = (long)cols * sizeof(float);
        long perChunk = PlanRowsPerChunk(rows, rowBytes, EffectiveLimit(device));
        int count = checked((int)((rows + perChunk - 1) / perChunk));
        var chunks = new VulkanDevice.Buffer[count];
        long srcRowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(cols, qt);
        try
        {
            for (int c = 0; c < count; c++)
            {
                long first = c * perChunk;
                int n = (int)Math.Min(perChunk, rows - first);
                // Rows are contiguous in the source, so a chunk is the same upload with the
                // source pointer advanced to its first row.
                chunks[c] = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging,
                    srcPtr + (nint)(first * srcRowBytes), qt, n, cols,
                    forceF32: true, out _, out _);
            }
        }
        catch
        {
            for (int c = 0; c < count; c++) chunks[c]?.Dispose();
            throw;
        }
        return new VulkanChunkedRowTable(chunks, rows, rowBytes, perChunk);
    }

    /// <summary>Records a copy of row <paramref name="row"/> into <paramref name="dst"/> at <paramref name="dstOffset"/>.</summary>
    internal void RecordRowCopy(nint cmdBuf, long row, VulkanDevice.Buffer dst, long dstOffset)
    {
        if ((ulong)row >= (ulong)Rows)
            throw new ArgumentOutOfRangeException(nameof(row), $"Row {row} is out of range [0, {Rows}).");
        long chunk = row / RowsPerChunk;
        if (chunk != 0) Interlocked.Increment(ref NonFirstChunkCopies);
        var region = new VkBufferCopy
        {
            srcOffset = (ulong)((row - chunk * RowsPerChunk) * RowBytes),
            dstOffset = (ulong)dstOffset,
            size = (ulong)RowBytes,
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, _chunks[chunk].Handle, dst.Handle, 1, region);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _chunks.Length; i++) _chunks[i].Dispose();
    }
}
