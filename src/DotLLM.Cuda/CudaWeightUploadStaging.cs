using System.Globalization;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Issue #509 — pinned, chunked, double-buffered host→device staging for the weight-load path.
/// </summary>
/// <remarks>
/// <para>
/// <b>What this replaces.</b> <see cref="CudaWeights"/> uploaded every tensor with a single
/// synchronous <c>cuMemcpyHtoD_v2</c> whose source is the <i>pageable</i> mmap'd GGUF view. There
/// is no dotLLM-side host copy in that path — the driver reads the mmap pointer directly — but a
/// pageable source forces the driver through its own internal bounce buffer, and the transfer is
/// neither chunked nor overlapped. This class stages through a <b>bounded, reused pair</b> of
/// <c>cuMemHostAlloc</c> (page-locked) chunks: while chunk <c>i</c>'s
/// <c>cuMemcpyHtoDAsync_v2</c> is in flight on one stream, the host <c>memcpy</c> for chunk
/// <c>i+1</c> runs into the other chunk.
/// </para>
/// <para>
/// <b>The host-synchronous contract is preserved.</b> <see cref="Upload"/> does not return until
/// (a) every source byte has been copied out of the host source into pinned staging and
/// (b) both staging streams have been synchronised, so the device buffer is fully written. That is
/// exactly what <c>cuMemcpyHtoD_v2</c> guaranteed, and three callers depend on it:
/// <c>CudaWeights.LoadFromGguf</c>'s <c>onHostTensorUploaded</c> hook frees the host source
/// immediately after the copy; <c>UploadAndDequant</c> launches a dequant kernel on the caller's
/// stream that reads the just-uploaded device buffer; and the YaRN inverse-frequency upload passes
/// a <c>fixed</c>-pinned managed array whose pin ends at the end of the statement. Overlap is
/// therefore <i>within</i> a tensor, across its chunks — not across tensors. Cross-tensor overlap
/// would need hazard tracking against the caller's stream and is deliberately not attempted.
/// </para>
/// <para>
/// <b>Why 64 MiB chunks (128 MiB for the pair).</b> On the only link where this can matter
/// (~3.2–3.5 GB/s), a 64 MiB DMA takes ~19 ms while the async-issue plus stream-synchronise
/// overhead is tens of microseconds — under 0.1% fixed cost, so there is nothing to gain by going
/// larger. A single-threaded <c>memcpy</c> of 64 MiB at ~10 GB/s is ~6 ms, comfortably hidden under
/// the DMA it overlaps. Going to 256 MiB would quadruple the page-locked footprint (which the
/// operating system cannot reclaim) for no amortisation benefit, against issue #509's explicit
/// "peak host RAM does not regress" criterion; going much smaller starts exposing the per-chunk
/// fixed cost and shortens the window the memcpy has to hide in.
/// </para>
/// <para>
/// <b>Why the threshold is exactly one chunk.</b> A tensor of at most <see cref="ChunkBytes"/>
/// occupies a single chunk, so there is no second chunk to overlap with: the staged path would be
/// a strictly serial <c>memcpy</c> <i>then</i> DMA, i.e. one extra full host copy versus the plain
/// pageable transfer. Staging only earns its keep from two chunks up, so tensors at or below one
/// chunk keep the simple synchronous path. (This also leaves every small tensor — norms, biases,
/// attention sinks, the YaRN table — on the unchanged path.)
/// </para>
/// <para>
/// <b>Escape hatches</b> (read once, <see cref="CudaSmallSGemvDispatch"/> convention):
/// <c>DOTLLM_CUDA_PINNED_UPLOAD=0</c> forces the legacy synchronous path for every tensor;
/// <c>DOTLLM_CUDA_PINNED_UPLOAD_CHUNK_MIB</c> overrides the chunk size in MiB (clamped to
/// 1..1024). <see cref="TotalStagedBytes"/>, <see cref="TotalStagedChunks"/> and
/// <see cref="TotalDirectBytes"/> are process-wide counters a benchmark reads <i>after</i> a load
/// to prove which path actually ran — never infer it from a support check.
/// </para>
/// </remarks>
public sealed class CudaWeightUploadStaging : IDisposable
{
    /// <summary>Default staging chunk size in bytes (64 MiB); the pair costs twice this.</summary>
    public const long DefaultChunkBytes = 64L * 1024 * 1024;

    /// <summary>Environment variable that, when <c>0</c>, forces the legacy synchronous upload.</summary>
    public const string EnabledEnvVar = "DOTLLM_CUDA_PINNED_UPLOAD";

    /// <summary>Environment variable overriding the chunk size, in MiB (clamped to 1..1024).</summary>
    public const string ChunkMibEnvVar = "DOTLLM_CUDA_PINNED_UPLOAD_CHUNK_MIB";

    /// <summary><c>CU_MEMHOSTALLOC_PORTABLE</c> is not needed; plain page-locked is enough.</summary>
    private const uint HostAllocFlags = 0;

    /// <summary><c>CU_STREAM_NON_BLOCKING</c> — must not serialise against the legacy default stream.</summary>
    private const uint StreamNonBlocking = 1;

    private static readonly bool EnabledFromEnv =
        Environment.GetEnvironmentVariable(EnabledEnvVar) != "0";

    private static readonly long ChunkBytesFromEnv = ReadChunkBytes();

    [ThreadStatic]
    private static CudaWeightUploadStaging? _current;

    private static long _totalStagedBytes;
    private static long _totalStagedChunks;
    private static long _totalDirectBytes;

    private readonly CudaWeightUploadStaging? _previous;
    private readonly long _chunkBytes;
    private readonly long _minStagedBytes;
    private readonly nint[] _chunks = [0, 0];
    private readonly nint[] _streams = [0, 0];
    private bool _disposed;

    private CudaWeightUploadStaging(long chunkBytes, long minStagedBytes, CudaWeightUploadStaging? previous)
    {
        _chunkBytes = chunkBytes;
        _minStagedBytes = minStagedBytes;
        _previous = previous;
    }

    /// <summary>Whether staging is enabled at all (the <see cref="EnabledEnvVar"/> hatch).</summary>
    public static bool Enabled => EnabledOverride ?? EnabledFromEnv;

    /// <summary>Effective chunk size in bytes.</summary>
    public static long ConfiguredChunkBytes => ChunkBytesOverride ?? ChunkBytesFromEnv;

    /// <summary>In-process override of <see cref="Enabled"/> (tests and benches only).</summary>
    internal static bool? EnabledOverride { get; set; }

    /// <summary>In-process override of <see cref="ConfiguredChunkBytes"/> (tests and benches only).</summary>
    internal static long? ChunkBytesOverride { get; set; }

    /// <summary>
    /// In-process override of the "at least this many bytes to be worth staging" threshold, which
    /// otherwise equals the chunk size. Tests set it to 0 so a sub-chunk tensor still exercises the
    /// chunking loop.
    /// </summary>
    internal static long? MinStagedBytesOverride { get; set; }

    /// <summary>The staging scope active on this thread, or <c>null</c> when no load is running.</summary>
    internal static CudaWeightUploadStaging? Current => _current;

    /// <summary>Total bytes this process has pushed through the pinned staged path.</summary>
    public static long TotalStagedBytes => Interlocked.Read(ref _totalStagedBytes);

    /// <summary>Total staging chunks this process has issued (the <c>cuMemcpyHtoDAsync_v2</c> count).</summary>
    public static long TotalStagedChunks => Interlocked.Read(ref _totalStagedChunks);

    /// <summary>Total bytes this process has pushed through the legacy synchronous path.</summary>
    public static long TotalDirectBytes => Interlocked.Read(ref _totalDirectBytes);

    /// <summary>Bytes this scope has pushed through the pinned staged path.</summary>
    public long StagedBytes { get; private set; }

    /// <summary>Staging chunks this scope has issued.</summary>
    public long StagedChunks { get; private set; }

    /// <summary>Bytes this scope has pushed through the legacy synchronous path.</summary>
    public long DirectBytes { get; private set; }

    /// <summary>
    /// <c>true</c> when <c>cuMemHostAlloc</c> (or stream creation) failed, so every upload in this
    /// scope silently fell back to the synchronous path. Not an error: a constrained host that
    /// cannot page-lock 2 x <see cref="ChunkBytes"/> should still load the model.
    /// </summary>
    public bool PinnedAllocFailed { get; private set; }

    /// <summary>Chunk size this scope uses, in bytes.</summary>
    public long ChunkBytes => _chunkBytes;

    /// <summary>Resets the process-wide counters. For test isolation.</summary>
    public static void ResetCounters()
    {
        Interlocked.Exchange(ref _totalStagedBytes, 0);
        Interlocked.Exchange(ref _totalStagedChunks, 0);
        Interlocked.Exchange(ref _totalDirectBytes, 0);
    }

    /// <summary>
    /// Opens a staging scope on the calling thread. <b>Nothing is page-locked here</b> — the chunk
    /// pair is allocated lazily on the first upload that is actually large enough to stage, so a
    /// load made entirely of small tensors, or one run with the escape hatch off, never pins a
    /// byte. Restores the previously-current scope on <see cref="Dispose"/>.
    /// </summary>
    public static CudaWeightUploadStaging BeginScope()
    {
        long chunk = ConfiguredChunkBytes;
        var scope = new CudaWeightUploadStaging(chunk, MinStagedBytesOverride ?? chunk, _current);
        _current = scope;
        return scope;
    }

    /// <summary>Whether a transfer of <paramref name="bytes"/> should go through staging.</summary>
    public bool ShouldStage(long bytes)
        => Enabled && !PinnedAllocFailed && bytes > _minStagedBytes;

    /// <summary>Records a transfer that took the legacy synchronous path.</summary>
    internal void RecordDirect(long bytes)
    {
        DirectBytes += bytes;
        Interlocked.Add(ref _totalDirectBytes, bytes);
    }

    /// <summary>
    /// Copies <paramref name="bytes"/> from <paramref name="hostPtr"/> to <paramref name="devPtr"/>
    /// through the pinned chunk pair. Returns a CUDA status code (0 on success) so the caller can
    /// attach its own VRAM diagnostics; returns <c>-1</c> to mean "staging unavailable, use the
    /// direct path" (the pinned pair could not be created).
    /// </summary>
    /// <remarks>
    /// The fence scheme is one <c>cuStreamSynchronize</c> per chunk against the stream that owns
    /// the chunk about to be overwritten. Chunk <c>i</c> uses slot <c>i % 2</c>; before its host
    /// <c>memcpy</c> we wait on stream <c>i % 2</c>, which retires the DMA out of that same chunk
    /// issued two chunks ago. Chunk <c>i - 1</c>'s DMA, on the <i>other</i> stream, keeps running
    /// underneath that memcpy — that is the overlap. Both streams are synchronised before return,
    /// including on the error path, so no DMA is ever left reading a chunk this object is about to
    /// free.
    /// </remarks>
    public unsafe int Upload(nint devPtr, nint hostPtr, long bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (bytes <= 0) return 0;
        if (!TryEnsureBuffers()) return -1;

        int rc = 0;
        long offset = 0;
        int slot = 0;
        long chunks = 0;

        while (offset < bytes)
        {
            long n = Math.Min(_chunkBytes, bytes - offset);

            // Retire the DMA that last read THIS chunk (issued two iterations ago) before
            // overwriting it. The other slot's copy stays in flight across the memcpy below.
            rc = CudaDriverApi.cuStreamSynchronize(_streams[slot]);
            if (rc != 0) break;

            Buffer.MemoryCopy((void*)(hostPtr + (nint)offset), (void*)_chunks[slot], _chunkBytes, n);

            rc = CudaDriverApi.cuMemcpyHtoDAsync_v2(
                devPtr + (nint)offset, _chunks[slot], (nuint)n, _streams[slot]);
            if (rc != 0) break;

            chunks++;
            offset += n;
            slot ^= 1;
        }

        // Always drain both streams — on the error path too, so neither DMA is still reading a
        // pinned chunk once this returns (Dispose may free them straight after).
        int sync0 = CudaDriverApi.cuStreamSynchronize(_streams[0]);
        int sync1 = CudaDriverApi.cuStreamSynchronize(_streams[1]);
        if (rc == 0) rc = sync0 != 0 ? sync0 : sync1;

        if (rc == 0)
        {
            StagedBytes += bytes;
            StagedChunks += chunks;
            Interlocked.Add(ref _totalStagedBytes, bytes);
            Interlocked.Add(ref _totalStagedChunks, chunks);
        }
        return rc;
    }

    /// <summary>
    /// Lazily creates the pinned chunk pair and the two non-blocking streams. Returns <c>false</c>
    /// (and latches <see cref="PinnedAllocFailed"/>) if the host cannot page-lock the pair, so the
    /// load continues on the synchronous path rather than failing.
    /// </summary>
    private bool TryEnsureBuffers()
    {
        if (_chunks[0] != 0) return true;
        if (PinnedAllocFailed) return false;

        for (int i = 0; i < 2; i++)
        {
            if (CudaDriverApi.cuMemHostAlloc(out nint host, (nuint)_chunkBytes, HostAllocFlags) != 0)
            {
                ReleaseBuffers();
                PinnedAllocFailed = true;
                return false;
            }
            if (CudaDriverApi.cuStreamCreate(out nint stream, StreamNonBlocking) != 0)
            {
                _ = CudaDriverApi.cuMemFreeHost(host);
                ReleaseBuffers();
                PinnedAllocFailed = true;
                return false;
            }
            _chunks[i] = host;
            _streams[i] = stream;
        }
        return true;
    }

    private void ReleaseBuffers()
    {
        for (int i = 0; i < 2; i++)
        {
            if (_streams[i] != 0)
            {
                _ = CudaDriverApi.cuStreamSynchronize(_streams[i]);
                CudaTeardownDiagnostics.RecordDestroy(
                    "CUstream (weight staging)", CudaDriverApi.cuStreamDestroy_v2(_streams[i]));
                _streams[i] = 0;
            }
            if (_chunks[i] != 0)
            {
                CudaTeardownDiagnostics.RecordDestroy(
                    "pinned staging chunk", CudaDriverApi.cuMemFreeHost(_chunks[i]));
                _chunks[i] = 0;
            }
        }
    }

    /// <summary>
    /// Drains both streams, frees the pinned pair and destroys the streams, then restores the
    /// previously-current scope. Deterministic — the page-locked pair must never outlive the load.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        ReleaseBuffers();
        _current = _previous;
    }

    private static long ReadChunkBytes()
    {
        string? raw = Environment.GetEnvironmentVariable(ChunkMibEnvVar);
        if (string.IsNullOrWhiteSpace(raw) ||
            !int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int mib))
        {
            return DefaultChunkBytes;
        }
        return Math.Clamp(mib, 1, 1024) * 1024L * 1024L;
    }
}
