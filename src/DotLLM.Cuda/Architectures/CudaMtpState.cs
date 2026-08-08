using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA <see cref="IMtpState"/> implementation: the MTP head's own tiny KV-cache (sized for just
/// the trailing MTP block's attention, not the trunk) plus the pending-hidden-state handoff row
/// and the captured-rows buffer a verify-phase <c>Forward</c> call populates. Mirrors
/// <see cref="DotLLM.Models.Architectures.CpuMtpState"/> (see issue #253 and <see cref="IMtpState"/>
/// for the overall design) with device-resident K/V cache and pending-hidden buffers instead of
/// host <c>NativeMemory</c> — the MTP head's autoregressive draft loop
/// (<see cref="CudaQwen3HybridDenseTransformerModel.ForwardMtp"/>) runs entirely on-device between
/// rounds, so the pending-hidden handoff never round-trips through host memory except at
/// <see cref="SeedFromCapturedRow"/> (once per speculation round, from the host-resident captured
/// rows a verify-phase forward D2H-copies back).
/// </summary>
public sealed class CudaMtpState : IMtpState, IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSteps;
    private readonly int _kvStride; // numKvHeads * headDim

    private nint _keyCacheDevice;     // [maxSteps, kvStride] f32, device-resident
    private nint _valueCacheDevice;   // [maxSteps, kvStride] f32, device-resident
    private nint _pendingHiddenDevice; // [hiddenSize] f32, device-resident — seed for the next ForwardMtp call

    private float[] _capturedRows = []; // host [rowCount, hiddenSize], grown on demand
    private int _capturedRowCount;

    private int _currentLength;
    private bool _disposed;

    /// <summary>Max autoregressive MTP steps this state's KV-cache can hold before needing a reset/rollback.</summary>
    public int MaxSteps => _maxSteps;

    /// <summary>K/V stride (numKvHeads * headDim) — the per-step row width of the device K/V cache.</summary>
    public int KvStride => _kvStride;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _capturedRows.AsSpan(0, _capturedRowCount * _hiddenSize);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedRowCount;

    /// <inheritdoc/>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Creates a fresh MTP state. All device buffers are zero-initialised (an empty MTP KV-cache
    /// and a zero pending-hidden vector — the latter is always overwritten by
    /// <see cref="SeedFromCapturedRow"/> before the first <c>ForwardMtp</c> call in normal use).
    /// </summary>
    /// <param name="hiddenSize">Model hidden size (matches <see cref="ModelConfig.HiddenSize"/>).</param>
    /// <param name="numKvHeads">KV head count for the MTP block's own attention.</param>
    /// <param name="headDim">Per-head dimension for the MTP block's own attention.</param>
    /// <param name="maxSteps">Maximum autoregressive MTP draft steps to size the KV-cache for (typically the max candidate count K).</param>
    public CudaMtpState(int hiddenSize, int numKvHeads, int headDim, int maxSteps)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSteps <= 0) throw new ArgumentOutOfRangeException(nameof(maxSteps));

        _hiddenSize = hiddenSize;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSteps = maxSteps;
        _kvStride = numKvHeads * headDim;

        long kvBytes = (long)maxSteps * _kvStride * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _keyCacheDevice, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _valueCacheDevice, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_keyCacheDevice, 0, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_valueCacheDevice, 0, (nuint)kvBytes).ThrowOnError();

        long hiddenBytes = (long)hiddenSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _pendingHiddenDevice, (nuint)hiddenBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_pendingHiddenDevice, 0, (nuint)hiddenBytes).ThrowOnError();
    }

    /// <summary>Device pointer to the pending-hidden vector ([hiddenSize] f32) that seeds the next <c>ForwardMtp</c> call.</summary>
    internal nint PendingHiddenDevicePtr
    {
        get { ThrowIfDisposed(); return _pendingHiddenDevice; }
    }

    /// <summary>Device pointer to the key-cache buffer, shape <c>[maxSteps, kvStride]</c> row-major.</summary>
    internal nint KeyCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _keyCacheDevice; }
    }

    /// <summary>Device pointer to the value-cache buffer, shape <c>[maxSteps, kvStride]</c> row-major.</summary>
    internal nint ValueCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _valueCacheDevice; }
    }

    /// <summary>Device pointer to the key-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal nint GetKeyRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_maxSteps) throw new ArgumentOutOfRangeException(nameof(step));
        return _keyCacheDevice + (nint)((long)step * _kvStride * sizeof(float));
    }

    /// <summary>Device pointer to the value-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal nint GetValueRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_maxSteps) throw new ArgumentOutOfRangeException(nameof(step));
        return _valueCacheDevice + (nint)((long)step * _kvStride * sizeof(float));
    }

    /// <summary>Advances the MTP KV-cache length by one step after a successful <c>ForwardMtp</c> call.</summary>
    internal void Advance()
    {
        ThrowIfDisposed();
        if (_currentLength >= _maxSteps)
            throw new InvalidOperationException(
                $"CudaMtpState KV-cache exhausted: {_currentLength} steps already advanced against a " +
                $"MaxSteps={_maxSteps} cache. Size the state for at least numCandidates steps.");
        _currentLength++;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ThrowIfDisposed();
        if (length < 0 || length > _currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>
    /// Called by an MTP-supporting model's verify-phase <c>Forward</c> overload to D2H-copy the
    /// captured pre-final-norm hidden rows (a device buffer, e.g. <c>_state.HiddenState</c>) into
    /// this state's host-resident captured-rows buffer for this state's next round. The caller is
    /// responsible for ensuring the source device buffer's writes have completed (stream
    /// synchronized) before calling this — <c>cuMemcpyDtoH_v2</c> does not implicitly wait for a
    /// non-default stream's queued work.
    /// </summary>
    /// <param name="deviceHiddenState">Device pointer to row-major <c>[rowCount, hiddenSize]</c> hidden state rows.</param>
    /// <param name="rowCount">Number of rows to capture.</param>
    internal unsafe void SetCapturedRowsFromDevice(nint deviceHiddenState, int rowCount)
    {
        ThrowIfDisposed();
        int needed = rowCount * _hiddenSize;
        if (_capturedRows.Length < needed)
            _capturedRows = new float[needed];
        fixed (float* p = _capturedRows)
        {
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, deviceHiddenState,
                (nuint)((long)needed * sizeof(float))).ThrowOnError();
        }
        _capturedRowCount = rowCount;
    }

    /// <inheritdoc/>
    public unsafe void SeedFromCapturedRow(int rowIndex)
    {
        ThrowIfDisposed();
        if ((uint)rowIndex >= (uint)_capturedRowCount)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"rowIndex {rowIndex} out of range [0, {_capturedRowCount}) — CapturedHiddenRows was not populated " +
                "by a verify-phase Forward call, or has fewer rows than expected.");

        // H2D: host-captured row -> device pending-hidden buffer. The very next ForwardMtp call
        // consumes _pendingHiddenDevice directly on-device (RMSNorm), so no further round-trip is
        // needed until the NEXT round's SeedFromCapturedRow.
        fixed (float* p = &_capturedRows[rowIndex * _hiddenSize])
        {
            CudaDriverApi.cuMemcpyHtoD_v2(_pendingHiddenDevice, (nint)p,
                (nuint)((long)_hiddenSize * sizeof(float))).ThrowOnError();
        }
    }

    /// <summary>Total bytes allocated for this state's own KV-cache + pending-hidden buffer (device memory).</summary>
    public long AllocatedBytes => (2L * _maxSteps * _kvStride + _hiddenSize) * sizeof(float);

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaMtpState));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_keyCacheDevice != 0) { CudaDriverApi.cuMemFree_v2(_keyCacheDevice); _keyCacheDevice = 0; }
        if (_valueCacheDevice != 0) { CudaDriverApi.cuMemFree_v2(_valueCacheDevice); _valueCacheDevice = 0; }
        if (_pendingHiddenDevice != 0) { CudaDriverApi.cuMemFree_v2(_pendingHiddenDevice); _pendingHiddenDevice = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
