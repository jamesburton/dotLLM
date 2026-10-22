using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

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
    private int _maxSteps;            // hard ceiling on positions (the trunk KV-cache's MaxLength); raised by a resume (issue #1088)
    private int _capacity;            // rows currently allocated, <= _maxSteps; grows on demand (issue #966)
    private readonly int _kvStride; // numKvHeads * headDim

    private nint _keyCacheDevice;     // [capacity, kvStride] f32, device-resident
    private nint _valueCacheDevice;   // [capacity, kvStride] f32, device-resident
    private nint _pendingHiddenDevice; // [hiddenSize] f32, device-resident — seed for the next ForwardMtp call
    private nint _positionIotaDevice;  // [capacity] int32, device-resident — element p holds p (issue #482)

    private float[] _capturedRows = []; // host [rowCount, hiddenSize], grown on demand

    // Host copy of the trunk hidden state at the last absorbed position (issue #469): the
    // h_{p-1} the next absorb pairs its first token with. Drafting rewrites the device pending
    // hidden with the head's own output; this survives it.
    private float[] _carryHidden = [];
    private int _capturedRowCount;

    private int _currentLength;
    private bool _disposed;

    // Issue #1068: admission for the lazy KV growth (null = unchecked, for direct test construction).
    private readonly Action<long, string>? _admitGrowth;

    /// <summary>
    /// Hard ceiling on the positions this state can hold (the <c>maxSteps</c> it was created with).
    /// Exceeding it throws <see cref="InvalidOperationException"/> before any device write.
    /// </summary>
    public int MaxSteps => _maxSteps;

    /// <summary>Positions the device buffers currently hold; grows in <see cref="MtpKvCapacity.GrowthStep"/> steps up to <see cref="MaxSteps"/>.</summary>
    public int Capacity => _capacity;

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
    /// <param name="admitGrowth">
    /// Called with the bytes and purpose of each lazy KV growth before it allocates (issue #1068); it throws to refuse. The owning
    /// model passes its free-VRAM admission so the head's cache cannot grow into shared system memory. Null skips the check.
    /// </param>
    public CudaMtpState(int hiddenSize, int numKvHeads, int headDim, int maxSteps, Action<long, string>? admitGrowth = null)
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
        _admitGrowth = admitGrowth;

        // Lazy KV (issue #966): allocate min(ceiling, 1024) positions now and grow as positions are
        // reached. An F32 head cache is 8 KiB/position on Bonsai 2 (4 KV heads x 256 dim), i.e. 512 MiB
        // at 64k, which an eager allocation reserved even for a short prompt.
        _capacity = MtpKvCapacity.Initial(maxSteps);
        long kvBytes = (long)_capacity * _kvStride * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _keyCacheDevice, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out _valueCacheDevice, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_keyCacheDevice, 0, (nuint)kvBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_valueCacheDevice, 0, (nuint)kvBytes).ThrowOnError();

        long hiddenBytes = (long)hiddenSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _pendingHiddenDevice, (nuint)hiddenBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_pendingHiddenDevice, 0, (nuint)hiddenBytes).ThrowOnError();

        // Positions as a device-resident iota (issue #482): a draft step's RoPE and a batched absorb's
        // contiguous positions are slices of it, so neither uploads positions per call. Grows with the KV.
        UploadIota(_capacity, out _positionIotaDevice);
    }

    /// <summary>Allocates an iota table of <paramref name="count"/> int32 rows (values 0..count-1) on the device.</summary>
    private static unsafe void UploadIota(int count, out nint device)
    {
        int[] iota = new int[count];
        for (int i = 0; i < count; i++) iota[i] = i;
        long iotaBytes = (long)count * sizeof(int);
        CudaDriverApi.cuMemAlloc_v2(out device, (nuint)iotaBytes).ThrowOnError();
        fixed (int* p = iota)
            CudaDriverApi.cuMemcpyHtoD_v2(device, (nint)p, (nuint)iotaBytes).ThrowOnError();
    }

    /// <summary>
    /// Makes the device buffers cover at least <paramref name="positions"/> rows (issue #966), growing in
    /// <see cref="MtpKvCapacity.GrowthStep"/> steps up to <see cref="MaxSteps"/> and preserving the
    /// committed rows <c>[0, CurrentLength)</c>. Must run before any pointer into the buffers is derived
    /// for a write at <c>positions - 1</c>; no caller caches those pointers across calls. Throws
    /// <see cref="InvalidOperationException"/> above the ceiling: never a clamp, never an
    /// out-of-range device access.
    /// </summary>
    /// <param name="positions">Rows required (last position + 1).</param>
    /// <param name="stream">The stream the head's kernels run on: drained before the buffers are swapped.</param>
    internal void EnsureCapacity(int positions, nint stream)
    {
        ThrowIfDisposed();
        int target = MtpKvCapacity.Next(_capacity, positions, _maxSteps);
        if (target == _capacity)
            return;

        // Drain in-flight work that may still read/write the old buffers, then swap. The copy covers
        // only committed rows; speculative rows beyond CurrentLength are rewritten before they are read.
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        long newKvBytes = (long)target * _kvStride * sizeof(float);
        long keepBytes = (long)Math.Min(_currentLength, _capacity) * _kvStride * sizeof(float);
        // Old and new buffers coexist during the copy, so the whole new size is admitted. A refusal leaves the state valid.
        _admitGrowth?.Invoke(2 * newKvBytes + (long)target * sizeof(int), $"MTP head KV cache for {target} positions");
        nint newKey = 0, newValue = 0, newIota = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out newKey, (nuint)newKvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out newValue, (nuint)newKvBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(newKey, 0, (nuint)newKvBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(newValue, 0, (nuint)newKvBytes).ThrowOnError();
            if (keepBytes > 0)
            {
                CudaDriverApi.cuMemcpyDtoD_v2(newKey, _keyCacheDevice, (nuint)keepBytes).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoD_v2(newValue, _valueCacheDevice, (nuint)keepBytes).ThrowOnError();
            }
            UploadIota(target, out newIota);
        }
        catch
        {
            if (newKey != 0) CudaDriverApi.cuMemFree_v2(newKey);
            if (newValue != 0) CudaDriverApi.cuMemFree_v2(newValue);
            if (newIota != 0) CudaDriverApi.cuMemFree_v2(newIota);
            throw;   // the old buffers are untouched: the state stays valid at its previous capacity
        }

        CudaDriverApi.cuMemFree_v2(_keyCacheDevice);
        CudaDriverApi.cuMemFree_v2(_valueCacheDevice);
        CudaDriverApi.cuMemFree_v2(_positionIotaDevice);
        _keyCacheDevice = newKey;
        _valueCacheDevice = newValue;
        _positionIotaDevice = newIota;
        _capacity = target;
    }

    /// <summary>
    /// Device pointer to <paramref name="count"/> contiguous int32 positions starting at
    /// <paramref name="position"/> (<c>[position, position + count)</c>), read-only — a slice of a
    /// device iota table filled at construction.
    /// </summary>
    internal nint GetPositionDevicePtr(int position, int count = 1)
    {
        ThrowIfDisposed();
        if (position < 0 || count < 0 || (long)position + count > _capacity)
            throw new ArgumentOutOfRangeException(nameof(position),
                $"positions [{position}, {(long)position + count}) exceed the allocated capacity {_capacity} (MaxSteps={_maxSteps}).");
        return _positionIotaDevice + (nint)((long)position * sizeof(int));
    }

    /// <summary>Device pointer to the pending-hidden vector ([hiddenSize] f32) that seeds the next <c>ForwardMtp</c> call.</summary>
    internal nint PendingHiddenDevicePtr
    {
        get { ThrowIfDisposed(); return _pendingHiddenDevice; }
    }

    /// <summary>Device pointer to the key-cache buffer, shape <c>[Capacity, kvStride]</c> row-major; re-fetch after any call that may grow it.</summary>
    internal nint KeyCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _keyCacheDevice; }
    }

    /// <summary>Device pointer to the value-cache buffer, shape <c>[Capacity, kvStride]</c> row-major; re-fetch after any call that may grow it.</summary>
    internal nint ValueCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _valueCacheDevice; }
    }

    /// <summary>Device pointer to the key-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal nint GetKeyRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return _keyCacheDevice + (nint)((long)step * _kvStride * sizeof(float));
    }

    /// <summary>Device pointer to the value-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal nint GetValueRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return _valueCacheDevice + (nint)((long)step * _kvStride * sizeof(float));
    }

    /// <summary>Advances the MTP KV-cache length by one step after a successful <c>ForwardMtp</c> call.</summary>
    internal void Advance()
    {
        ThrowIfDisposed();
        if (_currentLength >= _maxSteps)
            throw new InvalidOperationException(
                $"CudaMtpState KV-cache exhausted: {_currentLength} steps already advanced against a " +
                $"MaxSteps={_maxSteps} cache. Size the state for the whole sequence (prompt + generated + draft steps).");
        _currentLength++;
    }

    /// <summary>
    /// Prepares a batched absorb of <paramref name="count"/> contiguous positions starting at
    /// <paramref name="firstPosition"/> (issue #472, ported in #478): rolls speculative slots back to
    /// it, rejects a gap, and bounds-checks the whole slab before anything is written. Mirrors
    /// <see cref="DotLLM.Models.Architectures.CpuMtpState"/>.
    /// </summary>
    internal void BeginAbsorb(int firstPosition, int count, nint stream)
    {
        ThrowIfDisposed();
        if (_currentLength > firstPosition)
            _currentLength = firstPosition;
        else if (_currentLength < firstPosition)
            throw new InvalidOperationException(
                $"MTP absorb at position {firstPosition} but the MTP KV-cache only covers {_currentLength} " +
                "positions. Every trunk Forward of the sequence must pass the MTP state so the head " +
                "absorbs it (prefill included).");
        if ((long)firstPosition + count > _maxSteps)
            throw new InvalidOperationException(
                $"CudaMtpState KV-cache exhausted absorbing positions [{firstPosition}, {firstPosition + count}) " +
                $"(MaxSteps={_maxSteps}). Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");
        EnsureCapacity(firstPosition + count, stream);   // issue #966: lazy growth, after the ceiling check
    }

    /// <summary>Completes a batched absorb begun by <see cref="BeginAbsorb"/>: the cache now covers <paramref name="length"/> positions.</summary>
    internal void EndAbsorb(int length)
    {
        ThrowIfDisposed();
        if (length < _currentLength || length > _maxSteps)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>
    /// Writes the hidden-state rows that batch rows <c>[startRow, startRow + count)</c> pair with
    /// (issue #469): row 0 is the carried row (the trunk hidden of the position before the batch —
    /// zero when nothing has been absorbed yet), row <c>i</c> is captured row <c>i - 1</c>.
    /// <paramref name="dest"/> is <c>[count, hiddenSize]</c>. Exactly the rows the per-token absorb
    /// uploads through <see cref="SetPendingFromCarry"/> / <see cref="SetPendingFromCapturedRow"/>.
    /// </summary>
    internal void CopyAbsorbPairingRows(int startRow, int count, Span<float> dest)
    {
        ThrowIfDisposed();
        if (startRow < 0 || count < 0 || startRow + count > _capturedRowCount)
            throw new ArgumentOutOfRangeException(nameof(count));
        if (dest.Length < count * _hiddenSize)
            throw new ArgumentException("dest too small.", nameof(dest));
        for (int i = 0; i < count; i++)
        {
            int row = startRow + i;
            var d = dest.Slice(i * _hiddenSize, _hiddenSize);
            if (row > 0)
                _capturedRows.AsSpan((row - 1) * _hiddenSize, _hiddenSize).CopyTo(d);
            else if (_carryHidden.Length == _hiddenSize)
                _carryHidden.CopyTo(d);
            else
                d.Clear();   // nothing absorbed yet: h_{-1} is zero (matches SetPendingFromCarry)
        }
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
        if (_carryHidden.Length != _hiddenSize)
            _carryHidden = new float[_hiddenSize];
        _capturedRows.AsSpan(rowIndex * _hiddenSize, _hiddenSize).CopyTo(_carryHidden);
        UploadPending(_capturedRows.AsSpan(rowIndex * _hiddenSize, _hiddenSize));
    }

    /// <summary>Seeds the device pending hidden from the carried (last absorbed) trunk row.</summary>
    internal void SetPendingFromCarry()
    {
        ThrowIfDisposed();
        if (_carryHidden.Length != _hiddenSize)
            _carryHidden = new float[_hiddenSize];   // nothing absorbed yet: h_{-1} is zero
        UploadPending(_carryHidden);
    }

    /// <summary>Seeds the device pending hidden from captured row <paramref name="row"/> without moving the carry.</summary>
    internal void SetPendingFromCapturedRow(int row)
    {
        ThrowIfDisposed();
        UploadPending(_capturedRows.AsSpan(row * _hiddenSize, _hiddenSize));
    }

    private unsafe void UploadPending(ReadOnlySpan<float> row)
    {
        fixed (float* p = row)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(_pendingHiddenDevice, (nint)p,
                (nuint)((long)_hiddenSize * sizeof(float))).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public object? MarkPrefix()
    {
        ThrowIfDisposed();
        // Nothing absorbed yet: h_{-1} is zero (matches SetPendingFromCarry).
        float[] carry = _carryHidden.Length == _hiddenSize ? (float[])_carryHidden.Clone() : new float[_hiddenSize];
        return new MtpPrefixMark(this, _currentLength, carry);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The device K/V rows below the mark are the ones the marked request absorbed: nothing writes below a position it
    /// absorbs from, and a lazy growth copies the committed rows. Only host-side fields change here, so no CUDA context
    /// needs to be current: the next trunk forward's absorb pairs its first row with the restored carry and then seeds
    /// the device pending hidden itself (<see cref="SeedFromCapturedRow"/>). A prefix hit always runs that forward.
    /// </remarks>
    public bool TryResumeAt(object mark, int maxSequenceLength)
    {
        ThrowIfDisposed();
        if (MtpPrefixMark.For(mark, this, _hiddenSize) is not { } m || maxSequenceLength < m.Length)
            return false;
        _maxSteps = Math.Max(_maxSteps, maxSequenceLength);
        _currentLength = m.Length;
        _carryHidden = (float[])m.CarryHidden.Clone();
        return true;
    }

    /// <summary>Total bytes allocated for this state's own KV-cache + pending-hidden buffer (device memory).</summary>
    public long AllocatedBytes => (2L * _capacity * _kvStride + _hiddenSize) * sizeof(float) + (long)_capacity * sizeof(int);

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
        if (_positionIotaDevice != 0) { CudaDriverApi.cuMemFree_v2(_positionIotaDevice); _positionIotaDevice = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
