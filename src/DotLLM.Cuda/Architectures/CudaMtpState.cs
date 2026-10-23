using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Element format of one side (K or V) of the CUDA MTP head's KV cache (issue #1095). The head follows the trunk's KV storage
/// (<see cref="F16"/>, <see cref="Q8_0"/>, <see cref="Q4_0"/>) so the fused decode kernels read its history in place, and
/// is F16 where they cannot. <see cref="F32"/> is a reference arm for tests and the acceptance-rate A/B only (no FP32 storage
/// on hot paths, #1107), and applies to both sides.
/// </summary>
public enum CudaMtpKvFormat
{
    /// <summary>F32 rows, the pre-#1095 layout: 4 bytes per element. Reference arm only.</summary>
    F32,

    /// <summary>F16 rows, read directly by the tensor-core decode kernel: 2 bytes per element.</summary>
    F16,

    /// <summary><c>block_q8_0</c> rows: 34 bytes per 32 elements.</summary>
    Q8_0,

    /// <summary><c>block_q4_0</c> rows: 18 bytes per 32 elements.</summary>
    Q4_0,
}

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
    private readonly CudaMtpKvFormat _keyFormat;
    private readonly CudaMtpKvFormat _valueFormat;
    private readonly long _keyRowBytes;
    private readonly long _valueRowBytes;

    private nint _keyCacheDevice;     // [capacity] rows of _keyRowBytes in _keyFormat, device-resident
    private nint _valueCacheDevice;   // [capacity] rows of _valueRowBytes in _valueFormat, device-resident

    // Issue #1095: in-place growth. With CUDA virtual memory the K, V and iota buffers reserve address space for MaxSteps
    // rows and map physical pages as they grow, so a growth keeps every row (and the pointer) where it is and costs only
    // the added pages. Null when the API is unavailable: the copying growth stays as the fallback.
    private readonly ICudaVirtualMemory? _vm;
    private CudaGrowableDeviceBuffer? _keyGrow;
    private CudaGrowableDeviceBuffer? _valueGrow;
    private CudaGrowableDeviceBuffer? _iotaGrow;
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

    /// <summary>K/V stride (numKvHeads * headDim) — the per-step row width of the device K/V cache, in elements.</summary>
    public int KvStride => _kvStride;

    /// <summary>Element format of the key cache (issue #1095).</summary>
    public CudaMtpKvFormat KeyFormat => _keyFormat;

    /// <summary>Element format of the value cache (issue #1095).</summary>
    public CudaMtpKvFormat ValueFormat => _valueFormat;

    /// <summary>Bytes of one key-cache row in <see cref="KeyFormat"/>.</summary>
    public long KeyRowBytes => _keyRowBytes;

    /// <summary>Bytes of one value-cache row in <see cref="ValueFormat"/>.</summary>
    public long ValueRowBytes => _valueRowBytes;

    /// <summary>True when the cache grows in place through CUDA virtual memory (issue #1095), false when a growth copies.</summary>
    public bool GrowsInPlace => _vm is not null;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _capturedRows.AsSpan(0, _capturedRowCount * _hiddenSize);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedRowCount;

    /// <inheritdoc/>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Creates a fresh MTP state with an F16 KV cache that grows by copying. All device buffers are zero-initialised (an
    /// empty MTP KV-cache and a zero pending-hidden vector — the latter is always overwritten by
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
        : this(hiddenSize, numKvHeads, headDim, maxSteps, CudaMtpKvFormat.F16, CudaMtpKvFormat.F16, virtualMemory: null, admitGrowth)
    {
    }

    /// <summary>
    /// Creates a fresh MTP state whose KV cache is stored in <paramref name="keyFormat"/> / <paramref name="valueFormat"/>
    /// (issue #1095) and, when <paramref name="virtualMemory"/> is given, grows in place.
    /// </summary>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numKvHeads">KV head count for the MTP block's own attention.</param>
    /// <param name="headDim">Per-head dimension for the MTP block's own attention.</param>
    /// <param name="maxSteps">Ceiling on positions; the address range reserved for in-place growth covers it.</param>
    /// <param name="keyFormat">Key-cache element format.</param>
    /// <param name="valueFormat">Value-cache element format.</param>
    /// <param name="virtualMemory">The device's virtual-memory API for in-place growth, or null to grow by copying.</param>
    /// <param name="admitGrowth">Free-VRAM admission for each growth, charged only the bytes it adds; null skips it.</param>
    /// <exception cref="ArgumentException">
    /// F32 on one side only, or a quantized side whose row (<c>numKvHeads * headDim</c>) is not a multiple of 32.
    /// </exception>
    internal CudaMtpState(int hiddenSize, int numKvHeads, int headDim, int maxSteps,
                          CudaMtpKvFormat keyFormat, CudaMtpKvFormat valueFormat,
                          ICudaVirtualMemory? virtualMemory, Action<long, string>? admitGrowth)
    {
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSteps <= 0) throw new ArgumentOutOfRangeException(nameof(maxSteps));
        if ((keyFormat == CudaMtpKvFormat.F32) != (valueFormat == CudaMtpKvFormat.F32))
            throw new ArgumentException("An F32 MTP head KV cache is F32 on both sides.", nameof(valueFormat));

        _hiddenSize = hiddenSize;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSteps = maxSteps;
        _kvStride = numKvHeads * headDim;
        _keyFormat = keyFormat;
        _valueFormat = valueFormat;
        _keyRowBytes = RowBytes(keyFormat, _kvStride);
        _valueRowBytes = RowBytes(valueFormat, _kvStride);
        _vm = virtualMemory;
        _admitGrowth = admitGrowth;

        // Lazy KV (issue #966): allocate min(ceiling, 1024) positions now and grow as positions are reached.
        _capacity = MtpKvCapacity.Initial(maxSteps);
        if (_vm is not null)
        {
            try
            {
                // Reserving address space costs no memory, so cover the whole ceiling: growths then never move the pointers.
                _keyGrow = new CudaGrowableDeviceBuffer(_vm, (long)maxSteps * _keyRowBytes);
                _valueGrow = new CudaGrowableDeviceBuffer(_vm, (long)maxSteps * _valueRowBytes);
                _iotaGrow = new CudaGrowableDeviceBuffer(_vm, (long)maxSteps * sizeof(int));
                MapRows(0, _capacity);
            }
            catch
            {
                DisposeDeviceBuffers();
                throw;
            }
        }
        else
        {
            long kBytes = (long)_capacity * _keyRowBytes;
            long vBytes = (long)_capacity * _valueRowBytes;
            CudaDriverApi.cuMemAlloc_v2(out _keyCacheDevice, (nuint)kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _valueCacheDevice, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(_keyCacheDevice, 0, (nuint)kBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(_valueCacheDevice, 0, (nuint)vBytes).ThrowOnError();

            // Positions as a device-resident iota (issue #482): a draft step's RoPE and a batched absorb's
            // contiguous positions are slices of it, so neither uploads positions per call. Grows with the KV.
            UploadIota(_capacity, out _positionIotaDevice);
        }

        long hiddenBytes = (long)hiddenSize * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out _pendingHiddenDevice, (nuint)hiddenBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_pendingHiddenDevice, 0, (nuint)hiddenBytes).ThrowOnError();
    }

    /// <summary>Bytes of one cache row of <paramref name="kvStride"/> elements in <paramref name="format"/>.</summary>
    /// <param name="format">Element format.</param>
    /// <param name="kvStride">Elements per row (<c>numKvHeads * headDim</c>).</param>
    /// <returns>Row bytes.</returns>
    /// <exception cref="ArgumentException">A quantized format with <paramref name="kvStride"/> not a multiple of 32.</exception>
    public static long RowBytes(CudaMtpKvFormat format, int kvStride) => format switch
    {
        CudaMtpKvFormat.F32 => (long)kvStride * sizeof(float),
        CudaMtpKvFormat.F16 => (long)kvStride * sizeof(ushort),
        _ when kvStride % 32 != 0 => throw new ArgumentException(
            $"A quantized MTP head KV row needs numKvHeads*headDim ({kvStride}) to be a multiple of 32.", nameof(kvStride)),
        CudaMtpKvFormat.Q8_0 => HybridKvStorage.RowBytes(KvCacheDType.Q8_0, kvStride),
        CudaMtpKvFormat.Q4_0 => HybridKvStorage.RowBytes(KvCacheDType.Q4_0, kvStride),
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };

    /// <summary>
    /// The quantized-KV kernels' name for <paramref name="format"/>: <see cref="KvCacheDType.Q8_0"/> /
    /// <see cref="KvCacheDType.Q4_0"/>, or <see cref="KvCacheDType.F32"/> ("unquantized", F16 on the GPU) for F16 rows.
    /// </summary>
    /// <param name="format">A non-F32 format.</param>
    /// <returns>The matching <see cref="KvCacheDType"/>.</returns>
    internal static KvCacheDType ToKvCacheDType(CudaMtpKvFormat format) => format switch
    {
        CudaMtpKvFormat.F16 => KvCacheDType.F32,
        CudaMtpKvFormat.Q8_0 => KvCacheDType.Q8_0,
        CudaMtpKvFormat.Q4_0 => KvCacheDType.Q4_0,
        _ => throw new ArgumentOutOfRangeException(nameof(format), "F32 rows have no quantized-KV kernel name."),
    };

    /// <summary>True for <see cref="CudaMtpKvFormat.Q8_0"/> and <see cref="CudaMtpKvFormat.Q4_0"/>.</summary>
    /// <param name="format">Element format.</param>
    /// <returns>Whether rows are quantized blocks.</returns>
    public static bool IsQuantized(CudaMtpKvFormat format) => format is CudaMtpKvFormat.Q8_0 or CudaMtpKvFormat.Q4_0;

    /// <summary>
    /// Device bytes one position costs in the head's cache under <paramref name="keyFormat"/> / <paramref name="valueFormat"/>:
    /// its K and V rows plus its iota entry. A copying growth holds old and new storage at once, so it costs twice that.
    /// </summary>
    /// <param name="keyFormat">Key-cache element format.</param>
    /// <param name="valueFormat">Value-cache element format.</param>
    /// <param name="kvStride">Elements per row.</param>
    /// <param name="growsInPlace">Whether the cache grows in place (CUDA virtual memory) rather than by copying.</param>
    /// <returns>Peak bytes per position.</returns>
    public static long BytesPerPosition(CudaMtpKvFormat keyFormat, CudaMtpKvFormat valueFormat, int kvStride, bool growsInPlace)
        => (RowBytes(keyFormat, kvStride) + RowBytes(valueFormat, kvStride) + sizeof(int)) * (growsInPlace ? 1 : 2);

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
    /// <remarks>
    /// In place (issue #1095) the growth maps pages behind the existing rows, so nothing is copied, the pointers stay put
    /// within the reservation, and admission is charged only the pages it adds. Otherwise the old and new buffers coexist
    /// during a copy and the whole new size is admitted. Either way a refusal or a failure leaves the state valid at its
    /// previous capacity.
    /// </remarks>
    /// <param name="positions">Rows required (last position + 1).</param>
    /// <param name="stream">The stream the head's kernels run on: drained before the buffers change.</param>
    internal void EnsureCapacity(int positions, nint stream)
    {
        ThrowIfDisposed();
        int target = MtpKvCapacity.Next(_capacity, positions, _maxSteps);
        if (target == _capacity)
            return;

        string what = $"MTP head KV cache for {target} positions";
        if (_vm is not null)
        {
            long added = _keyGrow!.GrowthBytes(target * _keyRowBytes)
                       + _valueGrow!.GrowthBytes(target * _valueRowBytes)
                       + _iotaGrow!.GrowthBytes((long)target * sizeof(int));
            if (added > 0)
                _admitGrowth?.Invoke(added, what);

            // A growth past the reservation (a resume raised MaxSteps) moves the ranges: nothing may still address them.
            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
            MapRows(_capacity, target);
            _capacity = target;
            return;
        }

        // Drain in-flight work that may still read/write the old buffers, then swap. The copy covers
        // only committed rows; speculative rows beyond CurrentLength are rewritten before they are read.
        CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

        long newKBytes = target * _keyRowBytes;
        long newVBytes = target * _valueRowBytes;
        int keepRows = Math.Min(_currentLength, _capacity);
        _admitGrowth?.Invoke(newKBytes + newVBytes + (long)target * sizeof(int), what);
        nint newKey = 0, newValue = 0, newIota = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out newKey, (nuint)newKBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out newValue, (nuint)newVBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(newKey, 0, (nuint)newKBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(newValue, 0, (nuint)newVBytes).ThrowOnError();
            if (keepRows > 0)
            {
                CudaDriverApi.cuMemcpyDtoD_v2(newKey, _keyCacheDevice, (nuint)(keepRows * _keyRowBytes)).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoD_v2(newValue, _valueCacheDevice, (nuint)(keepRows * _valueRowBytes)).ThrowOnError();
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
    /// In-place growth (issue #1095): maps pages so the growable buffers hold <paramref name="toRows"/> rows, zeroes K/V rows
    /// <c>[fromRows, toRows)</c> and writes their iota entries. The rows below <paramref name="fromRows"/> are not touched.
    /// The pointers are republished even when a buffer fails to grow (a re-reservation may have moved one).
    /// </summary>
    private unsafe void MapRows(int fromRows, int toRows)
    {
        try
        {
            _keyGrow!.EnsureCapacity(toRows * _keyRowBytes);
            _valueGrow!.EnsureCapacity(toRows * _valueRowBytes);
            _iotaGrow!.EnsureCapacity((long)toRows * sizeof(int));
        }
        finally
        {
            _keyCacheDevice = _keyGrow!.Pointer;
            _valueCacheDevice = _valueGrow!.Pointer;
            _positionIotaDevice = _iotaGrow!.Pointer;
        }

        int added = toRows - fromRows;
        CudaDriverApi.cuMemsetD8_v2(_keyCacheDevice + (nint)(fromRows * _keyRowBytes), 0, (nuint)(added * _keyRowBytes)).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(_valueCacheDevice + (nint)(fromRows * _valueRowBytes), 0, (nuint)(added * _valueRowBytes)).ThrowOnError();
        int[] iota = new int[added];
        for (int i = 0; i < added; i++) iota[i] = fromRows + i;
        fixed (int* p = iota)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(_positionIotaDevice + (nint)((long)fromRows * sizeof(int)), (nint)p,
                (nuint)((long)added * sizeof(int))).ThrowOnError();
        }
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

    /// <summary>Device pointer to the key-cache buffer, <see cref="Capacity"/> rows of <see cref="KeyRowBytes"/>; re-fetch after any call that may grow it.</summary>
    internal nint KeyCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _keyCacheDevice; }
    }

    /// <summary>Device pointer to the value-cache buffer, <see cref="Capacity"/> rows of <see cref="ValueRowBytes"/>; re-fetch after any call that may grow it.</summary>
    internal nint ValueCacheDevicePtr
    {
        get { ThrowIfDisposed(); return _valueCacheDevice; }
    }

    /// <summary>Device pointer to the key-cache row for MTP step <paramref name="step"/> (0-based): <see cref="KeyRowBytes"/> in <see cref="KeyFormat"/>.</summary>
    internal nint GetKeyRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return _keyCacheDevice + (nint)(step * _keyRowBytes);
    }

    /// <summary>Device pointer to the value-cache row for MTP step <paramref name="step"/> (0-based): <see cref="ValueRowBytes"/> in <see cref="ValueFormat"/>.</summary>
    internal nint GetValueRowDevicePtr(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return _valueCacheDevice + (nint)(step * _valueRowBytes);
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
    /// absorbs from, and a lazy growth keeps the committed rows (in place, or by copying them). Only host-side fields change here, so no CUDA context
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

    /// <summary>
    /// Total device bytes held by this state's own KV-cache, position iota and pending-hidden buffer. In place (issue #1095)
    /// that is the pages mapped so far, a whole number of allocation granules per buffer.
    /// </summary>
    public long AllocatedBytes => (long)_hiddenSize * sizeof(float) + (_vm is not null
        ? _keyGrow!.MappedBytes + _valueGrow!.MappedBytes + _iotaGrow!.MappedBytes
        : _capacity * (_keyRowBytes + _valueRowBytes + sizeof(int)));

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaMtpState));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        DisposeDeviceBuffers();
        if (_pendingHiddenDevice != 0) { CudaDriverApi.cuMemFree_v2(_pendingHiddenDevice); _pendingHiddenDevice = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>Frees the K, V and iota storage: unmaps the growable buffers, or frees the plain allocations.</summary>
    private void DisposeDeviceBuffers()
    {
        if (_vm is not null)
        {
            _keyGrow?.Dispose();
            _valueGrow?.Dispose();
            _iotaGrow?.Dispose();
            _keyGrow = _valueGrow = _iotaGrow = null;
            _keyCacheDevice = _valueCacheDevice = _positionIotaDevice = 0;
            return;
        }
        if (_keyCacheDevice != 0) { CudaDriverApi.cuMemFree_v2(_keyCacheDevice); _keyCacheDevice = 0; }
        if (_valueCacheDevice != 0) { CudaDriverApi.cuMemFree_v2(_valueCacheDevice); _valueCacheDevice = 0; }
        if (_positionIotaDevice != 0) { CudaDriverApi.cuMemFree_v2(_positionIotaDevice); _positionIotaDevice = 0; }
    }
}
