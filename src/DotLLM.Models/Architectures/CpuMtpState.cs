using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// CPU <see cref="IMtpState"/> implementation: the MTP head's own tiny KV-cache (sized for just
/// the trailing MTP block's attention, not the trunk) plus the pending-hidden-state handoff row
/// and the captured-rows buffer a verify-phase <c>Forward</c> call populates. See issue #253 and
/// <see cref="IMtpState"/> for the overall design; mirrors llama.cpp's
/// <c>common_speculative_state_draft_mtp</c> (<c>common/speculative.cpp</c>), translated from a
/// second <c>llama_context</c> into a plain per-sequence state object.
/// </summary>
/// <remarks>
/// <b>Head KV follows the trunk's KV element type (issue #1141, port of #1095).</b> The head's one attention layer stores K/V as
/// F16 by default (never wider: rule 1 of #1107), or Q8_0 / Q4_0 when the trunk's KV is quantized, and attends over those rows in
/// place through the tiled attention path. <see cref="CpuMtpKvFormat.F32"/> survives only as the reference arm
/// (<c>DOTLLM_CPU_MTP_KV=f32</c>, or the constructor argument in tests). Draft precision moves acceptance only, never the output:
/// the verify step keeps it exact. The pending / carried hidden rows stay F32 (docs/PRECISION.md: kept pending #1144).
/// </remarks>
public sealed unsafe class CpuMtpState : IMtpState, IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private int _maxSteps;            // hard ceiling on positions (the trunk KV-cache's MaxLength); raised by a resume (issue #1088)
    private int _capacity;            // rows currently allocated, <= _maxSteps; grows on demand (issue #966)
    private readonly int _kvStride; // numKvHeads * headDim
    private readonly CpuMtpKvFormat _format;
    private readonly int _rowBytes;   // bytes of one stored K (or V) row in _format

    private nint _keyCache;   // [capacity, _rowBytes] bytes
    private nint _valueCache; // [capacity, _rowBytes] bytes
    private readonly float[] _pendingHidden; // [hiddenSize] — seed for the next ForwardMtp call

    // The trunk's hidden state at the last position absorbed into the MTP KV-cache (issue #469):
    // the h_{p-1} that pairs with the next token x_p. Drafting rewrites _pendingHidden with the
    // head's own output; this survives it, so the next absorb starts from a verified row.
    private readonly float[] _carryHidden;

    private float[] _capturedRows = []; // [rowCount, hiddenSize], grown on demand
    private int _capturedRowCount;

    private int _currentLength;
    private bool _disposed;

    /// <summary>
    /// Hard ceiling on the positions this state can hold (the <c>maxSteps</c> it was created with).
    /// Exceeding it throws <see cref="InvalidOperationException"/> before any write.
    /// </summary>
    public int MaxSteps => _maxSteps;

    /// <summary>Element format the head's K/V rows are stored in.</summary>
    public CpuMtpKvFormat KvFormat => _format;

    /// <summary>Bytes of one stored K (or V) row.</summary>
    public int RowBytes => _rowBytes;

    /// <summary>Positions currently allocated; grows in <see cref="MtpKvCapacity.GrowthStep"/> steps up to <see cref="MaxSteps"/>.</summary>
    public int Capacity => _capacity;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _capturedRows.AsSpan(0, _capturedRowCount * _hiddenSize);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedRowCount;

    /// <inheritdoc/>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Creates a fresh MTP state. All buffers are zero-initialised (an empty MTP KV-cache and a
    /// zero pending-hidden vector — the latter is always overwritten by
    /// <see cref="SeedFromCapturedRow"/> before the first <c>ForwardMtp</c> call in normal use).
    /// </summary>
    /// <param name="hiddenSize">Model hidden size (matches <see cref="ModelConfig.HiddenSize"/>).</param>
    /// <param name="numKvHeads">KV head count for the MTP block's own attention.</param>
    /// <param name="headDim">Per-head dimension for the MTP block's own attention.</param>
    /// <param name="maxSteps">Maximum autoregressive MTP draft steps to size the KV-cache for (typically the max candidate count K).</param>
    /// <param name="format">Storage format of the head's K/V rows; <see langword="null"/> = the default (F16, or the <c>DOTLLM_CPU_MTP_KV</c> override).</param>
    public CpuMtpState(int hiddenSize, int numKvHeads, int headDim, int maxSteps, CpuMtpKvFormat? format = null)
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
        _format = ResolveFormat(format, headDim);
        _rowBytes = RowBytesOf(_format, _kvStride);

        // Lazy KV (issue #966): min(ceiling, 1024) positions now, grown as positions are reached.
        _capacity = MtpKvCapacity.Initial(maxSteps);
        long kvBytes = (long)_capacity * _rowBytes;
        _keyCache = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
        _valueCache = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
        NativeMemory.Clear((void*)_keyCache, (nuint)kvBytes);
        NativeMemory.Clear((void*)_valueCache, (nuint)kvBytes);

        _pendingHidden = new float[hiddenSize];
        _carryHidden = new float[hiddenSize];
    }

    /// <summary>Read-only view of the pending hidden-state vector that seeds the next <c>ForwardMtp</c> call.</summary>
    public ReadOnlySpan<float> PendingHidden => _pendingHidden;

    /// <summary>Mutable view — the MTP forward implementation writes its own output hidden state here after each step.</summary>
    internal Span<float> PendingHiddenMutable => _pendingHidden;

    /// <summary>
    /// Makes the buffers cover at least <paramref name="positions"/> rows (issue #966), preserving the
    /// committed rows <c>[0, CurrentLength)</c>. Row views from <see cref="GetKeyRowBytes"/> /
    /// <see cref="GetValueRowBytes"/> are invalidated by a grow: re-fetch after calling.
    /// Throws <see cref="InvalidOperationException"/> above <see cref="MaxSteps"/>.
    /// </summary>
    internal void EnsureCapacity(int positions)
    {
        ThrowIfDisposed();
        int target = MtpKvCapacity.Next(_capacity, positions, _maxSteps);
        if (target == _capacity)
            return;

        long newBytes = (long)target * _rowBytes;
        long keepBytes = (long)Math.Min(_currentLength, _capacity) * _rowBytes;
        nint newKey = (nint)NativeMemory.AlignedAlloc((nuint)newBytes, 64);
        nint newValue;
        try { newValue = (nint)NativeMemory.AlignedAlloc((nuint)newBytes, 64); }
        catch { NativeMemory.AlignedFree((void*)newKey); throw; }
        NativeMemory.Clear((void*)newKey, (nuint)newBytes);
        NativeMemory.Clear((void*)newValue, (nuint)newBytes);
        if (keepBytes > 0)
        {
            Buffer.MemoryCopy((void*)_keyCache, (void*)newKey, newBytes, keepBytes);
            Buffer.MemoryCopy((void*)_valueCache, (void*)newValue, newBytes, keepBytes);
        }
        NativeMemory.AlignedFree((void*)_keyCache);
        NativeMemory.AlignedFree((void*)_valueCache);
        _keyCache = newKey;
        _valueCache = newValue;
        _capacity = target;
    }

    /// <summary>Seeds the pending hidden from the carried (last absorbed) trunk row.</summary>
    internal void SetPendingFromCarry() => _carryHidden.CopyTo(_pendingHidden);

    /// <summary>Seeds the pending hidden from captured row <paramref name="row"/> without moving the carry.</summary>
    internal void SetPendingFromCapturedRow(int row)
        => _capturedRows.AsSpan(row * _hiddenSize, _hiddenSize).CopyTo(_pendingHidden);

    /// <summary>Environment variable selecting the head KV format: <c>f16</c> (default), <c>q8_0</c>, <c>q4_0</c> or <c>f32</c> (reference arm only).</summary>
    public const string KvEnvVar = "DOTLLM_CPU_MTP_KV";

    /// <summary>
    /// Resolves the storage format: an explicit request wins, then <see cref="KvEnvVar"/>, then F16. Quantized formats need
    /// <c>headDim % 32 == 0</c> (each head on whole blocks) and fall back to F16 otherwise.
    /// </summary>
    public static CpuMtpKvFormat ResolveFormat(CpuMtpKvFormat? requested, int headDim)
    {
        CpuMtpKvFormat f = requested ?? Environment.GetEnvironmentVariable(KvEnvVar)?.Trim().ToLowerInvariant() switch
        {
            "f32" => CpuMtpKvFormat.F32,
            "q8_0" or "q8" => CpuMtpKvFormat.Q8_0,
            "q4_0" or "q4" => CpuMtpKvFormat.Q4_0,
            _ => CpuMtpKvFormat.F16,
        };
        if (f is CpuMtpKvFormat.Q8_0 or CpuMtpKvFormat.Q4_0 && headDim % KvQuantize.BlockSize != 0)
            return CpuMtpKvFormat.F16;
        return f;
    }

    /// <summary>Maps a trunk KV element type to the head's format (the head follows the trunk; an unquantized trunk gives the default F16).</summary>
    public static CpuMtpKvFormat? FromTrunk(KvCacheDType? trunk) => trunk switch
    {
        KvCacheDType.Q8_0 => CpuMtpKvFormat.Q8_0,
        KvCacheDType.Q4_0 => CpuMtpKvFormat.Q4_0,
        _ => null,
    };

    private static int RowBytesOf(CpuMtpKvFormat f, int stride) => f switch
    {
        CpuMtpKvFormat.F32 => stride * sizeof(float),
        CpuMtpKvFormat.F16 => stride * 2,
        CpuMtpKvFormat.Q8_0 => KvQuantize.QuantizedRowBytes(stride, KvCacheDType.Q8_0),
        _ => KvQuantize.QuantizedRowBytes(stride, KvCacheDType.Q4_0),
    };

    /// <summary>Raw bytes of stored key row <paramref name="step"/>.</summary>
    internal Span<byte> GetKeyRowBytes(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return new Span<byte>((byte*)_keyCache + (long)step * _rowBytes, _rowBytes);
    }

    /// <summary>Raw bytes of stored value row <paramref name="step"/>.</summary>
    internal Span<byte> GetValueRowBytes(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_capacity) throw new ArgumentOutOfRangeException(nameof(step));
        return new Span<byte>((byte*)_valueCache + (long)step * _rowBytes, _rowBytes);
    }

    /// <summary>
    /// Stores <paramref name="count"/> F32 key rows (<c>[count, kvStride]</c>, already normed and rotated) at positions
    /// <c>[firstPosition, +count)</c>, converting to the head's format. Does not move the length.
    /// </summary>
    internal void StoreKeyRows(int firstPosition, ReadOnlySpan<float> rows, int count)
        => StoreRows((byte*)_keyCache, firstPosition, rows, count);

    /// <summary>Value-side twin of <see cref="StoreKeyRows"/>.</summary>
    internal void StoreValueRows(int firstPosition, ReadOnlySpan<float> rows, int count)
        => StoreRows((byte*)_valueCache, firstPosition, rows, count);

    private void StoreRows(byte* basePtr, int firstPosition, ReadOnlySpan<float> rows, int count)
    {
        ThrowIfDisposed();
        if (firstPosition < 0 || count < 0 || (long)firstPosition + count > _capacity)
            throw new ArgumentOutOfRangeException(nameof(count));
        if (rows.Length < count * _kvStride)
            throw new ArgumentException("rows is shorter than count * kvStride.", nameof(rows));
        fixed (float* src = rows)
        {
            for (int r = 0; r < count; r++)
            {
                float* s = src + (long)r * _kvStride;
                byte* d = basePtr + (long)(firstPosition + r) * _rowBytes;
                switch (_format)
                {
                    case CpuMtpKvFormat.F32: Buffer.MemoryCopy(s, d, _rowBytes, _rowBytes); break;
                    case CpuMtpKvFormat.F16:
                        TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(s, _kvStride), new Span<Half>(d, _kvStride));
                        break;
                    case CpuMtpKvFormat.Q8_0: KvQuantize.F32ToQ8_0(s, d, _kvStride); break;
                    default: KvQuantize.F32ToQ4_0(s, d, _kvStride); break;
                }
            }
        }
    }

    /// <summary>
    /// Single-token (<c>seqQ = 1</c>) causal attention of <paramref name="q"/> over the head's rows <c>[0, seqKv)</c>, read in
    /// place in the stored format. The F32 reference arm runs the original contiguous-span kernel (the true pre-#1141 baseline).
    /// </summary>
    internal void AttendOver(ReadOnlySpan<float> q, Span<float> output, int seqKv, int numHeads, int positionOffset)
    {
        ThrowIfDisposed();
        if (_format == CpuMtpKvFormat.F32)
        {
            Attention.Execute(q, new ReadOnlySpan<float>((float*)_keyCache, seqKv * _kvStride),
                new ReadOnlySpan<float>((float*)_valueCache, seqKv * _kvStride), output,
                1, seqKv, numHeads, _numKvHeads, _headDim, positionOffset, (int?)null);
            return;
        }
        var fmt = _format == CpuMtpKvFormat.F16 ? Attention.KvRowFormat.F16
            : _format == CpuMtpKvFormat.Q8_0 ? Attention.KvRowFormat.Q8_0 : Attention.KvRowFormat.Q4_0;
        fixed (float* qp = q, op = output)
            Attention.ExecuteContiguousRows(qp, (byte*)_keyCache, (byte*)_valueCache, fmt, _rowBytes, op,
                1, seqKv, numHeads, _numKvHeads, _headDim, positionOffset, pool: null);
    }

    /// <summary>Advances the MTP KV-cache length by one step after a successful <c>ForwardMtp</c> call.</summary>
    internal void Advance()
    {
        ThrowIfDisposed();
        if (_currentLength >= _maxSteps)
            throw new InvalidOperationException(
                $"CpuMtpState KV-cache exhausted: {_currentLength} steps already advanced against a " +
                $"MaxSteps={_maxSteps} cache. Size the state for the whole sequence (prompt + generated + draft steps).");
        _currentLength++;
    }

    /// <summary>
    /// Prepares a batched absorb of <paramref name="count"/> contiguous positions starting at
    /// <paramref name="firstPosition"/> (issue #472): rolls speculative slots back to it, rejects a
    /// gap, and bounds-checks the whole slab before anything is written.
    /// </summary>
    internal void BeginAbsorb(int firstPosition, int count)
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
                $"CpuMtpState KV-cache exhausted absorbing positions [{firstPosition}, {firstPosition + count}) " +
                $"(MaxSteps={_maxSteps}). Create the state with CreateMtpState(maxSequenceLength) covering the whole sequence.");
        EnsureCapacity(firstPosition + count);   // issue #966: lazy growth, after the ceiling check
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
    /// Writes the hidden-state rows a batch of <paramref name="count"/> tokens pairs with (issue
    /// #469): row 0 is the carried row (the trunk hidden of the position before the batch), row
    /// <c>i</c> is captured row <c>i - 1</c>. <paramref name="dest"/> is <c>[count, hiddenSize]</c>.
    /// </summary>
    internal void CopyAbsorbPairingRows(int count, Span<float> dest)
    {
        ThrowIfDisposed();
        if (count > _capturedRowCount)
            throw new ArgumentOutOfRangeException(nameof(count));
        _carryHidden.CopyTo(dest);
        if (count > 1)
            _capturedRows.AsSpan(0, (count - 1) * _hiddenSize).CopyTo(dest.Slice(_hiddenSize));
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
    /// Called by an MTP-supporting model's verify-phase <c>Forward</c> overload to stash the
    /// captured pre-final-norm hidden rows for this state's next round.
    /// </summary>
    /// <param name="rows">Row-major <c>[rowCount, hiddenSize]</c> hidden state rows.</param>
    /// <param name="rowCount">Number of rows in <paramref name="rows"/>.</param>
    internal void SetCapturedRows(ReadOnlySpan<float> rows, int rowCount)
    {
        ThrowIfDisposed();
        int needed = rowCount * _hiddenSize;
        if (_capturedRows.Length < needed)
            _capturedRows = new float[needed];
        rows.Slice(0, needed).CopyTo(_capturedRows);
        _capturedRowCount = rowCount;
    }

    /// <inheritdoc/>
    public void SeedFromCapturedRow(int rowIndex)
    {
        ThrowIfDisposed();
        if ((uint)rowIndex >= (uint)_capturedRowCount)
            throw new ArgumentOutOfRangeException(nameof(rowIndex),
                $"rowIndex {rowIndex} out of range [0, {_capturedRowCount}) — CapturedHiddenRows was not populated " +
                "by a verify-phase Forward call, or has fewer rows than expected.");
        var row = _capturedRows.AsSpan(rowIndex * _hiddenSize, _hiddenSize);
        row.CopyTo(_pendingHidden);
        row.CopyTo(_carryHidden);
    }

    /// <inheritdoc/>
    public object? MarkPrefix()
    {
        ThrowIfDisposed();
        return new MtpPrefixMark(this, _currentLength, (float[])_carryHidden.Clone());
    }

    /// <inheritdoc/>
    public bool TryResumeAt(object mark, int maxSequenceLength)
    {
        ThrowIfDisposed();
        if (MtpPrefixMark.For(mark, this, _hiddenSize) is not { } m || maxSequenceLength < m.Length)
            return false;
        _maxSteps = Math.Max(_maxSteps, maxSequenceLength);
        _currentLength = m.Length;
        m.CarryHidden.CopyTo(_carryHidden);
        m.CarryHidden.CopyTo(_pendingHidden);
        return true;
    }

    /// <summary>Total bytes allocated for this state's own KV-cache (excludes the small managed captured-rows buffer).</summary>
    public long AllocatedBytes => 2L * _capacity * _rowBytes;

    /// <summary>Stored KV bytes per position (K + V).</summary>
    public int KvBytesPerPosition => 2 * _rowBytes;

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CpuMtpState));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_keyCache != 0) { NativeMemory.AlignedFree((void*)_keyCache); _keyCache = 0; }
        if (_valueCache != 0) { NativeMemory.AlignedFree((void*)_valueCache); _valueCache = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>Finalizer — last-ditch free if the state was not disposed.</summary>
    ~CpuMtpState()
    {
        if (_disposed) return;
        if (_keyCache != 0) NativeMemory.AlignedFree((void*)_keyCache);
        if (_valueCache != 0) NativeMemory.AlignedFree((void*)_valueCache);
    }
}

/// <summary>Element format of the CPU MTP head's stored K/V rows (issue #1141).</summary>
public enum CpuMtpKvFormat
{
    /// <summary>F16 (the default: the widest the head stores).</summary>
    F16,
    /// <summary>Q8_0 blocks (follows a q8_0 trunk).</summary>
    Q8_0,
    /// <summary>Q4_0 blocks (follows a q4_0 trunk).</summary>
    Q4_0,
    /// <summary>F32: reference / A-B arm only (rule 1 of #1107).</summary>
    F32,
}
