using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// CPU <see cref="IMtpState"/> implementation: the MTP head's own tiny KV-cache (sized for just
/// the trailing MTP block's attention, not the trunk) plus the pending-hidden-state handoff row
/// and the captured-rows buffer a verify-phase <c>Forward</c> call populates. See issue #253 and
/// <see cref="IMtpState"/> for the overall design; mirrors llama.cpp's
/// <c>common_speculative_state_draft_mtp</c> (<c>common/speculative.cpp</c>), translated from a
/// second <c>llama_context</c> into a plain per-sequence state object.
/// </summary>
public sealed unsafe class CpuMtpState : IMtpState, IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSteps;
    private readonly int _kvStride; // numKvHeads * headDim

    private nint _keyCache;   // [maxSteps, kvStride]
    private nint _valueCache; // [maxSteps, kvStride]
    private readonly float[] _pendingHidden; // [hiddenSize] — seed for the next ForwardMtp call

    private float[] _capturedRows = []; // [rowCount, hiddenSize], grown on demand
    private int _capturedRowCount;

    private int _currentLength;
    private bool _disposed;

    /// <summary>Max autoregressive MTP steps this state's KV-cache can hold before needing a reset/rollback.</summary>
    public int MaxSteps => _maxSteps;

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
    public CpuMtpState(int hiddenSize, int numKvHeads, int headDim, int maxSteps)
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
        _keyCache = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
        _valueCache = (nint)NativeMemory.AlignedAlloc((nuint)kvBytes, 64);
        NativeMemory.Clear((void*)_keyCache, (nuint)kvBytes);
        NativeMemory.Clear((void*)_valueCache, (nuint)kvBytes);

        _pendingHidden = new float[hiddenSize];
    }

    /// <summary>Read-only view of the pending hidden-state vector that seeds the next <c>ForwardMtp</c> call.</summary>
    public ReadOnlySpan<float> PendingHidden => _pendingHidden;

    /// <summary>Mutable view — the MTP forward implementation writes its own output hidden state here after each step.</summary>
    internal Span<float> PendingHiddenMutable => _pendingHidden;

    /// <summary>Key-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal Span<float> GetKeyRow(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_maxSteps) throw new ArgumentOutOfRangeException(nameof(step));
        return new Span<float>((float*)_keyCache + (long)step * _kvStride, _kvStride);
    }

    /// <summary>Value-cache row for MTP step <paramref name="step"/> (0-based), shape <c>[kvStride]</c>.</summary>
    internal Span<float> GetValueRow(int step)
    {
        ThrowIfDisposed();
        if ((uint)step >= (uint)_maxSteps) throw new ArgumentOutOfRangeException(nameof(step));
        return new Span<float>((float*)_valueCache + (long)step * _kvStride, _kvStride);
    }

    /// <summary>
    /// Base pointer to the full key-cache buffer, shape <c>[maxSteps, kvStride]</c> row-major.
    /// Used to build a multi-row <c>ReadOnlySpan&lt;float&gt;</c> covering <c>[0, seqKv)</c> steps
    /// for the attention kernel (a single MTP decode step attends over every step drafted so far
    /// in the current round, not just the newest one).
    /// </summary>
    internal float* KeyCachePtr => (float*)_keyCache;

    /// <summary>Base pointer to the full value-cache buffer — see <see cref="KeyCachePtr"/>.</summary>
    internal float* ValueCachePtr => (float*)_valueCache;

    /// <summary>Advances the MTP KV-cache length by one step after a successful <c>ForwardMtp</c> call.</summary>
    internal void Advance()
    {
        ThrowIfDisposed();
        if (_currentLength >= _maxSteps)
            throw new InvalidOperationException(
                $"CpuMtpState KV-cache exhausted: {_currentLength} steps already advanced against a " +
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
        _capturedRows.AsSpan(rowIndex * _hiddenSize, _hiddenSize).CopyTo(_pendingHidden);
    }

    /// <summary>Total bytes allocated for this state's own KV-cache (excludes the small managed captured-rows buffer).</summary>
    public long AllocatedBytes => 2L * _maxSteps * _kvStride * sizeof(float);

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
