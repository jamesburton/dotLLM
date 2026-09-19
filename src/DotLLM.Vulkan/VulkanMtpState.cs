using DotLLM.Core.Models;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan <see cref="IMtpState"/> implementation (issue #435): the MTP ("NextN") head's own tiny
/// KV-cache — sized for just the trailing MTP block's attention, not the trunk — plus the
/// pending-hidden handoff row and the captured-rows buffer a verify-phase <c>Forward</c> call
/// populates. Mirrors <see cref="DotLLM.Models.Architectures.CpuMtpState"/> and
/// <c>DotLLM.Cuda.Architectures.CudaMtpState</c>; see issue #253 and <see cref="IMtpState"/> for
/// the overall design.
/// </summary>
/// <remarks>
/// <para>
/// K/V live in device-local buffers written by <c>vkCmdCopyBuffer</c> from the model's per-step
/// K/V scratch and read by the attention kernel, so the autoregressive draft loop never round-trips
/// a K/V row through the host. <see cref="PendingHidden"/> is host-visible instead: it is 1 row
/// (<c>hiddenSize</c> floats, ~20 KB on Bonsai 2) and <see cref="SeedFromCapturedRow"/> has to
/// write it from host memory once per speculation round, which a host-visible allocation does
/// without a staging buffer. It is also a legal <c>vkCmdCopyBuffer</c> destination, which is how
/// the MTP step writes back its own output hidden state.
/// </para>
/// <para>
/// <b>Descriptor-cache hazard.</b> These buffers are bound into handle-keyed descriptor sets by the
/// attention kernel. Allocating and freeing a state repeatedly can recycle a Vulkan buffer handle
/// into a new allocation and hit a stale cached set — the failure mode is "correct, then zeros".
/// The model invalidates its kernel descriptor caches whenever it sees an MTP state it has not
/// seen before (see <c>VulkanQwen3HybridDenseTransformerModel.ForwardMtp</c>); allocate one state
/// per in-flight sequence and keep it, rather than one per round.
/// </para>
/// </remarks>
public sealed class VulkanMtpState : IMtpState, IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSteps;
    private readonly int _kvStride; // numKvHeads * headDim

    private readonly VulkanDevice.Buffer _keyCache;      // [maxSteps, kvStride] f32, device-local
    private readonly VulkanDevice.Buffer _valueCache;    // [maxSteps, kvStride] f32, device-local
    private readonly VulkanDevice.Buffer _pendingHidden; // [hiddenSize] f32, host-visible

    private float[] _capturedRows = []; // host [rowCount, hiddenSize], grown on demand
    private int _capturedRowCount;

    private int _currentLength;
    private bool _disposed;

    /// <summary>Max autoregressive MTP steps this state's KV-cache can hold before needing a reset/rollback.</summary>
    public int MaxSteps => _maxSteps;

    /// <summary>K/V stride (<c>numKvHeads * headDim</c>) — the per-step row width of the device K/V cache.</summary>
    public int KvStride => _kvStride;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => _capturedRows.AsSpan(0, _capturedRowCount * _hiddenSize);

    /// <inheritdoc/>
    public int CapturedRowCount => _capturedRowCount;

    /// <inheritdoc/>
    public int HiddenSize => _hiddenSize;

    /// <summary>Device key-cache buffer, <c>[maxSteps, kvStride]</c> row-major F32.</summary>
    internal VulkanDevice.Buffer KeyCache => _keyCache;

    /// <summary>Device value-cache buffer, <c>[maxSteps, kvStride]</c> row-major F32.</summary>
    internal VulkanDevice.Buffer ValueCache => _valueCache;

    /// <summary>Host-visible pending-hidden buffer, <c>[hiddenSize]</c> F32 — the seed for the next <c>ForwardMtp</c> call.</summary>
    internal VulkanDevice.Buffer PendingHidden => _pendingHidden;

    /// <summary>
    /// Creates a fresh MTP state. All buffers are zero-initialised (an empty MTP KV-cache and a
    /// zero pending-hidden vector — the latter is always overwritten by
    /// <see cref="SeedFromCapturedRow"/> before the first <c>ForwardMtp</c> call in normal use).
    /// </summary>
    /// <param name="device">Vulkan device on which to allocate.</param>
    /// <param name="hiddenSize">Model hidden size (matches <see cref="ModelConfig.HiddenSize"/>).</param>
    /// <param name="numKvHeads">KV head count for the MTP block's own attention.</param>
    /// <param name="headDim">Per-head dimension for the MTP block's own attention.</param>
    /// <param name="maxSteps">Maximum autoregressive MTP draft steps to size the KV-cache for (typically the max candidate count K).</param>
    public VulkanMtpState(VulkanDevice device, int hiddenSize, int numKvHeads, int headDim, int maxSteps)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSteps <= 0) throw new ArgumentOutOfRangeException(nameof(maxSteps));

        _device = device;
        _hiddenSize = hiddenSize;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSteps = maxSteps;
        _kvStride = numKvHeads * headDim;

        long kvBytes = (long)maxSteps * _kvStride * sizeof(float);
        long hiddenBytes = (long)hiddenSize * sizeof(float);

        _keyCache = device.AllocateDeviceLocal(kvBytes);
        _valueCache = device.AllocateDeviceLocal(kvBytes);
        _pendingHidden = device.Allocate(hiddenBytes);

        // Zero every buffer. The KV-cache must start zeroed because a draft step attends over
        // [0, step] and a NaN/garbage row inside that range would poison the softmax even though
        // it is "before" any row this round wrote — Rollback(0) resets the length, not the bytes.
        var zeros = new byte[Math.Max(kvBytes, hiddenBytes)];
        using var staging = device.Allocate(zeros.Length);
        device.Upload(zeros.AsSpan(), staging);
        device.CopyBufferSynchronous(staging, _keyCache, (ulong)kvBytes);
        device.CopyBufferSynchronous(staging, _valueCache, (ulong)kvBytes);
        device.Upload(zeros.AsSpan(0, (int)hiddenBytes), _pendingHidden);
    }

    /// <summary>Advances the MTP KV-cache length by one step after a successful <c>ForwardMtp</c> call.</summary>
    internal void Advance()
    {
        ThrowIfDisposed();
        if (_currentLength >= _maxSteps)
            throw new InvalidOperationException(
                $"VulkanMtpState KV-cache exhausted: {_currentLength} steps already advanced against a " +
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
    /// Called by the model's verify-phase <c>Forward</c> overload to stash the captured pre-final-norm
    /// hidden rows for this state's next round.
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
        _device.Upload((ReadOnlySpan<float>)_capturedRows.AsSpan(rowIndex * _hiddenSize, _hiddenSize), _pendingHidden);
    }

    /// <summary>Total device bytes allocated for this state (excludes the small managed captured-rows buffer).</summary>
    public long AllocatedBytes =>
        2L * _maxSteps * _kvStride * sizeof(float) + (long)_hiddenSize * sizeof(float);

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanMtpState));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _keyCache.Dispose();
        _valueCache.Dispose();
        _pendingHidden.Dispose();
    }
}
