using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Sparse, F32, per-attention-layer-slot device KV cache for <c>CudaNemotronHTransformerModel</c>.
/// Sized to <c>attentionLayerCount</c> (the number of <see cref="DotLLM.Core.Models.HybridLayerKind.Attention"/>
/// layers), not the model's total layer count — the model maps a physical layer index to a slot via
/// its own <c>kvSlotForLayer</c> array (mirrors <c>NemotronHTransformerModel</c>/
/// <c>VulkanNemotronHKvCache</c>'s sparse-KV-slot design exactly).
/// </summary>
/// <remarks>
/// F32 storage (not F16, unlike the generic <see cref="CudaKvCache"/>) to keep the model's
/// activation precision uniform end-to-end — see this plan's Global Constraints.
/// <see cref="Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/> requires
/// <c>positions</c> to be contiguous and ascending starting at
/// <c>positions[0]</c> — the only write pattern NemotronH's CPU/Vulkan hosts ever produce
/// (prefill writes <c>[0, seqLen)</c>; decode writes one position). A non-contiguous call throws.
/// </remarks>
internal sealed class CudaNemotronHKvCache : IKvCache
{
    private readonly nint[] _keys;   // per-slot device buffers, [maxSeqLen, kvStride] F32
    private readonly nint[] _values;
    private readonly int _kvStride;  // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private readonly int _deviceId;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of attention-layer slots this cache covers.</summary>
    public int AttentionLayerCount => _keys.Length;

    /// <summary>Total bytes allocated across every slot's key + value buffers.</summary>
    public long AllocatedBytes =>
        2L * _keys.Length * _maxSeqLen * _kvStride * sizeof(float);

    public CudaNemotronHKvCache(int attentionLayerCount, int numKvHeads, int headDim, int maxSeqLen, int deviceId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(attentionLayerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _deviceId = deviceId;
        _keys = new nint[attentionLayerCount];
        _values = new nint[attentionLayerCount];

        long bytesPerSlot = (long)maxSeqLen * _kvStride * sizeof(float);
        for (int i = 0; i < attentionLayerCount; i++)
        {
            CudaDriverApi.cuMemAlloc_v2(out _keys[i], (nuint)bytesPerSlot).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out _values[i], (nuint)bytesPerSlot).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        ThrowIfDisposed();
        if (positions.IsEmpty)
            throw new ArgumentException("positions must be non-empty.", nameof(positions));
        int startPos = positions[0];
        for (int i = 1; i < positions.Length; i++)
        {
            if (positions[i] != startPos + i)
                throw new NotSupportedException(
                    $"{nameof(CudaNemotronHKvCache)}.Update requires contiguous ascending positions " +
                    $"starting at positions[0]={startPos}; got positions[{i}]={positions[i]}.");
        }

        int seqLen = keys.Dim0;
        if (startPos + seqLen > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions),
                $"positions extend to {startPos + seqLen}, exceeding MaxLength={_maxSeqLen}.");

        long rowBytes = (long)_kvStride * sizeof(float);
        long bytesToCopy = (long)seqLen * rowBytes;
        nint dstK = _keys[layerIndex] + (nint)((long)startPos * rowBytes);
        nint dstV = _values[layerIndex] + (nint)((long)startPos * rowBytes);

        CudaDriverApi.cuMemcpyDtoD_v2(dstK, keys.DataPointer, (nuint)bytesToCopy).ThrowOnError();
        CudaDriverApi.cuMemcpyDtoD_v2(dstV, values.DataPointer, (nuint)bytesToCopy).ThrowOnError();

        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        ThrowIfDisposed();
        return new TensorRef(_currentLength, _kvStride, DType.Float32, _deviceId, _keys[layerIndex]);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        ThrowIfDisposed();
        return new TensorRef(_currentLength, _kvStride, DType.Float32, _deviceId, _values[layerIndex]);
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ThrowIfDisposed();
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.Update(ITensor) not supported. Use Update(TensorRef).");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.GetKeys(ITensor) not supported. Use GetKeysRef.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException(
            $"{nameof(CudaNemotronHKvCache)}.GetValues(ITensor) not supported. Use GetValuesRef.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        for (int i = 0; i < _keys.Length; i++)
        {
            if (_keys[i] != 0) CudaDriverApi.cuMemFree_v2(_keys[i]);
            if (_values[i] != 0) CudaDriverApi.cuMemFree_v2(_values[i]);
        }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CudaNemotronHKvCache));
    }
}
