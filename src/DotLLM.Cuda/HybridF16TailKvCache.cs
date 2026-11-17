using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Cuda;

// TODO(#1870): temporary F16 fallback for head_dim % 32 != 0; remove when quantized KV supports it
/// <summary>
/// Host KV for the CPU tail of <see cref="HybridTransformerModel"/> as F16 rows (issue #1851), for models whose head dimension is
/// not a multiple of 32: Q8_0 blocks must not straddle heads, and the quantized attention path decodes one head at a time. F16 is
/// the widest format the no-FP32-storage rule allows; the CPU attention reads the rows in place
/// (<c>Attention.ExecuteContiguousRows</c>). Uniform geometry, contiguous rows per layer, position <c>p</c> at row <c>p</c>.
/// </summary>
internal sealed unsafe class HybridF16TailKvCache : IKvCache
{
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int _stride;
    private readonly int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;

    /// <summary>Creates the cache.</summary>
    /// <param name="numLayers">Tail layers.</param>
    /// <param name="numKvHeads">KV heads.</param>
    /// <param name="headDim">Head dimension.</param>
    /// <param name="maxSeqLen">Positions.</param>
    public HybridF16TailKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _keys = new nint[numLayers];
        _values = new nint[numLayers];
        nuint bytes = (nuint)checked((long)maxSeqLen * _stride * sizeof(ushort));
        for (int l = 0; l < numLayers; l++)
        {
            _keys[l] = (nint)NativeMemory.AlignedAlloc(bytes, 64);
            _values[l] = (nint)NativeMemory.AlignedAlloc(bytes, 64);
        }
    }

    /// <summary>Bytes per stored row (one position, K or V).</summary>
    public int RowBytes => _stride * sizeof(ushort);

    /// <summary>Host bytes held.</summary>
    public long AllocatedBytes => 2L * _keys.Length * _maxSeqLen * RowBytes;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Base of layer <paramref name="layer"/>'s K rows.</summary>
    public byte* KeysOf(int layer) => (byte*)_keys[layer];

    /// <summary>Base of layer <paramref name="layer"/>'s V rows.</summary>
    public byte* ValuesOf(int layer) => (byte*)_values[layer];

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        if (keys.DType != DType.Float32 || values.DType != DType.Float32)
            throw new ArgumentException("The F16 tail cache takes F32 rows from the CPU forward.");
        float* k = (float*)keys.DataPointer, v = (float*)values.DataPointer;
        Half* kd = (Half*)_keys[layerIndex], vd = (Half*)_values[layerIndex];
        int maxPos = -1;
        for (int i = 0; i < positions.Length; i++)
        {
            int pos = positions[i];
            if ((uint)pos >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), $"Position {pos} exceeds max cache length {_maxSeqLen}.");
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(k + (long)i * _stride, _stride), new Span<Half>(kd + (long)pos * _stride, _stride));
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(v + (long)i * _stride, _stride), new Span<Half>(vd + (long)pos * _stride, _stride));
            maxPos = Math.Max(maxPos, pos);
        }
        if (maxPos + 1 > _currentLength) _currentLength = maxPos + 1;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => Update(new TensorRef(positions.Length, _stride, keys.DType, keys.DeviceId, keys.DataPointer),
                  new TensorRef(positions.Length, _stride, values.DType, values.DeviceId, values.DataPointer), positions, layerIndex);

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) => new(_currentLength, _stride, DType.Float16, -1, _keys[layerIndex]);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) => new(_currentLength, _stride, DType.Float16, -1, _values[layerIndex]);

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex) => new TensorView(new TensorShape(_currentLength, _stride), DType.Float16, -1, _keys[layerIndex]);

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex) => new TensorView(new TensorShape(_currentLength, _stride), DType.Float16, -1, _values[layerIndex]);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (length < _currentLength) _currentLength = length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int l = 0; l < _keys.Length; l++)
        {
            NativeMemory.AlignedFree((void*)_keys[l]);
            NativeMemory.AlignedFree((void*)_values[l]);
        }
    }
}
