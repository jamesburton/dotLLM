using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side Q8_0/Q4_0 quantized KV cache. Stores per-layer quantized K/V in a GPU-friendly
/// split layout (float scales + packed quants in separate buffers) and dequantizes the live
/// range into a shared FP32 scratch on demand so the existing F32 attention kernels run unchanged.
/// </summary>
/// <remarks>
/// <para><b>Split layout.</b> Unlike the CPU/CUDA backends that use the standard GGUF block
/// layout (f16 scale + int8/nibble contiguous), this cache stores float scales and packed quant
/// bytes in separate buffers for GPU-friendly aligned access. The quantization algorithm is
/// identical (max-abs symmetric, same rounding); only the memory layout differs.</para>
///
/// <para><b>Pure-quantized.</b> All tokens are quantized on write — there is no FP32 recency
/// window. Quality impact is minimal for Q8_0 and small for Q4_0 (the CPU/CUDA window path
/// is a refinement for later).</para>
///
/// <para><b>Mixed K/V types.</b> Keys and values can use different quantization (e.g. K=Q8_0,
/// V=Q4_0 for the K8V4 configuration). Separate kernel instances handle each.</para>
/// </remarks>
public sealed class VulkanQuantizedKvCache : IKvCache, IPerLayerKvCache, IDisposable
{
    private const int BlockSize = 32;

    private readonly VulkanDevice _device;

    // Per-layer quantized storage (split: scales + packed quants).
    private readonly VulkanDevice.Buffer[] _keyScales;
    private readonly VulkanDevice.Buffer[] _keyQuants;
    private readonly VulkanDevice.Buffer[] _valueScales;
    private readonly VulkanDevice.Buffer[] _valueQuants;

    // Shared FP32 scratch for attention (dequant target, shared across layers).
    private readonly VulkanDevice.Buffer _kScratch;
    private readonly VulkanDevice.Buffer _vScratch;

    // Kernels.
    private readonly KvQuantizeKernel _keyQuantize;
    private readonly KvQuantizeKernel _valueQuantize;
    private readonly KvDequantKernel _keyDequant;
    private readonly KvDequantKernel _valueDequant;
    private readonly bool _ownsKernels;

    // Geometry.
    private readonly int _numLayers;
    private readonly int _stride;         // numKvHeads * headDim (uniform)
    private readonly int _blocksPerRow;   // stride / 32
    private readonly int _maxSeqLen;

    // State.
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;
    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;
    /// <summary>Number of layers.</summary>
    public int LayerCount => _numLayers;
    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _stride;
    /// <summary>Key quantization type.</summary>
    public KvCacheDType KeyDType { get; }
    /// <summary>Value quantization type.</summary>
    public KvCacheDType ValueDType { get; }

    /// <summary>Total bytes allocated for all quantized buffers + scratch.</summary>
    public long AllocatedBytes { get; }

    /// <summary>
    /// Creates a uniform-geometry quantized KV cache with caller-owned (shared) kernels.
    /// </summary>
    public VulkanQuantizedKvCache(
        VulkanDevice device,
        KvQuantizeKernel keyQuantize, KvQuantizeKernel valueQuantize,
        KvDequantKernel keyDequant, KvDequantKernel valueDequant,
        bool ownsKernels,
        int numLayers, int numKvHeads, int headDim, int maxSeqLen,
        KvCacheDType keyDType, KvCacheDType valueDType)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(keyQuantize);
        ArgumentNullException.ThrowIfNull(valueQuantize);
        ArgumentNullException.ThrowIfNull(keyDequant);
        ArgumentNullException.ThrowIfNull(valueDequant);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0 || headDim % BlockSize != 0)
            throw new ArgumentException($"headDim must be a positive multiple of {BlockSize}; got {headDim}.", nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (keyDType is not (KvCacheDType.Q8_0 or KvCacheDType.Q4_0))
            throw new ArgumentOutOfRangeException(nameof(keyDType));
        if (valueDType is not (KvCacheDType.Q8_0 or KvCacheDType.Q4_0))
            throw new ArgumentOutOfRangeException(nameof(valueDType));

        _device = device;
        _keyQuantize = keyQuantize;
        _valueQuantize = valueQuantize;
        _keyDequant = keyDequant;
        _valueDequant = valueDequant;
        _ownsKernels = ownsKernels;
        _numLayers = numLayers;
        _stride = numKvHeads * headDim;
        _blocksPerRow = _stride / BlockSize;
        _maxSeqLen = maxSeqLen;
        KeyDType = keyDType;
        ValueDType = valueDType;

        int kQuantUints = keyQuantize.QuantUintsPerBlock;
        int vQuantUints = valueQuantize.QuantUintsPerBlock;
        long totalBlocks = (long)maxSeqLen * _blocksPerRow;

        long keyScaleBytes = totalBlocks * sizeof(float);
        long keyQuantBytes = totalBlocks * kQuantUints * sizeof(uint);
        long valScaleBytes = totalBlocks * sizeof(float);
        long valQuantBytes = totalBlocks * vQuantUints * sizeof(uint);
        long scratchBytes = (long)maxSeqLen * _stride * sizeof(float);

        _keyScales = new VulkanDevice.Buffer[numLayers];
        _keyQuants = new VulkanDevice.Buffer[numLayers];
        _valueScales = new VulkanDevice.Buffer[numLayers];
        _valueQuants = new VulkanDevice.Buffer[numLayers];

        long allocated = 0;
        for (int i = 0; i < numLayers; i++)
        {
            _keyScales[i] = device.Allocate(keyScaleBytes);
            _keyQuants[i] = device.Allocate(keyQuantBytes);
            _valueScales[i] = device.Allocate(valScaleBytes);
            _valueQuants[i] = device.Allocate(valQuantBytes);
            allocated += keyScaleBytes + keyQuantBytes + valScaleBytes + valQuantBytes;
        }
        _kScratch = device.Allocate(scratchBytes);
        _vScratch = device.Allocate(scratchBytes);
        allocated += 2 * scratchBytes;
        AllocatedBytes = allocated;
    }

    /// <summary>
    /// Creates a uniform-geometry quantized KV cache, owning its own kernel instances.
    /// </summary>
    public VulkanQuantizedKvCache(
        VulkanDevice device, string spvDir,
        int numLayers, int numKvHeads, int headDim, int maxSeqLen,
        KvCacheDType keyDType, KvCacheDType valueDType)
        : this(device,
               KvQuantizeKernel.Create(device, spvDir, keyDType),
               KvQuantizeKernel.Create(device, spvDir, valueDType),
               KvDequantKernel.Create(device, spvDir, keyDType),
               KvDequantKernel.Create(device, spvDir, valueDType),
               ownsKernels: true,
               numLayers, numKvHeads, headDim, maxSeqLen, keyDType, valueDType)
    {
    }

    /// <summary>Device buffer holding the dequantized keys for the last layer dequantized via
    /// <see cref="RecordDequant"/>. The attention kernel reads this.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer() => _kScratch;
    /// <summary>Device buffer holding the dequantized values.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer() => _vScratch;

    /// <summary>
    /// Records quantization of fresh FP32 K/V into the layer's quantized store. Positions must
    /// be contiguous ascending (prefill / single-token decode). Advances <see cref="CurrentLength"/>.
    /// </summary>
    internal void RecordUpdate(nint cmdBuf,
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        int startPos = ValidateContiguous(positions, seqLen);
        int totalBlocks = seqLen * _blocksPerRow;
        int srcElementOffset = 0; // K/V buffers start at element 0 (fresh activations)
        int dstBlockOffset = startPos * _blocksPerRow;

        _keyQuantize.Record(cmdBuf, kDev, _keyScales[layerIndex], _keyQuants[layerIndex],
                            totalBlocks, srcElementOffset, dstBlockOffset);
        _valueQuantize.Record(cmdBuf, vDev, _valueScales[layerIndex], _valueQuants[layerIndex],
                              totalBlocks, srcElementOffset, dstBlockOffset);

        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <summary>
    /// Records dequantization of <c>[0, CurrentLength)</c> for <paramref name="layerIndex"/> into the
    /// shared K/V scratch buffers (read by attention via <see cref="GetKeysBuffer"/>/<see cref="GetValuesBuffer"/>).
    /// </summary>
    internal void RecordDequant(nint cmdBuf, int layerIndex)
    {
        int totalBlocks = _currentLength * _blocksPerRow;
        _keyDequant.Record(cmdBuf, _keyScales[layerIndex], _keyQuants[layerIndex], _kScratch,
                           totalBlocks, srcOffset: 0, dstOffset: 0);
        _valueDequant.Record(cmdBuf, _valueScales[layerIndex], _valueQuants[layerIndex], _vScratch,
                             totalBlocks, srcOffset: 0, dstOffset: 0);
    }

    /// <summary>Synchronous quantize + store for one layer (one-shot submit). For tests.</summary>
    public void UpdateSync(VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
                           ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordUpdate(ctx.CommandBuffer, kDev, vDev, positions, seqLen, layerIndex);
        ctx.SubmitAndWait();
    }

    /// <summary>Synchronous dequant of one layer into the scratch buffers (one-shot submit). For tests.</summary>
    public void DequantSync(int layerIndex)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordDequant(ctx.CommandBuffer, layerIndex);
        ctx.SubmitAndWait();
    }

    /// <summary>Downloads the dequantized keys for a layer to host (dequant + download). Test helper.</summary>
    public void DequantKeysToHost(int layerIndex, Span<float> destination)
    {
        DequantSync(layerIndex);
        _device.Download(_kScratch, destination[..(_currentLength * _stride)]);
    }

    /// <summary>Downloads the dequantized values for a layer to host. Test helper.</summary>
    public void DequantValuesToHost(int layerIndex, Span<float> destination)
    {
        DequantSync(layerIndex);
        _device.Download(_vScratch, destination[..(_currentLength * _stride)]);
    }

    private int ValidateContiguous(ReadOnlySpan<int> positions, int seqLen)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        int start = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] != start + i)
                throw new NotSupportedException(
                    "VulkanQuantizedKvCache requires contiguous ascending positions (prefill / decode).");
            if ((uint)positions[i] >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"position {positions[i]} exceeds max {_maxSeqLen}.");
        }
        return start;
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength) throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length (new sequence).</summary>
    public void Reset() => _currentLength = 0;

    // IKvCache host overloads are not used on the Vulkan device path.
    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache is updated via RecordUpdate from the Vulkan forward pass.");
    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache is updated via RecordUpdate from the Vulkan forward pass.");
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache exposes device buffers, not TensorRef.");
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache exposes device buffers, not TensorRef.");
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache does not materialise cached keys as CPU tensors.");
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanQuantizedKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keyScales[i]?.Dispose();
            _keyQuants[i]?.Dispose();
            _valueScales[i]?.Dispose();
            _valueQuants[i]?.Dispose();
        }
        _kScratch?.Dispose();
        _vScratch?.Dispose();
        if (_ownsKernels)
        {
            _keyQuantize.Dispose();
            _valueQuantize.Dispose();
            _keyDequant.Dispose();
            _valueDequant.Dispose();
        }
    }
}
