using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side TurboQuant (MSE-stage) KV cache. Stores per-layer compressed codes + an fp32 norm
/// per cached head-vector in device buffers, and dequantizes the live range into a shared fp32
/// scratch on demand so the existing F32 attention kernel runs unchanged — the GPU analogue of the
/// CPU <c>TurboQuantKvCache</c>.
/// </summary>
/// <remarks>
/// <para><b>GPU-resident encode + decode.</b> Update dispatches the
/// <see cref="TurboQuantEncodeF32Kernel"/> (fresh device K/V → codes), and attention prep dispatches
/// the <see cref="TurboQuantDequantF32Kernel"/> (codes → fp32 scratch). Both stay on-device, so the
/// cache fits the fence-pipelined forward graph without a host round-trip. The codec constants
/// (centroids/signs/invSqrtD) are supplied by the caller — the Vulkan project does not depend on the
/// Engine codec; the bridging test passes <c>TurboQuantCodec</c>'s exposed constants.</para>
///
/// <para><b>Scope.</b> Uniform geometry only (single <c>headDim</c>/<c>numKvHeads</c>),
/// pure-quantized (no fp32 recency window), <c>headDim</c> a power of two ≤ 256 — matching the CPU
/// cache and the GQA models that pick TurboQuant. K and V use independent rotations (distinct sign
/// sets) but share the centroid table. The dequant scratch is shared across layers (filled
/// just-in-time per layer), so the permanent footprint is codes+norms, not F32.</para>
/// </remarks>
public sealed class VulkanTurboQuantKvCache : IKvCache, IPerLayerKvCache
{
    private readonly VulkanDevice _device;
    private readonly TurboQuantEncodeF32Kernel _encode;
    private readonly TurboQuantDequantF32Kernel _dequant;
    private readonly bool _ownsKernels;

    private readonly VulkanDevice.Buffer[] _keyCodes;   // [numLayers]
    private readonly VulkanDevice.Buffer[] _valueCodes;
    private readonly VulkanDevice.Buffer[] _keyNorms;
    private readonly VulkanDevice.Buffer[] _valueNorms;
    private readonly VulkanDevice.Buffer _centroids;
    private readonly VulkanDevice.Buffer _signsK;
    private readonly VulkanDevice.Buffer _signsV;
    private readonly VulkanDevice.Buffer _kScratch;     // shared across layers
    private readonly VulkanDevice.Buffer _vScratch;

    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _stride;            // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private readonly int _mseBits;
    private readonly int _codeUintsPerVec;
    private readonly float _invSqrtD;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;
    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;
    /// <summary>Number of layers in this cache.</summary>
    public int LayerCount => _numLayers;
    /// <inheritdoc/>
    public int KvStrideOf(int layerIndex) => _stride;
    /// <summary>MSE bits per coordinate.</summary>
    public int MseBits => _mseBits;

    /// <summary>Permanent compressed store bytes (codes + norms; excludes the shared scratch).</summary>
    public long AllocatedBytes =>
        (long)_numLayers * 2 * ((long)_maxSeqLen * _numKvHeads * _codeUintsPerVec * sizeof(uint)
                                + (long)_maxSeqLen * _numKvHeads * sizeof(float));

    /// <summary>
    /// Creates a uniform-geometry TurboQuant KV cache. The caller supplies the codec constants:
    /// <paramref name="centroids"/> (length <c>2^mseBits</c>, scaled by 1/√d), the per-coordinate
    /// RHT sign sets for keys/values (length <c>headDim</c>, ±1), <paramref name="invSqrtD"/>, and the
    /// MSE bit-width. Kernels are created from <paramref name="spvDir"/> and owned by this cache.
    /// </summary>
    public VulkanTurboQuantKvCache(
        VulkanDevice device, string spvDir,
        int numLayers, int numKvHeads, int headDim, int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
        : this(device,
               TurboQuantEncodeF32Kernel.Create(device, spvDir),
               TurboQuantDequantF32Kernel.Create(device, spvDir),
               ownsKernels: true,
               numLayers, numKvHeads, headDim, maxSeqLen, mseBits, centroids, signsK, signsV, invSqrtD)
    {
    }

    /// <summary>Creates the cache with caller-owned (shared) kernels — preferred when the model holds
    /// one encode/dequant kernel pair across all caches.</summary>
    public VulkanTurboQuantKvCache(
        VulkanDevice device, TurboQuantEncodeF32Kernel encode, TurboQuantDequantF32Kernel dequant, bool ownsKernels,
        int numLayers, int numKvHeads, int headDim, int maxSeqLen, int mseBits,
        ReadOnlySpan<float> centroids, ReadOnlySpan<float> signsK, ReadOnlySpan<float> signsV, float invSqrtD)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(encode);
        ArgumentNullException.ThrowIfNull(dequant);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > TurboQuantDequantF32Kernel.MaxHeadDim)
            throw new ArgumentException($"headDim must be a power of two in [1,256]; got {headDim}.", nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (mseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(mseBits));
        if (centroids.Length != (1 << mseBits)) throw new ArgumentException($"centroids must have 2^mseBits = {1 << mseBits} entries.", nameof(centroids));
        if (signsK.Length != headDim || signsV.Length != headDim) throw new ArgumentException("signs must have headDim entries.");

        _device = device;
        _encode = encode;
        _dequant = dequant;
        _ownsKernels = ownsKernels;
        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _stride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _mseBits = mseBits;
        _invSqrtD = invSqrtD;

        int codeBytes = (headDim * mseBits + 7) / 8;
        _codeUintsPerVec = (codeBytes + 3) / 4;

        // +1 guard uint per layer for the dequant straddle read of the final coordinate.
        long codeUintsPerLayer = (long)maxSeqLen * numKvHeads * _codeUintsPerVec + 1;
        long normsPerLayer = (long)maxSeqLen * numKvHeads;
        long scratchFloats = (long)maxSeqLen * _stride;

        _keyCodes = new VulkanDevice.Buffer[numLayers];
        _valueCodes = new VulkanDevice.Buffer[numLayers];
        _keyNorms = new VulkanDevice.Buffer[numLayers];
        _valueNorms = new VulkanDevice.Buffer[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _keyCodes[i] = device.Allocate(codeUintsPerLayer * sizeof(uint));
            _valueCodes[i] = device.Allocate(codeUintsPerLayer * sizeof(uint));
            _keyNorms[i] = device.Allocate(normsPerLayer * sizeof(float));
            _valueNorms[i] = device.Allocate(normsPerLayer * sizeof(float));
        }
        _kScratch = device.Allocate(scratchFloats * sizeof(float));
        _vScratch = device.Allocate(scratchFloats * sizeof(float));

        _centroids = device.Allocate((long)centroids.Length * sizeof(float));
        _signsK = device.Allocate((long)headDim * sizeof(float));
        _signsV = device.Allocate((long)headDim * sizeof(float));
        _device.Upload(centroids, _centroids);
        _device.Upload(signsK, _signsK);
        _device.Upload(signsV, _signsV);
    }

    /// <summary>Device buffer holding the dequantized keys for the last layer dequantized via
    /// <see cref="RecordDequant"/> / <see cref="DequantSync"/>. The attention kernel reads this.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer() => _kScratch;
    /// <summary>Device buffer holding the dequantized values (see <see cref="GetKeysBuffer"/>).</summary>
    internal VulkanDevice.Buffer GetValuesBuffer() => _vScratch;

    /// <summary>Records the encode of fresh device K/V into the layer's code/norm store. Positions
    /// must be contiguous ascending (prefill / single-token decode). Advances CurrentLength.</summary>
    internal void RecordUpdate(nint cmdBuf, VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
                               ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        int startPos = ValidateContiguous(positions, seqLen);
        _encode.Record(cmdBuf, kDev, _centroids, _signsK, _keyCodes[layerIndex], _keyNorms[layerIndex],
                       seqLen, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, startPos, _invSqrtD);
        _encode.Record(cmdBuf, vDev, _centroids, _signsV, _valueCodes[layerIndex], _valueNorms[layerIndex],
                       seqLen, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, startPos, _invSqrtD);
        int newLength = startPos + seqLen;
        if (newLength > _currentLength) _currentLength = newLength;
    }

    /// <summary>Records the dequant of <c>[0, CurrentLength)</c> for the layer into the shared K/V
    /// scratch buffers (read by attention via <see cref="GetKeysBuffer"/>/<see cref="GetValuesBuffer"/>).</summary>
    internal void RecordDequant(nint cmdBuf, int layerIndex)
    {
        int numVectors = _currentLength * _numKvHeads;
        _dequant.Record(cmdBuf, _keyCodes[layerIndex], _keyNorms[layerIndex], _centroids, _signsK, _kScratch,
                        numVectors, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, _invSqrtD);
        _dequant.Record(cmdBuf, _valueCodes[layerIndex], _valueNorms[layerIndex], _centroids, _signsV, _vScratch,
                        numVectors, _headDim, _numKvHeads, _mseBits, _codeUintsPerVec, _invSqrtD);
    }

    /// <summary>Synchronous encode of device K/V for one layer (one-shot submit). For tests / non-pipelined use.</summary>
    public void UpdateSync(VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
                           ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        RecordUpdate(ctx.CommandBuffer, kDev, vDev, positions, seqLen, layerIndex);
        ctx.SubmitAndWait();
    }

    /// <summary>Synchronous dequant of one layer into the scratch buffers (one-shot submit).</summary>
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
        if (positions.Length != seqLen) throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));
        int start = positions[0];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] != start + i)
                throw new NotSupportedException("VulkanTurboQuantKvCache requires contiguous ascending positions (prefill / decode).");
            if ((uint)positions[i] >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions), $"position {positions[i]} exceeds max {_maxSeqLen}.");
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

    // IKvCache host overloads are not used on the Vulkan device path (mirrors VulkanKvCache).
    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache is updated via RecordUpdate from the Vulkan forward pass.");
    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache is updated via RecordUpdate from the Vulkan forward pass.");
    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache exposes device buffers, not TensorRef.");
    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache exposes device buffers, not TensorRef.");
    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache does not materialise cached keys as CPU tensors.");
    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanTurboQuantKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keyCodes[i]?.Dispose();
            _valueCodes[i]?.Dispose();
            _keyNorms[i]?.Dispose();
            _valueNorms[i]?.Dispose();
        }
        _centroids?.Dispose();
        _signsK?.Dispose();
        _signsV?.Dispose();
        _kScratch?.Dispose();
        _vScratch?.Dispose();
        if (_ownsKernels)
        {
            _encode.Dispose();
            _dequant.Dispose();
        }
    }
}
