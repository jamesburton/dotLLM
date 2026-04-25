using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side expanded MLA KV cache (DeepSeek-V2/V3). Per-layer device-local
/// buffers for per-head <c>K_nope</c>, per-head <c>V</c>, and the MQA-shared
/// <c>K_pe</c> (one rope-K per token, broadcast across heads — RoPE already
/// applied on write).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the CPU <c>DotLLM.Models.Architectures.MlaExpandedKvState</c>
/// layout one-for-one so the kernel can record straight <c>vkCmdCopyBuffer</c>
/// rows from the per-step activation buffers into the cache without shape
/// translation. Three physical buffers per layer:
/// <list type="bullet">
///   <item><c>K_nope[layer]</c> : <c>[maxSeqLen, numHeads * qkNopeHeadDim]</c></item>
///   <item><c>V[layer]</c>      : <c>[maxSeqLen, numHeads * vHeadDim]</c></item>
///   <item><c>K_pe[layer]</c>   : <c>[maxSeqLen, qkRopeHeadDim]</c> — shared across heads</item>
/// </list>
/// </para>
/// <para>
/// First-pass cache form is the "expanded" Phase A layout — the latent
/// MLA cache (<c>c_kv + k_pe</c>, ~7× smaller) is a follow-up that
/// requires the absorbed-form attention kernel and is out of scope for
/// the Vulkan port's correctness wave.
/// </para>
/// <para>
/// Implements <see cref="IKvCache"/> with the same not-supported pattern
/// as <see cref="VulkanKvCache"/> — host-side <c>Update</c> overloads are
/// rejected; updates flow exclusively through <c>RecordUpdate</c>
/// on the forward-pass command buffer. <c>GetKNopeBuffer</c>-style
/// accessors return the underlying VRAM buffers for the kernel layer.
/// </para>
/// </remarks>
public sealed class MlaVulkanKvCache : IKvCache, IDisposable
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _kNopeBuffers;
    private readonly VulkanDevice.Buffer[] _vBuffers;
    private readonly VulkanDevice.Buffer[] _kPeBuffers;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _qkNopeHeadDim;
    private readonly int _vHeadDim;
    private readonly int _qkRopeHeadDim;
    private readonly int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Number of MLA-bearing layers in the model.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Per-token byte size of one layer's K_nope row.</summary>
    public int KNopeRowBytes => _numHeads * _qkNopeHeadDim * sizeof(float);

    /// <summary>Per-token byte size of one layer's V row.</summary>
    public int VRowBytes => _numHeads * _vHeadDim * sizeof(float);

    /// <summary>Per-token byte size of one layer's K_pe row.</summary>
    public int KPeRowBytes => _qkRopeHeadDim * sizeof(float);

    /// <summary>
    /// Total VRAM footprint across all layers at the configured max seq len.
    /// </summary>
    public long AllocatedBytes
    {
        get
        {
            long perLayerBytes =
                (long)_maxSeqLen * (KNopeRowBytes + VRowBytes + KPeRowBytes);
            return _numLayers * perLayerBytes;
        }
    }

    /// <summary>Allocates per-layer K_nope / V / K_pe buffers sized for <paramref name="maxSeqLen"/>.</summary>
    /// <param name="device">Vulkan device that owns the allocations.</param>
    /// <param name="numLayers">Number of MLA-bearing layers in the model.</param>
    /// <param name="maxSeqLen">Maximum cached sequence length.</param>
    /// <param name="numHeads">Number of attention heads (per the MLA config; numHeads = numKvHeads).</param>
    /// <param name="qkNopeHeadDim">Per-head non-rope K dimension.</param>
    /// <param name="vHeadDim">Per-head V dimension.</param>
    /// <param name="qkRopeHeadDim">Shared rope-K dimension (broadcast across heads).</param>
    public MlaVulkanKvCache(
        VulkanDevice device,
        int numLayers, int maxSeqLen,
        int numHeads, int qkNopeHeadDim, int vHeadDim, int qkRopeHeadDim)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));
        if (qkRopeHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(qkRopeHeadDim));

        _device = device;
        _numLayers = numLayers;
        _maxSeqLen = maxSeqLen;
        _numHeads = numHeads;
        _qkNopeHeadDim = qkNopeHeadDim;
        _vHeadDim = vHeadDim;
        _qkRopeHeadDim = qkRopeHeadDim;

        _kNopeBuffers = new VulkanDevice.Buffer[numLayers];
        _vBuffers = new VulkanDevice.Buffer[numLayers];
        _kPeBuffers = new VulkanDevice.Buffer[numLayers];

        long kNopeBytesPerLayer = (long)maxSeqLen * KNopeRowBytes;
        long vBytesPerLayer = (long)maxSeqLen * VRowBytes;
        long kPeBytesPerLayer = (long)maxSeqLen * KPeRowBytes;

        for (int i = 0; i < numLayers; i++)
        {
            _kNopeBuffers[i] = device.AllocateDeviceLocal(kNopeBytesPerLayer);
            _vBuffers[i] = device.AllocateDeviceLocal(vBytesPerLayer);
            _kPeBuffers[i] = device.AllocateDeviceLocal(kPeBytesPerLayer);
        }
    }

    /// <summary>Cached K_nope buffer for the given layer.</summary>
    internal VulkanDevice.Buffer GetKNopeBuffer(int layerIndex) => _kNopeBuffers[layerIndex];

    /// <summary>Cached V buffer for the given layer.</summary>
    internal VulkanDevice.Buffer GetVBuffer(int layerIndex) => _vBuffers[layerIndex];

    /// <summary>Cached K_pe buffer for the given layer.</summary>
    internal VulkanDevice.Buffer GetKPeBuffer(int layerIndex) => _kPeBuffers[layerIndex];

    /// <summary>
    /// Appends K_nope / V / K_pe copy commands onto <paramref name="cmdBuf"/>.
    /// The caller owns the <c>TRANSFER → COMPUTE</c> barrier that follows.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="kNopeDev">Per-step K_nope activation [seqLen, numHeads * qkNopeHeadDim].</param>
    /// <param name="vDev">Per-step V activation [seqLen, numHeads * vHeadDim].</param>
    /// <param name="kPeDev">Per-step K_pe activation [seqLen, qkRopeHeadDim] — RoPE-applied.</param>
    /// <param name="positions">Token positions (must be contiguous ascending — not validated for now).</param>
    /// <param name="seqLen">Number of new tokens being appended.</param>
    /// <param name="layerIndex">Target layer.</param>
    internal unsafe void RecordUpdate(
        nint cmdBuf,
        VulkanDevice.Buffer kNopeDev, VulkanDevice.Buffer vDev, VulkanDevice.Buffer kPeDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int startPos = positions[0];
        int maxPos = startPos;
        bool contiguous = true;
        for (int i = 1; i < seqLen; i++)
        {
            if (positions[i] != positions[i - 1] + 1) contiguous = false;
            if (positions[i] > maxPos) maxPos = positions[i];
        }
        if (!contiguous)
            throw new NotSupportedException(
                "MlaVulkanKvCache only supports contiguous-ascending position updates; got non-contiguous positions.");
        if ((uint)maxPos >= (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions),
                $"Position {maxPos} exceeds max cache length {_maxSeqLen}.");

        // K_nope copy
        var kRegion = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = (ulong)((long)startPos * KNopeRowBytes),
            size = (ulong)KNopeRowBytes * (ulong)seqLen,
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, kNopeDev.Handle, _kNopeBuffers[layerIndex].Handle, 1, kRegion);

        // V copy
        var vRegion = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = (ulong)((long)startPos * VRowBytes),
            size = (ulong)VRowBytes * (ulong)seqLen,
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, vDev.Handle, _vBuffers[layerIndex].Handle, 1, vRegion);

        // K_pe copy (shared across heads — single contiguous slab per token)
        var peRegion = new VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = (ulong)((long)startPos * KPeRowBytes),
            size = (ulong)KPeRowBytes * (ulong)seqLen,
        };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, kPeDev.Handle, _kPeBuffers[layerIndex].Handle, 1, peRegion);

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "MlaVulkanKvCache is updated via RecordUpdate from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "MlaVulkanKvCache is updated via RecordUpdate from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("MlaVulkanKvCache exposes device buffers via GetKNopeBuffer / GetKPeBuffer, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("MlaVulkanKvCache exposes device buffers via GetVBuffer, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("MlaVulkanKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("MlaVulkanKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _kNopeBuffers[i]?.Dispose();
            _vBuffers[i]?.Dispose();
            _kPeBuffers[i]?.Dispose();
        }
    }
}
