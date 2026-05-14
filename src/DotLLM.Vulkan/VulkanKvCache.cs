using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan-side KV cache. Per-layer device-resident buffers of shape
/// <c>[maxSeqLen, numKvHeads * headDim]</c> FP32, host-visible and
/// host-coherent (scaffold memory type — no staging ring yet).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <c>DotLLM.Engine.KvCache.SimpleKvCache</c> semantics: <c>Update</c>
/// appends new K/V rows at the supplied position indices. The attention kernel
/// reads straight from the device buffers via <see cref="GetKeysBuffer"/> /
/// <see cref="GetValuesBuffer"/>; no staging copies are required.
/// </para>
/// <para>
/// Implements <see cref="IKvCache"/> so code that already knows about the CPU
/// cache semantics (text-generation loop, tests) can swap the backing store
/// transparently. The <see cref="IKvCache.Update(ITensor, ITensor, ReadOnlySpan{int}, int)"/>
/// and <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/>
/// overloads expect CPU-resident tensor pointers (the caller is responsible
/// for uploading); we only use the device-side path from
/// <see cref="VulkanTransformerModel"/>, but the IKvCache contract lets this
/// object satisfy the same API.
/// </para>
/// </remarks>
public sealed class VulkanKvCache : IKvCache
{
    private readonly VulkanDevice _device;
    private readonly VulkanDevice.Buffer[] _keys;
    private readonly VulkanDevice.Buffer[] _values;
    private readonly int _numLayers;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly int _kvStride;
    private int _currentLength;
    private bool _disposed;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Creates the per-layer K/V buffers. Memory is not zeroed — the forward pass only reads positions it has written.</summary>
    public VulkanKvCache(VulkanDevice device, int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));
        if (maxSeqLen <= 0) throw new ArgumentOutOfRangeException(nameof(maxSeqLen));

        _device = device;
        _numLayers = numLayers;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _kvStride = numKvHeads * headDim;

        _keys = new VulkanDevice.Buffer[numLayers];
        _values = new VulkanDevice.Buffer[numLayers];

        long bytesPerLayer = (long)maxSeqLen * _kvStride * sizeof(float);
        for (int i = 0; i < numLayers; i++)
        {
            _keys[i] = device.Allocate(bytesPerLayer);
            _values[i] = device.Allocate(bytesPerLayer);
        }
    }

    /// <summary>Returns the device buffer holding cached keys for the given layer.</summary>
    internal VulkanDevice.Buffer GetKeysBuffer(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns the device buffer holding cached values for the given layer.</summary>
    internal VulkanDevice.Buffer GetValuesBuffer(int layerIndex) => _values[layerIndex];

    /// <summary>
    /// Copies new <paramref name="kDev"/> / <paramref name="vDev"/> rows into
    /// the cached K/V buffers at the given positions. Both inputs are
    /// device-resident FP32 buffers with shape <c>[seqLen, kvStride]</c>.
    /// </summary>
    /// <remarks>
    /// The scaffold path writes via host-mapped pointers: both the source and
    /// destination buffers are host-visible host-coherent, so a single
    /// <c>Buffer.MemoryCopy</c> per position suffices. A proper implementation
    /// would use <c>vkCmdCopyBuffer</c> on the compute queue (TODO alongside
    /// the staging ring / device-local memory migration).
    /// </remarks>
    internal unsafe void UpdateDevice(
        VulkanDevice.Buffer kDev, VulkanDevice.Buffer vDev,
        ReadOnlySpan<int> positions, int seqLen, int layerIndex)
    {
        if (positions.Length != seqLen)
            throw new ArgumentException("positions.Length must equal seqLen", nameof(positions));

        int rowBytes = _kvStride * sizeof(float);

        // Map sources and destinations. All four mappings operate on the same
        // device, and Vulkan lets a buffer be mapped at most once at a time —
        // so we map all four here and copy row by row.
        VulkanApi.vkMapMemory(_device.Handle, kDev.Memory, 0, (ulong)(seqLen * rowBytes), 0, out nint kSrcMapped)
            .ThrowOnError("vkMapMemory kv.kDev");
        VulkanApi.vkMapMemory(_device.Handle, vDev.Memory, 0, (ulong)(seqLen * rowBytes), 0, out nint vSrcMapped)
            .ThrowOnError("vkMapMemory kv.vDev");
        VulkanApi.vkMapMemory(_device.Handle, _keys[layerIndex].Memory, 0, (ulong)_keys[layerIndex].Size, 0, out nint kDstMapped)
            .ThrowOnError("vkMapMemory kv.keys[layer]");
        VulkanApi.vkMapMemory(_device.Handle, _values[layerIndex].Memory, 0, (ulong)_values[layerIndex].Size, 0, out nint vDstMapped)
            .ThrowOnError("vkMapMemory kv.values[layer]");

        int maxPos = -1;
        try
        {
            byte* kSrc = (byte*)kSrcMapped;
            byte* vSrc = (byte*)vSrcMapped;
            byte* kDst = (byte*)kDstMapped;
            byte* vDst = (byte*)vDstMapped;

            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                if ((uint)pos >= (uint)_maxSeqLen)
                    throw new ArgumentOutOfRangeException(nameof(positions),
                        $"Position {pos} exceeds max cache length {_maxSeqLen}.");
                if (pos > maxPos) maxPos = pos;

                System.Buffer.MemoryCopy(kSrc + i * rowBytes, kDst + (long)pos * rowBytes, rowBytes, rowBytes);
                System.Buffer.MemoryCopy(vSrc + i * rowBytes, vDst + (long)pos * rowBytes, rowBytes, rowBytes);
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, _values[layerIndex].Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, _keys[layerIndex].Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, vDev.Memory);
            VulkanApi.vkUnmapMemory(_device.Handle, kDev.Memory);
        }

        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException(
            "VulkanKvCache is updated via UpdateDevice from the Vulkan forward pass; the host-side Update overload is not supported.");

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetKeysBuffer, not TensorRef.");

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache exposes device buffers via GetValuesBuffer, not TensorRef.");

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached keys as CPU tensors.");

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
        => throw new NotSupportedException("VulkanKvCache does not materialise cached values as CPU tensors.");

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <summary>Resets the visible length. Used when starting a new sequence.</summary>
    public void Reset() => _currentLength = 0;

    /// <summary>
    /// Number of layers in this cache.
    /// </summary>
    public int LayerCount => _numLayers;

    /// <summary>
    /// Per-row stride (<c>numKvHeads * headDim</c>, FP32 elements).
    /// </summary>
    public int KvStride => _kvStride;

    /// <summary>
    /// Ingests host-resident K/V rows (FP32, layout <c>[length, kvStride]</c>)
    /// for the given <paramref name="layerIndex"/> at positions <c>[0, length)</c>.
    /// Used by the hybrid CPU-prefill / iGPU-decode handoff: after CPU prefill
    /// has populated a <c>SimpleKvCache</c>, each layer's contiguous host buffer
    /// is uploaded into the device-local Vulkan KV cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In this scaffold parent every KV layer buffer is host-visible
    /// host-coherent (see <see cref="VulkanDevice.Allocate"/>), so the handoff
    /// is a single mapped <c>memcpy</c> per K/V buffer via
    /// <see cref="VulkanDevice.Upload(System.ReadOnlySpan{float}, VulkanDevice.Buffer)"/>.
    /// When the scaffold gains a staged device-local KV cache in a later wave
    /// this method becomes the natural seam for a <c>vkCmdCopyBuffer</c>
    /// transfer (staging buffer → device-local destination) — the public
    /// contract (per-layer ingest, advances <see cref="CurrentLength"/>) is
    /// stable across both backings.
    /// </para>
    /// <para>
    /// Advances <see cref="CurrentLength"/> to <c>max(CurrentLength, length)</c>
    /// so the subsequent device decode sees positions <c>[0, length)</c> as
    /// already-cached. Both <c>keys</c> and <c>values</c> must cover exactly
    /// <c>length × KvStride</c> FP32 elements.
    /// </para>
    /// </remarks>
    public void IngestFromHost(int layerIndex, int length,
        ReadOnlySpan<float> keys, ReadOnlySpan<float> values)
    {
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), "length must be positive.");
        if (length > _maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length),
                $"length {length} exceeds cache MaxLength {_maxSeqLen}.");
        long expectedFloats = (long)length * _kvStride;
        if (keys.Length != expectedFloats || values.Length != expectedFloats)
            throw new ArgumentException(
                $"keys/values must contain exactly length × kvStride = {expectedFloats} floats; "
                + $"got keys={keys.Length}, values={values.Length}.");

        // Direct mapped-memory upload into each layer's host-visible buffer.
        // On a UMA APU (Strix Halo) the bytes never leave system DRAM.
        _device.Upload(keys, _keys[layerIndex]);
        _device.Upload(values, _values[layerIndex]);

        if (length > _currentLength)
            _currentLength = length;
    }

    /// <summary>
    /// Sets the visible length without changing buffer contents. Used after
    /// <see cref="IngestFromHost"/> calls for every layer to advance the
    /// observed length atomically across layers (the per-layer call already
    /// advances individually; this is a no-op for single-layer ingest but
    /// makes the multi-layer code path explicit at the call site).
    /// </summary>
    public void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _keys[i]?.Dispose();
            _values[i]?.Dispose();
        }
    }
}
