using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan;

namespace DotLLM.Cuda;

/// <summary>
/// Split KV-cache for hybrid Vulkan+CUDA inference. Routes layers
/// 0..<see cref="NumVulkanLayers"/>-1 to a GPU-resident
/// <see cref="VulkanKvCache"/> and layers
/// <see cref="NumVulkanLayers"/>..<c>L-1</c> to a
/// <see cref="CudaKvCache"/>. Vulkan layers are updated via
/// <see cref="VulkanCache"/> directly; CUDA layers via
/// <see cref="CudaCache"/> using 0-based (per-cache) layer indices.
/// </summary>
public sealed class VulkanCudaKvCache : IKvCache
{
    private readonly int _numVulkanLayers;

    /// <summary>Vulkan-side KV-cache for layers 0..N-1 (FP32, device memory).</summary>
    internal VulkanKvCache VulkanCache { get; }

    /// <summary>CUDA-side KV-cache for layers N..L-1 (FP16, device memory).</summary>
    internal CudaKvCache CudaCache { get; }

    /// <summary>Number of transformer layers assigned to the Vulkan device.</summary>
    public int NumVulkanLayers => _numVulkanLayers;

    /// <inheritdoc/>
    public int CurrentLength
    {
        get
        {
            Debug.Assert(VulkanCache.CurrentLength == CudaCache.CurrentLength,
                "Vulkan and CUDA KV-caches must advance in lockstep.");
            return CudaCache.CurrentLength;
        }
    }

    /// <inheritdoc/>
    public int MaxLength => CudaCache.MaxLength;

    /// <summary>
    /// Creates a Vulkan+CUDA split KV-cache.
    /// </summary>
    /// <param name="vulkanCache">Vulkan cache for the first <paramref name="numVulkanLayers"/> layers.</param>
    /// <param name="cudaCache">CUDA cache for the remaining layers.</param>
    /// <param name="numVulkanLayers">Number of layers handled by the Vulkan cache.</param>
    public VulkanCudaKvCache(VulkanKvCache vulkanCache, CudaKvCache cudaCache, int numVulkanLayers)
    {
        ArgumentNullException.ThrowIfNull(vulkanCache);
        ArgumentNullException.ThrowIfNull(cudaCache);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numVulkanLayers);

        VulkanCache = vulkanCache;
        CudaCache = cudaCache;
        _numVulkanLayers = numVulkanLayers;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// CPU-tensor update path. Vulkan layers must be updated via
    /// <see cref="VulkanCache"/> directly; CUDA layers are remapped to
    /// 0-based indices for <see cref="CudaCache"/>.
    /// </remarks>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a Vulkan layer — use VulkanCache directly.");

        // CudaKvCache.Update(ITensor) is the IKvCache host-copy path — not available
        // for device-side CudaKvCache; callers must use UpdateDevice. Surface a clear error.
        throw new NotSupportedException(
            "Use CudaCache.UpdateDevice() for CUDA layers in a VulkanCudaKvCache.");
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a Vulkan layer — use VulkanCache directly.");

        throw new NotSupportedException(
            "Use CudaCache.UpdateDevice() for CUDA layers in a VulkanCudaKvCache.");
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a Vulkan layer — use VulkanCache.GetKeys().");

        throw new NotSupportedException(
            "Use CudaCache.GetKeysPtr() for CUDA layers in a VulkanCudaKvCache.");
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            throw new InvalidOperationException(
                $"Layer {layerIndex} is a Vulkan layer — use VulkanCache.GetValues().");

        throw new NotSupportedException(
            "Use CudaCache.GetValuesPtr() for CUDA layers in a VulkanCudaKvCache.");
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            return VulkanCache.GetKeysRef(layerIndex);

        return CudaCache.GetKeysRef(layerIndex - _numVulkanLayers);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        if (layerIndex < _numVulkanLayers)
            return VulkanCache.GetValuesRef(layerIndex);

        return CudaCache.GetValuesRef(layerIndex - _numVulkanLayers);
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        VulkanCache.Rollback(length);
        CudaCache.Rollback(length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        VulkanCache.Dispose();
        CudaCache.Dispose();
    }
}
