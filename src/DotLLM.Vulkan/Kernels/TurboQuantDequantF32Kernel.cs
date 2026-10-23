using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// TurboQuant (MSE stage) KV dequantization to FP32 — the GPU port of
/// <c>DotLLM.Engine.KvCache.Codecs.TurboQuantCodec.Decode</c> (without the QJL residual).
/// Reads per-head-vector packed Lloyd–Max codes + an fp32 norm and writes the contiguous
/// fp32 attention scratch <c>[positions, numKvHeads*headDim]</c>.
/// </summary>
/// <remarks>
/// Per head-vector: centroid lookup (codes → <c>centroids</c>) → unnormalized Walsh–Hadamard →
/// <c>× invSqrtD × signs × norm</c>. <c>centroids</c> (already scaled by <c>1/√d</c>), <c>signs</c>
/// (±1) and <c>invSqrtD</c> are codec constants supplied as buffers / push constants, so the
/// kernel stays backend-pure (no dependency on the codec type). One workgroup per head-vector,
/// 256 threads, one coordinate per thread. <b>Supports <c>headDim</c> a power of two ≤ 256</b>
/// (128/256 cover the GQA models that pick TurboQuant; 512 would need a cooperative variant).
/// </remarks>
public sealed class TurboQuantDequantF32Kernel : IDisposable
{
    /// <summary>Maximum supported per-head dimension (workgroup size / shared-array bound).</summary>
    public const int MaxHeadDim = 256;

    private const int BindingCount = 5;
    private const int PushConstantBytes = 5 * sizeof(uint); // 4 uints + 1 float

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private TurboQuantDequantF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: BindingCount);
    }

    /// <summary>Loads <c>turboquant_dequant_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static TurboQuantDequantF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "turboquant_dequant_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[BindingCount];
            for (int i = 0; i < BindingCount; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: BindingCount);
        return new TurboQuantDequantF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Dispatches the dequant synchronously.</summary>
    public void Launch(
        VulkanDevice.Buffer codes, VulkanDevice.Buffer norms, VulkanDevice.Buffer centroids,
        VulkanDevice.Buffer signs, VulkanDevice.Buffer dst,
        int numVectors, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, float invSqrtD)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, codes, norms, centroids, signs, dst,
               numVectors, headDim, numKvHeads, mseBits, codeUintsPerVec, invSqrtD);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dequant into <paramref name="cmdBuf"/> without submitting. One workgroup
    /// per head-vector (<paramref name="numVectors"/> = positions × numKvHeads).</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer codes, VulkanDevice.Buffer norms, VulkanDevice.Buffer centroids,
        VulkanDevice.Buffer signs, VulkanDevice.Buffer dst,
        int numVectors, int headDim, int numKvHeads, int mseBits, int codeUintsPerVec, float invSqrtD)
    {
        if (numVectors <= 0) throw new ArgumentOutOfRangeException(nameof(numVectors));
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim)
            throw new ArgumentException($"headDim must be a power of two in [1,{MaxHeadDim}]; got {headDim}.", nameof(headDim));
        if (numKvHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numKvHeads));
        if (mseBits is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(mseBits));
        if ((numVectors % numKvHeads) != 0)
            throw new ArgumentException("numVectors must be a multiple of numKvHeads.", nameof(numVectors));

        // The shader reads codes[widx+1] when a code straddles a uint boundary; require the codes
        // buffer to cover every vector's uint stride plus one guard uint for that final read.
        long codeUintsMin = (long)numVectors * codeUintsPerVec + 1;
        if (codes.Size < codeUintsMin * sizeof(uint))
            throw new ArgumentException($"codes buffer too small: need >= {codeUintsMin * sizeof(uint)} bytes.", nameof(codes));
        if (norms.Size < (long)numVectors * sizeof(float))
            throw new ArgumentException("norms buffer too small.", nameof(norms));
        if (centroids.Size < (long)(1 << mseBits) * sizeof(float))
            throw new ArgumentException("centroids buffer too small.", nameof(centroids));
        if (signs.Size < (long)headDim * sizeof(float))
            throw new ArgumentException("signs buffer too small.", nameof(signs));
        int positions = numVectors / numKvHeads;
        long dstMin = (long)positions * numKvHeads * headDim * sizeof(float);
        if (dst.Size < dstMin)
            throw new ArgumentException($"dst buffer too small: need >= {dstMin} bytes.", nameof(dst));

        Span<nint> buffers = stackalloc nint[BindingCount]
            { codes.Handle, norms.Handle, centroids.Handle, signs.Handle, dst.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)headDim, (uint)numKvHeads, (uint)mseBits, (uint)codeUintsPerVec,
            BitConverter.SingleToUInt32Bits(invSqrtD),
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)numVectors, 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_descriptorPool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
