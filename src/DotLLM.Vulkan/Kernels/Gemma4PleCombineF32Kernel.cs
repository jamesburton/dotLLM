using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Gemma-4 dense-PLE (E2B/E4B) per-layer-input combine (#916): device side of
/// <c>PerLayerEmbeddings.ComputeInputs</c> after its context-projection GEMM. For every (token, layer) block of
/// <c>pleDim</c> floats: <c>out = (rmsnorm(proj * projScale) * projNorm + identity) * combineScale</c>, written
/// LAYER-MAJOR <c>[layers, seq, pleDim]</c> so each layer's injection reads one contiguous slice.
/// </summary>
public sealed class Gemma4PleCombineF32Kernel : IDisposable
{
    private const int PushConstantBytes = 6 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private Gemma4PleCombineF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>gemma4_ple_combine_f32.spv</c> from the given directory and creates the pipeline.</summary>
    public static Gemma4PleCombineF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "gemma4_ple_combine_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (uint i = 0; i < 4; i++) bindings[(int)i] = new VkDescriptorBinding(i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new Gemma4PleCombineF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Records the combine into <paramref name="cmdBuf"/> without submitting.</summary>
    /// <param name="proj">Context projection <c>[seq, layers*pleDim]</c> (token-major, un-scaled GEMM output).</param>
    /// <param name="identity">Token-identity rows <c>[seq, layers*pleDim]</c>, already multiplied by sqrt(pleDim).</param>
    /// <param name="projNorm"><c>[pleDim]</c> per-layer projection norm weight.</param>
    /// <param name="output">Layer-major result <c>[layers, seq, pleDim]</c>.</param>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="seq">Token rows.</param>
    /// <param name="layers">Decoder layers.</param>
    /// <param name="pleDim">Per-layer embedding width.</param>
    /// <param name="projScale">Scale applied to the projection before the RMSNorm (1/sqrt(hidden)).</param>
    /// <param name="eps">RMSNorm epsilon.</param>
    /// <param name="combineScale">Scale applied to (norm + identity) (1/sqrt(2)).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer proj, VulkanDevice.Buffer identity, VulkanDevice.Buffer projNorm, VulkanDevice.Buffer output,
        int seq, int layers, int pleDim, float projScale, float eps, float combineScale)
    {
        if (seq <= 0 || layers <= 0 || pleDim <= 0) throw new ArgumentOutOfRangeException(nameof(seq));
        long elems = (long)seq * layers * pleDim;
        long bytes = elems * sizeof(float);
        if (proj.Size < bytes) throw new ArgumentException("Proj buffer too small.", nameof(proj));
        if (identity.Size < bytes) throw new ArgumentException("Identity buffer too small.", nameof(identity));
        if (output.Size < bytes) throw new ArgumentException("Output buffer too small.", nameof(output));
        if (projNorm.Size < (long)pleDim * sizeof(float)) throw new ArgumentException("Norm buffer too small.", nameof(projNorm));

        Span<nint> buffers = stackalloc nint[4] { proj.Handle, identity.Handle, projNorm.Handle, output.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> push = stackalloc uint[6]
        {
            (uint)pleDim, (uint)layers, (uint)seq,
            BitConverter.SingleToUInt32Bits(projScale), BitConverter.SingleToUInt32Bits(eps), BitConverter.SingleToUInt32Bits(combineScale),
        };
        fixed (uint* p = push)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)p);

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)(seq * layers), 1, 1);
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
