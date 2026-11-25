using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// One compiled Olala compute pipeline (issue #1983): a SPIR-V blob from <c>native/vulkan/spv/olala_*.spv</c>, N storage-buffer bindings, a
/// block of 32-bit push constants and optional specialization constants, with its own descriptor-set cache. The Olala kernels share this
/// one wrapper instead of a class each: their host side is identical (bind buffers, push words, dispatch) and every semantic lives in the shader.
/// </summary>
internal sealed class OlalaKernel : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private readonly int _buffers;
    private readonly uint _pushBytes;
    private bool _disposed;

    /// <summary>Shader name (file stem), for diagnostics.</summary>
    public string Name { get; }

    /// <summary>Loads <paramref name="name"/>.spv from <paramref name="spvDir"/> and builds the pipeline.</summary>
    /// <param name="device">Device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="name">Shader file stem, e.g. <c>olala_gemv_q8_0</c>.</param>
    /// <param name="buffers">Storage-buffer bindings 0..buffers-1.</param>
    /// <param name="pushWords">Number of 32-bit push-constant words.</param>
    /// <param name="spec">Specialization constants for ids 0, 1, ...</param>
    public OlalaKernel(VulkanDevice device, string spvDir, string name, int buffers, int pushWords, ReadOnlySpan<uint> spec = default)
    {
        string path = Path.Combine(spvDir, name + ".spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Build the DotLLM.Vulkan project (glslc compiles native/vulkan/shaders/{name}.comp).");
        _device = device;
        Name = name;
        _buffers = buffers;
        _pushBytes = (uint)(pushWords * sizeof(uint));
        _module = VulkanModule.LoadFromFile(device, path);
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffers];
            for (int i = 0; i < buffers; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            _pipeline = _module.CreateComputePipeline("main", bindings, _pushBytes, specConstants: spec);
        }
        catch
        {
            _module.Dispose();
            throw;
        }
        _pool = KernelSupport.CreateDescriptorPool(device, (uint)buffers);
        _cache = new DescriptorSetCache(device, _pool, _pipeline, buffers);
    }

    /// <summary>Records one dispatch into <paramref name="cmdBuf"/> (no barrier: the caller orders dependent dispatches).</summary>
    public unsafe void Record(nint cmdBuf, ReadOnlySpan<VulkanDevice.Buffer> bufs, ReadOnlySpan<uint> push, uint gx, uint gy, uint gz)
    {
        if (bufs.Length != _buffers) throw new ArgumentException($"{Name}: expected {_buffers} buffers, got {bufs.Length}.");
        if (push.Length * sizeof(uint) != _pushBytes) throw new ArgumentException($"{Name}: expected {_pushBytes / 4} push words, got {push.Length}.");
        Span<nint> handles = stackalloc nint[_buffers];
        for (int i = 0; i < _buffers; i++) handles[i] = bufs[i].Handle;
        nint set = _cache.GetOrCreate(handles);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        if (_pushBytes > 0)
            fixed (uint* pc = push)
                VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, _pushBytes, (nint)pc);
        if (gx > 0 && gy > 0 && gz > 0) VulkanApi.vkCmdDispatch(cmdBuf, gx, gy, gz);
    }

    /// <summary>Drops cached descriptor sets (call after re-allocating buffers).</summary>
    public void InvalidateDescriptorCache() => _cache.Reset();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
