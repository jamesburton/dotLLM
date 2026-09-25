using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Issue #474 — one compiled variant of <c>matmul_pq2_0_f32_gemv_multirow.glsl</c>: <see cref="Rows"/>
/// output rows and up to <see cref="Columns"/> activation columns per workgroup.
/// </summary>
/// <remarks>
/// Owns its module, pipeline, descriptor pool and handle-keyed <see cref="DescriptorSetCache"/>.
/// The owner must forward <see cref="InvalidateDescriptorCache"/> (the #467 recycled-handle hazard).
/// </remarks>
internal sealed class PQ2_0GemvMultiRowPipeline : IDisposable
{
    private const int PushConstantBytes = 7 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private bool _disposed;

    /// <summary>Output rows per workgroup.</summary>
    public int Rows { get; }

    /// <summary>Compiled column capacity; a dispatch may use 1..<see cref="Columns"/> live columns.</summary>
    public int Columns { get; }

    /// <summary>The SPIR-V file this pipeline was built from.</summary>
    public string SpvFileName { get; }

    private PQ2_0GemvMultiRowPipeline(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool,
        int rows, int columns, string spvFileName)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _pool = pool;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
        Rows = rows;
        Columns = columns;
        SpvFileName = spvFileName;
    }

    /// <summary>Loads <paramref name="spvFileName"/> from <paramref name="spvDir"/>.</summary>
    public static PQ2_0GemvMultiRowPipeline Create(
        VulkanDevice device, string spvDir, string spvFileName, int rows, int columns)
    {
        if (rows < 1) throw new ArgumentOutOfRangeException(nameof(rows));
        if (columns < 1 || columns > MatMulPQ2_0GemvF32Kernel.MaxColumns) throw new ArgumentOutOfRangeException(nameof(columns));

        string path = Path.Combine(spvDir, spvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            pipeline = module.CreateComputePipeline(
                entryPoint: "main", bindings: bindings, pushConstantBytes: PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new PQ2_0GemvMultiRowPipeline(device, module, pipeline, pool, rows, columns, spvFileName);
    }

    /// <summary>Drops every cached descriptor set (see #467).</summary>
    public void InvalidateDescriptorCache() => _cache.Reset();

    /// <summary>
    /// Records <c>y[s, :] = W @ x[s, :]</c> for <paramref name="columns"/> columns. Arguments are
    /// validated by the caller (<see cref="MatMulPQ2_0GemvF32Kernel"/>).
    /// </summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsPQ2_0, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k, int columns, int xOffsetElements, int yOffsetElements)
    {
        int blocksPerRow = k / MatMulPQ2_0GemvF32Kernel.PQ2_0GroupSize;
        long rowBytes = (long)blocksPerRow * MatMulPQ2_0GemvF32Kernel.PQ2_0GroupBytes;

        Span<nint> buffers = stackalloc nint[3] { weightsPQ2_0.Handle, x.Handle, y.Handle };
        nint descriptorSet = _cache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[7]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)((rowBytes + 3) / 4),
            (uint)xOffsetElements,
            (uint)yOffsetElements,
            (uint)columns,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Rows - 1) / Rows), 1, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0)
            VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
