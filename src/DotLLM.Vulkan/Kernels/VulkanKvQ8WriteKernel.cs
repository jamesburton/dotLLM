using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>Writes F32 K / V rows into a block-quantised (Q8_0 or Q4_0, #1884) <see cref="VulkanKvCache"/> (the device-side quantise the cache records).</summary>
internal interface IVulkanKvQ8Writer
{
    /// <summary>
    /// Records the quantisation of <paramref name="count"/> F32 elements from the start of <paramref name="src"/> into Q8_0 blocks (36 bytes per 32
    /// elements: F32 scale = amax / 127, then 32 int8 = roundEven(x / scale)) starting at element <paramref name="dstElemOffset"/> of <paramref name="dst"/>.
    /// </summary>
    /// <param name="cmd">The command buffer.</param>
    /// <param name="src">F32 source rows.</param>
    /// <param name="dst">The Q8_0 (uint-backed) destination.</param>
    /// <param name="dstElemOffset">Destination element offset (a multiple of 32).</param>
    /// <param name="count">Elements to quantise (a multiple of 32).</param>
    void RecordF32ToQ8(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long dstElemOffset, long count);

    /// <summary>The block format this writer produces (<see cref="VulkanKvStorage.Q8_0"/> or <see cref="VulkanKvStorage.Q4_0"/>).</summary>
    VulkanKvStorage Storage { get; }

    /// <summary>True when <see cref="RecordScatterRows"/> can run (its SPIR-V is present).</summary>
    bool SupportsScatter { get; }

    /// <summary>
    /// Records the quantisation of <paramref name="rows"/> F32 rows of <paramref name="kvStride"/> elements, row <c>srcRow0 + t</c> of
    /// <paramref name="src"/> into cache row <c>positions[srcRow0 + t] - rowBase</c> of <paramref name="dst"/> (issue #1884: selective
    /// recompute and chunk-KV reuse write non-contiguous positions). The bytes equal a contiguous write of the same row.
    /// </summary>
    void RecordScatterRows(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer positions, int kvStride, int rows, int rowBase, int srcRow0 = 0);
}

/// <summary>
/// The F32-to-Q8_0 KV row writer (Flint Q8_0 KV): one <c>convert_f32_to_q8_0_kv_range.spv</c> pipeline, one invocation per 32-element block.
/// The scale and the rounding are fixed in the shader (roundEven), so the stored bytes equal the C# reference quantiser the parity tests use.
/// Issue #1884: the same kernel loads <c>convert_f32_to_q4_0_kv_range.spv</c> for <see cref="VulkanKvStorage.Q4_0"/> (20-byte blocks, the CPU / CUDA
/// KV rule: scale = amax / 7, nibble = clamp(roundEven(x / scale) + 8, 0, 15), adjacent pairs). The method keeps its Q8 name for its callers.
/// </summary>
internal sealed class VulkanKvQ8WriteKernel : IVulkanKvQ8Writer, IDisposable
{
    private const int PushConstantBytes = 2 * sizeof(uint);
    private const int WorkgroupSize = 64;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private VulkanModule? _rowsModule;           // the row-scatter twin (#1884); null when its SPIR-V is missing
    private ComputePipeline? _rowsPipeline;
    private nint _rowsPool;
    private DescriptorSetCache? _rowsCache;
    private bool _disposed;

    private VulkanKvQ8WriteKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, VulkanKvStorage storage)
    {
        Storage = storage;
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _pool = pool;
        _cache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 2);
    }

    /// <inheritdoc/>
    public VulkanKvStorage Storage { get; }

    /// <summary>Dispatches recorded so far (test hook).</summary>
    public long Dispatches { get; private set; }

    /// <summary>Loads <c>convert_f32_to_q8_0_kv_range.spv</c>.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <returns>The kernel.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V is missing.</exception>
    public static VulkanKvQ8WriteKernel Create(VulkanDevice device, string spvDir) => Create(device, spvDir, VulkanKvStorage.Q8_0);

    /// <summary>Loads the writer for <paramref name="storage"/>: Q8_0 or Q4_0 (#1884).</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="storage">The block format to write.</param>
    /// <returns>The kernel.</returns>
    public static VulkanKvQ8WriteKernel Create(VulkanDevice device, string spvDir, VulkanKvStorage storage)
    {
        string shader = storage switch
        {
            VulkanKvStorage.Q8_0 => "convert_f32_to_q8_0_kv_range",
            VulkanKvStorage.Q4_0 => "convert_f32_to_q4_0_kv_range",
            _ => throw new ArgumentOutOfRangeException(nameof(storage)),
        };
        string path = Path.Combine(spvDir, shader + ".spv");
        if (!File.Exists(path))
            throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");
        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[2];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        var kernel = new VulkanKvQ8WriteKernel(device, module, pipeline, KernelSupport.CreateDescriptorPool(device, buffersPerSet: 2), storage);
        kernel.TryLoadRows(spvDir, shader.Replace("_range", "_rows"));
        return kernel;
    }

    /// <inheritdoc/>
    public unsafe void RecordF32ToQ8(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long dstElemOffset, long count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (count <= 0) return;
        if ((dstElemOffset & 31) != 0 || (count & 31) != 0) throw new ArgumentException("Block quantise offsets and counts must be multiples of 32.");
        long blocks = count / 32, firstBlock = dstElemOffset / 32;
        long blockBytes = Storage.RowBytes(32);
        if (firstBlock + blocks > uint.MaxValue / (blockBytes / 4)) throw new ArgumentOutOfRangeException(nameof(count));
        if (src.Size < count * sizeof(float)) throw new ArgumentException("Source buffer too small.", nameof(src));
        if (dst.Size < (firstBlock + blocks) * blockBytes) throw new ArgumentException("Destination buffer too small.", nameof(dst));

        Span<nint> buffers = stackalloc nint[2] { src.Handle, dst.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        uint* pc = stackalloc uint[2];
        pc[0] = (uint)firstBlock;
        pc[1] = (uint)blocks;
        VulkanApi.vkCmdPushConstants(cmd, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pc);
        VulkanApi.vkCmdDispatch(cmd, (uint)((blocks + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
        Dispatches++;
    }

    /// <inheritdoc/>
    public bool SupportsScatter => _rowsPipeline is not null;

    private void TryLoadRows(string spvDir, string shader)
    {
        string path = Path.Combine(spvDir, shader + ".spv");
        if (!File.Exists(path)) return;
        _rowsModule = VulkanModule.LoadFromFile(_device, path);
        Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
        bindings[0] = new VkDescriptorBinding(0);
        bindings[1] = new VkDescriptorBinding(1);
        bindings[2] = new VkDescriptorBinding(2);
        _rowsPipeline = _rowsModule.CreateComputePipeline("main", bindings, 4 * sizeof(uint));
        _rowsPool = KernelSupport.CreateDescriptorPool(_device, buffersPerSet: 3);
        _rowsCache = new DescriptorSetCache(_device, _rowsPool, _rowsPipeline, buffersPerSet: 3);
    }

    /// <inheritdoc/>
    public unsafe void RecordScatterRows(nint cmd, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer positions, int kvStride, int rows, int rowBase, int srcRow0 = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_rowsPipeline is null) throw new NotSupportedException($"The {Storage} KV row-scatter SPIR-V is missing; rebuild the Vulkan shaders.");
        if (rows <= 0) return;
        if ((kvStride & 31) != 0) throw new ArgumentException("Block KV row width must be a multiple of 32.", nameof(kvStride));
        long blocks = (long)rows * (kvStride / 32);
        if (blocks > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(rows));
        if (src.Size < ((long)srcRow0 + rows) * kvStride * sizeof(float)) throw new ArgumentException("Source buffer too small.", nameof(src));
        if (positions.Size < ((long)srcRow0 + rows) * sizeof(int)) throw new ArgumentException("Positions buffer too small.", nameof(positions));

        Span<nint> buffers = stackalloc nint[3] { src.Handle, dst.Handle, positions.Handle };
        nint set = _rowsCache!.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, _rowsPipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, _rowsPipeline.Layout, 0, 1, set, 0, 0);
        uint* pc = stackalloc uint[4];
        pc[0] = (uint)(kvStride / 32);
        pc[1] = (uint)rows;
        pc[2] = (uint)rowBase;
        pc[3] = (uint)srcRow0;
        VulkanApi.vkCmdPushConstants(cmd, _rowsPipeline.Layout, VkShaderStageFlags.Compute, 0, 4 * sizeof(uint), (nint)pc);
        VulkanApi.vkCmdDispatch(cmd, (uint)((blocks + WorkgroupSize - 1) / WorkgroupSize), 1, 1);
        Dispatches++;
    }

    /// <summary>Drops the cached descriptor sets (a KV buffer was freed; its handle may be recycled).</summary>
    public void InvalidateDescriptorCache()
    {
        _cache.Reset();
        _rowsCache?.Reset();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        if (_rowsPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _rowsPool, 0);
        _rowsPipeline?.Dispose();
        _rowsModule?.Dispose();
        _pipeline.Dispose();
        _module.Dispose();
    }
}
