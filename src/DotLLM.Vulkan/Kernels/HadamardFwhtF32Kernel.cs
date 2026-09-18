using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// PrismML blockwise normalized Walsh-Hadamard activation transform
/// (<c>prism.hadamard.*</c>) — the GPU twin of <c>DotLLM.Cpu.Kernels.Hadamard</c>.
/// </summary>
/// <remarks>
/// <para>
/// Bonsai 2 stores its ternary weights in a rotated basis, so every folded weight's input
/// activation must be rotated to match before the matmul. Running it on the device rather than
/// round-tripping to the host is the whole point: a host transform would add a
/// download/upload per folded projection per token, which on a 64-layer model is hundreds of
/// stalls per decode step.
/// </para>
/// <para>
/// The transform is O(n log n) over the activation and touches no weight bytes, so it is cheap
/// relative to the ternary GEMMs it feeds.
/// </para>
/// </remarks>
public sealed class HadamardFwhtF32Kernel : IDisposable
{
    private const int WorkgroupSize = 256;

    /// <summary>Largest block the shader's shared-memory staging can hold (4 KiB of floats).</summary>
    public const int MaxBlockSize = 1024;

    private const int PushConstantBytes = 10 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private HadamardFwhtF32Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>Loads <c>hadamard_fwht_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <returns>The created kernel.</returns>
    public static HadamardFwhtF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "hadamard_fwht_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        return new HadamardFwhtF32Kernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    /// <param name="src">Source activation buffer, <c>[rows, width]</c> F32.</param>
    /// <param name="dst">Destination buffer, same shape. May be the same buffer as <paramref name="src"/>.</param>
    /// <param name="signs">±1 vector of <paramref name="width"/> floats; may be any bound buffer when <paramref name="applySigns"/> is false.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="width">Row width; must be a positive multiple of <paramref name="blockSize"/>.</param>
    /// <param name="blockSize">Hadamard block width; a power of two, at most <see cref="MaxBlockSize"/>.</param>
    /// <param name="applySigns">Whether the ±1 sign step runs.</param>
    /// <param name="inverse">False for the forward order (signs then rotation), true for the inverse (rotation then signs).</param>
    /// <param name="permute">GDN tiled→grouped value-head remap, for <c>ssm_out</c> only.</param>
    public void Launch(
        VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer signs,
        int rows, int width, int blockSize,
        bool applySigns, bool inverse, GdnPermute? permute = null)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, src, dst, signs, rows, width, blockSize, applySigns, inverse, permute);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// GDN value-head permutation geometry for <c>*.ssm_out.weight</c>
    /// (<c>prism.hadamard.gdn_v_grouped</c>).
    /// </summary>
    /// <param name="DState">Per-head width, contiguous and minor (128 for Bonsai 2).</param>
    /// <param name="NKHead">Key heads (16).</param>
    /// <param name="Rep">Value heads per key head, <c>NVHead / NKHead</c> (3).</param>
    public readonly record struct GdnPermute(int DState, int NKHead, int Rep);

    /// <summary>
    /// Records the transform. See <see cref="Launch"/> for parameter semantics.
    /// </summary>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="src">Source activation buffer.</param>
    /// <param name="dst">Destination buffer.</param>
    /// <param name="signs">±1 vector buffer.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="width">Row width.</param>
    /// <param name="blockSize">Hadamard block width.</param>
    /// <param name="applySigns">Whether the sign step runs.</param>
    /// <param name="inverse">Forward (false) or inverse (true) step order.</param>
    /// <param name="permute">Optional GDN value-head remap.</param>
    public unsafe void Record(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, VulkanDevice.Buffer signs,
        int rows, int width, int blockSize,
        bool applySigns, bool inverse, GdnPermute? permute = null)
    {
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
            throw new ArgumentException($"Hadamard block size must be a power of two, got {blockSize}.", nameof(blockSize));
        if (blockSize > MaxBlockSize)
            throw new ArgumentException(
                $"Hadamard block size {blockSize} exceeds the shader's shared-memory staging limit of {MaxBlockSize}.",
                nameof(blockSize));
        if (width <= 0 || width % blockSize != 0)
            throw new ArgumentException(
                $"Row width {width} is not a positive multiple of block size {blockSize}.", nameof(width));

        long bytes = (long)rows * width * sizeof(float);
        if (src.Size < bytes) throw new ArgumentException("src buffer too small.", nameof(src));
        if (dst.Size < bytes) throw new ArgumentException("dst buffer too small.", nameof(dst));
        if (applySigns && signs.Size < (long)width * sizeof(float))
            throw new ArgumentException("signs buffer too small.", nameof(signs));

        int blocksPerRow = width / blockSize;
        var perm = permute ?? default;
        if (permute is { } p)
        {
            if (p.DState <= 0 || p.NKHead <= 0 || p.Rep <= 0 || p.DState * p.NKHead * p.Rep != width)
                throw new ArgumentException(
                    $"GDN permute geometry {p.DState}x{p.NKHead}x{p.Rep} does not cover width {width}.",
                    nameof(permute));
        }

        Span<nint> buffers = stackalloc nint[3] { src.Handle, dst.Handle, signs.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        var w = System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian;
        w(pcBytes, (uint)rows);
        w(pcBytes[4..], (uint)width);
        w(pcBytes[8..], (uint)blockSize);
        w(pcBytes[12..], (uint)blocksPerRow);
        w(pcBytes[16..], applySigns ? 1u : 0u);
        w(pcBytes[20..], inverse ? 1u : 0u);
        w(pcBytes[24..], permute is not null ? 1u : 0u);
        w(pcBytes[28..], (uint)perm.DState);
        w(pcBytes[32..], (uint)perm.NKHead);
        w(pcBytes[36..], (uint)perm.Rep);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // One workgroup per (row, block): x = block within the row, y = row.
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)blocksPerRow, (uint)rows, 1);
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
