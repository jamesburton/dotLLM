using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Identifies a grouped-MoE F16 coopmat shader variant — see
/// <see cref="Q8_0GemmCoopmatVariant"/> (issue #240) for the rationale; this is the same
/// wave-width-pin fix applied to <see cref="MoeGroupedMatmulF16CoopmatKernel"/>.
/// </summary>
/// <param name="SpvFileName">File name of the SPIR-V module within the <c>spv</c> directory.</param>
/// <param name="RequiredSubgroupSize">Non-zero pins the pipeline to this wave width.</param>
public readonly record struct MoeGroupedCoopmatVariant(string SpvFileName, int RequiredSubgroupSize)
{
    /// <summary>Baseline 64-thread coopmat kernel.</summary>
    public static MoeGroupedCoopmatVariant Coopmat64 => new("moe_grouped_matmul_f16_coopmat.spv", 0);

    /// <summary>32-thread workgroup pinned to wave32.</summary>
    public static MoeGroupedCoopmatVariant Coopmat32 => new("moe_grouped_matmul_f16_coopmat32.spv", 32);

    /// <summary>
    /// Whether <paramref name="device"/> can create a pipeline for this variant right now: the
    /// pinnable subgroup size (when declared) AND the compiled SPIR-V present in
    /// <paramref name="spvDir"/> — see <see cref="Q8_0GemmCoopmatVariant.IsSupportedOn"/>'s
    /// remarks for why the file-existence check matters.
    /// </summary>
    public bool IsSupportedOn(VulkanDevice device, string spvDir)
    {
        if (RequiredSubgroupSize != 0
            && !device.SupportsRequiredSubgroupSize((uint)RequiredSubgroupSize, VkShaderStageFlags.Compute))
            return false;
        return File.Exists(Path.Combine(spvDir, SpvFileName));
    }

    /// <summary>Picks the fastest variant this device can actually run right now.</summary>
    public static MoeGroupedCoopmatVariant SelectFor(VulkanDevice device, string spvDir)
        => Coopmat32.IsSupportedOn(device, spvDir) ? Coopmat32 : Coopmat64;
}

/// <summary>
/// Grouped MoE dense expert projection over packed F16 expert banks using cooperative matrices.
/// </summary>
public sealed class MoeGroupedMatmulF16CoopmatKernel : IDisposable
{
    /// <summary>Bytes per F16 element on device.</summary>
    public const int F16ElementBytes = 2;
    /// <summary>K must be a multiple of this value.</summary>
    public const int KChunk = 32;

    private const int TileM = 16;
    private const int TileN = 16;
    private const int PushConstantBytes = 6 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeGroupedMatmulF16CoopmatKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>
    /// Loads the fastest coopmat variant <paramref name="device"/> can run right now
    /// (<see cref="MoeGroupedCoopmatVariant.SelectFor"/>) from <paramref name="spvDir"/>.
    /// </summary>
    public static MoeGroupedMatmulF16CoopmatKernel Create(VulkanDevice device, string spvDir)
        => Create(device, spvDir, MoeGroupedCoopmatVariant.SelectFor(device, spvDir));

    /// <summary>
    /// Loads the SPIR-V for <paramref name="variant"/> and creates the pipeline. The explicit
    /// overload exists so a benchmark can A/B <see cref="MoeGroupedCoopmatVariant.Coopmat64"/> vs
    /// <see cref="MoeGroupedCoopmatVariant.Coopmat32"/> side by side in one process.
    /// </summary>
    /// <exception cref="FileNotFoundException">The variant's SPIR-V is missing from <paramref name="spvDir"/>.</exception>
    public static MoeGroupedMatmulF16CoopmatKernel Create(VulkanDevice device, string spvDir, MoeGroupedCoopmatVariant variant)
    {
        if (!device.HasCooperativeMatrix)
            throw new InvalidOperationException(
                "MoeGroupedMatmulF16CoopmatKernel requires VK_KHR_cooperative_matrix support.");

        string path = Path.Combine(spvDir, variant.SpvFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline(
                "main", bindings, PushConstantBytes,
                requiredSubgroupSize: (uint)variant.RequiredSubgroupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeGroupedMatmulF16CoopmatKernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer expertBankF16, VulkanDevice.Buffer packedInput,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, expertBankF16, packedInput, offsets, output,
            m, k, rows, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records grouped F16 coopmat matmul into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer expertBankF16, VulkanDevice.Buffer packedInput,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output,
        int m, int k, int rows, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % KChunk) != 0)
            throw new ArgumentException($"k must be a multiple of {KChunk}, got {k}", nameof(k));

        long weightBytes = (long)numExperts * m * k * F16ElementBytes;
        long inputBytes = (long)rows * k * sizeof(float);
        long offsetsBytes = (long)(numExperts + 1) * sizeof(uint);
        long outputBytes = (long)rows * m * sizeof(float);
        if (expertBankF16.Size < weightBytes) throw new ArgumentException("expertBankF16 buffer too small.", nameof(expertBankF16));
        if (packedInput.Size < inputBytes) throw new ArgumentException("packedInput buffer too small.", nameof(packedInput));
        if (offsets.Size < offsetsBytes) throw new ArgumentException("offsets buffer too small.", nameof(offsets));
        if (output.Size < outputBytes) throw new ArgumentException("output buffer too small.", nameof(output));

        Span<nint> buffers = stackalloc nint[4]
        {
            expertBankF16.Handle, packedInput.Handle, offsets.Handle, output.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        int pairsPerRow = k / 2;
        Span<uint> pc = stackalloc uint[6]
        {
            (uint)m,
            (uint)k,
            (uint)rows,
            (uint)numExperts,
            (uint)pairsPerRow,
            (uint)pairsPerRow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupsX = (uint)((m + TileM - 1) / TileM);
        uint groupsY = (uint)((rows + TileN - 1) / TileN);
        VulkanApi.vkCmdDispatch(cmdBuf, groupsX, groupsY, (uint)numExperts);
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
