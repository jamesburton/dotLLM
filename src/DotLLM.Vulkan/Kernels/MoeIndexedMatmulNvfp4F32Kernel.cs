using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed NVFP4 expert bank (36-byte blocks of 64, per-expert output scale, #1579):
/// <c>y[n, m] = dequant(bank[indices[n], m, :]) dot x[n, :]</c>.
/// </summary>
public sealed class MoeIndexedMatmulNvfp4F32Kernel : IDisposable
{
    /// <summary>NVFP4 block: 4 UE4M3 sub-block scales + 32 bytes of packed nibbles (36 bytes, 64 elements).</summary>
    public const int Nvfp4BlockBytes = QuantFormat.Nvfp4BlockBytes;
    /// <summary>Elements per NVFP4 block.</summary>
    public const int Nvfp4GroupSize = QuantFormat.Nvfp4GroupSize;

    private const int WorkgroupX = 16;
    private const int WorkgroupY = 16;
    // M, K, N, numExperts, blocksPerRow
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MoeIndexedMatmulNvfp4F32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 5);
    }

    /// <summary>Loads <c>moe_indexed_matmul_nvfp4_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    public static MoeIndexedMatmulNvfp4F32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_nvfp4_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[5];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
            bindings[4] = new VkDescriptorBinding(4);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 5);
        return new MoeIndexedMatmulNvfp4F32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    public void Launch(
        VulkanDevice.Buffer bankNvfp4, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y, VulkanDevice.Buffer expertScale,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankNvfp4, x, indices, y, expertScale, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed NVFP4 expert-bank matmul dispatch into <paramref name="cmdBuf"/>.</summary>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bankNvfp4, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y, VulkanDevice.Buffer expertScale,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Nvfp4GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Nvfp4GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Nvfp4GroupSize;
        long rowBytes = (long)blocksPerRow * Nvfp4BlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankNvfp4.Size < bankBytes) throw new ArgumentException("bankNvfp4 buffer too small.", nameof(bankNvfp4));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));
        if (expertScale.Size < (long)numExperts * sizeof(float)) throw new ArgumentException("expertScale buffer too small.", nameof(expertScale));

        Span<nint> buffers = stackalloc nint[5]
        {
            bankNvfp4.Handle, x.Handle, indices.Handle, y.Handle, expertScale.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[5]
        {
            (uint)m,
            (uint)k,
            (uint)n,
            (uint)numExperts,
            (uint)blocksPerRow,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        uint groupX = (uint)((m + WorkgroupX - 1) / WorkgroupX);
        uint groupY = (uint)((n + WorkgroupY - 1) / WorkgroupY);
        VulkanApi.vkCmdDispatch(cmdBuf, groupX, groupY, 1);
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
