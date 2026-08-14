using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed Q5_0 expert bank (issue #407):
/// <c>y[n, m] = dequant(bank[indices[n], m, :]) dot x[n, :]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Direct sibling of <see cref="MoeIndexedMatmulQ8_0F32Kernel"/> — same four
/// bindings, same five push constants, same <c>[numExperts, M, K]</c> bank
/// addressing. Only the per-block decode differs: Q5_0 is symmetric
/// (<c>d·(q−16)</c>) with the 5th bit of every weight held in a separate
/// 32-bit <c>qh</c> field.
/// </para>
/// <para>
/// Without this kernel (and the matching <c>MoeRoutedRawDeviceQuantType</c>
/// wiring) a Q5_0 routed expert bank falls back to a host F32 dequant — a
/// 5.8x expansion that turns Nemotron-3.5-Lightning's 18.9 GB into ~100 GB of
/// host allocation (#344, #407).
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulQ5_0F32Kernel : IDisposable
{
    /// <summary>Q5_0 block: fp16 d + 4-byte qh + 16-byte qs = 22 bytes.</summary>
    public const int Q5_0BlockBytes = QuantFormat.Q5_0BlockBytes;

    /// <summary>Elements per Q5_0 block.</summary>
    public const int Q5_0GroupSize = QuantFormat.LegacyGroupSize;

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

    private MoeIndexedMatmulQ5_0F32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_indexed_matmul_q5_0_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    /// <param name="device">Device the pipeline is created on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <returns>An owned kernel instance; dispose it with the device.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V blob is missing.</exception>
    public static MoeIndexedMatmulQ5_0F32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_q5_0_f32.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        VulkanModule module = VulkanModule.LoadFromFile(device, path);
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[4];
            bindings[0] = new VkDescriptorBinding(0);
            bindings[1] = new VkDescriptorBinding(1);
            bindings[2] = new VkDescriptorBinding(2);
            bindings[3] = new VkDescriptorBinding(3);
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

        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 4);
        return new MoeIndexedMatmulQ5_0F32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    /// <param name="bankQ5_0">Packed Q5_0 expert bank, <c>[numExperts, m, k]</c>.</param>
    /// <param name="x">Activations, <c>[n, k]</c> F32.</param>
    /// <param name="indices">Per-row expert index, <c>[n]</c> int32.</param>
    /// <param name="y">Output, <c>[n, m]</c> F32.</param>
    /// <param name="m">Per-expert weight row count.</param>
    /// <param name="k">Contraction dim; must be a multiple of 32.</param>
    /// <param name="n">Total expanded rows.</param>
    /// <param name="numExperts">Bank's first-axis size.</param>
    public void Launch(
        VulkanDevice.Buffer bankQ5_0, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankQ5_0, x, indices, y, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed Q5_0 expert-bank matmul dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="bankQ5_0">Packed Q5_0 expert bank, <c>[numExperts, m, k]</c>.</param>
    /// <param name="x">Activations, <c>[n, k]</c> F32.</param>
    /// <param name="indices">Per-row expert index, <c>[n]</c> int32.</param>
    /// <param name="y">Output, <c>[n, m]</c> F32.</param>
    /// <param name="m">Per-expert weight row count.</param>
    /// <param name="k">Contraction dim; must be a multiple of 32.</param>
    /// <param name="n">Total expanded rows.</param>
    /// <param name="numExperts">Bank's first-axis size.</param>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is non-positive.</exception>
    /// <exception cref="ArgumentException"><paramref name="k"/> is misaligned or a buffer is too small.</exception>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer bankQ5_0, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Q5_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q5_0BlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankQ5_0.Size < bankBytes) throw new ArgumentException("bankQ5_0 buffer too small.", nameof(bankQ5_0));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4]
        {
            bankQ5_0.Handle, x.Handle, indices.Handle, y.Handle,
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
