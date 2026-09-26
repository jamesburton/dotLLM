using DotLLM.Vulkan.Interop;
using DotLLM.Core.Configuration;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// MoE indexed expert matmul over a packed IQ4_NL expert bank (issue #407):
/// <c>y[n, m] = dequant(bank[indices[n], m, :]) dot x[n, :]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Direct sibling of <see cref="MoeIndexedMatmulQ8_0F32Kernel"/> — same four
/// bindings, same five push constants, same <c>[numExperts, M, K]</c> bank
/// addressing. Only the per-block decode differs: an IQ4_NL nibble is an index
/// into ggml's 16-entry non-linear signed codebook (<c>kvalues_iq4nl</c>), not a
/// signed integer, so the element value is <c>d · kvalues_iq4nl[q]</c>.
/// </para>
/// <para>
/// IQ4_NL's dense Vulkan path has been complete for some time; it is the
/// <i>routed</i> path that was missing, and a fully-covered dense type with no
/// <c>moe_indexed</c> variant still expanded to F32 for every expert bank —
/// 91% of Nemotron-3.5-Lightning's tensors by count (#344, #407).
/// </para>
/// </remarks>
public sealed class MoeIndexedMatmulIq4NlF32Kernel : IDisposable
{
    /// <summary>IQ4_NL block: fp16 d + 16-byte qs = 18 bytes.</summary>
    public const int Iq4NlBlockBytes = QuantFormat.IQ4_NLBlockBytes;

    /// <summary>Elements per IQ4_NL block.</summary>
    public const int Iq4NlGroupSize = QuantFormat.LegacyGroupSize;

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

    private MoeIndexedMatmulIq4NlF32Kernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: 4);
    }

    /// <summary>Loads <c>moe_indexed_matmul_iq4_nl_f32.spv</c> from <paramref name="spvDir"/>.</summary>
    /// <param name="device">Device the pipeline is created on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <returns>An owned kernel instance; dispose it with the device.</returns>
    /// <exception cref="FileNotFoundException">The SPIR-V blob is missing.</exception>
    public static MoeIndexedMatmulIq4NlF32Kernel Create(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "moe_indexed_matmul_iq4_nl_f32.spv");
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
        return new MoeIndexedMatmulIq4NlF32Kernel(device, module, pipeline, pool);
    }

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous launch — wraps <see cref="Record"/>; used by unit tests.</summary>
    /// <param name="bankIq4Nl">Packed IQ4_NL expert bank, <c>[numExperts, m, k]</c>.</param>
    /// <param name="x">Activations, <c>[n, k]</c> F32.</param>
    /// <param name="indices">Per-row expert index, <c>[n]</c> int32.</param>
    /// <param name="y">Output, <c>[n, m]</c> F32.</param>
    /// <param name="m">Per-expert weight row count.</param>
    /// <param name="k">Contraction dim; must be a multiple of 32.</param>
    /// <param name="n">Total expanded rows.</param>
    /// <param name="numExperts">Bank's first-axis size.</param>
    public void Launch(
        VulkanDevice.Buffer bankIq4Nl, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, bankIq4Nl, x, indices, y, m, k, n, numExperts);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the indexed IQ4_NL expert-bank matmul dispatch into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="bankIq4Nl">Packed IQ4_NL expert bank, <c>[numExperts, m, k]</c>.</param>
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
        VulkanDevice.Buffer bankIq4Nl, VulkanDevice.Buffer x, VulkanDevice.Buffer indices, VulkanDevice.Buffer y,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if ((k % Iq4NlGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4NlGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4NlGroupSize;
        long rowBytes = (long)blocksPerRow * Iq4NlBlockBytes;
        long bankBytes = (long)numExperts * m * rowBytes;
        long xBytes = (long)n * k * sizeof(float);
        long idxBytes = (long)n * sizeof(int);
        long yBytes = (long)n * m * sizeof(float);
        if (bankIq4Nl.Size < bankBytes) throw new ArgumentException("bankIq4Nl buffer too small.", nameof(bankIq4Nl));
        if (x.Size < xBytes) throw new ArgumentException("x buffer too small.", nameof(x));
        if (indices.Size < idxBytes) throw new ArgumentException("indices buffer too small.", nameof(indices));
        if (y.Size < yBytes) throw new ArgumentException("y buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[4]
        {
            bankIq4Nl.Handle, x.Handle, indices.Handle, y.Handle,
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
