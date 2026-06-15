using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Q8_0 decode-path GEMV using the hardware integer dot product (DP4a):
/// <c>y[M] = W_q8[M,K] @ x[K]</c>.
/// </summary>
/// <remarks>
/// <para>
/// Numerically equivalent to <see cref="MatMulQ8_0Kernel"/> (same weight layout,
/// buffers, push constants and dispatch), but instead of dequantizing each int8
/// weight to FP32, it quantizes the FP32 activation slice to INT8 per 32-element
/// block and uses <c>dotPacked4x8AccSatEXT</c> (4×8-bit packed signed dot,
/// hardware-accelerated via DP4a on Xe-LPG / sm_61+) to accumulate four products
/// per instruction. The activation is rounded to INT8 first (≈0.8% per element),
/// so this trades a small accuracy cost for INT8 throughput.
/// </para>
/// <para>
/// Requires <see cref="VulkanDevice.HasIntegerDotProduct"/>. Callers that cannot
/// guarantee the feature should fall back to <see cref="MatMulQ8_0Kernel"/>.
/// </para>
/// </remarks>
public sealed class MatMulQ8_0Dp4aKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    private const int PushConstantBytes = 4 * sizeof(uint); // M, K, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private MatMulQ8_0Dp4aKernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 3);
    }

    /// <summary>
    /// Loads the DP4a Q8_0 GEMV SPIR-V (<c>matmul_q8_0_dp4a.spv</c>) and creates the pipeline.
    /// </summary>
    /// <param name="device">
    /// Vulkan device; must report <see cref="VulkanDevice.HasIntegerDotProduct"/> = <c>true</c>
    /// (the shader uses <c>GL_EXT_integer_dot_product</c>, which needs the extension enabled at
    /// device creation).
    /// </param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V blobs.</param>
    /// <returns>An initialized kernel.</returns>
    /// <exception cref="NotSupportedException">If the device lacks integer dot product support.</exception>
    /// <exception cref="FileNotFoundException">If the SPIR-V blob is missing.</exception>
    public static MatMulQ8_0Dp4aKernel Create(VulkanDevice device, string spvDir)
    {
        if (!device.HasIntegerDotProduct)
            throw new NotSupportedException(
                "Device does not support VK_KHR_shader_integer_dot_product; use MatMulQ8_0Kernel instead.");

        string path = Path.Combine(spvDir, "matmul_q8_0_dp4a.spv");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.ps1 after installing the Vulkan SDK.");

        var module = VulkanModule.LoadFromFile(device, path);
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
        return new MatMulQ8_0Dp4aKernel(device, module, pipeline, pool);
    }

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the GEMV: <c>y[M] = W[M,K] @ x[K]</c>. Synchronous — returns after
    /// <c>vkQueueWaitIdle</c>.
    /// </summary>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public void Launch(
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weightsQ8, x, y, m, k);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the GEMV into <paramref name="cmdBuf"/> without submitting.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer to append commands to.</param>
    /// <param name="weightsQ8">Raw Q8_0 blob of <c>M * (K/32) * 34</c> bytes, rows contiguous.</param>
    /// <param name="x">FP32 activation buffer of length <paramref name="k"/>.</param>
    /// <param name="y">FP32 output buffer of length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Input dimension (must be a multiple of 32).</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
        int m, int k)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long weightsMin = (long)m * rowBytes;
        if (weightsQ8.Size < weightsMin)
            throw new ArgumentException(
                $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.",
                nameof(weightsQ8));
        if (x.Size < (long)k * sizeof(float))
            throw new ArgumentException("Input buffer too small.", nameof(x));
        if (y.Size < (long)m * sizeof(float))
            throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[3] { weightsQ8.Handle, x.Handle, y.Handle };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        Span<uint> pc = stackalloc uint[4]
        {
            (uint)m,
            (uint)k,
            (uint)blocksPerRow,
            (uint)rowUints,
        };
        fixed (uint* pcPtr = pc)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        VulkanApi.vkCmdDispatch(cmdBuf, (uint)m, 1, 1);
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
