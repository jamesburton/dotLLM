using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Decode-path fusion of RMSNorm + Q8_0 GEMV into a single dispatch.
/// Replaces the (rmsnorm_f32 → matmul_q8_0) pair at the attention-norm + Q
/// projection and the FFN-norm + Gate projection sites — eliminating one
/// dispatch and one pipeline barrier per layer per fusion.
/// </summary>
/// <remarks>
/// <para>
/// Each workgroup recomputes the rmsnorm reduction redundantly; the cost
/// (~K muls + log2(WG_SIZE) barriers) is small compared to the dispatch
/// overhead it eliminates. The matmul phase reads the normalized values
/// from on-chip shared memory instead of round-tripping through VRAM.
/// </para>
/// <para>
/// The shader also writes <c>normOutput</c> from the first
/// <c>ceil(K / WG_SIZE)</c> workgroups so downstream non-fused matmuls
/// (K, V, Up) can consume the normalized hidden state as before.
/// </para>
/// <para>
/// Constraints: decode path only (single-token; the shader's shared
/// <c>sharedNorm</c> scratch holds the entire hidden row). The shader's
/// fixed <c>K_MAX</c> caps the supported hidden size at 4096 — call
/// <see cref="SupportsHiddenSize"/> at the routing site.
/// </para>
/// </remarks>
public sealed class RmsNormMatmulQ8_0FusedKernel : IDisposable
{
    /// <summary>Q8_0 block: 2 bytes fp16 scale + 32 signed int8 values.</summary>
    public const int Q8_0BlockBytes = 34;

    /// <summary>Elements per Q8_0 block.</summary>
    public const int Q8_0GroupSize = 32;

    /// <summary>
    /// Maximum hidden size supported by the shader's on-chip scratch buffer.
    /// Mirrors <c>K_MAX</c> in <c>rmsnorm_matmul_q8_0.comp</c>. Sized to keep
    /// shared-memory pressure low (4 KB) so the GPU can keep many workgroups
    /// in flight per CU. Larger models fall back to the standalone pair.
    /// </summary>
    public const int MaxHiddenSize = 1024;

    private const int WorkgroupSize = 128;
    /// <summary>Matmul rows produced per workgroup. Must mirror <c>ROWS_PER_WG</c> in the shader.</summary>
    private const int RowsPerWorkgroup = 8;
    // M, K, blocksPerRow, rowUints, eps  →  4×u32 + 1×f32
    private const int PushConstantBytes = 5 * sizeof(uint);

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    private RmsNormMatmulQ8_0FusedKernel(
        VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline.DescriptorSetLayout, buffersPerSet: 5);
    }

    /// <summary>
    /// Loads <c>rmsnorm_matmul_q8_0.spv</c> from <paramref name="spvDir"/> and
    /// builds the compute pipeline. Returns <c>null</c> when the SPV is
    /// missing — older builds without the fused shader stay working via the
    /// non-fused fallback in <c>VulkanTransformerModel</c>.
    /// </summary>
    public static RmsNormMatmulQ8_0FusedKernel? TryCreate(VulkanDevice device, string spvDir)
    {
        string path = Path.Combine(spvDir, "rmsnorm_matmul_q8_0.spv");
        if (!File.Exists(path))
            return null;

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
        return new RmsNormMatmulQ8_0FusedKernel(device, module, pipeline, pool);
    }

    /// <summary>
    /// Returns true when this kernel can handle a fusion with the given
    /// hidden size. Above the on-chip cap the caller must fall back to the
    /// separate rmsnorm + matmul dispatches.
    /// </summary>
    public static bool SupportsHiddenSize(int k) => k > 0 && k <= MaxHiddenSize;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Dispatches the fused rmsnorm + Q8_0 GEMV. Synchronous wrapper around
    /// <see cref="Record"/> — used by unit tests; the forward pass uses
    /// <see cref="Record"/> directly into a persistent command buffer.
    /// </summary>
    public void Launch(
        VulkanDevice.Buffer hidden, VulkanDevice.Buffer normWeight,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer normOutput,
        VulkanDevice.Buffer y,
        int m, int k, float eps)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, hidden, normWeight, weightsQ8, normOutput, y, m, k, eps);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// Records the fused rmsnorm + Q8_0 GEMV into <paramref name="cmdBuf"/>.
    /// </summary>
    /// <param name="cmdBuf">Open Vulkan command buffer.</param>
    /// <param name="hidden">FP32 hidden state, length <paramref name="k"/>.</param>
    /// <param name="normWeight">FP32 RMSNorm per-feature scale, length <paramref name="k"/>.</param>
    /// <param name="weightsQ8">Raw Q8_0 weight blob, <c>m * (k/32) * 34</c> bytes.</param>
    /// <param name="normOutput">FP32 buffer that receives the normalized hidden state, length <paramref name="k"/>.</param>
    /// <param name="y">FP32 matmul output, length <paramref name="m"/>.</param>
    /// <param name="m">Output dimension.</param>
    /// <param name="k">Hidden / input dimension. Must be a multiple of 32 and ≤ <see cref="MaxHiddenSize"/>.</param>
    /// <param name="eps">Epsilon under the rmsnorm sqrt.</param>
    public unsafe void Record(
        nint cmdBuf,
        VulkanDevice.Buffer hidden, VulkanDevice.Buffer normWeight,
        VulkanDevice.Buffer weightsQ8, VulkanDevice.Buffer normOutput,
        VulkanDevice.Buffer y,
        int m, int k, float eps)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        if ((k % Q8_0GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q8_0GroupSize}, got {k}", nameof(k));
        if (k > MaxHiddenSize)
            throw new ArgumentException(
                $"k must be <= {MaxHiddenSize} (shader on-chip scratch cap), got {k}", nameof(k));

        int blocksPerRow = k / Q8_0GroupSize;
        long rowBytes = (long)blocksPerRow * Q8_0BlockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);

        long hiddenMin = (long)k * sizeof(float);
        long weightsMin = (long)m * rowBytes;
        long yMin = (long)m * sizeof(float);
        if (hidden.Size < hiddenMin) throw new ArgumentException("Hidden buffer too small.", nameof(hidden));
        if (normWeight.Size < hiddenMin) throw new ArgumentException("NormWeight buffer too small.", nameof(normWeight));
        if (weightsQ8.Size < weightsMin) throw new ArgumentException(
            $"Weights buffer too small: need >= {weightsMin} bytes, got {weightsQ8.Size}.", nameof(weightsQ8));
        if (normOutput.Size < hiddenMin) throw new ArgumentException("NormOutput buffer too small.", nameof(normOutput));
        if (y.Size < yMin) throw new ArgumentException("Output buffer too small.", nameof(y));

        Span<nint> buffers = stackalloc nint[5]
        {
            hidden.Handle, normWeight.Handle, weightsQ8.Handle, normOutput.Handle, y.Handle,
        };
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);

        // Push constants: M, K, blocksPerRow, rowUints (all u32) + eps (f32) — 20 bytes total.
        Span<byte> pcBytes = stackalloc byte[PushConstantBytes];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes, (uint)m);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[4..], (uint)k);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[8..], (uint)blocksPerRow);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(pcBytes[12..], (uint)rowUints);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(pcBytes[16..], eps);
        fixed (byte* pcPtr = pcBytes)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, PushConstantBytes, (nint)pcPtr);
        }

        // Sub-tile dispatch: each workgroup outputs RowsPerWorkgroup matmul
        // rows. Tail tile (if M is not divisible) is bounds-checked in the
        // shader.
        uint groupCount = (uint)((m + RowsPerWorkgroup - 1) / RowsPerWorkgroup);
        VulkanApi.vkCmdDispatch(cmdBuf, groupCount, 1, 1);
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
