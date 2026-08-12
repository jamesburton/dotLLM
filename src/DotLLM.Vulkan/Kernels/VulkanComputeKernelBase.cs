using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Shared plumbing for single-descriptor-set compute kernel wrappers: loads a
/// compiled <c>.spv</c> module, builds the compute pipeline (one storage-buffer
/// binding per index in <c>[0, buffersPerSet)</c>, entry point <c>"main"</c>),
/// the descriptor pool, and the <see cref="DescriptorSetCache"/>, and disposes
/// them in the correct order (pool, then pipeline, then module).
/// </summary>
/// <remarks>
/// <para>
/// This region — <c>_device</c>, <c>_module</c>, <c>_pipeline</c>,
/// <c>_descriptorPool</c>, <c>_descriptorCache</c>, the load/create flow, and
/// <see cref="Dispose"/> — was byte-identical across every dequant kernel
/// wrapper (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q5_0). The <see cref="DescriptorSetCache"/>
/// handle-aliasing hazard lives in exactly that duplicated region, so before
/// this extraction a fix to it needed applying in six places.
/// </para>
/// <para>
/// Derived classes keep their own <c>Create</c> factory, buffer-size guards,
/// push-constant <c>stackalloc</c>, and <c>vkCmdDispatch</c> geometry — none of
/// that is shared here. <see cref="BindAndPush"/> only binds the pipeline,
/// binds the cached descriptor set, and pushes constants; it never dispatches,
/// and it never appears behind a virtual call on the per-dispatch path (derived
/// <c>Record</c> methods stay concrete — <c>buffersPerSet</c> and
/// <c>pushConstantBytes</c> are resolved once, at construction).
/// </para>
/// <para>
/// <b>Important:</b> <see cref="DescriptorSetCache.GetOrCreate"/> (called from
/// <see cref="BindOnly"/>, and hence from <see cref="BindAndPush"/>) declares
/// the dispatch's buffer-access set to the device's active hazard tracker
/// every time it runs — not only on a cache miss. For a single-dispatch
/// <c>Record</c>, call <see cref="BindAndPush"/> once. For a chunked dispatch
/// (multiple <c>vkCmdDispatch</c> calls against the *same* buffers, differing
/// only in push constants — e.g. the K-quant dequant kernels'
/// <c>firstBlock</c>-chunked loop), call <see cref="BindOnly"/> exactly once
/// before the loop and <see cref="PushConstantsOnly"/> per chunk inside it:
/// calling <see cref="BindAndPush"/> (or <see cref="BindOnly"/>) once per
/// chunk would re-arm hazard tracking on each iteration and — if a tracked
/// forward is ever active — spuriously treat each chunk's write as a hazard
/// against the previous chunk's write to the same destination buffer,
/// emitting a redundant <c>vkCmdPipelineBarrier</c> per chunk even though the
/// chunks write disjoint output and no inter-chunk barrier is needed.
/// </para>
/// </remarks>
public abstract class VulkanComputeKernelBase : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private bool _disposed;

    /// <summary>The device this kernel was created against.</summary>
    protected VulkanDevice Device => _device;

    /// <summary>
    /// Loads <paramref name="shaderFileName"/> from <paramref name="spvDir"/> and
    /// builds the compute pipeline, descriptor pool, and descriptor-set cache.
    /// </summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V modules.</param>
    /// <param name="shaderFileName">File name of the compiled <c>.spv</c> module (e.g. <c>"q5_0_dequant_f32.spv"</c>).</param>
    /// <param name="buffersPerSet">Number of storage-buffer bindings (and descriptor-set-cache key width) this kernel uses.</param>
    /// <param name="pushConstantBytes">Size of the push-constant block in bytes.</param>
    /// <exception cref="FileNotFoundException">The <c>.spv</c> file does not exist under <paramref name="spvDir"/>.</exception>
    protected VulkanComputeKernelBase(
        VulkanDevice device, string spvDir, string shaderFileName, int buffersPerSet, int pushConstantBytes)
    {
        _device = device;

        string path = Path.Combine(spvDir, shaderFileName);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Vulkan SPIR-V not found: {path}. Run native/vulkan/build.sh (or build.ps1) after installing the Vulkan SDK.");

        _module = VulkanModule.LoadFromFile(device, path);
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[buffersPerSet];
            for (int i = 0; i < buffersPerSet; i++)
                bindings[i] = new VkDescriptorBinding((uint)i);

            _pipeline = _module.CreateComputePipeline(
                entryPoint: "main",
                bindings: bindings,
                pushConstantBytes: (uint)pushConstantBytes);
        }
        catch
        {
            _module.Dispose();
            throw;
        }

        _descriptorPool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffersPerSet);
        _descriptorCache = new DescriptorSetCache(device, _descriptorPool, _pipeline, buffersPerSet);
    }

    /// <summary>Raw <c>VkPipeline</c> handle — for diagnostics only (e.g. <see cref="VulkanDevice.GetShaderStatisticsAmd"/>).</summary>
    internal nint PipelineHandle => _pipeline.Pipeline;

    /// <summary>Drops every cached descriptor set; call when scratch buffers have been re-allocated.</summary>
    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>
    /// Binds the pipeline and binds the (allocated-or-cached) descriptor set for
    /// <paramref name="buffers"/> into <paramref name="cmdBuf"/>. Does not push
    /// constants or dispatch. This is the call that declares the dispatch's
    /// buffer-access set to the device's active hazard tracker (via
    /// <see cref="DescriptorSetCache.GetOrCreate"/>) — for a chunked dispatch that
    /// reuses the same buffers across multiple <c>vkCmdDispatch</c> calls, call this
    /// once before the loop, not once per chunk (see the hazard-tracking note on
    /// this class).
    /// </summary>
    protected void BindOnly(nint cmdBuf, ReadOnlySpan<nint> buffers)
    {
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);

        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(
            cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout,
            0, 1, descriptorSet, 0, 0);
    }

    /// <summary>
    /// Pushes <paramref name="pushConstants"/> into <paramref name="cmdBuf"/> without
    /// touching the pipeline/descriptor-set bind or the descriptor-set cache. Combine
    /// with <see cref="BindOnly"/> for a chunked dispatch: bind once before the loop,
    /// then call this once per chunk.
    /// </summary>
    protected unsafe void PushConstantsOnly(nint cmdBuf, ReadOnlySpan<uint> pushConstants)
    {
        fixed (uint* pcPtr = pushConstants)
        {
            VulkanApi.vkCmdPushConstants(
                cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute,
                0, (uint)(pushConstants.Length * sizeof(uint)), (nint)pcPtr);
        }
    }

    /// <summary>
    /// Binds the pipeline, binds the (allocated-or-cached) descriptor set for
    /// <paramref name="buffers"/>, and pushes <paramref name="pushConstants"/> into
    /// <paramref name="cmdBuf"/>. Does not dispatch — the caller records
    /// <c>vkCmdDispatch</c> itself. Convenience wrapper of <see cref="BindOnly"/> +
    /// <see cref="PushConstantsOnly"/> for the (common) single-dispatch case; do
    /// <b>not</b> call this per chunk in a chunked dispatch — see the hazard-tracking
    /// note on this class.
    /// </summary>
    protected void BindAndPush(nint cmdBuf, ReadOnlySpan<nint> buffers, ReadOnlySpan<uint> pushConstants)
    {
        BindOnly(cmdBuf, buffers);
        PushConstantsOnly(cmdBuf, pushConstants);
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
