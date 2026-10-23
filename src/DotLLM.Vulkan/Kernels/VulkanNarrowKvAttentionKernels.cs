namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The attention kernels for one narrow (non-F32) <see cref="VulkanKvStorage"/>, plus the writer that fills such a cache (issue #1102).
/// Every kernel here reads the narrow rows in place: decode (split-KV), prefill (coopmat flash, F32 flash, naive) and the per-token
/// fallback each load their K / V elements through <c>kv_load.glsl</c>, so nothing widens the history into an F32 scratch.
/// </summary>
/// <remarks>
/// Built on demand by a host that selects a narrow cache, with the same gates as its F32 set (the env opt-outs, the head_dim bounds):
/// a host never pays for pipelines it will not dispatch. The writer is mandatory (a cache that cannot be written is useless) and so is
/// the per-token kernel (the fallback every shape can reach).
/// </remarks>
internal sealed class VulkanNarrowKvAttentionKernels : IDisposable
{
    private bool _disposed;

    private VulkanNarrowKvAttentionKernels(
        VulkanKvStorage storage, AttentionF32Kernel attention, VulkanFlashAttentionF32Kernel? flash,
        VulkanFlashAttentionCoopmatKernel? coopmat, VulkanSplitKvAttentionKernel? splitKv, VulkanKvF16WriteKernel writer)
    {
        Storage = storage;
        Attention = attention;
        FlashAttention = flash;
        FlashAttentionCoopmat = coopmat;
        SplitKvAttention = splitKv;
        Writer = writer;
    }

    /// <summary>The KV element format every kernel here reads.</summary>
    public VulkanKvStorage Storage { get; }

    /// <summary>Per-token attention (always present; the fallback for any shape).</summary>
    public AttentionF32Kernel Attention { get; }

    /// <summary>Scalar flash prefill attention, or null when disabled or unable to dispatch the head_dim.</summary>
    public VulkanFlashAttentionF32Kernel? FlashAttention { get; }

    /// <summary>Cooperative-matrix flash prefill attention, or null when unavailable for the device / head_dim.</summary>
    public VulkanFlashAttentionCoopmatKernel? FlashAttentionCoopmat { get; }

    /// <summary>Split-KV decode attention, or null when disabled or the head_dim exceeds its bound.</summary>
    public VulkanSplitKvAttentionKernel? SplitKvAttention { get; }

    /// <summary>The F32-to-F16 row writer the cache records its updates with.</summary>
    public VulkanKvF16WriteKernel Writer { get; }

    /// <summary>
    /// Builds the set for <paramref name="storage"/> and a model whose heads are <paramref name="headDim"/> wide.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the SPIR-V blobs.</param>
    /// <param name="headDim">The model's per-head dimension.</param>
    /// <param name="storage">The narrow storage (only <see cref="VulkanKvStorage.F16"/> so far).</param>
    /// <returns>The kernel set.</returns>
    /// <exception cref="NotSupportedException">The storage is not narrow, or a mandatory SPIR-V blob is missing.</exception>
    public static VulkanNarrowKvAttentionKernels Create(VulkanDevice device, string spvDir, int headDim, VulkanKvStorage storage)
    {
        if (storage != VulkanKvStorage.F16)
            throw new NotSupportedException($"No narrow attention kernels exist for KV storage {storage}.");

        AttentionF32Kernel? attention = null;
        VulkanKvF16WriteKernel? writer = null;
        VulkanFlashAttentionF32Kernel? flash = null;
        VulkanFlashAttentionCoopmatKernel? coopmat = null;
        VulkanSplitKvAttentionKernel? splitKv = null;
        try
        {
            attention = AttentionF32Kernel.Create(device, spvDir, storage);
            writer = VulkanKvF16WriteKernel.Create(device, spvDir);
        }
        catch (FileNotFoundException e)
        {
            attention?.Dispose();
            throw new NotSupportedException($"F16 KV needs SPIR-V that this build lacks ({e.Message}).", e);
        }

        try
        {
            // Same gates as VulkanQwen3MoeHybridKernels.Create for the F32 set.
            if (!VulkanTransformerModel.IsFlashAttentionDisabled())
            {
                var wide = headDim <= VulkanFlashAttentionF32Kernel.MaxHeadDim
                    ? FlashAttentionWideVariant.None
                    : VulkanFlashAttentionF32Kernel.WideVariantFromEnvironment();
                flash = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir, wide, includeHd512: false, kvStorage: storage);
                if (flash is not null && headDim > flash.SupportedMaxHeadDim)
                {
                    flash.Dispose();
                    flash = null;
                }

                if (!VulkanTransformerModel.IsCoopmatAttentionDisabled() && headDim <= VulkanFlashAttentionCoopmatKernel.WideMaxHeadDim)
                {
                    coopmat = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, storage);
                    if (coopmat is not null && headDim > coopmat.SupportedMaxHeadDim)
                    {
                        coopmat.Dispose();
                        coopmat = null;
                    }
                }
            }

            if (!VulkanTransformerModel.IsSplitDecodeDisabled() && headDim <= VulkanSplitKvAttentionKernel.MaxHeadDim)
                splitKv = VulkanSplitKvAttentionKernel.TryCreate(device, spvDir, storage);
        }
        catch
        {
            splitKv?.Dispose();
            coopmat?.Dispose();
            flash?.Dispose();
            writer.Dispose();
            attention.Dispose();
            throw;
        }

        return new VulkanNarrowKvAttentionKernels(storage, attention, flash, coopmat, splitKv, writer);
    }

    /// <summary>Drops every kernel's cached descriptor sets (a buffer they may name was freed or re-created).</summary>
    public void InvalidateAll()
    {
        Attention.InvalidateDescriptorCache();
        FlashAttention?.InvalidateDescriptorCache();
        FlashAttentionCoopmat?.InvalidateDescriptorCache();
        SplitKvAttention?.InvalidateDescriptorCache();
        Writer.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        SplitKvAttention?.Dispose();
        FlashAttentionCoopmat?.Dispose();
        FlashAttention?.Dispose();
        Writer.Dispose();
        Attention.Dispose();
    }
}
