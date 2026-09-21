using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// The kernels an F16 <see cref="VulkanNemotronHKvCache"/> needs: the F32 -> F16 pack that writes
/// it, and the <c>_kvf16</c> attention variants that can read it (issue #437).
/// </summary>
/// <remarks>
/// <para><b>Why the cache owns these rather than the model.</b> An F16 KV store is unreadable by
/// the ordinary F32 attention kernels — they would interpret packed halves as floats and return
/// garbage, silently. Binding the readers to the store makes that pairing impossible to get wrong:
/// a model asks the cache for its kernel set and gets one that matches the bytes. It also keeps the
/// growth path self-contained, because <see cref="VulkanNemotronHKvCache.EnsureCapacity"/> can
/// invalidate these descriptor caches itself rather than relying on each model to remember.</para>
///
/// <para><b>Coverage.</b> <see cref="SplitKv"/> carries decode from seqKv &gt;= 17 (flash-decoding,
/// the long-context win) and <see cref="Attention"/> carries prefill and short-context decode.
/// There is no F16 flash-attention variant: <c>VulkanFlashAttentionF32Kernel.MaxHeadDim</c> is 128,
/// so Bonsai 2 (headDim 256) never reaches it anyway, and models that would are simply served by
/// <see cref="Attention"/> instead. Coopmat likewise has no F16 variant; it is opt-in and measured
/// slower on the reference hardware, and <see cref="AttentionF32Kernel"/> degrades to the subgroup
/// path when the blob is absent. Both are prefill-throughput caveats of the opt-in F16 mode, not
/// correctness ones.</para>
/// </remarks>
public sealed class VulkanKvF16Kernels : IDisposable
{
    private bool _disposed;

    /// <summary>F32 -> F16 pack used by the cache's append path.</summary>
    public KvPackF16Kernel Pack { get; }

    /// <summary>Tiled attention reading F16 K/V — prefill and short-context decode.</summary>
    public AttentionF32Kernel Attention { get; }

    /// <summary>Split-KV flash-decoding reading F16 K/V; null when the SPV set is unavailable.</summary>
    public VulkanSplitKvAttentionKernel? SplitKv { get; }

    private VulkanKvF16Kernels(KvPackF16Kernel pack, AttentionF32Kernel attention,
                               VulkanSplitKvAttentionKernel? splitKv)
    {
        Pack = pack;
        Attention = attention;
        SplitKv = splitKv;
    }

    /// <summary>
    /// Loads the F16 kernel set from <paramref name="spvDir"/>. Throws when the required blobs
    /// (<c>kv_pack_f16.spv</c>, <c>attention_f32_kvf16.spv</c>) are missing — an F16 cache that
    /// cannot be read is worse than no F16 cache, so this fails loudly rather than degrading.
    /// </summary>
    public static VulkanKvF16Kernels Create(VulkanDevice device, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentException.ThrowIfNullOrEmpty(spvDir);

        KvPackF16Kernel pack = KvPackF16Kernel.Create(device, spvDir);
        AttentionF32Kernel? attention = null;
        VulkanSplitKvAttentionKernel? splitKv = null;
        try
        {
            attention = AttentionF32Kernel.Create(device, spvDir, kvF16: true);
            splitKv = VulkanSplitKvAttentionKernel.TryCreate(device, spvDir, kvF16: true);
        }
        catch
        {
            attention?.Dispose();
            pack.Dispose();
            throw;
        }

        return new VulkanKvF16Kernels(pack, attention, splitKv);
    }

    /// <summary>Drops every cached descriptor set — called after the KV buffers are re-allocated.</summary>
    internal void InvalidateDescriptorCaches()
    {
        Pack.InvalidateDescriptorCache();
        Attention.InvalidateDescriptorCache();
        SplitKv?.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        SplitKv?.Dispose();
        Attention.Dispose();
        Pack.Dispose();
    }
}
