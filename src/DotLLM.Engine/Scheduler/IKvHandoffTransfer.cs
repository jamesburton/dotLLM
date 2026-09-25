using System.Buffers;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Scheduler;

/// <summary>
/// Pluggable strategy for handing a sequence's KV-cache from the prefill replica to the decode replica
/// in a <see cref="DisaggregatedScheduler"/>. Abstracts <em>how</em> the decode replica obtains an
/// equivalent KV-cache for a just-prefilled sequence:
/// <list type="bullet">
/// <item><see cref="ReferenceKvHandoffTransfer"/> — the two replicas share one pool, so the same cache
/// object is reused by reference (zero copy; the original #354 behaviour).</item>
/// <item><see cref="CopyKvHandoffTransfer"/> — the replicas have <em>separate</em> pools, so the K/V
/// block contents are serialized out of the prefill cache and rebuilt in a fresh decode-pool cache.
/// This is the in-process stand-in for a real cross-process / cross-device transfer (where the source
/// and destination caches live in different address spaces or on different GPUs).</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>Future work (needs hardware):</b> a true cross-device transfer copies through a device→host→
/// device (or NCCL / RDMA / NVLink) path with explicit device placement on both ends. That cannot be
/// validated on a single-GPU box, so <see cref="CopyKvHandoffTransfer"/> only proves the
/// <em>content-transfer correctness</em> by copying between two CPU pools on one machine; the transport
/// is swapped for a same-host copy. When multi-device hardware is available, an additional
/// implementation can route the copy across devices behind this same seam without touching the scheduler.</para>
/// </remarks>
public interface IKvHandoffTransfer
{
    /// <summary>
    /// Produces the KV-cache the decode replica should use for a sequence whose prefill KV-cache is
    /// <paramref name="source"/>. Implementations either return <paramref name="source"/> unchanged
    /// (shared-pool, by reference) or materialize a fresh equivalent cache from
    /// <paramref name="destinationFactory"/> (separate-pool, by copy) and dispose <paramref name="source"/>.
    /// </summary>
    /// <param name="source">The KV-cache the prefill replica built (over the prefill pool).</param>
    /// <param name="config">Model config of the replica pair (identical architecture / KV layout).</param>
    /// <param name="destinationFactory">Factory creating a cache from the <em>decode</em> pool, given
    /// the config and a max-sequence-length. Only used by copy-based implementations.</param>
    /// <returns>The cache the decode replica must attend over and append to. May be the same instance
    /// as <paramref name="source"/> (reference) or a new instance (copy).</returns>
    IKvCache Transfer(IKvCache source, ModelConfig config, Func<ModelConfig, int, IKvCache> destinationFactory);
}

/// <summary>
/// Zero-copy handoff: the prefill and decode replicas share one KV pool, so the decode replica attends
/// the exact cache object the prefill replica populated. This is the default and matches the original
/// shared-pool disaggregation behaviour (PR #354).
/// </summary>
public sealed class ReferenceKvHandoffTransfer : IKvHandoffTransfer
{
    /// <summary>Shared singleton — the strategy is stateless.</summary>
    public static readonly ReferenceKvHandoffTransfer Instance = new();

    /// <inheritdoc/>
    public IKvCache Transfer(IKvCache source, ModelConfig config, Func<ModelConfig, int, IKvCache> destinationFactory)
        => source; // same pool — hand off by reference, no copy.
}

/// <summary>
/// Copy-based handoff: the prefill and decode replicas use <em>separate</em> pools. The prefill cache's
/// per-layer K/V contents are read out and written into a freshly allocated decode-pool cache, then the
/// prefill cache is disposed (returning its blocks to the prefill pool). The decode replica decodes over
/// the new cache. This simulates a cross-device KV transfer on a single box: the transferred state is a
/// byte-for-byte copy, so decode output is identical to the by-reference path.
/// </summary>
public sealed class CopyKvHandoffTransfer : IKvHandoffTransfer
{
    /// <summary>Shared singleton — the strategy is stateless.</summary>
    public static readonly CopyKvHandoffTransfer Instance = new();

    /// <inheritdoc/>
    public IKvCache Transfer(IKvCache source, ModelConfig config, Func<ModelConfig, int, IKvCache> destinationFactory)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destinationFactory);

        // Allocate a fresh cache from the destination (decode) pool, sized to hold the full sequence.
        IKvCache destination = destinationFactory(config, source.MaxLength);
        try
        {
            KvCacheCopy.CopyContents(source, destination, config.NumLayers);
        }
        catch
        {
            destination.Dispose();
            throw;
        }

        // The source cache (on the prefill pool) is no longer needed; free its blocks back to that pool.
        source.Dispose();
        return destination;
    }
}

/// <summary>
/// Cross-device / cross-pool handoff via an explicit host hop. Each layer's K/V is copied
/// <em>device→host</em> out of the prefill cache and <em>host→device</em> into a fresh decode-pool cache
/// over the backend-agnostic <see cref="IHostStagedKvCache"/> seam, then the prefill cache is disposed.
/// Unlike <see cref="CopyKvHandoffTransfer"/> (which uses the <c>GetKeysRef</c>/<c>Update(TensorRef)</c>
/// surface that device-local caches do not support), this path works for Vulkan/CUDA caches because the
/// host staging is explicit — it is the production transport for a two-GPU prefill→decode handoff where the
/// devices cannot DMA to each other directly. On a single CPU box (separate pools) it is byte-equivalent to
/// the copy transfer, so it is unit-tested there; the real device→host→device run is hardware-gated.
/// </summary>
public sealed class StagedKvHandoffTransfer : IKvHandoffTransfer
{
    /// <summary>Shared singleton — the strategy is stateless.</summary>
    public static readonly StagedKvHandoffTransfer Instance = new();

    /// <inheritdoc/>
    public IKvCache Transfer(IKvCache source, ModelConfig config, Func<ModelConfig, int, IKvCache> destinationFactory)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destinationFactory);

        if (source is not IHostStagedKvCache stagedSource)
            throw new NotSupportedException(
                $"Source KV-cache {source.GetType().Name} does not implement {nameof(IHostStagedKvCache)}; "
                + "staged transfer requires host-staging support on both ends.");

        // Allocate a fresh cache from the destination (decode) pool/device, sized to hold the full sequence.
        IKvCache destination = destinationFactory(config, source.MaxLength);
        try
        {
            if (destination is not IHostStagedKvCache stagedDestination)
                throw new NotSupportedException(
                    $"Destination KV-cache {destination.GetType().Name} does not implement {nameof(IHostStagedKvCache)}.");

            int length = source.CurrentLength;
            if (length > 0)
                StageLayers(stagedSource, stagedDestination, config.NumLayers, length);
        }
        catch
        {
            destination.Dispose();
            throw;
        }

        // The source cache (on the prefill pool/device) is no longer needed; free its blocks.
        source.Dispose();
        return destination;
    }

    private static void StageLayers(IHostStagedKvCache source, IHostStagedKvCache destination, int numLayers, int length)
    {
        for (int layer = 0; layer < numLayers; layer++)
        {
            // Per-layer rent: KV row width can differ per layer (e.g. Gemma-4 sliding vs global layers).
            int count = source.StagedLayerElementCount(layer);
            float[] keys = ArrayPool<float>.Shared.Rent(count);
            float[] values = ArrayPool<float>.Shared.Rent(count);
            try
            {
                source.DownloadLayer(layer, keys.AsSpan(0, count), values.AsSpan(0, count));
                destination.UploadLayer(layer, length, keys.AsSpan(0, count), values.AsSpan(0, count));
            }
            finally
            {
                ArrayPool<float>.Shared.Return(keys);
                ArrayPool<float>.Shared.Return(values);
            }
        }
    }
}

/// <summary>
/// Backend-agnostic copy of KV-cache contents between two <see cref="IKvCache"/> instances, layer by
/// layer, over the <see cref="IKvCache.GetKeysRef"/>/<see cref="IKvCache.GetValuesRef"/> /
/// <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/> public surface. Works for
/// any cache implementation that exposes contiguous per-layer K/V views (paged, simple, quantized).
/// </summary>
internal static class KvCacheCopy
{
    /// <summary>
    /// Copies all <paramref name="numLayers"/> layers' K/V entries from <paramref name="source"/> into
    /// <paramref name="destination"/>, preserving positions <c>0..CurrentLength-1</c>. The destination
    /// must be empty and large enough (<c>MaxLength &gt;= source.CurrentLength</c>).
    /// </summary>
    public static void CopyContents(IKvCache source, IKvCache destination, int numLayers)
    {
        int length = source.CurrentLength;
        if (length == 0) return;
        if (destination.MaxLength < length)
            throw new ArgumentException(
                $"Destination cache MaxLength {destination.MaxLength} cannot hold {length} positions.",
                nameof(destination));

        int[] positions = ArrayPool<int>.Shared.Rent(length);
        try
        {
            for (int p = 0; p < length; p++) positions[p] = p;
            var posSpan = positions.AsSpan(0, length);

            for (int layer = 0; layer < numLayers; layer++)
            {
                // Each GetKeysRef/GetValuesRef gathers into the source's per-layer staging buffer; we
                // consume it (copy into the destination) before advancing to the next layer, so the
                // single-staging-buffer reuse across layers is safe.
                TensorRef keys = source.GetKeysRef(layer);
                TensorRef values = source.GetValuesRef(layer);
                destination.Update(keys, values, posSpan, layer);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(positions);
        }
    }
}
