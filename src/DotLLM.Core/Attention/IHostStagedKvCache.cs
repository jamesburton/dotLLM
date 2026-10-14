namespace DotLLM.Core.Attention;

/// <summary>
/// Capability for an <see cref="IKvCache"/> that can stage its per-layer K/V contents through host
/// memory — copy <em>device→host</em> (<see cref="DownloadLayer"/>) and <em>host→device</em>
/// (<see cref="UploadLayer"/>). This is the backend-agnostic seam a cross-device / cross-pool KV handoff
/// uses (see <c>StagedKvHandoffTransfer</c>): the transfer reads a sequence's cache out of one pool/device
/// and rebuilds it in another without the scheduler or the Engine layer knowing which backend is involved.
/// </summary>
/// <remarks>
/// <para>CPU caches implement this almost for free — their "staging" buffer is already host memory, so a
/// download is a contiguous gather and an upload is an ordinary append. Device-local caches (Vulkan/CUDA)
/// bridge their device storage and a host buffer, which is exactly the transport a two-GPU handoff needs
/// when the devices cannot DMA to each other directly (no P2P/NVLink).</para>
/// <para>Unlike the <c>GetKeysRef</c>/<c>Update(TensorRef)</c> surface — which device-local caches do not
/// support (they throw, since their K/V never materializes as a host pointer) — this interface makes the
/// host hop <em>explicit</em>, so the same transfer works for paged-CPU, Vulkan, and (future) CUDA caches.</para>
/// </remarks>
public interface IHostStagedKvCache
{
    /// <summary>
    /// FP32 element count of one layer's key tensor (equal to the value tensor) for the current visible
    /// length: <c>CurrentLength × kvStride(layerIndex)</c>. Sizes the host staging buffer for that layer.
    /// Per-layer because some architectures (e.g. Gemma-4) carry different KV widths per layer.
    /// </summary>
    int StagedLayerElementCount(int layerIndex);

    /// <summary>
    /// Copies layer <paramref name="layerIndex"/>'s cached keys and values, for the current visible length,
    /// into the supplied host spans. Each span must be at least <see cref="StagedLayerElementCount"/> long;
    /// only the leading element-count entries are written. Device-local caches perform a device→host copy.
    /// </summary>
    void DownloadLayer(int layerIndex, Span<float> keys, Span<float> values);

    /// <summary>
    /// Writes <paramref name="length"/> rows of host-resident keys/values into layer
    /// <paramref name="layerIndex"/> at positions <c>[0, length)</c>, advancing the cache's visible length.
    /// Each span must contain exactly <c>length × kvStride(layerIndex)</c> FP32 elements. Device-local
    /// caches perform a host→device copy.
    /// </summary>
    void UploadLayer(int layerIndex, int length, ReadOnlySpan<float> keys, ReadOnlySpan<float> values);
}
