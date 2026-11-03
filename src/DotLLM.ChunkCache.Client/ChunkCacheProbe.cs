// SPDX-License-Identifier: MIT
namespace DotLLM.ChunkCache.Client;

/// <summary>The outcome of a fail-closed capability probe (contract ruling 16).</summary>
/// <param name="Capabilities">The server's capabilities when it speaks convention v1; otherwise <see langword="null"/>.</param>
/// <param name="Failure">Why the probe failed (401 / 403 / 5xx, transport error, timeout); <see langword="null"/> when the
/// probe succeeded or the convention is simply absent (404, no v1).</param>
public sealed record ChunkCacheProbeResult(ChunkCacheCapabilities? Capabilities, Exception? Failure)
{
    /// <summary>
    /// Whether requests to this server may carry <c>chunk_cache</c> annotations. When <see langword="false"/>, the
    /// caller strips every annotation and sends a plain request.
    /// </summary>
    public bool IsCapable => Capabilities is not null;
}

/// <summary>The fail-closed probe over any <see cref="IChunkCacheClient"/>.</summary>
public static class ChunkCacheProbe
{
    /// <summary>
    /// Probes the server and never throws for a server-side or transport failure: contract ruling 16 says any
    /// exception or non-2xx from the probe (401, 403, 5xx, timeout, 404) means "strip annotations and send a plain
    /// request". The caller's own cancellation of <paramref name="ct"/> still propagates.
    /// </summary>
    /// <remarks>
    /// <see cref="IChunkCacheClient.GetCapabilitiesAsync"/> keeps the spec 8.2 shape (<see langword="null"/> = absent,
    /// exceptions for a present server that refuses or fails, useful for diagnostics). Installers of the annotating
    /// pipeline should use this method, so a failing probe can never leave annotations on.
    /// </remarks>
    public static async ValueTask<ChunkCacheProbeResult> ProbeAsync(this IChunkCacheClient client, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(client);
        try
        {
            return new ChunkCacheProbeResult(await client.GetCapabilitiesAsync(ct).ConfigureAwait(false), null);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new ChunkCacheProbeResult(null, ex);
        }
    }
}
