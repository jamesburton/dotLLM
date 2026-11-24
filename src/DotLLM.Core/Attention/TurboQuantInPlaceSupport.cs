using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Core.Attention;

/// <summary>
/// The host-independent half of "can this model run in-place TurboQuant KV" (#1892): everything decidable from the KV request and the
/// model's geometry. Every host's <c>IModel.SupportsInPlaceTurboQuant</c> starts here, so CPU, CUDA and Vulkan accept and refuse the
/// same configurations; each then adds what only it knows (loaded kernels, twin sets).
/// </summary>
public static class TurboQuantInPlaceSupport
{
    /// <summary>Largest head dimension the rotation supports (a power of two).</summary>
    public const int MaxHeadDim = 256;

    /// <summary>
    /// True when <paramref name="kv"/> asks for TurboQuant on BOTH sides, MSE stage only (no <c>tq*q</c>/QJL: step 5, #1894), with no
    /// original-space window (there is none with TurboQuant), and <paramref name="config"/> has one uniform KV width, a head dim that
    /// is a power of two of at most <see cref="MaxHeadDim"/> (#1897), the same K and V head dim (#1898) and no MLA latent KV (#1899).
    /// </summary>
    /// <param name="kv">The KV request.</param>
    /// <param name="config">The model configuration.</param>
    /// <returns>Whether the request is one in-place TurboQuant can serve on this geometry.</returns>
    public static bool Eligible(KvCacheConfig kv, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (kv.KeyDType != KvCacheDType.TurboQuant || kv.ValueDType != KvCacheDType.TurboQuant) return false;
        if (kv.TurboQuantUseQjl || kv.MixedPrecisionWindowSize > 0) return false;
        if (kv.TurboQuantBits is < 1 or > 8) return false;
        if (config.MlaConfig is not null || config.HasDistinctValueHeadDim) return false;
        int headDim = config.HeadDim;
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim) return false;
        return KvGeometry.FromConfig(config).IsUniform;
    }

    /// <summary>
    /// Bytes one context token costs in an in-place TurboQuant cache of <paramref name="config"/>: codes plus norms (the
    /// <see cref="TurboQuantLayout"/>), never an F16 figure. For sizing and admission (CLAUDE.md rule 2: the cache holds nothing wider).
    /// </summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="kv">The TurboQuant request (bits).</param>
    /// <returns>Bytes per token over every KV slot.</returns>
    public static long KvBytesPerToken(ModelConfig config, KvCacheConfig kv)
    {
        ArgumentNullException.ThrowIfNull(config);
        var layout = new TurboQuantLayout(config.HeadDim, config.NumKvHeads, kv.TurboQuantBits, TurboQuantLayout.DefaultNorm);
        return layout.KvBytesPerPosition(KvGeometry.SlotCount(config));
    }
}
