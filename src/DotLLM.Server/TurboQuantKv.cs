using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Architectures;

namespace DotLLM.Server;

/// <summary>
/// The one place <c>serve</c>, <c>run</c>, <c>chat</c> and <c>bench</c> turn a TurboQuant KV request into a cache (#1892): parse the flags
/// per side, prepare the host, ask <see cref="TurboQuantKvSupport"/> whether it really runs, and build the in-place cache of the
/// backend. Four commands calling one helper is how they stay identical on CPU, CUDA and Vulkan; nothing here falls back to another cache.
/// </summary>
public static class TurboQuantKv
{
    private static long s_inPlaceCaches, s_scratchCaches;

    /// <summary>In-place TurboQuant caches the factories of <see cref="CreateFactory"/> have built since process start (diagnostic / test: proves a command built the in-place cache).</summary>
    public static long InPlaceCachesBuilt => Interlocked.Read(ref s_inPlaceCaches);

    /// <summary>Dequant-scratch TurboQuant caches (the CPU <c>tq*q</c> path) built since process start (diagnostic / test).</summary>
    public static long ScratchCachesBuilt => Interlocked.Read(ref s_scratchCaches);

    /// <summary>
    /// Parses the raw K and V flags (and <c>--cache-window</c>) into one <see cref="KvCacheConfig"/>, K and V independently (#1888): a TurboQuant
    /// side carries its own bit width and QJL flag, and <paramref name="formatMismatch"/> reports two TurboQuant sides that disagree.
    /// </summary>
    /// <param name="keyText">The K flag as given (null = f32).</param>
    /// <param name="valueText">The V flag as given (null = f32).</param>
    /// <param name="cacheWindow">The <c>--cache-window</c> value.</param>
    /// <param name="formatMismatch">The mixed-format refusal, or null.</param>
    /// <returns>The KV configuration.</returns>
    public static KvCacheConfig Parse(string? keyText, string? valueText, int cacheWindow, out string? formatMismatch)
    {
        var keyDType = KvCacheConfig.ParseDType(keyText ?? "f32", out int kBits, out bool kQjl);
        var valueDType = KvCacheConfig.ParseDType(valueText ?? "f32", out int vBits, out bool vQjl);
        bool kTq = keyDType == KvCacheDType.TurboQuant, vTq = valueDType == KvCacheDType.TurboQuant;
        int bits = kTq ? kBits : vTq ? vBits : 0;
        bool qjl = kTq ? kQjl : vTq && vQjl;
        formatMismatch = kTq && vTq && TurboQuantKvSupport.FormatsDiffer(kBits, kQjl, vBits, vQjl)
            ? TurboQuantKvSupport.MixedFormatRefusal(keyText!, valueText!)
            : null;
        return new KvCacheConfig(keyDType, valueDType, Math.Max(0, cacheWindow), TurboQuantBits: bits > 0 ? bits : TurboQuantKvSupport.DefaultBits, TurboQuantUseQjl: qjl);
    }

    /// <summary>
    /// The refusal decidable before the model loads: mixed formats, QJL on an explicit GPU, the cache window, the paged cache, the
    /// geometry, a draft model. Null when nothing is wrong yet; <see cref="PostLoadRefusal"/> then decides the rest.
    /// </summary>
    /// <param name="kv">The parsed request.</param>
    /// <param name="formatMismatch">The mismatch message from <see cref="Parse"/>.</param>
    /// <param name="command">The command.</param>
    /// <param name="backend">The backend, <see cref="KvBackendKind.Unknown"/> for an <c>auto</c> device.</param>
    /// <param name="paged">Whether the paged cache is selected.</param>
    /// <param name="config">The model configuration, or null when not known.</param>
    /// <param name="hasDraftModel">Whether a speculative draft model is configured.</param>
    /// <returns>The refusal, or null.</returns>
    public static string? PreLoadRefusal(KvCacheConfig kv, string? formatMismatch, KvCommand command, KvBackendKind backend, bool paged,
        ModelConfig? config, bool hasDraftModel)
    {
        if (formatMismatch is not null) return formatMismatch;
        return TurboQuantKvSupport.RefusalReason(kv, new TurboQuantContext(
            command, backend, paged, config, HostSupportsInPlace: null, hasDraftModel, MtpWillAttach: false, CacheWindow: kv.MixedPrecisionWindowSize));
    }

    /// <summary>
    /// The authoritative refusal once <paramref name="model"/> has loaded. For a Vulkan host it first builds the in-place twin set (the
    /// codec constants are the Engine's, which the Vulkan project cannot see), then asks the host's own
    /// <see cref="IModel.SupportsInPlaceTurboQuant"/> and runs every rule of <see cref="TurboQuantKvSupport.RefusalReason"/>.
    /// Null means <see cref="CreateFactory"/> builds a cache that really runs TurboQuant.
    /// </summary>
    /// <param name="kv">The parsed request.</param>
    /// <param name="formatMismatch">The mismatch message from <see cref="Parse"/>.</param>
    /// <param name="command">The command.</param>
    /// <param name="paged">Whether the paged cache is selected.</param>
    /// <param name="config">The model configuration.</param>
    /// <param name="model">The loaded model.</param>
    /// <param name="hasDraftModel">Whether a speculative draft model is configured.</param>
    /// <param name="mtpWillAttach">Whether an MTP head will attach.</param>
    /// <returns>The refusal, or null.</returns>
    public static string? PostLoadRefusal(KvCacheConfig kv, string? formatMismatch, KvCommand command, bool paged, ModelConfig config,
        IModel model, bool hasDraftModel, bool mtpWillAttach)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (formatMismatch is not null) return formatMismatch;
        if (!kv.IsTurboQuantAny) return null;
        PrepareHost(model, kv, config);
        return TurboQuantKvSupport.RefusalReason(kv, new TurboQuantContext(
            command, TurboQuantKvSupport.BackendOf(model), paged, config, model.SupportsInPlaceTurboQuant(kv), hasDraftModel, mtpWillAttach,
            CacheWindow: kv.MixedPrecisionWindowSize, HostIsPlainCpuTransformer: model.GetType() == typeof(TransformerModel)));
    }

    /// <summary>
    /// Builds a Vulkan host's in-place twin set for the request when the geometry qualifies, so its capability can answer. A refused or
    /// impossible build changes nothing (the capability then says no and the request is refused).
    /// </summary>
    private static void PrepareHost(IModel model, KvCacheConfig kv, ModelConfig config)
    {
        if (model is not DotLLM.Vulkan.VulkanTransformerModel vk || !TurboQuantInPlaceSupport.Eligible(kv, config)) return;
        var codecK = new TurboQuantCodec(config.HeadDim, kv.TurboQuantBits, kv.TurboQuantSeed);
        var codecV = new TurboQuantCodec(config.HeadDim, kv.TurboQuantBits, kv.TurboQuantSeed ^ TurboQuantCodec.VSeedMask);
        vk.TryUseInPlaceTurboQuantKv(DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir(), codecK.MseBits,
            codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD);
    }

    /// <summary>
    /// The per-request KV factory for a TurboQuant request that <see cref="PostLoadRefusal"/> accepted: the CPU
    /// <see cref="TurboQuantInPlaceKvCache"/> (MSE; <c>tq*q</c> keeps the scratch <see cref="TurboQuantKvCache"/> until step 5), the CUDA
    /// host's <c>CreateKvCache</c>, or the Vulkan host's in-place cache. A host that cannot throws: there is no other cache to build.
    /// </summary>
    /// <param name="model">The loaded model.</param>
    /// <param name="kv">The accepted request.</param>
    /// <returns>The factory.</returns>
    /// <exception cref="NotSupportedException">The host does not run this request in place and it is not the CPU QJL path.</exception>
    public static Func<ModelConfig, int, IKvCache> CreateFactory(IModel model, KvCacheConfig kv)
    {
        ArgumentNullException.ThrowIfNull(model);
        switch (model)
        {
            case DotLLM.Cuda.CudaTransformerModel cuda when cuda.SupportsInPlaceTurboQuant(kv):
                return (_, size) => { Interlocked.Increment(ref s_inPlaceCaches); return cuda.CreateKvCache(size, kv); };
            case DotLLM.Vulkan.VulkanTransformerModel vk when vk.SupportsInPlaceTurboQuant(kv):
                return (_, size) => { Interlocked.Increment(ref s_inPlaceCaches); return vk.CreateKvCache(size); };
            case TransformerModel when kv.TurboQuantUseQjl:
                return (cfg, size) =>
                {
                    Interlocked.Increment(ref s_scratchCaches);
                    return new TurboQuantKvCache(KvGeometry.SlotCount(cfg), cfg.NumKvHeads, cfg.HeadDim, size,
                        kv.TurboQuantBits, kv.TurboQuantSeed, useQjl: true);
                };
            case TransformerModel tm when tm.SupportsInPlaceTurboQuant(kv):
                return (cfg, size) =>
                {
                    Interlocked.Increment(ref s_inPlaceCaches);
                    return new TurboQuantInPlaceKvCache(KvGeometry.SlotCount(cfg), cfg.NumKvHeads, cfg.HeadDim, size,
                        kv.TurboQuantBits, kv.TurboQuantSeed);
                };
            default:
                throw new NotSupportedException(
                    $"TurboQuant KV does not run in place on {model.GetType().Name} (#1900); there is no other TurboQuant cache to build.");
        }
    }
}
