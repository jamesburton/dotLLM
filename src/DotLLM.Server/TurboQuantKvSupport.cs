using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Server;

/// <summary>Backend a KV cache would live on, as far as the TurboQuant decision cares (#1888).</summary>
public enum KvBackendKind
{
    /// <summary>Not known yet: <c>--device auto</c> before the model loads.</summary>
    Unknown,
    /// <summary>CPU.</summary>
    Cpu,
    /// <summary>CUDA, including the GPU/CPU layer split.</summary>
    Cuda,
    /// <summary>Vulkan.</summary>
    Vulkan,
    /// <summary>HIP (ROCm).</summary>
    Hip,
}

/// <summary>The command a KV request comes from.</summary>
public enum KvCommand
{
    /// <summary><c>dotllm serve</c>.</summary>
    Serve,
    /// <summary><c>dotllm run</c>.</summary>
    Run,
    /// <summary><c>dotllm chat</c>.</summary>
    Chat,
    /// <summary><c>dotllm bench</c>.</summary>
    Bench,
}

/// <summary>What <see cref="TurboQuantKvSupport.RefusalReason"/> needs to know about where a TurboQuant cache would run.</summary>
/// <param name="Command">The command the request comes from.</param>
/// <param name="Backend">The backend; <see cref="KvBackendKind.Unknown"/> before an <c>auto</c> device has loaded.</param>
/// <param name="Paged">Whether the paged KV cache is selected.</param>
/// <param name="Config">The model configuration (geometry); null when no model config is known yet (the geometry rules are then skipped).</param>
/// <param name="HostSupportsInPlace">The loaded model's <c>SupportsInPlaceTurboQuant(kv)</c>; null before load (the host rule is then skipped).</param>
/// <param name="HasDraftModel">Whether a speculative draft model is configured (its KV would not be TurboQuant).</param>
/// <param name="MtpWillAttach">Whether an MTP head will attach (its KV would not be TurboQuant).</param>
/// <param name="CacheWindow">The <c>--cache-window</c> value (0 = none); TurboQuant has no original-space ring to keep a window in.</param>
/// <param name="HostIsPlainCpuTransformer">Whether the loaded model is the plain CPU <c>TransformerModel</c>, the one host that still serves
/// <c>tq*q</c> (QJL) on the dequant-scratch cache until step 5 (#1894); null before load.</param>
public sealed record TurboQuantContext(
    KvCommand Command, KvBackendKind Backend, bool Paged,
    ModelConfig? Config, bool? HostSupportsInPlace,
    bool HasDraftModel, bool MtpWillAttach,
    int CacheWindow = 0, bool? HostIsPlainCpuTransformer = null);

/// <summary>
/// Decides whether a TurboQuant KV request really runs TurboQuant where it was made (#1888, #1887). Every KV path asks this before
/// choosing a cache, so a TurboQuant request is never silently served by another cache: it runs, or startup fails with the reason.
/// Since #1892 the MSE stage runs in place (codes + norms, never a decoded history) on the CPU <c>TransformerModel</c>, the dense CUDA
/// host and the Vulkan dense chain and Flint, for serve, run, chat and bench. What is left refused names its gap ticket.
/// </summary>
public static class TurboQuantKvSupport
{
    /// <summary>Largest head dimension the TurboQuant rotation supports (a power of two).</summary>
    public const int MaxHeadDim = 256;

    /// <summary>Bits per coordinate when a TurboQuant flag gives none (<c>tq</c>, <c>turboquant</c>): <c>KvCacheConfig.TurboQuantBits</c>'s default.</summary>
    public const int DefaultBits = 4;

    /// <summary>
    /// Whether two TurboQuant flags name different formats. A bit width of 0 (none given) means <see cref="DefaultBits"/>, so
    /// <c>tq</c> and <c>tq4</c> are the same format.
    /// </summary>
    public static bool FormatsDiffer(int keyBits, bool keyQjl, int valueBits, bool valueQjl)
        => (keyBits > 0 ? keyBits : DefaultBits) != (valueBits > 0 ? valueBits : DefaultBits) || keyQjl != valueQjl;

    /// <summary>The epic tracking TurboQuant on every backend; cited in every refusal.</summary>
    public static int EpicIssue => 1887;

    private const string Alternative = "use --cache-type-k/-v q8_0 or q4_0 instead";

    /// <summary>
    /// Returns null when <paramref name="kv"/> asks for no TurboQuant or TurboQuant really runs in <paramref name="ctx"/>; otherwise the
    /// user-facing reason it cannot. Rules that need the loaded model (backend, host) are skipped while they are unknown, so the same call
    /// serves a cheap pre-load pass and the authoritative post-load pass. Order: mixed, HIP, QJL on a GPU, the cache window, paged, the
    /// geometry (MLA, head dim, per-layer or distinct V width), draft, MTP, then the host's own capability.
    /// </summary>
    /// <param name="kv">The KV storage request.</param>
    /// <param name="ctx">Where the cache would run.</param>
    /// <returns>The refusal reason, or null.</returns>
    public static string? RefusalReason(KvCacheConfig kv, TurboQuantContext ctx)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        if (!kv.IsTurboQuantAny) return null;

        if (kv.IsTurboQuantMixed)
            return Refuse(Gap.Mixed, $"TurboQuant KV must be used for both K and V (got K={kv.KeyDType}, V={kv.ValueDType}): pass the same tq value to both " +
                "--cache-type-k and --cache-type-v; mixed K/V pairs with TurboQuant arrive in step 6");
        if (ctx.Backend == KvBackendKind.Hip)
            return Refuse(Gap.Hip, "TurboQuant KV does not run on hip yet");
        if (kv.TurboQuantUseQjl && ctx.Backend is KvBackendKind.Cuda or KvBackendKind.Vulkan)
            return Refuse(Gap.Qjl, $"TurboQuant with QJL (tq*q) does not run on {ctx.Backend.ToString().ToLowerInvariant()} yet: " +
                "the in-place GPU kernels cover the MSE stage only (step 5); use tq without the trailing q, or --device cpu for tq*q (QJL)");
        // Not a gap and not a ticket to close: there is no original-space ring in a TurboQuant cache (codes are all it holds), so a
        // mixed-precision window has nothing to keep recent tokens in. Refused before any cache is built.
        if (ctx.CacheWindow > 0)
            return $"--cache-window {ctx.CacheWindow} does not apply to TurboQuant KV: a TurboQuant cache holds codes only, with no original-space " +
                   "window to keep recent tokens in; drop --cache-window (or use --cache-type-k/-v q8_0 or q4_0 for a mixed-precision window). " +
                   $"TurboQuant on every backend is tracked in #{EpicIssue}.";
        if (ctx.Paged)
            return Refuse(Gap.Paged, "the paged KV cache has no TurboQuant blocks yet (step 3); pass --no-paged");

        if (ctx.Config is { } config)
        {
            // MLA (DeepSeek-V2/V3, GLM-4.7-Flash, Kimi) keeps its own latent KV state and ignores the caller's cache: a TurboQuant cache
            // would be allocated and never used. Checked before the head dim, which on MLA is qk_nope + qk_rope and can be a power of two.
            if (config.MlaConfig is not null)
                return Refuse(Gap.Mla, $"TurboQuant KV does not run on MLA models ({config.Architecture}) yet: their attention keeps its own latent KV");
            int headDim = config.HeadDim;
            if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim)
                return Refuse(Gap.HeadDim, $"TurboQuant KV needs a power-of-two head dim of at most {MaxHeadDim}; this model has head dim {headDim}");
            if (config.HasDistinctValueHeadDim)
                return Refuse(Gap.PerLayer, $"TurboQuant KV rotates K and V in one head dim; this model's V head dim ({config.GetValueHeadDim()}) differs from its K head dim ({headDim})");
            if (!KvGeometry.FromConfig(config).IsUniform)
                return Refuse(Gap.PerLayer, "TurboQuant KV needs the same KV width on every layer; this model's layers differ");
        }
        if (ctx.HasDraftModel)
            return Refuse(Gap.DraftMtp, "the speculative draft model's KV would not be TurboQuant; drop --speculative-model");
        if (ctx.MtpWillAttach)
            return Refuse(Gap.DraftMtp, "the MTP head's KV would not be TurboQuant; pass --no-mtp");
        if (ctx.HostSupportsInPlace == false)
        {
            // The one host that still serves tq*q (QJL) is the plain CPU transformer, on the dequant-scratch cache until step 5 (#1894).
            bool qjlScratch = kv.TurboQuantUseQjl && ctx.Backend == KvBackendKind.Cpu && ctx.HostIsPlainCpuTransformer == true;
            if (!qjlScratch)
                return Refuse(Gap.Host, $"TurboQuant KV does not run in place on this {ctx.Config?.Architecture.ToString() ?? "loaded"} " +
                    $"{(ctx.Backend == KvBackendKind.Unknown ? "" : ctx.Backend.ToString().ToLowerInvariant() + " ")}host: its attention arms have no TurboQuant twin " +
                    "(hybrid, recurrent, Qwen4Exp and pipeline hosts) or a kernel it needs is not loaded");
        }
        return null;
    }

    /// <summary>The refusal for two TurboQuant sides whose bit width or QJL flag differ (a mixed format, #1887 step 6).</summary>
    /// <param name="keyText">The K flag as given (e.g. <c>tq3</c>).</param>
    /// <param name="valueText">The V flag as given.</param>
    /// <returns>The refusal reason.</returns>
    public static string MixedFormatRefusal(string keyText, string valueText)
        => Refuse(Gap.Mixed, $"TurboQuant K and V must use the same format (got --cache-type-k {keyText}, --cache-type-v {valueText}); mixed TurboQuant formats arrive in step 6");

    /// <summary>The backend <paramref name="model"/> runs on, by its type: Vulkan, CUDA (including the GPU/CPU split) or HIP models, else CPU.
    /// The post-load host rule (an exact <c>TransformerModel</c> type check) backstops any model this misclassifies.</summary>
    /// <param name="model">A loaded model.</param>
    /// <returns>The backend.</returns>
    public static KvBackendKind BackendOf(IModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        string ns = model.GetType().Namespace ?? "";
        if (ns.StartsWith("DotLLM.Vulkan", StringComparison.Ordinal)) return KvBackendKind.Vulkan;
        if (ns.StartsWith("DotLLM.Cuda", StringComparison.Ordinal)) return KvBackendKind.Cuda;
        if (ns.StartsWith("DotLLM.Hip", StringComparison.Ordinal)) return KvBackendKind.Hip;
        return KvBackendKind.Cpu;
    }

    /// <summary>
    /// The ticket that closes each refused gap (#1887). Each carries a checklist to remove the refusal, its hint and the
    /// "use q8_0 or q4_0 instead" suggestion once the gap is closed: every "use something else" here is a gap, not guidance.
    /// </summary>
    internal static class Gap
    {
        public const int Paged = 1893;     // step 3: paged pools, scheduler, Vulkan pool (#1882), tray
        public const int Mixed = 1895;     // step 6: mixed K/V pairs, V-only, differing bits
        public const int Hip = 1896;
        public const int HeadDim = 1897;
        public const int PerLayer = 1898;
        public const int Mla = 1899;
        public const int Host = 1900;
        public const int Qjl = 1894;       // step 5: tq*q (QJL) on CUDA / Vulkan
        public const int DraftMtp = 1901;
    }

    private static string Refuse(int gap, string what)
        => $"{what}; {Alternative} for now. Closing this gap: #{gap}; TurboQuant on every backend is tracked in #{EpicIssue}.";
}
