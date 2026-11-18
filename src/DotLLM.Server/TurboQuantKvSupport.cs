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
/// <param name="HostIsPlainCpuTransformer">Whether the loaded model is the plain CPU <c>TransformerModel</c>; null before load.</param>
/// <param name="HasDraftModel">Whether a speculative draft model is configured (its KV would not be TurboQuant).</param>
/// <param name="MtpWillAttach">Whether an MTP head will attach (its KV would not be TurboQuant).</param>
public sealed record TurboQuantContext(
    KvCommand Command, KvBackendKind Backend, bool Paged,
    ModelConfig? Config, bool? HostIsPlainCpuTransformer,
    bool HasDraftModel, bool MtpWillAttach);

/// <summary>
/// Decides whether a TurboQuant KV request really runs TurboQuant where it was made (#1888, #1887 step 1). Every KV path asks this before
/// choosing a cache, so a TurboQuant request is never silently served by another cache: it runs, or startup fails with the reason.
/// In step 1 the only real path is CPU <c>serve --no-paged</c> on the plain transformer (<c>TurboQuantKvCache</c>).
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
    /// serves a cheap pre-load pass and the authoritative post-load pass.
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
        if (ctx.Backend is KvBackendKind.Cuda or KvBackendKind.Vulkan or KvBackendKind.Hip)
            return Refuse(ctx.Backend == KvBackendKind.Hip ? Gap.Hip : Gap.InPlace, $"TurboQuant KV does not run on {ctx.Backend.ToString().ToLowerInvariant()} yet (step 2)" +
                (ctx.Command == KvCommand.Serve ? "; for the CPU path pass --device cpu --no-paged" : ""));
        if (ctx.Command is KvCommand.Run or KvCommand.Chat or KvCommand.Bench)
            return Refuse(Gap.InPlace, $"'{ctx.Command.ToString().ToLowerInvariant()}' does not run TurboQuant KV yet (step 2); use `dotllm serve --device cpu --no-paged` until then");
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
            if (!KvGeometry.FromConfig(config).IsUniform)
                return Refuse(Gap.PerLayer, "TurboQuant KV needs the same KV width on every layer; this model's layers differ");
        }
        if (ctx.HasDraftModel)
            return Refuse(Gap.DraftMtp, "the speculative draft model's KV would not be TurboQuant; drop --speculative-model");
        if (ctx.MtpWillAttach)
            return Refuse(Gap.DraftMtp, "the MTP head's KV would not be TurboQuant; pass --no-mtp");
        if (ctx.HostIsPlainCpuTransformer == false)
            return Refuse(Gap.Host, $"TurboQuant KV does not run on the {ctx.Config?.Architecture.ToString() ?? "loaded"} CPU model yet");
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
        public const int InPlace = 1892;   // step 2: CUDA/Vulkan in-place attention; run/chat/bench
        public const int Paged = 1893;     // step 3: paged pools, scheduler, Vulkan pool (#1882), tray
        public const int Mixed = 1895;     // step 6: mixed K/V pairs, V-only, differing bits
        public const int Hip = 1896;
        public const int HeadDim = 1897;
        public const int PerLayer = 1898;
        public const int Mla = 1899;
        public const int Host = 1900;
        public const int DraftMtp = 1901;
    }

    private static string Refuse(int gap, string what)
        => $"{what}; {Alternative} for now. Closing this gap: #{gap}; TurboQuant on every backend is tracked in #{EpicIssue}.";
}
