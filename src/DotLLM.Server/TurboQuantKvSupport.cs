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
}

/// <summary>What <see cref="TurboQuantKvSupport.RefusalReason"/> needs to know about where a TurboQuant cache would run.</summary>
/// <param name="Command">The command the request comes from.</param>
/// <param name="Backend">The backend; <see cref="KvBackendKind.Unknown"/> before an <c>auto</c> device has loaded.</param>
/// <param name="Paged">Whether the paged KV cache is selected.</param>
/// <param name="Config">The model configuration (geometry).</param>
/// <param name="HostIsPlainCpuTransformer">Whether the loaded model is the plain CPU <c>TransformerModel</c>; null before load.</param>
/// <param name="HasDraftModel">Whether a speculative draft model is configured (its KV would not be TurboQuant).</param>
/// <param name="MtpWillAttach">Whether an MTP head will attach (its KV would not be TurboQuant).</param>
public sealed record TurboQuantContext(
    KvCommand Command, KvBackendKind Backend, bool Paged,
    ModelConfig Config, bool? HostIsPlainCpuTransformer,
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
            return Refuse($"TurboQuant KV must be used for both K and V (got K={kv.KeyDType}, V={kv.ValueDType}); mixed K/V pairs with TurboQuant arrive in step 6");
        if (ctx.Backend is KvBackendKind.Cuda or KvBackendKind.Vulkan or KvBackendKind.Hip)
            return Refuse($"TurboQuant KV does not run on {ctx.Backend.ToString().ToLowerInvariant()} yet (step 2)");
        if (ctx.Command is KvCommand.Run or KvCommand.Chat)
            return Refuse($"'{ctx.Command.ToString().ToLowerInvariant()}' does not run TurboQuant KV yet (step 2); use `dotllm serve --device cpu --no-paged` until then");
        if (ctx.Paged)
            return Refuse("the paged KV cache has no TurboQuant blocks yet (step 3); pass --no-paged");

        int headDim = ctx.Config.HeadDim;
        if (headDim <= 0 || (headDim & (headDim - 1)) != 0 || headDim > MaxHeadDim)
            return Refuse($"TurboQuant KV needs a power-of-two head dim of at most {MaxHeadDim}; this model has head dim {headDim}");
        if (!KvGeometry.FromConfig(ctx.Config).IsUniform)
            return Refuse("TurboQuant KV needs the same KV width on every layer; this model's layers differ");
        if (ctx.HasDraftModel)
            return Refuse("the speculative draft model's KV would not be TurboQuant; drop --speculative-model");
        if (ctx.MtpWillAttach)
            return Refuse("the MTP head's KV would not be TurboQuant; pass --no-mtp");
        if (ctx.HostIsPlainCpuTransformer == false)
            return Refuse($"TurboQuant KV does not run on the {ctx.Config.Architecture} CPU model yet");
        return null;
    }

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

    private static string Refuse(string what) => $"{what}; {Alternative}. TurboQuant on every backend is tracked in #{EpicIssue}.";
}
