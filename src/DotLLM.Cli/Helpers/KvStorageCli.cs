using System.ComponentModel;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda.Architectures;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// Shared <c>--kv-quant-k</c> / <c>--kv-quant-v</c> options (issue #967) for <c>run</c>, <c>bench</c>
/// and <c>serve</c>. Same value grammar as <c>--cache-type-k/-v</c> (<c>f16</c>, <c>q8_0</c>, <c>q4_0</c>);
/// where a command also has <c>--cache-type-*</c> the two are synonyms and must not conflict.
/// </summary>
public abstract class KvQuantSettings : CommandSettings
{
    /// <summary>KV key storage format.</summary>
    [CommandOption("--kv-quant-k")]
    [Description("KV-cache key storage: f16, q8_0, q4_0. Synonym of --cache-type-k. For the CUDA hybrid-dense " +
                 "model (Bonsai-2 / qwen35) overrides DOTLLM_HYBRID_KV; quantized KV needs numKvHeads*headDim % 32 == 0.")]
    public string? KvQuantK { get; set; }

    /// <summary>KV value storage format.</summary>
    [CommandOption("--kv-quant-v")]
    [Description("KV-cache value storage: f16, q8_0, q4_0. Synonym of --cache-type-v. See --kv-quant-k.")]
    public string? KvQuantV { get; set; }
}

/// <summary>Where a model's KV cache lives, as far as honouring KV-storage flags is concerned.</summary>
public enum KvFlagTarget
{
    /// <summary>CUDA Qwen3HybridDense: owns its KV and honours the flags (this issue).</summary>
    CudaHybridDense,
    /// <summary>CUDA plain transformer: honours quantized KV via the engine cache (run/serve only).</summary>
    CudaPlain,
    /// <summary>CPU backends: honour via the engine's QuantizedKvCache (run/serve only).</summary>
    Cpu,
    /// <summary>Vulkan VulkanTransformerModel: device-resident KV, quantized storage supported.</summary>
    VulkanQuantSupported,
    /// <summary>Vulkan: device-resident KV, no quantized storage.</summary>
    Vulkan,
    /// <summary>A model that owns its KV (split GPU/CPU, MoE-hybrid, Nemotron-H) and has no quantized storage.</summary>
    OwnCacheUnsupported,
    /// <summary>Unknown model type: handled by the generic engine cache path.</summary>
    Other,
}

/// <summary>A parsed KV storage request: which flags were given and what they resolved to.</summary>
internal sealed record KvStorageRequest(
    KvCacheDType? Key, KvCacheDType? Value, string? KeyFlag, string? ValueFlag, int CacheWindow = 0)
{
    /// <summary>True when any key/value flag (or a non-zero window) was supplied.</summary>
    public bool IsSpecified => Key is not null || Value is not null || CacheWindow > 0;

    /// <summary>No flags given.</summary>
    public static KvStorageRequest None { get; } = new(null, null, null, null);
}

/// <summary>Flag parsing, the warn/reject matrix and apply/summary glue shared by run, bench and serve.</summary>
internal static class KvStorageCli
{
    /// <summary>
    /// Builds a request from the raw option strings. <c>--kv-quant-*</c> and <c>--cache-type-*</c> are
    /// synonyms; giving both with different values is an error.
    /// </summary>
    public static bool TryBuild(
        string? cacheTypeK, string? cacheTypeV, string? kvQuantK, string? kvQuantV, int cacheWindow,
        out KvStorageRequest request, out string? error)
    {
        request = KvStorageRequest.None;
        error = null;
        try
        {
            if (!TrySide("key", "--cache-type-k", cacheTypeK, "--kv-quant-k", kvQuantK,
                    out var key, out var keyFlag, out error)) return false;
            if (!TrySide("value", "--cache-type-v", cacheTypeV, "--kv-quant-v", kvQuantV,
                    out var value, out var valueFlag, out error)) return false;
            request = new KvStorageRequest(key, value, keyFlag, valueFlag, Math.Max(0, cacheWindow));
            return true;
        }
        catch (ArgumentException ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private static bool TrySide(
        string side, string nameA, string? rawA, string nameB, string? rawB,
        out KvCacheDType? dtype, out string? flag, out string? error)
    {
        dtype = null; flag = null; error = null;
        var a = HybridKvStorage.ParseFlag(rawA);
        var b = HybridKvStorage.ParseFlag(rawB);
        if (a is not null && b is not null && a != b)
        {
            error = $"{nameA} ({rawA}) and {nameB} ({rawB}) are synonyms for the KV {side} format but disagree; pass only one.";
            return false;
        }
        dtype = b ?? a;
        flag = b is not null ? nameB : a is not null ? nameA : null;
        return true;
    }

    /// <summary>
    /// The warning matrix: one message per flag that the selected model/backend cannot honour (never silent).
    /// </summary>
    /// <param name="request">Parsed flags.</param>
    /// <param name="target">Model/backend classification.</param>
    /// <param name="modelName">Architecture / model label for the message.</param>
    /// <param name="backend">Backend label (cpu, cuda, vulkan).</param>
    /// <param name="engineCacheHonoured">True for run/serve, whose engine cache implements quantized KV for plain and CPU models; false for bench (SimpleKvCache).</param>
    public static IReadOnlyList<string> Warnings(
        KvStorageRequest request, KvFlagTarget target, string modelName, string backend, bool engineCacheHonoured)
    {
        var list = new List<string>();
        if (!request.IsSpecified) return list;

        bool honoured = target switch
        {
            KvFlagTarget.CudaHybridDense => true,
            KvFlagTarget.VulkanQuantSupported => true,
            KvFlagTarget.CudaPlain or KvFlagTarget.Cpu or KvFlagTarget.Other => engineCacheHonoured,
            _ => false,
        };

        string Why() => target switch
        {
            KvFlagTarget.Vulkan => "the Vulkan backend has no quantized KV storage",
            KvFlagTarget.OwnCacheUnsupported => "this model owns its KV cache and has no quantized storage",
            KvFlagTarget.Cpu or KvFlagTarget.CudaPlain or KvFlagTarget.Other =>
                "this command uses a plain full-precision KV cache (quantized KV is honoured by 'run' and 'serve')",
            _ => "unsupported",
        };

        if (!honoured)
        {
            if (request.Key is not null)
                list.Add($"{request.KeyFlag} {HybridKvStorage.Label(request.Key.Value, request.Key.Value)} is ignored: {Why()} (model {modelName}, backend {backend}).");
            if (request.Value is not null)
                list.Add($"{request.ValueFlag} {HybridKvStorage.Label(request.Value.Value, request.Value.Value)} is ignored: {Why()} (model {modelName}, backend {backend}).");
        }

        if (request.CacheWindow > 0 && target == KvFlagTarget.CudaHybridDense)
            list.Add($"--cache-window {request.CacheWindow} is ignored: the CUDA hybrid-dense model stores the whole KV in one format " +
                     $"(no mixed-precision window) (model {modelName}, backend {backend}).");
        return list;
    }

    /// <summary>Classifies a loaded model.</summary>
    public static KvFlagTarget Classify(IModel model, bool isVulkan)
    {
        if (isVulkan)
            return model is DotLLM.Vulkan.VulkanTransformerModel ? KvFlagTarget.VulkanQuantSupported : KvFlagTarget.Vulkan;
        return model switch
        {
            CudaQwen3HybridDenseTransformerModel => KvFlagTarget.CudaHybridDense,
            HybridQwen3HybridDenseTransformerModel or DotLLM.Cuda.HybridTransformerModel
                or CudaQwen3MoeHybridTransformerModel or CudaNemotronHTransformerModel or CudaLfm2MoeTransformerModel => KvFlagTarget.OwnCacheUnsupported,
            DotLLM.Cuda.CudaTransformerModel => KvFlagTarget.CudaPlain,
            _ when model.GetType().Namespace?.StartsWith("DotLLM.Vulkan", StringComparison.Ordinal) == true => KvFlagTarget.Vulkan,
            _ when model.GetType().Namespace?.StartsWith("DotLLM.Cuda", StringComparison.Ordinal) == true => KvFlagTarget.OwnCacheUnsupported,
            _ => KvFlagTarget.Cpu,
        };
    }

    /// <summary>
    /// Applies an explicit request to a CUDA hybrid-dense model (no-op for others). Throws
    /// <see cref="NotSupportedException"/> with a user-facing message for combinations that would be wrong.
    /// </summary>
    public static void Apply(IModel model, KvStorageRequest request)
    {
        if (model is CudaLfm2MoeTransformerModel)
            CudaLfm2MoeTransformerModel.ValidateKvStorage(request.Key ?? KvCacheDType.F32, request.Value ?? KvCacheDType.F32, request.CacheWindow);
        if (model is CudaQwen3HybridDenseTransformerModel dense && (request.Key is not null || request.Value is not null))
            dense.ConfigureKvStorage(request.Key, request.Value);
    }

    /// <summary>
    /// Effective KV format and size for the summary line of a hybrid-dense CUDA model, or null for other
    /// models. Bytes are the allocated size when known, else the estimate for <paramref name="cacheTokens"/>.
    /// </summary>
    public static (string Label, long Bytes, string Text)? Describe(IModel model, int cacheTokens)
    {
        if (model is not CudaQwen3HybridDenseTransformerModel dense) return null;
        long bytes = dense.KvAllocatedBytes > 0 ? dense.KvAllocatedBytes : dense.ComputeKvBytes(cacheTokens);
        string label = dense.KvStorageLabel;
        return (label, bytes, $"KV {label} (K:{dense.KvKeyStorage} V:{dense.KvValueStorage}), {bytes / 1048576.0:F0} MiB");
    }
}
