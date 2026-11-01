using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Identity of the producer of a <see cref="ChunkKvEntry"/>: chunks are only reusable by the exact model, weights, tokenizer,
/// RoPE configuration, KV layout and engine build that encoded them.
/// </summary>
public static class ChunkFingerprint
{
    /// <summary>
    /// Hashes (SHA-256, lowercase hex) the caller-supplied identities that <see cref="ModelConfig"/> does not carry together
    /// with the configuration's shape, its full RoPE configuration, NoPE layers, per-layer KV strides and the engine version.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="modelId">Model identity, e.g. a repo id or the weights file hash.</param>
    /// <param name="quantization">Weight quantization, e.g. <c>Q4_K_M</c>.</param>
    /// <param name="tokenizerId">Tokenizer identity, e.g. a hash of the vocabulary and merges.</param>
    /// <returns>The fingerprint.</returns>
    /// <remarks>Means the CPU backend; equal to <see cref="Compute(ModelConfig, string, string, string, string)"/> with <see cref="ChunkBackendTags.Cpu"/>.</remarks>
    public static string Compute(ModelConfig config, string modelId, string quantization, string tokenizerId)
        => Compute(config, modelId, quantization, tokenizerId, ChunkBackendTags.Cpu);

    /// <summary>As <see cref="Compute(ModelConfig, string, string, string)"/>, for chunks encoded by <paramref name="backend"/>.</summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="modelId">Model identity, e.g. a repo id or the weights file hash.</param>
    /// <param name="quantization">Weight quantization, e.g. <c>Q4_K_M</c>.</param>
    /// <param name="tokenizerId">Tokenizer identity, e.g. a hash of the vocabulary and merges.</param>
    /// <param name="backend">Backend tag (<see cref="ChunkBackendTags"/>): CPU and Vulkan K/V differ by a numerical floor.</param>
    /// <returns>The fingerprint.</returns>
    public static string Compute(ModelConfig config, string modelId, string quantization, string tokenizerId, string backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentException.ThrowIfNullOrEmpty(backend);
        ArgumentNullException.ThrowIfNull(modelId);
        ArgumentNullException.ThrowIfNull(quantization);
        ArgumentNullException.ThrowIfNull(tokenizerId);

        var ci = CultureInfo.InvariantCulture;
        var sb = new StringBuilder("chunk-kv/v1");
        Append(sb, "model", modelId);
        Append(sb, "quant", quantization);
        Append(sb, "tokenizer", tokenizerId);
        Append(sb, "arch", config.Architecture.ToString());
        Append(sb, "shape", string.Join(',',
            config.NumLayers, config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads, config.HeadDim, config.VocabSize));
        if (config.RoPEConfig is { } r)
        {
            Append(sb, "rope", string.Join(',',
                r.Theta.ToString("R", ci), r.DimensionCount.ToString(ci), r.Type.ToString(), r.ScalingType.ToString(),
                r.ScalingFactor.ToString("R", ci), r.OrigMaxSeqLen.ToString(ci), r.AttnFactor.ToString("R", ci),
                r.BetaFast.ToString("R", ci), r.BetaSlow.ToString("R", ci)));
        }
        else
        {
            Append(sb, "rope", "none");
        }
        Append(sb, "nope", config.NoRopeLayers is { Count: > 0 } nope ? string.Join(',', nope) : "");
        var geometry = KvGeometry.FromConfig(config, compactSlidingWindows: false);
        var strides = new string[geometry.LayerCount];
        for (int l = 0; l < strides.Length; l++) strides[l] = geometry.KvStrideOf(l).ToString(ci);
        Append(sb, "kv", string.Join(',', strides));
        Append(sb, "engine", EngineVersion);
        Append(sb, "backend", backend);

        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString())));
    }

    /// <summary>dotLLM build identity (informational version of <c>DotLLM.Core</c>).</summary>
    public static string EngineVersion { get; } =
        typeof(ModelConfig).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
        ?? typeof(ModelConfig).Assembly.GetName().Version?.ToString()
        ?? "unknown";

    // Length-prefixed fields so no value can forge a field boundary.
    private static void Append(StringBuilder sb, string key, string value)
        => sb.Append('|').Append(key).Append('=').Append(value.Length.ToString(CultureInfo.InvariantCulture)).Append(':').Append(value);
}
