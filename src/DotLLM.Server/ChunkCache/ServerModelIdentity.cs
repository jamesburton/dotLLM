using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The identity the server keys chunk-store lookups on: the content model id (<see cref="GgufIdentity.ModelId"/>, sampled),
/// the tokenizer id (<see cref="GgufIdentity.TokenizerId"/>), the quantisation label, the backend tag and the stored K/V
/// element type, folded into the same fingerprint the <c>chunkcache</c> CLI uses
/// (<see cref="StoredContextAssembler.ForModel(ModelConfig, string, string, string)"/>), so an entry the CLI built for this
/// GGUF (without <c>--strict-model-id</c>) is a hit here and nothing else is.
/// </summary>
/// <param name="ModelId">Content model id.</param>
/// <param name="TokenizerId">Content tokenizer id.</param>
/// <param name="Quantization">Quantisation label (<c>ftype-N</c> from <c>general.file_type</c>, as the CLI writes it).</param>
/// <param name="Backend">Backend tag (<see cref="ChunkBackendTags"/>).</param>
/// <param name="KvDtype">Element type of stored K/V (F32: the store's encode type, whatever the serving KV cache uses).</param>
/// <param name="Fingerprint">The model fingerprint hashed into every key.</param>
/// <param name="Rope">RoPE parameters hashed into every key.</param>
/// <param name="UnsupportedReason">Why chunk reuse cannot apply to this model or backend (<c>chunk_model_unsupported</c>), or null.</param>
internal sealed record ServerModelIdentity(
    string ModelId, string TokenizerId, string Quantization, string Backend, KvDtype KvDtype, string Fingerprint,
    RopeParameters Rope, string? UnsupportedReason)
{
    /// <summary>Whether chunk keys can be computed for this model (not recurrent, hybrid, diffusion, or on a backend without assembly).</summary>
    public bool Supported => UnsupportedReason is null;

    /// <summary>The store key of a chunk with <paramref name="tokenIds"/> in namespace <paramref name="ns"/> (isolated positioning).</summary>
    /// <param name="tokenIds">The chunk's ids (<see cref="DotLLM.ChunkReuse.CodeCorpus.ChunkTokenization.Encode"/>).</param>
    /// <param name="ns">The caller's namespace (server-assigned).</param>
    /// <returns>The key.</returns>
    /// <exception cref="InvalidOperationException">The model is not supported.</exception>
    public ChunkKey KeyOf(ReadOnlySpan<int> tokenIds, string ns)
    {
        if (!Supported)
            throw new InvalidOperationException("Chunk keys are not defined for an unsupported model.");
        return ChunkKey.Create(Fingerprint, TokenizerId, Rope, Backend, KvDtype, tokenIds,
            positioning: ChunkPositioning.Isolated, @namespace: ns);
    }

    /// <summary>An identity that supports nothing.</summary>
    /// <param name="reason">Why (operator-facing; never echoed to callers).</param>
    /// <returns>The identity.</returns>
    public static ServerModelIdentity Unsupported(string reason) =>
        new("", "", "", "", KvDtype.F32, "", default, reason);

    /// <summary>Builds the identity from its parts (tests and <see cref="FromGguf"/>).</summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="modelId">Content model id.</param>
    /// <param name="quantization">Quantisation label.</param>
    /// <param name="tokenizerId">Content tokenizer id.</param>
    /// <param name="backend">Backend tag.</param>
    /// <returns>The identity, unsupported when the RoPE variant cannot be re-rotated.</returns>
    public static ServerModelIdentity Create(ModelConfig config, string modelId, string quantization, string tokenizerId, string backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        RopeParameters rope;
        try
        {
            rope = RopeParameters.FromConfig(config);
        }
        catch (NotSupportedException)
        {
            return Unsupported("RoPE variant not supported by chunk re-rotation");
        }
        string fingerprint = ChunkFingerprint.Compute(config, modelId, quantization, tokenizerId, backend);
        return new ServerModelIdentity(modelId, tokenizerId, quantization, backend, KvDtype.F32, fingerprint, rope, null);
    }

    /// <summary>The identity of a loaded GGUF model on <paramref name="backend"/>.</summary>
    /// <param name="gguf">The model file.</param>
    /// <param name="config">The configuration the server loaded.</param>
    /// <param name="backend">Backend tag.</param>
    /// <returns>The identity.</returns>
    public static ServerModelIdentity FromGguf(GgufFile gguf, ModelConfig config, string backend)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        string modelId = GgufIdentity.ModelId(gguf);
        string tokenizerId = GgufIdentity.TokenizerId(gguf.Metadata);
        string quant = GgufIdentity.QuantizationLabel(gguf.Metadata);
        return Create(config, modelId, quant, tokenizerId, backend);
    }

    /// <summary>
    /// The backend tag for a server device string (<see cref="ServerOptions.ResolvedDevice"/> or <see cref="ServerOptions.Device"/>),
    /// or null when that backend has no chunk assembly (CUDA and anything else).
    /// </summary>
    /// <param name="device">Device string, e.g. <c>cpu</c>, <c>vulkan</c>, <c>vulkan:1</c>.</param>
    /// <returns>The tag or null.</returns>
    public static string? BackendTagFor(string? device)
    {
        if (string.IsNullOrEmpty(device) || string.Equals(device, "cpu", StringComparison.OrdinalIgnoreCase))
            return ChunkBackendTags.Cpu;
        if (device.StartsWith("vulkan", StringComparison.OrdinalIgnoreCase))
            return ChunkBackendTags.Vulkan;
        return null;
    }
}
