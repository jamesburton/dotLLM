// Ported from SyzygyResearch/Mach-1-Additive-35B's decode.py (Apache License 2.0):
// https://huggingface.co/SyzygyResearch/Mach-1-Additive-35B/blob/main/decode.py
// Apache-2.0 is one-way compatible with dotLLM's GPLv3 (see CLAUDE.md). Ported for
// issue #266 (Phase A: codec decoder).
namespace DotLLM.Models.Quantization.Mach1;

/// <summary>
/// Which expert-tier container a given <c>packed/experts/L{LL}.safetensors</c>
/// file uses. <c>codec.json</c>'s own <c>container</c> field is versioned —
/// the vendor documents it as expected to change — so every caller MUST
/// dispatch on the file's own metadata via <see cref="Mach1ExpertContainer.Detect"/>
/// rather than assuming a container kind.
/// </summary>
public enum Mach1ExpertContainerKind
{
    /// <summary>Unrecognized metadata — neither a <c>manifest</c> key nor a <c>wave_gamma</c> field was found.</summary>
    Unknown = 0,

    /// <summary>
    /// The chunked container (<c>trained_susv_wave_gamma_chunked_v1</c>):
    /// 32-expert stacked tensors, continuous su/sv, per-wavefront gamma.
    /// See <see cref="Mach1ExpertLayerDecoderV3T"/>.
    /// </summary>
    ChunkedV3T,

    /// <summary>
    /// The older per-expert manifest-driven container: separate tensors per
    /// expert, K1 (demoted, int8 signs + Wscale [+ low-rank basis]) or K2
    /// (continuous su/sv) experts. See <see cref="Mach1ExpertLayerDecoderV2"/>.
    /// </summary>
    ManifestV2,
}

/// <summary>
/// Detects which expert-tier container a layer file uses, from its own
/// safetensors metadata — mirrors decode.py's <c>decode_expert_layer</c>
/// dispatch (<c>"wave_gamma" in meta.get("fields")</c> vs. a <c>manifest</c> key).
/// </summary>
public static class Mach1ExpertContainer
{
    /// <summary>Detects the container kind from a layer file's <c>__metadata__</c> dictionary.</summary>
    public static Mach1ExpertContainerKind Detect(IReadOnlyDictionary<string, string> metadata)
    {
        if (Mach1ExpertLayerDecoderV3T.IsV3TContainer(metadata))
            return Mach1ExpertContainerKind.ChunkedV3T;
        if (Mach1ExpertLayerDecoderV2.IsV2Container(metadata))
            return Mach1ExpertContainerKind.ManifestV2;
        return Mach1ExpertContainerKind.Unknown;
    }
}
