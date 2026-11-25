using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// The FP8 n-gram (PLE) table GGUF as Edge0 ships it (<c>ple-fp8.gguf</c>, written by edge0-strata <c>tools/ple_fp8_pack.py</c>): one tensor
/// <c>per_layer_token_embd.weight</c>, ne = <c>[rowDim, rows]</c> (<c>[160, 320001536]</c>), GGUF type I8 holding FP8 E4M3FN bytes, with the
/// metadata <c>general.architecture = strata-ple</c>, <c>strata.ple.format = f8_e4m3</c> and the table-wide F32 <c>strata.ple.scale</c>.
/// GGUF has no FP8 type, so the loader accepts the raw I8 container ONLY for a file that declares exactly that (I8 stays refused everywhere
/// else, apart from the tensor-overlay container). Decoding is <see cref="DotLLM.Cpu.Kernels.PleRowCodec"/>.
/// </summary>
public static class Fp8PleTable
{
    /// <summary><c>general.architecture</c> of the FP8 PLE table file.</summary>
    public const string Architecture = "strata-ple";

    /// <summary>Metadata key naming the row encoding.</summary>
    public const string FormatKey = "strata.ple.format";

    /// <summary>The only supported value of <see cref="FormatKey"/>.</summary>
    public const string FormatE4M3 = "f8_e4m3";

    /// <summary>Metadata key of the table-wide F32 scale.</summary>
    public const string ScaleKey = "strata.ple.scale";

    /// <summary>The tensor name (same as the IQ4_NL table of the base model).</summary>
    public const string TensorName = "per_layer_token_embd.weight";

    /// <summary>True when the raw metadata declares an FP8 E4M3 PLE table (architecture, format and a positive finite scale).</summary>
    internal static bool IsFp8Table(Dictionary<string, GgufMetadataValue> raw)
        => raw.TryGetValue("general.architecture", out var a) && a.Value is string arch && string.Equals(arch, Architecture, StringComparison.Ordinal)
           && raw.TryGetValue(FormatKey, out var f) && f.Value is string fmt && string.Equals(fmt, FormatE4M3, StringComparison.Ordinal)
           && raw.TryGetValue(ScaleKey, out var s) && s.Value is float scale && scale > 0f && float.IsFinite(scale);

    /// <summary>
    /// Opens the table of an FP8 PLE GGUF: the tensor's pointer into the (lazy) mapping, its row count and width, and the table scale.
    /// </summary>
    /// <param name="gguf">The opened <c>ple-fp8.gguf</c>; must outlive the returned pointer.</param>
    /// <param name="rowDim">Expected elements per row (160); a different width is refused.</param>
    /// <returns>The table pointer, its row count and the scale.</returns>
    public static (nint Table, long Rows, float Scale) Open(GgufFile gguf, int rowDim)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        if (!gguf.TensorsByName.TryGetValue(TensorName, out var d))
            throw new InvalidDataException($"FP8 PLE file has no tensor '{TensorName}'.");
        if (d.QuantizationType != QuantizationType.I8 || d.Shape.Rank != 2 || d.Shape[0] != rowDim)
            throw new InvalidDataException(
                $"FP8 PLE tensor must be I8 [{rowDim}, rows]; got {d.QuantizationType} [{string.Join(", ", d.Shape.Dimensions.ToArray())}].");
        if (!gguf.Metadata.TryGetValue(FormatKey, out var f) || f.Value is not string fmt || fmt != FormatE4M3)
            throw new InvalidDataException($"FP8 PLE file does not declare {FormatKey} = {FormatE4M3}.");
        float scale = gguf.Metadata.GetFloat32(ScaleKey);
        if (!(scale > 0f && float.IsFinite(scale)))
            throw new InvalidDataException($"FP8 PLE file has a non-positive {ScaleKey} ({scale}).");
        return (gguf.TensorDataPointer(d), d.Shape[1], scale);
    }
}
