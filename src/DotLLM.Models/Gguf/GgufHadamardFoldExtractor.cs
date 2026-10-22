using System.Collections.Frozen;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Extracts the PrismML blockwise-Hadamard fold declaration (<c>prism.hadamard.*</c>) from GGUF
/// metadata into a <see cref="HadamardFoldConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// Every validation here is deliberately fatal rather than a warning-and-continue. A folded
/// checkpoint whose transform is skipped or applied wrongly does not fail — it generates fluent
/// nonsense, because the weights themselves are well-formed ternary values in the wrong basis.
/// Stock llama.cpp exhibits exactly this failure on these files. Refusing to load is the only
/// behaviour that surfaces the problem.
/// </para>
/// </remarks>
public static class GgufHadamardFoldExtractor
{
    private const string KeyVersion = "prism.hadamard.version";
    private const string KeyBlockSize = "prism.hadamard.block_size";
    private const string KeyTransform = "prism.hadamard.transform";
    private const string KeyAxis = "prism.hadamard.axis";
    private const string KeySignMode = "prism.hadamard.sign_mode";
    private const string KeyWeightNames = "prism.hadamard.weight_names";
    private const string KeySignWidths = "prism.hadamard.sign_widths";
    private const string KeySignValues = "prism.hadamard.sign_values";
    private const string KeyInverseNames = "prism.hadamard.inverse_weight_names";
    private const string KeyGdnVGrouped = "prism.hadamard.gdn_v_grouped";

    private const string SignModeExplicit = "explicit";
    private const string SignModeIdentity = "identity";

    /// <summary>
    /// Reads the fold declaration, or returns <see langword="null"/> when the checkpoint declares
    /// none (the ordinary case — Bonsai 1 and every non-PrismML model).
    /// </summary>
    /// <param name="metadata">Parsed GGUF metadata.</param>
    /// <returns>The fold configuration, or <see langword="null"/> if absent.</returns>
    /// <exception cref="NotSupportedException">
    /// A fold is declared but uses a version, transform, axis or sign mode this build does not
    /// implement.
    /// </exception>
    /// <exception cref="InvalidDataException">The declaration is internally inconsistent.</exception>
    public static HadamardFoldConfig? TryExtract(GgufMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);

        if (!metadata.ContainsKey(KeyVersion))
            return null;

        uint version = metadata.GetUInt32(KeyVersion);
        if (version != HadamardFoldConfig.SupportedVersion)
            throw new NotSupportedException(
                $"Unsupported {KeyVersion}: {version} (this build implements " +
                $"{HadamardFoldConfig.SupportedVersion}). Refusing to load — an unapplied or " +
                "mismatched Hadamard fold produces plausible-looking garbage, not an error.");

        int blockSize = checked((int)metadata.GetUInt32(KeyBlockSize));
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0)
            throw new InvalidDataException($"Invalid {KeyBlockSize}: {blockSize} (must be a power of two).");

        string transform = metadata.GetString(KeyTransform);
        if (!string.Equals(transform, HadamardFoldConfig.SupportedTransform, StringComparison.Ordinal))
            throw new NotSupportedException(
                $"Unsupported {KeyTransform}: '{transform}' (this build implements " +
                $"'{HadamardFoldConfig.SupportedTransform}').");

        string axis = metadata.GetString(KeyAxis);
        if (!string.Equals(axis, HadamardFoldConfig.SupportedAxis, StringComparison.Ordinal))
            throw new NotSupportedException(
                $"Unsupported {KeyAxis}: '{axis}' (this build implements '{HadamardFoldConfig.SupportedAxis}').");

        string[] weightNames = metadata.GetStringArray(KeyWeightNames);
        if (weightNames.Length == 0)
            throw new InvalidDataException($"{KeyWeightNames} is empty — a declared fold must name its weights.");

        var signsByWidth = ReadSigns(metadata);

        string[] inverseNames = metadata.ContainsKey(KeyInverseNames)
            ? metadata.GetStringArray(KeyInverseNames)
            : [];

        // A name in both sets would be transformed twice, in opposite orders — reject rather than
        // pick one.
        var folded = weightNames.ToFrozenSet(StringComparer.Ordinal);
        foreach (string name in inverseNames)
        {
            if (folded.Contains(name))
                throw new InvalidDataException(
                    $"'{name}' appears in both {KeyWeightNames} and {KeyInverseNames}.");
        }

        return new HadamardFoldConfig(
            BlockSize: blockSize,
            SignsByWidth: signsByWidth,
            FoldedWeights: folded,
            InverseWeights: inverseNames.ToFrozenSet(StringComparer.Ordinal),
            GdnVGrouped: metadata.GetBoolOrDefault(KeyGdnVGrouped, false));
    }

    /// <summary>
    /// Reads the concatenated per-width sign vectors. <c>sign_values</c> is one flat array holding
    /// each width's vector back to back, in <c>sign_widths</c> order.
    /// </summary>
    private static FrozenDictionary<int, sbyte[]> ReadSigns(GgufMetadata metadata)
    {
        string signMode = metadata.GetStringOrDefault(KeySignMode, SignModeIdentity);

        if (string.Equals(signMode, SignModeIdentity, StringComparison.Ordinal))
            return FrozenDictionary<int, sbyte[]>.Empty;

        if (!string.Equals(signMode, SignModeExplicit, StringComparison.Ordinal))
            throw new NotSupportedException(
                $"Unsupported {KeySignMode}: '{signMode}' (this build implements " +
                $"'{SignModeExplicit}' and '{SignModeIdentity}').");

        int[] widths = metadata.GetInt32Array(KeySignWidths);
        int[] values = metadata.GetInt32Array(KeySignValues);

        if (widths.Length == 0)
            throw new InvalidDataException($"{KeySignMode} is '{SignModeExplicit}' but {KeySignWidths} is empty.");

        long total = 0;
        foreach (int width in widths)
        {
            if (width <= 0)
                throw new InvalidDataException($"Invalid sign width {width} in {KeySignWidths}.");
            total += width;
        }

        if (total != values.Length)
            throw new InvalidDataException(
                $"{KeySignValues} holds {values.Length} entries but {KeySignWidths} sums to {total}.");

        var result = new Dictionary<int, sbyte[]>(widths.Length);
        int cursor = 0;
        foreach (int width in widths)
        {
            var vector = new sbyte[width];
            for (int i = 0; i < width; i++)
            {
                int v = values[cursor + i];
                if (v != 1 && v != -1)
                    throw new InvalidDataException(
                        $"{KeySignValues}[{cursor + i}] is {v}; Hadamard sign values must be +1 or -1.");
                vector[i] = (sbyte)v;
            }

            if (!result.TryAdd(width, vector))
                throw new InvalidDataException($"Duplicate sign width {width} in {KeySignWidths}.");

            cursor += width;
        }

        return result.ToFrozenDictionary();
    }
}
