using System.Collections.Frozen;

namespace DotLLM.Core.Models;

/// <summary>
/// PrismML blockwise-Hadamard weight folding (GGUF <c>prism.hadamard.*</c>), as shipped by the
/// Bonsai 2 ternary checkpoints.
/// </summary>
/// <remarks>
/// <para>
/// The checkpoint stores its low-bit weights in a <i>rotated basis</i>: every folded matrix was
/// transformed by an orthogonal blockwise Hadamard rotation (optionally preceded by a fixed ±1 sign
/// flip) before the ternary assignment, and the inverse transform is applied to the <b>activations</b>
/// at runtime. The rotation costs no extra bits and no extra weight traffic — it is folded into the
/// stored weights offline — so the only runtime cost is an O(n log n) pass over activations.
/// </para>
/// <para>
/// A runtime that loads such a file without applying the transform produces fluent-looking garbage
/// rather than failing, so the GGUF config extractor refuses any
/// declaration it does not fully understand instead of guessing.
/// </para>
/// <para>
/// Transform order differs between the two directions, and getting it backwards is silent
/// corruption:
/// </para>
/// <list type="bullet">
///   <item><b>Forward</b> (a folded weight, applied to its input activation):
///     optional <see cref="GdnVGrouped"/> permutation → sign flip → Hadamard.</item>
///   <item><b>Inverse</b> (a table consumed by row lookup, applied to the fetched row):
///     Hadamard → sign flip. This is <c>(H∘S)⁻¹ = S∘H</c>.</item>
/// </list>
/// </remarks>
/// <param name="BlockSize">
/// Hadamard block width (GGUF: <c>prism.hadamard.block_size</c>), always a power of two — 1024 for
/// Bonsai 2. Activation rows are transformed in independent contiguous blocks of this width, so
/// every folded weight's input width must be a multiple of it.
/// </param>
/// <param name="SignsByWidth">
/// Per-input-width sign vectors (GGUF: <c>prism.hadamard.sign_widths</c> +
/// <c>prism.hadamard.sign_values</c>), keyed by activation width. Empty when
/// <c>prism.hadamard.sign_mode</c> is not <c>explicit</c>, in which case the sign step is the
/// identity. Values are exactly ±1.
/// </param>
/// <param name="FoldedWeights">
/// Names of the weights whose <b>input activation</b> must be transformed before the matmul
/// (GGUF: <c>prism.hadamard.weight_names</c>) — 401 entries for Bonsai 2.
/// </param>
/// <param name="InverseWeights">
/// Names of tables consumed by row lookup rather than matmul, whose <b>fetched row</b> must be
/// transformed after the lookup (GGUF: <c>prism.hadamard.inverse_weight_names</c>) — just
/// <c>token_embd.weight</c> for Bonsai 2.
/// </param>
/// <param name="GdnVGrouped">
/// When true (GGUF: <c>prism.hadamard.gdn_v_grouped</c>), the fold for <c>*.ssm_out.weight</c> was
/// computed in <i>grouped</i> GDN value-head order, while the recurrence emits <i>tiled</i> order.
/// The activation must be permuted from <c>[dState, nKHead, rep]</c> to <c>[dState, rep, nKHead]</c>
/// before the sign flip and rotation. See <see cref="GatedDeltaNetConfig"/> for the head geometry;
/// <c>rep = NVHead / NKHead</c>.
/// </param>
public sealed record HadamardFoldConfig(
    int BlockSize,
    FrozenDictionary<int, sbyte[]> SignsByWidth,
    FrozenSet<string> FoldedWeights,
    FrozenSet<string> InverseWeights,
    bool GdnVGrouped)
{
    /// <summary>
    /// The only <c>prism.hadamard.version</c> this implementation understands.
    /// </summary>
    public const uint SupportedVersion = 1;

    /// <summary>
    /// The only <c>prism.hadamard.transform</c> this implementation understands — a Sylvester-ordered
    /// Walsh-Hadamard transform scaled by <c>1/sqrt(BlockSize)</c>, which is what
    /// consumers implement as an in-place FWHT.
    /// </summary>
    public const string SupportedTransform = "normalized-sylvester-walsh-hadamard";

    /// <summary>
    /// The only <c>prism.hadamard.axis</c> this implementation understands: the transform applies
    /// along the activation's last (contiguous, fastest-varying) dimension.
    /// </summary>
    public const string SupportedAxis = "input-last-dimension";

    /// <summary>Scale folded into the transform: <c>1/sqrt(BlockSize)</c>.</summary>
    public float Scale => 1f / MathF.Sqrt(BlockSize);

    /// <summary>
    /// Returns the sign vector for an activation of <paramref name="width"/> elements, or
    /// <see langword="null"/> when the sign step is the identity for this model.
    /// </summary>
    /// <param name="width">Activation width (the folded weight's input dimension).</param>
    /// <returns>The ±1 vector of length <paramref name="width"/>, or <see langword="null"/>.</returns>
    /// <exception cref="InvalidOperationException">
    /// Signs are declared but none covers <paramref name="width"/> — the file is internally
    /// inconsistent and would otherwise decode to garbage.
    /// </exception>
    public sbyte[]? SignsFor(int width)
    {
        if (SignsByWidth.Count == 0)
            return null;

        if (!SignsByWidth.TryGetValue(width, out var signs))
            throw new InvalidOperationException(
                $"prism.hadamard declares explicit signs but has no vector for activation width {width}.");

        return signs;
    }

    /// <summary>
    /// True when <paramref name="tensorName"/> is a folded weight whose input activation needs the
    /// forward transform.
    /// </summary>
    /// <param name="tensorName">GGUF tensor name.</param>
    /// <returns>Whether the forward transform applies.</returns>
    public bool IsFolded(string tensorName) => FoldedWeights.Contains(tensorName);

    /// <summary>
    /// True when <paramref name="tensorName"/> is a lookup table whose fetched rows need the
    /// inverse transform.
    /// </summary>
    /// <param name="tensorName">GGUF tensor name.</param>
    /// <returns>Whether the inverse transform applies.</returns>
    public bool IsInverse(string tensorName) => InverseWeights.Contains(tensorName);
}
