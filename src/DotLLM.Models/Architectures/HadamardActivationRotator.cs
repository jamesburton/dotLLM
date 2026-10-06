using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Applies the PrismML activation-side transform for Hadamard-folded checkpoints
/// (<see cref="HadamardFoldConfig"/>) to a <c>[seqLen, width]</c> activation buffer.
/// </summary>
/// <remarks>
/// <para>
/// The folded weights live in a rotated basis, so each folded weight's input must be rotated to
/// match. The transform writes to a <b>separate</b> destination rather than in place, and that is
/// load-bearing rather than stylistic: in the GDN layers, <c>ssm_alpha</c> and <c>ssm_beta</c> are
/// <b>not</b> folded yet read the very same <c>attn_norm</c> output that the folded <c>attn_qkv</c>
/// and <c>attn_gate</c> consume, and the residual stream reads it too. Rotating in place would
/// silently corrupt all three.
/// </para>
/// <para>
/// One rotation is shared by every folded consumer of the same activation (<c>attn_qkv</c> +
/// <c>attn_gate</c>, <c>ffn_gate</c> + <c>ffn_up</c>, <c>attn_q</c> + <c>attn_k</c> + <c>attn_v</c>),
/// matching the fork's per-(activation, rotation) memoization.
/// </para>
/// </remarks>
public sealed class HadamardActivationRotator
{
    private readonly HadamardFoldConfig _fold;
    private readonly int _dState;
    private readonly int _nKHead;
    private readonly int _rep;
    private readonly float[]? _permScratch;

    /// <summary>
    /// Creates a rotator for a model that declares a Hadamard fold.
    /// </summary>
    /// <param name="fold">The parsed <c>prism.hadamard.*</c> declaration.</param>
    /// <param name="gdn">
    /// GDN geometry, used only for the <c>ssm_out</c> value-head permutation. May be
    /// <see langword="null"/> for non-hybrid architectures.
    /// </param>
    public HadamardActivationRotator(HadamardFoldConfig fold, GatedDeltaNetConfig? gdn)
    {
        _fold = fold ?? throw new ArgumentNullException(nameof(fold));

        if (fold.GdnVGrouped)
        {
            if (gdn is not { } g)
                throw new InvalidOperationException(
                    "prism.hadamard.gdn_v_grouped is set but the model has no GDN configuration.");
            if (g.NKHead <= 0 || g.NVHead <= 0 || g.NVHead % g.NKHead != 0)
                throw new InvalidOperationException(
                    $"prism.hadamard.gdn_v_grouped: bad head geometry NVHead={g.NVHead}, NKHead={g.NKHead}.");

            _dState = g.DState;
            _nKHead = g.NKHead;
            _rep = g.VHeadsPerKHead;
            _permScratch = new float[_dState * _nKHead * _rep];
        }
    }

    /// <summary>The fold declaration this rotator applies.</summary>
    public HadamardFoldConfig Fold => _fold;

    /// <summary>
    /// True when <paramref name="tensorName"/> needs its input activation rotated.
    /// </summary>
    /// <param name="tensorName">GGUF tensor name of the weight about to be multiplied.</param>
    /// <returns>Whether <see cref="RotateForward"/> must run first.</returns>
    public bool IsFolded(string tensorName) => _fold.IsFolded(tensorName);

    /// <summary>
    /// Rotates every row of a <c>[seqLen, width]</c> activation into <paramref name="dst"/>, ready
    /// to be multiplied by the named folded weight.
    /// </summary>
    /// <param name="src">Source activation, <c>seqLen · width</c> floats.</param>
    /// <param name="dst">Destination, same length. Must not alias <paramref name="src"/>.</param>
    /// <param name="seqLen">Number of token rows.</param>
    /// <param name="width">Activation width (the folded weight's input dimension).</param>
    /// <param name="permuteGdnValueHeads">
    /// True only for <c>*.ssm_out.weight</c>, whose fold was computed in grouped value-head order.
    /// Passed as a flag rather than derived from a tensor name so the hot path allocates nothing.
    /// Ignored when the checkpoint does not set <c>prism.hadamard.gdn_v_grouped</c>.
    /// </param>
    public unsafe void RotateForward(
        float* src, float* dst, int seqLen, int width, bool permuteGdnValueHeads = false)
    {
        var signs = _fold.SignsFor(width);
        var signSpan = signs is null ? ReadOnlySpan<sbyte>.Empty : signs.AsSpan();
        bool permute = _fold.GdnVGrouped && permuteGdnValueHeads;

        if (permute && width != _dState * _nKHead * _rep)
            throw new InvalidOperationException(
                $"ssm_out input width {width} does not match GDN geometry " +
                $"{_dState}x{_nKHead}x{_rep} = {_dState * _nKHead * _rep}.");

        for (int t = 0; t < seqLen; t++)
        {
            var srcRow = new ReadOnlySpan<float>(src + (long)t * width, width);
            var dstRow = new Span<float>(dst + (long)t * width, width);

            if (permute)
            {
                // Tiled -> grouped first: the fold was computed in grouped value-head order while
                // the recurrence emits tiled. Signs are indexed in the permuted order.
                Hadamard.PermuteTiledToGrouped(srcRow, _permScratch!, _dState, _nKHead, _rep);
                Hadamard.ForwardRow(_permScratch!, signSpan, dstRow, _fold.BlockSize);
            }
            else
            {
                Hadamard.ForwardRow(srcRow, signSpan, dstRow, _fold.BlockSize);
            }
        }
    }

    /// <summary>
    /// Applies the inverse transform in place to rows fetched from a rotated lookup table
    /// (<c>token_embd.weight</c>): rotation first, then signs.
    /// </summary>
    /// <param name="rows">Fetched rows, <c>seqLen · width</c> floats, transformed in place.</param>
    /// <param name="seqLen">Number of token rows.</param>
    /// <param name="width">Row width.</param>
    public unsafe void RotateInverseInPlace(float* rows, int seqLen, int width)
    {
        var signs = _fold.SignsFor(width);
        var signSpan = signs is null ? ReadOnlySpan<sbyte>.Empty : signs.AsSpan();

        for (int t = 0; t < seqLen; t++)
        {
            var row = new Span<float>(rows + (long)t * width, width);
            Hadamard.InverseRow(row, signSpan, row, _fold.BlockSize);
        }
    }

    /// <summary>
    /// Verifies that the checkpoint folds exactly the set of weights the <c>qwen35</c> forward pass
    /// rotates, and nothing else.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The forward pass rotates at fixed, known sites rather than consulting the fold list per
    /// matmul — that keeps the hot path free of name lookups and string formatting. The safety of
    /// that shortcut rests entirely on the declared set matching what those sites cover, so it is
    /// checked once here at load time.
    /// </para>
    /// <para>
    /// If a future PrismML checkpoint folds a different subset (say it starts folding
    /// <c>ssm_alpha</c>, or stops folding <c>output.weight</c>), this throws instead of quietly
    /// generating text in the wrong basis.
    /// </para>
    /// <para>
    /// <b>Partial-offload splits (issue #481).</b> A CPU/GPU split loads each half as its own model
    /// instance, and each half rotates only the blocks it owns. Such a half passes its GLOBAL owned
    /// layer range (<paramref name="ownedFirstLayer"/>, <paramref name="ownedLayerCount"/>) and whether
    /// it runs the lm_head (<paramref name="ownsLmHead"/>). Two checks then run against different sets:
    /// every declared name must be one the WHOLE model rotates (so a declaration neither half covers,
    /// e.g. a folded MTP block or a folded <c>ssm_alpha</c>, is still refused by both halves), and
    /// every name THIS instance rotates must be declared. The defaults describe a whole-model
    /// instance, for which the two checks together are exact set equality.
    /// </para>
    /// </remarks>
    /// <param name="layerCount">
    /// Number of transformer blocks in the FULL trunk (not a partial-offload half's slice): the
    /// checkpoint's declaration covers every block.
    /// </param>
    /// <param name="fullAttentionInterval">
    /// <c>qwen35.full_attention_interval</c>: block <c>i</c> (1-indexed, GLOBAL) is full GQA attention
    /// when <c>i % interval == 0</c>, and Gated DeltaNet otherwise.
    /// </param>
    /// <param name="ownedFirstLayer">First GLOBAL block index this instance rotates (0 for a whole model).</param>
    /// <param name="ownedLayerCount">
    /// Number of blocks this instance rotates, starting at <paramref name="ownedFirstLayer"/>; negative
    /// (the default) means through the end of the trunk.
    /// </param>
    /// <param name="ownsLmHead">
    /// Whether this instance runs the folded lm_head (<c>output.weight</c>). False for the GPU head of
    /// a split, which stops at the boundary hidden state.
    /// </param>
    /// <exception cref="NotSupportedException">The declared fold set differs from the implemented one.</exception>
    public void ValidateQwen35FoldSet(int layerCount, int fullAttentionInterval,
        int ownedFirstLayer = 0, int ownedLayerCount = -1, bool ownsLmHead = true)
    {
        if (ownedLayerCount < 0)
            ownedLayerCount = layerCount - ownedFirstLayer;
        if (ownedFirstLayer < 0 || ownedLayerCount < 0 || ownedFirstLayer + ownedLayerCount > layerCount)
            throw new ArgumentOutOfRangeException(nameof(ownedFirstLayer),
                $"Owned block range [{ownedFirstLayer}, {ownedFirstLayer + ownedLayerCount}) is outside the " +
                $"{layerCount}-block trunk.");

        // What the whole model rotates: anything declared outside this set is rotated by nobody.
        var expected = Qwen35FoldedNames(0, layerCount, fullAttentionInterval, includeLmHead: true);
        // What this instance rotates: every one of these must be declared.
        var owned = Qwen35FoldedNames(ownedFirstLayer, ownedFirstLayer + ownedLayerCount,
            fullAttentionInterval, ownsLmHead);

        var declared = _fold.FoldedWeights;

        var missing = owned.Where(n => !declared.Contains(n)).Order(StringComparer.Ordinal).Take(5).ToArray();
        var extra = declared.Where(n => !expected.Contains(n)).Order(StringComparer.Ordinal).Take(5).ToArray();

        if (missing.Length > 0 || extra.Length > 0)
        {
            throw new NotSupportedException(
                "prism.hadamard.weight_names does not match the set this build rotates. " +
                $"Declared {declared.Count}, implemented {expected.Count}" +
                (owned.Count != expected.Count
                    ? $" (this instance owns blocks [{ownedFirstLayer}, {ownedFirstLayer + ownedLayerCount})" +
                      $"{(ownsLmHead ? " + lm_head" : "")}: {owned.Count})"
                    : "") + ". " +
                (missing.Length > 0 ? $"Rotated by us but not declared: {string.Join(", ", missing)}. " : "") +
                (extra.Length > 0 ? $"Declared but not rotated by us: {string.Join(", ", extra)}. " : "") +
                "Refusing to load — running with a mismatched fold set produces fluent garbage.");
        }

        // token_embd is the only inverse table the forward pass un-rotates after lookup.
        var unexpectedInverse = _fold.InverseWeights
            .Where(n => !string.Equals(n, "token_embd.weight", StringComparison.Ordinal))
            .Order(StringComparer.Ordinal)
            .ToArray();
        if (unexpectedInverse.Length > 0)
            throw new NotSupportedException(
                $"prism.hadamard.inverse_weight_names contains unsupported entries: " +
                $"{string.Join(", ", unexpectedInverse)}. Only token_embd.weight is un-rotated after lookup.");
    }

    /// <summary>
    /// The folded weight names the <c>qwen35</c> forward pass rotates for GLOBAL blocks
    /// <c>[firstLayer, endLayer)</c>, plus <c>output.weight</c> when <paramref name="includeLmHead"/>.
    /// </summary>
    private static HashSet<string> Qwen35FoldedNames(int firstLayer, int endLayer, int fullAttentionInterval,
        bool includeLmHead)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        if (includeLmHead)
            names.Add("output.weight");

        for (int layer = firstLayer; layer < endLayer; layer++)
        {
            string prefix = $"blk.{layer}";
            bool fullAttention = fullAttentionInterval > 0 && (layer + 1) % fullAttentionInterval == 0;

            if (fullAttention)
            {
                names.Add($"{prefix}.attn_q.weight");
                names.Add($"{prefix}.attn_k.weight");
                names.Add($"{prefix}.attn_v.weight");
                names.Add($"{prefix}.attn_output.weight");
            }
            else
            {
                names.Add($"{prefix}.attn_qkv.weight");
                names.Add($"{prefix}.attn_gate.weight");
                names.Add($"{prefix}.ssm_out.weight");
            }

            names.Add($"{prefix}.ffn_gate.weight");
            names.Add($"{prefix}.ffn_up.weight");
            names.Add($"{prefix}.ffn_down.weight");
        }

        return names;
    }

    private static bool IsSsmOut(string tensorName) =>
        tensorName.EndsWith(".ssm_out.weight", StringComparison.Ordinal);
}
