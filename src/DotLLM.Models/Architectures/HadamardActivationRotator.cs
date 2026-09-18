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
    /// <param name="tensorName">
    /// The folded weight's GGUF name. Only used to decide whether the GDN value-head permutation
    /// applies (<c>*.ssm_out.weight</c>); the rotation itself depends only on the width.
    /// </param>
    /// <param name="src">Source activation, <c>seqLen · width</c> floats.</param>
    /// <param name="dst">Destination, same length. Must not alias <paramref name="src"/>.</param>
    /// <param name="seqLen">Number of token rows.</param>
    /// <param name="width">Activation width (the folded weight's input dimension).</param>
    public unsafe void RotateForward(string tensorName, float* src, float* dst, int seqLen, int width)
    {
        var signs = _fold.SignsFor(width);
        var signSpan = signs is null ? ReadOnlySpan<sbyte>.Empty : signs.AsSpan();
        bool permute = _fold.GdnVGrouped && IsSsmOut(tensorName);

        if (permute && width != _dState * _nKHead * _rep)
            throw new InvalidOperationException(
                $"'{tensorName}' input width {width} does not match GDN geometry " +
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

    private static bool IsSsmOut(string tensorName) =>
        tensorName.EndsWith(".ssm_out.weight", StringComparison.Ordinal);
}
