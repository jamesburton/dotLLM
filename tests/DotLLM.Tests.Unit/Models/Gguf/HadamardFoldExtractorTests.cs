using System;
using System.Collections.Generic;
using System.Linq;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Verifies the Hadamard fold-set validation that lets the <c>qwen35</c> forward pass rotate at
/// fixed sites instead of consulting the fold list per matmul.
/// </summary>
/// <remarks>
/// The expected set is the one dumped from the real
/// <c>prism-ml/Ternary-Bonsai-2-27B-gguf</c> PQ2_0 file: 401 names over 64 blocks with
/// <c>full_attention_interval = 4</c>.
/// </remarks>
public sealed class HadamardFoldExtractorTests
{
    private const int LayerCount = 64;
    private const int FullAttentionInterval = 4;

    /// <summary>Rebuilds Bonsai 2's declared fold list exactly as the file carries it.</summary>
    private static List<string> Bonsai2FoldNames()
    {
        var names = new List<string> { "output.weight" };
        for (int layer = 0; layer < LayerCount; layer++)
        {
            string p = $"blk.{layer}";
            if ((layer + 1) % FullAttentionInterval == 0)
            {
                names.Add($"{p}.attn_q.weight");
                names.Add($"{p}.attn_k.weight");
                names.Add($"{p}.attn_v.weight");
                names.Add($"{p}.attn_output.weight");
            }
            else
            {
                names.Add($"{p}.attn_qkv.weight");
                names.Add($"{p}.attn_gate.weight");
                names.Add($"{p}.ssm_out.weight");
            }
            names.Add($"{p}.ffn_gate.weight");
            names.Add($"{p}.ffn_up.weight");
            names.Add($"{p}.ffn_down.weight");
        }
        return names;
    }

    private static HadamardActivationRotator RotatorFor(IEnumerable<string> foldNames)
    {
        var config = new HadamardFoldConfig(
            BlockSize: 1024,
            SignsByWidth: System.Collections.Frozen.FrozenDictionary<int, sbyte[]>.Empty,
            FoldedWeights: foldNames.ToFrozenSetOrdinal(),
            InverseWeights: new[] { "token_embd.weight" }.ToFrozenSetOrdinal(),
            GdnVGrouped: true);

        // Bonsai 2 GDN geometry: NVHead 48, NKHead 16, DState 128, DInner 6144, DConv 4.
        var gdn = new GatedDeltaNetConfig(
            FullAttnInterval: FullAttentionInterval,
            NVHead: 48,
            NKHead: 16,
            DState: 128,
            DInner: 6144,
            DConv: 4);

        return new HadamardActivationRotator(config, gdn);
    }

    /// <summary>
    /// The real Bonsai 2 declaration has 401 folded weights — a direct check that the layer-type
    /// split the validator assumes matches the shipped file.
    /// </summary>
    [Fact]
    public void Bonsai2FoldList_Has401Entries_MatchingTheShippedFile()
    {
        var names = Bonsai2FoldNames();
        Assert.Equal(401, names.Count);
        Assert.Equal(401, names.Distinct(StringComparer.Ordinal).Count());

        // 48 GDN layers x 3 + 16 full-attn layers x 4 + 64 ffn x 3 + 1 lm_head.
        Assert.Equal(48, names.Count(n => n.EndsWith(".ssm_out.weight", StringComparison.Ordinal)));
        Assert.Equal(48, names.Count(n => n.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)));
        Assert.Equal(16, names.Count(n => n.EndsWith(".attn_output.weight", StringComparison.Ordinal)));
        Assert.Equal(64, names.Count(n => n.EndsWith(".ffn_down.weight", StringComparison.Ordinal)));
    }

    [Fact]
    public void ValidateQwen35FoldSet_AcceptsTheRealBonsai2Declaration()
    {
        var rotator = RotatorFor(Bonsai2FoldNames());
        rotator.ValidateQwen35FoldSet(LayerCount, FullAttentionInterval);
    }

    /// <summary>
    /// A checkpoint that folds something we do not rotate must be refused, not silently run — the
    /// forward pass would multiply a rotated weight by an unrotated activation.
    /// </summary>
    [Fact]
    public void ValidateQwen35FoldSet_RejectsUndeclaredExtraWeight()
    {
        var names = Bonsai2FoldNames();
        names.Add("blk.0.ssm_alpha.weight"); // not folded in Bonsai 2, and we never rotate its input

        var rotator = RotatorFor(names);
        var ex = Assert.Throws<NotSupportedException>(
            () => rotator.ValidateQwen35FoldSet(LayerCount, FullAttentionInterval));
        Assert.Contains("ssm_alpha", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The mirror case: we rotate an activation the checkpoint did NOT fold, which is equally wrong.
    /// </summary>
    [Fact]
    public void ValidateQwen35FoldSet_RejectsMissingWeight()
    {
        var names = Bonsai2FoldNames();
        names.Remove("output.weight");

        var rotator = RotatorFor(names);
        var ex = Assert.Throws<NotSupportedException>(
            () => rotator.ValidateQwen35FoldSet(LayerCount, FullAttentionInterval));
        Assert.Contains("output.weight", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateQwen35FoldSet_RejectsUnsupportedInverseTable()
    {
        var config = new HadamardFoldConfig(
            BlockSize: 1024,
            SignsByWidth: System.Collections.Frozen.FrozenDictionary<int, sbyte[]>.Empty,
            FoldedWeights: Bonsai2FoldNames().ToFrozenSetOrdinal(),
            InverseWeights: new[] { "token_embd.weight", "blk.0.ffn_down.weight" }.ToFrozenSetOrdinal(),
            GdnVGrouped: false);

        var rotator = new HadamardActivationRotator(config, gdn: null);
        var ex = Assert.Throws<NotSupportedException>(
            () => rotator.ValidateQwen35FoldSet(LayerCount, FullAttentionInterval));
        Assert.Contains("ffn_down", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// <c>gdn_v_grouped</c> without GDN geometry is a contradiction — the permutation has no shape
    /// to work with.
    /// </summary>
    [Fact]
    public void Constructor_RejectsGdnVGrouped_WithoutGdnConfig()
    {
        var config = new HadamardFoldConfig(
            BlockSize: 1024,
            SignsByWidth: System.Collections.Frozen.FrozenDictionary<int, sbyte[]>.Empty,
            FoldedWeights: new[] { "output.weight" }.ToFrozenSetOrdinal(),
            InverseWeights: System.Collections.Frozen.FrozenSet<string>.Empty,
            GdnVGrouped: true);

        Assert.Throws<InvalidOperationException>(() => new HadamardActivationRotator(config, gdn: null));
    }
}

internal static class FrozenSetTestExtensions
{
    public static System.Collections.Frozen.FrozenSet<string> ToFrozenSetOrdinal(this IEnumerable<string> source) =>
        System.Collections.Frozen.FrozenSet.ToFrozenSet(source, StringComparer.Ordinal);
}
