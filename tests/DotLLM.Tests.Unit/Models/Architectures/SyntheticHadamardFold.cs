using System.Collections.Frozen;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Builds a PrismML Hadamard fold declaration (<see cref="HadamardFoldConfig"/>) for the synthetic
/// <c>qwen35</c> fixture, so a backend's fold plumbing can be exercised without the 7.66 GB Bonsai 2
/// checkpoint (issue #479).
/// </summary>
/// <remarks>
/// <para>
/// The fixture's weights are NOT actually folded, so a model loaded with this declaration computes
/// a different function from the unfolded one. That is fine for what these tests assert —
/// cross-backend parity with the same declaration, and that each transform site changes the
/// output — and it makes the fold's effect maximally visible: nothing cancels it out.
/// </para>
/// <para>
/// The geometry is chosen so no transform is degenerate: <see cref="GdnKeyHeads"/> = 2 and
/// <see cref="GdnValueHeads"/> = 4 make the <c>gdn_v_grouped</c> tiled→grouped permute a real
/// reordering (it is the identity at the default fixture's single key head), the block size of 8
/// gives every activation several independent blocks, and explicit random signs make the forward
/// (signs → rotation) and inverse (rotation → signs) orders produce different answers.
/// </para>
/// </remarks>
internal static class SyntheticHadamardFold
{
    /// <summary>GDN key heads for the fold fixture (non-degenerate permute needs at least 2).</summary>
    internal const int GdnKeyHeads = 2;

    /// <summary>GDN value heads for the fold fixture (rep = 2 value heads per key head).</summary>
    internal const int GdnValueHeads = 4;

    /// <summary>
    /// Hadamard block width. Must divide every folded width of the fixture: hidden 32, attention
    /// output 32, <c>ssm_out</c> input 4·8 = 32, and <c>ffn_down</c> input 24.
    /// </summary>
    internal const int BlockSize = 8;

    /// <summary>
    /// Writes the fold fixture: a 2-layer (GDN, attention) trunk with the non-degenerate GDN head
    /// geometry, and an MTP head WITHOUT its own <c>nextn.embed_tokens</c> /
    /// <c>nextn.shared_head_head</c>, so the head falls back to the trunk's Hadamard-latent
    /// embedding and folded lm_head — the two MTP fold sites.
    /// </summary>
    internal static string WriteFixture(string path, bool withMtp = true) =>
        SyntheticQwen35HybridDenseMtpGguf.Write(
            path, withMtp: withMtp, mtpHasOwnHeadTensors: false,
            gdnKeyHeads: GdnKeyHeads, gdnValueHeads: GdnValueHeads);

    /// <summary>
    /// Returns a fold declaration covering exactly the weights the <c>qwen35</c> forward pass
    /// rotates for <paramref name="config"/> (what <c>ValidateQwen35FoldSet</c> requires), with
    /// <c>token_embd.weight</c> as the only inverse table.
    /// </summary>
    /// <param name="config">The fixture's extracted config.</param>
    /// <param name="gdnVGrouped">Value of <c>prism.hadamard.gdn_v_grouped</c>.</param>
    /// <param name="withSigns">False for <c>sign_mode</c> other than explicit (identity sign step).</param>
    /// <param name="seed">Seed for the deterministic ±1 sign vectors.</param>
    internal static HadamardFoldConfig For(ModelConfig config, bool gdnVGrouped = true, bool withSigns = true,
        int seed = 0x479)
    {
        var gdn = config.GdnConfig ?? throw new ArgumentException("qwen35 config must carry a GDN config.", nameof(config));
        var kinds = config.HybridLayout!.LayerKind;

        var names = new List<string> { "output.weight" };
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            string p = $"blk.{layer}";
            if (kinds[layer] == HybridLayerKind.Attention)
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

        var signs = new Dictionary<int, sbyte[]>();
        if (withSigns)
        {
            var rng = new Random(seed);
            int[] widths =
            [
                config.HiddenSize,
                config.NumAttentionHeads * config.HeadDim,
                gdn.NVHead * gdn.DState,
                config.IntermediateSize,
            ];
            foreach (int width in widths.Distinct())
            {
                var v = new sbyte[width];
                for (int i = 0; i < width; i++) v[i] = rng.Next(2) == 0 ? (sbyte)-1 : (sbyte)1;
                signs[width] = v;
            }
        }

        return new HadamardFoldConfig(
            BlockSize: BlockSize,
            SignsByWidth: signs.ToFrozenDictionary(),
            FoldedWeights: names.ToFrozenSet(StringComparer.Ordinal),
            InverseWeights: new[] { "token_embd.weight" }.ToFrozenSet(StringComparer.Ordinal),
            GdnVGrouped: gdnVGrouped);
    }
}
