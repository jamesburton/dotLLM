using System.Collections.ObjectModel;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Canonical safetensors tensor names for a Mamba-3 checkpoint, matching the
/// layout of <c>ib-ssm/mamba3-370M-10BT</c> (commit
/// <c>02943831ad63d36783f41fa872f08cc8631538ee</c>, 2026-04-15).
/// </summary>
/// <remarks>
/// <para>
/// Stage D1 is scaffold-only: this class lists the expected tensor names so
/// Stage D2's safetensors loader has a single source of truth to match
/// against. No weights are allocated here, no tensor shapes are inspected,
/// and the mapping is not yet fed into any model forward.
/// </para>
/// <para>
/// <b>Naming divergence from the reference.</b>
/// <c>VikramKarLex/mamba3-minimal</c> organises each layer as
/// <c>{mixer_norm, mixer, mlp_norm, mlp}</c>, but the HF 370M checkpoint
/// drops the MLP branch entirely and keeps a single pre-mixer norm named
/// <c>norm</c> (not <c>mixer_norm</c>). This loader follows the HF
/// convention — any future checkpoint that re-introduces an MLP path
/// will need a different mapping. Additionally, the HF checkpoint does not
/// store <c>A_log</c> (the reference's per-head decay log-parameter); Stage
/// D2 must resolve where that parameter lives (see the
/// <c>ib_ssm_config.README.md</c> fixture note).
/// </para>
/// <para>
/// Layer tensor names contain an <c>{i}</c> placeholder formatted as the
/// zero-indexed layer number with no zero-padding (e.g.
/// <c>backbone.layers.0.mixer.B_bias</c>, …, <c>backbone.layers.47.mixer.B_bias</c>).
/// </para>
/// </remarks>
public static class Mamba3TensorMapping
{
    /// <summary>
    /// Token-embedding weight, shape <c>[vocab_size, hidden_size]</c>.
    /// </summary>
    public const string TokenEmbedding = "backbone.embeddings.weight";

    /// <summary>
    /// Final pre-LM-head RMSNorm weight, shape <c>[hidden_size]</c>.
    /// </summary>
    public const string FinalNorm = "backbone.norm_f.weight";

    /// <summary>
    /// Language-modelling head weight, shape <c>[vocab_size, hidden_size]</c>.
    /// NB: the HF 370M checkpoint does NOT tie <see cref="TokenEmbedding"/> to
    /// <see cref="LmHead"/> (<c>tie_word_embeddings=false</c>); both are
    /// stored distinctly. Stage D2 must honour the config flag rather than
    /// assuming tying.
    /// </summary>
    public const string LmHead = "lm_head.weight";

    /// <summary>
    /// Pre-mixer RMSNorm weight for block <paramref name="layerIndex"/>,
    /// shape <c>[hidden_size]</c>. Only one norm per layer (no MLP).
    /// </summary>
    public static string LayerNorm(int layerIndex) =>
        $"backbone.layers.{layerIndex}.norm.weight";

    /// <summary>
    /// Input projection for block <paramref name="layerIndex"/>, shape
    /// <c>[d_in_proj, hidden_size]</c>. <c>d_in_proj</c> decomposes as
    /// <c>2*d_inner + 2*bc_dim + 2*num_heads + state_size/2</c>
    /// = the concatenated seven splits
    /// <c>[z, x, B, C, dt, λ, θ]</c>.
    /// </summary>
    public static string InProj(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.in_proj.weight";

    /// <summary>
    /// Output projection for block <paramref name="layerIndex"/>, shape
    /// <c>[hidden_size, d_inner]</c>.
    /// </summary>
    public static string OutProj(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.out_proj.weight";

    /// <summary>
    /// QK-Norm weight on B (pre-DataRoPE) for block
    /// <paramref name="layerIndex"/>, shape <c>[state_size]</c>.
    /// </summary>
    public static string BNorm(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.B_norm.weight";

    /// <summary>
    /// QK-Norm weight on C (pre-DataRoPE) for block
    /// <paramref name="layerIndex"/>, shape <c>[state_size]</c>.
    /// </summary>
    public static string CNorm(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.C_norm.weight";

    /// <summary>
    /// Per-head learned bias added to B after QK-Norm and before DataRoPE.
    /// Shape depends on <c>is_mimo</c>:
    /// <list type="bullet">
    ///   <item>
    ///     <description>
    ///       <b>SISO</b>: <c>[num_heads, 1, state_size]</c> — the HF 370M
    ///       convention with a singleton middle axis (the block squeezes it).
    ///     </description>
    ///   </item>
    ///   <item>
    ///     <description>
    ///       <b>MIMO</b>: <c>[num_heads, mimo_rank, state_size]</c> — the
    ///       canonical rank-expanded layout. Matches <c>B_bias</c> as produced
    ///       by the canonical capture script at
    ///       <c>tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py</c>.
    ///     </description>
    ///   </item>
    /// </list>
    /// </summary>
    public static string BBias(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.B_bias";

    /// <summary>
    /// Per-head learned bias for C; shape matches <see cref="BBias(int)"/>.
    /// </summary>
    public static string CBias(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.C_bias";

    /// <summary>
    /// Per-head skip-connection coefficient <c>D</c>, shape <c>[num_heads]</c>.
    /// </summary>
    public static string D(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.D";

    /// <summary>
    /// Per-head bias added to <c>dt</c> before softplus, shape
    /// <c>[num_heads]</c>.
    /// </summary>
    public static string DtBias(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.dt_bias";

    /// <summary>
    /// MIMO-only per-rank V expansion weight, shape
    /// <c>[num_heads, mimo_rank, head_dim]</c>. Canonical init: <c>1/R</c>.
    /// Present only when the config has <see cref="DotLLM.Core.Models.Mamba3Config.IsMimo"/>.
    /// Consumed by <c>canonical_mimo_scan</c>'s <c>V_mimo[...,r] = V · mimo_x[h,r,p]</c>;
    /// dotLLM's canonical kernel folds this into the rank-summed <c>K.sum</c>
    /// state update, so the tensor is loaded for compatibility with canonical
    /// checkpoints but not consumed directly by the forward today. See
    /// <c>state-spaces/mamba</c> <c>mamba3.py</c> and the canonical capture
    /// script at <c>tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py</c>.
    /// </summary>
    public static string MimoX(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.mimo_x";

    /// <summary>
    /// MIMO-only per-rank gate expansion weight, shape
    /// <c>[num_heads, mimo_rank, head_dim]</c>. Canonical init: <c>1</c>.
    /// Consumed by <see cref="Mamba3Block.ForwardMimo(Mamba3ForwardScratch, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, Span{float}, int, int, int, int, int, int, int, int, int, float, float)"/>
    /// to rank-expand the gate <c>z</c> per rank before the silu gate.
    /// </summary>
    public static string MimoZ(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.mimo_z";

    /// <summary>
    /// MIMO-only per-rank output contraction weight, shape
    /// <c>[num_heads, mimo_rank, head_dim]</c>. Canonical init: <c>1/R</c>.
    /// Contracts the per-rank SSD output back to <c>[n_head, head_dim]</c>
    /// after the scan.
    /// </summary>
    public static string MimoO(int layerIndex) =>
        $"backbone.layers.{layerIndex}.mixer.mimo_o";

    /// <summary>
    /// Enumerates every tensor name expected for a <paramref name="numLayers"/>-layer
    /// Mamba-3 checkpoint. For a SISO checkpoint (<paramref name="isMimo"/> = false)
    /// this is three globals + nine per-layer tensors. For a MIMO checkpoint the
    /// per-layer count increases by three (<c>mimo_x</c>, <c>mimo_z</c>,
    /// <c>mimo_o</c>). Order: globals first, then layers in ascending index
    /// with per-layer tensors in declaration order.
    /// </summary>
    /// <param name="numLayers">Number of <c>backbone.layers.*</c> blocks.</param>
    /// <param name="isMimo">Whether the checkpoint carries the MIMO-only per-rank weights.</param>
    /// <returns>
    /// A fresh read-only list — does NOT cache across calls since the expected
    /// count depends on <paramref name="numLayers"/>.
    /// </returns>
    public static IReadOnlyList<string> ExpectedTensorNames(int numLayers, bool isMimo = false)
    {
        if (numLayers < 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers));

        int perLayer = isMimo ? PerLayerMimoTensorCount : PerLayerTensorCount;
        var names = new List<string>(3 + numLayers * perLayer);
        names.Add(TokenEmbedding);
        names.Add(FinalNorm);
        names.Add(LmHead);
        for (int i = 0; i < numLayers; i++)
        {
            names.Add(LayerNorm(i));
            names.Add(InProj(i));
            names.Add(OutProj(i));
            names.Add(BNorm(i));
            names.Add(CNorm(i));
            names.Add(BBias(i));
            names.Add(CBias(i));
            names.Add(D(i));
            names.Add(DtBias(i));
            if (isMimo)
            {
                names.Add(MimoX(i));
                names.Add(MimoZ(i));
                names.Add(MimoO(i));
            }
        }
        return new ReadOnlyCollection<string>(names);
    }

    /// <summary>
    /// Number of tensors contributed by a single SISO Mamba-3 block:
    /// <c>9</c> — <c>norm + {in_proj, out_proj, B_norm, C_norm, B_bias,
    /// C_bias, D, dt_bias}</c>. Does not include any MLP tensors (the HF
    /// 370M checkpoint has none).
    /// </summary>
    public const int PerLayerTensorCount = 9;

    /// <summary>
    /// Number of tensors contributed by a single MIMO Mamba-3 block:
    /// <see cref="PerLayerTensorCount"/> plus three MIMO-only per-rank weights
    /// (<c>mimo_x</c>, <c>mimo_z</c>, <c>mimo_o</c>).
    /// </summary>
    public const int PerLayerMimoTensorCount = PerLayerTensorCount + 3;

    /// <summary>
    /// Reference (<c>VikramKarLex/mamba3-minimal</c>) state_dict key shapes,
    /// for Stage D3 cross-reference numerical validation. Keys here are
    /// documented as strings rather than used at load time — the HF names
    /// above are what Stage D2's loader actually reads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mapping between HF safetensors names and the reference
    /// <c>state_dict</c> keys. Differences:
    /// </para>
    /// <list type="bullet">
    ///   <item>
    ///     Reference uses <c>backbone.embedding.weight</c> (singular);
    ///     HF uses <c>backbone.embeddings.weight</c> (plural).
    ///   </item>
    ///   <item>
    ///     Reference uses per-layer <c>mixer_norm.weight</c> +
    ///     <c>mlp_norm.weight</c> and a full SwiGLU MLP
    ///     (<c>mlp.w_gate.weight</c>, <c>mlp.w_up.weight</c>,
    ///     <c>mlp.w_down.weight</c>); HF drops the MLP and uses a single
    ///     <c>norm.weight</c>.
    ///   </item>
    ///   <item>
    ///     Reference stores <c>mixer.A_log</c> (shape <c>[num_heads]</c>);
    ///     HF does not. Resolve in Stage D2 before any forward pass.
    ///   </item>
    ///   <item>
    ///     Reference SISO <c>B_bias</c>/<c>C_bias</c> are 2-D
    ///     <c>[num_heads, state_size]</c>; HF is 3-D
    ///     <c>[num_heads, 1, state_size]</c>. Same element count, different
    ///     rank — squeeze the middle axis on load.
    ///   </item>
    /// </list>
    /// </remarks>
    public static class ReferenceKeys
    {
        /// <summary>Reference key for the token embedding.</summary>
        public const string TokenEmbedding = "backbone.embedding.weight";
        /// <summary>Reference key for the final norm.</summary>
        public const string FinalNorm = "backbone.norm_f.weight";
        /// <summary>Reference key for the LM head (tied to embedding by default).</summary>
        public const string LmHead = "lm_head.weight";
        /// <summary>Per-head decay log-parameter (absent from HF checkpoint).</summary>
        public static string ALog(int layerIndex) =>
            $"backbone.layers.{layerIndex}.mixer.A_log";
    }
}
