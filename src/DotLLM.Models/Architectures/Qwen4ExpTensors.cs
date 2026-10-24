using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// The GGUF tensor-name / shape contract of a <c>qwen4exp</c> checkpoint, derived from <see cref="ModelConfig"/>.
/// Mirrors llama.cpp's <c>llama_model_qwen4exp::load_arch_tensors</c> and was checked tensor-for-tensor against the real
/// <c>unsloth/Qwen3.8-Flash-Next-GGUF</c> headers (1224 trunk tensors over 4 shards, 34 MTP tensors).
/// </summary>
/// <remarks>
/// <para>Dimensions are in GGUF order (<c>ne0</c> first, i.e. the contraction/input axis first for a matrix).
/// Multi-stream norm gammas are stored flat (<c>[hc * n_embd]</c>); llama.cpp reshapes them to
/// <c>[n_embd, hc]</c> at load.</para>
/// <para>Differences from the pre-implementation design notes (§1.7), established from the real header: indexer tensors are
/// DOTTED (<c>blk.N.indexer.q_proj</c>); the final head mixer is <c>output_hc_{norm,down,up}</c> (the <c>hc_head_*</c> prefix
/// exists only inside <c>blk.48.nextn.*</c>); routed experts ship SPLIT (<c>ffn_gate_exps</c> + <c>ffn_up_exps</c>, not a fused
/// <c>gate_up</c>); <c>ssm_a</c> has no <c>.weight</c> suffix and <c>ssm_dt</c> is a <c>.bias</c>; the PLE table is the single
/// tensor <c>per_layer_token_embd.weight</c> (<c>[row_dim, rows]</c>, <c>rows</c> padded above the head ranges).</para>
/// <para>The MTP block is NOT in the trunk file (no <c>nextn_predict_layers</c> key, 48 blocks): it ships as a separate
/// 49-block GGUF carrying only <c>token_embd</c>, <c>output</c> and <c>blk.48.*</c> (no <c>output_hc_*</c>, no PLE).</para>
/// </remarks>
public static class Qwen4ExpTensors
{
    /// <summary>One tensor the model expects.</summary>
    /// <param name="Name">GGUF tensor name.</param>
    /// <param name="Dims">Expected dimensions, <c>ne0</c> first. For <paramref name="LastDimIsMinimum"/> tensors the last entry is a lower bound.</param>
    /// <param name="Required">False for tensors llama.cpp loads with <c>TENSOR_NOT_REQUIRED</c> (<c>output.weight</c>: tied to <c>token_embd</c> when absent).</param>
    /// <param name="LastDimIsMinimum">True for the padded n-gram table, whose row count only has to cover the head ranges.</param>
    public readonly record struct Expected(string Name, long[] Dims, bool Required = true, bool LastDimIsMinimum = false);

    // ───────────────────────────── name builders ─────────────────────────────

    /// <summary>Token embedding.</summary>
    public const string TokenEmbd = "token_embd.weight";

    /// <summary>Output (LM head); optional — tied to <see cref="TokenEmbd"/> when absent.</summary>
    public const string Output = "output.weight";

    /// <summary>The n-gram hash table, <c>[row_dim, rows]</c>. Lives in its own shard; must never be imported to a device.</summary>
    public const string PerLayerTokenEmbd = "per_layer_token_embd.weight";

    /// <summary>Final head mixer gamma (replaces <c>output_norm</c>).</summary>
    public const string OutputHcNorm = "output_hc_norm.weight";

    /// <summary>Final head mixer down projection.</summary>
    public const string OutputHcDown = "output_hc_down.weight";

    /// <summary>Final head mixer up projection.</summary>
    public const string OutputHcUp = "output_hc_up.weight";

    /// <summary>Name of a per-block tensor: <c>blk.{il}.{suffix}</c>.</summary>
    public static string Block(int il, string suffix) => $"blk.{il}.{suffix}";

    /// <summary>Per-block name suffixes (relative to <c>blk.{il}.</c>).</summary>
    public static class Suffix
    {
        /// <summary>Hyper-connection module before the token mixer: gamma, down, up, inject.</summary>
        public const string HcAttnNorm = "hc_attn_norm.weight", HcAttnDown = "hc_attn_down.weight",
            HcAttnUp = "hc_attn_up.weight", HcAttnInject = "hc_attn_inject.weight";

        /// <summary>Hyper-connection module before the MoE: gamma, down, up, inject.</summary>
        public const string HcFfnNorm = "hc_ffn_norm.weight", HcFfnDown = "hc_ffn_down.weight",
            HcFfnUp = "hc_ffn_up.weight", HcFfnInject = "hc_ffn_inject.weight";

        /// <summary>Gated-DeltaNet (linear-attention) layer tensors.</summary>
        public const string AttnQkv = "attn_qkv.weight", AttnGate = "attn_gate.weight", SsmConv1d = "ssm_conv1d.weight",
            SsmDtBias = "ssm_dt.bias", SsmA = "ssm_a", SsmBeta = "ssm_beta.weight", SsmAlpha = "ssm_alpha.weight",
            SsmNorm = "ssm_norm.weight", SsmOut = "ssm_out.weight";

        /// <summary>QSA (full-attention) layer tensors; <c>attn_q</c> holds <c>[q | gate]</c> per head.</summary>
        public const string AttnQ = "attn_q.weight", AttnK = "attn_k.weight", AttnV = "attn_v.weight",
            AttnOutput = "attn_output.weight", AttnQNorm = "attn_q_norm.weight", AttnKNorm = "attn_k_norm.weight";

        /// <summary>QSA indexer tensors.</summary>
        public const string IndexerQProj = "indexer.q_proj.weight", IndexerKProj = "indexer.k_proj.weight",
            IndexerQNorm = "indexer.q_norm.weight", IndexerKNorm = "indexer.k_norm.weight";

        /// <summary>N-gram (PLE) module tensors, present on the PLE layer only.</summary>
        public const string PleKey = "ple_key.weight", PleValue = "ple_value.weight", PleNormKey = "ple_norm_key.weight",
            PleNormQuery = "ple_norm_query.weight", PleNormConv = "ple_norm_conv.weight", PleConv1d = "ple_conv1d.weight";

        /// <summary>MoE tensors (every block): router, split routed experts, shared expert and its sigmoid gate.</summary>
        public const string FfnGateInp = "ffn_gate_inp.weight", FfnGateExps = "ffn_gate_exps.weight",
            FfnUpExps = "ffn_up_exps.weight", FfnDownExps = "ffn_down_exps.weight", FfnGateShexp = "ffn_gate_shexp.weight",
            FfnUpShexp = "ffn_up_shexp.weight", FfnDownShexp = "ffn_down_shexp.weight", FfnGateInpShexp = "ffn_gate_inp_shexp.weight";

        /// <summary>MTP ("nextn") block tensors (<c>blk.{n_layer}.nextn.*</c>).</summary>
        public const string NextnEhProj = "nextn.eh_proj.weight", NextnEnorm = "nextn.enorm.weight", NextnHnorm = "nextn.hnorm.weight",
            NextnHcHeadNorm = "nextn.hc_head_norm.weight", NextnHcHeadDown = "nextn.hc_head_down.weight",
            NextnHcHeadUp = "nextn.hc_head_up.weight";
    }

    // ───────────────────────────── expected set ─────────────────────────────

    /// <summary>
    /// Enumerates every tensor the model needs.
    /// </summary>
    /// <param name="config">A <see cref="Architecture.Qwen4Exp"/> configuration.</param>
    /// <param name="includeTrunk">Include the trunk blocks, the final head mixer and the n-gram table (false for a standalone MTP file).</param>
    /// <param name="includeMtp">Include the trailing MTP block (<c>blk.{NumLayers}.*</c>); requires <see cref="ModelConfig.NextnPredictLayers"/> &gt; 0.</param>
    public static IEnumerable<Expected> Enumerate(ModelConfig config, bool includeTrunk = true, bool includeMtp = false)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Qwen4Exp || config.Qwen4Exp is not { } q4
            || config.GdnConfig is not { } gdn || config.Moe is not { } moe || config.HybridLayout is not { } layout)
            throw new ArgumentException("Configuration is not a fully-populated Qwen4Exp configuration.", nameof(config));
        if (includeMtp && config.NextnPredictLayers <= 0)
            throw new ArgumentException("includeMtp requires a configuration with nextn_predict_layers > 0.", nameof(config));

        long embd = config.HiddenSize;
        long hc = q4.HyperConnectionCount;
        long hcDim = hc * embd;
        long lr = q4.HyperConnectionLowRank;
        long vocab = config.VocabSize;

        yield return new Expected(TokenEmbd, [embd, vocab]);
        yield return new Expected(Output, [embd, vocab], Required: false);

        if (includeTrunk)
        {
            yield return new Expected(OutputHcNorm, [hcDim]);
            yield return new Expected(OutputHcDown, [hcDim, lr]);
            yield return new Expected(OutputHcUp, [lr, hcDim]);

            if (q4.Ple is { } ple)
                yield return new Expected(PerLayerTokenEmbd, [ple.RowDim, checked((long)ple.MinTableRows)], LastDimIsMinimum: true);

            for (int il = 0; il < config.NumLayers; il++)
            {
                bool attention = layout.LayerKind[il] == HybridLayerKind.Attention;
                bool hasPle = q4.Ple is { } p && p.Layers.Contains(il);
                foreach (var e in BlockTensors(config, q4, gdn, moe, il, attention, hasPle))
                    yield return e;
            }
        }

        if (includeMtp)
        {
            // The MTP block is a full-attention (QSA) block, never a GDN one; it has no PLE module.
            int il = config.NumLayers;
            foreach (var e in BlockTensors(config, q4, gdn, moe, il, attention: true, hasPle: false))
                yield return e;

            yield return new Expected(Block(il, Suffix.NextnEhProj), [2 * embd, embd]);
            yield return new Expected(Block(il, Suffix.NextnEnorm), [embd]);
            yield return new Expected(Block(il, Suffix.NextnHnorm), [hcDim]);
            yield return new Expected(Block(il, Suffix.NextnHcHeadNorm), [hcDim]);
            yield return new Expected(Block(il, Suffix.NextnHcHeadDown), [hcDim, lr]);
            yield return new Expected(Block(il, Suffix.NextnHcHeadUp), [lr, hcDim]);
        }
    }

    private static IEnumerable<Expected> BlockTensors(
        ModelConfig config, Qwen4ExpConfig q4, GatedDeltaNetConfig gdn, MoeConfig moe, int il, bool attention, bool hasPle)
    {
        long embd = config.HiddenSize;
        long hc = q4.HyperConnectionCount;
        long hcDim = hc * embd;
        long lr = q4.HyperConnectionLowRank;

        // Two hyper-connection modules per block: before the token mixer, before the MoE.
        yield return new Expected(Block(il, Suffix.HcAttnNorm), [hcDim]);
        yield return new Expected(Block(il, Suffix.HcAttnDown), [hcDim, lr]);
        yield return new Expected(Block(il, Suffix.HcAttnUp), [lr, hcDim]);
        yield return new Expected(Block(il, Suffix.HcAttnInject), [hcDim, hc]);
        yield return new Expected(Block(il, Suffix.HcFfnNorm), [hcDim]);
        yield return new Expected(Block(il, Suffix.HcFfnDown), [hcDim, lr]);
        yield return new Expected(Block(il, Suffix.HcFfnUp), [lr, hcDim]);
        yield return new Expected(Block(il, Suffix.HcFfnInject), [hcDim, hc]);

        if (attention)
        {
            long headDim = config.HeadDim;
            long qOut = headDim * config.NumAttentionHeads * 2; // [q | gate] per head
            long kvOut = headDim * config.NumKvHeads;
            long idxDim = q4.IndexerKeyLength;

            yield return new Expected(Block(il, Suffix.AttnQ), [embd, qOut]);
            yield return new Expected(Block(il, Suffix.AttnK), [embd, kvOut]);
            yield return new Expected(Block(il, Suffix.AttnV), [embd, kvOut]);
            yield return new Expected(Block(il, Suffix.AttnOutput), [headDim * config.NumAttentionHeads, embd]);
            yield return new Expected(Block(il, Suffix.AttnQNorm), [headDim]);
            yield return new Expected(Block(il, Suffix.AttnKNorm), [headDim]);
            yield return new Expected(Block(il, Suffix.IndexerQProj), [embd, q4.IndexerHeadCount * idxDim]);
            yield return new Expected(Block(il, Suffix.IndexerKProj), [embd, idxDim]);
            yield return new Expected(Block(il, Suffix.IndexerQNorm), [idxDim]);
            yield return new Expected(Block(il, Suffix.IndexerKNorm), [idxDim]);
        }
        else
        {
            long keyDim = (long)gdn.DState * gdn.NKHead;
            long valueDim = (long)gdn.DState * gdn.NVHead;
            long convDim = keyDim * 2 + valueDim;

            yield return new Expected(Block(il, Suffix.AttnQkv), [embd, convDim]);
            yield return new Expected(Block(il, Suffix.AttnGate), [embd, valueDim]);
            yield return new Expected(Block(il, Suffix.SsmConv1d), [gdn.DConv, convDim]);
            yield return new Expected(Block(il, Suffix.SsmDtBias), [gdn.NVHead]);
            yield return new Expected(Block(il, Suffix.SsmA), [gdn.NVHead]);
            yield return new Expected(Block(il, Suffix.SsmBeta), [embd, gdn.NVHead]);
            yield return new Expected(Block(il, Suffix.SsmAlpha), [embd, gdn.NVHead]);
            yield return new Expected(Block(il, Suffix.SsmNorm), [gdn.DState]);
            yield return new Expected(Block(il, Suffix.SsmOut), [valueDim, embd]);
        }

        if (hasPle)
        {
            yield return new Expected(Block(il, Suffix.PleKey), [embd, hcDim]);
            yield return new Expected(Block(il, Suffix.PleValue), [embd, embd]);
            yield return new Expected(Block(il, Suffix.PleNormKey), [hcDim]);
            yield return new Expected(Block(il, Suffix.PleNormQuery), [hcDim]);
            yield return new Expected(Block(il, Suffix.PleNormConv), [hcDim]);
            yield return new Expected(Block(il, Suffix.PleConv1d), [q4.Ple!.ConvKernel, hcDim]);
        }

        long ffExp = moe.MoeIntermediateSize;
        long ffShared = moe.SharedExpertIntermediateSize ?? ffExp;
        long experts = moe.NumExperts;
        yield return new Expected(Block(il, Suffix.FfnGateInp), [embd, experts]);
        yield return new Expected(Block(il, Suffix.FfnDownExps), [ffExp, embd, experts]);
        yield return new Expected(Block(il, Suffix.FfnGateExps), [embd, ffExp, experts]);
        yield return new Expected(Block(il, Suffix.FfnUpExps), [embd, ffExp, experts]);
        yield return new Expected(Block(il, Suffix.FfnGateInpShexp), [embd]);
        yield return new Expected(Block(il, Suffix.FfnGateShexp), [embd, ffShared]);
        yield return new Expected(Block(il, Suffix.FfnUpShexp), [embd, ffShared]);
        yield return new Expected(Block(il, Suffix.FfnDownShexp), [ffShared, embd]);
    }

    // ───────────────────────────── verification ─────────────────────────────

    /// <summary>
    /// Compares a tensor table against the contract. Returns one human-readable problem per missing required tensor, per
    /// shape mismatch, and per tensor the model does not know — an empty list means the table is exactly the Qwen4Exp layout.
    /// </summary>
    /// <param name="tensors">Unified tensor table (<see cref="GgufFile.TensorsByName"/>, or any name → descriptor map).</param>
    /// <param name="config">A Qwen4Exp configuration extracted from the same file.</param>
    /// <param name="includeTrunk">See <see cref="Enumerate"/>.</param>
    /// <param name="includeMtp">See <see cref="Enumerate"/>.</param>
    public static IReadOnlyList<string> FindProblems(
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config, bool includeTrunk = true, bool includeMtp = false)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var problems = new List<string>();
        var known = new HashSet<string>(StringComparer.Ordinal);

        foreach (Expected e in Enumerate(config, includeTrunk, includeMtp))
        {
            known.Add(e.Name);
            if (!tensors.TryGetValue(e.Name, out GgufTensorDescriptor actual))
            {
                if (e.Required)
                    problems.Add($"missing tensor '{e.Name}'");
                continue;
            }

            int[] dims = actual.Shape.Dimensions.ToArray();
            bool ok = dims.Length == e.Dims.Length;
            for (int d = 0; ok && d < dims.Length; d++)
            {
                bool minimum = e.LastDimIsMinimum && d == dims.Length - 1;
                ok = minimum ? dims[d] >= e.Dims[d] : dims[d] == e.Dims[d];
            }
            if (!ok)
                problems.Add($"tensor '{e.Name}' has shape [{string.Join(", ", dims)}], expected [{string.Join(", ", e.Dims)}]" +
                             (e.LastDimIsMinimum ? " (last dim is a minimum)" : ""));
        }

        foreach (string name in tensors.Keys)
            if (!known.Contains(name))
                problems.Add($"unexpected tensor '{name}'");

        return problems;
    }
}
