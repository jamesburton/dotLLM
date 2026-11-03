using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Outcome of <see cref="Glm4MoeTensorCensus.Check(ModelConfig, IEnumerable{ValueTuple{string, int[]}})"/>: the tensors a
/// <c>glm4moe</c> GGUF is expected to carry (derived from its <see cref="ModelConfig"/>) compared with the tensors it actually carries.
/// </summary>
public sealed class Glm4MoeCensusResult
{
    /// <summary>Expected tensors that are absent from the file.</summary>
    public required IReadOnlyList<string> Missing { get; init; }

    /// <summary>Tensors present in the file that the glm4moe layout does not define.</summary>
    public required IReadOnlyList<string> Unexpected { get; init; }

    /// <summary>Tensors present with a shape other than the one the config implies (human-readable, <c>name: expected [..] got [..]</c>).</summary>
    public required IReadOnlyList<string> ShapeMismatches { get; init; }

    /// <summary>Total number of tensors in the file.</summary>
    public required int TensorCount { get; init; }

    /// <summary>Whether per-head Q/K RMS-norm tensors (<c>attn_q_norm</c> / <c>attn_k_norm</c>) were found (GLM-4.5 355B: yes, Air: no).</summary>
    public required bool HasQkNorm { get; init; }

    /// <summary>
    /// Whether the NextN/MTP block(s) at raw index <see cref="ModelConfig.NumLayers"/>.. are present in this file. <see langword="false"/> for
    /// trunk-only conversions (e.g. ggml-org ships the head as a separate <c>mtp-*.gguf</c>), which is NOT a mismatch.
    /// </summary>
    public required bool NextnPresent { get; init; }

    /// <summary><see langword="true"/> when nothing is missing, unexpected or mis-shaped.</summary>
    public bool IsClean => Missing.Count == 0 && Unexpected.Count == 0 && ShapeMismatches.Count == 0;

    /// <summary>One-line summary for diagnostics.</summary>
    public override string ToString()
        => $"tensors={TensorCount} missing={Missing.Count} unexpected={Unexpected.Count} shapeMismatches={ShapeMismatches.Count} qkNorm={HasQkNorm} nextn={NextnPresent}";
}

/// <summary>
/// Read-only tensor census for <c>glm4moe</c> GGUFs (issue #1598). Derives the expected tensor names and GGUF (ne-order) shapes from a
/// <see cref="ModelConfig"/> per llama.cpp's <c>glm4-moe.cpp</c> <c>load_arch_tensors</c> and checks a file against them, so a wrong
/// or partial conversion fails loudly before any forward pass exists. Names verified against the real
/// <c>bartowski/cerebras_GLM-4.5-Air-REAP-82B-A12B-GGUF</c> IQ4_XS header (803 tensors, 47 blocks).
/// </summary>
public static class Glm4MoeTensorCensus
{
    /// <summary>
    /// Builds the expected (required) and optional tensor sets for <paramref name="config"/>.
    /// </summary>
    /// <param name="config">A <see cref="Architecture.Glm4Moe"/> config with <see cref="ModelConfig.Moe"/> set.</param>
    /// <param name="qkNorm">Whether per-head Q/K norm tensors are required (<see langword="true"/>), forbidden (<see langword="false"/>).</param>
    /// <param name="includeNextn">Whether the NextN block(s) are required.</param>
    /// <returns>Required tensors and optional tensors, name to GGUF dims.</returns>
    public static (Dictionary<string, int[]> Required, Dictionary<string, int[]> Optional) Expected(
        ModelConfig config, bool qkNorm, bool includeNextn)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.Glm4Moe || config.Moe is not { } moe)
            throw new ArgumentException("The census needs an Architecture.Glm4Moe config with MoE parameters.", nameof(config));

        int h = config.HiddenSize, v = config.VocabSize, hd = config.HeadDim;
        int nq = config.NumAttentionHeads * hd, nkv = config.NumKvHeads * hd;
        int e = moe.NumExperts, mi = moe.MoeIntermediateSize, ff = config.IntermediateSize;
        int sh = moe.SharedExpertIntermediateSize ?? 0;
        int denseLead = moe.MlpOnlyLayers?.Count ?? 0;
        int trunk = config.NumLayers;

        var req = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["token_embd.weight"] = [h, v],
            ["output_norm.weight"] = [h],
        };
        var opt = new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["output.weight"] = [h, v],   // llama.cpp: TENSOR_NOT_REQUIRED, falls back to token_embd (tied)
        };

        void Layer(int l, bool nextn)
        {
            string p = $"blk.{l}";
            req[$"{p}.attn_norm.weight"] = [h];
            req[$"{p}.attn_q.weight"] = [h, nq];
            req[$"{p}.attn_q.bias"] = [nq];
            req[$"{p}.attn_k.weight"] = [h, nkv];
            req[$"{p}.attn_k.bias"] = [nkv];
            req[$"{p}.attn_v.weight"] = [h, nkv];
            req[$"{p}.attn_v.bias"] = [nkv];
            req[$"{p}.attn_output.weight"] = [nq, h];                // no attn_output.bias
            req[$"{p}.post_attention_norm.weight"] = [h];            // the PRE-FFN norm (HF post_attention_layernorm)
            if (qkNorm)
            {
                req[$"{p}.attn_q_norm.weight"] = [hd];
                req[$"{p}.attn_k_norm.weight"] = [hd];
            }
            if (!nextn && l < denseLead)
            {
                req[$"{p}.ffn_gate.weight"] = [h, ff];
                req[$"{p}.ffn_up.weight"] = [h, ff];
                req[$"{p}.ffn_down.weight"] = [ff, h];
            }
            else
            {
                req[$"{p}.ffn_gate_inp.weight"] = [h, e];
                req[$"{p}.exp_probs_b.bias"] = [e];
                req[$"{p}.ffn_gate_exps.weight"] = [h, mi, e];
                req[$"{p}.ffn_up_exps.weight"] = [h, mi, e];
                req[$"{p}.ffn_down_exps.weight"] = [mi, h, e];
                if (sh > 0)
                {
                    req[$"{p}.ffn_gate_shexp.weight"] = [h, sh];
                    req[$"{p}.ffn_up_shexp.weight"] = [h, sh];
                    req[$"{p}.ffn_down_shexp.weight"] = [sh, h];
                }
            }
            if (nextn)
            {
                req[$"{p}.nextn.eh_proj.weight"] = [2 * h, h];
                req[$"{p}.nextn.enorm.weight"] = [h];
                req[$"{p}.nextn.hnorm.weight"] = [h];
                opt[$"{p}.nextn.embed_tokens.weight"] = [h, v];       // optional: llama.cpp falls back to token_embd
                opt[$"{p}.nextn.shared_head_head.weight"] = [h, v];   // optional: falls back to output
                opt[$"{p}.nextn.shared_head_norm.weight"] = [h];      // optional: falls back to output_norm
            }
        }

        for (int l = 0; l < trunk; l++) Layer(l, nextn: false);
        if (includeNextn)
            for (int l = trunk; l < trunk + config.NextnPredictLayers; l++) Layer(l, nextn: true);
        return (req, opt);
    }

    /// <summary>Runs the census against an opened GGUF file.</summary>
    /// <param name="config">The config extracted from the same file.</param>
    /// <param name="gguf">The opened file.</param>
    /// <returns>The comparison result.</returns>
    public static Glm4MoeCensusResult Check(ModelConfig config, GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        return Check(config, gguf.Tensors.Select(t => (t.Name, t.Shape.Dimensions)));
    }

    /// <summary>
    /// Runs the census against an arbitrary tensor listing (name, GGUF ne-order dims), e.g. a header-only range read.
    /// Q/K norm is detected from the presence of <c>blk.0.attn_q_norm.weight</c> unless <see cref="ModelConfig.UseQkNorm"/> declares it.
    /// The NextN block is "present" when any tensor of block <see cref="ModelConfig.NumLayers"/> exists; a partially present block is reported as missing tensors.
    /// </summary>
    /// <param name="config">A glm4moe config.</param>
    /// <param name="actual">Actual tensors.</param>
    /// <returns>The comparison result.</returns>
    public static Glm4MoeCensusResult Check(ModelConfig config, IEnumerable<(string Name, int[] Dims)> actual)
    {
        ArgumentNullException.ThrowIfNull(actual);
        var have = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var (n, d) in actual) have[n] = d;

        bool qk = config.UseQkNorm ?? have.ContainsKey("blk.0.attn_q_norm.weight");
        string nextnPrefix = $"blk.{config.NumLayers}.";
        bool nextnPresent = config.NextnPredictLayers > 0 && have.Keys.Any(k => k.StartsWith(nextnPrefix, StringComparison.Ordinal));

        var (req, opt) = Expected(config, qk, includeNextn: nextnPresent);
        var missing = new List<string>();
        var mismatches = new List<string>();
        foreach (var (name, dims) in req)
        {
            if (!have.TryGetValue(name, out var got)) { missing.Add(name); continue; }
            if (!got.AsSpan().SequenceEqual(dims))
                mismatches.Add($"{name}: expected [{string.Join(", ", dims)}] got [{string.Join(", ", got)}]");
        }
        foreach (var (name, dims) in opt)
            if (have.TryGetValue(name, out var got) && !got.AsSpan().SequenceEqual(dims))
                mismatches.Add($"{name}: expected [{string.Join(", ", dims)}] got [{string.Join(", ", got)}]");

        var unexpected = have.Keys.Where(k => !req.ContainsKey(k) && !opt.ContainsKey(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        missing.Sort(StringComparer.Ordinal);
        mismatches.Sort(StringComparer.Ordinal);
        return new Glm4MoeCensusResult
        {
            Missing = missing,
            Unexpected = unexpected,
            ShapeMismatches = mismatches,
            TensorCount = have.Count,
            HasQkNorm = have.ContainsKey("blk.0.attn_q_norm.weight"),
            NextnPresent = nextnPresent,
        };
    }
}
