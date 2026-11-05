using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

/// <summary>Outcome of <see cref="RicursiveTensorCensus.Check(ModelConfig, IEnumerable{ValueTuple{string, int[]}})"/>.</summary>
public sealed class RicursiveCensusResult
{
    /// <summary>Expected tensors that are absent.</summary>
    public required IReadOnlyList<string> Missing { get; init; }

    /// <summary>Tensors present that the ricursive layout does not define (including a second value-residual spelling).</summary>
    public required IReadOnlyList<string> Unexpected { get; init; }

    /// <summary>Tensors whose shape differs from the config (<c>name: expected [..] got [..]</c>).</summary>
    public required IReadOnlyList<string> ShapeMismatches { get; init; }

    /// <summary>Total tensors in the listing.</summary>
    public required int TensorCount { get; init; }

    /// <summary>
    /// The value-residual tensor stem the file uses (<c>attn_vres_lambda</c> / <c>attn_value_res</c> for GGUF,
    /// <c>self_attn.value_res_lambda</c> for HF), or <see langword="null"/> when the file has no value residual.
    /// </summary>
    public required string? ValueResidualName { get; init; }

    /// <summary>Whether the per-layer sigmoid output gate (<c>attn_gate</c> / <c>g_proj</c>) is present.</summary>
    public required bool HasGate { get; init; }

    /// <summary>Whether an untied LM head (<c>output.weight</c> / <c>lm_head.weight</c>) is present.</summary>
    public required bool HasOutputHead { get; init; }

    /// <summary><see langword="true"/> when nothing is missing, unexpected or mis-shaped.</summary>
    public bool IsClean => Missing.Count == 0 && Unexpected.Count == 0 && ShapeMismatches.Count == 0;

    /// <summary>Throws an <see cref="InvalidDataException"/> describing the first problems when the census is not clean.</summary>
    /// <param name="source">What was checked (shown in the message).</param>
    public void ThrowIfNotClean(string source)
    {
        if (IsClean) return;
        static string Head(IReadOnlyList<string> l) => l.Count == 0 ? "-" : string.Join(", ", l.Take(5)) + (l.Count > 5 ? $" (+{l.Count - 5} more)" : "");
        throw new InvalidDataException(
            $"{source} is not a valid ricursive (Ripple) checkpoint: missing [{Head(Missing)}], unexpected [{Head(Unexpected)}], "
            + $"mis-shaped [{Head(ShapeMismatches)}]. Accepted value-residual names: blk.N.attn_vres_lambda.weight (llama.cpp PR #31692 / AtomicChat) "
            + "or blk.N.attn_value_res.weight (prometheusAIR), on every layer or none.");
    }

    /// <summary>One-line summary.</summary>
    public override string ToString()
        => $"tensors={TensorCount} missing={Missing.Count} unexpected={Unexpected.Count} shapeMismatches={ShapeMismatches.Count} "
         + $"vres={ValueResidualName ?? "(none)"} gate={HasGate} outputHead={HasOutputHead}";
}

/// <summary>
/// Read-only tensor census for <c>ricursive</c> (Ripple) checkpoints (issue #1643). Derives the expected names and shapes from a
/// <see cref="ModelConfig"/> and compares a tensor listing against them, so an unknown or partial conversion fails loudly before any
/// forward pass exists. GGUF names and ne-order shapes follow llama.cpp's <c>ricursive.cpp</c> <c>load_arch_tensors</c> (both
/// community ports and PR #31692) and were checked against the real AtomicChat Q4_K_M and prometheusAIR Q6_K tensor tables
/// (1203 tensors = 3 + 80 x 15); HF names follow <c>model.safetensors.index.json</c> of <c>ricursiveAI/Ripple-65B-Preview</c>.
/// </summary>
/// <remarks>
/// Optional groups follow llama.cpp's rules: the gate and the value-residual logit are present on every layer or on none (llama.cpp
/// PR #31692 rejects a mix), and <c>output.weight</c> may be absent (tied fallback). When the config declares a switch
/// (HF <see cref="RicursiveConfig.UseGatedAttention"/> / <see cref="RicursiveConfig.UseValueResidual"/>) the group is required or
/// forbidden; when it is <see langword="null"/> (GGUF) layer 0 decides. The value-residual logit has two GGUF spellings; exactly one may
/// be used, chosen by layer 0, and the other spelling anywhere is reported as unexpected.
/// </remarks>
public static class RicursiveTensorCensus
{
    /// <summary>Runs the GGUF census against an opened file.</summary>
    /// <param name="config">The config extracted from the same file.</param>
    /// <param name="gguf">The opened file.</param>
    /// <returns>The comparison.</returns>
    public static RicursiveCensusResult Check(ModelConfig config, GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        return Check(config, gguf.Tensors.Select(t => (t.Name, t.Shape.Dimensions)));
    }

    /// <summary>Runs the GGUF census against a tensor listing (name, GGUF ne-order dims), e.g. a header-only read.</summary>
    /// <param name="config">A ricursive config.</param>
    /// <param name="actual">The tensors.</param>
    /// <returns>The comparison.</returns>
    public static RicursiveCensusResult Check(ModelConfig config, IEnumerable<(string Name, int[] Dims)> actual)
        => CheckCore(config, actual, hf: false);

    /// <summary>
    /// Runs the census against HF safetensors names (<c>model.layers.N.self_attn.q_proj.weight</c>, ...) with torch-order shapes
    /// (<c>[out, in]</c>; the value-residual logit is <c>[1]</c> and has no <c>.weight</c> suffix).
    /// </summary>
    /// <param name="config">A ricursive config.</param>
    /// <param name="actual">The tensors.</param>
    /// <returns>The comparison.</returns>
    public static RicursiveCensusResult CheckHf(ModelConfig config, IEnumerable<(string Name, int[] Dims)> actual)
        => CheckCore(config, actual, hf: true);

    private static RicursiveCensusResult CheckCore(ModelConfig config, IEnumerable<(string Name, int[] Dims)> actual, bool hf)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(actual);
        if (config.Architecture != Architecture.Ricursive || config.Ricursive is not { } rc)
            throw new ArgumentException("The census needs an Architecture.Ricursive config.", nameof(config));

        var have = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var (n, d) in actual) have[n] = d;

        int h = config.HiddenSize, v = config.VocabSize, hd = config.HeadDim, ff = config.IntermediateSize;
        int nq = config.NumAttentionHeads * hd, nkv = config.NumKvHeads * hd;
        // GGUF dims are ne-order ([in, out]); HF/torch is [out, in]. M(in, out) gives the right order for either.
        int[] M(int inDim, int outDim) => hf ? [outDim, inDim] : [inDim, outDim];
        string L(int l, string gguf, string hfName) => hf ? $"model.layers.{l}.{hfName}" : $"blk.{l}.{gguf}";

        var req = new Dictionary<string, int[]>(StringComparer.Ordinal);
        var opt = new Dictionary<string, int[]>(StringComparer.Ordinal);
        req[hf ? "model.embed_tokens.weight" : "token_embd.weight"] = M(h, v);
        req[hf ? "model.norm.weight" : "output_norm.weight"] = [h];
        string head = hf ? "lm_head.weight" : "output.weight";
        if (hf && !config.TiedEmbeddings) req[head] = M(h, v); else opt[head] = M(h, v);

        // Optional groups: config switch wins, else layer 0 decides.
        string gate0 = L(0, "attn_gate.weight", "self_attn.g_proj.weight");
        bool gate = rc.UseGatedAttention ?? have.ContainsKey(gate0);

        string[] vresStems = hf ? ["self_attn.value_res_lambda"] : [.. RicursiveConfig.ValueResidualTensorNames];
        string VresName(int l, string stem) => hf ? $"model.layers.{l}.{stem}" : $"blk.{l}.{stem}.weight";
        string? vres = vresStems.FirstOrDefault(s => have.ContainsKey(VresName(0, s)));
        bool useVres = rc.UseValueResidual ?? vres is not null;
        if (useVres) vres ??= vresStems[0];   // declared but absent on layer 0: require the canonical spelling
        else vres = null;

        for (int l = 0; l < config.NumLayers; l++)
        {
            req[L(l, "attn_norm.weight", "input_layernorm.weight")] = [h];
            req[L(l, "attn_q.weight", "self_attn.q_proj.weight")] = M(h, nq);
            req[L(l, "attn_k.weight", "self_attn.k_proj.weight")] = M(h, nkv);
            req[L(l, "attn_v.weight", "self_attn.v_proj.weight")] = M(h, nkv);
            req[L(l, "attn_output.weight", "self_attn.o_proj.weight")] = M(nq, h);
            req[L(l, "attn_q_norm.weight", "self_attn.q_norm.weight")] = [hd];
            req[L(l, "attn_k_norm.weight", "self_attn.k_norm.weight")] = [hd];
            if (gate) req[L(l, "attn_gate.weight", "self_attn.g_proj.weight")] = M(h, nq);
            if (vres is not null) req[VresName(l, vres)] = [1];
            req[L(l, "post_attention_norm.weight", "post_attention_layernorm.weight")] = [h];
            req[L(l, "ffn_norm.weight", "pre_feedforward_layernorm.weight")] = [h];
            req[L(l, "ffn_gate.weight", "mlp.gate_proj.weight")] = M(h, ff);
            req[L(l, "ffn_up.weight", "mlp.up_proj.weight")] = M(h, ff);
            req[L(l, "ffn_down.weight", "mlp.down_proj.weight")] = M(ff, h);
            req[L(l, "post_ffw_norm.weight", "post_feedforward_layernorm.weight")] = [h];
        }

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
        return new RicursiveCensusResult
        {
            Missing = missing,
            Unexpected = unexpected,
            ShapeMismatches = mismatches,
            TensorCount = have.Count,
            ValueResidualName = vres,
            HasGate = gate,
            HasOutputHead = have.ContainsKey(head),
        };
    }
}
