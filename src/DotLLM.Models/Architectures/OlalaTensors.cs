using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// GGUF tensor contract of the <c>olala</c> architecture (OVH llama.cpp fork <c>LLM_ARCH_OLALA</c>, verified against the
/// <c>Olala-v1.0-mini-7A1B</c> bf16 GGUF header). Shapes are listed in GGUF order (fastest dimension first).
/// </summary>
internal static class OlalaTensors
{
    public const string TokenEmbd = "token_embd.weight";
    public const string Output = "output.weight";

    /// <summary>Per-block names that every block carries.</summary>
    private static readonly string[] CommonBlock =
    [
        "geo_mixer_scale", "geo_mixer_bias", "geo_mlp_scale", "geo_mlp_bias",
        "attn_output.weight", "ffn_gate_inp.weight", "exp_probs_b", "ffn_latent_down.weight", "ffn_latent_up.weight",
        "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    ];

    private static readonly string[] SsmBlock =
    [
        "ssm_in.weight", "ssm_in_dyn.weight", "ssm_b_norm.weight", "ssm_c_norm.weight", "ssm_b_bias", "ssm_c_bias",
        "ssm_dt_bias", "ssm_d", "ssm_mimo_x", "ssm_mimo_z", "ssm_mimo_o",
    ];

    private static readonly string[] AttnBlock =
    [
        "attn_q.weight", "attn_wa_k.weight", "attn_wa_v.weight", "attn_wb_k.weight", "attn_wb_v.weight", "attn_shift_k.weight",
        "attn_shift_v.weight", "attn_lambda.weight", "attn_softmax_scaler", "attn_q_norm.weight", "attn_k_norm.weight", "attn_gate.weight",
    ];

    /// <summary>
    /// Optional per-block packed copies of the M-layer constants (<c>conversion/olala.py _M3_PACKS</c>, written for the GPU kernels of the fork):
    /// duplicates of tensors the loader already reads, ignored.
    /// </summary>
    private static readonly string[] IgnoredPacked = ["ssm_m3_bias", "ssm_m3_mxz", "ssm_m3_norms", "ssm_m3_misc"];

    public static string Block(int il, string suffix) => $"blk.{il}.{suffix}";

    /// <summary>
    /// Validates the tensor table against <paramref name="config"/>: every required tensor present with the expected shape, nothing
    /// unknown (an unknown tensor means a checkpoint variant — cosnet branches, value embeddings, a final norm — that would silently
    /// run as a different model).
    /// </summary>
    public static List<string> FindProblems(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config)
    {
        var o = config.Olala ?? throw new ArgumentException("olala config missing", nameof(config));
        var problems = new List<string>();
        int hidden = config.HiddenSize, vocab = config.VocabSize;
        int latent = o.MoeLatentSize > 0 ? o.MoeLatentSize : hidden;

        void Expect(string name, params long[] dims)
        {
            if (!tensors.TryGetValue(name, out var d)) { problems.Add($"missing tensor {name}"); return; }
            if (d.Shape.Rank != dims.Length) { problems.Add($"{name}: rank {d.Shape.Rank}, expected {dims.Length}"); return; }
            for (int i = 0; i < dims.Length; i++)
                if (d.Shape[i] != dims[i]) { problems.Add($"{name}: dim {i} is {d.Shape[i]}, expected {dims[i]}"); return; }
        }

        Expect(TokenEmbd, hidden, vocab);
        if (tensors.ContainsKey(Output)) Expect(Output, hidden, vocab);

        int H = o.NumSsmHeads, P = o.MambaHeadDim, N = o.StateSize, R = o.MimoRank;
        int sig = o.NumSignalHeads, noise = o.NumNoiseHeads, hd = config.HeadDim, rank = o.TpaRank;
        var known = new HashSet<string> { TokenEmbd, Output };
        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => Block(il, s);
            foreach (var s in CommonBlock) known.Add(B(s));
            Expect(B("geo_mixer_scale"), 1); Expect(B("geo_mixer_bias"), 1); Expect(B("geo_mlp_scale"), 1); Expect(B("geo_mlp_bias"), 1);
            Expect(B("ffn_gate_inp.weight"), hidden, o.ExpertCount);
            Expect(B("exp_probs_b"), o.ExpertCount);
            Expect(B("ffn_latent_down.weight"), hidden, latent);
            Expect(B("ffn_latent_up.weight"), latent, hidden);
            Expect(B("ffn_up_exps.weight"), latent, o.ExpertFfLength, o.ExpertCount);
            Expect(B("ffn_down_exps.weight"), o.ExpertFfLength, latent, o.ExpertCount);
            if (o.SharedExpertFfLength > 0)
            {
                Expect(B("ffn_up_shexp.weight"), hidden, o.SharedExpertFfLength);
                Expect(B("ffn_down_shexp.weight"), o.SharedExpertFfLength, hidden);
            }
            if (o.IsSsmLayer(il))
            {
                foreach (var s in SsmBlock) known.Add(B(s));
                foreach (var s in IgnoredPacked) known.Add(B(s));
                Expect(B("ssm_in.weight"), hidden, (long)H * o.SsmInPerHead);
                Expect(B("ssm_in_dyn.weight"), hidden, 2L * o.GroupCount * N * R + o.NumRopeAngles);
                Expect(B("ssm_b_norm.weight"), N); Expect(B("ssm_c_norm.weight"), N);
                Expect(B("ssm_b_bias"), N, R, H); Expect(B("ssm_c_bias"), N, R, H);
                Expect(B("ssm_dt_bias"), H); Expect(B("ssm_d"), H);
                Expect(B("ssm_mimo_x"), P, R, H); Expect(B("ssm_mimo_z"), P, R, H); Expect(B("ssm_mimo_o"), P, R, H);
                Expect(B("attn_output.weight"), (long)H * P, hidden);
            }
            else
            {
                foreach (var s in AttnBlock) known.Add(B(s));
                Expect(B("attn_q.weight"), hidden, (long)(sig + noise) * hd);
                Expect(B("attn_wa_k.weight"), hidden, (long)noise * rank); Expect(B("attn_wa_v.weight"), hidden, (long)noise * rank);
                Expect(B("attn_wb_k.weight"), hidden, (long)rank * hd); Expect(B("attn_wb_v.weight"), hidden, (long)rank * hd);
                Expect(B("attn_shift_k.weight"), hidden, noise); Expect(B("attn_shift_v.weight"), hidden, noise);
                Expect(B("attn_lambda.weight"), hidden, noise);
                Expect(B("attn_softmax_scaler"), sig + noise);
                Expect(B("attn_q_norm.weight"), hd); Expect(B("attn_k_norm.weight"), hd);
                Expect(B("attn_gate.weight"), hidden, (long)sig * hd);
                Expect(B("attn_output.weight"), (long)sig * hd, hidden);
            }
        }
        foreach (var name in tensors.Keys)
            if (!known.Contains(name)) problems.Add($"unexpected tensor {name} (checkpoint variant not implemented)");
        return problems;
    }
}
