using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// GGUF tensor contract of <c>inkling2</c> / <c>inkling2_loop</c> (llama.cpp <c>src/models/inkling2.cpp</c> <c>load_arch_tensors</c> and
/// <c>inkling2-loop.cpp</c>; verified against the <c>Inkling-2-Mini-NVFP4.gguf</c> header). Shapes are in GGUF order (fastest dimension first);
/// trailing 1-dimensions are dropped by GGUF, so a <c>{n, 1}</c> tensor appears as <c>{n}</c>.
/// </summary>
internal static class Inkling2Tensors
{
    public const string TokenEmbd = "token_embd.weight";
    public const string TokenEmbdNorm = "token_embd_norm.weight";
    public const string Output = "output.weight";
    public const string OutputNorm = "output_norm.weight";
    public const string OutputResQuery = "output_res_query.weight";

    /// <summary>Loop-controller tensor stems, in controller order Q, K, V, FFN.</summary>
    public static readonly string[] CtrlKinds = ["q", "k", "v", "ffn"];

    public static string Block(int il, string suffix) => $"blk.{il}.{suffix}";

    /// <summary>
    /// Validates the tensor table against <paramref name="config"/>: every required tensor present with the expected shape and nothing unknown
    /// (an unknown tensor means a checkpoint variant that would silently run as a different model). Returns the set of NVFP4 <c>.scale</c>
    /// tensors the loader applies (the per-expert scales of the two expert banks) so the caller can pass it to
    /// <see cref="Nvfp4Support.RejectUnappliedScales"/>.
    /// </summary>
    public static List<string> FindProblems(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, ModelConfig config, out HashSet<string> consumedScales)
    {
        var c = config.Inkling2 ?? throw new ArgumentException("inkling2 config missing", nameof(config));
        var problems = new List<string>();
        var known = new HashSet<string>();
        consumedScales = new HashSet<string>();

        int H = config.HiddenSize, nh = config.NumAttentionHeads, nkv = config.NumKvHeads, hd = config.HeadDim;
        int K = c.ConvKernel, rel = c.RelativeLength, E = c.ExpertCount, ff = c.ExpertFfLength;

        void Expect(string name, params long[] dims)
        {
            known.Add(name);
            if (!tensors.TryGetValue(name, out var d)) { problems.Add($"missing tensor {name}"); return; }
            int want = dims.Length;
            while (want > 1 && dims[want - 1] == 1) want--;
            if (d.Shape.Rank != want) { problems.Add($"{name}: rank {d.Shape.Rank}, expected {want}"); return; }
            for (int i = 0; i < want; i++)
                if (d.Shape[i] != dims[i]) { problems.Add($"{name}: dim {i} is {d.Shape[i]}, expected {dims[i]}"); return; }
        }

        void ExpectF32(string name, params long[] dims)
        {
            Expect(name, dims);
            if (tensors.TryGetValue(name, out var d) && d.QuantizationType != QuantizationType.F32)
                problems.Add($"{name} is {d.QuantizationType}; the contract keeps it F32");
        }

        Expect(TokenEmbd, H, c.PaddedVocabSize);
        Expect(Output, H, c.PaddedVocabSize);
        ExpectF32(TokenEmbdNorm, H);
        ExpectF32(OutputNorm, H);
        ExpectF32(OutputResQuery, H);

        for (int il = 0; il < config.NumLayers; il++)
        {
            string B(string s) => Block(il, s);
            bool local = c.IsLocalLayer[il];
            int extent = local ? c.SlidingWindow : c.RelativeExtent;

            Expect(B("attn_q.weight"), H, nh * hd);
            Expect(B("attn_k.weight"), H, nkv * hd);
            Expect(B("attn_v.weight"), H, nkv * hd);
            Expect(B("attn_output.weight"), nh * hd, H);
            Expect(B("attn_gate.weight"), H, nh * hd);
            Expect(B("attn_sink_proj.weight"), H, nh);
            Expect(B("attn_rel.weight"), H, rel * nh);
            ExpectF32(B("attn_q_clip.weight"), hd, nh);
            ExpectF32(B("attn_rel_bias.weight"), rel, extent);
            ExpectF32(B("attn_rel_clip.weight"), rel, nh);
            ExpectF32(B("attn_k_conv.weight"), K, nkv * hd);
            ExpectF32(B("attn_v_conv.weight"), K, nkv * hd);
            ExpectF32(B("attn_post_conv.weight"), K, H);
            ExpectF32(B("ffn_post_conv.weight"), K, H);
            ExpectF32(B("ffn_res_query.weight"), H);
            ExpectF32(B("ffn_norm.weight"), H);
            Expect(B("ffn_gate_inp.weight"), H, E);
            if (tensors.TryGetValue(B("ffn_gate_inp.weight"), out var router) && router.QuantizationType != QuantizationType.F32)
                problems.Add($"{B("ffn_gate_inp.weight")} is {router.QuantizationType}; the router is read as F32");
            ExpectF32(B("exp_probs_b.bias"), E);
            Expect(B("ffn_gate_up_exps.weight"), H, 2 * ff, E);
            Expect(B("ffn_down_exps.weight"), ff, H, E);
            foreach (var bank in new[] { "ffn_gate_up_exps", "ffn_down_exps" })
            {
                string w = B(bank + ".weight"), s = B(bank + ".scale");
                if (tensors.TryGetValue(w, out var wd) && wd.QuantizationType is not (QuantizationType.NVFP4 or QuantizationType.Q8_0))
                    problems.Add($"{w} is {wd.QuantizationType}; only NVFP4 and Q8_0 expert banks are implemented");
                if (tensors.ContainsKey(s))
                {
                    ExpectF32(s, E);
                    if (tensors.TryGetValue(w, out wd) && wd.QuantizationType == QuantizationType.NVFP4) consumedScales.Add(s);
                    else problems.Add($"{s}: a per-expert scale next to a non-NVFP4 bank is not implemented");
                }
            }

            if (local) ExpectF32(B("attn_norm.weight"), H);
            else
            {
                ExpectF32(B("attn_q_inp_norm.weight"), H);
                ExpectF32(B("attn_k_inp_norm.weight"), H);
                ExpectF32(B("attn_v_inp_norm.weight"), H);
                Expect(B("attn_k_conv_proj.weight"), H, K * nkv);
                Expect(B("attn_v_conv_proj.weight"), H, K * nkv);
            }

            if (il > 0) ExpectF32(B("attn_res_query.weight"), H, local ? 1 : 3);

            if (c.LoopPasses > 1 && il >= c.LoopStart && il < c.LoopEnd)
            {
                Expect(B("loop_msg_query.weight"), H, 4);
                foreach (var kind in CtrlKinds)
                {
                    Expect(B($"loop_{kind}_ctrl_down.weight"), 2 * H, c.LoopGateRank);
                    Expect(B($"loop_{kind}_ctrl_gain.weight"), c.LoopGateRank, H);
                    Expect(B($"loop_{kind}_ctrl_forget.weight"), c.LoopGateRank, H);
                }
            }
        }

        foreach (var name in tensors.Keys)
            if (!known.Contains(name) && !consumedScales.Contains(name))
                problems.Add($"unexpected tensor {name}");
        return problems;
    }
}
