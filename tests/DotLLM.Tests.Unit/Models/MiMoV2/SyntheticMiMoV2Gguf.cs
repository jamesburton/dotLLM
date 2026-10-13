using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.MiMoV2;

/// <summary>
/// Deterministic random-weight <c>mimo2</c> GGUF writer for hermetic tests (no fixture files). The geometry mirrors the
/// HF/llama.cpp oracle model of <c>tools/mimo2-tiny-oracle/make_tiny.py</c> with a 40-token vocabulary so the expected
/// llama.cpp logits fit in the test source: every ratio is distinct (K 48 vs V 32, rotary 16 of 48, GQA 4 on global layers vs 2 on
/// sliding layers, window 4 over a longer prompt, 2 global + 4 sliding layers, dense layer 0, 3 trailing NextN blocks that the
/// trunk must ignore).
/// </summary>
internal static class SyntheticMiMoV2Gguf
{
    public const int Hidden = 64, Layers = 6, NextN = 3, Heads = 8, GlobalKv = 2, SlidingKv = 4, KDim = 48, VDim = 32, RopeDim = 16;
    public const int Window = 4, Experts = 8, TopK = 2, MoeFf = 32, DenseFf = 96, Vocab = 40, Context = 256;
    public static readonly int[] Pattern = [0, 1, 1, 1, 0, 1];   // 1 = sliding-window layer

    public sealed record Options
    {
        public int Seed { get; init; } = 20260713;
        /// <summary>Scale applied to attn_sinks (0 draws a constant very negative sink = "no sink").</summary>
        public bool Sinks { get; init; } = true;
        public bool WriteTokenizer { get; init; } = true;
    }

    public static byte[] Build(Options? options = null)
    {
        options ??= new Options();
        var rng = new Random(options.Seed);
        var w = new GgufWriter();
        const string arch = "mimo2";
        int blocks = Layers + NextN;
        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", Context);
        w.AddUInt32($"{arch}.embedding_length", Hidden);
        w.AddUInt32($"{arch}.block_count", (uint)blocks);
        w.AddUInt32($"{arch}.feed_forward_length", DenseFf);
        w.AddUInt32($"{arch}.attention.head_count", Heads);
        w.AddInt32Array($"{arch}.attention.head_count_kv", Enumerable.Range(0, blocks).Select(i => IsSliding(i) ? SlidingKv : GlobalKv).ToArray());
        w.AddFloat32($"{arch}.rope.freq_base", 1e7f);
        w.AddFloat32($"{arch}.rope.freq_base_swa", 1e4f);
        w.AddUInt32($"{arch}.expert_used_count", TopK);
        w.AddUInt32($"{arch}.expert_group_count", 1);
        w.AddUInt32($"{arch}.expert_group_used_count", 1);
        w.AddUInt32($"{arch}.expert_gating_func", 2);
        w.AddUInt32($"{arch}.attention.key_length", KDim);
        w.AddUInt32($"{arch}.attention.value_length", VDim);
        w.AddUInt32($"{arch}.attention.sliding_window", Window);
        w.AddInt32Array($"{arch}.attention.sliding_window_pattern", Enumerable.Range(0, blocks).Select(i => IsSliding(i) ? 1 : 0).ToArray());
        w.AddUInt32($"{arch}.expert_count", Experts);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeFf);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);
        w.AddFloat32($"{arch}.attention.value_scale", 0.707f);
        w.AddUInt32($"{arch}.nextn_predict_layers", NextN);

        if (options.WriteTokenizer)
        {
            var tokens = new string[Vocab];
            for (int i = 0; i < Vocab; i++) tokens[i] = ((char)('a' + i % 26)).ToString() + (i >= 26 ? ((char)('a' + (i / 26) % 26)).ToString() : "");
            w.AddString("tokenizer.ggml.model", "gpt2");
            w.AddString("tokenizer.ggml.pre", "qwen2");
            w.AddStringArray("tokenizer.ggml.tokens", tokens);
            w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, Vocab).ToArray());
            w.AddStringArray("tokenizer.ggml.merges", []);
        }

        Matrix(w, rng, "token_embd.weight", Hidden, Vocab, 0.6f);
        Norm(w, rng, "output_norm.weight", Hidden);
        Matrix(w, rng, "output.weight", Hidden, Vocab, 0.25f);

        for (int i = 0; i < blocks; i++)
        {
            string p = $"blk.{i}";
            bool sliding = IsSliding(i);
            bool trunk = i < Layers;
            int kv = sliding ? SlidingKv : GlobalKv;
            Norm(w, rng, $"{p}.attn_norm.weight", Hidden);
            Matrix(w, rng, $"{p}.attn_qkv.weight", Hidden, Heads * KDim + kv * KDim + kv * VDim, 1.4f / MathF.Sqrt(Hidden));
            Matrix(w, rng, $"{p}.attn_output.weight", Heads * VDim, Hidden, 1.4f / MathF.Sqrt(Heads * VDim));
            if (sliding)
            {
                var sinks = new float[Heads];
                for (int h = 0; h < Heads; h++) sinks[h] = options.Sinks ? (float)(Gauss(rng) * 1.5) : -30f;
                w.AddTensor($"{p}.attn_sinks.weight", [Heads], (uint)QuantizationType.F32, Bytes(sinks));
            }
            Norm(w, rng, $"{p}.ffn_norm.weight", Hidden);
            bool dense = i == 0 || !trunk;
            if (dense)
            {
                Matrix(w, rng, $"{p}.ffn_gate.weight", Hidden, DenseFf, 1.4f / MathF.Sqrt(Hidden));
                Matrix(w, rng, $"{p}.ffn_up.weight", Hidden, DenseFf, 1.4f / MathF.Sqrt(Hidden));
                Matrix(w, rng, $"{p}.ffn_down.weight", DenseFf, Hidden, 1.4f / MathF.Sqrt(DenseFf));
            }
            else
            {
                Matrix(w, rng, $"{p}.ffn_gate_inp.weight", Hidden, Experts, 0.6f);
                Bank(w, rng, $"{p}.ffn_gate_exps.weight", Hidden, MoeFf, 1.4f / MathF.Sqrt(Hidden));
                Bank(w, rng, $"{p}.ffn_up_exps.weight", Hidden, MoeFf, 1.4f / MathF.Sqrt(Hidden));
                Bank(w, rng, $"{p}.ffn_down_exps.weight", MoeFf, Hidden, 1.4f / MathF.Sqrt(MoeFf));
                var bias = new float[Experts];
                for (int e = 0; e < Experts; e++) bias[e] = (float)(Gauss(rng) * 0.15);
                w.AddTensor($"{p}.exp_probs_b.bias", [Experts], (uint)QuantizationType.F32, Bytes(bias));
            }
            if (!trunk)
            {
                Matrix(w, rng, $"{p}.nextn.eh_proj.weight", 2 * Hidden, Hidden, 1.4f / MathF.Sqrt(2 * Hidden));
                Norm(w, rng, $"{p}.nextn.enorm.weight", Hidden);
                Norm(w, rng, $"{p}.nextn.hnorm.weight", Hidden);
                Norm(w, rng, $"{p}.layer_output_norm.weight", Hidden);
            }
        }
        return w.Build();
    }

    internal static bool IsSliding(int block) => block >= Pattern.Length || Pattern[block] == 1;

    private static double Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static byte[] Bytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static void Matrix(GgufWriter w, Random rng, string name, int inK, int outM, float std)
    {
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, Bytes(f));
    }

    private static void Bank(GgufWriter w, Random rng, string name, int inK, int outM, float std)
    {
        var f = new float[(long)inK * outM * Experts];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM, Experts], (uint)QuantizationType.F32, Bytes(f));
    }

    private static void Norm(GgufWriter w, Random rng, string name, int n)
    {
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + 0.2f * (float)Gauss(rng);
        w.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(f));
    }
}
