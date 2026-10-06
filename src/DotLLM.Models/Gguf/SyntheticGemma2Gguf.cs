using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Shape knobs for <see cref="SyntheticGemma2Gguf"/>. Defaults are chosen so every Gemma-2
/// feature is <i>exercised</i>, not merely present: GQA (4 heads / 2 KV heads, not the MQA
/// degenerate case), a sliding window much shorter than the test sequence, soft-caps small
/// enough that the tanh actually bites, and a layer count / hidden / head_dim combination
/// where <c>query_pre_attn_scalar</c> differs from <c>head_dim</c>.
/// </summary>
public sealed record SyntheticGemma2Config
{
    /// <summary><c>gemma2</c> (four-norm, soft-caps, SWA), <c>gemma3</c> (four-norm, QK-norm, dual RoPE, 5:1 SWA) or <c>gemma</c> (Gemma 1 / CodeGemma two-norm).</summary>
    public string Arch { get; init; } = "gemma2";

    /// <summary>
    /// Layer count. 46 is deliberate: llama.cpp derives Gemma-2's <c>query_pre_attn_scalar</c>
    /// from the model type, and only the 46-layer 27B uses <c>n_embd / n_head</c> instead of
    /// <c>head_dim</c>. That is the only way a GGUF fixture can make the two differ.
    /// </summary>
    public int Layers { get; init; } = 46;
    /// <summary>Hidden size.</summary>
    public int Hidden { get; init; } = 64;
    /// <summary>Query heads.</summary>
    public int Heads { get; init; } = 4;
    /// <summary>KV heads (2 = real GQA; 1 would be the degenerate MQA case).</summary>
    public int KvHeads { get; init; } = 2;
    /// <summary>Per-head dimension (independent of hidden/heads, as in the real 7B/27B).</summary>
    public int HeadDim { get; init; } = 8;
    /// <summary>FFN width.</summary>
    public int FeedForward { get; init; } = 96;
    /// <summary>Vocabulary size.</summary>
    public int Vocab { get; init; } = 96;
    /// <summary>Context length.</summary>
    public int Context { get; init; } = 128;
    /// <summary>Sliding window (gemma2 only; 0 = none).</summary>
    public int SlidingWindow { get; init; } = 3;
    /// <summary>Attention-logit soft-cap (gemma2 only; 0 = key omitted).</summary>
    public float AttnSoftcap { get; init; } = 2.0f;
    /// <summary>Final-logit soft-cap (gemma2 only; 0 = key omitted).</summary>
    public float FinalSoftcap { get; init; } = 3.0f;
    /// <summary>Gemma-3 global RoPE base (<c>rope.freq_base</c>). Deliberately far from <see cref="RopeBaseSwa"/> so a single-table bug shows.</summary>
    public float RopeBase { get; init; } = 50000f;
    /// <summary>Gemma-3 local (windowed-layer) RoPE base (<c>rope.freq_base_swa</c>); 0 = key omitted (loader default 10000).</summary>
    public float RopeBaseSwa { get; init; } = 7000f;
    /// <summary>Gemma-3 linear RoPE scaling factor on the global layers (4B+ ship 8); 0 = no scaling keys.</summary>
    public float RopeLinearFactor { get; init; } = 8f;
    /// <summary>Gemma-3 sliding-window pattern N (llama.cpp <c>load_swa_pattern(ml, 6)</c>): every Nth layer is global. 0 = key omitted (default 6).</summary>
    public int SlidingPattern { get; init; } = 3;
    /// <summary>Write the Gemma-3 sliding pattern as a per-layer bool array instead of the scalar.</summary>
    public bool SlidingPatternAsArray { get; init; }
    /// <summary>RMSNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-6f;
    /// <summary>PRNG seed.</summary>
    public uint Seed { get; init; } = 0x6E3A2u;
}

/// <summary>
/// Deterministic tiny <c>gemma</c> / <c>gemma2</c> GGUF fixture builder (F32 tensors, tied
/// embeddings, no rope keys — mirroring what llama.cpp's converter emits for real Gemma
/// checkpoints). Norm weights are stored the way the real GGUFs store them: the converter's
/// <c>+1</c> is already baked in, so they are consumed as plain RMSNorm gains.
/// </summary>
public static class SyntheticGemma2Gguf
{
    /// <summary>Per-layer tensor bundle, exposed so test oracles can run a reference forward on the exact weights.</summary>
    public sealed class Weights
    {
        /// <summary>Config these weights were built for.</summary>
        public required SyntheticGemma2Config Config { get; init; }
        /// <summary>Token embedding [vocab, hidden] (also the tied LM head).</summary>
        public required float[] TokenEmbd { get; init; }
        /// <summary>Final norm gain [hidden].</summary>
        public required float[] OutputNorm { get; init; }
        /// <summary>Per-layer weights.</summary>
        public required Layer[] Layers { get; init; }
    }

    /// <summary>One layer's weights (row-major [out, in]).</summary>
    public sealed class Layer
    {
        /// <summary>attn_norm [hidden].</summary>
        public required float[] AttnNorm { get; init; }
        /// <summary>attn_q [heads*hd, hidden].</summary>
        public required float[] Q { get; init; }
        /// <summary>attn_k [kv*hd, hidden].</summary>
        public required float[] K { get; init; }
        /// <summary>attn_v [kv*hd, hidden].</summary>
        public required float[] V { get; init; }
        /// <summary>attn_output [hidden, heads*hd].</summary>
        public required float[] O { get; init; }
        /// <summary>attn_q_norm [head_dim] (gemma3 only).</summary>
        public float[]? QNorm { get; init; }
        /// <summary>attn_k_norm [head_dim] (gemma3 only).</summary>
        public float[]? KNorm { get; init; }
        /// <summary>post_attention_norm [hidden] (gemma2 / gemma3).</summary>
        public float[]? PostAttnNorm { get; init; }
        /// <summary>ffn_norm [hidden].</summary>
        public required float[] FfnNorm { get; init; }
        /// <summary>ffn_gate [ff, hidden].</summary>
        public required float[] Gate { get; init; }
        /// <summary>ffn_up [ff, hidden].</summary>
        public required float[] Up { get; init; }
        /// <summary>ffn_down [hidden, ff].</summary>
        public required float[] Down { get; init; }
        /// <summary>post_ffw_norm [hidden] (gemma2 / gemma3).</summary>
        public float[]? PostFfnNorm { get; init; }
    }

    /// <summary>Generates the deterministic weights for <paramref name="cfg"/>.</summary>
    public static Weights BuildWeights(SyntheticGemma2Config cfg)
    {
        bool g2 = cfg.Arch is "gemma2" or "gemma3";
        bool g3 = cfg.Arch == "gemma3";
        var rng = new SyntheticGemma4Gguf.Xorshift(cfg.Seed);
        int qDim = cfg.Heads * cfg.HeadDim, kvDim = cfg.KvHeads * cfg.HeadDim;

        float[] Mat(int rows, int cols, float scale)
        {
            var f = new float[(long)rows * cols];
            for (long i = 0; i < f.Length; i++) f[i] = rng.NextSigned(scale);
            return f;
        }
        float[] Norm(int n)
        {
            // Baked (1 + w) gains: centred on 1 with real spread, like the converter's output.
            var f = new float[n];
            for (int i = 0; i < n; i++) f[i] = 1.0f + rng.NextSigned(0.4f);
            return f;
        }

        var layers = new Layer[cfg.Layers];
        var embd = Mat(cfg.Vocab, cfg.Hidden, 0.5f);
        for (int l = 0; l < cfg.Layers; l++)
        {
            layers[l] = new Layer
            {
                AttnNorm = Norm(cfg.Hidden),
                Q = Mat(qDim, cfg.Hidden, 0.5f),
                K = Mat(kvDim, cfg.Hidden, 0.5f),
                V = Mat(kvDim, cfg.Hidden, 0.5f),
                O = Mat(cfg.Hidden, qDim, 0.25f),
                QNorm = g3 ? Norm(cfg.HeadDim) : null,
                KNorm = g3 ? Norm(cfg.HeadDim) : null,
                PostAttnNorm = g2 ? Norm(cfg.Hidden) : null,
                FfnNorm = Norm(cfg.Hidden),
                Gate = Mat(cfg.FeedForward, cfg.Hidden, 0.3f),
                Up = Mat(cfg.FeedForward, cfg.Hidden, 0.3f),
                Down = Mat(cfg.Hidden, cfg.FeedForward, 0.2f),
                PostFfnNorm = g2 ? Norm(cfg.Hidden) : null,
            };
        }
        return new Weights { Config = cfg, TokenEmbd = embd, OutputNorm = Norm(cfg.Hidden), Layers = layers };
    }

    /// <summary>Writes the fixture and returns the weights it was built from.</summary>
    public static Weights Write(string path, SyntheticGemma2Config? config = null)
    {
        var cfg = config ?? new SyntheticGemma2Config();
        var wts = BuildWeights(cfg);
        File.WriteAllBytes(path, Serialize(wts));
        return wts;
    }

    /// <summary>Serialises <paramref name="wts"/> to GGUF bytes.</summary>
    public static byte[] Serialize(Weights wts)
    {
        var cfg = wts.Config;
        string arch = cfg.Arch;
        bool g2 = arch is "gemma2" or "gemma3";
        bool g3 = arch == "gemma3";
        var w = new GgufWriter();

        w.AddString("general.architecture", arch);
        w.AddString("general.name", $"synthetic-{arch}");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", (uint)cfg.Context);
        w.AddUInt32($"{arch}.embedding_length", (uint)cfg.Hidden);
        w.AddUInt32($"{arch}.block_count", (uint)cfg.Layers);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)cfg.FeedForward);
        w.AddUInt32($"{arch}.attention.head_count", (uint)cfg.Heads);
        w.AddUInt32($"{arch}.attention.head_count_kv", (uint)cfg.KvHeads);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", cfg.NormEps);
        w.AddUInt32($"{arch}.attention.key_length", (uint)cfg.HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)cfg.HeadDim);
        if (g2 && !g3)
        {
            if (cfg.AttnSoftcap > 0) w.AddFloat32($"{arch}.attn_logit_softcapping", cfg.AttnSoftcap);
            if (cfg.FinalSoftcap > 0) w.AddFloat32($"{arch}.final_logit_softcapping", cfg.FinalSoftcap);
            if (cfg.SlidingWindow > 0) w.AddUInt32($"{arch}.attention.sliding_window", (uint)cfg.SlidingWindow);
        }
        if (g3)
        {
            // Gemma-3 has no soft-caps; it carries a window, a 1-in-N global pattern and dual RoPE keys.
            if (cfg.SlidingWindow > 0) w.AddUInt32($"{arch}.attention.sliding_window", (uint)cfg.SlidingWindow);
            if (cfg.SlidingPatternAsArray)
            {
                int n = cfg.SlidingPattern > 0 ? cfg.SlidingPattern : 6;
                var flags = new bool[cfg.Layers];
                for (int i = 0; i < flags.Length; i++) flags[i] = (i % n) < n - 1;
                w.AddBoolArray($"{arch}.attention.sliding_window_pattern", flags);
            }
            else if (cfg.SlidingPattern > 0)
                w.AddUInt32($"{arch}.attention.sliding_window_pattern", (uint)cfg.SlidingPattern);
            w.AddFloat32($"{arch}.rope.freq_base", cfg.RopeBase);
            if (cfg.RopeBaseSwa > 0) w.AddFloat32($"{arch}.rope.freq_base_swa", cfg.RopeBaseSwa);
            if (cfg.RopeLinearFactor > 0)
            {
                w.AddString($"{arch}.rope.scaling.type", "linear");
                w.AddFloat32($"{arch}.rope.scaling.factor", cfg.RopeLinearFactor);
            }
        }
        // gemma / gemma2: deliberately NO rope.* keys - real GGUFs carry none.

        // Tokenizer: BOS=2 like the real vocab.
        w.AddString("tokenizer.ggml.model", "llama");
        var tokens = new string[cfg.Vocab];
        var types = new int[cfg.Vocab];
        for (int i = 0; i < cfg.Vocab; i++)
        {
            tokens[i] = i switch { 0 => "<pad>", 1 => "<eos>", 2 => "<bos>", 3 => "<unk>", _ => $"tok{i}" };
            types[i] = i is 1 or 2 or 0 ? 3 : i == 3 ? 2 : 1;
        }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddFloat32Array("tokenizer.ggml.scores", new float[cfg.Vocab]);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddUInt32("tokenizer.ggml.bos_token_id", 2);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 3);
        w.AddBool("tokenizer.ggml.add_bos_token", true);
        w.AddBool("tokenizer.ggml.add_space_prefix", false);

        void Mat(string name, float[] f, int cols, int rows) =>
            w.AddTensor(name, [cols, rows], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
        void Vec(string name, float[] f) =>
            w.AddTensor(name, [f.Length], (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());

        Mat("token_embd.weight", wts.TokenEmbd, cfg.Hidden, cfg.Vocab);
        for (int l = 0; l < cfg.Layers; l++)
        {
            var L = wts.Layers[l];
            string p = $"blk.{l}";
            Vec($"{p}.attn_norm.weight", L.AttnNorm);
            Mat($"{p}.attn_q.weight", L.Q, cfg.Hidden, cfg.Heads * cfg.HeadDim);
            Mat($"{p}.attn_k.weight", L.K, cfg.Hidden, cfg.KvHeads * cfg.HeadDim);
            Mat($"{p}.attn_v.weight", L.V, cfg.Hidden, cfg.KvHeads * cfg.HeadDim);
            Mat($"{p}.attn_output.weight", L.O, cfg.Heads * cfg.HeadDim, cfg.Hidden);
            if (L.QNorm is not null) Vec($"{p}.attn_q_norm.weight", L.QNorm);
            if (L.KNorm is not null) Vec($"{p}.attn_k_norm.weight", L.KNorm);
            if (L.PostAttnNorm is not null) Vec($"{p}.post_attention_norm.weight", L.PostAttnNorm);
            Vec($"{p}.ffn_norm.weight", L.FfnNorm);
            Mat($"{p}.ffn_gate.weight", L.Gate, cfg.Hidden, cfg.FeedForward);
            Mat($"{p}.ffn_up.weight", L.Up, cfg.Hidden, cfg.FeedForward);
            Mat($"{p}.ffn_down.weight", L.Down, cfg.FeedForward, cfg.Hidden);
            if (L.PostFfnNorm is not null) Vec($"{p}.post_ffw_norm.weight", L.PostFfnNorm);
        }
        Vec("output_norm.weight", wts.OutputNorm);
        return w.Build();
    }
}
