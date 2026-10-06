using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Shape knobs for <see cref="SyntheticGraniteGguf"/>. The four Granite scalars default to values that are
/// each != 1 AND mutually distinct (embedding 3.0, attention 0.07, residual 0.4, logit 5.0), so a swapped,
/// missing or doubly-applied scalar cannot produce the same logits as the correct model. (The pre-#764
/// parity tests used the HF defaults and could not tell.)
/// </summary>
public sealed record SyntheticGraniteConfig
{
    /// <summary>
    /// <c>granite</c> (dense), <c>granitemoe</c> (stacked-expert MoE), or the other Llama-family arms sharing this fixture:
    /// <c>smollm3</c> (Llama + NoPE every 4th layer), <c>olmo2</c> (post-norm-only layout, whole-projection QK-norm, NeoX)
    /// and <c>olmoe</c> (QK-norm whole projection, softmax-no-renorm MoE, NeoX). Granite scalars are meaningful only for
    /// the granite arches (set them to 0 for the others).
    /// </summary>
    public string Arch { get; init; } = "granite";
    /// <summary>Layer count.</summary>
    public int Layers { get; init; } = 3;
    /// <summary>Hidden size (head_dim = hidden / heads = 32, a multiple of the Q8_0 block so the quantized-KV kernel runs; Granite has no head_dim key).</summary>
    public int Hidden { get; init; } = 128;
    /// <summary>Query heads.</summary>
    public int Heads { get; init; } = 4;
    /// <summary>KV heads (2 = real GQA).</summary>
    public int KvHeads { get; init; } = 2;
    /// <summary>FFN width (per expert for MoE).</summary>
    public int FeedForward { get; init; } = 96;
    /// <summary>Vocabulary size.</summary>
    public int Vocab { get; init; } = 96;
    /// <summary>Context length.</summary>
    public int Context { get; init; } = 128;
    /// <summary>Experts (MoE only).</summary>
    public int Experts { get; init; } = 4;
    /// <summary>Experts per token (MoE only).</summary>
    public int ExpertsUsed { get; init; } = 2;
    /// <summary><c>embedding_scale</c>; 0 = key omitted.</summary>
    public float EmbeddingScale { get; init; } = 3.0f;
    /// <summary><c>attention.scale</c>; 0 = key omitted (default 1/sqrt(head_dim)).</summary>
    public float AttentionScale { get; init; } = 0.07f;
    /// <summary><c>residual_scale</c>; 0 = key omitted.</summary>
    public float ResidualScale { get; init; } = 0.4f;
    /// <summary><c>logit_scale</c> (logits are divided by it); 0 = key omitted (the extractor then rejects the file).</summary>
    public float LogitScale { get; init; } = 5.0f;
    /// <summary>RoPE base.</summary>
    public float RopeBase { get; init; } = 10000f;
    /// <summary>Omit <c>output.weight</c> (tied LM head, as Granite-3.x 2B/8B ship).</summary>
    public bool Tied { get; init; }
    /// <summary>
    /// Store the stacked expert banks as Q8_0 (as real granitemoe GGUFs do) instead of F32. Required for the GPU backends,
    /// whose routed-MoE loaders consume raw quantized expert banks. The CPU-vs-reference test keeps F32 for exactness.
    /// </summary>
    public bool ExpertsQ8_0 { get; init; }
    /// <summary>Non-zero writes <c>expert_shared_feed_forward_length</c> (MoE-shared; the loader must reject it).</summary>
    public int SharedFeedForward { get; init; }
    /// <summary>Writes <c>attention.sliding_window</c> (OLMo 3 style; the olmo2 loader must refuse it). 0 = key omitted.</summary>
    public int SlidingWindow { get; init; }
    /// <summary>RMSNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;
    /// <summary>PRNG seed.</summary>
    public uint Seed { get; init; } = 0x6A41u;

    /// <summary>True for the MoE variant.</summary>
    public bool IsMoe => Arch is "granitemoe" or "olmoe" or "qwen2moe";
    /// <summary>True for OLMo 2 (no pre-norms, post-attention/post-FFN norms).</summary>
    public bool IsOlmo2 => Arch == "olmo2";
    /// <summary>True for the archs with a whole-projection Q/K RMSNorm (OLMo 2, OLMoE).</summary>
    public bool HasQkNorm => Arch is "olmo2" or "olmoe";
    /// <summary>NeoX (rotate-half) RoPE pairing (OLMo 2 / OLMoE); the others use adjacent-pair "Norm" pairing.</summary>
    public bool NeoXRope => Arch is "olmo2" or "olmoe";
    /// <summary>True for SmolLM3: every 4th layer ((il+1) % 4 == 0) skips RoPE.</summary>
    public bool NoPeEvery4th => Arch == "smollm3";
    /// <summary>Per-head dimension (hidden / heads).</summary>
    public int HeadDim => Hidden / Heads;
}

/// <summary>
/// Deterministic tiny <c>granite</c> / <c>granitemoe</c> GGUF fixture builder (F32 tensors). Tensor names and
/// the stacked-expert layout follow llama.cpp's <c>granite.cpp</c> / <c>granite-moe.cpp</c>.
/// </summary>
public static class SyntheticGraniteGguf
{
    /// <summary>One layer's weights (row-major [out, in]; experts [E, out, in]).</summary>
    public sealed class Layer
    {
        /// <summary>attn_norm.</summary>
        public required float[] AttnNorm { get; init; }
        /// <summary>attn_q.</summary>
        public required float[] Q { get; init; }
        /// <summary>attn_k.</summary>
        public required float[] K { get; init; }
        /// <summary>attn_v.</summary>
        public required float[] V { get; init; }
        /// <summary>attn_output.</summary>
        public required float[] O { get; init; }
        /// <summary>ffn_norm.</summary>
        public required float[] FfnNorm { get; init; }
        /// <summary>ffn_gate (dense) [ff, hidden].</summary>
        public float[]? Gate { get; init; }
        /// <summary>ffn_up (dense) [ff, hidden].</summary>
        public float[]? Up { get; init; }
        /// <summary>ffn_down (dense) [hidden, ff].</summary>
        public float[]? Down { get; init; }
        /// <summary>attn_q_norm over the whole Q projection [heads*head_dim] (OLMo 2 / OLMoE).</summary>
        public float[]? QNorm { get; init; }
        /// <summary>attn_k_norm over the whole K projection [kv_heads*head_dim] (OLMo 2 / OLMoE).</summary>
        public float[]? KNorm { get; init; }
        /// <summary>post_attention_norm [hidden] (OLMo 2).</summary>
        public float[]? PostAttnNorm { get; init; }
        /// <summary>post_ffw_norm [hidden] (OLMo 2).</summary>
        public float[]? PostFfnNorm { get; init; }
        /// <summary>ffn_gate_inp (MoE) [E, hidden].</summary>
        public float[]? Router { get; init; }
        /// <summary>ffn_gate_exps (MoE) [E, ff, hidden].</summary>
        public float[]? GateExps { get; init; }
        /// <summary>ffn_up_exps (MoE) [E, ff, hidden].</summary>
        public float[]? UpExps { get; init; }
        /// <summary>ffn_down_exps (MoE) [E, hidden, ff].</summary>
        public float[]? DownExps { get; init; }
    }

    /// <summary>Whole-model weights, exposed so test oracles run a reference forward on the exact values.</summary>
    public sealed class Weights
    {
        /// <summary>Config these weights were built for.</summary>
        public required SyntheticGraniteConfig Config { get; init; }
        /// <summary>token_embd [vocab, hidden].</summary>
        public required float[] TokenEmbd { get; init; }
        /// <summary>output_norm.</summary>
        public required float[] OutputNorm { get; init; }
        /// <summary>output [vocab, hidden] (== TokenEmbd when tied).</summary>
        public required float[] Output { get; init; }
        /// <summary>Layers.</summary>
        public required Layer[] Layers { get; init; }
    }

    /// <summary>Generates the deterministic weights.</summary>
    public static Weights BuildWeights(SyntheticGraniteConfig cfg)
    {
        var rng = new SyntheticGemma4Gguf.Xorshift(cfg.Seed);
        int qDim = cfg.Heads * cfg.HeadDim, kvDim = cfg.KvHeads * cfg.HeadDim;
        float[] Mat(long n, float scale)
        {
            var f = new float[n];
            for (long i = 0; i < n; i++) f[i] = rng.NextSigned(scale);
            return f;
        }
        float[] Norm(int n)
        {
            var f = new float[n];
            for (int i = 0; i < n; i++) f[i] = 1.0f + rng.NextSigned(0.3f);
            return f;
        }

        var embd = Mat((long)cfg.Vocab * cfg.Hidden, 0.5f);
        var layers = new Layer[cfg.Layers];
        for (int l = 0; l < cfg.Layers; l++)
        {
            layers[l] = new Layer
            {
                AttnNorm = Norm(cfg.Hidden),
                Q = Mat((long)qDim * cfg.Hidden, 0.5f),
                K = Mat((long)kvDim * cfg.Hidden, 0.5f),
                V = Mat((long)kvDim * cfg.Hidden, 0.5f),
                O = Mat((long)cfg.Hidden * qDim, 0.4f),
                FfnNorm = Norm(cfg.Hidden),
                Gate = cfg.IsMoe ? null : Mat((long)cfg.FeedForward * cfg.Hidden, 0.3f),
                Up = cfg.IsMoe ? null : Mat((long)cfg.FeedForward * cfg.Hidden, 0.3f),
                Down = cfg.IsMoe ? null : Mat((long)cfg.Hidden * cfg.FeedForward, 0.3f),
                QNorm = cfg.HasQkNorm ? Norm(qDim) : null,
                KNorm = cfg.HasQkNorm ? Norm(kvDim) : null,
                PostAttnNorm = cfg.IsOlmo2 ? Norm(cfg.Hidden) : null,
                PostFfnNorm = cfg.IsOlmo2 ? Norm(cfg.Hidden) : null,
                Router = cfg.IsMoe ? Mat((long)cfg.Experts * cfg.Hidden, 1.0f) : null,
                GateExps = cfg.IsMoe ? Mat((long)cfg.Experts * cfg.FeedForward * cfg.Hidden, 0.3f) : null,
                UpExps = cfg.IsMoe ? Mat((long)cfg.Experts * cfg.FeedForward * cfg.Hidden, 0.3f) : null,
                DownExps = cfg.IsMoe ? Mat((long)cfg.Experts * cfg.Hidden * cfg.FeedForward, 0.3f) : null,
            };
        }
        // Q8_0 experts: hold the DEQUANTIZED values so an oracle running on these floats sees exactly what the file stores.
        if (cfg.ExpertsQ8_0)
            foreach (var L in layers)
            {
                if (L.GateExps is null) continue;
                RoundTripQ8_0(L.GateExps); RoundTripQ8_0(L.UpExps!); RoundTripQ8_0(L.DownExps!);
            }
        var outNorm = Norm(cfg.Hidden);
        var output = cfg.Tied ? embd : Mat((long)cfg.Vocab * cfg.Hidden, 0.5f);
        return new Weights { Config = cfg, TokenEmbd = embd, OutputNorm = outNorm, Output = output, Layers = layers };
    }

    /// <summary>Writes the fixture and returns the weights it was built from.</summary>
    public static Weights Write(string path, SyntheticGraniteConfig? config = null)
    {
        var wts = BuildWeights(config ?? new SyntheticGraniteConfig());
        File.WriteAllBytes(path, Serialize(wts));
        return wts;
    }

    /// <summary>Serialises <paramref name="wts"/> to GGUF bytes.</summary>
    public static byte[] Serialize(Weights wts)
    {
        var cfg = wts.Config;
        string arch = cfg.Arch;
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
        w.AddFloat32($"{arch}.rope.freq_base", cfg.RopeBase);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)cfg.HeadDim);
        if (cfg.SlidingWindow > 0) w.AddUInt32($"{arch}.attention.sliding_window", (uint)cfg.SlidingWindow);
        if (cfg.IsMoe)
        {
            w.AddUInt32($"{arch}.expert_count", (uint)cfg.Experts);
            w.AddUInt32($"{arch}.expert_used_count", (uint)cfg.ExpertsUsed);
            w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)cfg.FeedForward);
            if (cfg.SharedFeedForward > 0)
                w.AddUInt32($"{arch}.expert_shared_feed_forward_length", (uint)cfg.SharedFeedForward);
        }
        // llama.cpp conversion/granite.py: HF *_multiplier / logits_scaling -> *_scale keys.
        if (cfg.EmbeddingScale > 0) w.AddFloat32($"{arch}.embedding_scale", cfg.EmbeddingScale);
        if (cfg.AttentionScale > 0) w.AddFloat32($"{arch}.attention.scale", cfg.AttentionScale);
        if (cfg.ResidualScale > 0) w.AddFloat32($"{arch}.residual_scale", cfg.ResidualScale);
        if (cfg.LogitScale > 0) w.AddFloat32($"{arch}.logit_scale", cfg.LogitScale);

        w.AddString("tokenizer.ggml.model", "llama");
        var tokens = new string[cfg.Vocab];
        var types = new int[cfg.Vocab];
        for (int i = 0; i < cfg.Vocab; i++)
        {
            tokens[i] = i switch { 0 => "<unk>", 1 => "<s>", 2 => "</s>", _ => $"tok{i}" };
            types[i] = i == 0 ? 2 : i is 1 or 2 ? 3 : 1;
        }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddFloat32Array("tokenizer.ggml.scores", new float[cfg.Vocab]);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddUInt32("tokenizer.ggml.bos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 2);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);

        void Mat(string name, float[] f, params int[] dims) =>
            w.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(f.AsSpan()).ToArray());
        void Exps(string name, float[] f, params int[] dims)
        {
            if (cfg.ExpertsQ8_0)
                w.AddTensor(name, dims, (uint)QuantizationType.Q8_0, QuantizeQ8_0(f, dims[0]));
            else
                Mat(name, f, dims);
        }

        int qDim = cfg.Heads * cfg.HeadDim, kvDim = cfg.KvHeads * cfg.HeadDim;
        Mat("token_embd.weight", wts.TokenEmbd, cfg.Hidden, cfg.Vocab);
        for (int l = 0; l < cfg.Layers; l++)
        {
            var L = wts.Layers[l];
            string p = $"blk.{l}";
            if (!cfg.IsOlmo2) Mat($"{p}.attn_norm.weight", L.AttnNorm, cfg.Hidden);   // OLMo 2 has no pre-norm tensors
            Mat($"{p}.attn_q.weight", L.Q, cfg.Hidden, qDim);
            Mat($"{p}.attn_k.weight", L.K, cfg.Hidden, kvDim);
            Mat($"{p}.attn_v.weight", L.V, cfg.Hidden, kvDim);
            Mat($"{p}.attn_output.weight", L.O, qDim, cfg.Hidden);
            if (L.QNorm is not null) Mat($"{p}.attn_q_norm.weight", L.QNorm, qDim);
            if (L.KNorm is not null) Mat($"{p}.attn_k_norm.weight", L.KNorm, kvDim);
            if (L.PostAttnNorm is not null) Mat($"{p}.post_attention_norm.weight", L.PostAttnNorm, cfg.Hidden);
            if (L.PostFfnNorm is not null) Mat($"{p}.post_ffw_norm.weight", L.PostFfnNorm, cfg.Hidden);
            if (!cfg.IsOlmo2) Mat($"{p}.ffn_norm.weight", L.FfnNorm, cfg.Hidden);
            if (cfg.IsMoe)
            {
                Mat($"{p}.ffn_gate_inp.weight", L.Router!, cfg.Hidden, cfg.Experts);
                Exps($"{p}.ffn_gate_exps.weight", L.GateExps!, cfg.Hidden, cfg.FeedForward, cfg.Experts);
                Exps($"{p}.ffn_up_exps.weight", L.UpExps!, cfg.Hidden, cfg.FeedForward, cfg.Experts);
                Exps($"{p}.ffn_down_exps.weight", L.DownExps!, cfg.FeedForward, cfg.Hidden, cfg.Experts);
            }
            else
            {
                Mat($"{p}.ffn_gate.weight", L.Gate!, cfg.Hidden, cfg.FeedForward);
                Mat($"{p}.ffn_up.weight", L.Up!, cfg.Hidden, cfg.FeedForward);
                Mat($"{p}.ffn_down.weight", L.Down!, cfg.FeedForward, cfg.Hidden);
            }
        }
        Mat("output_norm.weight", wts.OutputNorm, cfg.Hidden);
        if (!cfg.Tied) Mat("output.weight", wts.Output, cfg.Hidden, cfg.Vocab);
        return w.Build();
    }

    /// <summary>In-place Q8_0 quantize -> dequantize round trip (32-element blocks).</summary>
    private static void RoundTripQ8_0(float[] f)
    {
        byte[] q = QuantizeQ8_0(f, 32);
        for (int b = 0; b < f.Length / 32; b++)
        {
            float d = (float)BitConverter.ToHalf(q, b * 34);
            for (int i = 0; i < 32; i++) f[b * 32 + i] = d * (sbyte)q[b * 34 + 2 + i];
        }
    }

    /// <summary>Quantizes row-major rows of length <paramref name="k"/> (multiple of 32) to GGUF Q8_0 (34-byte blocks: f16 scale + 32 int8).</summary>
    private static byte[] QuantizeQ8_0(float[] f, int k)
    {
        if (k % 32 != 0) throw new ArgumentException("Q8_0 rows must be a multiple of 32.", nameof(k));
        int blocks = f.Length / 32;
        var bytes = new byte[blocks * 34];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(f[b * 32 + i]));
            float d = amax / 127f;
            float inv = d != 0 ? 1f / d : 0f;
            Half hd = (Half)d;
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 34, 2), hd);
            for (int i = 0; i < 32; i++)
                bytes[b * 34 + 2 + i] = (byte)(sbyte)MathF.Round(f[b * 32 + i] * inv);
        }
        return bytes;
    }
}
