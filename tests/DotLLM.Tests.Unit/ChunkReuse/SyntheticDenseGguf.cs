using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Deterministic random-weight dense <c>qwen3</c> / <c>llama</c> GGUF for the chunk-reuse tests (no fixture files; same writer
/// and RNG pattern as <c>SyntheticFlintGguf</c>). Chosen to be discriminating for RoPE re-rotation: GQA 4:2 (two KV heads, so a
/// per-head slicing bug cannot hide behind one head), head dim 16 (8 rotation pairs), a small RoPE base (100) so even the
/// lowest-frequency pair turns by a visible angle over a few positions, and three layers so every layer above 0 depends on
/// attention. <c>qwen3</c> carries per-head Q/K RMSNorm (applied before RoPE) and NeoX pairing; <c>llama</c> has no QK norm and
/// interleaved (Norm) pairing.
/// </summary>
internal static class SyntheticDenseGguf
{
    public const int Hidden = 64, Layers = 3, Heads = 4, Kv = 2, HeadDim = 16, Ff = 96, Vocab = 48, Context = 64;
    public const float RopeTheta = 100f;

    /// <summary>Vocabulary size and context length of <see cref="Build"/> with <c>byteLevelVocab</c>.</summary>
    public const int ByteVocab = 256, ByteContext = 8192;

    /// <summary>Builds the GGUF bytes for <paramref name="arch"/> (<c>qwen3</c> or <c>llama</c>).</summary>
    /// <param name="arch">Architecture.</param>
    /// <param name="seed">Weight seed.</param>
    /// <param name="byteLevelVocab">
    /// False (default): the 48-token <c>t0..t47</c> vocabulary of the id-level tests. True: a GPT-2 byte-level vocabulary of the
    /// 256 byte tokens (no merges), so the GGUF's own tokenizer encodes any UTF-8 text one token per byte and decodes it back
    /// exactly, and a context length of <see cref="ByteContext"/>: a whole-tool test model (chunkcache) that reads source files.
    /// </param>
    public static byte[] Build(string arch, int seed = 20261031, bool byteLevelVocab = false)
    {
        int vocab = byteLevelVocab ? ByteVocab : Vocab;
        bool qkNorm = arch switch
        {
            "qwen3" => true,
            "llama" => false,
            _ => throw new ArgumentOutOfRangeException(nameof(arch), arch, "qwen3 or llama"),
        };

        var rng = new Random(seed);
        var w = new GgufWriter();
        w.AddString("general.architecture", arch);
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.context_length", (uint)(byteLevelVocab ? ByteContext : Context));
        w.AddUInt32($"{arch}.embedding_length", Hidden);
        w.AddUInt32($"{arch}.block_count", Layers);
        w.AddUInt32($"{arch}.feed_forward_length", Ff);
        w.AddUInt32($"{arch}.attention.head_count", Heads);
        w.AddUInt32($"{arch}.attention.head_count_kv", Kv);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        // Like the real Qwen3-4B-Instruct-2507 GGUF: freq_base only, no rope.dimension_count (full-head rotation) and no
        // rope.scaling.* keys.
        w.AddFloat32($"{arch}.rope.freq_base", RopeTheta);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f);

        var tokens = byteLevelVocab ? ByteLevelTokens() : new string[Vocab];
        if (!byteLevelVocab)
            for (int i = 0; i < Vocab; i++)
                tokens[i] = "t" + i.ToString(System.Globalization.CultureInfo.InvariantCulture);
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, vocab).ToArray());
        w.AddStringArray("tokenizer.ggml.merges", []);

        Matrix(w, rng, "token_embd.weight", Hidden, vocab, 0.6f);
        Norm(w, rng, "output_norm.weight", Hidden);
        Matrix(w, rng, "output.weight", Hidden, vocab, 0.25f);

        float sh = 1.4f / MathF.Sqrt(Hidden);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"blk.{i}";
            Norm(w, rng, $"{p}.attn_norm.weight", Hidden);
            Matrix(w, rng, $"{p}.attn_q.weight", Hidden, Heads * HeadDim, sh);
            Matrix(w, rng, $"{p}.attn_k.weight", Hidden, Kv * HeadDim, sh);
            Matrix(w, rng, $"{p}.attn_v.weight", Hidden, Kv * HeadDim, sh);
            Matrix(w, rng, $"{p}.attn_output.weight", Heads * HeadDim, Hidden, 1.4f / MathF.Sqrt(Heads * HeadDim));
            if (qkNorm)
            {
                Norm(w, rng, $"{p}.attn_q_norm.weight", HeadDim);
                Norm(w, rng, $"{p}.attn_k_norm.weight", HeadDim);
            }
            Norm(w, rng, $"{p}.ffn_norm.weight", Hidden);
            Matrix(w, rng, $"{p}.ffn_gate.weight", Hidden, Ff, sh);
            Matrix(w, rng, $"{p}.ffn_up.weight", Hidden, Ff, sh);
            Matrix(w, rng, $"{p}.ffn_down.weight", Ff, Hidden, 1.4f / MathF.Sqrt(Ff));
        }
        return w.Build();
    }

    /// <summary>GPT-2's bytes_to_unicode: token i is the printable character standing for byte i.</summary>
    private static string[] ByteLevelTokens()
    {
        var printable = new List<int>();
        for (int b = '!'; b <= '~'; b++) printable.Add(b);
        for (int b = 0xA1; b <= 0xAC; b++) printable.Add(b);
        for (int b = 0xAE; b <= 0xFF; b++) printable.Add(b);
        var tokens = new string[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
            tokens[b] = ((char)(printable.Contains(b) ? b : 256 + extra++)).ToString();
        return tokens;
    }

    private static double Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static byte[] Bytes(float[] f) => MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    private static void Matrix(GgufWriter w, Random rng, string name, int inK, int outM, float std)
    {
        var f = new float[inK * outM];
        for (int i = 0; i < f.Length; i++) f[i] = (float)(Gauss(rng) * std);
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.F32, Bytes(f));
    }

    private static void Norm(GgufWriter w, Random rng, string name, int n)
    {
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + 0.2f * (float)Gauss(rng);
        w.AddTensor(name, [n], (uint)QuantizationType.F32, Bytes(f));
    }
}
