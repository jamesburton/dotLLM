using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Deterministic synthetic <c>glm4moe</c> (GLM-4.5-Air / REAP / GLM-4.5) GGUF fixture builder (issue #1598). Emits a TINY file with the
/// exact key and tensor naming of a real llama.cpp conversion (verified against the bartowski REAP-82B IQ4_XS header): GQA attention with
/// Q/K/V bias and no O bias, partial NeoX RoPE (<c>rope.dimension_count</c> = head_dim / 2), <c>post_attention_norm</c> as the pre-FFN norm,
/// a dense SwiGLU layer 0, sigmoid + <c>exp_probs_b</c> MoE layers with an ungated shared expert, and one trailing NextN block counted in
/// <c>block_count</c>. All tensors are F32. There is no forward pass for this architecture yet: the fixture exists for config extraction,
/// the tensor census and the loud-refusal paths.
/// </summary>
/// <remarks>
/// <para>
/// The shapes are scaled-down images of the real ones (Air: hidden 4096, 96 q / 8 KV heads, head_dim 128, rope 64, dense FFN 10944,
/// MoE 1408, 128 experts top-8, 46 trunk layers + 1 NextN). the <c>numExperts</c> argument of <see cref="Build"/> is a parameter so the 128-expert (Air) and 96-expert
/// (REAP) cases are both reproducible; the expert counts are kept distinct from top-k, shared count and every width so a swapped field is observable.
/// </para>
/// </remarks>
public static class SyntheticGlm4MoeGguf
{
    /// <summary>Vocabulary size (also the tokenizer array length).</summary>
    public const int VocabSize = 16;
    /// <summary>Hidden size.</summary>
    public const int HiddenSize = 32;
    /// <summary>Attention heads.</summary>
    public const int NumAttentionHeads = 6;
    /// <summary>KV heads (GQA group 3).</summary>
    public const int NumKvHeads = 2;
    /// <summary>Head dimension.</summary>
    public const int HeadDim = 16;
    /// <summary>Rotated slice per head: <c>partial_rotary_factor</c> 0.5 of <see cref="HeadDim"/>.</summary>
    public const int RopeDim = 8;
    /// <summary>Dense FFN width (layer 0).</summary>
    public const int DenseIntermediate = 40;
    /// <summary>Per-expert FFN width.</summary>
    public const int MoeIntermediate = 24;
    /// <summary>Routed experts used per token.</summary>
    public const int ExpertsUsed = 3;
    /// <summary>Shared experts (fused into one SwiGLU of width <see cref="MoeIntermediate"/> * count).</summary>
    public const int SharedExperts = 2;
    /// <summary>Default routed expert count (stands in for Air's 128).</summary>
    public const int DefaultExperts = 8;
    /// <summary>Default trunk depth: layer 0 dense, layers 1..N-1 MoE (stands in for Air's 46).</summary>
    public const int DefaultTrunkLayers = 3;
    /// <summary>Leading dense blocks.</summary>
    public const int LeadingDense = 1;
    /// <summary>Context length.</summary>
    public const int ContextLength = 64;

    /// <summary>Builds the synthetic <c>glm4moe</c> fixture.</summary>
    /// <param name="numExperts">Routed experts (default 8; pass a different value to emulate the REAP-pruned shape).</param>
    /// <param name="trunkLayers">Trunk depth excluding the NextN block (must exceed <see cref="LeadingDense"/>).</param>
    /// <param name="withNextn">Append one NextN/MTP block (counted in <c>block_count</c>, <c>nextn_predict_layers</c> = 1), as the bartowski/unsloth files do.</param>
    /// <param name="withQkNorm">Emit per-head <c>attn_q_norm</c>/<c>attn_k_norm</c> (GLM-4.5 355B); Air has none.</param>
    /// <param name="seed">Xorshift PRNG seed.</param>
    /// <param name="omitTensor">Tensor name to leave out (census mismatch tests).</param>
    /// <param name="wrongShapeTensor">Tensor name to emit with an extra trailing element (census shape-mismatch tests).</param>
    /// <returns>The GGUF bytes.</returns>
    public static byte[] Build(int numExperts = DefaultExperts, int trunkLayers = DefaultTrunkLayers, bool withNextn = true,
                               bool withQkNorm = false, uint seed = 0xA1B2C3u, string? omitTensor = null, string? wrongShapeTensor = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(trunkLayers, LeadingDense);
        ArgumentOutOfRangeException.ThrowIfLessThan(numExperts, ExpertsUsed);
        var w = new GgufWriter();
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        const string arch = "glm4moe";
        int nextn = withNextn ? 1 : 0;

        w.AddString("general.architecture", arch);
        w.AddString("general.type", "model");
        w.AddString("general.name", "synthetic-glm4moe-tiny");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.block_count", (uint)(trunkLayers + nextn));
        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.feed_forward_length", DenseIntermediate);
        w.AddUInt32($"{arch}.attention.head_count", NumAttentionHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddFloat32($"{arch}.rope.freq_base", 1000000.0f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.expert_used_count", ExpertsUsed);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.attention.value_length", HeadDim);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);
        w.AddUInt32($"{arch}.expert_count", (uint)numExperts);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeIntermediate);
        w.AddUInt32($"{arch}.expert_shared_count", SharedExperts);
        w.AddUInt32($"{arch}.leading_dense_block_count", LeadingDense);
        w.AddUInt32($"{arch}.expert_gating_func", 2);            // sigmoid
        w.AddFloat32($"{arch}.expert_weights_scale", 1.0f);
        w.AddBool($"{arch}.expert_weights_norm", true);
        if (withNextn)
            w.AddUInt32($"{arch}.nextn_predict_layers", 1);
        AddTokenizer(w);

        void Mat(string name, int inK, int outM, float amp = 0.05f)
        {
            if (name == omitTensor) return;
            int extra = name == wrongShapeTensor ? 1 : 0;
            var f = new float[(long)(inK + extra) * outM];
            for (int i = 0; i < f.Length; i++) f[i] = rng.NextSigned(amp);
            Add(w, name, [inK + extra, outM], f);
        }
        void Vec(string name, int n, float center, float amp = 0.05f)
        {
            if (name == omitTensor) return;
            int len = n + (name == wrongShapeTensor ? 1 : 0);
            var f = new float[len];
            for (int i = 0; i < f.Length; i++) f[i] = center + rng.NextSigned(amp);
            Add(w, name, [len], f);
        }
        void Bank(string name, int inK, int mid)
        {
            if (name == omitTensor) return;
            int extra = name == wrongShapeTensor ? 1 : 0;
            var f = new float[(long)(inK + extra) * mid * numExperts];
            for (int i = 0; i < f.Length; i++) f[i] = rng.NextSigned(0.4f);
            Add(w, name, [inK + extra, mid, numExperts], f);
        }

        int nq = NumAttentionHeads * HeadDim, nkv = NumKvHeads * HeadDim;
        int shared = MoeIntermediate * SharedExperts;

        Mat("token_embd.weight", HiddenSize, VocabSize);
        Vec("output_norm.weight", HiddenSize, 1f);
        Mat("output.weight", HiddenSize, VocabSize);

        for (int l = 0; l < trunkLayers + nextn; l++)
        {
            string p = $"blk.{l}";
            bool isNextn = l >= trunkLayers;
            Vec($"{p}.attn_norm.weight", HiddenSize, 1f);
            // Q/K amplitudes (and the biases) are large on purpose: with +-0.05 the softmax is near-uniform and RoPE, the QKV bias and
            // the attention path are all but invisible in the logits, so a test could not tell a wrong rotation from a right one.
            Mat($"{p}.attn_q.weight", HiddenSize, nq, 1.0f);
            Vec($"{p}.attn_q.bias", nq, 0f, 0.8f);
            Mat($"{p}.attn_k.weight", HiddenSize, nkv, 1.0f);
            Vec($"{p}.attn_k.bias", nkv, 0f, 0.8f);
            Mat($"{p}.attn_v.weight", HiddenSize, nkv);
            Vec($"{p}.attn_v.bias", nkv, 0f, 0.3f);
            Mat($"{p}.attn_output.weight", nq, HiddenSize);
            if (withQkNorm)
            {
                Vec($"{p}.attn_q_norm.weight", HeadDim, 1f);
                Vec($"{p}.attn_k_norm.weight", HeadDim, 1f);
            }
            Vec($"{p}.post_attention_norm.weight", HiddenSize, 1f);
            if (!isNextn && l < LeadingDense)
            {
                Mat($"{p}.ffn_gate.weight", HiddenSize, DenseIntermediate);
                Mat($"{p}.ffn_up.weight", HiddenSize, DenseIntermediate);
                Mat($"{p}.ffn_down.weight", DenseIntermediate, HiddenSize);
            }
            else
            {
                // Router + selection bias large enough that routing is peaky, the bias flips selections, and the sigmoid scores are not all ~0.5.
                Mat($"{p}.ffn_gate_inp.weight", HiddenSize, numExperts, 0.6f);
                Vec($"{p}.exp_probs_b.bias", numExperts, 0f, 0.3f);
                Bank($"{p}.ffn_gate_exps.weight", HiddenSize, MoeIntermediate);
                Bank($"{p}.ffn_up_exps.weight", HiddenSize, MoeIntermediate);
                Bank($"{p}.ffn_down_exps.weight", MoeIntermediate, HiddenSize);
                Mat($"{p}.ffn_gate_shexp.weight", HiddenSize, shared, 0.4f);
                Mat($"{p}.ffn_up_shexp.weight", HiddenSize, shared, 0.4f);
                Mat($"{p}.ffn_down_shexp.weight", shared, HiddenSize, 0.4f);
            }
            if (isNextn)
            {
                Mat($"{p}.nextn.eh_proj.weight", 2 * HiddenSize, HiddenSize);
                Vec($"{p}.nextn.enorm.weight", HiddenSize, 1f);
                Vec($"{p}.nextn.hnorm.weight", HiddenSize, 1f);
                Mat($"{p}.nextn.embed_tokens.weight", HiddenSize, VocabSize);
                Mat($"{p}.nextn.shared_head_head.weight", HiddenSize, VocabSize);
                Vec($"{p}.nextn.shared_head_norm.weight", HiddenSize, 1f);
            }
        }
        return w.Build();
    }

    /// <summary>Writes the synthetic <c>glm4moe</c> fixture to <paramref name="path"/>.</summary>
    /// <param name="path">Destination file.</param>
    /// <param name="numExperts">See <see cref="Build"/>.</param>
    /// <param name="trunkLayers">See <see cref="Build"/>.</param>
    /// <param name="withNextn">See <see cref="Build"/>.</param>
    /// <param name="withQkNorm">See <see cref="Build"/>.</param>
    /// <returns><paramref name="path"/>.</returns>
    public static string Write(string path, int numExperts = DefaultExperts, int trunkLayers = DefaultTrunkLayers, bool withNextn = true,
                               bool withQkNorm = false)
    {
        File.WriteAllBytes(path, Build(numExperts, trunkLayers, withNextn, withQkNorm));
        return path;
    }

    private static void Add(GgufWriter w, string name, int[] dims, float[] values)
        => w.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    private static void AddTokenizer(GgufWriter w)
    {
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddString("tokenizer.ggml.pre", "glm4");
        var tokens = new string[VocabSize];
        var types = new int[VocabSize];
        for (int i = 0; i < VocabSize; i++) { tokens[i] = $"tok{i}"; types[i] = 1; }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddStringArray("tokenizer.ggml.merges", []);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);
    }
}
