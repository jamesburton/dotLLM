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
    /// <param name="headDim">Per-head width (default 16; 128 reproduces the real Air head geometry, rope = headDim / 2).</param>
    /// <param name="q8Weights">Store every matrix / expert bank as Q8_0 (widths 64 / 32; for GPU loaders that keep banks packed).</param>
    /// <param name="q8AsF32">As <paramref name="q8Weights"/> but write the dequantised values as F32 (exact-weight oracle).</param>
    /// <param name="omitTensor">Tensor name to leave out (census mismatch tests).</param>
    /// <param name="wrongShapeTensor">Tensor name to emit with an extra trailing element (census shape-mismatch tests).</param>
    /// <returns>The GGUF bytes.</returns>
    public static byte[] Build(int numExperts = DefaultExperts, int trunkLayers = DefaultTrunkLayers, bool withNextn = true,
                               bool withQkNorm = false, uint seed = 0xA1B2C3u, string? omitTensor = null, string? wrongShapeTensor = null, int headDim = HeadDim,
                               bool q8Weights = false, bool q8AsF32 = false)
    {
        // q8Weights (#1600): every matrix and expert bank is stored Q8_0 (the Vulkan loader only keeps packed banks resident, F32 banks are refused), which
        // needs every reduction length to be a multiple of 32: the FFN widths become 64 (dense) / 32 (per expert, 64 fused shared).
        q8Weights |= q8AsF32;
        int denseInter = q8Weights ? 64 : DenseIntermediate, moeInter = q8Weights ? 32 : MoeIntermediate;
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
        w.AddUInt32($"{arch}.context_length", headDim == HeadDim ? (uint)ContextLength : 512u);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)denseInter);
        w.AddUInt32($"{arch}.attention.head_count", NumAttentionHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddFloat32($"{arch}.rope.freq_base", 1000000.0f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.expert_used_count", ExpertsUsed);
        w.AddUInt32($"{arch}.attention.key_length", (uint)headDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)headDim);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)(headDim / 2));
        w.AddUInt32($"{arch}.expert_count", (uint)numExperts);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)moeInter);
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
            AddW(w, name, [inK + extra, outM], f, q8Weights, q8AsF32);
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
            AddW(w, name, [inK + extra, mid, numExperts], f, q8Weights, q8AsF32);
        }

        int nq = NumAttentionHeads * headDim, nkv = NumKvHeads * headDim;
        float qkAmp = headDim == HeadDim ? 1.0f : MathF.Sqrt((float)HeadDim / headDim);   // keep attention logits at the same scale
        int shared = moeInter * SharedExperts;

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
            Mat($"{p}.attn_q.weight", HiddenSize, nq, qkAmp);
            Vec($"{p}.attn_q.bias", nq, 0f, 0.8f);
            Mat($"{p}.attn_k.weight", HiddenSize, nkv, qkAmp);
            Vec($"{p}.attn_k.bias", nkv, 0f, 0.8f);
            Mat($"{p}.attn_v.weight", HiddenSize, nkv);
            Vec($"{p}.attn_v.bias", nkv, 0f, 0.3f);
            Mat($"{p}.attn_output.weight", nq, HiddenSize);
            if (withQkNorm)
            {
                Vec($"{p}.attn_q_norm.weight", headDim, 1f);
                Vec($"{p}.attn_k_norm.weight", headDim, 1f);
            }
            Vec($"{p}.post_attention_norm.weight", HiddenSize, 1f);
            if (!isNextn && l < LeadingDense)
            {
                Mat($"{p}.ffn_gate.weight", HiddenSize, denseInter);
                Mat($"{p}.ffn_up.weight", HiddenSize, denseInter);
                Mat($"{p}.ffn_down.weight", denseInter, HiddenSize);
            }
            else
            {
                // Router + selection bias large enough that routing is peaky, the bias flips selections, and the sigmoid scores are not all ~0.5.
                Mat($"{p}.ffn_gate_inp.weight", HiddenSize, numExperts, 0.6f);
                Vec($"{p}.exp_probs_b.bias", numExperts, 0f, 0.3f);
                Bank($"{p}.ffn_gate_exps.weight", HiddenSize, moeInter);
                Bank($"{p}.ffn_up_exps.weight", HiddenSize, moeInter);
                Bank($"{p}.ffn_down_exps.weight", moeInter, HiddenSize);
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

    private static void AddW(GgufWriter w, string name, int[] dims, float[] values, bool q8, bool asF32)
    {
        if (!q8) { Add(w, name, dims, values); return; }
        if (dims[0] % 32 != 0) throw new ArgumentException($"{name}: Q8_0 needs a reduction length multiple of 32, got {dims[0]}.");
        var bytes = new byte[values.Length / 32 * 34];
        for (int b = 0; b < values.Length / 32; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(values[b * 32 + i]));
            float d = amax / 127f, inv = d != 0 ? 1f / d : 0f;
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 34, 2), (Half)d);
            for (int i = 0; i < 32; i++) bytes[b * 34 + 2 + i] = (byte)(sbyte)MathF.Round(values[b * 32 + i] * inv);
        }
        if (asF32)
        {
            // The same quantised values widened to F32: the exact-weight oracle for the Q8_0 file (activation precision is the only difference).
            var deq = new float[values.Length];
            for (int b = 0; b < values.Length / 32; b++)
            {
                float d = (float)BitConverter.ToHalf(bytes, b * 34);
                for (int i = 0; i < 32; i++) deq[b * 32 + i] = d * (sbyte)bytes[b * 34 + 2 + i];
            }
            Add(w, name, dims, deq);
            return;
        }
        w.AddTensor(name, dims, (uint)QuantizationType.Q8_0, bytes);
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
