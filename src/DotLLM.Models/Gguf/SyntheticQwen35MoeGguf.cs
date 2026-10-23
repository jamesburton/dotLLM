using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Deterministic synthetic <c>qwen35moe</c> (Qwen3.6 Gated-DeltaNet + sparse-MoE hybrid)
/// GGUF fixture builder. Emits a TINY but architecturally complete checkpoint: layer 0 is a
/// Gated DeltaNet (linear-attention) layer with NO <c>attn_output.weight</c>, layer 1 is a
/// full-attention layer with the fused Q+Gate projection, and every layer carries a sparse
/// MoE FFN with a Qwen1.5-style shared expert + per-token sigmoid gate. The produced bytes
/// load through the normal <c>ModelLoader.LoadFromGguf</c> /
/// <c>ModelLoader.CreateCpuModelFromGguf</c> path.
/// </summary>
/// <remarks>
/// <para><b>Why it exists.</b> The hybrid layout is exactly what discriminates the CPU
/// dispatch regression (issue #135): routing a <c>qwen35moe</c> GGUF to the plain
/// <c>TransformerModel</c> loader fails with "blk.0.attn_output.weight not present" because
/// GDN layers have no attention output projection. Real Qwen3.6 files are tens of GB; this
/// fixture is a few hundred KB.</para>
/// <para><b>Determinism.</b> Weights come from the same seeded xorshift PRNG as
/// <see cref="SyntheticGemma4Gguf"/> — no <c>Random</c>/time. All tensors are F32 (dispatch
/// and forward-path coverage, not quantization coverage).</para>
/// </remarks>
public static class SyntheticQwen35MoeGguf
{
    // Compact but architecturally valid shapes (mirrors the CPU unit-test fixture dims).
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;          // GQA repeat factor = 2
    private const int HeadDim = 16;
    private const int RopeDim = 8;             // partial-rotary slice < HeadDim
    private const int ContextLength = 8;
    private const int MoeIntermediate = 32;
    private const int SharedIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;
    private const int BlockCount = 2;          // layer 0 GDN, layer 1 full-attn
    private const int FullAttnInterval = 2;    // (i+1) % 2 == 0 → layer 1 is attention

    // GDN config ({arch}.ssm.* keys, qwen35moe semantics).
    private const int NVHead = 2;              // ssm.time_step_rank
    private const int NKHead = 1;              // ssm.group_count
    private const int DState = 8;              // ssm.state_size
    private const int DConv = 4;               // ssm.conv_kernel
    private const int DInner = NVHead * DState; // ssm.inner_size

    /// <summary>Builds the synthetic <c>qwen35moe</c> fixture to a byte array.</summary>
    public static byte[] Build(uint seed = 0xC0FFEEu)
    {
        var w = new GgufWriter();
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        const string arch = "qwen35moe";

        // ── Metadata ──────────────────────────────────────────────────────
        w.AddString("general.architecture", arch);
        w.AddString("general.name", "synthetic-qwen35moe-tiny");
        w.AddUInt32("general.alignment", 32);

        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.block_count", BlockCount);
        w.AddUInt32($"{arch}.attention.head_count", NumAttentionHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.vocab_size", VocabSize);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);

        // Hybrid layout: layer i is full attention when (i+1) % full_attention_interval == 0.
        w.AddUInt32($"{arch}.full_attention_interval", FullAttnInterval);

        // GDN ({arch}.ssm.* reused with qwen35moe semantics — see TryExtractGdnConfig).
        w.AddUInt32($"{arch}.ssm.inner_size", DInner);
        w.AddUInt32($"{arch}.ssm.state_size", DState);
        w.AddUInt32($"{arch}.ssm.time_step_rank", NVHead);
        w.AddUInt32($"{arch}.ssm.group_count", NKHead);
        w.AddUInt32($"{arch}.ssm.conv_kernel", DConv);

        // Sparse MoE + implicit-count shared expert (qwen35moe convention: only the
        // shared intermediate width is stored; count is implicit 1 + sigmoid gate).
        w.AddUInt32($"{arch}.expert_count", NumExperts);
        w.AddUInt32($"{arch}.expert_used_count", NumExpertsPerTok);
        w.AddUInt32($"{arch}.expert_feed_forward_length", MoeIntermediate);
        w.AddUInt32($"{arch}.expert_shared_feed_forward_length", SharedIntermediate);

        AddTokenizer(w);

        // ── Tensors ───────────────────────────────────────────────────────
        // token_embd [ne0=hidden, ne1=vocab]; output tied to token_embd (no output.weight).
        AddMatrixF32(w, rng, "token_embd.weight", inK: HiddenSize, outM: VocabSize, 0.05f);
        AddNorm(w, rng, "output_norm.weight", HiddenSize);

        for (int i = 0; i < BlockCount; i++)
        {
            bool fullAttn = (i + 1) % FullAttnInterval == 0;
            string p = $"blk.{i}";

            AddNorm(w, rng, $"{p}.attn_norm.weight", HiddenSize);
            AddNorm(w, rng, $"{p}.post_attention_norm.weight", HiddenSize);

            if (fullAttn)
                AddFullAttnLayer(w, rng, p);
            else
                AddGdnLayer(w, rng, p);

            AddMoeLayer(w, rng, p);
        }

        return w.Build();
    }

    /// <summary>Writes the synthetic <c>qwen35moe</c> fixture to <paramref name="path"/>.</summary>
    public static string Write(string path, uint seed = 0xC0FFEEu)
    {
        File.WriteAllBytes(path, Build(seed));
        return path;
    }

    /// <summary>
    /// GDN (Gated DeltaNet) token-mixing tensors. Note: NO attn_output.weight — that absence
    /// is what discriminates the hybrid loader from the plain TransformerModel loader.
    /// </summary>
    private static void AddGdnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p)
    {
        int gdnKDim = NKHead * DState;
        int gdnVDim = NVHead * DState;
        int qkvOut = 2 * gdnKDim + gdnVDim;
        int convDim = qkvOut; // (2*NKHead + NVHead) * DState

        AddMatrixF32(w, rng, $"{p}.attn_qkv.weight", inK: HiddenSize, outM: qkvOut, 0.05f);
        AddMatrixF32(w, rng, $"{p}.attn_gate.weight", inK: HiddenSize, outM: gdnVDim, 0.05f);

        // ssm_a: per-V-head decay base — must be negative (exp(a·dt) < 1).
        var a = new float[NVHead];
        for (int i = 0; i < NVHead; i++) a[i] = -0.5f + rng.NextSigned(0.25f);
        AddF32Tensor(w, $"{p}.ssm_a", [NVHead], a);

        AddMatrixF32(w, rng, $"{p}.ssm_alpha.weight", inK: HiddenSize, outM: NVHead, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ssm_beta.weight", inK: HiddenSize, outM: NVHead, 0.05f);

        var conv = new float[DConv * convDim];
        for (int i = 0; i < conv.Length; i++) conv[i] = rng.NextSigned(0.1f);
        AddF32Tensor(w, $"{p}.ssm_conv1d.weight", [DConv, convDim], conv);

        var dtBias = new float[NVHead];
        for (int i = 0; i < NVHead; i++) dtBias[i] = rng.NextSigned(0.1f);
        AddF32Tensor(w, $"{p}.ssm_dt.bias", [NVHead], dtBias);

        AddNorm(w, rng, $"{p}.ssm_norm.weight", DState);
        AddMatrixF32(w, rng, $"{p}.ssm_out.weight", inK: gdnVDim, outM: HiddenSize, 0.05f);
    }

    /// <summary>Full-attention tensors: fused Q+Gate projection + QK-norm, per qwen35moe.</summary>
    private static void AddFullAttnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p)
    {
        int qOut = 2 * NumAttentionHeads * HeadDim; // Q + Gate fused per head
        int kvOut = NumKvHeads * HeadDim;
        int oIn = NumAttentionHeads * HeadDim;

        AddMatrixF32(w, rng, $"{p}.attn_q.weight", inK: HiddenSize, outM: qOut, 0.05f);
        AddMatrixF32(w, rng, $"{p}.attn_k.weight", inK: HiddenSize, outM: kvOut, 0.05f);
        AddMatrixF32(w, rng, $"{p}.attn_v.weight", inK: HiddenSize, outM: kvOut, 0.05f);
        AddMatrixF32(w, rng, $"{p}.attn_output.weight", inK: oIn, outM: HiddenSize, 0.05f);
        AddNorm(w, rng, $"{p}.attn_q_norm.weight", HeadDim);
        AddNorm(w, rng, $"{p}.attn_k_norm.weight", HeadDim);
    }

    /// <summary>Sparse MoE + shared expert + sigmoid shared-expert gate (all layers).</summary>
    private static void AddMoeLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p)
    {
        // Router [ne0=hidden, ne1=experts].
        AddMatrixF32(w, rng, $"{p}.ffn_gate_inp.weight", inK: HiddenSize, outM: NumExperts, 0.05f);

        // Routed experts [ne0=K, ne1=M, ne2=experts].
        AddExpertBankF32(w, rng, $"{p}.ffn_gate_exps.weight", inK: HiddenSize, midOut: MoeIntermediate, 0.05f);
        AddExpertBankF32(w, rng, $"{p}.ffn_up_exps.weight", inK: HiddenSize, midOut: MoeIntermediate, 0.05f);
        AddExpertBankF32(w, rng, $"{p}.ffn_down_exps.weight", inK: MoeIntermediate, midOut: HiddenSize, 0.05f);

        // Shared expert (implicit count 1) + per-token sigmoid gate.
        AddMatrixF32(w, rng, $"{p}.ffn_gate_shexp.weight", inK: HiddenSize, outM: SharedIntermediate, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ffn_up_shexp.weight", inK: HiddenSize, outM: SharedIntermediate, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ffn_down_shexp.weight", inK: SharedIntermediate, outM: HiddenSize, 0.05f);
        var shGate = new float[HiddenSize];
        for (int i = 0; i < shGate.Length; i++) shGate[i] = rng.NextSigned(0.05f);
        AddF32Tensor(w, $"{p}.ffn_gate_inp_shexp.weight", [HiddenSize], shGate);
    }

    // ──────────────────── Tensor emit helpers (all F32) ────────────────────

    private static void AddMatrixF32(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name,
        int inK, int outM, float scale)
    {
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.LongLength; i++) f[i] = rng.NextSigned(scale);
        AddF32Tensor(w, name, [inK, outM], f);
    }

    private static void AddExpertBankF32(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name,
        int inK, int midOut, float scale)
    {
        var f = new float[(long)inK * midOut * NumExperts];
        for (long i = 0; i < f.LongLength; i++) f[i] = rng.NextSigned(scale);
        AddF32Tensor(w, name, [inK, midOut, NumExperts], f);
    }

    private static void AddNorm(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name, int n)
    {
        var f = new float[n];
        for (int i = 0; i < n; i++) f[i] = 1.0f + rng.NextSigned(0.05f);
        AddF32Tensor(w, name, [n], f);
    }

    private static void AddF32Tensor(GgufWriter w, string name, int[] dims, float[] values)
        => w.AddTensor(name, dims, (uint)QuantizationType.F32,
            MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    private static void AddTokenizer(GgufWriter w)
    {
        w.AddString("tokenizer.ggml.model", "llama");
        var tokens = new string[VocabSize];
        var scores = new float[VocabSize];
        var types = new int[VocabSize];
        for (int i = 0; i < VocabSize; i++)
        {
            tokens[i] = i switch { 0 => "<unk>", 1 => "<bos>", 2 => "<eos>", _ => $"tok{i}" };
            scores[i] = 0f;
            types[i] = i switch { 0 => 2, 1 or 2 => 3, _ => 1 };
        }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddFloat32Array("tokenizer.ggml.scores", scores);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddUInt32("tokenizer.ggml.bos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 2);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);
    }
}
