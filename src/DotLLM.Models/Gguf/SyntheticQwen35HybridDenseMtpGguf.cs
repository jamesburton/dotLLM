using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Deterministic synthetic <c>qwen35</c> (Qwen3HybridDense — Gated-DeltaNet + dense SwiGLU FFN)
/// GGUF fixture builder, WITH a trailing Multi-Token Prediction (MTP / "NextN") head appended
/// (issue #253). Emits a TINY but architecturally complete checkpoint: <c>BlockCount</c> trunk
/// layers (layer 0 GDN, layer 1 full-attention, per <see cref="FullAttnInterval"/>) plus one
/// extra full-attention MTP block at raw index <c>BlockCount</c>, exactly mirroring how
/// llama.cpp's <c>convert_hf_to_gguf.py</c> (PR ggml-org/llama.cpp#22673) appends the MTP
/// block(s) after the trunk: <c>block_count = num_hidden_layers + mtp_num_hidden_layers</c>.
/// </summary>
/// <remarks>
/// <para><b>Why it exists.</b> There is no locally-cached real Qwen3.6-MTP-GGUF fixture (checked
/// <c>~/.dotllm/test-cache/</c> and the HF hub cache — see issue #253's fixture-availability
/// note); this fixture lets the MTP GGUF loader and the MTP head's forward math be exercised
/// end-to-end through the real <c>GgufModelConfigExtractor</c> / <c>Qwen3HybridDenseTransformerModel.LoadFromGguf</c>
/// path without a multi-GB download. Real Bonsai/Qwen3.6 files are tens of GB; this fixture is a
/// few hundred KB.</para>
/// <para><b>Determinism.</b> Weights come from the same seeded xorshift PRNG as
/// <see cref="SyntheticGemma4Gguf"/> and <see cref="SyntheticQwen35MoeGguf"/> — no
/// <c>Random</c>/time. All tensors are F32 (forward-path coverage, not quantization coverage).</para>
/// <para><b>Sibling without MTP.</b> <see cref="Build"/> takes a <c>withMtp</c> flag so the exact
/// same trunk can be built with or without the MTP head — the zero-behavior-change contract for
/// non-MTP checkpoints is proven by loading the <c>withMtp: false</c> variant and confirming
/// <c>SupportsMtp</c> is false and every other field matches.</para>
/// </remarks>
public static class SyntheticQwen35HybridDenseMtpGguf
{
    // Compact but architecturally valid shapes. Internal (not private) so tests can reference
    // the exact dims when constructing KV-caches / asserting shapes against the fixture.
    internal const int VocabSize = 12;
    internal const int HiddenSize = 32;
    internal const int NumAttentionHeads = 2;
    internal const int NumKvHeads = 1;          // GQA repeat factor = 2
    internal const int HeadDim = 16;
    internal const int RopeDim = 8;             // partial-rotary slice < HeadDim
    internal const int ContextLength = 16;
    internal const int IntermediateSize = 24;
    internal const int BlockCount = 2;          // trunk: layer 0 GDN, layer 1 full-attn
    internal const int FullAttnInterval = 2;    // (i+1) % 2 == 0 → layer 1 is attention

    // GDN config ({arch}.ssm.* keys, qwen35/qwen35moe shared semantics).
    private const int NVHead = 2;              // ssm.time_step_rank
    private const int NKHead = 1;              // ssm.group_count
    private const int DState = 8;              // ssm.state_size
    private const int DConv = 4;               // ssm.conv_kernel
    private const int DInner = NVHead * DState; // ssm.inner_size

    /// <summary>Builds the synthetic <c>qwen35</c> fixture, optionally with a trailing MTP head.</summary>
    /// <param name="seed">Xorshift PRNG seed — deterministic weights, no <c>Random</c>/time.</param>
    /// <param name="withMtp">
    /// When <see langword="true"/> (default), appends one MTP block at raw block index
    /// <see cref="BlockCount"/> and sets <c>qwen35.nextn_predict_layers = 1</c>. When
    /// <see langword="false"/>, produces a plain trunk-only <c>qwen35</c> GGUF — the "no MTP head"
    /// control fixture for the zero-behavior-change assertion.
    /// </param>
    /// <param name="mtpHasOwnHeadTensors">
    /// When <see langword="true"/> (default), the MTP block also carries its own
    /// <c>nextn.embed_tokens</c> / <c>nextn.shared_head_head</c> / <c>nextn.shared_head_norm</c>
    /// tensors. When <see langword="false"/>, those three are omitted so the loader exercises the
    /// "fall back to the trunk's token_embd/output/output_norm" path. Ignored when
    /// <paramref name="withMtp"/> is <see langword="false"/>.
    /// </param>
    /// <param name="fullAttnInterval">
    /// Overrides <see cref="FullAttnInterval"/>'s default mixed GDN+attention layout. Pass 1 for an
    /// all-full-attention trunk (no GDN layer at all) — useful for isolating a test from the
    /// separate, pre-existing "speculative decoding has no rollback for recurrent trunk state"
    /// limitation (see <c>MtpSpeculativeDecoder</c>'s remarks).
    /// </param>
    public static byte[] Build(uint seed = 0xC0FFEEu, bool withMtp = true, bool mtpHasOwnHeadTensors = true,
        int fullAttnInterval = FullAttnInterval)
    {
        var w = new GgufWriter();
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        const string arch = "qwen35";

        // ── Metadata ──────────────────────────────────────────────────────
        w.AddString("general.architecture", arch);
        w.AddString("general.name", "synthetic-qwen35-hybriddense-mtp-tiny");
        w.AddUInt32("general.alignment", 32);

        int rawBlockCount = withMtp ? BlockCount + 1 : BlockCount;
        w.AddUInt32($"{arch}.context_length", ContextLength);
        w.AddUInt32($"{arch}.embedding_length", HiddenSize);
        w.AddUInt32($"{arch}.block_count", (uint)rawBlockCount);
        w.AddUInt32($"{arch}.attention.head_count", NumAttentionHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddUInt32($"{arch}.attention.key_length", HeadDim);
        w.AddUInt32($"{arch}.feed_forward_length", IntermediateSize);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        w.AddUInt32($"{arch}.vocab_size", VocabSize);
        w.AddFloat32($"{arch}.rope.freq_base", 10000.0f);
        w.AddUInt32($"{arch}.rope.dimension_count", RopeDim);

        // Hybrid layout: trunk layer i is full attention when (i+1) % full_attention_interval == 0.
        // fullAttnInterval defaults to the standard fixture shape (mixed GDN+attention); callers that
        // need an all-full-attention trunk (e.g. to isolate a test from the separate, pre-existing
        // "speculative decoding + recurrent trunk state has no rollback" limitation — see
        // MtpSpeculativeDecoder's remarks / issue #253's CUDA follow-up notes) pass 1.
        w.AddUInt32($"{arch}.full_attention_interval", (uint)fullAttnInterval);

        if (withMtp)
            w.AddUInt32($"{arch}.nextn_predict_layers", 1);

        // GDN ({arch}.ssm.* reused with qwen35 semantics — see TryExtractGdnConfig).
        w.AddUInt32($"{arch}.ssm.inner_size", DInner);
        w.AddUInt32($"{arch}.ssm.state_size", DState);
        w.AddUInt32($"{arch}.ssm.time_step_rank", NVHead);
        w.AddUInt32($"{arch}.ssm.group_count", NKHead);
        w.AddUInt32($"{arch}.ssm.conv_kernel", DConv);

        AddTokenizer(w);

        // ── Tensors ───────────────────────────────────────────────────────
        // token_embd [ne0=hidden, ne1=vocab]; separate output.weight (untied, exercises the
        // trunk-fallback path distinctly from the tied-embedding case the MoE fixture covers).
        AddMatrixF32(w, rng, "token_embd.weight", inK: HiddenSize, outM: VocabSize, 0.05f);
        AddMatrixF32(w, rng, "output.weight", inK: HiddenSize, outM: VocabSize, 0.05f);
        AddNorm(w, rng, "output_norm.weight", HiddenSize);

        for (int i = 0; i < BlockCount; i++)
        {
            bool fullAttn = (i + 1) % fullAttnInterval == 0;
            string p = $"blk.{i}";

            AddNorm(w, rng, $"{p}.attn_norm.weight", HiddenSize);
            AddNorm(w, rng, $"{p}.post_attention_norm.weight", HiddenSize);

            if (fullAttn)
                AddFullAttnLayer(w, rng, p);
            else
                AddGdnLayer(w, rng, p);

            AddDenseFfnLayer(w, rng, p);
        }

        if (withMtp)
        {
            string mp = $"blk.{BlockCount}";
            AddNorm(w, rng, $"{mp}.attn_norm.weight", HiddenSize);
            AddNorm(w, rng, $"{mp}.post_attention_norm.weight", HiddenSize);
            AddFullAttnLayer(w, rng, mp); // MTP block is always full-attention
            AddDenseFfnLayer(w, rng, mp);

            AddMatrixF32(w, rng, $"{mp}.nextn.eh_proj.weight", inK: 2 * HiddenSize, outM: HiddenSize, 0.05f);
            AddNorm(w, rng, $"{mp}.nextn.enorm.weight", HiddenSize);
            AddNorm(w, rng, $"{mp}.nextn.hnorm.weight", HiddenSize);

            if (mtpHasOwnHeadTensors)
            {
                AddMatrixF32(w, rng, $"{mp}.nextn.embed_tokens.weight", inK: HiddenSize, outM: VocabSize, 0.05f);
                AddMatrixF32(w, rng, $"{mp}.nextn.shared_head_head.weight", inK: HiddenSize, outM: VocabSize, 0.05f);
                AddNorm(w, rng, $"{mp}.nextn.shared_head_norm.weight", HiddenSize);
            }
        }

        return w.Build();
    }

    /// <summary>Writes the synthetic fixture to <paramref name="path"/>.</summary>
    public static string Write(string path, uint seed = 0xC0FFEEu, bool withMtp = true, bool mtpHasOwnHeadTensors = true,
        int fullAttnInterval = FullAttnInterval)
    {
        File.WriteAllBytes(path, Build(seed, withMtp, mtpHasOwnHeadTensors, fullAttnInterval));
        return path;
    }

    /// <summary>
    /// GDN (Gated DeltaNet) token-mixing tensors. Note: NO attn_output.weight — GDN layers have
    /// no attention output projection.
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

    /// <summary>Full-attention tensors: fused Q+Gate projection + QK-norm, per qwen35(moe).</summary>
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

    /// <summary>Dense SwiGLU FFN — standard ffn_gate/up/down naming (no MoE routing).</summary>
    private static void AddDenseFfnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p)
    {
        AddMatrixF32(w, rng, $"{p}.ffn_gate.weight", inK: HiddenSize, outM: IntermediateSize, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ffn_up.weight", inK: HiddenSize, outM: IntermediateSize, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ffn_down.weight", inK: IntermediateSize, outM: HiddenSize, 0.05f);
    }

    // ──────────────────── Tensor emit helpers (all F32) ────────────────────

    private static void AddMatrixF32(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name,
        int inK, int outM, float scale)
    {
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.LongLength; i++) f[i] = rng.NextSigned(scale);
        AddF32Tensor(w, name, [inK, outM], f);
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
