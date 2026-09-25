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
    /// <param name="blockCount">
    /// Overrides <see cref="BlockCount"/>'s default trunk depth (2). Issue #291's CPU/GPU
    /// partial-offload split regression coverage needs a trunk deep enough that a split boundary
    /// can land with BOTH a GDN and a full-attention layer on EACH side of the boundary (a 2-layer
    /// trunk can only ever put one layer on each side) — every other existing caller keeps the
    /// default 2-layer trunk unchanged.
    /// </param>
    /// <param name="contextLength">
    /// Overrides <see cref="ContextLength"/>'s default of 16. Issue #435 needs a fixture whose
    /// context exceeds a backend's all-row-logits bound (also 16), so that a test can assert the
    /// behaviour on BOTH sides of that bound; every other caller keeps the default.
    /// </param>
    /// <param name="gdnKeyHeads">
    /// Overrides the GDN key-head count (<c>ssm.group_count</c>, default 1). Issue #479 needs
    /// <c>NKHead &gt;= 2</c>: with a single key head the PrismML <c>gdn_v_grouped</c> tiled→grouped
    /// value-head permute is the identity, so a test on the default fixture cannot tell a missing
    /// permute from a correct one.
    /// </param>
    /// <param name="gdnValueHeads">
    /// Overrides the GDN value-head count (<c>ssm.time_step_rank</c>, default 2). Must be a multiple
    /// of <paramref name="gdnKeyHeads"/>.
    /// </param>
    /// <param name="pq2_0Projections">
    /// Issue #482: when <see langword="true"/>, widens the fixture so every projection input width
    /// is a multiple of 128 (see <see cref="Dims.Pq2_0"/>) and stores the trunk projections,
    /// <c>token_embd</c> and <c>output</c> as random ternary PQ2_0 — the Bonsai 2 type mix, so the
    /// CUDA PQ2_0 GEMV paths (single- and multi-column) are exercised end to end. <c>ssm_alpha</c> /
    /// <c>ssm_beta</c> and the MTP block stay F32. Default <see langword="false"/> leaves the fixture
    /// byte-identical to before.
    /// </param>
    /// <param name="q8_0MtpHead">
    /// Issue #486: when <see langword="true"/>, stores the MTP block's projections (attention Q/K/V/O,
    /// FFN gate/up/down and <c>nextn.eh_proj</c>) as Q8_0 — Bonsai 2's MTP head type — and widens the
    /// fixture to <see cref="Dims.Pq2_0"/> so every input width is a multiple of 32. Exercises the CUDA
    /// Q8_0 MTP GEMV paths (single-row draft/absorb and the multi-column batched absorb). The head-local
    /// embedding / lm_head stay F32. Default <see langword="false"/> leaves the fixture unchanged.
    /// </param>
    public static byte[] Build(uint seed = 0xC0FFEEu, bool withMtp = true, bool mtpHasOwnHeadTensors = true,
        int fullAttnInterval = FullAttnInterval, int blockCount = BlockCount,
        int contextLength = ContextLength, int gdnKeyHeads = NKHead, int gdnValueHeads = NVHead,
        bool pq2_0Projections = false, bool q8_0MtpHead = false)
    {
        if (gdnKeyHeads <= 0 || gdnValueHeads <= 0 || gdnValueHeads % gdnKeyHeads != 0)
            throw new ArgumentException(
                $"GDN value heads ({gdnValueHeads}) must be a positive multiple of key heads ({gdnKeyHeads}).");

        var w = new GgufWriter();
        var rng = new SyntheticGemma4Gguf.Xorshift(seed);
        const string arch = "qwen35";
        Dims d = pq2_0Projections || q8_0MtpHead ? Dims.Pq2_0 : Dims.Default;

        // ── Metadata ──────────────────────────────────────────────────────
        w.AddString("general.architecture", arch);
        w.AddString("general.name", "synthetic-qwen35-hybriddense-mtp-tiny");
        w.AddUInt32("general.alignment", 32);

        int rawBlockCount = withMtp ? blockCount + 1 : blockCount;
        w.AddUInt32($"{arch}.context_length", (uint)contextLength);
        w.AddUInt32($"{arch}.embedding_length", (uint)d.Hidden);
        w.AddUInt32($"{arch}.block_count", (uint)rawBlockCount);
        w.AddUInt32($"{arch}.attention.head_count", NumAttentionHeads);
        w.AddUInt32($"{arch}.attention.head_count_kv", NumKvHeads);
        w.AddUInt32($"{arch}.attention.key_length", (uint)d.HeadDim);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)d.Ffn);
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
        w.AddUInt32($"{arch}.ssm.inner_size", (uint)(gdnValueHeads * d.DState));
        w.AddUInt32($"{arch}.ssm.state_size", (uint)d.DState);
        w.AddUInt32($"{arch}.ssm.time_step_rank", (uint)gdnValueHeads);
        w.AddUInt32($"{arch}.ssm.group_count", (uint)gdnKeyHeads);
        w.AddUInt32($"{arch}.ssm.conv_kernel", DConv);

        AddTokenizer(w);

        // ── Tensors ───────────────────────────────────────────────────────
        // token_embd [ne0=hidden, ne1=vocab]; separate output.weight (untied, exercises the
        // trunk-fallback path distinctly from the tied-embedding case the MoE fixture covers).
        // With pq2_0Projections the trunk's projections, token_embd and output are PQ2_0 (as in
        // Bonsai 2); ssm_alpha/ssm_beta and the MTP block stay F32 (BF16 / Q8_0 in Bonsai 2).
        bool pq2 = pq2_0Projections;
        AddProjection(w, rng, "token_embd.weight", inK: d.Hidden, outM: VocabSize, 0.05f, pq2);
        AddProjection(w, rng, "output.weight", inK: d.Hidden, outM: VocabSize, 0.05f, pq2);
        AddNorm(w, rng, "output_norm.weight", d.Hidden);

        for (int i = 0; i < blockCount; i++)
        {
            bool fullAttn = (i + 1) % fullAttnInterval == 0;
            string p = $"blk.{i}";

            AddNorm(w, rng, $"{p}.attn_norm.weight", d.Hidden);
            AddNorm(w, rng, $"{p}.post_attention_norm.weight", d.Hidden);

            if (fullAttn)
                AddFullAttnLayer(w, rng, p, d, pq2);
            else
                AddGdnLayer(w, rng, p, gdnKeyHeads, gdnValueHeads, d, pq2);

            AddDenseFfnLayer(w, rng, p, d, pq2);
        }

        if (withMtp)
        {
            string mp = $"blk.{blockCount}";
            AddNorm(w, rng, $"{mp}.attn_norm.weight", d.Hidden);
            AddNorm(w, rng, $"{mp}.post_attention_norm.weight", d.Hidden);
            AddFullAttnLayer(w, rng, mp, d, pq2: false, q8: q8_0MtpHead); // MTP block is always full-attention
            AddDenseFfnLayer(w, rng, mp, d, pq2: false, q8: q8_0MtpHead);

            AddProjection(w, rng, $"{mp}.nextn.eh_proj.weight", inK: 2 * d.Hidden, outM: d.Hidden, 0.05f,
                pq2: false, q8: q8_0MtpHead);
            AddNorm(w, rng, $"{mp}.nextn.enorm.weight", d.Hidden);
            AddNorm(w, rng, $"{mp}.nextn.hnorm.weight", d.Hidden);

            if (mtpHasOwnHeadTensors)
            {
                AddMatrixF32(w, rng, $"{mp}.nextn.embed_tokens.weight", inK: d.Hidden, outM: VocabSize, 0.05f);
                AddMatrixF32(w, rng, $"{mp}.nextn.shared_head_head.weight", inK: d.Hidden, outM: VocabSize, 0.05f);
                AddNorm(w, rng, $"{mp}.nextn.shared_head_norm.weight", d.Hidden);
            }
        }

        return w.Build();
    }

    /// <summary>Writes the synthetic fixture to <paramref name="path"/>.</summary>
    public static string Write(string path, uint seed = 0xC0FFEEu, bool withMtp = true, bool mtpHasOwnHeadTensors = true,
        int fullAttnInterval = FullAttnInterval, int blockCount = BlockCount,
        int contextLength = ContextLength, int gdnKeyHeads = NKHead, int gdnValueHeads = NVHead,
        bool pq2_0Projections = false, bool q8_0MtpHead = false)
    {
        File.WriteAllBytes(path, Build(seed, withMtp, mtpHasOwnHeadTensors, fullAttnInterval, blockCount, contextLength,
            gdnKeyHeads, gdnValueHeads, pq2_0Projections, q8_0MtpHead));
        return path;
    }

    /// <summary>
    /// Tensor dimensions. <see cref="Default"/> is the historical tiny fixture (every dim &lt; 128).
    /// <see cref="Pq2_0"/> (issue #482) makes every projection input width a multiple of 128 so the
    /// projections can be stored as PQ2_0: hidden 128, ffn 256, head_dim 256 and GDN state 128 (the
    /// last two are Bonsai 2's real values), giving input widths 128 / 256 / 512.
    /// </summary>
    private readonly record struct Dims(int Hidden, int Ffn, int HeadDim, int DState)
    {
        public static Dims Default => new(
            SyntheticQwen35HybridDenseMtpGguf.HiddenSize, SyntheticQwen35HybridDenseMtpGguf.IntermediateSize,
            SyntheticQwen35HybridDenseMtpGguf.HeadDim, SyntheticQwen35HybridDenseMtpGguf.DState);
        public static Dims Pq2_0 => new(128, 256, 256, 128);
    }

    /// <summary>
    /// Emits a projection matrix as F32, or — when <paramref name="pq2"/> is set — as a random
    /// ternary PQ2_0 matrix (<paramref name="inK"/> must then be a multiple of 128).
    /// </summary>
    private static void AddProjection(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name,
        int inK, int outM, float scale, bool pq2, bool q8 = false)
    {
        if (q8)
        {
            AddMatrixQ8_0(w, rng, name, inK, outM, scale);
            return;
        }
        if (!pq2)
        {
            AddMatrixF32(w, rng, name, inK, outM, scale);
            return;
        }
        if (inK % 128 != 0)
            throw new ArgumentException($"PQ2_0 tensor {name} needs an input width that is a multiple of 128, got {inK}.");

        // dotLLM's interleaved on-disk PQ2_0 layout: per 128-element group an fp16 scale, then 32
        // bytes holding elements 4b..4b+3 at ascending 2-bit offsets, code = value + 1.
        int groups = inK / 128;
        byte[] buf = new byte[(long)outM * groups * 34];
        for (long g = 0; g < (long)outM * groups; g++)
        {
            long gb = g * 34;
            // Per-group scale of the same order as the F32 fixture's weights, with some spread.
            float s = scale * (0.5f + (rng.NextUInt() & 0xFFFF) / 65536f);
            ushort bits = BitConverter.HalfToUInt16Bits((Half)s);
            buf[gb] = (byte)bits;
            buf[gb + 1] = (byte)(bits >> 8);
            for (int b = 0; b < 32; b++)
            {
                uint r = rng.NextUInt();
                int packed = 0;
                for (int i = 0; i < 4; i++)
                    packed |= (int)((r >> (8 * i)) % 3u) << (2 * i);
                buf[gb + 2 + b] = (byte)packed;
            }
        }
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.PQ2_0, buf);
    }

    /// <summary>
    /// GDN (Gated DeltaNet) token-mixing tensors. Note: NO attn_output.weight — GDN layers have
    /// no attention output projection.
    /// </summary>
    private static void AddGdnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p,
        int nKHead, int nVHead, Dims d, bool pq2)
    {
        int gdnKDim = nKHead * d.DState;
        int gdnVDim = nVHead * d.DState;
        int qkvOut = 2 * gdnKDim + gdnVDim;
        int convDim = qkvOut; // (2*NKHead + NVHead) * DState

        AddProjection(w, rng, $"{p}.attn_qkv.weight", inK: d.Hidden, outM: qkvOut, 0.05f, pq2);
        AddProjection(w, rng, $"{p}.attn_gate.weight", inK: d.Hidden, outM: gdnVDim, 0.05f, pq2);

        // ssm_a: per-V-head decay base — must be negative (exp(a·dt) < 1).
        var a = new float[nVHead];
        for (int i = 0; i < nVHead; i++) a[i] = -0.5f + rng.NextSigned(0.25f);
        AddF32Tensor(w, $"{p}.ssm_a", [nVHead], a);

        AddMatrixF32(w, rng, $"{p}.ssm_alpha.weight", inK: d.Hidden, outM: nVHead, 0.05f);
        AddMatrixF32(w, rng, $"{p}.ssm_beta.weight", inK: d.Hidden, outM: nVHead, 0.05f);

        var conv = new float[DConv * convDim];
        for (int i = 0; i < conv.Length; i++) conv[i] = rng.NextSigned(0.1f);
        AddF32Tensor(w, $"{p}.ssm_conv1d.weight", [DConv, convDim], conv);

        var dtBias = new float[nVHead];
        for (int i = 0; i < nVHead; i++) dtBias[i] = rng.NextSigned(0.1f);
        AddF32Tensor(w, $"{p}.ssm_dt.bias", [nVHead], dtBias);

        AddNorm(w, rng, $"{p}.ssm_norm.weight", d.DState);
        AddProjection(w, rng, $"{p}.ssm_out.weight", inK: gdnVDim, outM: d.Hidden, 0.05f, pq2);
    }

    /// <summary>Full-attention tensors: fused Q+Gate projection + QK-norm, per qwen35(moe).</summary>
    private static void AddFullAttnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p, Dims d, bool pq2,
        bool q8 = false)
    {
        int qOut = 2 * NumAttentionHeads * d.HeadDim; // Q + Gate fused per head
        int kvOut = NumKvHeads * d.HeadDim;
        int oIn = NumAttentionHeads * d.HeadDim;

        AddProjection(w, rng, $"{p}.attn_q.weight", inK: d.Hidden, outM: qOut, 0.05f, pq2, q8);
        AddProjection(w, rng, $"{p}.attn_k.weight", inK: d.Hidden, outM: kvOut, 0.05f, pq2, q8);
        AddProjection(w, rng, $"{p}.attn_v.weight", inK: d.Hidden, outM: kvOut, 0.05f, pq2, q8);
        AddProjection(w, rng, $"{p}.attn_output.weight", inK: oIn, outM: d.Hidden, 0.05f, pq2, q8);
        AddNorm(w, rng, $"{p}.attn_q_norm.weight", d.HeadDim);
        AddNorm(w, rng, $"{p}.attn_k_norm.weight", d.HeadDim);
    }

    /// <summary>Dense SwiGLU FFN — standard ffn_gate/up/down naming (no MoE routing).</summary>
    private static void AddDenseFfnLayer(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string p, Dims d, bool pq2,
        bool q8 = false)
    {
        AddProjection(w, rng, $"{p}.ffn_gate.weight", inK: d.Hidden, outM: d.Ffn, 0.05f, pq2, q8);
        AddProjection(w, rng, $"{p}.ffn_up.weight", inK: d.Hidden, outM: d.Ffn, 0.05f, pq2, q8);
        AddProjection(w, rng, $"{p}.ffn_down.weight", inK: d.Ffn, outM: d.Hidden, 0.05f, pq2, q8);
    }

    /// <summary>
    /// Random matrix like <see cref="AddMatrixF32"/> (same values, same RNG consumption), stored as
    /// Q8_0: per 32 elements an fp16 scale <c>max|v| / 127</c> and 32 int8 <c>round(v / scale)</c>.
    /// </summary>
    private static void AddMatrixQ8_0(GgufWriter w, SyntheticGemma4Gguf.Xorshift rng, string name,
        int inK, int outM, float scale)
    {
        if (inK % 32 != 0)
            throw new ArgumentException($"Q8_0 tensor {name} needs an input width that is a multiple of 32, got {inK}.");
        var f = new float[(long)inK * outM];
        for (long i = 0; i < f.LongLength; i++) f[i] = rng.NextSigned(scale);
        long blocks = f.LongLength / 32;
        byte[] buf = new byte[blocks * 34];
        for (long b = 0; b < blocks; b++)
        {
            float amax = 0f;
            for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(f[b * 32 + j]));
            Half dh = (Half)(amax / 127f);
            float dq = (float)dh;
            ushort bits = BitConverter.HalfToUInt16Bits(dh);
            buf[b * 34] = (byte)bits;
            buf[b * 34 + 1] = (byte)(bits >> 8);
            for (int j = 0; j < 32; j++)
            {
                int q = dq == 0f ? 0 : (int)MathF.Round(f[b * 32 + j] / dq);
                buf[b * 34 + 2 + j] = (byte)(sbyte)Math.Clamp(q, -127, 127);
            }
        }
        w.AddTensor(name, [inK, outM], (uint)QuantizationType.Q8_0, buf);
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
