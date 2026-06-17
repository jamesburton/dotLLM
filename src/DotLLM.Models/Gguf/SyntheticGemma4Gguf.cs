using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Deterministic synthetic <c>gemma4</c> (autoregressive) and <c>diffusion-gemma</c>
/// (masked-diffusion) GGUF fixture builder. Emits a TINY but architecturally-complete
/// checkpoint that exercises every feature of the real 26B-A4B (dual head dim, V-less
/// global layers, dual/partial RoPE, dual parallel dense+MoE FFN, fused gate_up expert
/// split, custom router + per-expert down scale, layer_output_scale, final-logit softcap,
/// and — for diffusion-gemma — enc_layer_output_scale, self-conditioning, canvas length,
/// and a mask token). The produced bytes load through the normal
/// <c>ModelLoader.LoadFromGguf</c> path on every backend (CPU/Vulkan/CUDA/HIP), so a 12 GB
/// T5500 can run + optimize kernels against the same artifact without the full 26B.
/// </summary>
/// <remarks>
/// <para><b>Why these dims.</b> The real-model quant mix puts Q4_K on the fused gate_up
/// experts and Q5_1 on the down experts. Those formats have a fixed block size (Q4_K =
/// 256 elements, Q5_1 = 32) and the loader computes per-expert row strides from the
/// tensor's K dimension, so the tiny config must keep every quantized row K-dimension a
/// multiple of its format's block size: <c>hidden = 256</c> (Q4_K gate_up rows, K = hidden),
/// <c>expert_ff = 32</c> (Q5_1 down rows, K = expert_ff), dense <c>ff = 64</c> and all
/// attention K dims multiples of 32 (Q8_0). This is the smallest block-valid config that
/// still exercises all four quant types; the design doc's illustrative <c>hidden = 64</c>
/// cannot carry a Q4_K row (64 % 256 != 0), so we bump to 256 and document it here.</para>
/// <para><b>Determinism.</b> Weights come from a seeded xorshift PRNG (no
/// <c>Random</c>/time), so a given <see cref="SyntheticGemma4Config"/> + seed always produces byte-identical
/// output (modulo the quantizer, which is itself deterministic).</para>
/// </remarks>
public static class SyntheticGemma4Gguf
{
    /// <summary>Tiny default size preset — fast regression + kernel smoke. ~hundreds of KB.</summary>
    public static SyntheticGemma4Config Tiny => new();

    /// <summary>
    /// Real-26B-like preset for catching head-dim-&gt;256 / partial-rope divergence WITHOUT the
    /// 26B. Mirrors the real model's distinguishing attention config that the Tiny/Bench presets
    /// do NOT exercise: <b>GlobalHeadDim = 512</b> (over the Vulkan attention shader's old 256
    /// bound) and a <b>full-head-dim rope.dimension_count</b> so the 0.25 partial-rotary factor
    /// actually fires (rotate 128 of 512 dims, freq denominator = 512). Sliding layers use
    /// head_dim 256 / 8 kv-heads, global layers head_dim 512 / 2 kv-heads + V-from-K — exactly
    /// the 26B's dual schedule, only with a tiny hidden/vocab/expert count so a single forward is
    /// milliseconds. The 16-head Q matches the 26B. Stays well under any memory budget.
    /// </summary>
    /// <remarks>
    /// Because <see cref="SyntheticGemma4Config.GlobalHeadDim"/> (512) differs from the
    /// <c>rope.dimension_count</c> the fixture emits, the partial-rope path is driven by the
    /// config extractor's hardcoded 0.25 factor over the full 512-dim head — the SAME code path
    /// as the real 26B. This is the fixture the CPU↔Vulkan partial-rope-denominator parity test
    /// runs on.
    /// </remarks>
    public static SyntheticGemma4Config Real26BLike => new()
    {
        BlockCount = 6,            // layers 0-4 sliding, layer 5 global (matches GlobalLayerStride 6)
        HiddenSize = 256,          // Q4_K gate_up rows need K = hidden % 256 == 0
        HeadCount = 16,            // 26B has 16 Q heads
        SlidingHeadDim = 256,      // 26B sliding head_dim
        GlobalHeadDim = 512,       // 26B global head_dim — EXERCISES head_dim > 256
        SlidingKvHeads = 8,        // 26B sliding kv-heads
        GlobalKvHeads = 2,         // 26B global kv-heads
        DenseFeedForward = 64,
        ExpertFeedForward = 32,
        ExpertCount = 8,
        ExpertUsedCount = 2,
        VocabSize = 256,
        ContextLength = 512,
    };

    /// <summary>
    /// Larger "bench" preset — compute-bound enough for cross-backend timing while staying
    /// far under 12 GB. Still block-valid (hidden 1024, expert_ff 128, dense ff 512).
    /// </summary>
    public static SyntheticGemma4Config Bench => new()
    {
        BlockCount = 8,
        HiddenSize = 1024,
        HeadCount = 8,
        SlidingHeadDim = 64,
        GlobalHeadDim = 128,
        SlidingKvHeads = 4,
        GlobalKvHeads = 2,
        DenseFeedForward = 512,
        ExpertFeedForward = 128,
        ExpertCount = 16,
        ExpertUsedCount = 4,
        VocabSize = 1024,
        ContextLength = 1024,
        CanvasLength = 32,
    };

    /// <summary>Builds the autoregressive <c>gemma4</c> fixture to a byte array.</summary>
    public static byte[] BuildGemma4(SyntheticGemma4Config? config = null, uint seed = 0xC0FFEEu)
        => Build(config ?? Tiny, diffusion: false, seed);

    /// <summary>Builds the masked-diffusion <c>diffusion-gemma</c> fixture to a byte array.</summary>
    public static byte[] BuildDiffusionGemma(SyntheticGemma4Config? config = null, uint seed = 0xC0FFEEu)
        => Build(config ?? Tiny, diffusion: true, seed);

    /// <summary>Writes the autoregressive <c>gemma4</c> fixture to <paramref name="path"/>.</summary>
    public static string WriteGemma4(string path, SyntheticGemma4Config? config = null, uint seed = 0xC0FFEEu)
    {
        File.WriteAllBytes(path, BuildGemma4(config, seed));
        return path;
    }

    /// <summary>Writes the masked-diffusion <c>diffusion-gemma</c> fixture to <paramref name="path"/>.</summary>
    public static string WriteDiffusionGemma(string path, SyntheticGemma4Config? config = null, uint seed = 0xC0FFEEu)
    {
        File.WriteAllBytes(path, BuildDiffusionGemma(config, seed));
        return path;
    }

    private static byte[] Build(SyntheticGemma4Config cfg, bool diffusion, uint seed)
    {
        cfg.Validate();
        string arch = diffusion ? "diffusion-gemma" : "gemma4";
        var w = new GgufWriter();
        var rng = new Xorshift(seed);

        // ── Metadata ──────────────────────────────────────────────────────
        w.AddString("general.architecture", arch);
        w.AddString("general.name", diffusion ? "synthetic-diffusion-gemma-tiny" : "synthetic-gemma4-tiny");
        w.AddUInt32("general.alignment", 32);

        w.AddUInt32($"{arch}.context_length", (uint)cfg.ContextLength);
        w.AddUInt32($"{arch}.embedding_length", (uint)cfg.HiddenSize);
        w.AddUInt32($"{arch}.block_count", (uint)cfg.BlockCount);
        w.AddUInt32($"{arch}.feed_forward_length", (uint)cfg.DenseFeedForward);
        w.AddUInt32($"{arch}.attention.head_count", (uint)cfg.HeadCount);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", cfg.NormEpsilon);
        w.AddUInt32($"{arch}.attention.sliding_window", (uint)cfg.SlidingWindow);
        w.AddUInt32($"{arch}.vocab_size", (uint)cfg.VocabSize);
        w.AddFloat32($"{arch}.final_logit_softcapping", cfg.FinalLogitSoftcap);

        // Per-layer arrays: sliding_window_pattern (1 = sliding, 0 = global) and
        // head_count_kv (sliding kv on sliding layers, global kv on global layers).
        var swPattern = new int[cfg.BlockCount];
        var kvArray = new int[cfg.BlockCount];
        for (int i = 0; i < cfg.BlockCount; i++)
        {
            bool sliding = !cfg.IsGlobalLayer(i);
            swPattern[i] = sliding ? 1 : 0;
            kvArray[i] = sliding ? cfg.SlidingKvHeads : cfg.GlobalKvHeads;
        }
        w.AddInt32Array($"{arch}.attention.sliding_window_pattern", swPattern);
        w.AddInt32Array($"{arch}.attention.head_count_kv", kvArray);

        // Dual head dim + dual RoPE (sliding = *_swa, global = bare keys).
        w.AddUInt32($"{arch}.attention.key_length", (uint)cfg.GlobalHeadDim);
        w.AddUInt32($"{arch}.attention.key_length_swa", (uint)cfg.SlidingHeadDim);
        w.AddUInt32($"{arch}.attention.value_length", (uint)cfg.GlobalHeadDim);
        w.AddUInt32($"{arch}.attention.value_length_swa", (uint)cfg.SlidingHeadDim);
        w.AddFloat32($"{arch}.rope.freq_base", cfg.GlobalRopeFreqBase);
        w.AddFloat32($"{arch}.rope.freq_base_swa", cfg.SlidingRopeFreqBase);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)cfg.GlobalHeadDim);        // full head, partial 0.25 in forward
        w.AddUInt32($"{arch}.rope.dimension_count_swa", (uint)cfg.SlidingHeadDim);   // full rotation

        // MoE.
        w.AddUInt32($"{arch}.expert_count", (uint)cfg.ExpertCount);
        w.AddUInt32($"{arch}.expert_used_count", (uint)cfg.ExpertUsedCount);
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)cfg.ExpertFeedForward);

        // Diffusion-only metadata.
        if (diffusion)
        {
            w.AddUInt32("diffusion.canvas_length", (uint)cfg.CanvasLength);
            w.AddBool($"{arch}.attention.causal", false);
            w.AddUInt32("tokenizer.ggml.mask_token_id", (uint)cfg.MaskTokenId);
        }

        // ── Tokenizer metadata (minimal — RAW token ids drive the forward) ──
        // SentencePiece "llama" model: simple token strings + scores + token_type.
        AddTokenizer(w, cfg);

        // ── Tensors ───────────────────────────────────────────────────────
        int H = cfg.HiddenSize;
        int vocab = cfg.VocabSize;

        // token_embd [K=hidden, M=vocab].
        AddMatrix(w, rng, "token_embd.weight", inK: H, outM: vocab, cfg.TokenEmbdQuant, 0.02f);

        for (int i = 0; i < cfg.BlockCount; i++)
            AddLayer(w, rng, cfg, i, diffusion);

        // output_norm [hidden] (Gemma ties output to token_embd → no output.weight).
        AddNorm(w, rng, "output_norm.weight", H);

        // Diffusion self-conditioning (model-level). n_ff = dense feed_forward_length.
        if (diffusion)
        {
            int ff = cfg.DenseFeedForward;
            AddNorm(w, rng, "self_cond_pre_norm.weight", H);
            AddMatrix(w, rng, "self_cond_gate.weight", inK: H, outM: ff, cfg.SelfCondQuant, 0.05f);
            AddMatrix(w, rng, "self_cond_up.weight", inK: H, outM: ff, cfg.SelfCondQuant, 0.05f);
            AddMatrix(w, rng, "self_cond_down.weight", inK: ff, outM: H, cfg.SelfCondQuant, 0.05f);
        }

        return w.Build();
    }

    private static void AddLayer(GgufWriter w, Xorshift rng, SyntheticGemma4Config cfg, int layer, bool diffusion)
    {
        string p = $"blk.{layer}";
        int H = cfg.HiddenSize;
        bool global = cfg.IsGlobalLayer(layer);
        int hd = global ? cfg.GlobalHeadDim : cfg.SlidingHeadDim;
        int kv = global ? cfg.GlobalKvHeads : cfg.SlidingKvHeads;
        int qOut = cfg.HeadCount * hd;
        int kvOut = kv * hd;
        int ff = cfg.DenseFeedForward;
        int ie = cfg.ExpertFeedForward;
        int e = cfg.ExpertCount;

        AddNorm(w, rng, $"{p}.attn_norm.weight", H);
        AddMatrix(w, rng, $"{p}.attn_q.weight", inK: H, outM: qOut, cfg.AttnQuant, 0.05f);
        AddMatrix(w, rng, $"{p}.attn_k.weight", inK: H, outM: kvOut, cfg.AttnQuant, 0.05f);
        // V-less on global layers (V-from-K). Sliding layers carry attn_v.
        if (!global)
            AddMatrix(w, rng, $"{p}.attn_v.weight", inK: H, outM: kvOut, cfg.AttnQuant, 0.05f);
        AddMatrix(w, rng, $"{p}.attn_output.weight", inK: qOut, outM: H, cfg.AttnQuant, 0.05f);
        // QK-norm: head_dim sized (plain RMSNorm weight, F32).
        AddNorm(w, rng, $"{p}.attn_q_norm.weight", hd);
        AddNorm(w, rng, $"{p}.attn_k_norm.weight", hd);
        AddNorm(w, rng, $"{p}.post_attention_norm.weight", H);

        // Dense FFN branch.
        AddNorm(w, rng, $"{p}.ffn_norm.weight", H);
        AddMatrix(w, rng, $"{p}.ffn_gate.weight", inK: H, outM: ff, cfg.DenseFfnQuant, 0.05f);
        AddMatrix(w, rng, $"{p}.ffn_up.weight", inK: H, outM: ff, cfg.DenseFfnQuant, 0.05f);
        AddMatrix(w, rng, $"{p}.ffn_down.weight", inK: ff, outM: H, cfg.DenseFfnQuant, 0.05f);

        // Router (ffn_gate_inp [K=hidden, M=experts]) + channel scale [hidden] (F32).
        AddMatrix(w, rng, $"{p}.ffn_gate_inp.weight", inK: H, outM: e, QuantizationType.F32, 0.05f);
        AddNorm(w, rng, $"{p}.ffn_gate_inp.scale", H);

        // Fused gate_up experts [K=hidden, 2*Ie, E]: per expert a [2*Ie, hidden] slab.
        AddExpertBank(w, rng, $"{p}.ffn_gate_up_exps.weight",
            inK: H, midOut: 2 * ie, experts: e, cfg.ExpertGateUpQuant, 0.05f);
        // Down experts [K=Ie, hidden, E]: per expert a [hidden, Ie] slab.
        AddExpertBank(w, rng, $"{p}.ffn_down_exps.weight",
            inK: ie, midOut: H, experts: e, cfg.ExpertDownQuant, 0.05f);
        // Per-expert down scale [E] (F32).
        AddNorm(w, rng, $"{p}.ffn_down_exps.scale", e);

        // Five-norm dual-FFN extras.
        AddNorm(w, rng, $"{p}.pre_ffw_norm_2.weight", H);
        AddNorm(w, rng, $"{p}.post_ffw_norm_1.weight", H);
        AddNorm(w, rng, $"{p}.post_ffw_norm_2.weight", H);
        AddNorm(w, rng, $"{p}.post_ffw_norm.weight", H);

        // Per-layer output scale [1] (F32), near 1.0 to keep activations stable.
        AddScalarNear(w, rng, $"{p}.layer_output_scale.weight", 1.0f, 0.05f);
        if (diffusion)
            AddScalarNear(w, rng, $"{p}.enc_layer_output_scale.weight", 1.0f, 0.05f);
    }

    // ──────────────────── Tensor emit helpers ────────────────────

    /// <summary>Emits a 2D matrix [ne0=K, ne1=M] of deterministic small weights, quantized.</summary>
    private static void AddMatrix(GgufWriter w, Xorshift rng, string name, int inK, int outM,
        QuantizationType qt, float scale)
    {
        long count = (long)inK * outM;
        var f = new float[count];
        for (long i = 0; i < count; i++) f[i] = rng.NextSigned(scale);
        byte[] data = QuantizeRowMajor(f, rowLen: inK, rows: outM, qt);
        w.AddTensor(name, new[] { inK, outM }, (uint)qt, data);
    }

    /// <summary>
    /// Emits a 3D expert bank [ne0=K, ne1=midOut, ne2=experts]: <paramref name="experts"/>
    /// contiguous [midOut, K] slabs, each row K-quantized. Matches the loader's per-expert
    /// stride = midOut * RowByteSize(K).
    /// </summary>
    private static void AddExpertBank(GgufWriter w, Xorshift rng, string name, int inK, int midOut,
        int experts, QuantizationType qt, float scale)
    {
        long rowsTotal = (long)midOut * experts;
        long count = rowsTotal * inK;
        var f = new float[count];
        for (long i = 0; i < count; i++) f[i] = rng.NextSigned(scale);
        byte[] data = QuantizeRowMajor(f, rowLen: inK, rows: rowsTotal, qt);
        w.AddTensor(name, new[] { inK, midOut, experts }, (uint)qt, data);
    }

    /// <summary>Emits a 1D F32 RMSNorm/scale vector [n] (plain weights — Gemma4 adds NO +1).</summary>
    private static void AddNorm(GgufWriter w, Xorshift rng, string name, int n)
    {
        var f = new float[n];
        // Norm weights near 1.0 (RMSNorm gain), scales near 1.0 — keeps activations stable.
        for (int i = 0; i < n; i++) f[i] = 1.0f + rng.NextSigned(0.05f);
        byte[] data = System.Runtime.InteropServices.MemoryMarshal.AsBytes(f.AsSpan()).ToArray();
        w.AddTensor(name, new[] { n }, (uint)QuantizationType.F32, data);
    }

    /// <summary>Emits a 1D F32 scalar [1] near <paramref name="center"/>.</summary>
    private static void AddScalarNear(GgufWriter w, Xorshift rng, string name, float center, float jitter)
    {
        var f = new[] { center + rng.NextSigned(jitter) };
        byte[] data = System.Runtime.InteropServices.MemoryMarshal.AsBytes(f.AsSpan()).ToArray();
        w.AddTensor(name, new[] { 1 }, (uint)QuantizationType.F32, data);
    }

    /// <summary>
    /// Quantizes a row-major [rows, rowLen] float buffer row-by-row (each row independently,
    /// matching GGUF's per-row block layout) and concatenates the quantized rows.
    /// </summary>
    private static byte[] QuantizeRowMajor(float[] f, int rowLen, long rows, QuantizationType qt)
    {
        long rowBytes = Quantize.RowByteSize(rowLen, qt);
        var data = new byte[rowBytes * rows];
        var rowDest = new byte[rowBytes];
        for (long r = 0; r < rows; r++)
        {
            var rowSrc = f.AsSpan((int)(r * rowLen), rowLen);
            Quantize.FromFloat32(rowSrc, rowLen, qt, rowDest);
            rowDest.AsSpan().CopyTo(data.AsSpan((int)(r * rowBytes)));
        }
        return data;
    }

    private static void AddTokenizer(GgufWriter w, SyntheticGemma4Config cfg)
    {
        w.AddString("tokenizer.ggml.model", "llama");
        var tokens = new string[cfg.VocabSize];
        var scores = new float[cfg.VocabSize];
        var types = new int[cfg.VocabSize];
        for (int i = 0; i < cfg.VocabSize; i++)
        {
            tokens[i] = i switch
            {
                _ when i == cfg.BosTokenId => "<bos>",
                _ when i == cfg.EosTokenId => "<eos>",
                _ when i == cfg.UnknownTokenId => "<unk>",
                _ when i == cfg.MaskTokenId => "<mask>",
                _ => $"tok{i}",
            };
            scores[i] = 0f;
            // 1 = normal, 2 = unknown, 3 = control. Special tokens → control.
            types[i] = (i == cfg.BosTokenId || i == cfg.EosTokenId || i == cfg.MaskTokenId) ? 3
                : i == cfg.UnknownTokenId ? 2 : 1;
        }
        w.AddStringArray("tokenizer.ggml.tokens", tokens);
        w.AddFloat32Array("tokenizer.ggml.scores", scores);
        w.AddInt32Array("tokenizer.ggml.token_type", types);
        w.AddUInt32("tokenizer.ggml.bos_token_id", (uint)cfg.BosTokenId);
        w.AddUInt32("tokenizer.ggml.eos_token_id", (uint)cfg.EosTokenId);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", (uint)cfg.UnknownTokenId);
    }

    /// <summary>Deterministic xorshift32 PRNG → signed floats. No <c>Random</c>/time.</summary>
    internal sealed class Xorshift
    {
        private uint _s;
        public Xorshift(uint seed) => _s = seed == 0 ? 0x9E3779B9u : seed;
        public uint NextUInt() { uint x = _s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; _s = x; return x; }
        /// <summary>Uniform float in [-scale, scale).</summary>
        public float NextSigned(float scale)
        {
            float u = (NextUInt() >> 8) * (1.0f / 16777216.0f);
            return (u * 2f - 1f) * scale;
        }
    }
}
