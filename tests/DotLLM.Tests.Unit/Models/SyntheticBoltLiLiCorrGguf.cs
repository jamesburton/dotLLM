using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Synthetic Nemotron-3.5 Bolt (<c>nemotron_h_moe</c>) target and LiLiCorr drafter GGUFs for #1237, in the published files' formats:
/// NVFP4 + <c>X.scale</c> SSM projections, IQ2_XXS routed and shared experts behind a Q8_0 LatentMoE bottleneck, an F32 router with a
/// selection bias, Q8_0 attention, embedding and LM head; the drafter has NVFP4 + scale FFN/fc, BF16 attention and conv projections,
/// attention sinks, the DFlash2 convolutions and the LiLiCorr lattice head. Shapes are tiny but keep the published kernels' paths:
/// d_state 128 (the register scan), K multiples of 256 (IQ2_XXS), KV heads &gt; 1 in the drafter, a window shorter than the context.
/// </summary>
internal static class SyntheticBoltLiLiCorrGguf
{
    public const string Arch = "nemotron_h_moe";
    public const int Hidden = 256, Vocab = 32, Layers = 6;
    public const int DState = 128, NHead = 8, DInner = 256, DConv = 4;
    public const int Experts = 8, TopK = 2, ExpertFf = 256, SharedFf = 256, Latent = 256;
    public const int AttnHeads = 2, AttnHeadDim = 128;

    // Layer pattern: SSM, MoE, attention, MoE, SSM, MoE (head_count_kv / feed_forward_length select the kind).
    public static readonly HybridLayerKind[] Kinds =
        [HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Attention, HybridLayerKind.Ffn, HybridLayerKind.Ssm, HybridLayerKind.Ffn];

    // Drafter.
    public const int DHeads = 4, DKvHeads = 2, DHeadDim = 64, DFfn = 512, DLayers = 2, Block = 8, Window = 16, Mask = Vocab - 1;
    public const int ConvKernel = 2, ConvGroup = 16;
    public const int LiHidden = 128, LiTopK = 4, LiHeads = 4, LiLayers = 2, LiFactor = 64, LiMlp = 256;
    public static readonly int[] TargetLayers = [1, 3, Layers];

    /// <summary>Writes the target GGUF and returns its path.</summary>
    public static string WriteTarget(string path, int seed = 1237, int contextLength = 128)
    {
        var rng = new Random(seed);
        var g = new GgufWriter();
        g.AddString("general.architecture", Arch);
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32($"{Arch}.context_length", (uint)contextLength);
        g.AddUInt32($"{Arch}.embedding_length", Hidden);
        g.AddUInt32($"{Arch}.block_count", Layers);
        g.AddUInt32($"{Arch}.attention.head_count", AttnHeads);
        g.AddInt32Array($"{Arch}.attention.head_count_kv", Kinds.Select(k => k == HybridLayerKind.Attention ? 1 : 0).ToArray());
        int[] ff = Kinds.Select(k => k == HybridLayerKind.Ffn ? ExpertFf : 0).ToArray();
        g.AddInt32Array($"{Arch}.feed_forward_length", ff);
        g.AddUInt32($"{Arch}.attention.key_length", AttnHeadDim);
        g.AddUInt32($"{Arch}.attention.value_length", AttnHeadDim);
        g.AddUInt32($"{Arch}.vocab_size", Vocab);
        g.AddFloat32($"{Arch}.attention.layer_norm_rms_epsilon", 1e-5f);
        g.AddUInt32($"{Arch}.ssm.conv_kernel", DConv);
        g.AddUInt32($"{Arch}.ssm.state_size", DState);
        g.AddUInt32($"{Arch}.ssm.group_count", 1);
        g.AddUInt32($"{Arch}.ssm.time_step_rank", NHead);
        g.AddUInt32($"{Arch}.ssm.inner_size", DInner);
        g.AddUInt32($"{Arch}.expert_count", Experts);
        g.AddInt32Array($"{Arch}.expert_used_count", Kinds.Select(k => k == HybridLayerKind.Ffn ? TopK : 0).ToArray());
        g.AddInt32Array($"{Arch}.expert_feed_forward_length", ff);
        g.AddUInt32($"{Arch}.expert_shared_feed_forward_length", SharedFf);
        g.AddUInt32($"{Arch}.expert_shared_count", 1);
        g.AddUInt32($"{Arch}.expert_gating_func", 2);
        g.AddBool($"{Arch}.expert_weights_norm", true);
        g.AddFloat32($"{Arch}.expert_weights_scale", 2.5f);
        g.AddUInt32($"{Arch}.expert_group_count", 1);
        g.AddUInt32($"{Arch}.expert_group_used_count", 1);
        g.AddUInt32($"{Arch}.moe_latent_size", Latent);

        Q8(g, rng, "token_embd.weight", Hidden, Vocab, 1.0f);
        int convDim = DInner + 2 * DState, inProj = 2 * DInner + 2 * DState + NHead;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}";
            F32(g, $"{p}.attn_norm.weight", [Hidden], Ones(rng, Hidden));
            switch (Kinds[l])
            {
                case HybridLayerKind.Ssm:
                    Nvfp4(g, rng, $"{p}.ssm_in", Hidden, inProj, 0.6f);
                    F32(g, $"{p}.ssm_conv1d.weight", [DConv, convDim], Rand(rng, DConv * convDim, 0.5f));
                    F32(g, $"{p}.ssm_conv1d.bias", [convDim], Rand(rng, convDim, 0.1f));
                    F32(g, $"{p}.ssm_a", [1, NHead], Enumerable.Range(0, NHead).Select(_ => -(float)(rng.NextDouble() * 0.5 + 0.2)).ToArray());
                    F32(g, $"{p}.ssm_d", [1, NHead], Rand(rng, NHead, 0.5f));
                    F32(g, $"{p}.ssm_dt.bias", [NHead], Rand(rng, NHead, 0.5f));
                    F32(g, $"{p}.ssm_norm.weight", [DInner, 1], Ones(rng, DInner));
                    Nvfp4(g, rng, $"{p}.ssm_out", DInner, Hidden, 0.8f);
                    break;
                case HybridLayerKind.Attention:
                    Q8(g, rng, $"{p}.attn_q.weight", Hidden, AttnHeads * AttnHeadDim, 0.3f);
                    Q8(g, rng, $"{p}.attn_k.weight", Hidden, AttnHeadDim, 0.3f);
                    Q8(g, rng, $"{p}.attn_v.weight", Hidden, AttnHeadDim, 0.3f);
                    Q8(g, rng, $"{p}.attn_output.weight", AttnHeads * AttnHeadDim, Hidden, 0.3f);
                    break;
                default:
                    F32(g, $"{p}.ffn_gate_inp.weight", [Hidden, Experts], Rand(rng, Hidden * Experts, 0.3f));
                    F32(g, $"{p}.exp_probs_b.bias", [Experts], Rand(rng, Experts, 0.2f));
                    Iq2Xxs(g, rng, $"{p}.ffn_up_exps.weight", [Latent, ExpertFf, Experts]);
                    Iq2Xxs(g, rng, $"{p}.ffn_down_exps.weight", [ExpertFf, Latent, Experts]);
                    Iq2Xxs(g, rng, $"{p}.ffn_up_shexp.weight", [Hidden, SharedFf]);
                    Iq2Xxs(g, rng, $"{p}.ffn_down_shexp.weight", [SharedFf, Hidden]);
                    Q8(g, rng, $"{p}.ffn_latent_down.weight", Hidden, Latent, 0.3f);
                    Q8(g, rng, $"{p}.ffn_latent_up.weight", Latent, Hidden, 0.3f);
                    break;
            }
        }
        F32(g, "output_norm.weight", [Hidden], Ones(rng, Hidden));
        Q8(g, rng, "output.weight", Hidden, Vocab, 0.6f);
        File.WriteAllBytes(path, g.Build());
        return path;
    }

    /// <summary>Writes a random LiLiCorr drafter for the target above and returns its path.</summary>
    public static string WriteDrafter(string path, int seed)
    {
        var rng = new Random(seed);
        var g = new GgufWriter();
        g.AddString("general.architecture", "dflash");
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32("dflash.embedding_length", Hidden);
        g.AddUInt32("dflash.block_count", DLayers);
        g.AddUInt32("dflash.attention.head_count", DHeads);
        g.AddUInt32("dflash.attention.head_count_kv", DKvHeads);
        g.AddUInt32("dflash.attention.key_length", DHeadDim);
        g.AddUInt32("dflash.attention.value_length", DHeadDim);
        g.AddUInt32("dflash.feed_forward_length", DFfn);
        g.AddFloat32("dflash.attention.layer_norm_rms_epsilon", 1e-6f);
        g.AddFloat32("dflash.rope.freq_base", 10000f);
        g.AddBool("dflash.attention.causal", false);
        g.AddUInt32("dflash.block_size", Block);
        g.AddUInt32("dflash.attention.sliding_window", Window);
        g.AddInt32Array("dflash.target_layers", TargetLayers);
        g.AddUInt32("dflash.conv_kernel_size", ConvKernel);
        g.AddUInt32("dflash.conv_group_size", ConvGroup);
        g.AddUInt32("dflash.selector_rank", 0);
        g.AddUInt32("dflash.selector_top_k", 0);
        g.AddUInt32("tokenizer.ggml.mask_token_id", Mask);
        g.AddUInt32("lilicorr.hidden_size", LiHidden);
        g.AddUInt32("lilicorr.candidate_topk", LiTopK);
        g.AddUInt32("lilicorr.num_heads", LiHeads);
        g.AddUInt32("lilicorr.num_layers", LiLayers);
        g.AddUInt32("lilicorr.factor_dim", LiFactor);
        g.AddUInt32("lilicorr.mlp_hidden_size", LiMlp);
        g.AddFloat32("lilicorr.vector_eps", 1e-4f);
        g.AddFloat32("lilicorr.logit_scale", 8f);

        int q = DHeads * DHeadDim, kv = DKvHeads * DHeadDim, convProj = 2 * ConvKernel * (Hidden / ConvGroup);
        Nvfp4(g, rng, "fc", TargetLayers.Length * Hidden, Hidden, 0.5f);
        F32(g, "enc.output_norm.weight", [Hidden], Ones(rng, Hidden));
        F32(g, "output_norm.weight", [Hidden], Ones(rng, Hidden));
        for (int i = 0; i < DLayers; i++)
        {
            string p = $"blk.{i}";
            F32(g, $"{p}.attn_norm.weight", [Hidden], Ones(rng, Hidden));
            F32(g, $"{p}.ffn_norm.weight", [Hidden], Ones(rng, Hidden));
            F32(g, $"{p}.attn_q_norm.weight", [DHeadDim], Ones(rng, DHeadDim));
            F32(g, $"{p}.attn_k_norm.weight", [DHeadDim], Ones(rng, DHeadDim));
            F32(g, $"{p}.attn_sinks.weight", [DHeads], Rand(rng, DHeads, 1.5f));
            Bf16(g, rng, $"{p}.attn_q.weight", Hidden, q, 0.1f);
            Bf16(g, rng, $"{p}.attn_k.weight", Hidden, kv, 0.1f);
            Bf16(g, rng, $"{p}.attn_v.weight", Hidden, kv, 0.1f);
            Bf16(g, rng, $"{p}.attn_output.weight", q, Hidden, 0.1f);
            Nvfp4(g, rng, $"{p}.ffn_gate", Hidden, DFfn, 0.4f);
            Nvfp4(g, rng, $"{p}.ffn_up", Hidden, DFfn, 0.4f);
            Nvfp4(g, rng, $"{p}.ffn_down", DFfn, Hidden, 0.4f);
            F32(g, $"{p}.attn_conv_base", [Hidden, ConvKernel, 2], Rand(rng, Hidden * ConvKernel * 2, 0.5f));
            Bf16(g, rng, $"{p}.attn_conv_proj.weight", Hidden, convProj, 0.05f);
            F32(g, $"{p}.ffn_conv_base", [Hidden, ConvKernel, 2], Rand(rng, Hidden * ConvKernel * 2, 0.5f));
            Bf16(g, rng, $"{p}.ffn_conv_proj.weight", Hidden, convProj, 0.05f);
        }

        // LiLiCorr head (llama.cpp's converted layout): head layers at blk.{DLayers + i}, the feature MLP at blk.{DLayers + LiLayers}.
        int slots = Block - 1, nodes = slots * LiTopK, d = LiHidden;
        Nvfp4(g, rng, "lilicorr.token_proj", Hidden, d, 0.5f);
        F32(g, "lilicorr.token_proj.bias", [d], Rand(rng, d, 0.1f));
        Nvfp4(g, rng, "lilicorr.pass_hidden_proj", Hidden, d, 0.5f);
        F32(g, "lilicorr.pass_hidden_proj.bias", [d], Rand(rng, d, 0.1f));
        Nvfp4(g, rng, "lilicorr.context_proj", Hidden, d, 0.5f);
        F32(g, "lilicorr.context_proj.bias", [d], Rand(rng, d, 0.1f));
        int fb = DLayers + LiLayers;
        F32(g, $"blk.{fb}.ffn_norm.weight", [5], Ones(rng, 5));
        F32(g, $"blk.{fb}.ffn_norm.bias", [5], Rand(rng, 5, 0.1f));
        F32(g, $"blk.{fb}.ffn_up.weight", [5, d], Rand(rng, 5 * d, 0.5f));
        F32(g, $"blk.{fb}.ffn_up.bias", [d], Rand(rng, d, 0.1f));
        F32(g, $"blk.{fb}.ffn_down.weight", [d, d], Rand(rng, d * d, 0.1f));
        F32(g, $"blk.{fb}.ffn_down.bias", [d], Rand(rng, d, 0.1f));
        F32(g, "lilicorr.slot_embedding.weight", [d, slots], Rand(rng, d * slots, 0.5f));
        F32(g, "lilicorr.rank_embedding.weight", [d, LiTopK], Rand(rng, d * LiTopK, 0.5f));
        F32(g, "lilicorr.attention_bias.bias", [nodes, nodes, LiHeads], Rand(rng, nodes * nodes * LiHeads, 1f));
        for (int i = 0; i < LiLayers; i++)
        {
            string p = $"blk.{DLayers + i}";
            F32(g, $"{p}.attn_norm.weight", [d], Ones(rng, d));
            F32(g, $"{p}.ffn_norm.weight", [d], Ones(rng, d));
            F32(g, $"lilicorr.layers.{i}.attn.in_proj.weight", [d, 3 * d], Rand(rng, d * 3 * d, 0.1f));
            F32(g, $"lilicorr.layers.{i}.attn.in_proj.bias", [3 * d], Rand(rng, 3 * d, 0.1f));
            Nvfp4(g, rng, $"{p}.attn_output", d, d, 0.3f);
            F32(g, $"{p}.attn_output.bias", [d], Rand(rng, d, 0.1f));
            Nvfp4(g, rng, $"{p}.ffn_up", d, LiMlp, 0.3f);
            F32(g, $"{p}.ffn_up.bias", [LiMlp], Rand(rng, LiMlp, 0.1f));
            Nvfp4(g, rng, $"{p}.ffn_down", LiMlp, d, 0.3f);
            F32(g, $"{p}.ffn_down.bias", [d], Rand(rng, d, 0.1f));
        }
        F32(g, "lilicorr.output_norm.weight", [d], Ones(rng, d));
        F32(g, "lilicorr.anchor_norm.weight", [d], Ones(rng, d));
        F32(g, "lilicorr.factor_input_proj.weight", [3 * d, d], Rand(rng, 3 * d * d, 0.1f));
        F32(g, "lilicorr.factor_input_proj.bias", [d], Rand(rng, d, 0.1f));
        foreach (string head in new[] { "out_head", "in_head", "anchor_out_head" })
        {
            F32(g, $"lilicorr.{head}.weight", [d, LiFactor], Rand(rng, d * LiFactor, 0.2f));
            F32(g, $"lilicorr.{head}.bias", [LiFactor], Rand(rng, LiFactor, 0.1f));
        }
        File.WriteAllBytes(path, g.Build());
        return path;
    }

    // ── Writers ─────────────────────────────────────────────────────

    private static void F32(GgufWriter g, string name, int[] dims, float[] data)
        => g.AddTensor(name, dims, (uint)QuantizationType.F32, MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    private static void Q8(GgufWriter g, Random rng, string name, int inK, int outM, float amp)
    {
        float[] w = Rand(rng, inK * outM, amp);
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(inK, QuantizationType.Q8_0);
        byte[] b = new byte[rowBytes * outM];
        for (int r = 0; r < outM; r++)
            DotLLM.Cpu.Kernels.Quantize.FromFloat32(w.AsSpan(r * inK, inK), inK, QuantizationType.Q8_0, b.AsSpan((int)(r * rowBytes), (int)rowBytes));
        g.AddTensor(name, [inK, outM], (uint)QuantizationType.Q8_0, b);
    }

    private static void Bf16(GgufWriter g, Random rng, string name, int inK, int outM, float amp)
    {
        float[] w = Rand(rng, inK * outM, amp);
        byte[] b = new byte[w.Length * 2];
        for (int i = 0; i < w.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(w[i]) >> 16;
            b[2 * i] = (byte)bits;
            b[2 * i + 1] = (byte)(bits >> 8);
        }
        g.AddTensor(name, [inK, outM], (uint)QuantizationType.BF16, b);
    }

    /// <summary><c>{stem}.weight</c> NVFP4 [inK, outM] with random codes and UE4M3 scales in 0x28..0x37, plus <c>{stem}.scale</c>.</summary>
    private static void Nvfp4(GgufWriter g, Random rng, string stem, int inK, int outM, float scale)
    {
        byte[] b = new byte[(long)inK * outM / 64 * 36];
        rng.NextBytes(b);
        for (int blk = 0; blk < b.Length; blk += 36)
            for (int s = 0; s < 4; s++)
                b[blk + s] = (byte)rng.Next(0x28, 0x38);
        g.AddTensor($"{stem}.weight", [inK, outM], (uint)QuantizationType.NVFP4, b);
        F32(g, $"{stem}.scale", [1], [scale]);
    }

    /// <summary>IQ2_XXS of the given dims with random grid/sign/scale codes and a small fp16 block scale.</summary>
    private static void Iq2Xxs(GgufWriter g, Random rng, string name, int[] dims)
    {
        long elems = 1;
        foreach (int dd in dims) elems *= dd;
        byte[] b = new byte[elems / 256 * 66];
        rng.NextBytes(b);
        for (long blk = 0; blk < b.Length; blk += 66)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.004f + 0.004f * (float)rng.NextDouble()));
            b[blk] = (byte)d;
            b[blk + 1] = (byte)(d >> 8);
        }
        g.AddTensor(name, dims, (uint)QuantizationType.IQ2_XXS, b);
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * amp;
        return a;
    }

    private static float[] Ones(Random rng, int n)
    {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 1.0f + (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return a;
    }
}
