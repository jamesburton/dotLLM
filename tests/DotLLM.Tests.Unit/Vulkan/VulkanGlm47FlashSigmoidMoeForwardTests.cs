using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #742 end-to-end: a tiny GLM-4.7-Flash-shaped deepseek2 GGUF (split k_b/v_b MLA, one dense layer then
/// a sigmoid + selection-bias MoE layer with a shared expert, renorm and scale) must produce the same
/// logits on Vulkan as on the CPU reference. A control proves the selection bias changes the output,
/// so a Vulkan path that ignored the bias (or used softmax) could not pass.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGlm47FlashSigmoidMoeForwardTests
{
    private const int Hidden = 16, Heads = 2, Nope = 4, Rope = 4, VHead = 4, KvLora = 8,
        Inter = 24, Vocab = 8, E = 8, K = 2, MoeI = 12, Layers = 2;

    [SkippableFact]
    public void Forward_SigmoidBiasMoe_MatchesCpu_AndBiasMatters()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string biased = Write(strongBias: true);
        string unbiased = Write(strongBias: false);
        try
        {
            int[] tokens = [1, 3, 5];
            int[] pos = [0, 1, 2];
            float[] cpu = Cpu(biased, tokens, pos);
            float[] cpuNoBias = Cpu(unbiased, tokens, pos);
            float[] vk = Vk(biased, spvDir, tokens, pos);

            float maxControl = 0;
            for (int c = 0; c < Vocab; c++) maxControl = MathF.Max(maxControl, MathF.Abs(cpu[c] - cpuNoBias[c]));
            Assert.True(maxControl > 5e-3f, $"control insensitive: bias moved CPU logits by only {maxControl:E3}");

            for (int c = 0; c < Vocab; c++)
            {
                float diff = MathF.Abs(cpu[c] - vk[c]);
                Assert.True(diff <= 5e-3f + 1e-3f * MathF.Abs(cpu[c]), $"col={c}: cpu={cpu[c]:F6} vk={vk[c]:F6}");
            }
        }
        finally { File.Delete(biased); File.Delete(unbiased); }
    }

    private static unsafe float[] Copy(ITensor t)
    {
        int n = t.Shape[0] * t.Shape[1];
        float[] a = new float[n];
        new ReadOnlySpan<float>((void*)t.DataPointer, n).CopyTo(a);
        return a;
    }

    private static float[] Cpu(string path, int[] tokens, int[] pos)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        using ITensor l = model.Forward(tokens, pos, deviceId: -1);
        float[] all = Copy(l);
        return all[((tokens.Length - 1) * Vocab)..];
    }

    private static float[] Vk(string path, string spvDir, int[] tokens, int[] pos)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        using ITensor l = model.Forward(tokens, pos, deviceId: -1);
        return Copy(l);
    }

    private static string Write(bool strongBias)
    {
        var b = new GgufTestData(version: 3);
        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", Hidden);
        b.AddUInt32("deepseek2.block_count", Layers);
        b.AddUInt32("deepseek2.feed_forward_length", Inter);
        b.AddUInt32("deepseek2.attention.head_count", Heads);
        b.AddUInt32("deepseek2.attention.head_count_kv", 1);
        b.AddUInt32("deepseek2.context_length", 16);
        b.AddFloat32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        b.AddUInt32("deepseek2.vocab_size", Vocab);
        b.AddFloat32("deepseek2.rope.freq_base", 10000.0f);
        b.AddUInt32("deepseek2.rope.dimension_count", Rope);
        b.AddUInt32("deepseek2.attention.q_lora_rank", 0);
        b.AddUInt32("deepseek2.attention.kv_lora_rank", KvLora);
        b.AddUInt32("deepseek2.attention.key_length", KvLora + Rope);
        b.AddUInt32("deepseek2.attention.value_length", KvLora);
        b.AddUInt32("deepseek2.attention.key_length_mla", Nope + Rope);
        b.AddUInt32("deepseek2.attention.value_length_mla", VHead);
        b.AddUInt32("deepseek2.leading_dense_block_count", 1);
        b.AddUInt32("deepseek2.expert_count", E);
        b.AddUInt32("deepseek2.expert_used_count", K);
        b.AddUInt32("deepseek2.expert_feed_forward_length", MoeI);
        b.AddUInt32("deepseek2.expert_shared_count", 1);
        b.AddUInt32("deepseek2.expert_gating_func", 2);
        b.AddBool("deepseek2.expert_weights_norm", true);
        b.AddFloat32("deepseek2.expert_weights_scale", 1.8f);

        F32(b, "token_embd.weight", [Hidden, Vocab], 1);
        F32(b, "output_norm.weight", [Hidden], 2, 1f);
        F32(b, "output.weight", [Hidden, Vocab], 3);
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            int s = 100 * (l + 1);
            F32(b, p + "attn_norm.weight", [Hidden], s + 4, 1f);
            F32(b, p + "ffn_norm.weight", [Hidden], s + 5, 1f);
            F32(b, p + "attn_q.weight", [Hidden, Heads * (Nope + Rope)], s + 6);
            F32(b, p + "attn_kv_a_mqa.weight", [Hidden, KvLora + Rope], s + 7);
            F32(b, p + "attn_kv_a_norm.weight", [KvLora], s + 8, 1f);
            F32(b, p + "attn_k_b.weight", [Nope, KvLora, Heads], s + 9);
            F32(b, p + "attn_v_b.weight", [KvLora, VHead, Heads], s + 10);
            F32(b, p + "attn_output.weight", [Heads * VHead, Hidden], s + 11);
            if (l == 0)
            {
                F32(b, p + "ffn_gate.weight", [Hidden, Inter], s + 12);
                F32(b, p + "ffn_up.weight", [Hidden, Inter], s + 13);
                F32(b, p + "ffn_down.weight", [Inter, Hidden], s + 14);
            }
            else
            {
                F32(b, p + "ffn_gate_inp.weight", [Hidden, E], s + 12, scale: 0.6f);
                float[] bias = new float[E];
                if (strongBias) for (int e = 0; e < E; e++) bias[e] = (e * 5 % E) * 0.12f;
                Raw(b, p + "exp_probs_b.bias", [E], bias);
                F32(b, p + "ffn_gate_exps.weight", [Hidden, MoeI, E], s + 13);
                F32(b, p + "ffn_up_exps.weight", [Hidden, MoeI, E], s + 14);
                F32(b, p + "ffn_down_exps.weight", [MoeI, Hidden, E], s + 15);
                F32(b, p + "ffn_gate_shexp.weight", [Hidden, MoeI], s + 16);
                F32(b, p + "ffn_up_shexp.weight", [Hidden, MoeI], s + 17);
                F32(b, p + "ffn_down_shexp.weight", [MoeI, Hidden], s + 18);
            }
        }
        return b.WriteToTempFile();
    }

    private static void Raw(GgufTestData b, string name, int[] shape, float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        for (int i = 0; i < values.Length; i++)
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        b.AddTensor(name, shape, quantType: 0, bytes);
    }

    private static void F32(GgufTestData b, string name, int[] shape, int seed, float center = 0f, float scale = 0.1f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        float[] v = new float[n];
        for (long i = 0; i < n; i++)
            v[i] = center + scale * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
        Raw(b, name, shape, v);
    }
}
