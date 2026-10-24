using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// #742 — GLM-4.7-Flash (deepseek2 arch) specifics: the pre-split MLA up-projection
/// (<c>attn_k_b</c> / <c>attn_v_b</c>) and the sigmoid + selection-bias router.
/// </summary>
public sealed class Glm47FlashMlaMoeTests
{
    private const int Hidden = 16, Heads = 2, Nope = 4, Rope = 4, VHead = 4, KvLora = 8, Inter = 24, Vocab = 8;

    /// <summary>
    /// The split k_b/v_b tensors must reassemble into exactly the legacy fused kv_b matrix.
    /// k_b is stored TRANSPOSED (element (n,l,h) at n + nope*(l + kvLora*h)); a wrong
    /// index order permutes weights and every layer's attention silently, while still
    /// producing finite output — so this compares against the fused-layout load element by element.
    /// </summary>
    [Fact]
    public void SplitKvB_ReassemblesToTheFusedLayout_ElementForElement()
    {
        int kvBOut = Heads * (Nope + VHead);
        // Distinct value per (row, col) so any permutation is detected.
        float[,] fused = new float[kvBOut, KvLora];
        for (int r = 0; r < kvBOut; r++)
            for (int c = 0; c < KvLora; c++)
                fused[r, c] = (r + 1) * 0.01f + (c + 1) * 0.0001f;

        string fusedPath = WriteGguf(fused, split: false);
        string splitPath = WriteGguf(fused, split: true);
        try
        {
            float[] a = LoadKvB(fusedPath);
            float[] b = LoadKvB(splitPath);
            Assert.Equal(kvBOut * KvLora, a.Length);
            for (int r = 0; r < kvBOut; r++)
                for (int c = 0; c < KvLora; c++)
                {
                    Assert.Equal(fused[r, c], a[r * KvLora + c]);       // control: fused path
                    Assert.Equal(fused[r, c], b[r * KvLora + c]);       // split path under test
                }
        }
        finally { File.Delete(fusedPath); File.Delete(splitPath); }
    }

    private static unsafe float[] LoadKvB(string path)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config);
        var mla = weights.Layers[0].Mla!;
        int n = Heads * (Nope + VHead) * KvLora;
        return new ReadOnlySpan<float>((void*)mla.KvBProj, n).ToArray();
    }

    private static string WriteGguf(float[,] fused, bool split)
    {
        var b = new GgufTestData(version: 3);
        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", Hidden);
        b.AddUInt32("deepseek2.block_count", 1);
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
        if (split)
        {
            // llama.cpp MLA convention: absorbed sizes in key_length/value_length, real in *_mla.
            b.AddUInt32("deepseek2.attention.key_length", KvLora + Rope);
            b.AddUInt32("deepseek2.attention.value_length", KvLora);
            b.AddUInt32("deepseek2.attention.key_length_mla", Nope + Rope);
            b.AddUInt32("deepseek2.attention.value_length_mla", VHead);
        }
        else
        {
            b.AddUInt32("deepseek2.attention.key_length", Nope + Rope);
            b.AddUInt32("deepseek2.attention.value_length", VHead);
        }

        F32(b, "token_embd.weight", [Hidden, Vocab], 1);
        F32(b, "output_norm.weight", [Hidden], 2, 1f);
        F32(b, "output.weight", [Hidden, Vocab], 3);
        F32(b, "blk.0.attn_norm.weight", [Hidden], 4, 1f);
        F32(b, "blk.0.ffn_norm.weight", [Hidden], 5, 1f);
        F32(b, "blk.0.attn_q.weight", [Hidden, Heads * (Nope + Rope)], 6);
        F32(b, "blk.0.attn_kv_a_mqa.weight", [Hidden, KvLora + Rope], 7);
        F32(b, "blk.0.attn_kv_a_norm.weight", [KvLora], 8, 1f);
        F32(b, "blk.0.attn_output.weight", [Heads * VHead, Hidden], 9);
        F32(b, "blk.0.ffn_gate.weight", [Hidden, Inter], 10);
        F32(b, "blk.0.ffn_up.weight", [Hidden, Inter], 11);
        F32(b, "blk.0.ffn_down.weight", [Inter, Hidden], 12);

        int perHead = Nope + VHead;
        if (!split)
        {
            float[] data = new float[Heads * perHead * KvLora];
            for (int r = 0; r < Heads * perHead; r++)
                for (int c = 0; c < KvLora; c++) data[r * KvLora + c] = fused[r, c];
            Raw(b, "blk.0.attn_kv_b.weight", [KvLora, Heads * perHead], data);
        }
        else
        {
            float[] kb = new float[Nope * KvLora * Heads];
            float[] vb = new float[KvLora * VHead * Heads];
            for (int h = 0; h < Heads; h++)
            {
                for (int n = 0; n < Nope; n++)
                    for (int l = 0; l < KvLora; l++)
                        kb[n + Nope * (l + KvLora * h)] = fused[h * perHead + n, l];
                for (int vi = 0; vi < VHead; vi++)
                    for (int l = 0; l < KvLora; l++)
                        vb[l + KvLora * (vi + VHead * h)] = fused[h * perHead + Nope + vi, l];
            }
            Raw(b, "blk.0.attn_k_b.weight", [Nope, KvLora, Heads], kb);
            Raw(b, "blk.0.attn_v_b.weight", [KvLora, VHead, Heads], vb);
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

    private static void F32(GgufTestData b, string name, int[] shape, int seed, float center = 0f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        float[] v = new float[n];
        for (long i = 0; i < n; i++)
            v[i] = center + 0.1f * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
        Raw(b, name, shape, v);
    }

    /// <summary>
    /// DeepSeek-V3 routing: top-k is chosen on sigmoid(logit)+bias, but the weights are the
    /// UNBIASED sigmoids, then renormalised and scaled. Each of the four properties is
    /// observable here: bias changes the experts, selection order follows the biased score,
    /// weights ignore the bias, norm and scale are applied.
    /// </summary>
    [Fact]
    public void Route_SigmoidWithSelectionBias_SelectsOnBiasedScoreButWeightsUnbiased()
    {
        const int E = 4, K = 2, H = 2;
        // logits for hidden=[1,0] are gate[e][0] = [2, 1, 0, -1].
        float[] gate = [2f, 0f, 1f, 0f, 0f, 0f, -1f, 0f];
        float[] hidden = [1f, 0f];
        float[] bias = [0f, 0f, 0.5f, 0.5f];

        (int[] ids, float[] w) Run(bool sigmoid, float[]? b, bool norm, float scale)
        {
            int[] assignExpert = new int[K]; float[] assignWeight = new float[K];
            int[] cursors = new int[E + 1]; int[] tokens = new int[K]; int[] slots = new int[K];
            int[] unique = new int[E];
            MoeSwiGluMlp.Route(hidden, gate, assignExpert, assignWeight, cursors, tokens, slots, unique,
                E, K, H, 1, normTopKProb: norm, sigmoidGating: sigmoid,
                selectionBias: b is null ? default : b, weightsScale: scale);
            return (assignExpert, assignWeight);
        }

        static float Sig(float x) => 1f / (1f + MathF.Exp(-x));

        var biased = Run(true, bias, norm: true, scale: 2.0f);
        Assert.Equal([2, 0], biased.ids);                       // bias promotes expert 2 over expert 1
        float s2 = Sig(0f), s0 = Sig(2f), sum = s2 + s0;        // UNBIASED probs
        Assert.Equal(s2 / sum * 2.0f, biased.w[0], 5);
        Assert.Equal(s0 / sum * 2.0f, biased.w[1], 5);

        var unbiased = Run(true, null, norm: true, scale: 2.0f);
        Assert.Equal([0, 1], unbiased.ids);                     // control: no bias => experts 0,1

        var softmax = Run(false, null, norm: false, scale: 1.0f);
        Assert.Equal([0, 1], softmax.ids);
        Assert.True(softmax.w[0] < 1f && softmax.w[0] > softmax.w[1]);   // softmax probs, not sigmoid
        Assert.NotEqual(Sig(2f), softmax.w[0], 3);
    }
}
