using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end CPU GGUF DeepSeek-V2 load + forward integration test. Builds a
/// synthetic minimal-shaped DeepSeek-V2-Lite-style GGUF in memory (2 layers:
/// layer 0 dense FFN, layer 1 MoE; all tensors F32 to keep the fixture small
/// and avoid Q4_K block-alignment constraints on hidden=16), loads it via
/// <see cref="TransformerModel.LoadFromGguf"/>, and runs prefill to verify
/// finite logits. The decisive evidence that the GGUF MLA + MoE loader chain
/// actually works end-to-end without needing a 10 GB downloaded checkpoint.
/// </summary>
public sealed class DeepSeekV2GgufLoadTests
{
    // Tiny shapes that exercise both monolithic-Q (V2-Lite) and the
    // MoE 3D-stacked expert layout. Keep hidden small enough that the
    // F32 fixture fits comfortably in memory but large enough that
    // block-aligned dequant paths (Q4_K) would still apply if/when the
    // test is parameterised over quant types.
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int QkNope = 4;
    private const int QkRope = 4;        // RoPE pairs → must be even
    private const int VHead = 4;
    private const int KvLoraRank = 8;
    private const int IntermediateSize = 24;
    private const int MoeIntermediate = 24;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;
    private const int LeadingDenseBlocks = 1;   // layer 0 stays dense; layer 1+ is MoE

    [Fact]
    public void LoadFromGguf_DeepSeekV2Lite_MonolithicQ_Loads()
    {
        string ggufPath = WriteFixture(qLoraRank: 0, seed: 42);
        try
        {
            using var gguf = GgufFile.Open(ggufPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            // Sanity-check the extracted config
            Assert.Equal(Architecture.DeepSeekV2, config.Architecture);
            Assert.Equal(AttentionType.MLA, config.AttentionType);
            Assert.NotNull(config.MlaConfig);
            Assert.Equal(0, config.MlaConfig.QLoraRank);
            Assert.NotNull(config.Moe);
            Assert.Equal(NumExperts, config.Moe.NumExperts);

            // Load the weights
            using var weights = TransformerWeights.LoadFromGguf(gguf, config);
            Assert.Equal(NumLayers, weights.Layers.Length);

            // Layer 0: dense FFN, MLA attention
            ref readonly var layer0 = ref weights.Layers[0];
            Assert.NotNull(layer0.Mla);
            Assert.Null(layer0.Moe);
            Assert.NotEqual((nint)0, layer0.GateWeight);

            // Layer 1: MoE FFN, MLA attention
            ref readonly var layer1 = ref weights.Layers[1];
            Assert.NotNull(layer1.Mla);
            Assert.NotNull(layer1.Moe);
            Assert.Equal(NumExperts, layer1.Moe.W1.Length);
            Assert.Equal((nint)0, layer1.GateWeight);  // dense slots zeroed
        }
        finally
        {
            File.Delete(ggufPath);
        }
    }

    /// <summary>
    /// Real-checkpoint config-extraction smoke test against a cached
    /// <c>DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf</c> (~10.4 GB,
    /// 16B-class with MLA + 64 routed + 2 shared experts). Skipped when
    /// the file isn't downloaded. This validates that the metadata
    /// extractor handles the real key naming + type encoding without
    /// blowing the host RAM (only metadata is read; tensor data isn't
    /// touched). Full TransformerWeights load + forward exercises tasks
    /// #9 / #10 (on-device dequant) — gated on the F32 dequant pressure
    /// being addressed.
    /// </summary>
    [SkippableFact]
    public void RealGguf_ConfigExtractor_ParsesDeepSeekV2Lite()
    {
        string path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf");
        Skip.If(!File.Exists(path), $"Real DeepSeek-V2-Lite GGUF not cached at {path}");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.DeepSeekV2, config.Architecture);
        Assert.Equal(AttentionType.MLA, config.AttentionType);

        // V2-Lite expected hyperparameters (per HF config.json on
        // deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct):
        //   hidden_size = 2048, num_hidden_layers = 27,
        //   num_attention_heads = 16, num_key_value_heads = 16,
        //   intermediate_size = 10944, vocab_size ≈ 102400,
        //   first_k_dense_replace = 1, n_routed_experts = 64,
        //   num_experts_per_tok = 6, n_shared_experts = 2,
        //   moe_intermediate_size = 1408,
        //   q_lora_rank = 0 (monolithic Q on V2-Lite),
        //   kv_lora_rank = 512, qk_nope_head_dim = 128,
        //   qk_rope_head_dim = 64, v_head_dim = 128.
        Assert.Equal(2048, config.HiddenSize);
        Assert.Equal(27, config.NumLayers);
        Assert.Equal(16, config.NumAttentionHeads);

        Assert.NotNull(config.MlaConfig);
        Assert.Equal(0, config.MlaConfig.QLoraRank);   // V2-Lite is monolithic-Q
        Assert.Equal(512, config.MlaConfig.KvLoraRank);
        Assert.Equal(128, config.MlaConfig.QkNopeHeadDim);
        Assert.Equal(64, config.MlaConfig.QkRopeHeadDim);
        Assert.Equal(128, config.MlaConfig.VHeadDim);
        Assert.Equal(192, config.HeadDim);             // patched to qk_nope + qk_rope

        Assert.NotNull(config.Moe);
        Assert.Equal(64, config.Moe.NumExperts);
        Assert.Equal(6, config.Moe.NumExpertsPerTok);
        Assert.Equal(2, config.Moe.NumSharedExperts);
        Assert.Equal(1408, config.Moe.MoeIntermediateSize);
        Assert.Equal(1408 * 2, config.Moe.SharedExpertIntermediateSize);
        // leading_dense_block_count = 1 → layer 0 dense, 1..26 MoE
        Assert.False(config.Moe.IsMoeLayer(0));
        Assert.True(config.Moe.IsMoeLayer(1));
        Assert.True(config.Moe.IsMoeLayer(26));
    }

    /// <summary>
    /// Narrow real-checkpoint smoke against the cached
    /// <c>DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf</c>. Trims the model to
    /// <see cref="ModelConfig.NumLayers"/>=1 (only layer 0, which is dense FFN
    /// per <c>leading_dense_block_count=1</c>) so the GGUF MoE tensor loader
    /// (still F32-host-dequant for now → would blow ~57 GB host RAM at full
    /// V2-Lite scale) is bypassed entirely. Exercises the new quantized MLA
    /// path on real Q4_K_M weights — proves the GGUF→CudaTransformerModel→
    /// CudaMlaAttention.ForwardF16(Quantized) chain works end-to-end on real
    /// data without crashing or producing NaN logits. The full multi-layer
    /// run gates on task #10 (quantized MoE + on-device dequant for the 3D
    /// expert tensors).
    /// </summary>
    [SkippableFact]
    [Trait("Category", "GPU")]
    public void RealGguf_QuantizedMla_Layer0Smoke()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf");
        Skip.If(!File.Exists(path), $"Real DeepSeek-V2-Lite GGUF not cached at {path}");

        using var gguf = GgufFile.Open(path);
        var fullConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);

        // Trim to layer 0 (dense FFN, no MoE) so the F32-host MoE dequant
        // is skipped entirely. Also clear Moe so the dispatcher doesn't
        // expect MoE entries on a 1-layer model.
        var config = fullConfig with { NumLayers = 1, Moe = null };

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        Assert.Equal(MlaPrecision.Quantized, ModelLayer0MlaPrecision(model));

        // Tiny prefill — first 4 tokens of any prompt. We don't have the
        // tokenizer wired up here so we use raw token IDs from the BOS region.
        int[] tokenIds = [100000, 261, 1559, 11];   // arbitrary valid V2-Lite ids
        int[] positions = [0, 1, 2, 3];

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: 0, kvCache: null);
        Assert.Equal(fullConfig.VocabSize, logits.Shape[1]);

        unsafe
        {
            int finite = 0;
            int total = logits.Shape.Rank == 1 ? logits.Shape[0] : logits.Shape[1];
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
            foreach (float v in span)
                if (float.IsFinite(v)) finite++;
            Assert.True(finite == total,
                $"Expected all {total} logits finite; got {finite} finite. " +
                $"First 10: [{string.Join(", ", span.Slice(0, Math.Min(10, total)).ToArray().Select(v => v.ToString("F3")))}]");
        }
    }

    /// <summary>
    /// Helper: peek at layer 0's MLA precision via the model's internal weights.
    /// Confirms the GGUF loader routed through LoadLayerQuant rather than
    /// falling back to the F16-cast path.
    /// </summary>
    private static MlaPrecision ModelLayer0MlaPrecision(CudaTransformerModel model)
    {
        // Reach in via reflection to the private _weights field. This test
        // file uses InternalsVisibleTo, but the field is private — short of
        // adding an internal accessor we use reflection. Acceptable for this
        // single one-off test (kept narrow on purpose).
        var weightsField = typeof(CudaTransformerModel).GetField("_weights",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        var weights = weightsField!.GetValue(model);
        var mlaLayersProp = weights!.GetType().GetProperty("MlaLayers");
        var mlaLayers = (Array)mlaLayersProp!.GetValue(weights)!;
        var layer0 = mlaLayers.GetValue(0)!;
        var precisionField = layer0.GetType().GetField("Precision");
        return (MlaPrecision)precisionField!.GetValue(layer0)!;
    }

    [Fact]
    public void LoadFromGguf_DeepSeekV2_LoraQ_Loads()
    {
        string ggufPath = WriteFixture(qLoraRank: 8, seed: 7);
        try
        {
            using var gguf = GgufFile.Open(ggufPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            Assert.Equal(8, config.MlaConfig!.QLoraRank);

            using var weights = TransformerWeights.LoadFromGguf(gguf, config);
            ref readonly var layer0 = ref weights.Layers[0];
            Assert.NotNull(layer0.Mla);
            Assert.Equal(8, layer0.Mla.QLoraRank);
            Assert.NotEqual((nint)0, layer0.Mla.QAProj);
            Assert.Equal((nint)0, layer0.Mla.QProj);
        }
        finally
        {
            File.Delete(ggufPath);
        }
    }

    /// <summary>
    /// Writes a synthetic minimal DeepSeek-V2-Lite-shaped GGUF in F32 (no
    /// quantization) to a temp file. F32 keeps the byte layout trivial:
    /// <c>elementCount * 4</c> bytes per tensor, no per-block alignment
    /// concerns. The same fixture exercises Q4_K once per-block alignment
    /// constraints (hidden % 256 == 0) are accommodated.
    /// </summary>
    private static string WriteFixture(int qLoraRank, int seed)
    {
        var b = new GgufTestData(version: 3);

        int qkHead = QkNope + QkRope;
        int qTotal = NumHeads * qkHead;
        int kvAOut = KvLoraRank + QkRope;
        int kvBOut = NumHeads * (QkNope + VHead);
        int oInput = NumHeads * VHead;

        // ── Metadata ──────────────────────────────────────────────────
        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", (uint)HiddenSize);
        b.AddUInt32("deepseek2.block_count", (uint)NumLayers);
        b.AddUInt32("deepseek2.feed_forward_length", (uint)IntermediateSize);
        b.AddUInt32("deepseek2.attention.head_count", (uint)NumHeads);
        b.AddUInt32("deepseek2.attention.head_count_kv", (uint)NumHeads);
        b.AddUInt32("deepseek2.context_length", 16);
        b.AddFloat32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        b.AddUInt32("deepseek2.vocab_size", (uint)VocabSize);
        b.AddFloat32("deepseek2.rope.freq_base", 10000.0f);
        b.AddUInt32("deepseek2.rope.dimension_count", (uint)QkRope);

        // MLA
        b.AddUInt32("deepseek2.attention.q_lora_rank", (uint)qLoraRank);
        b.AddUInt32("deepseek2.attention.kv_lora_rank", (uint)KvLoraRank);
        // attention.key_length = TOTAL per-head qk dim (qk_nope + qk_rope), per
        // llama.cpp's gguf_writer convention for MLA models. The extractor
        // derives qk_nope = key_length - rope.dimension_count.
        b.AddUInt32("deepseek2.attention.key_length", (uint)(QkNope + QkRope));
        b.AddUInt32("deepseek2.attention.value_length", (uint)VHead);

        // MoE
        b.AddUInt32("deepseek2.expert_count", (uint)NumExperts);
        b.AddUInt32("deepseek2.expert_used_count", (uint)NumExpertsPerTok);
        b.AddUInt32("deepseek2.expert_shared_count", 0);
        b.AddUInt32("deepseek2.expert_feed_forward_length", (uint)MoeIntermediate);
        b.AddUInt32("deepseek2.leading_dense_block_count", (uint)LeadingDenseBlocks);

        // ── Tensors (F32) ─────────────────────────────────────────────
        // Globals
        AddF32Tensor(b, "token_embd.weight", [HiddenSize, VocabSize], seed + 100);  // GGUF: [K, M]
        AddF32Tensor(b, "output_norm.weight", [HiddenSize], seed + 101, center: 1.0f, jitter: 0.05f);
        AddF32Tensor(b, "output.weight", [HiddenSize, VocabSize], seed + 102);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 1000 * (i + 1);
            string p = $"blk.{i}";

            AddF32Tensor(b, $"{p}.attn_norm.weight", [HiddenSize], s + 0, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.ffn_norm.weight", [HiddenSize], s + 1, center: 1.0f, jitter: 0.05f);

            // MLA attention
            if (qLoraRank > 0)
            {
                AddF32Tensor(b, $"{p}.attn_q_a.weight", [HiddenSize, qLoraRank], s + 2);
                AddF32Tensor(b, $"{p}.attn_q_a_norm.weight", [qLoraRank], s + 3, center: 1.0f, jitter: 0.05f);
                AddF32Tensor(b, $"{p}.attn_q_b.weight", [qLoraRank, qTotal], s + 4);
            }
            else
            {
                AddF32Tensor(b, $"{p}.attn_q.weight", [HiddenSize, qTotal], s + 2);
            }
            AddF32Tensor(b, $"{p}.attn_kv_a_mqa.weight", [HiddenSize, kvAOut], s + 5);
            AddF32Tensor(b, $"{p}.attn_kv_a_norm.weight", [KvLoraRank], s + 6, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.attn_kv_b.weight", [KvLoraRank, kvBOut], s + 7);
            AddF32Tensor(b, $"{p}.attn_output.weight", [oInput, HiddenSize], s + 8);

            // FFN: layer 0 dense; layer 1+ MoE.
            if (i < LeadingDenseBlocks)
            {
                AddF32Tensor(b, $"{p}.ffn_gate.weight", [HiddenSize, IntermediateSize], s + 9);
                AddF32Tensor(b, $"{p}.ffn_up.weight", [HiddenSize, IntermediateSize], s + 10);
                AddF32Tensor(b, $"{p}.ffn_down.weight", [IntermediateSize, HiddenSize], s + 11);
            }
            else
            {
                // Router: [hidden, num_experts]
                AddF32Tensor(b, $"{p}.ffn_gate_inp.weight", [HiddenSize, NumExperts], s + 12);
                // Fused experts (3D): [hidden, intermediate, num_experts] for gate/up
                AddF32Tensor(b, $"{p}.ffn_gate_exps.weight",
                    [HiddenSize, MoeIntermediate, NumExperts], s + 13);
                AddF32Tensor(b, $"{p}.ffn_up_exps.weight",
                    [HiddenSize, MoeIntermediate, NumExperts], s + 14);
                // Fused down (3D): [intermediate, hidden, num_experts]
                AddF32Tensor(b, $"{p}.ffn_down_exps.weight",
                    [MoeIntermediate, HiddenSize, NumExperts], s + 15);
            }
        }

        return b.WriteToTempFile();
    }

    /// <summary>
    /// Adds an F32 tensor with a deterministic cos-based fill. <paramref name="center"/>
    /// + <paramref name="jitter"/> form the near-unity range used by norm weights.
    /// </summary>
    private static void AddF32Tensor(GgufTestData b, string name, int[] shape, int seed,
                                     float amplitude = 0.1f,
                                     float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        byte[] bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            float v = jitter > 0f ? center + jitter * cos : amplitude * cos;
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(
                bytes.AsSpan((int)(i * sizeof(float)), sizeof(float)), v);
        }
        // GGUF tensor type IDs: F32=0, F16=1, ...
        b.AddTensor(name, shape, quantType: 0, bytes);
    }
}
