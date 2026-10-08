using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Writes the HF-generated tiny Qwen4-Exp checkpoint (<c>Fixtures/tiny_model.json</c>, GGUF-convention tensors) as a real
/// <c>qwen4exp</c> GGUF with matching metadata, so the HF parity test drives the production path end to end:
/// <see cref="GgufFile"/> → <see cref="GgufModelConfigExtractor"/> → <c>ModelLoader</c> → forward.
/// </summary>
internal static class Qwen4ExpTinyGguf
{
    /// <summary>Non-tensor entries of the fixture (reference values / hash constants).</summary>
    private static bool IsTensor(string name) =>
        !(name.StartsWith("l_out.", StringComparison.Ordinal) || name is "ids" or "hidden_final" or "logits"
          or "ids_img" or "img_positions" or "img_embeds" or "hidden_final_img" or "logits_img"
          || name.StartsWith("ple.", StringComparison.Ordinal));

    /// <summary>
    /// The real UD-Q4_K_XL mix where the tiny geometry allows: BF16 indexer projections, Q4_K hc-down (K = 4*H = 256), Q8_0
    /// projections / router / embedding / shared expert, Q8_0 gate+up and Q5_1 down expert banks (the Q5_1 bank exercises the
    /// no-direct-kernel fallback), F16 n-gram table; norms, convs, inject, hc-up (K = 20) and scalars stay F32.
    /// </summary>
    private static QuantizationType ChooseQuant(string name, int[] shape)
    {
        if (name == "per_layer_token_embd.weight") return QuantizationType.F16;
        if (name.Contains("indexer.", StringComparison.Ordinal) && name.EndsWith("_proj.weight", StringComparison.Ordinal)) return QuantizationType.BF16;
        if (name.Contains("ffn_down_exps", StringComparison.Ordinal)) return QuantizationType.Q5_1;
        if (name.Contains("_down.weight", StringComparison.Ordinal) && name.Contains("hc_", StringComparison.Ordinal)) return QuantizationType.Q4_K;
        if (shape.Length >= 2 && name.EndsWith(".weight", StringComparison.Ordinal) && shape[^1] % 32 == 0
            && !name.Contains("conv1d", StringComparison.Ordinal) && !name.Contains("inject", StringComparison.Ordinal))
            return QuantizationType.Q8_0;
        return QuantizationType.F32;
    }

    /// <summary>Builds the GGUF bytes. <paramref name="shards"/> &gt; 1 is not supported here (see the split test).</summary>
    public static byte[] Build(Qwen4ExpReferenceFixture fx, int contextLength = 256, bool quantize = false, int? budgetTokens = null, bool omitImageTokenId = false)
    {
        const string arch = "qwen4exp";
        int layers = fx.Int("num_layers"), blockSize = fx.Int("block");
        var w = new GgufWriter();
        w.AddString("general.architecture", arch);
        w.AddString("general.name", "hf-tiny-qwen4exp");
        w.AddUInt32("general.alignment", 32);
        w.AddUInt32($"{arch}.block_count", (uint)layers);
        w.AddUInt32($"{arch}.context_length", (uint)contextLength);
        w.AddUInt32($"{arch}.embedding_length", (uint)fx.Int("hidden_size"));
        w.AddUInt32($"{arch}.attention.head_count", (uint)fx.Int("heads"));
        w.AddUInt32($"{arch}.attention.head_count_kv", (uint)fx.Int("kv_heads"));
        w.AddInt32Array($"{arch}.rope.dimension_sections", [1, 1, 2, 0]);
        w.AddFloat32($"{arch}.rope.freq_base", 1.0e7f);
        w.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", (float)fx.Meta.GetProperty("eps").GetDouble());
        w.AddUInt32($"{arch}.expert_count", (uint)fx.Int("experts"));
        w.AddUInt32($"{arch}.expert_used_count", (uint)fx.Int("top_k"));
        w.AddUInt32($"{arch}.attention.key_length", (uint)fx.Int("head_dim"));
        w.AddUInt32($"{arch}.attention.value_length", (uint)fx.Int("head_dim"));
        w.AddUInt32($"{arch}.expert_feed_forward_length", (uint)fx.Int("moe_inter"));
        w.AddUInt32($"{arch}.expert_shared_feed_forward_length", (uint)fx.Int("shared_inter"));
        w.AddUInt32($"{arch}.ssm.conv_kernel", (uint)fx.Int("conv_k"));
        w.AddUInt32($"{arch}.ssm.state_size", (uint)fx.Int("dk"));
        w.AddUInt32($"{arch}.ssm.group_count", (uint)fx.Int("nk"));
        w.AddUInt32($"{arch}.ssm.time_step_rank", (uint)fx.Int("nv"));
        w.AddUInt32($"{arch}.ssm.inner_size", (uint)(fx.Int("dv") * fx.Int("nv")));
        w.AddUInt32($"{arch}.full_attention_interval", 4);
        w.AddUInt32($"{arch}.rope.dimension_count", (uint)fx.Int("rope_dim"));
        w.AddUInt32($"{arch}.hyper_connection.count", (uint)fx.Int("hc_count"));
        w.AddUInt32($"{arch}.hyper_connection.low_rank", (uint)fx.Int("hc_lowrank"));
        w.AddUInt32($"{arch}.attention.indexer.head_count", (uint)fx.Int("idx_heads"));
        w.AddUInt32($"{arch}.attention.indexer.key_length", (uint)fx.Int("idx_dim"));
        w.AddUInt32($"{arch}.attention.indexer.top_k", (uint)(budgetTokens ?? fx.Int("budget")));
        var ratios = new int[layers];
        for (int i = 0; i < layers; i++) ratios[i] = (i + 1) % 4 == 0 ? blockSize : 0;
        w.AddInt32Array($"{arch}.attention.compress_ratios", ratios);

        // zero-based block indices (HF ple_layer_ids=[2] is 1-based); fixtures with several modules list them in "ple_layers"
        int[] pleLayers = fx.Meta.TryGetProperty("ple_layers", out var pl) ? pl.EnumerateArray().Select(e => e.GetInt32()).ToArray() : [1];
        w.AddInt32Array($"{arch}.ple.layers", pleLayers);
        if (!omitImageTokenId && fx.Meta.TryGetProperty("image_token_id", out var imageId))
            w.AddUInt32($"{arch}.ple.image_token_id", (uint)imageId.GetInt32());
        w.AddUInt32($"{arch}.ple.ngram_size", (uint)fx.Int("ngram"));
        w.AddUInt32($"{arch}.ple.heads_per_ngram", (uint)fx.Int("heads_per_ngram"));
        w.AddUInt32($"{arch}.ple.conv_kernel", (uint)fx.Int("ple_conv_k"));
        w.AddUInt32($"{arch}.ple.eos_token_id", (uint)fx.Int("eos"));
        w.AddUInt32($"{arch}.embedding_length_per_layer_input", (uint)fx.Shape("per_layer_token_embd.weight")[1]);
        w.AddUInt64Array($"{arch}.ple.layer_multipliers", fx.I64("ple.layer_multipliers").Select(v => unchecked((ulong)v)).ToArray());
        w.AddUInt64Array($"{arch}.ple.head_offsets", fx.I64("ple.head_offsets").Select(v => (ulong)v).ToArray());
        w.AddUInt64Array($"{arch}.ple.head_vocab_sizes", fx.I64("ple.head_vocab_sizes").Select(v => (ulong)v).ToArray());

        int vocab = fx.Int("vocab");
        w.AddString("tokenizer.ggml.model", "llama");
        w.AddStringArray("tokenizer.ggml.tokens", Enumerable.Range(0, vocab).Select(i => i == 5 ? "<eos>" : $"tok{i}").ToArray());
        w.AddFloat32Array("tokenizer.ggml.scores", new float[vocab]);
        w.AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(1, vocab).ToArray());
        w.AddUInt32("tokenizer.ggml.bos_token_id", 1);
        w.AddUInt32("tokenizer.ggml.eos_token_id", 5);
        w.AddUInt32("tokenizer.ggml.unknown_token_id", 0);

        foreach (string name in fx.Names.Where(IsTensor))
        {
            int[] shape = fx.Shape(name);
            int[] dims = shape.Reverse().ToArray();          // numpy [out, in] row-major == GGUF ne [in, out]
            float[] data = fx.F32(name);
            var qt = QuantizationType.F32;
            if (quantize) qt = ChooseQuant(name, shape);
            byte[] bytes;
            if (qt == QuantizationType.F32)
            {
                bytes = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            }
            else if (qt == QuantizationType.BF16)
            {
                bytes = new byte[data.Length * 2];
                for (int i = 0; i < data.Length; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(data[i]);
                    ushort bf = (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);   // round-to-nearest-even
                    bytes[2 * i] = (byte)bf; bytes[2 * i + 1] = (byte)(bf >> 8);
                }
            }
            else
            {
                bytes = DotLLM.Cpu.Kernels.Quantize.FromFloat32(data, data.Length, qt);
            }
            w.AddTensor(name, dims, (uint)qt, bytes);
        }
        return w.Build();
    }
}
