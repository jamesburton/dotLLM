using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Tests.Unit.Server;

internal sealed class HfApplicationFixture : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-hf-app-{Guid.NewGuid():N}");
    private const string Embedding = "model.embed_tokens.weight";
    private const string Template = "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";

    internal string WriteCheckpoint(string name, bool sharded = false)
    {
        string directory = Path.Combine(_scratch, name);
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), """
            {"model_type":"llama","hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,
             "num_attention_heads":1,"num_key_value_heads":1,"vocab_size":258,"max_position_embeddings":64,
             "rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":true,"bos_token_id":256,"eos_token_id":257}
            """);
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < 256; i++) vocab.Add($"<0x{i.ToString("X2", CultureInfo.InvariantCulture)}>", i);
        vocab.Add("<s>", 256); vocab.Add("</s>", 257);
        File.WriteAllText(Path.Combine(directory, "tokenizer.json"), JsonSerializer.Serialize(new
        {
            model = new { type = "BPE", byte_fallback = true, vocab, merges = Array.Empty<string>() },
            pre_tokenizer = new { type = "Metaspace", replacement = "▁", prepend_scheme = "never" },
            added_tokens = new[] { new { id = 256, content = "<s>", special = true }, new { id = 257, content = "</s>", special = true } },
        }));
        File.WriteAllText(Path.Combine(directory, "tokenizer_config.json"), JsonSerializer.Serialize(new { chat_template = Template }));
        var builder = new SafetensorsFixtureBuilder();
        var random = new Random(867);
        void Weight(string tensor, int rows, int columns)
        {
            Half[] data = new Half[rows * columns];
            for (int i = 0; i < data.Length; i++) data[i] = (Half)((random.NextDouble() - 0.5) * 0.1);
            builder.AddRaw(tensor, "F16", [rows, columns], MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
        }
        void Norm(string tensor) => builder.AddFloat32(tensor, [32], Enumerable.Repeat(1f, 32).ToArray());
        Weight(Embedding, 258, 32);
        Norm("model.norm.weight");
        const string layer = "model.layers.0";
        Norm($"{layer}.input_layernorm.weight");
        Norm($"{layer}.post_attention_layernorm.weight");
        foreach (string projection in new[] { "q_proj", "k_proj", "v_proj", "o_proj" })
            Weight($"{layer}.self_attn.{projection}.weight", 32, 32);
        Weight($"{layer}.mlp.gate_proj.weight", 64, 32);
        Weight($"{layer}.mlp.up_proj.weight", 64, 32);
        Weight($"{layer}.mlp.down_proj.weight", 32, 64);
        builder.WriteTo(Path.Combine(directory, "model.safetensors"));
        if (sharded)
        {
            var shards = new[] { new SafetensorsFixtureBuilder(), new SafetensorsFixtureBuilder() };
            var map = new Dictionary<string, string>(StringComparer.Ordinal);
            using (var source = SafetensorsFile.Open(Path.Combine(directory, "model.safetensors")))
            {
                int index = 0;
                foreach (var (tensorName, tensor) in source.TensorsByName)
                {
                    int shard = index++ % 2;
                    shards[shard].AddRaw(tensorName, tensor.DType.ToString(), tensor.Shape.ToArray(), source.GetTensorSpan(tensorName).ToArray());
                    map.Add(tensorName, $"model-0000{shard + 1}-of-00002.safetensors");
                }
            }
            for (int i = 0; i < shards.Length; i++)
                shards[i].WriteTo(Path.Combine(directory, $"model-0000{i + 1}-of-00002.safetensors"));
            File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"), JsonSerializer.Serialize(new { weight_map = map }));
            File.Delete(Path.Combine(directory, "model.safetensors"));
        }
        return directory;
    }

    internal string WriteGgufForLifetimeCheck()
    {
        string directory = WriteCheckpoint("gguf-lifetime");
        using var source = SafetensorsFile.Open(Path.Combine(directory, "model.safetensors"));
        var writer = new GgufWriter()
            .AddString("general.architecture", "llama").AddUInt32("llama.context_length", 64)
            .AddUInt32("llama.embedding_length", 32).AddUInt32("llama.block_count", 1)
            .AddUInt32("llama.feed_forward_length", 64).AddUInt32("llama.attention.head_count", 1)
            .AddUInt32("llama.attention.head_count_kv", 1).AddUInt32("llama.rope.dimension_count", 32)
            .AddFloat32("llama.attention.layer_norm_rms_epsilon", 0.00001f).AddFloat32("llama.rope.freq_base", 10000)
            .AddString("tokenizer.ggml.model", "llama")
            .AddStringArray("tokenizer.ggml.tokens", Enumerable.Range(0, 256).Select(i => $"<0x{i:X2}>").Concat(["<s>", "</s>"]).ToArray())
            .AddFloat32Array("tokenizer.ggml.scores", new float[258])
            .AddInt32Array("tokenizer.ggml.token_type", Enumerable.Repeat(6, 256).Concat([3, 3]).ToArray())
            .AddUInt32("tokenizer.ggml.bos_token_id", 256).AddUInt32("tokenizer.ggml.eos_token_id", 257);
        string[] hf = [Embedding, "model.norm.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight"];
        string[] gguf = ["token_embd.weight", "output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight",
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"];
        for (int i = 0; i < hf.Length; i++)
        {
            var tensor = source.TensorsByName[hf[i]];
            writer.AddTensor(gguf[i], tensor.Shape.Reverse().ToArray(), tensor.DType == SafetensorsDType.F16 ? 1u : 0u,
                source.GetTensorSpan(hf[i]).ToArray());
        }
        string path = Path.Combine(directory, "lifetime.gguf");
        File.WriteAllBytes(path, writer.Build());
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(_scratch)) Directory.Delete(_scratch, recursive: true);
    }
}
