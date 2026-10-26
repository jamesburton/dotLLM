using System.IO;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1289: <c>/v1/models</c> carries llama.cpp's <c>meta</c> object (plus the post-fit <c>n_ctx</c>) and
/// <c>/health</c> carries the served context, slots and prompt-cache state, under llama.cpp/azllm field names.
/// </summary>
public sealed class ModelMetaHealthTests
{
    private static ModelConfig Config(int ctx) => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = 32,
        HiddenSize = 4,
        IntermediateSize = 16,
        NumLayers = 1,
        NumAttentionHeads = 2,
        NumKvHeads = 2,
        HeadDim = 2,
        MaxSequenceLength = ctx,
        PositionEncodingType = PositionEncodingType.ALiBi,
        RoPEConfig = null,
    };

    /// <summary>One Q8_0 [64,4] tensor (256 elements, 272 bytes) and one F32 [4] tensor (16 bytes).</summary>
    private static GgufFile WriteGguf(string path)
    {
        var g = new GgufWriter();
        g.AddString("general.architecture", "llama");
        g.AddUInt32("general.alignment", 32);
        g.AddUInt32("llama.context_length", 4096);
        g.AddString("tokenizer.ggml.model", "gpt2");
        g.AddTensor("blk.0.attn_q.weight", [64, 4], 8, new byte[272]);
        g.AddTensor("output_norm.weight", [4], 0, new byte[16]);
        File.WriteAllBytes(path, g.Build());
        return GgufFile.Open(path);
    }

    [Fact]
    public void Meta_ReportsServedAndTrainedContext_SizeParamsAndMeasuredFtype()
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var gguf = WriteGguf(path))
            {
                var meta = ModelMeta.Build(Config(1024), gguf)!;

                Assert.Equal(1024, meta.NCtx);
                Assert.Equal(4096, meta.NCtxTrain);
                Assert.Equal(260, meta.NParams);
                Assert.Equal(288, meta.Size);
                Assert.Equal(2, meta.VocabType);
                Assert.Equal(32, meta.NVocab);
                Assert.Equal(4, meta.NEmbd);
                Assert.Equal("Q8_0 - 8.86 bpw", meta.Ftype);

                // Field names are what llama.cpp clients read.
                string json = JsonSerializer.Serialize(meta, ServerJsonContext.Default.ModelMetaDto);
                foreach (var name in new[] { "vocab_type", "n_vocab", "n_ctx", "n_ctx_train", "n_embd", "n_params", "size", "ftype" })
                    Assert.Contains($"\"{name}\":", json);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Meta_IsNullWithoutAModel_AndOmittedFromTheList()
    {
        Assert.Null(ModelMeta.Build(null, null));

        using var state = new ServerState { Options = new ServerOptions { Model = "m", ModelId = "m" } };
        string json = JsonSerializer.Serialize(ModelEndpoint.BuildList(state), ServerJsonContext.Default.ModelListResponse);
        Assert.DoesNotContain("\"meta\"", json);
    }

    [Fact]
    public void Health_IsBareOk_WhenNoModelIsLoaded()
    {
        using var state = new ServerState { Options = new ServerOptions { Model = "m", ModelId = "m" } };

        string json = JsonSerializer.Serialize(HealthEndpoint.BuildHealth(state), ServerJsonContext.Default.HealthResponse);

        Assert.Equal("{\"status\":\"ok\"}", json);
    }

    [Fact]
    public void Health_CarriesContextSlotsAndPromptCache_WhenLoaded()
    {
        using var state = new ServerState
        {
            Options = new ServerOptions { Model = "m", ModelId = "m" },
            Config = Config(2048),
            IsReady = true,
        };

        var health = HealthEndpoint.BuildHealth(state);

        Assert.Equal("ok", health.Status);
        Assert.Equal(2048, health.MaxContext);
        Assert.Equal(2048, health.NCtxTrain); // no GGUF: falls back to the served window
        Assert.Equal(1, health.Slots);
        Assert.False(health.PromptCacheActive);
        string json = JsonSerializer.Serialize(health, ServerJsonContext.Default.HealthResponse);
        Assert.Contains("\"max_context\":2048", json);
        Assert.Contains("\"prompt_cache_active\":false", json);
    }
}
