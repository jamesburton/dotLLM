using System.Net;
using System.Text;
using System.Text.Json;
using DotLLM.Server;
using DotLLM.Tests.Integration.Fixtures;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// <c>POST /v1/embeddings</c> served by a real BERT-class encoder (all-MiniLM-L6-v2 f16) through the
/// full server path, including <c>dimensions</c> truncation (#739/#740).
/// </summary>
public sealed class EncoderEmbeddingsServerTests
{
    private static async Task<(HttpClient, Microsoft.AspNetCore.Builder.WebApplication)?> Boot(string device = "cpu")
    {
        var loc = TestFixtureResolver.ResolveFile("DOTLLM_MINILM_GGUF", "second-state",
            "All-MiniLM-L6-v2-Embedding-GGUF", "all-MiniLM-L6-v2-ggml-model-f16.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("all-MiniLM-L6-v2 f16"));
        var state = ServerStartup.LoadModel(loc.Path!, new ServerOptions { Model = loc.Path!, Device = device, ModelId = "minilm" });
        var app = ServerStartup.BuildApp(state, ["--urls", "http://127.0.0.1:0"]);
        await app.StartAsync();
        string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        return (new HttpClient { BaseAddress = new Uri(address) }, app);
    }

    private static async Task<float[]> Embed(HttpClient c, string json)
    {
        var r = await c.PostAsync("/v1/embeddings", new StringContent(json, Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, r.StatusCode);
        using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
        return doc.RootElement.GetProperty("data")[0].GetProperty("embedding").EnumerateArray().Select(x => x.GetSingle()).ToArray();
    }

    [SkippableFact]
    public async Task Encoder_serves_embeddings_with_dimensions_truncation()
    {
        var (client, app) = (await Boot())!.Value;
        try
        {
            var full = await Embed(client, "{\"input\":\"hello world\"}");
            Assert.Equal(384, full.Length);
            Assert.InRange(full.Sum(x => (double)x * x), 0.999, 1.001);

            var trunc = await Embed(client, "{\"input\":\"hello world\",\"dimensions\":128}");
            Assert.Equal(128, trunc.Length);
            Assert.InRange(trunc.Sum(x => (double)x * x), 0.999, 1.001);
            // Truncate-then-renormalise: direction of the leading 128 components is preserved.
            double dot = 0, nf = 0;
            for (int i = 0; i < 128; i++) { dot += full[i] * trunc[i]; nf += full[i] * full[i]; }
            Assert.InRange(dot / Math.Sqrt(nf), 0.9999, 1.0001);

            var bad = await client.PostAsync("/v1/embeddings",
                new StringContent("{\"input\":\"x\",\"dimensions\":99999}", Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.BadRequest, bad.StatusCode);
        }
        finally { client.Dispose(); await app.StopAsync(); await app.DisposeAsync(); }
    }

    [SkippableFact]
    public async Task Ollama_api_embed_matches_v1_embeddings()
    {
        var (client, app) = (await Boot())!.Value;
        try
        {
            var v1 = await Embed(client, "{\"input\":\"hello world\",\"dimensions\":64}");
            var r = await client.PostAsync("/api/embed",
                new StringContent("{\"model\":\"minilm\",\"input\":[\"hello world\",\"second\"],\"dimensions\":64}", Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.OK, r.StatusCode);
            using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
            var embs = doc.RootElement.GetProperty("embeddings");
            Assert.Equal(2, embs.GetArrayLength());
            var first = embs[0].EnumerateArray().Select(x => x.GetSingle()).ToArray();
            Assert.Equal(v1, first);
            Assert.True(doc.RootElement.GetProperty("prompt_eval_count").GetInt32() > 0);

            var legacy = await client.PostAsync("/api/embeddings",
                new StringContent("{\"model\":\"minilm\",\"prompt\":\"hello world\"}", Encoding.UTF8, "application/json"));
            Assert.Equal(HttpStatusCode.OK, legacy.StatusCode);
            using var ld = JsonDocument.Parse(await legacy.Content.ReadAsStringAsync());
            Assert.Equal(384, ld.RootElement.GetProperty("embedding").GetArrayLength());
        }
        finally { client.Dispose(); await app.StopAsync(); await app.DisposeAsync(); }
    }

    [SkippableFact]
    public void Encoder_on_gpu_device_fails_loudly_instead_of_falling_back()
    {
        var loc = TestFixtureResolver.ResolveFile("DOTLLM_MINILM_GGUF", "second-state",
            "All-MiniLM-L6-v2-Embedding-GGUF", "all-MiniLM-L6-v2-ggml-model-f16.gguf");
        Skip.If(!loc.Found, loc.SkipMessage("all-MiniLM-L6-v2 f16"));
        Assert.Throws<NotSupportedException>(() =>
            ServerStartup.LoadModel(loc.Path!, new ServerOptions { Model = loc.Path!, Device = "vulkan", ModelId = "minilm" }));
    }
}
