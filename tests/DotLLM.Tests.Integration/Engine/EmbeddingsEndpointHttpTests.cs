using System.Buffers.Binary;
using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using DotLLM.Server;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// End-to-end HTTP tests for <c>POST /v1/embeddings</c> (issue #451): the real dotLLM server,
/// booted in-process with a real GGUF model, exercised over a real HTTP socket.
/// </summary>
/// <remarks>
/// These cover the wire contract the OpenAI SDK depends on — response shape, per-item ordering,
/// <c>encoding_format</c>, <c>usage.prompt_tokens</c>, and the error statuses — which the pooling
/// tests in <see cref="EmbeddingLlamaCppParityTests"/> deliberately do not touch. Numerical
/// correctness is anchored there, against llama.cpp; nothing here asserts a vector's values.
/// </remarks>
[Collection("EmbeddingsHttp")]
public sealed class EmbeddingsEndpointHttpTests(EmbeddingsServerFixture fixture, ITestOutputHelper output)
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    private HttpClient Client => fixture.Client ?? throw new InvalidOperationException("no server");

    private void SkipIfNoModel() => Skip.If(fixture.SkipReason is not null, fixture.SkipReason ?? "");

    private async Task<(HttpStatusCode Status, JsonDocument Body)> PostAsync(object request)
    {
        var response = await Client.PostAsJsonAsync("/v1/embeddings", request, Json);
        string text = await response.Content.ReadAsStringAsync();
        return (response.StatusCode, JsonDocument.Parse(text));
    }

    [SkippableFact]
    public async Task Single_string_returns_one_wellformed_embedding()
    {
        SkipIfNoModel();

        var (status, body) = await PostAsync(new { model = fixture.ModelId, input = "The quick brown fox." });
        Assert.Equal(HttpStatusCode.OK, status);

        var root = body.RootElement;
        Assert.Equal("list", root.GetProperty("object").GetString());
        Assert.False(string.IsNullOrEmpty(root.GetProperty("model").GetString()));

        var data = root.GetProperty("data");
        Assert.Equal(1, data.GetArrayLength());
        Assert.Equal("embedding", data[0].GetProperty("object").GetString());
        Assert.Equal(0, data[0].GetProperty("index").GetInt32());

        var vector = data[0].GetProperty("embedding");
        Assert.Equal(JsonValueKind.Array, vector.ValueKind);
        Assert.Equal(fixture.HiddenSize, vector.GetArrayLength());

        int promptTokens = root.GetProperty("usage").GetProperty("prompt_tokens").GetInt32();
        Assert.True(promptTokens > 0);
        Assert.Equal(promptTokens, root.GetProperty("usage").GetProperty("total_tokens").GetInt32());

        // Default is L2-normalised, as OpenAI's embeddings are.
        double norm = 0;
        foreach (var x in vector.EnumerateArray()) norm += Math.Pow(x.GetDouble(), 2);
        Assert.Equal(1.0, Math.Sqrt(norm), 4);
    }

    [SkippableFact]
    public async Task Array_of_strings_returns_one_embedding_per_item_in_order()
    {
        SkipIfNoModel();

        string[] inputs = ["alpha", "beta gamma delta", "epsilon"];
        var (status, body) = await PostAsync(new { model = fixture.ModelId, input = inputs });
        Assert.Equal(HttpStatusCode.OK, status);

        var data = body.RootElement.GetProperty("data");
        Assert.Equal(3, data.GetArrayLength());
        for (int i = 0; i < 3; i++)
            Assert.Equal(i, data[i].GetProperty("index").GetInt32());

        // Distinct inputs must give distinct vectors — a handler that embedded only the first
        // item and copied it would otherwise pass every structural assertion above.
        float[] a = ReadVector(data[0].GetProperty("embedding"));
        float[] b = ReadVector(data[1].GetProperty("embedding"));
        Assert.False(a.SequenceEqual(b), "embeddings for different inputs are byte-identical.");
    }

    /// <summary>
    /// <c>usage.prompt_tokens</c> must be the sum over items. Verified against the server's own
    /// <c>/v1/tokenize</c> endpoint rather than a hard-coded number, so the assertion stays true
    /// if the fixture model changes.
    /// </summary>
    [SkippableFact]
    public async Task Usage_prompt_tokens_is_the_sum_over_items()
    {
        SkipIfNoModel();

        string[] inputs = ["one two three", "four five"];
        int expected = 0;
        foreach (string text in inputs)
        {
            var r = await Client.PostAsJsonAsync("/v1/tokenize", new { text }, Json);
            using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
            expected += doc.RootElement.GetProperty("count").GetInt32();
        }
        Assert.True(expected > 2, "fixture inputs must tokenise to more than one token each.");

        var (status, body) = await PostAsync(new { model = fixture.ModelId, input = inputs });
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(expected, body.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    [SkippableFact]
    public async Task Pretokenised_ids_produce_the_same_vector_as_the_equivalent_text()
    {
        SkipIfNoModel();

        const string text = "dotLLM embeddings round trip";
        var tokenResponse = await Client.PostAsJsonAsync("/v1/tokenize", new { text }, Json);
        using var tokenDoc = JsonDocument.Parse(await tokenResponse.Content.ReadAsStringAsync());
        int[] tokens = tokenDoc.RootElement.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray();

        var (s1, b1) = await PostAsync(new { model = fixture.ModelId, input = text });
        var (s2, b2) = await PostAsync(new { model = fixture.ModelId, input = tokens });
        Assert.Equal(HttpStatusCode.OK, s1);
        Assert.Equal(HttpStatusCode.OK, s2);

        // A flat int array is ONE sequence, not one per token.
        Assert.Equal(1, b2.RootElement.GetProperty("data").GetArrayLength());

        float[] fromText = ReadVector(b1.RootElement.GetProperty("data")[0].GetProperty("embedding"));
        float[] fromIds = ReadVector(b2.RootElement.GetProperty("data")[0].GetProperty("embedding"));
        Assert.Equal(fromText, fromIds);
    }

    [SkippableFact]
    public async Task Nested_token_arrays_produce_one_embedding_each()
    {
        SkipIfNoModel();

        int[][] inputs = [[1, 2, 3], [4, 5]];
        var (status, body) = await PostAsync(new { model = fixture.ModelId, input = inputs });
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(2, body.RootElement.GetProperty("data").GetArrayLength());
        Assert.Equal(5, body.RootElement.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    /// <summary>
    /// base64 must decode to exactly the float payload the <c>float</c> format returns — the
    /// encoding is raw little-endian float32, which is what the OpenAI SDK's numpy path expects.
    /// </summary>
    [SkippableFact]
    public async Task Base64_encoding_decodes_to_the_same_floats()
    {
        SkipIfNoModel();

        const string text = "base64 round trip";
        var (s1, b1) = await PostAsync(new { model = fixture.ModelId, input = text, encoding_format = "float" });
        var (s2, b2) = await PostAsync(new { model = fixture.ModelId, input = text, encoding_format = "base64" });
        Assert.Equal(HttpStatusCode.OK, s1);
        Assert.Equal(HttpStatusCode.OK, s2);

        float[] asFloats = ReadVector(b1.RootElement.GetProperty("data")[0].GetProperty("embedding"));

        var encoded = b2.RootElement.GetProperty("data")[0].GetProperty("embedding");
        Assert.Equal(JsonValueKind.String, encoded.ValueKind);
        byte[] bytes = Convert.FromBase64String(encoded.GetString()!);
        Assert.Equal(asFloats.Length * sizeof(float), bytes.Length);

        var decoded = new float[asFloats.Length];
        for (int i = 0; i < decoded.Length; i++)
            decoded[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * sizeof(float)));
        Assert.Equal(asFloats, decoded);
    }

    /// <summary>
    /// The three pooling modes must produce three different vectors over the wire. If the
    /// <c>pooling</c> parameter were ignored, every other test here would still pass.
    /// </summary>
    [SkippableFact]
    public async Task Pooling_parameter_changes_the_result()
    {
        SkipIfNoModel();

        const string text = "pooling must actually be honoured";
        var vectors = new List<float[]>();
        foreach (string pooling in new[] { "last", "mean", "cls" })
        {
            var (status, body) = await PostAsync(new { model = fixture.ModelId, input = text, pooling });
            Assert.Equal(HttpStatusCode.OK, status);
            vectors.Add(ReadVector(body.RootElement.GetProperty("data")[0].GetProperty("embedding")));
        }

        Assert.False(vectors[0].SequenceEqual(vectors[1]), "last == mean");
        Assert.False(vectors[1].SequenceEqual(vectors[2]), "mean == cls");
        Assert.False(vectors[0].SequenceEqual(vectors[2]), "last == cls");

        // No explicit pooling must equal the resolved default. This fixture's GGUF declares no
        // pooling_type, so the default is `last`.
        var (defaultStatus, defaultBody) = await PostAsync(new { model = fixture.ModelId, input = text });
        Assert.Equal(HttpStatusCode.OK, defaultStatus);
        float[] byDefault = ReadVector(defaultBody.RootElement.GetProperty("data")[0].GetProperty("embedding"));
        Assert.Equal(vectors[0], byDefault);
    }

    [SkippableFact]
    public async Task Normalize_false_returns_an_unnormalised_vector()
    {
        SkipIfNoModel();

        var (status, body) = await PostAsync(new { model = fixture.ModelId, input = "norm off", normalize = false });
        Assert.Equal(HttpStatusCode.OK, status);

        double norm = 0;
        foreach (var x in body.RootElement.GetProperty("data")[0].GetProperty("embedding").EnumerateArray())
            norm += Math.Pow(x.GetDouble(), 2);
        norm = Math.Sqrt(norm);
        output.WriteLine($"un-normalised norm = {norm:F4}");
        Assert.True(Math.Abs(norm - 1.0) > 0.1, $"expected a non-unit norm, got {norm:F6}.");
    }

    [SkippableTheory]
    [InlineData("{\"input\": \"\"}")]
    [InlineData("{\"input\": []}")]
    [InlineData("{\"input\": null}")]
    [InlineData("{\"input\": 42}")]
    [InlineData("{\"input\": [\"a\", 1]}")]
    [InlineData("{\"input\": [1, 2, 99999999]}")]
    [InlineData("{\"input\": \"hi\", \"encoding_format\": \"utf8\"}")]
    [InlineData("{\"input\": \"hi\", \"pooling\": \"sum\"}")]
    [InlineData("{\"input\": \"hi\", \"dimensions\": 64}")]
    public async Task Malformed_requests_return_400_with_a_message(string json)
    {
        SkipIfNoModel();

        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await Client.PostAsync("/v1/embeddings", content);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);

        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync());

        // #452 replaced the flat {"error": "<string>"} body with the SDK envelope
        // {"error": {"message", "type", ...}} — neither SDK can read .type/.code/.param off a bare
        // string. Assert the envelope's shape, not just its message, so the next shape change fails
        // here rather than silently reading a null message. (#523: this test kept the old shape and
        // threw on every case for as long as the fixture happened to be present.)
        var envelope = body.RootElement.GetProperty("error");
        Assert.Equal(JsonValueKind.Object, envelope.ValueKind);
        Assert.False(string.IsNullOrWhiteSpace(envelope.GetProperty("type").GetString()));

        string error = envelope.GetProperty("message").GetString() ?? "";
        Assert.False(string.IsNullOrWhiteSpace(error));

        // The 400 must come from validating THIS request, not from model activation — an earlier
        // revision of these tests passed only because the request named a model the server did
        // not have, so every payload returned 400 regardless of its contents.
        Assert.DoesNotContain("Model not found", error, StringComparison.Ordinal);
        Assert.DoesNotContain("No model loaded", error, StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task A_prompt_longer_than_the_context_returns_400_not_500()
    {
        SkipIfNoModel();

        int[] tooMany = Enumerable.Repeat(1, fixture.MaxSequenceLength + 1).ToArray();
        using var content = new StringContent(
            JsonSerializer.Serialize(new { input = tooMany }, Json), Encoding.UTF8, "application/json");
        var response = await Client.PostAsync("/v1/embeddings", content);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    /// <summary>
    /// An embedding computed while a completion is generating must be byte-identical to the same
    /// embedding computed alone. The model's scratch buffers are shared mutable state, so this is
    /// the discriminating test for the request gate: remove <c>ExecuteAsync</c> from the handler
    /// and the two forwards interleave through the same <c>_state</c> buffers.
    /// </summary>
    /// <remarks>
    /// This covers the direct-generator path, which is the default. When the continuous-batch
    /// scheduler is enabled the endpoint refuses with 503 instead, because the scheduler runs
    /// forward passes outside the gate — see <c>EmbeddingsEndpoint</c>.
    /// </remarks>
    [SkippableFact]
    public async Task An_embedding_taken_during_a_generation_matches_the_sequential_result()
    {
        SkipIfNoModel();

        const string text = "concurrency must not corrupt the scratch buffers";

        var (baselineStatus, baselineBody) = await PostAsync(new { model = fixture.ModelId, input = text });
        Assert.Equal(HttpStatusCode.OK, baselineStatus);
        float[] sequential = ReadVector(baselineBody.RootElement.GetProperty("data")[0].GetProperty("embedding"));

        var generation = Client.PostAsJsonAsync("/v1/completions",
            new { model = fixture.ModelId, prompt = "Once upon a time", max_tokens = 48 }, Json);
        var embedding = PostAsync(new { model = fixture.ModelId, input = text });

        await Task.WhenAll(generation, embedding);
        var (status, body) = await embedding;
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(HttpStatusCode.OK, (await generation).StatusCode);

        float[] concurrent = ReadVector(body.RootElement.GetProperty("data")[0].GetProperty("embedding"));
        Assert.Equal(sequential, concurrent);
    }

    private static float[] ReadVector(JsonElement array)
        => array.EnumerateArray().Select(x => x.GetSingle()).ToArray();
}
