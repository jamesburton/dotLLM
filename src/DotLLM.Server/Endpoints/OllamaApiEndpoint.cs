using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using DotLLM.HuggingFace;
using DotLLM.Server.RateLimiting;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Reasoning;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Ollama-compatible <c>/api/*</c> surface (issue #720): <c>chat</c>, <c>generate</c>, <c>tags</c>, <c>show</c>, <c>ps</c>, <c>version</c>,
/// <c>pull</c> and <c>delete</c>, with ollama's NDJSON streaming, so ollama clients (Open WebUI, Continue, ollama-python/js, LangChain's ChatOllama) can
/// point at a dotLLM server unchanged.
/// </summary>
/// <remarks>
/// <para>
/// A shim over the existing paths: model names are profiles / local models / ollama names (<see cref="ServerState.EnsureActiveAsync"/>), requests
/// run through the same generator (streaming) or scheduler (non-streaming) as <c>/v1/*</c>, and the profile's system prompt and sampling defaults apply.
/// Not covered, and answered with a clear 501: <c>create</c>, <c>copy</c>, <c>push</c>, and tools / images
/// inside chat (use <c>/v1/chat/completions</c>). Thinking is supported (#767): <c>think</c> (bool, or a
/// <c>low</c>/<c>medium</c>/<c>high</c> level) goes to the chat template, and the model's reasoning comes back in
/// <c>message.thinking</c> (<c>thinking</c> on <c>/api/generate</c>), streamed as separate chunks. <c>pull</c> and <c>delete</c> need <c>--allow-model-admin</c> like <c>/v1/models/*</c>.
/// </para>
/// <para>Responses are written with <see cref="Utf8JsonWriter"/> directly: ollama's shapes are small and this keeps the surface reflection-free.</para>
/// </remarks>
public static class OllamaApiEndpoint
{
    /// <summary>The ollama API level this shim emulates (clients gate features on <c>/api/version</c>).</summary>
    public const string EmulatedVersion = "0.6.0";

    private const string Ndjson = "application/x-ndjson";

    public static void Map(WebApplication app)
    {
        app.MapGet("/api/version", (HttpContext c) => WriteJson(c, 200, w => { w.WriteString("version", EmulatedVersion); }));
        app.MapGet("/api/tags", (ServerState state, HttpContext c) => WriteJson(c, 200, w => Tags(w, state)));
        app.MapGet("/api/ps", (ServerState state, HttpContext c) => WriteJson(c, 200, w => Ps(w, state)));
        app.MapPost("/api/show", ShowAsync);
        app.MapPost("/api/chat", (ServerState s, HttpContext c) => GenerateAsync(s, c, chat: true));
        app.MapPost("/api/generate", (ServerState s, HttpContext c) => GenerateAsync(s, c, chat: false));
        app.MapPost("/api/pull", PullAsync);
        app.MapDelete("/api/delete", DeleteAsync);
        app.MapPost("/api/embed", (ServerState s, HttpContext c) => EmbedAsync(s, c, legacy: false));
        app.MapPost("/api/embeddings", (ServerState s, HttpContext c) => EmbedAsync(s, c, legacy: true));
        foreach (string path in new[] { "/api/create", "/api/copy", "/api/push" })
            app.MapPost(path, (HttpContext c) => Error(c, 501,
                $"{c.Request.Path} is not implemented by dotLLM's ollama compatibility layer. Use `dotllm model create` / `dotllm model cp`, or the OpenAI-compatible /v1 API."));
    }

    // ───────────────────────────────────── helpers ─────────────────────────────────────

    /// <summary>
    /// <c>POST /api/embed</c> (<c>input</c> string or array, optional <c>dimensions</c>) and the legacy
    /// <c>POST /api/embeddings</c> (<c>prompt</c> string -> single un-normalised <c>embedding</c>), over the
    /// same core as <c>/v1/embeddings</c> (#740). The ollama <c>truncate</c> option is accepted; over-length
    /// inputs are rejected rather than truncated.
    /// </summary>
    private static async Task EmbedAsync(ServerState state, HttpContext c, bool legacy)
    {
        using var doc = await ReadBodyAsync(c);
        if (doc is null) return;
        var root = doc.RootElement;
        var ct = c.RequestAborted;

        string? name = ModelName(root);
        if (string.IsNullOrWhiteSpace(name)) { await Error(c, 400, "model is required"); return; }

        JsonElement input;
        if (legacy)
        {
            string? prompt = Str(root, "prompt");
            if (string.IsNullOrEmpty(prompt)) { await Error(c, 400, "prompt is required"); return; }
            input = JsonSerializer.SerializeToElement(prompt);
        }
        else if (!root.TryGetProperty("input", out input)) { await Error(c, 400, "input is required"); return; }

        int? dims = root.TryGetProperty("dimensions", out var d) && d.ValueKind == JsonValueKind.Number ? d.GetInt32() : null;
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var request = new DotLLM.Server.Models.EmbeddingRequest
        {
            Input = input, Model = name, Dimensions = dims, Normalize = legacy ? false : null,
        };
        var result = await EmbeddingsEndpoint.ComputeAsync(state, request, ct);
        if (result.Error is not null)
        {
            await Error(c, result.Status == 400 && result.Error.Contains("not found", StringComparison.OrdinalIgnoreCase) ? 404 : result.Status, result.Error);
            return;
        }
        double totalNs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds * 1e6;

        await WriteJson(c, 200, w =>
        {
            if (legacy)
            {
                w.WriteStartArray("embedding");
                foreach (float f in result.Vectors![0]) w.WriteNumberValue(f);
                w.WriteEndArray();
                return;
            }
            w.WriteString("model", name);
            w.WriteStartArray("embeddings");
            foreach (var v in result.Vectors!)
            {
                w.WriteStartArray();
                foreach (float f in v) w.WriteNumberValue(f);
                w.WriteEndArray();
            }
            w.WriteEndArray();
            w.WriteNumber("total_duration", (long)totalNs);
            w.WriteNumber("prompt_eval_count", result.PromptTokens);
        });
    }

    private static Task WriteJson(HttpContext c, int status, Action<Utf8JsonWriter> body)
    {
        c.Response.StatusCode = status;
        c.Response.ContentType = "application/json; charset=utf-8";
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms)) { w.WriteStartObject(); body(w); w.WriteEndObject(); }
        return c.Response.Body.WriteAsync(ms.ToArray(), c.RequestAborted).AsTask();
    }

    private static Task Error(HttpContext c, int status, string message) =>
        WriteJson(c, status, w => w.WriteString("error", message));

    private static async Task<JsonDocument?> ReadBodyAsync(HttpContext c)
    {
        try { return await JsonDocument.ParseAsync(c.Request.Body, cancellationToken: c.RequestAborted); }
        catch (JsonException) { await Error(c, 400, "invalid JSON request body"); return null; }
    }

    private static string? Str(JsonElement e, string name) =>
        e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;

    /// <summary>
    /// ollama's <c>think</c>: <c>true</c>/<c>false</c>, or a level string (<c>low</c>/<c>medium</c>/<c>high</c>), which
    /// means "think" and is handed to the template as <c>reasoning_effort</c>. Absent leaves the template default.
    /// </summary>
    internal static void ParseThink(JsonElement root, out bool? think, out string? level)
    {
        think = null;
        level = null;
        if (!root.TryGetProperty("think", out var p))
            return;
        if (p.ValueKind == JsonValueKind.True) think = true;
        else if (p.ValueKind == JsonValueKind.False) think = false;
        else if (p.ValueKind == JsonValueKind.String) { think = true; level = p.GetString(); }
    }

    private static string? ModelName(JsonElement root) => Str(root, "model") ?? Str(root, "name");

    /// <summary>ollama keep_alive: a number of seconds, or a duration string ("5m", "30s", "1h", "-1", "0").</summary>
    internal static double? ParseKeepAlive(JsonElement root)
    {
        if (!root.TryGetProperty("keep_alive", out var v)) return null;
        if (v.ValueKind == JsonValueKind.Number) return v.GetDouble();
        if (v.ValueKind != JsonValueKind.String) return null;
        string s = v.GetString()!.Trim();
        if (s.Length == 0) return null;
        if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double plain)) return plain;
        var m = Regex.Match(s, @"^(-?\d+(?:\.\d+)?)(ms|s|m|h)$");
        if (!m.Success) return null;
        double n = double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
        return m.Groups[2].Value switch { "ms" => n / 1000, "s" => n, "m" => n * 60, _ => n * 3600 };
    }

    private static long Ns(double ms) => (long)(ms * 1_000_000);

    private static readonly Regex QuantRx = new(@"(IQ\d_[A-Z]+|Q\d_K_[A-Z]+|Q\d_K|Q\d_\d|BF16|F16|F32)", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    // ───────────────────────────────────── tags / ps / show ─────────────────────────────────────

    /// <summary>The names a client can use: profiles first, then every local model (ollama-store ones as name:tag).</summary>
    private static List<(string Name, string Path, long Size, DateTimeOffset Modified, string? Quant)> ListNames()
    {
        var result = new List<(string, string, long, DateTimeOffset, string?)>();
        foreach (var (name, profile) in ModelProfileStore.List())
        {
            string? path = ModelResolver.ResolveLocal(profile.From ?? "", null, includeOllama: true);
            long size = path is null ? 0 : ModelResolver.FileLength(path);
            result.Add((name, path ?? profile.From ?? "", size, path is null ? DateTimeOffset.UtcNow : File.GetLastWriteTimeUtc(path), QuantOf(path ?? profile.From)));
        }
        foreach (var m in ModelResolver.EnumerateLocal(includeOllama: true))
        {
            string stem = System.IO.Path.GetFileNameWithoutExtension(m.Filename);
            string name = stem;
            if (m.RepoId.StartsWith("ollama/", StringComparison.Ordinal))
            {
                string safe = m.RepoId["ollama/".Length..];
                if (stem.StartsWith(safe + "-", StringComparison.Ordinal)) name = safe.Replace('_', '/') + ":" + stem[(safe.Length + 1)..];
            }
            if (result.Any(r => r.Item1.Equals(name, StringComparison.OrdinalIgnoreCase))) continue;   // a profile of that name shadows it
            result.Add((name, m.FullPath, m.SizeBytes, m.DownloadedAt, QuantOf(m.Filename)));
        }
        return result;
    }

    private static string? QuantOf(string? text) => text is null ? null : QuantRx.Match(text) is { Success: true } q ? q.Value.ToUpperInvariant() : null;

    private static string Digest(string path, long size)
    {
        byte[] h = System.Security.Cryptography.SHA256.HashData(Encoding.UTF8.GetBytes(path + "|" + size));
        return Convert.ToHexString(h).ToLowerInvariant();
    }

    private static void Details(Utf8JsonWriter w, string? quant)
    {
        w.WriteStartObject("details");
        w.WriteString("parent_model", "");
        w.WriteString("format", "gguf");
        w.WriteString("family", "");
        w.WriteNull("families");
        w.WriteString("parameter_size", "");
        w.WriteString("quantization_level", quant ?? "");
        w.WriteEndObject();
    }

    private static void Tags(Utf8JsonWriter w, ServerState state)
    {
        w.WriteStartArray("models");
        foreach (var (name, path, size, modified, quant) in ListNames())
        {
            w.WriteStartObject();
            w.WriteString("name", name); w.WriteString("model", name);
            w.WriteString("modified_at", modified.ToString("O", CultureInfo.InvariantCulture));
            w.WriteNumber("size", size);
            w.WriteString("digest", Digest(path, size));
            Details(w, quant);
            w.WriteEndObject();
        }
        w.WriteEndArray();
    }

    private static void Ps(Utf8JsonWriter w, ServerState state)
    {
        bool gpu = !string.Equals(state.Options.ResolvedDevice ?? state.Options.Device, "cpu", StringComparison.OrdinalIgnoreCase);
        w.WriteStartArray("models");
        foreach (var m in state.ListResidentModels())
        {
            w.WriteStartObject();
            w.WriteString("name", m.Key); w.WriteString("model", m.Key);
            w.WriteNumber("size", m.EstimatedBytes);
            w.WriteString("digest", Digest(m.Key, m.EstimatedBytes));
            Details(w, QuantOf(state.LoadedModelPath));
            w.WriteString("expires_at", (m.ExpiresInSeconds is { } s ? DateTimeOffset.UtcNow.AddSeconds(s) : DateTimeOffset.UtcNow.AddYears(100)).ToString("O", CultureInfo.InvariantCulture));
            w.WriteNumber("size_vram", gpu ? m.EstimatedBytes : 0);
            w.WriteEndObject();
        }
        w.WriteEndArray();
    }

    private static async Task ShowAsync(ServerState state, HttpContext c)
    {
        using var doc = await ReadBodyAsync(c);
        if (doc is null) return;
        string? name = ModelName(doc.RootElement);
        if (string.IsNullOrWhiteSpace(name)) { await Error(c, 400, "model is required"); return; }

        var profileChain = ModelProfileStore.Resolve(name);
        var profile = profileChain is { } rc ? ModelProfileStore.Merge(rc.Chain) : null;
        string? path = ServerStartup.ResolveModelPath(name, null);
        if (path is null) { await Error(c, 404, $"model '{name}' not found"); return; }

        var mf = new StringBuilder($"FROM {profile?.From ?? path}\n");
        if (profile?.System is { } sys) mf.Append("SYSTEM \"\"\"").Append(sys).Append("\"\"\"\n");
        var prm = new StringBuilder();
        void P(string k, object? v) { if (v is not null) { string line = $"{k} {Convert.ToString(v, CultureInfo.InvariantCulture)}"; mf.Append("PARAMETER ").Append(line).Append('\n'); prm.Append(line).Append('\n'); } }
        P("temperature", profile?.Temperature); P("top_p", profile?.TopP); P("top_k", profile?.TopK); P("min_p", profile?.MinP);
        P("repeat_penalty", profile?.RepeatPenalty); P("num_predict", profile?.MaxTokens); P("seed", profile?.Seed);
        foreach (string stop in profile?.Stop ?? []) P("stop", "\"" + stop + "\"");

        await WriteJson(c, 200, w =>
        {
            w.WriteString("modelfile", mf.ToString());
            w.WriteString("parameters", prm.ToString());
            w.WriteString("template", "");   // dotLLM renders the chat template embedded in the GGUF
            if (profile?.System is { } s2) w.WriteString("system", s2);
            Details(w, QuantOf(path));
            w.WriteStartObject("model_info"); w.WriteEndObject();
            w.WriteStartArray("capabilities"); w.WriteStringValue("completion"); w.WriteEndArray();
            w.WriteString("modified_at", File.GetLastWriteTimeUtc(path).ToString("O", CultureInfo.InvariantCulture));
        });
    }

    // ───────────────────────────────────── chat / generate ─────────────────────────────────────

    /// <summary>Maps ollama's <c>options</c> (and <c>format</c>) over the effective sampling defaults.</summary>
    internal static InferenceOptions BuildOptions(JsonElement root, SamplingDefaults d, ThreadingConfig threading)
    {
        JsonElement o = root.TryGetProperty("options", out var oe) && oe.ValueKind == JsonValueKind.Object ? oe : default;
        float? F(string k) => o.ValueKind == JsonValueKind.Object && o.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : null;
        int? I(string k) => o.ValueKind == JsonValueKind.Object && o.TryGetProperty(k, out var v) && v.ValueKind == JsonValueKind.Number ? (int)v.GetDouble() : null;

        var stops = new List<string>();
        if (o.ValueKind == JsonValueKind.Object && o.TryGetProperty("stop", out var st))
        {
            if (st.ValueKind == JsonValueKind.String) stops.Add(st.GetString()!);
            else if (st.ValueKind == JsonValueKind.Array) stops.AddRange(st.EnumerateArray().Where(e => e.ValueKind == JsonValueKind.String).Select(e => e.GetString()!));
        }
        if (d.StopSequences is { Count: > 0 } ps) stops.AddRange(ps.Where(x => !stops.Contains(x)));

        int numPredict = I("num_predict") ?? d.MaxTokens;
        ResponseFormat? format = null;
        if (root.TryGetProperty("format", out var f))
        {
            if (f.ValueKind == JsonValueKind.String && f.GetString() == "json") format = new ResponseFormat.JsonObject();
            else if (f.ValueKind == JsonValueKind.Object) format = new ResponseFormat.JsonSchema { Schema = f.GetRawText() };
        }

        return new InferenceOptions
        {
            Temperature = F("temperature") ?? d.Temperature,
            TopP = F("top_p") ?? d.TopP,
            TopK = I("top_k") ?? d.TopK,
            MinP = F("min_p") ?? d.MinP,
            RepetitionPenalty = F("repeat_penalty") ?? d.RepetitionPenalty,
            FrequencyPenalty = F("frequency_penalty") ?? 0f,
            PresencePenalty = F("presence_penalty") ?? 0f,
            Seed = I("seed") ?? d.Seed,
            MaxTokens = numPredict > 0 ? numPredict : d.MaxTokens,   // -1 / -2 mean "no limit" in ollama: fall back to the server default
            StopSequences = stops,
            ResponseFormat = format,
            Threading = threading,
        };
    }

    private static async Task GenerateAsync(ServerState state, HttpContext c, bool chat)
    {
        using var doc = await ReadBodyAsync(c);
        if (doc is null) return;
        var root = doc.RootElement;
        var ct = c.RequestAborted;

        string? name = ModelName(root);
        if (string.IsNullOrWhiteSpace(name)) { await Error(c, 400, "model is required"); return; }
        if (chat && root.TryGetProperty("tools", out var tools) && tools.ValueKind == JsonValueKind.Array && tools.GetArrayLength() > 0)
        { await Error(c, 501, "tools are not supported on /api/chat by dotLLM's ollama compatibility layer; use /v1/chat/completions."); return; }

        double? keepAlive = ParseKeepAlive(root);
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var activationError = await state.EnsureActiveAsync(name, keepAlive, ct);
        if (activationError is not null)
        {
            await Error(c, activationError.Contains("not found", StringComparison.OrdinalIgnoreCase) ? 404 : 400,
                activationError.Contains("not found", StringComparison.OrdinalIgnoreCase)
                    ? $"model '{name}' not found, try pulling it first" : activationError);
            return;
        }
        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null || state.Tokenizer is null)
        { await Error(c, 503, "no model loaded"); return; }
        double loadMs = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;

        // /api/generate with no prompt is ollama's "load this model" call; keep_alive 0 unloads it.
        string? prompt = chat ? null : Str(root, "prompt");
        if (!chat && string.IsNullOrEmpty(prompt))
        {
            string reason = "load";
            if (keepAlive == 0) { await state.UnloadAsync(name, all: false, ct); reason = "unload"; }
            await WriteJson(c, 200, w =>
            {
                w.WriteString("model", name); w.WriteString("created_at", DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture));
                w.WriteString("response", ""); w.WriteBoolean("done", true); w.WriteString("done_reason", reason);
            });
            return;
        }

        // (#767) `think` -> the template's enable_thinking (and a level string -> reasoning_effort). A `format`
        // (JSON / schema) constrains decoding from the first token and cannot coexist with an open think block,
        // so it defaults thinking off unless `think: true` was asked for.
        var options = BuildOptions(root, state.EffectiveSamplingDefaults, new ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads));
        bool constrained = options.ResponseFormat is not (null or ResponseFormat.Text);
        ParseThink(root, out bool? think, out string? thinkLevel);
        var templateOptions = ReasoningSupport.BuildTemplateOptions(null, think, thinkLevel, null, constrained);

        string finalPrompt;
        if (chat)
        {
            if (!root.TryGetProperty("messages", out var ms) || ms.ValueKind != JsonValueKind.Array) { await Error(c, 400, "messages is required"); return; }
            var messages = new List<ChatMessage>();
            foreach (var m in ms.EnumerateArray())
                messages.Add(new ChatMessage
                {
                    Role = Str(m, "role") ?? "user",
                    Content = Str(m, "content") ?? "",
                    // ollama replays an earlier turn's reasoning in `thinking`; the template decides whether to render it.
                    ReasoningContent = Str(m, "thinking"),
                });
            if (messages.Count == 0) { await Error(c, 400, "messages must not be empty"); return; }
            if (!ReasoningSupport.TryApply(state.ChatTemplate, ProfileSystemPrompt.Apply(state.ActiveProfile?.System, messages.ToArray()),
                    templateOptions, out finalPrompt, out string? chatErr, out _))
            { await Error(c, 400, chatErr!); return; }
        }
        else if (root.TryGetProperty("raw", out var raw) && raw.ValueKind == JsonValueKind.True)
            finalPrompt = prompt!;
        else
        {
            var messages = new List<ChatMessage>();
            if (Str(root, "system") is { Length: > 0 } sys) messages.Add(new ChatMessage { Role = "system", Content = sys });
            messages.Add(new ChatMessage { Role = "user", Content = prompt! });
            if (!ReasoningSupport.TryApply(state.ChatTemplate, ProfileSystemPrompt.Apply(state.ActiveProfile?.System, messages.ToArray()),
                    templateOptions, out finalPrompt, out string? genErr, out _))
            { await Error(c, 400, genErr!); return; }
        }

        var plan = ReasoningSupport.Plan(state.Options.ReasoningFormat, null, constrained, finalPrompt, out _);
        options = plan.Gate(options, ReasoningSupport.UngatedStops);
        bool stream = !(root.TryGetProperty("stream", out var sv) && sv.ValueKind == JsonValueKind.False);
        string modelId = state.Options.ModelId;
        var generator = state.Generator;
        string created() => DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture);

        void Chunk(Utf8JsonWriter w, string text, bool done, string? doneReason, InferenceTimings? t, int evalCount, double totalMs, string? thinking = null)
        {
            w.WriteStartObject();
            w.WriteString("model", modelId); w.WriteString("created_at", created());
            if (chat)
            {
                w.WriteStartObject("message"); w.WriteString("role", "assistant"); w.WriteString("content", text);
                if (thinking is not null) w.WriteString("thinking", thinking);
                w.WriteEndObject();
            }
            else
            {
                w.WriteString("response", text);
                if (thinking is not null) w.WriteString("thinking", thinking);
            }
            w.WriteBoolean("done", done);
            if (done)
            {
                w.WriteString("done_reason", doneReason ?? "stop");
                w.WriteNumber("total_duration", Ns(totalMs)); w.WriteNumber("load_duration", Ns(loadMs));
                w.WriteNumber("prompt_eval_count", t?.PrefillTokenCount ?? 0); w.WriteNumber("prompt_eval_duration", Ns(t?.PrefillTimeMs ?? 0));
                w.WriteNumber("eval_count", evalCount); w.WriteNumber("eval_duration", Ns((t?.DecodeTimeMs ?? 0) + (t?.SamplingTimeMs ?? 0)));
            }
            w.WriteEndObject();
        }

        if (!stream)
        {
            InferenceResponse? result = null;
            if (state.Scheduler is { } scheduler)
            {
                result = await scheduler.EnqueueAsync(new InferenceRequest
                {
                    TokenIds = state.Tokenizer.Encode(finalPrompt),
                    Options = options,
                    ApiKey = c.Items.TryGetValue(RateLimitMiddleware.ApiKeyItemKey, out var k) ? k as string : null,
                }, ct);
            }
            else
                await state.ExecuteAsync(() => { result = generator.Generate(finalPrompt, options); return Task.CompletedTask; }, ct);

            RateLimitMiddleware.GetLease(c)?.ReportActualTokens(result!.PromptTokenCount + result.GeneratedTokenCount);
            c.Response.ContentType = "application/json; charset=utf-8";
            using var ms2 = new MemoryStream();
            var (reasoning, answer, _) = plan.SplitComplete(result!.Text, state.Tokenizer);
            using (var w = new Utf8JsonWriter(ms2))
                Chunk(w, answer, true, result.FinishReason == FinishReason.Length ? "length" : "stop", result.Timings, result.GeneratedTokenCount,
                    System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds, reasoning);
            await c.Response.Body.WriteAsync(ms2.ToArray(), ct);
            return;
        }

        c.Response.ContentType = Ndjson;
        c.Response.Headers.CacheControl = "no-cache";
        int generated = 0;
        FinishReason? finish = null;
        InferenceTimings? timings = null;
        var splitter = plan.NewSplitter();
        async Task Send(string text, string? thinking)
        {
            using var ms3 = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms3)) Chunk(w, text, false, null, null, 0, 0, thinking);
            ms3.WriteByte((byte)'\n');
            await c.Response.Body.WriteAsync(ms3.ToArray(), ct);
            await c.Response.Body.FlushAsync(ct);
        }
        async Task Emit(ReasoningChunk chunk)
        {
            // Reasoning and answer go out as separate chunks (reasoning first), as ollama does.
            if (chunk.Reasoning.Length > 0) await Send("", chunk.Reasoning);
            if (chunk.Content.Length > 0) await Send(chunk.Content, null);
        }
        await state.ExecuteAsync(async () =>
        {
            await foreach (var token in generator.GenerateStreamingTokensAsync(finalPrompt, options, ct))
            {
                if (token.Text.Length > 0)
                {
                    generated++;
                    await Emit(splitter is null ? new ReasoningChunk("", token.Text) : splitter.Feed(token.Text));
                }
                if (token.FinishReason.HasValue) finish = token.FinishReason;
                if (token.Timings.HasValue) timings = token.Timings;
            }
            if (splitter is not null) await Emit(splitter.Finish());
        }, ct);

        RateLimitMiddleware.GetLease(c)?.ReportActualTokens((timings?.PrefillTokenCount ?? 0) + generated);
        using var fin = new MemoryStream();
        using (var w = new Utf8JsonWriter(fin))
            Chunk(w, "", true, finish == FinishReason.Length ? "length" : "stop", timings, generated, System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        fin.WriteByte((byte)'\n');
        await c.Response.Body.WriteAsync(fin.ToArray(), ct);
    }

    // ───────────────────────────────────── pull / delete ─────────────────────────────────────

    private static async Task PullAsync(ServerState state, HttpContext c)
    {
        if (!state.Options.AllowModelAdminApi)
        { await Error(c, 403, "POST /api/pull is disabled. Start the server with --allow-model-admin to enable model administration."); return; }
        using var doc = await ReadBodyAsync(c);
        if (doc is null) return;
        string? name = ModelName(doc.RootElement);
        if (string.IsNullOrWhiteSpace(name)) { await Error(c, 400, "model is required"); return; }
        bool stream = !(doc.RootElement.TryGetProperty("stream", out var sv) && sv.ValueKind == JsonValueKind.False);
        var ct = c.RequestAborted;

        var hf = ModelResolver.Parse(name);
        var oll = OllamaRef.TryParse(name);
        if (!hf.IsRepo && oll is null) { await Error(c, 400, $"'{name}' is not a Hugging Face repo (owner/repo[:tag]) or an ollama model name"); return; }

        var channel = System.Threading.Channels.Channel.CreateUnbounded<(long Done, long? Total)>();
        var progress = new Progress<(long, long?)>(p => channel.Writer.TryWrite(p));
        Task<string> pull = Task.Run(async () =>
        {
            try
            {
                if (hf.IsRepo)
                {
                    using var client = new HuggingFaceClient();
                    using var downloader = new HuggingFaceDownloader();
                    return await ModelResolver.PullAsync(hf, null, client, downloader, progress, ct);
                }
                using var registry = new OllamaRegistry();
                return await ModelResolver.PullOllamaAsync(oll!.Value, registry, progress, ct);
            }
            finally { channel.Writer.TryComplete(); }
        }, ct);

        async Task Line(Action<Utf8JsonWriter> body)
        {
            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms)) { w.WriteStartObject(); body(w); w.WriteEndObject(); }
            ms.WriteByte((byte)'\n');
            await c.Response.Body.WriteAsync(ms.ToArray(), ct);
            await c.Response.Body.FlushAsync(ct);
        }

        try
        {
            if (stream)
            {
                c.Response.ContentType = Ndjson;
                await Line(w => w.WriteString("status", "pulling manifest"));
                long last = 0;
                await foreach (var (done, total) in channel.Reader.ReadAllAsync(ct))
                {
                    if (done - last < (1 << 20) && done != total) continue;   // throttle: a line per MiB
                    last = done;
                    await Line(w => { w.WriteString("status", "downloading"); w.WriteString("digest", name); if (total.HasValue) w.WriteNumber("total", total.Value); w.WriteNumber("completed", done); });
                }
                await pull;
                await Line(w => w.WriteString("status", "success"));
            }
            else
            {
                await foreach (var _ in channel.Reader.ReadAllAsync(ct)) { }
                await pull;
                await WriteJson(c, 200, w => w.WriteString("status", "success"));
            }
        }
        catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException or IOException)
        {
            if (stream && c.Response.HasStarted) await Line(w => w.WriteString("error", ex.Message));
            else await Error(c, ex is InvalidOperationException && ex.Message.Contains("not found") ? 404 : 502, ex.Message);
        }
    }

    private static bool OwnedOllamaPull(string path)
    {
        string owned = System.IO.Path.GetFullPath(System.IO.Path.Combine(HuggingFaceDownloader.DefaultModelsDirectory, "ollama")) + System.IO.Path.DirectorySeparatorChar;
        return System.IO.Path.IsPathRooted(path) && System.IO.Path.GetFullPath(path).StartsWith(owned, StringComparison.OrdinalIgnoreCase) && File.Exists(path);
    }

    private static async Task DeleteAsync(ServerState state, HttpContext c)
    {
        if (!state.Options.AllowModelAdminApi)
        { await Error(c, 403, "DELETE /api/delete is disabled. Start the server with --allow-model-admin to enable model administration."); return; }
        using var doc = await ReadBodyAsync(c);
        if (doc is null) return;
        string? name = ModelName(doc.RootElement);
        if (string.IsNullOrWhiteSpace(name)) { await Error(c, 400, "model is required"); return; }

        // A profile is removed. Its base model is kept, except when dotLLM pulled that file itself from the ollama registry (it lives under
        // models/ollama/ and no other profile uses it): ollama's "delete" means "remove the model", so a pulled model goes with its name.
        if (ModelProfileStore.TryGet(name) is { } profile)
        {
            ModelProfileStore.Delete(name);
            if (profile.From is { Length: > 0 } from && OwnedOllamaPull(from) && !ModelProfileStore.List().Any(p => string.Equals(p.Profile.From, from, StringComparison.OrdinalIgnoreCase)))
            {
                try { File.Delete(from); ModelResolver.PruneEmptyDirectories(System.IO.Path.GetDirectoryName(from), HuggingFaceDownloader.DefaultModelsDirectory); }
                catch (IOException) { /* in use: the profile is gone either way */ }
            }
            await WriteJson(c, 200, _ => { });
            return;
        }

        var match = ModelResolver.EnumerateLocal().FirstOrDefault(m =>
            System.IO.Path.GetFileNameWithoutExtension(m.Filename).Equals(name, StringComparison.OrdinalIgnoreCase)
            || m.Filename.Equals(name, StringComparison.OrdinalIgnoreCase));
        if (match is null)
        {
            bool inOllama = OllamaRef.TryParse(name) is { } o && OllamaStore.TryFind(o) is not null;
            await Error(c, inOllama ? 400 : 404, inOllama
                ? $"'{name}' lives in the ollama store, which dotLLM never modifies; remove it with ollama."
                : $"model '{name}' not found");
            return;
        }
        long freed = ModelResolver.DeleteLocal(match);
        await WriteJson(c, freed > 0 ? 200 : 500, w => { if (freed == 0) w.WriteString("error", "nothing was removed"); });
    }
}
