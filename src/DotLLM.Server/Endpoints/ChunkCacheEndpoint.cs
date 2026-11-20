using System.Text.Json;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using Microsoft.Extensions.DependencyInjection;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// The chunk-cache control plane (spec §5.9, §7.2; plan Tasks 14 and 16): <c>GET /v1/chunk-cache</c> (capabilities),
/// <c>POST /v1/chunk-cache/explain</c> (a dry run of a request's annotations), and with <c>--chunk-store-write</c>
/// <c>POST /v1/chunk-cache/prewarm</c> (queue text for a background encode) and <c>POST /v1/chunk-cache/evict</c> (orphans
/// and forget). Mapped only with <c>--chunk-cache</c>: with
/// the flag off the routes do not exist (404, exactly as before the convention). Every call is authenticated by
/// <see cref="ChunkCacheAuth"/> (401 / 403) and, when rate limiting is configured, metered like inference (<c>/v1/</c> paths are metered
/// unless exempted, and these are not). The POST routes bound their concurrency (429) and body size (413), read only
/// <c>application/json</c> (415), and every route refuses cross-site browser requests (403), so a web page can never drive
/// prewarm or evict (CSRF).
/// </summary>
/// <remarks>
/// Only prewarm and evict change the store, through the sole writer (<see cref="ChunkStoreService"/>). Responses and logs never
/// carry a namespace, a key or a key digest (the producer hash in <c>model.producer</c> is a key of no tokens, not a
/// namespace), chunk text, a text hash or a source path: they carry counts.
/// </remarks>
public static class ChunkCacheEndpoint
{
    /// <summary>The capabilities route.</summary>
    public const string CapabilitiesPath = "/v1/chunk-cache";

    /// <summary>The explain route.</summary>
    public const string ExplainPath = "/v1/chunk-cache/explain";

    /// <summary>The prewarm route.</summary>
    public const string PrewarmPath = "/v1/chunk-cache/prewarm";

    /// <summary>The evict route.</summary>
    public const string EvictPath = "/v1/chunk-cache/evict";

    /// <summary>Explain calls that may run at once per server; more get 429.</summary>
    public const int MaxConcurrentExplains = 2;

    /// <summary>Largest explain body, bytes; larger is 413.</summary>
    public const int MaxExplainBodyBytes = 4 * 1024 * 1024;

    /// <summary>
    /// Seconds an explain body may take to arrive; slower is 408. The body is read before an explain slot is taken, so a
    /// slow sender never holds one.
    /// </summary>
    public const int ExplainBodyReadTimeoutSeconds = 10;
    /// <summary>Most texts in one prewarm call; more is a 400 (<c>chunk_too_many</c>).</summary>
    public const int MaxPrewarmItems = 256;

    /// <summary>Largest prewarm body, bytes; larger is 413.</summary>
    public const int MaxPrewarmBodyBytes = 4 * 1024 * 1024;

    /// <summary>Default of <see cref="BodyReadTimeout"/>.</summary>
    public static readonly TimeSpan DefaultBodyReadTimeout = TimeSpan.FromSeconds(10);

    /// <summary>
    /// How long a POST body may take to arrive (explain, prewarm, evict); later is 408. The body is read before any slot is
    /// taken, so a trickling body never holds one (Task 16 review S3). A constant; tests shorten it.
    /// </summary>
    internal static TimeSpan BodyReadTimeout { get; set; } = DefaultBodyReadTimeout;

    /// <summary>Longest evict <c>path</c> glob, characters; longer is 400.</summary>
    public const int MaxGlobChars = 1024;

    /// <summary>Largest evict body, bytes; larger is 413.</summary>
    public const int MaxEvictBodyBytes = 64 * 1024;

    /// <summary>Prewarm and evict calls that may run at once per server; more get 429.</summary>
    public const int MaxConcurrentWrites = 2;

    /// <summary>The evict warning when a delete failed (a file held open): the counts are in <c>failed</c>.</summary>
    public const string EvictPartial = "chunk_evict_partial";

    /// <summary>The evict modes (spec §7.2).</summary>
    private static readonly Dictionary<string, EvictMode> Modes = new(StringComparer.Ordinal)
    {
        ["orphans"] = EvictMode.Orphans, ["path"] = EvictMode.Path, ["namespace"] = EvictMode.Namespace, ["all"] = EvictMode.All,
    };

    /// <summary>Maps the routes when the server runs with <c>--chunk-cache</c>; otherwise maps nothing.</summary>
    /// <param name="app">The web application (its services hold the <see cref="ServerState"/>).</param>
    public static void Map(WebApplication app)
    {
        ArgumentNullException.ThrowIfNull(app);
        var state = app.Services.GetService<ServerState>();
        if (state is null || !state.Options.ChunkCache.Enabled)
            return;

        if (state.Options.ChunkCache.ApiKeys is { } keys)
        {
            ServerLog.Info($"chunk-cache: {keys.Count} API key(s) ({keys.AdminCount} admin) guard /v1/chunk-cache/*");
            foreach (string warning in keys.Warnings)
                ServerLog.Info(warning);
        }
        else
        {
            ServerLog.Info("chunk-cache: no --chunk-cache-api-keys; /v1/chunk-cache/* answers local callers only");
        }

        // --chunk-store-write: become the store's sole writer now, at startup, so a second server on the same root fails
        // before it listens (InvalidServerOptionException), and stop the worker before the model goes away.
        if (state.ChunkCacheRuntime.StartWriter(state) is not null)
            app.Lifetime.ApplicationStopping.Register(() => state.ChunkCacheRuntime.Dispose());

        app.MapGet(CapabilitiesPath, GetCapabilitiesAsync);
        app.MapPost(ExplainPath, ExplainAsync);
        app.MapPost(PrewarmPath, PrewarmAsync);
        app.MapPost(EvictPath, EvictAsync);
    }

    private static async Task GetCapabilitiesAsync(HttpContext context, ServerState state)
    {
        var options = state.Options.ChunkCache;
        if (!options.Enabled)
        {
            context.Response.StatusCode = StatusCodes.Status404NotFound;
            return;
        }
        if (ChunkCacheAuth.IsCrossSite(context))
        {
            await WriteCrossSiteAsync(context);
            return;
        }
        var caller = ChunkCacheAuth.Resolve(context, options, out var failure);
        if (failure != ChunkAuthFailure.None)
        {
            await ChunkCacheAuth.WriteFailureAsync(context, failure);
            return;
        }
        await context.Response.WriteAsJsonAsync(BuildCapabilities(state, caller), ServerJsonContext.Default.ChunkCacheCapabilitiesDto,
            cancellationToken: context.RequestAborted);
    }

    /// <summary>The capabilities body for <paramref name="caller"/>. Never throws.</summary>
    internal static ChunkCacheCapabilitiesDto BuildCapabilities(ServerState state, ChunkCaller caller)
    {
        var options = state.Options.ChunkCache;
        ChunkCacheModelDto model;
        try
        {
            var identity = state.ChunkCacheRuntime.IdentityFor(state);
            model = identity.Supported
                ? new ChunkCacheModelDto
                {
                    Id = identity.ModelId,
                    Supported = true,
                    Reason = null,
                    IdKind = "sampled",
                    TokenizerId = identity.TokenizerId,
                    Producer = identity.KeyOf([], caller.Namespace).Hex,
                    Quantization = identity.Quantization,
                    Backend = identity.Backend,
                    KvDtype = DtypeName(identity.KvDtype),
                    Rope = new ChunkCacheRopeDto
                    {
                        Dim = identity.Rope.RopeDim,
                        Theta = identity.Rope.Theta,
                        Type = identity.Rope.Type.ToString().ToLowerInvariant(),
                    },
                }
                : Unsupported();
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            ServerLog.Info($"chunk_cache: capabilities identity failed ({ex.GetType().Name})");
            model = Unsupported();
        }

        return new ChunkCacheCapabilitiesDto
        {
            Versions = [ChunkCacheWire.Version],
            Policies = ["exact", "shadow", "balanced", "fast", "auto"],
            Selectors = [ChunkCacheReport.PlanSelector],
            WireChunkers = [.. ChunkCacheWire.WireChunkers.Order(StringComparer.Ordinal)],
            AutoValidated = [],
            Model = model,
            Store = StoreFigures(state, caller),
            MaxChunksPerRequest = options.MaxChunks,
        };

        static ChunkCacheModelDto Unsupported() =>
            new() { Id = null, Supported = false, Reason = ChunkCacheCodes.ModelUnsupported };
    }

    /// <summary>
    /// <c>capabilities.store</c> (Task 16): a non-admin sees its own namespace's ledger entries and the budget that applies to
    /// it (the per-namespace budget when set); an admin sees every entry file in the store and the store budget. Without a
    /// writer there is no ledger: non-admins see 0, an admin the read-only store's files. Never throws.
    /// </summary>
    private static ChunkCacheStoreDto StoreFigures(ServerState state, ChunkCaller caller)
    {
        var options = state.Options.ChunkCache;
        long entries = 0, bytes = 0, budget = 0;
        try
        {
            if (state.ChunkCacheRuntime.Writer is { } writer)
            {
                if (caller.IsAdmin)
                {
                    (entries, bytes) = state.ChunkCacheRuntime.StoreFigures(writer.StoreFigures);
                    budget = writer.BudgetBytes;
                }
                else
                {
                    (entries, bytes) = writer.NamespaceFigures(caller.Namespace);
                    budget = writer.NamespaceBudgetBytes > 0 ? writer.NamespaceBudgetBytes : writer.BudgetBytes;
                }
            }
            else if (caller.IsAdmin && state.ChunkCacheRuntime.Store(options, out _) is FileChunkStore files)
            {
                (entries, bytes) = state.ChunkCacheRuntime.StoreFigures(() =>
                {
                    var stats = files.GetStats();
                    return (stats.Entries, stats.EntryBytes);
                });
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            ServerLog.Info($"chunk_cache: store figures unavailable ({ex.GetType().Name})");
            (entries, bytes) = (0, 0);
        }
        return new ChunkCacheStoreDto
        {
            // Ruling 52: always an object; configured: false without --chunk-store (absent otherwise).
            Configured = options.StoreRoot is null && state.ChunkCacheRuntime.StoreOverride is null ? false : null,
            Entries = entries,
            Bytes = bytes,
            BudgetBytes = budget,
            Dtype = DtypeName(KvDtype.F32),
        };
    }

    private static string DtypeName(KvDtype dtype) => dtype.ToString().ToLowerInvariant();

    private static async Task ExplainAsync(HttpContext context, ServerState state)
    {
        try
        {
            await ExplainCoreAsync(context, state);
        }
        catch (Microsoft.AspNetCore.Http.BadHttpRequestException ex)
        {
            // The server's own body limit, or a truncated body: the client's fault, not a 500.
            if (!context.Response.HasStarted)
                await WriteAsync(context, ex.StatusCode == StatusCodes.Status413PayloadTooLarge ? StatusCodes.Status413PayloadTooLarge : StatusCodes.Status400BadRequest,
                    ErrorResponse.InvalidRequest("explain: the request body could not be read (too large or truncated)", param: "request"));
        }
        catch (Exception ex) when (ex is not (OperationCanceledException or OutOfMemoryException))
        {
            // Never a crash on hostile input; the log carries the exception type only (no content).
            ServerLog.Info($"chunk_cache: explain failed ({ex.GetType().Name})");
            if (!context.Response.HasStarted)
                await WriteAsync(context, StatusCodes.Status500InternalServerError, ErrorResponse.Internal("chunk_cache: explain failed"));
        }
    }

    private static async Task ExplainCoreAsync(HttpContext context, ServerState state)
    {
        var options = state.Options.ChunkCache;
        if (!options.Enabled)
        {
            context.Response.StatusCode = StatusCodes.Status404NotFound;
            return;
        }
        // CSRF (before auth, before the body): a cross-site browser request is refused, and only application/json is read,
        // so a page cannot send a "simple" (unpreflighted) POST here.
        if (ChunkCacheAuth.IsCrossSite(context))
        {
            await WriteCrossSiteAsync(context);
            return;
        }
        if (!IsJson(context.Request.ContentType))
        {
            await WriteAsync(context, StatusCodes.Status415UnsupportedMediaType,
                ErrorResponse.InvalidRequest("explain takes Content-Type: application/json", code: "unsupported_media_type"));
            return;
        }
        var caller = ChunkCacheAuth.Resolve(context, options, out var failure);
        if (failure != ChunkAuthFailure.None)
        {
            await ChunkCacheAuth.WriteFailureAsync(context, failure);
            return;
        }

        // The body first, bounded in size and time, and before a slot: a slow sender (Kestrel's minimum data rate is only
        // ~240 B/s, i.e. hours for 4 MiB) must never hold one of the few explain slots for everyone else.
        if (context.Request.ContentLength > MaxExplainBodyBytes)
        {
            await WriteTooLargeAsync(context);
            return;
        }
        var runtime = state.ChunkCacheRuntime;
        ReadOnlyMemory<byte>? body;
        using (var deadline = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted))
        {
            deadline.CancelAfter(runtime.ExplainBodyReadTimeout);
            runtime.EnterExplainBodyRead();
            try
            {
                body = await ReadLimitedAsync(context, MaxExplainBodyBytes, deadline.Token);
            }
            catch (OperationCanceledException) when (deadline.IsCancellationRequested && !context.RequestAborted.IsCancellationRequested)
            {
                // The rest of the body is never read: close the connection rather than drain it.
                context.Response.Headers.Connection = "close";
                await WriteAsync(context, StatusCodes.Status408RequestTimeout, ErrorResponse.InvalidRequest(
                    $"explain: the request body did not arrive within {(int)runtime.ExplainBodyReadTimeout.TotalSeconds} s",
                    param: "request", code: "request_timeout"));
                return;
            }
            finally
            {
                runtime.ExitExplainBodyRead();
            }
        }
        if (body is null)
        {
            await WriteTooLargeAsync(context);
            return;
        }

        // Bounded: explain renders and tokenises on the request thread, outside the scheduler.
        var gate = runtime.ExplainGate;
        if (!gate.Wait(0))
        {
            context.Response.Headers.RetryAfter = "1";
            await WriteAsync(context, StatusCodes.Status429TooManyRequests,
                ErrorResponse.RateLimit($"explain: at most {MaxConcurrentExplains} run at once; retry shortly", code: "chunk_explain_busy"));
            return;
        }
        try
        {
            await ExplainAdmittedAsync(context, state, caller, body.Value);
        }
        finally
        {
            gate.Release();
        }
    }

    private static async Task ExplainAdmittedAsync(HttpContext context, ServerState state, ChunkCaller caller, ReadOnlyMemory<byte> body)
    {
        var options = state.Options.ChunkCache;
        var (request, bodyError) = ParseBody(body);
        if (request is null)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(bodyError!, param: "request"));
            return;
        }

        if (!state.IsReady || state.ChatTemplate is null || state.Tokenizer is null || state.Config is null)
        {
            await WriteAsync(context, StatusCodes.Status503ServiceUnavailable, ErrorResponse.Internal("No model loaded", code: "model_not_loaded"));
            return;
        }

        var validationError = RequestValidator.ValidateChatRequest(request, state.Options.EffectiveMaxMessages);
        if (validationError is not null)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(validationError));
            return;
        }

        var parsed = ChunkAnnotationParser.Parse(request, options, caller.Namespace);
        if (parsed is null)
        {
            await WriteExplainAsync(context, NothingAnnotated());
            return;
        }
        if (parsed.Error is { } parseError)
        {
            await WriteChunkErrorAsync(context, parseError);
            return;
        }

        if (!ChatCompletionEndpoint.TryRender(request, state, out var rendered, out string? templateError, out string? templateParam))
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(templateError!, param: templateParam));
            return;
        }
        // One encode serves the context-length check and the plan (the chat path's ValidatePromptLength would be a second).
        int[] promptIds = state.Tokenizer.Encode(rendered.Prompt);
        if (promptIds.Length >= state.Config.MaxSequenceLength)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(
                $"prompt ({promptIds.Length} tokens) exceeds model context length ({state.Config.MaxSequenceLength})",
                param: "messages", code: "context_length_exceeded"));
            return;
        }

        var exchange = ChunkCacheExchange.Begin(parsed, caller)!;
        if (exchange.Prepare(state, rendered.Messages, rendered.Messages.Length - rendered.EngineMessages.Length,
                rendered.TemplateOptions, rendered.Prompt, promptIds) is { } planError)
        {
            await WriteChunkErrorAsync(context, planError);
            return;
        }

        await WriteExplainAsync(context, BuildExplain(exchange.Plan!, caller));
    }

    /// <summary>The explain body for a plan (no prompt-cache probe: nothing is <c>prefix</c>).</summary>
    internal static ChunkCacheExplainDto BuildExplain(ChunkCachePlan plan, ChunkCaller caller)
    {
        // The shadow plan's prediction, whatever the requested policy: what reusing the stored chunks would take.
        var report = ChunkCacheReport.Build(plan with { Shadow = true }, plan.PromptTokens, 0, 0, ChunkCostModel.For(plan.Backend));
        return new ChunkCacheExplainDto
        {
            PolicyWouldApply = ChunkCacheReport.PolicyName(ChunkPolicy.Exact),
            Reason = plan.Reason,
            Chunks = caller.Authenticated ? ChunkCacheExchange.PerChunk(plan.Statuses, 0) : null,
            Counts = report.Chunks,
            PredictedMs = new ChunkCacheExplainPredictedDto
            {
                Reuse = report.Plan?.PredictedTtftMs ?? 0,
                FullPrefill = report.Plan?.FullPrefillEstimateMs ?? 0,
            },
            Warnings = plan.Warnings.Select(w => w.Code).ToArray(),
        };
    }

    private static ChunkCacheExplainDto NothingAnnotated() => new()
    {
        PolicyWouldApply = ChunkCacheReport.PolicyName(ChunkPolicy.Exact),
        Reason = null,
        Chunks = [],
        Counts = new ChunkCacheChunksDto(),
        PredictedMs = new ChunkCacheExplainPredictedDto(),
        Warnings = [],
    };

    /// <summary>
    /// Reads <c>{"request": &lt;chat completion request&gt;}</c> (ruling 15). Any other top-level field is refused, so a hash
    /// list (a membership probe without the text) has no way in. Never throws on content.
    /// </summary>
    private static (ChatCompletionRequest? Request, string? Error) ParseBody(ReadOnlyMemory<byte> body)
    {
        const string shape = "explain takes {\"request\": <a /v1/chat/completions body>} and nothing else (no hash lists: send the text)";
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(body);
        }
        catch (JsonException)
        {
            return (null, shape + "; the body is not JSON");
        }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return (null, shape);
            JsonElement? inner = null;
            foreach (var property in root.EnumerateObject())
            {
                if (!string.Equals(property.Name, "request", StringComparison.Ordinal) || inner is not null)
                    return (null, shape + "; the body has another field");
                inner = property.Value;
            }
            if (inner is not { ValueKind: JsonValueKind.Object } requestElement)
                return (null, shape + "; request must be an object");
            try
            {
                var request = requestElement.Deserialize(ServerJsonContext.Default.ChatCompletionRequest);
                return request is null ? (null, shape) : (request, null);
            }
            catch (Exception ex) when (ex is JsonException or NotSupportedException or InvalidOperationException or ArgumentException or FormatException)
            {
                return (null, "request is not a valid /v1/chat/completions body (string message content only)");
            }
        }
    }

    /// <summary>
    /// Reads the body, at most <paramref name="max"/> bytes (also without a <c>Content-Length</c>, e.g. chunked); null when
    /// it is larger.
    /// </summary>
    private static async Task<ReadOnlyMemory<byte>?> ReadLimitedAsync(HttpContext context, int max, CancellationToken ct)
    {
        if (context.Request.ContentLength > max)
            return null;
        var buffer = new MemoryStream();
        byte[] chunk = new byte[64 * 1024];
        int read;
        while ((read = await context.Request.Body.ReadAsync(chunk, ct)) > 0)
        {
            if (buffer.Length + read > max)
                return null;
            buffer.Write(chunk, 0, read);
        }
        return buffer.GetBuffer().AsMemory(0, (int)buffer.Length);
    }

    // ------------------------------------------------------------------------------------------------ prewarm and evict (Task 16)

    /// <summary>
    /// The gates every write route passes, in order: cross-site (403), <c>application/json</c> (415), authentication (401 /
    /// 403), a writer (503 <c>chunk_store_unavailable</c> without <c>--chunk-store-write</c>). Null when a response was written.
    /// </summary>
    private static async Task<(ChunkCaller Caller, ChunkStoreService Writer)?> AdmitWriteAsync(HttpContext context, ServerState state, string route)
    {
        if (ChunkCacheAuth.IsCrossSite(context))
        {
            await WriteCrossSiteAsync(context);
            return null;
        }
        if (!IsJson(context.Request.ContentType))
        {
            await WriteAsync(context, StatusCodes.Status415UnsupportedMediaType,
                ErrorResponse.InvalidRequest($"{route} takes Content-Type: application/json", code: "unsupported_media_type"));
            return null;
        }
        var caller = ChunkCacheAuth.Resolve(context, state.Options.ChunkCache, out var failure);
        if (failure != ChunkAuthFailure.None)
        {
            await ChunkCacheAuth.WriteFailureAsync(context, failure);
            return null;
        }
        if (state.ChunkCacheRuntime.Writer is not { } writer)
        {
            await WriteAsync(context, StatusCodes.Status503ServiceUnavailable, ErrorResponse.Internal(
                $"{route}: this server does not write its chunk store (start it with --chunk-store and --chunk-store-write)",
                code: ChunkCacheCodes.StoreUnavailable));
            return null;
        }
        return (caller, writer);
    }

    /// <summary>Runs a write route under the write gate (429 when full), mapping body-read failures to 413 / 400 and anything unexpected to 500.</summary>
    private static async Task RunWriteAsync(HttpContext context, ServerState state, string route, int maxBody,
        Func<ChunkCaller, ChunkStoreService, ReadOnlyMemory<byte>, Task> handler)
    {
        try
        {
            if (!state.Options.ChunkCache.Enabled)
            {
                context.Response.StatusCode = StatusCodes.Status404NotFound;
                return;
            }
            if (await AdmitWriteAsync(context, state, route) is not { } admitted)
                return;
            // The body first (deadline and size limit): a trickling body never holds a slot (review S3).
            if (await ReadBodyAsync(context, maxBody, route) is not { } body)
                return;
            // One write call per namespace at a time (one tenant cannot hold both slots), at most MaxConcurrentWrites in all.
            var runtime = state.ChunkCacheRuntime;
            if (!runtime.TryEnterWriteSlot(admitted.Caller.Namespace))
            {
                await WriteBusyAsync(context, $"{route}: one prewarm / evict call per caller at a time; retry shortly");
                return;
            }
            try
            {
                var gate = runtime.WriteGate;
                if (!gate.Wait(0))
                {
                    await WriteBusyAsync(context, $"{route}: at most {MaxConcurrentWrites} prewarm / evict calls run at once; retry shortly");
                    return;
                }
                try
                {
                    await handler(admitted.Caller, admitted.Writer, body);
                }
                finally
                {
                    gate.Release();
                }
            }
            finally
            {
                runtime.LeaveWriteSlot(admitted.Caller.Namespace);
            }
        }
        catch (Microsoft.AspNetCore.Http.BadHttpRequestException ex)
        {
            if (!context.Response.HasStarted)
                await WriteAsync(context, ex.StatusCode == StatusCodes.Status413PayloadTooLarge ? StatusCodes.Status413PayloadTooLarge : StatusCodes.Status400BadRequest,
                    ErrorResponse.InvalidRequest($"{route}: the request body could not be read (too large or truncated)"));
        }
        catch (Exception ex) when (ex is not (OperationCanceledException or OutOfMemoryException))
        {
            ServerLog.Info($"chunk_cache: {route} failed ({ex.GetType().Name})"); // type only: no content
            if (!context.Response.HasStarted)
                await WriteAsync(context, StatusCodes.Status500InternalServerError, ErrorResponse.Internal($"chunk_cache: {route} failed"));
        }
    }

    private static Task WriteBusyAsync(HttpContext context, string message)
    {
        context.Response.Headers.RetryAfter = "1";
        return WriteAsync(context, StatusCodes.Status429TooManyRequests, ErrorResponse.RateLimit(message, code: "chunk_control_busy"));
    }

    private static Task PrewarmAsync(HttpContext context, ServerState state) =>
        RunWriteAsync(context, state, "prewarm", MaxPrewarmBodyBytes, (caller, writer, body) => PrewarmAdmittedAsync(context, state, caller, writer, body));

    /// <summary>One text of a prewarm call.</summary>
    private sealed record PrewarmItem(string Text, string? SourcePath, bool SourceUnreadable)
    {
        private bool PrintMembers(System.Text.StringBuilder builder)
        {
            builder.Append(System.Globalization.CultureInfo.InvariantCulture, $"TextLength = {Text.Length}");
            return true;
        }
    }

    /// <summary>
    /// <c>POST /v1/chunk-cache/prewarm</c> (spec §7.2, ruling 14): <c>{"texts": [{"id"?, "text", "source"?: {"path"?}}],
    /// "max_tokens"?}</c>. Text only (no hash list: a hash-only API would be a membership oracle). Each text is tokenised as the
    /// chat path does (<see cref="ChunkTokenization.Encode"/>), keyed in the caller's namespace, and either found stored, or
    /// admitted by the sole writer's rules except seen-twice (<see cref="ChunkStoreService.AdmitPrewarm"/>: deny-list on
    /// <c>source.path</c>, PEM sniff, CPU 512-token cap, budget, queue shares), without a sighting. Request thread only: no
    /// lock file, no store write.
    /// </summary>
    private static async Task PrewarmAdmittedAsync(HttpContext context, ServerState state, ChunkCaller caller, ChunkStoreService writer, ReadOnlyMemory<byte> body)
    {
        var (items, maxTokens, error, tooMany) = ParsePrewarm(body);
        if (items is null)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest,
                ErrorResponse.InvalidRequest(error!, param: tooMany ? "texts" : null, code: tooMany ? ChunkCacheCodes.TooMany : null));
            return;
        }
        if (!state.IsReady || state.Tokenizer is null)
        {
            await WriteAsync(context, StatusCodes.Status503ServiceUnavailable, ErrorResponse.Internal("No model loaded", code: "model_not_loaded"));
            return;
        }
        var identity = state.ChunkCacheRuntime.IdentityFor(state);
        if (!identity.Supported)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest,
                ErrorResponse.InvalidRequest("prewarm: chunk reuse does not support the served model or backend", code: ChunkCacheCodes.ModelUnsupported));
            return;
        }

        int already = 0, queued = 0, refused = 0;
        long queuedTokens = 0;
        var reasons = new SortedSet<string>(StringComparer.Ordinal);
        var decided = new Dictionary<string, (bool Queued, bool Stored, string? Reason)>(StringComparer.Ordinal);
        foreach (var item in items)
        {
            int[] ids = ChunkTokenization.Encode(state.Tokenizer, item.Text);
            (bool Queued, bool Stored, string? Reason) outcome;
            if (ids.Length == 0)
            {
                outcome = (false, false, AdmissionReasons.Empty);
            }
            else
            {
                var key = identity.KeyOf(ids, caller.Namespace);
                if (decided.TryGetValue(key.Hex, out var earlier))
                {
                    outcome = earlier; // the same text twice in one call: decided once, its tokens counted once
                }
                else
                {
                    if (writer.IsStored(key))
                    {
                        outcome = (false, true, null);
                    }
                    else if (maxTokens is { } cap && queuedTokens + ids.Length > cap)
                    {
                        outcome = (false, false, AdmissionReasons.MaxTokens);
                    }
                    else
                    {
                        var span = new ChunkSpanRef(0, 0, 0, System.Text.Encoding.UTF8.GetByteCount(item.Text), "", WireChunking.Hash(item.Text),
                            null, false, false, item.Text, new EngineSpan(0, 0, item.Text.Length))
                        {
                            Index = 0, SourcePath = item.SourcePath, SourceUnreadable = item.SourceUnreadable,
                        };
                        var result = writer.AdmitPrewarm(span, key, ids.Length, caller, identity.Backend);
                        outcome = (result.Queued, false, result.Reason);
                        if (result.Queued)
                            queuedTokens += ids.Length;
                    }
                    decided[key.Hex] = outcome;
                }
            }
            if (outcome.Stored)
                already++;
            else if (outcome.Queued)
                queued++;
            else
            {
                refused++;
                reasons.Add(outcome.Reason ?? AdmissionReasons.QueueFull);
            }
        }

        await context.Response.WriteAsJsonAsync(new ChunkCachePrewarmDto
        {
            Requested = items.Count,
            AlreadyStored = already,
            Queued = queued,
            QueuedTokens = (int)Math.Min(int.MaxValue, queuedTokens),
            Refused = refused,
            RefusedReasons = [.. reasons],
        }, ServerJsonContext.Default.ChunkCachePrewarmDto, cancellationToken: context.RequestAborted);
    }

    /// <summary>
    /// Parses a prewarm body strictly: only <c>texts</c> and <c>max_tokens</c> at the top, only <c>id</c>, <c>text</c> and
    /// <c>source</c> (<c>{"path"}</c>) per item, no repeated property; anything else is refused (so no hash list rides along).
    /// Error messages never echo the content.
    /// </summary>
    private static (List<PrewarmItem>? Items, long? MaxTokens, string? Error, bool TooMany) ParsePrewarm(ReadOnlyMemory<byte> body)
    {
        const string shape = "prewarm takes {\"texts\": [{\"id\"?, \"text\", \"source\"?: {\"path\"?}}], \"max_tokens\"?} and nothing else (text only: no hashes)";
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(body);
        }
        catch (JsonException)
        {
            return (null, null, shape + "; the body is not JSON", false);
        }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return (null, null, shape, false);
            JsonElement? texts = null;
            long? maxTokens = null;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in root.EnumerateObject())
            {
                if (!seen.Add(property.Name))
                    return (null, null, shape + "; a field is repeated", false);
                switch (property.Name)
                {
                    case "texts":
                        texts = property.Value;
                        break;
                    case "max_tokens":
                        if (property.Value.ValueKind != JsonValueKind.Number || !property.Value.TryGetInt64(out long cap) || cap < 1)
                            return (null, null, "prewarm: max_tokens must be a positive integer", false);
                        maxTokens = cap;
                        break;
                    default:
                        return (null, null, shape + "; the body has another field", false);
                }
            }
            if (texts is not { ValueKind: JsonValueKind.Array } array)
                return (null, null, shape + "; texts must be an array", false);
            if (array.GetArrayLength() > MaxPrewarmItems)
                return (null, null, $"prewarm: at most {MaxPrewarmItems} texts per call", true);
            var items = new List<PrewarmItem>(array.GetArrayLength());
            foreach (var element in array.EnumerateArray())
            {
                if (element.ValueKind != JsonValueKind.Object)
                    return (null, null, shape + "; each text is an object", false);
                string? text = null, path = null;
                bool sourceUnreadable = false;
                var fields = new HashSet<string>(StringComparer.Ordinal);
                foreach (var f in element.EnumerateObject())
                {
                    if (!fields.Add(f.Name))
                        return (null, null, shape + "; a field is repeated", false);
                    switch (f.Name)
                    {
                        case "id":
                            if (f.Value.ValueKind != JsonValueKind.String)
                                return (null, null, "prewarm: id must be a string", false);
                            break;
                        case "text":
                            if (f.Value.ValueKind != JsonValueKind.String)
                                return (null, null, "prewarm: text must be a string", false);
                            text = f.Value.GetString();
                            break;
                        case "source":
                            if (f.Value.ValueKind != JsonValueKind.Object)
                                return (null, null, "prewarm: source must be an object ({\"path\": ...})", false);
                            foreach (var sp in f.Value.EnumerateObject())
                            {
                                if (!string.Equals(sp.Name, "path", StringComparison.Ordinal) || path is not null || sourceUnreadable)
                                    return (null, null, "prewarm: source takes path only", false);
                                if (sp.Value.ValueKind == JsonValueKind.String)
                                    path = sp.Value.GetString();
                                else
                                    return (null, null, "prewarm: source.path must be a string", false);
                            }
                            break;
                        default:
                            return (null, null, shape + "; an item has another field", false);
                    }
                }
                if (text is null)
                    return (null, null, shape + "; each item needs its text", false);
                items.Add(new PrewarmItem(text, path, sourceUnreadable));
            }
            return (items, maxTokens, null, false);
        }
    }

    private static Task EvictAsync(HttpContext context, ServerState state) =>
        RunWriteAsync(context, state, "evict", MaxEvictBodyBytes, (caller, writer, body) => EvictAdmittedAsync(context, caller, writer, body));

    /// <summary>
    /// <c>POST /v1/chunk-cache/evict</c> (spec §7.2, §10): <c>{"mode"?: "orphans" | "path" | "namespace" | "all", "path"?,
    /// "namespace"?, "dry_run"?}</c>. <c>dry_run</c> defaults to true: only an explicit JSON <c>false</c> deletes. The namespace
    /// is the caller's (server-assigned); the body's <c>namespace</c> is honoured only for an admin key (else ignored with the
    /// warning <c>chunk_namespace_ignored</c>). <c>orphans</c> is store-wide and admin-only; <c>all</c> without a namespace
    /// wipes the store for an admin and the caller's namespace for anyone else. Fails closed (503) without deleting anything.
    /// </summary>
    private static async Task EvictAdmittedAsync(HttpContext context, ChunkCaller caller, ChunkStoreService writer, ReadOnlyMemory<byte> body)
    {
        var (parsed, error) = ParseEvict(body);
        if (parsed is null)
        {
            await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(error!));
            return;
        }
        var (mode, glob, bodyNamespace, dryRun) = parsed.Value;

        string[]? warnings = null;
        string? target = caller.Namespace;
        if (bodyNamespace is not null)
        {
            if (caller.IsAdmin)
            {
                string? named = bodyNamespace == ChunkNamespace.ReservedDefaultName ? ChunkNamespace.Default
                    : bodyNamespace.Length > 0 && ChunkNamespace.IsValid(bodyNamespace) ? bodyNamespace : null;
                if (named is null)
                {
                    await WriteAsync(context, StatusCodes.Status400BadRequest, ErrorResponse.InvalidRequest(
                        "evict: namespace must be 'default' or 1-64 characters of [a-z0-9._-] (as in the keys file's ns=, or k-<16 hex>)", param: "namespace"));
                    return;
                }
                if (mode == EvictMode.Orphans)
                {
                    await WriteAsync(context, StatusCodes.Status400BadRequest,
                        ErrorResponse.InvalidRequest("evict: orphans are store-wide; namespace does not apply", param: "namespace"));
                    return;
                }
                target = named;
            }
            else if (!string.Equals(bodyNamespace, caller.Namespace, StringComparison.Ordinal))
            {
                warnings = [ChunkCacheCodes.NamespaceIgnored]; // spec §10: namespaces are server-assigned
            }
        }
        if (mode == EvictMode.Orphans)
        {
            if (!caller.IsAdmin)
            {
                context.Response.StatusCode = StatusCodes.Status403Forbidden;
                await context.Response.WriteAsJsonAsync(new ErrorResponse
                {
                    Error = new ErrorDetail
                    {
                        Message = "evict: orphans are store-wide; only an admin key (or the local user of a server without keys) may evict them.",
                        Type = ChunkCacheAuth.PermissionErrorType,
                        Code = "chunk_cache_admin_only",
                    },
                }, ServerJsonContext.Default.ErrorResponse, cancellationToken: context.RequestAborted);
                return;
            }
            target = null;
        }
        else if (mode == EvictMode.All && caller.IsAdmin && bodyNamespace is null)
        {
            target = null; // an admin's all: the whole store
        }

        // Ruling (Task 16 review S2): deleting store-wide (orphans, an admin's all) or in another namespace needs an admin
        // KEY; the local user of a keyless server may dry-run those and delete in its own namespace only.
        if (!dryRun && (target is null || !string.Equals(target, caller.Namespace, StringComparison.Ordinal)) && !caller.IsAdminKey)
        {
            context.Response.StatusCode = StatusCodes.Status403Forbidden;
            await context.Response.WriteAsJsonAsync(new ErrorResponse
            {
                Error = new ErrorDetail
                {
                    Message = "evict: deleting store-wide (orphans, all) or in another namespace needs an admin key from --chunk-cache-api-keys "
                        + "(a server without keys allows dry runs of those and deletes in its own namespace only).",
                    Type = ChunkCacheAuth.PermissionErrorType,
                    Code = "chunk_cache_admin_key_required",
                },
            }, ServerJsonContext.Default.ErrorResponse, cancellationToken: context.RequestAborted);
            return;
        }

        EvictOutcome outcome;
        try
        {
            outcome = await writer.EvictAsync(new EvictRequest(mode, target, glob, dryRun), context.RequestAborted);
        }
        catch (EvictRefusedException ex)
        {
            await WriteAsync(context, StatusCodes.Status503ServiceUnavailable,
                ErrorResponse.Internal($"evict: {ex.Message} ({ex.Detail})", code: ChunkCacheCodes.StoreUnavailable));
            return;
        }
        await context.Response.WriteAsJsonAsync(new ChunkCacheEvictDto
        {
            Evicted = outcome.Evicted,
            FreedBytes = outcome.FreedBytes,
            KeptReferenced = outcome.KeptReferenced,
            DryRun = dryRun,
            Removed = new ChunkCacheEvictRemovedDto
            {
                LedgerRows = outcome.LedgerRows,
                IndexFiles = outcome.IndexFiles,
                Sightings = outcome.Sightings,
                Queued = outcome.Queued,
            },
            Failed = outcome.Partial
                ? new ChunkCacheEvictFailedDto { EntryFiles = outcome.FailedEntryFiles, IndexFiles = outcome.FailedIndexFiles }
                : null,
            Warnings = outcome.Partial ? [.. warnings ?? [], EvictPartial] : warnings,
        }, ServerJsonContext.Default.ChunkCacheEvictDto, cancellationToken: context.RequestAborted);
    }

    /// <summary>
    /// Parses an evict body strictly (only <c>mode</c>, <c>path</c>, <c>namespace</c>, <c>dry_run</c>, each once and of its
    /// type; <c>path</c> required by and only allowed for <c>mode: path</c>, a valid non-negated relative glob). Error
    /// messages never echo the content.
    /// </summary>
    private static ((EvictMode Mode, string? Glob, string? Namespace, bool DryRun)? Parsed, string? Error) ParseEvict(ReadOnlyMemory<byte> body)
    {
        const string shape = "evict takes {\"mode\"?: \"orphans\" | \"path\" | \"namespace\" | \"all\", \"path\"?, \"namespace\"?, \"dry_run\"?: true | false}";
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(body);
        }
        catch (JsonException)
        {
            return (null, shape + "; the body is not JSON");
        }
        using (doc)
        {
            if (doc.RootElement.ValueKind != JsonValueKind.Object)
                return (null, shape);
            var mode = EvictMode.Orphans;
            string? glob = null, ns = null;
            bool dryRun = true;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in doc.RootElement.EnumerateObject())
            {
                if (!seen.Add(p.Name))
                    return (null, shape + "; a field is repeated");
                switch (p.Name)
                {
                    case "mode":
                        if (p.Value.ValueKind != JsonValueKind.String || !Modes.TryGetValue(p.Value.GetString()!, out mode))
                            return (null, "evict: mode is one of orphans, path, namespace, all (older CLI names such as all-for-model are not modes)");
                        break;
                    case "path":
                        if (p.Value.ValueKind != JsonValueKind.String)
                            return (null, "evict: path must be a string (a glob)");
                        glob = p.Value.GetString();
                        break;
                    case "namespace":
                        if (p.Value.ValueKind != JsonValueKind.String)
                            return (null, "evict: namespace must be a string");
                        ns = p.Value.GetString();
                        break;
                    case "dry_run":
                        if (p.Value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                            return (null, "evict: dry_run must be true or false (JSON booleans; it defaults to true)");
                        dryRun = p.Value.GetBoolean();
                        break;
                    default:
                        return (null, shape + "; the body has another field");
                }
            }
            if (mode == EvictMode.Path)
            {
                if (string.IsNullOrWhiteSpace(glob))
                    return (null, "evict: mode path needs path (a glob relative to the tree root, e.g. src/**)");
                if (glob.Length > MaxGlobChars)
                    return (null, $"evict: path is longer than {MaxGlobChars} characters");
                try
                {
                    if (DotLLM.ChunkReuse.Tooling.PathGlob.Parse(glob).Negated)
                        return (null, "evict: path cannot be a negated ('!') glob");
                }
                catch (ArgumentException)
                {
                    return (null, "evict: path must be a relative glob (not empty, not rooted, no '..' segment)");
                }
            }
            else if (glob is not null)
            {
                return (null, "evict: path applies to mode path only");
            }
            return ((mode, glob, ns, dryRun), null);
        }
    }

    /// <summary>
    /// Reads a POST body within <see cref="BodyReadTimeout"/> and <paramref name="max"/> bytes, before any slot is taken.
    /// Null when a 413 (too large) or 408 (too slow) was written.
    /// </summary>
    private static async Task<ReadOnlyMemory<byte>?> ReadBodyAsync(HttpContext context, int max, string route)
    {
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
        deadline.CancelAfter(BodyReadTimeout);
        ReadOnlyMemory<byte>? body;
        try
        {
            body = await ReadLimitedAsync(context, max, deadline.Token);
        }
        catch (OperationCanceledException) when (!context.RequestAborted.IsCancellationRequested)
        {
            // The rest of the body is never read: close the connection rather than drain it (as explain does).
            context.Response.Headers.Connection = "close";
            await WriteAsync(context, StatusCodes.Status408RequestTimeout, ErrorResponse.InvalidRequest(
                $"{route}: the request body did not arrive within {BodyReadTimeout.TotalSeconds:0.#} s", code: "request_timeout"));
            return null;
        }
        if (body is null)
        {
            await WriteAsync(context, StatusCodes.Status413PayloadTooLarge,
                ErrorResponse.InvalidRequest($"{route}: the body is larger than {max} bytes", param: route == "explain" ? "request" : null, code: "request_too_large"));
            return null;
        }
        return body;
    }

    /// <summary>Whether a <c>Content-Type</c> is <c>application/json</c> (parameters such as <c>charset</c> allowed).</summary>
    private static bool IsJson(string? contentType) =>
        Microsoft.Net.Http.Headers.MediaTypeHeaderValue.TryParse(contentType, out var media)
        && string.Equals(media.MediaType.Value, "application/json", StringComparison.OrdinalIgnoreCase);

    private static Task WriteTooLargeAsync(HttpContext context) =>
        WriteAsync(context, StatusCodes.Status413PayloadTooLarge,
            ErrorResponse.InvalidRequest($"explain: the body is larger than {MaxExplainBodyBytes} bytes", param: "request", code: "request_too_large"));

    private static Task WriteCrossSiteAsync(HttpContext context)
    {
        context.Response.StatusCode = StatusCodes.Status403Forbidden;
        return context.Response.WriteAsJsonAsync(new ErrorResponse
        {
            Error = new ErrorDetail
            {
                Message = "The chunk-cache API refuses cross-site browser requests (Sec-Fetch-Site: cross-site).",
                Type = ChunkCacheAuth.PermissionErrorType,
                Code = "chunk_cache_cross_site",
            },
        }, ServerJsonContext.Default.ErrorResponse, cancellationToken: context.RequestAborted);
    }

    private static Task WriteExplainAsync(HttpContext context, ChunkCacheExplainDto body) =>
        context.Response.WriteAsJsonAsync(body, ServerJsonContext.Default.ChunkCacheExplainDto, cancellationToken: context.RequestAborted);

    private static Task WriteChunkErrorAsync(HttpContext context, ChunkCacheError error) =>
        WriteAsync(context, error.Status, error.Status == StatusCodes.Status400BadRequest
            ? ErrorResponse.InvalidRequest(error.Message, param: "chunk_cache", code: error.Code)
            : ErrorResponse.Internal(error.Message, code: error.Code));

    private static Task WriteAsync(HttpContext context, int status, ErrorResponse body)
    {
        context.Response.StatusCode = status;
        return context.Response.WriteAsJsonAsync(body, ServerJsonContext.Default.ErrorResponse, cancellationToken: context.RequestAborted);
    }
}
