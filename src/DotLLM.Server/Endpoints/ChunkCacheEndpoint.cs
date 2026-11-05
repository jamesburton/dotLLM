using System.Text.Json;
using DotLLM.ChunkReuse.Store;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using Microsoft.Extensions.DependencyInjection;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// The chunk-cache control plane (spec §5.9, §7.2; plan Task 14): <c>GET /v1/chunk-cache</c> (capabilities) and
/// <c>POST /v1/chunk-cache/explain</c> (a dry run of a request's annotations). Mapped only with <c>--chunk-cache</c>: with
/// the flag off the routes do not exist (404, exactly as before the convention). Every call is authenticated by
/// <see cref="ChunkCacheAuth"/> (401 / 403) and, when rate limiting is configured, metered like inference (<c>/v1/</c> paths are metered
/// unless exempted, and these are not). Explain also bounds its own concurrency (429), body size (413) and body read time (408; read before a concurrency slot is taken), and both routes refuse cross-site browser requests; explain reads only <c>application/json</c> (415).
/// </summary>
/// <remarks>
/// Nothing here writes the store, generates, or counts in <c>/metrics</c>; responses and logs never carry a namespace, a key
/// or a key digest (the producer hash in <c>model.producer</c> is a key of no tokens, not a namespace).
/// </remarks>
public static class ChunkCacheEndpoint
{
    /// <summary>The capabilities route.</summary>
    public const string CapabilitiesPath = "/v1/chunk-cache";

    /// <summary>The explain route.</summary>
    public const string ExplainPath = "/v1/chunk-cache/explain";

    /// <summary>Explain calls that may run at once per server; more get 429.</summary>
    public const int MaxConcurrentExplains = 2;

    /// <summary>Largest explain body, bytes; larger is 413.</summary>
    public const int MaxExplainBodyBytes = 4 * 1024 * 1024;

    /// <summary>
    /// Seconds an explain body may take to arrive; slower is 408. The body is read before an explain slot is taken, so a
    /// slow sender never holds one.
    /// </summary>
    public const int ExplainBodyReadTimeoutSeconds = 10;

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

        app.MapGet(CapabilitiesPath, GetCapabilitiesAsync);
        app.MapPost(ExplainPath, ExplainAsync);
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
            // Per-namespace figures need the server ledger (plan Tasks 15-16); until then every caller sees 0, and no caller
            // ever sees store-wide figures (IChunkStore.Keys / TotalBytes span every namespace).
            Store = new ChunkCacheStoreDto
            {
                // Ruling 52: always an object; configured: false without --chunk-store (absent otherwise).
                Configured = options.StoreRoot is null && state.ChunkCacheRuntime.StoreOverride is null ? false : null,
                Entries = 0,
                Bytes = 0,
                BudgetBytes = 0,
                Dtype = DtypeName(KvDtype.F32),
            },
            MaxChunksPerRequest = options.MaxChunks,
        };

        static ChunkCacheModelDto Unsupported() =>
            new() { Id = null, Supported = false, Reason = ChunkCacheCodes.ModelUnsupported };
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
                body = await ReadLimitedAsync(context, deadline.Token);
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
    /// Reads the body, at most <see cref="MaxExplainBodyBytes"/> (also without a <c>Content-Length</c>, e.g. chunked);
    /// null when it is larger. <paramref name="cancellationToken"/> carries the read deadline.
    /// </summary>
    private static async Task<ReadOnlyMemory<byte>?> ReadLimitedAsync(HttpContext context, CancellationToken cancellationToken)
    {
        var buffer = new MemoryStream();
        byte[] chunk = new byte[64 * 1024];
        int read;
        while ((read = await context.Request.Body.ReadAsync(chunk, cancellationToken)) > 0)
        {
            if (buffer.Length + read > MaxExplainBodyBytes)
                return null;
            buffer.Write(chunk, 0, read);
        }
        return buffer.GetBuffer().AsMemory(0, (int)buffer.Length);
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
