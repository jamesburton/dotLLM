using System.Text;
using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Server.RateLimiting;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/completions — OpenAI-compatible raw completion endpoint (no chat template).
/// </summary>
public static class CompletionEndpoint
{

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/completions", HandleAsync);

    private static async Task HandleAsync(
        CompletionRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        // (#369) Activate the requested model — see ChatCompletionEndpoint for the same pattern.
        // (#1023 review) Keep-alive comments while a streaming request waits here for a load or the lock, as on chat.
        await using var keepAlive = request.Stream ? SseKeepAlive.Start(httpContext, state.DescribeWait, state.KeepAliveInterval) : null;
        var activationError = await state.EnsureActiveAsync(request.Model, request.KeepAlive, httpContext.RequestAborted);
        if (activationError is not null)
        {
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 400, ErrorResponse.InvalidRequest(activationError, param: "model", code: DotLLM.HuggingFace.AmbiguousModelReferenceException.CodeFor(activationError)));
            return;
        }

        if (!state.IsReady || state.Generator is null)
        {
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 503, ErrorResponse.Internal("No model loaded", code: "model_not_loaded"));
            return;
        }

        // Validate request structure
        var validationError = RequestValidator.ValidateCompletionRequest(request);
        if (validationError is not null)
        {
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 400, ErrorResponse.InvalidRequest(validationError));
            return;
        }

        var ct = httpContext.RequestAborted;
        var requestId = RequestConverter.GenerateRequestId();
        var modelId = state.Options.ModelId;

        // Validate prefix_id reference (Step 37).
        if (!string.IsNullOrWhiteSpace(request.PrefixId))
        {
            var mgr = state.PrefixTrieManager;
            if (mgr is null || mgr.InspectNamedPrefix(request.PrefixId) is null)
            {
                await SseResponse.WriteOpenAiErrorAsync(httpContext, 400, ErrorResponse.InvalidRequest($"prefix_id '{request.PrefixId}' is not registered. POST /v1/prompt-cache/{request.PrefixId} first.", param: "prefix_id"));
                return;
            }
        }

        // Resolve LoRA adapter (if requested) — bad name → 400 with available list
        DotLLM.Core.Lora.ILoraAdapter? adapter;
        try
        {
            adapter = LoraEndpoints.Resolve(request.LoraAdapter, state);
        }
        catch (LoraAdapterNotFoundException ex)
        {
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 400, ErrorResponse.InvalidRequest(ex.Message, param: "lora_adapter"));
            return;
        }

        // Validate prompt length against model context
        int maxTokens = request.MaxTokens ?? state.EffectiveSamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            request.Prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
            maxTokens, out int effectiveMaxTokens, out _);
        if (promptError is not null)
        {
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 400, ErrorResponse.InvalidRequest(promptError, param: "prompt", code: "context_length_exceeded"));
            return;
        }

        var options = RequestConverter.ToInferenceOptions(request,
            state.EffectiveSamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(
                state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        if (request.Stream && state.Scheduler is { } scheduler && adapter is null && !options.Logprobs)
        {
            // (#1266) Same routing as the non-streaming path: on the scheduler, with no gate or model lease.
            await WriteCompletionStreamAsync(httpContext,
                ServerState.SchedulerStream(scheduler, httpContext, state.Tokenizer!.Encode(request.Prompt), options),
                ServerState.WithoutGate,
                requestId, modelId, request.WantsUsageChunk, ct, state.KeepAliveInterval);
        }
        else if (request.Stream)
        {
            // (#1023) ": waiting for ..." keep-alive comments while the gate is contended; the generator is read once it is held.
            TextGenerator? current = null;
            string prompt = request.Prompt;
            await WriteCompletionStreamAsync(httpContext,
                innerCt => current!.GenerateStreamingTokensAsync(prompt, options, innerCt, adapter),
                SseKeepAlive.Wrap(httpContext,
                    (work, gateCt) => state.ExecuteAsync(modelId, g => { current = g; return work(); }, gateCt),
                    state.DescribeWait, state.KeepAliveInterval),
                requestId, modelId, request.WantsUsageChunk, ct, state.KeepAliveInterval);
        }
        else
            await HandleNonStreamingAsync(state, httpContext, request.Prompt, options,
                requestId, modelId, adapter, ct);
    }

    private static async Task HandleNonStreamingAsync(
        ServerState state, HttpContext httpContext,
        string prompt, DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        DotLLM.Core.Lora.ILoraAdapter? adapter,
        CancellationToken ct)
    {
        InferenceResponse? result = null;

        // Route through the continuous-batch scheduler when it's the right shape for it: no LoRA
        // adapter, no logprobs capture (scheduler doesn't surface per-token logprobs yet). Multiple
        // concurrent requests pipeline through one model dispatch per scheduler iteration.
        if (state.Scheduler is { } scheduler && adapter is null && !options.Logprobs)
        {
            int[] promptIds = state.Tokenizer!.Encode(prompt);
            var inferenceRequest = new InferenceRequest
            {
                TokenIds = promptIds,
                Options = options,
                // Carry the resolved API key (stashed by RateLimitMiddleware) for the scheduler's
                // per-key admission fairness; null when rate limiting / key resolution is off.
                ApiKey = httpContext.Items.TryGetValue(RateLimitMiddleware.ApiKeyItemKey, out var k)
                    ? k as string
                    : null,
            };
            result = await scheduler.EnqueueAsync(inferenceRequest, ct);
        }
        else
        {
            // (#1023) The generator current under the gate, not the one captured before a possible queued swap.
            await state.ExecuteAsync(modelId, current =>
            {
                result = current.Generate(prompt, options, adapter: adapter, cancellationToken: ct);
                return Task.CompletedTask;
            }, ct);
        }

        var logprobsDto = result!.Logprobs is { Length: > 0 }
            ? RequestConverter.ToLogprobsDto(result.Logprobs)
            : null;

        var response = new CompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new CompletionChoiceDto
            {
                Index = 0,
                Text = result.Text,
                Logprobs = logprobsDto,
                FinishReason = RequestConverter.ToFinishReasonString(result.FinishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
                PromptTokensDetails = new PromptTokensDetailsDto { CachedTokens = Math.Clamp(result.Timings.CachedTokenCount, 0, result.PromptTokenCount) },
            },
        };

        // Report actuals to the rate-limit lease so unused token budget is refunded.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(result.PromptTokenCount + result.GeneratedTokenCount);

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response, ServerJsonContext.Default.CompletionResponse, ct);
    }

    /// <summary>
    /// Emits the OpenAI legacy-completions SSE stream for one request. The token source and the model gate are injected, as in
    /// <see cref="ChatCompletionEndpoint.WriteChatStreamAsync"/>, so keep-alives and error handling can be tested without a model.
    /// </summary>
    /// <param name="httpContext">The streaming request.</param>
    /// <param name="tokenSource">Generation, started under the gate. Its first item waits for the prefill.</param>
    /// <param name="execute">The model gate.</param>
    /// <param name="requestId">The <c>cmpl-...</c> id.</param>
    /// <param name="modelId">The model id reported in each chunk.</param>
    /// <param name="includeUsageChunk">Whether to end with a usage-only chunk (<c>stream_options.include_usage</c>).</param>
    /// <param name="ct">The request token.</param>
    /// <param name="keepAliveInterval">Interval between keep-alive comments; the default when null.</param>
    /// <returns>A task that completes when the stream is written.</returns>
    internal static async Task WriteCompletionStreamAsync(
        HttpContext httpContext,
        Func<CancellationToken, IAsyncEnumerable<GenerationToken>> tokenSource,
        Func<Func<Task>, CancellationToken, Task> execute,
        string requestId, string modelId,
        bool includeUsageChunk,
        CancellationToken ct,
        TimeSpan? keepAliveInterval = null)
    {
        // (#1023 review) Stop the activation-phase keep-alive before touching headers (it commits them itself).
        await SseKeepAlive.StopForAsync(httpContext);

        // No Connection header: it is connection-specific and illegal over HTTP/2+. See SseResponse.
        SseResponse.ApplyHeaders(httpContext);

        int completionTokens = 0;
        int promptTokens = 0;
        int cachedTokens = 0;

        try
        {
            await execute(async () =>
            {
                // (#1025) Keep-alive comments while the prefill runs, until the first token.
                await foreach (var token in SseKeepAlive.UntilFirstItem(httpContext, tokenSource(ct), keepAliveInterval, ct))
                {
                    if (token.Text.Length > 0) completionTokens++;
                    if (token.GeneratedTokenCount is int generated) completionTokens = generated;
                    if (token.Timings.HasValue)
                    {
                        promptTokens = token.Timings.Value.PrefillTokenCount;
                        cachedTokens = token.Timings.Value.CachedTokenCount;
                    }

                    var tokenLogprobs = token.Logprobs.HasValue
                        ? RequestConverter.ToLogprobsDto(token.Logprobs.Value)
                        : null;
                    var chunk = new CompletionChunk
                    {
                        Id = requestId,
                        Model = modelId,
                        Timings = token.Timings.HasValue ? TimingsDto.From(token.Timings.Value) : null,
                        Choices = [new CompletionChunkChoiceDto
                        {
                            Text = token.Text,
                            Logprobs = tokenLogprobs,
                            FinishReason = token.FinishReason.HasValue
                                ? RequestConverter.ToFinishReasonString(token.FinishReason.Value)
                                : null,
                        }],
                    };
                    await httpContext.Response.WriteAsync("data: ", ct);
                    await JsonSerializer.SerializeAsync(httpContext.Response.Body, chunk, ServerJsonContext.Default.CompletionChunk, ct);
                    await httpContext.Response.WriteAsync("\n\n", ct);
                    await httpContext.Response.Body.FlushAsync(ct);
                }
            }, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not ModelUnavailableException
                                   and not DotLLM.Core.Configuration.DeviceMemoryAdmissionException && !ct.IsCancellationRequested)
        {
            // A GPU-memory refusal (issue #1068) escapes to UseModelUnavailableHandler, which reports it as a 400
            // context_length_exceeded the client can act on, not a 500.
            // PR #1026 review: once a keep-alive (or a chunk) has committed the SSE response there is no status left to set,
            // and a bare truncation reads as a dropped connection. Report it in-stream, as /v1/messages does for its own errors
            // (chat does not yet).
            await SseResponse.WriteOpenAiErrorAsync(httpContext, 500, ErrorResponse.Internal(ex.Message));
            return;
        }

        // Report actuals to the rate-limit lease so unused token budget is refunded.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(promptTokens + completionTokens);

        // stream_options.include_usage (#450): usage-only chunk with an empty choices array,
        // which is the shape the SDKs match on to close out their token accounting.
        if (includeUsageChunk)
        {
            var usageChunk = new CompletionChunk
            {
                Id = requestId,
                Model = modelId,
                Choices = [],
                Usage = new UsageDto
                {
                    PromptTokens = promptTokens,
                    CompletionTokens = completionTokens,
                    TotalTokens = promptTokens + completionTokens,
                    PromptTokensDetails = new PromptTokensDetailsDto { CachedTokens = Math.Clamp(cachedTokens, 0, promptTokens) },
                },
            };
            await httpContext.Response.WriteAsync("data: ", ct);
            await JsonSerializer.SerializeAsync(httpContext.Response.Body, usageChunk, ServerJsonContext.Default.CompletionChunk, ct);
            await httpContext.Response.WriteAsync("\n\n", ct);
        }

        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }
}
