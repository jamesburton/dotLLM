using System.Text;
using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server.Models;
using DotLLM.Server.RateLimiting;
using DotLLM.Tokenizers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/chat/completions — OpenAI-compatible chat completion endpoint.
/// Supports both non-streaming (JSON response) and streaming (SSE).
/// </summary>
public static class ChatCompletionEndpoint
{
    private static readonly string[] CommonStopSequences =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>"];

    public static void Map(WebApplication app) =>
        app.MapPost("/v1/chat/completions", HandleAsync);

    private static async Task HandleAsync(
        ChatCompletionRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        // (#369) Activate the requested model — a cheap field-swap if it's already resident, a
        // lazy reload if it idled out, or a fresh load otherwise. No-op when request.Model is null
        // or already active.
        var activationError = await state.EnsureActiveAsync(request.Model, request.KeepAlive, httpContext.RequestAborted);
        if (activationError is not null)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.InvalidRequest(activationError, param: "model", code: "model_not_found"),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null)
        {
            httpContext.Response.StatusCode = 503;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.Internal("No model loaded", code: "model_not_loaded"),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // Validate request structure
        var validationError = RequestValidator.ValidateChatRequest(request);
        if (validationError is not null)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.InvalidRequest(validationError),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // n is implemented for the non-streaming path only (#460). Streaming would need per-choice
        // index on every delta and an n-way interleave of the SSE stream; rather than silently
        // returning one choice -- the defect this closes -- say so.
        if (request.Stream && request.ChoiceCount != 1)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.InvalidRequest(
                    "n > 1 is not supported with stream: true. Request the choices without streaming, "
                    + "or issue n separate streaming requests.",
                    param: "n"),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        var ct = httpContext.RequestAborted;
        var requestId = RequestConverter.GenerateRequestId();
        var modelId = state.Options.ModelId;
        var generator = state.Generator;

        // Validate prefix_id reference (Step 37): must be registered if supplied.
        if (!string.IsNullOrWhiteSpace(request.PrefixId))
        {
            var mgr = state.PrefixTrieManager;
            if (mgr is null || mgr.InspectNamedPrefix(request.PrefixId) is null)
            {
                httpContext.Response.StatusCode = 400;
                await httpContext.Response.WriteAsJsonAsync(
                    ErrorResponse.InvalidRequest($"prefix_id '{request.PrefixId}' is not registered. POST /v1/prompt-cache/{request.PrefixId} first.", param: "prefix_id"),
                    ServerJsonContext.Default.ErrorResponse,
                    contentType: null,
                    httpContext.RequestAborted);
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
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.InvalidRequest(ex.Message, param: "lora_adapter"),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // Convert DTOs to engine types
        var messages = RequestConverter.ToMessages(request.Messages);
        var tools = RequestConverter.ToTools(request.Tools);
        var toolChoice = RequestConverter.ParseToolChoice(request.ToolChoice);

        // Apply chat template
        var templateOptions = new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools,
        };
        string prompt = state.ChatTemplate.Apply(messages, templateOptions);

        // Validate prompt length against model context
        int maxTokens = request.MaxTokens ?? state.SamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
            maxTokens, out int effectiveMaxTokens, out _);
        if (promptError is not null)
        {
            httpContext.Response.StatusCode = 400;
            await httpContext.Response.WriteAsJsonAsync(
                ErrorResponse.InvalidRequest(promptError, param: "messages", code: "context_length_exceeded"),
                ServerJsonContext.Default.ErrorResponse,
                contentType: null,
                httpContext.RequestAborted);
            return;
        }

        // Build inference options with clamped max_tokens
        var stopSequences = CommonStopSequences;
        var options = RequestConverter.ToInferenceOptions(request, stopSequences,
            state.SamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(
                state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        // #456: tool_choice was parsed at the top of this method and then DISCARDED — the local
        // had exactly one reference, its own assignment — so required/none/named-function had no
        // effect on generation. The Anthropic surface has honoured it since #449; this shares
        // that implementation rather than forking a second one.
        var effectiveToolParser = ToolChoiceBinder.Apply(
            toolChoice, tools, state.ToolCallParser, ref options, out bool forcedToolCall);

        // Diffusion routing: when the loaded model is a masked text-diffusion model, generation runs
        // through DiffusionTextGenerator (canvas denoising) instead of the autoregressive TextGenerator.
        // AR models leave DiffusionGenerator null and fall through to the unchanged path below.
        if (state.DiffusionGenerator is { } diffusionGenerator)
        {
            // Per-request diffusion overrides (canvas length / steps / temperature schedule) build a
            // fresh generator over the same model with a tweaked DiffusionConfig; absent overrides reuse
            // the load-time generator (verified DiffusionConfig defaults). max_tokens → target length.
            var effective = ResolveDiffusionGenerator(diffusionGenerator, state, request.Diffusion);

            if (request.ChoiceCount != 1)
            {
                httpContext.Response.StatusCode = 400;
                await httpContext.Response.WriteAsJsonAsync(
                    ErrorResponse.InvalidRequest(
                        "n > 1 is not supported for diffusion models.", param: "n"),
                    ServerJsonContext.Default.ErrorResponse,
                    contentType: null, httpContext.RequestAborted);
                return;
            }

            if (request.Stream)
                await HandleDiffusionStreamingAsync(request, effective, state, httpContext,
                    prompt, effectiveMaxTokens, requestId, modelId, ct);
            else
                await HandleDiffusionNonStreamingAsync(request, effective, state, httpContext,
                    prompt, effectiveMaxTokens, requestId, modelId, ct);
            return;
        }

        if (request.Stream)
            await HandleStreamingAsync(request, generator, state, httpContext, prompt, options,
                requestId, modelId, tools, effectiveToolParser, adapter, ct);
        else
            await HandleNonStreamingAsync(request, generator, state, httpContext, prompt, options,
                requestId, modelId, tools, effectiveToolParser, adapter, ct);
    }

    private static async Task HandleNonStreamingAsync(
        ChatCompletionRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        ToolDefinition[]? tools,
        IToolCallParser? toolParser,
        DotLLM.Core.Lora.ILoraAdapter? adapter,
        CancellationToken ct)
    {
        int choiceCount = request.ChoiceCount;
        var results = new InferenceResponse[choiceCount];

        // Route through the continuous-batch scheduler when it's the right shape for it: no LoRA
        // adapter, no logprobs capture. Multiple concurrent requests pipeline through one model
        // dispatch per scheduler iteration.
        if (state.Scheduler is { } scheduler && adapter is null && !options.Logprobs)
        {
            int[] promptIds = state.Tokenizer!.Encode(prompt);
            var inFlight = new Task<InferenceResponse>[choiceCount];
            for (int i = 0; i < choiceCount; i++)
            {
                inFlight[i] = scheduler.EnqueueAsync(new InferenceRequest
                {
                    TokenIds = promptIds,
                    Options = SeedForChoice(options, i),
                }, ct);
            }
            // All n submitted before awaiting any: the scheduler batches concurrent sequences into
            // one model dispatch per iteration, so n choices cost far less than n sequential runs.
            results = await Task.WhenAll(inFlight);
        }
        else
        {
            // The generator path is serialized by ExecuteAsync, so the choices run one at a time.
            for (int i = 0; i < choiceCount; i++)
            {
                var choiceOptions = SeedForChoice(options, i);
                InferenceResponse? one = null;
                await state.ExecuteAsync(async () =>
                {
                    one = generator.Generate(prompt, choiceOptions, adapter: adapter);
                }, ct);
                results[i] = one!;
            }
        }

        var choices = new ChatChoiceDto[choiceCount];
        for (int i = 0; i < choiceCount; i++)
            choices[i] = BuildChoice(results[i], i);

        var usage = BuildMultiChoiceUsage(results);

        var response = new ChatCompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = choices,
            Usage = usage,
        };

        // Report actuals to the rate-limit lease so unused token budget is refunded. Every choice
        // ran the prompt, so the metered cost is n prompts plus the summed completions -- which is
        // NOT what the reported usage says, because OpenAI counts the prompt once however many
        // choices it produced. The limiter meters real work; the response reports the convention.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(results[0].PromptTokenCount * choiceCount
                + (usage.TotalTokens - usage.PromptTokens));

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response, ServerJsonContext.Default.ChatCompletionResponse, ct);
        return;

        // Local: turn one engine response into one choice. Identical to the single-choice path it
        // replaces; the index is the only thing that varies.
        ChatChoiceDto BuildChoice(InferenceResponse result, int index)
        {
        // Detect tool calls
        string text = result!.Text;
        ToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;

        // #456: the EFFECTIVE parser, not state's. tool_choice:none yields null here so no tool
        // call is ever reported, and a forced choice yields the markerless parser that matches
        // the constrained output.
        if (toolParser is not null && tools is { Length: > 0 })
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, toolParser);
            text = enriched.Text;
            toolCalls = ApplyParallelToolCalls(enriched.ToolCalls, request.ParallelToolCalls);
            finishReason = enriched.FinishReason;
        }

        // Strip stop-sequence suffixes. Only when the engine did NOT report a match: since #459 it
        // trims the matched stop string itself, and stripping again would eat a second copy from
        // text that legitimately ends with a repeat ("wait!!" with stop "!" becoming "wait").
        if (result.MatchedStopSequence is null)
        {
            foreach (var seq in options.StopSequences)
            {
                if (text.EndsWith(seq, StringComparison.Ordinal))
                {
                    text = text[..^seq.Length];
                    break;
                }
            }
        }

        var message = new ChatMessageDto
        {
            Role = "assistant",
            Content = toolCalls is { Length: > 0 } ? null : text,
            ToolCalls = toolCalls is { Length: > 0 }
                ? RequestConverter.ToToolCallDtos(toolCalls)
                : null,
        };

        var logprobsDto = result.Logprobs is { Length: > 0 }
            ? RequestConverter.ToLogprobsDto(result.Logprobs)
            : null;

        return new ChatChoiceDto
        {
            Index = index,
            Message = message,
            Logprobs = logprobsDto,
            FinishReason = RequestConverter.ToFinishReasonString(finishReason),
        };
        }
    }

    /// <summary>
    /// Usage across the <c>n</c> choices of one request. OpenAI reports the prompt <b>once</b>
    /// however many choices were produced, and sums <c>completion_tokens</c> across them -- so
    /// <c>total_tokens</c> is not <c>n x</c> anything, and a client reconciling spend against it
    /// would be misled by any other arrangement.
    /// </summary>
    internal static UsageDto BuildMultiChoiceUsage(IReadOnlyList<InferenceResponse> results)
    {
        int promptTokens = results.Count > 0 ? results[0].PromptTokenCount : 0;
        int completionTokens = 0;
        for (int i = 0; i < results.Count; i++)
            completionTokens += results[i].GeneratedTokenCount;

        return new UsageDto
        {
            PromptTokens = promptTokens,
            CompletionTokens = completionTokens,
            TotalTokens = promptTokens + completionTokens,
        };
    }

    /// <summary>
    /// Per-choice sampling seed for <c>n &gt; 1</c>. Choice 0 keeps the caller's seed exactly, so
    /// single-choice behaviour and its determinism are unchanged; later choices offset it so a
    /// seeded request does not return n identical completions. An unseeded request already varies
    /// (the pipeline builds a fresh <c>Random</c>) and is left alone.
    /// </summary>
    internal static DotLLM.Core.Configuration.InferenceOptions SeedForChoice(
        DotLLM.Core.Configuration.InferenceOptions options, int index)
        => index == 0 || options.Seed is null
            ? options
            : options with { Seed = unchecked(options.Seed.Value + index) };

    private static async Task HandleStreamingAsync(
        ChatCompletionRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        ToolDefinition[]? tools,
        IToolCallParser? toolParser,
        DotLLM.Core.Lora.ILoraAdapter? adapter,
        CancellationToken ct)
    {
        // No Connection header: it is connection-specific and illegal over HTTP/2+. See SseResponse.
        SseResponse.ApplyHeaders(httpContext);

        // First chunk: role
        var roleChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto
            {
                Delta = new ChatDeltaDto { Role = "assistant" },
            }],
        };
        await WriteSseChunk(httpContext, roleChunk, ct);

        var sb = new StringBuilder();
        FinishReason finishReason = FinishReason.Length;
        InferenceTimings? timings = null;
        int completionTokens = 0;

        await state.ExecuteAsync(async () =>
        {
            await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options, ct, adapter))
            {
                if (token.Text.Length > 0)
                {
                    completionTokens++;
                    sb.Append(token.Text);
                    var tokenLogprobs = token.Logprobs.HasValue
                        ? RequestConverter.ToLogprobsDto(token.Logprobs.Value)
                        : null;
                    var contentChunk = new ChatCompletionChunk
                    {
                        Id = requestId,
                        Model = modelId,
                        Choices = [new ChatChunkChoiceDto
                        {
                            Delta = new ChatDeltaDto { Content = token.Text },
                            Logprobs = tokenLogprobs,
                        }],
                    };
                    await WriteSseChunk(httpContext, contentChunk, ct);
                }

                if (token.FinishReason.HasValue)
                {
                    finishReason = token.FinishReason.Value;
                    timings = token.Timings;
                }
            }
        }, ct);

        // Detect tool calls in accumulated text
        string text = sb.ToString();
        ToolCall[]? toolCalls = null;
        // #456: see the non-streaming path — the effective parser honours tool_choice.
        if (toolParser is not null && tools is { Length: > 0 })
        {
            toolCalls = ApplyParallelToolCalls(toolParser.TryParse(text), request.ParallelToolCalls);
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        // Final chunk with finish_reason
        var finalDelta = toolCalls is { Length: > 0 }
            ? new ChatDeltaDto { ToolCalls = RequestConverter.ToToolCallDtos(toolCalls) }
            : new ChatDeltaDto();

        int promptTokens = timings?.PrefillTokenCount ?? 0;

        // Report actuals to the rate-limit lease so unused token budget is refunded.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(promptTokens + completionTokens);

        var finalChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto
            {
                Delta = finalDelta,
                FinishReason = RequestConverter.ToFinishReasonString(finishReason),
            }],
            // #450/#453: OpenAI puts usage in a DEDICATED final chunk (choices: []) and ONLY
            // when stream_options.include_usage was requested. Carrying it on the last content
            // chunk as well produced two usage-bearing chunks when requested and one when not
            // — the conformance rows openai/usage.stream{,.unrequested} caught both.
            // Timings stays: it is our own extension and no OpenAI client looks for it.
            Timings = timings.HasValue ? new TimingsDto
            {
                PrefillTimeMs = timings.Value.PrefillTimeMs,
                DecodeTimeMs = timings.Value.DecodeTimeMs,
                SamplingTimeMs = timings.Value.SamplingTimeMs,
                PrefillTokensPerSec = timings.Value.PrefillTokensPerSec,
                DecodeTokensPerSec = timings.Value.DecodeTokensPerSec,
                PromptTokens = timings.Value.PrefillTokenCount,
                GeneratedTokens = timings.Value.DecodeTokenCount,
                CachedTokens = timings.Value.CachedTokenCount,
                SpeculativeDraftTokens = timings.Value.SpeculativeDraftTokens,
                SpeculativeAcceptedTokens = timings.Value.SpeculativeAcceptedTokens,
                SpeculativeAcceptanceRate = timings.Value.SpeculativeAcceptanceRate,
            } : null,
        };
        await WriteSseChunk(httpContext, finalChunk, ct);

        // stream_options.include_usage (#450): OpenAI closes the stream with a usage-only chunk
        // (choices: []). The finish_reason chunk above also carries usage as a long-standing
        // dotLLM extension; this adds the shape the SDKs actually look for, without removing it.
        if (request.WantsUsageChunk)
            await WriteSseChunk(httpContext, BuildUsageChunk(requestId, modelId, promptTokens, completionTokens), ct);

        // [DONE] sentinel
        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }

    /// <summary>
    /// Builds OpenAI's final usage chunk (#450): <c>usage</c> populated, <c>choices</c> empty.
    /// SDKs tell it apart from a content chunk by exactly that empty array, so the shape is
    /// load-bearing and not merely cosmetic.
    /// </summary>
    internal static ChatCompletionChunk BuildUsageChunk(
        string requestId, string modelId, int promptTokens, int completionTokens) =>
        new()
        {
            Id = requestId,
            Model = modelId,
            Choices = [],
            Usage = new UsageDto
            {
                PromptTokens = promptTokens,
                CompletionTokens = completionTokens,
                TotalTokens = promptTokens + completionTokens,
            },
        };

    /// <summary>
    /// Enforces <c>parallel_tool_calls: false</c> (#450) by keeping at most the first detected
    /// call. Nothing constrains the model during decode, so the cap is applied on the way out.
    /// Null/true (the OpenAI default) passes the calls through untouched.
    /// </summary>
    internal static ToolCall[]? ApplyParallelToolCalls(ToolCall[]? toolCalls, bool? parallelToolCalls) =>
        parallelToolCalls == false && toolCalls is { Length: > 1 } ? [toolCalls[0]] : toolCalls;

    private static async Task WriteSseChunk(HttpContext ctx, ChatCompletionChunk chunk, CancellationToken ct)
    {
        await ctx.Response.WriteAsync("data: ", ct);
        await JsonSerializer.SerializeAsync(ctx.Response.Body, chunk, ServerJsonContext.Default.ChatCompletionChunk, ct);
        await ctx.Response.WriteAsync("\n\n", ct);
        await ctx.Response.Body.FlushAsync(ct);
    }

    // ─────────────────────────── Diffusion decode path ───────────────────────────

    /// <summary>
    /// Returns the diffusion generator to use for this request. When the request carries no diffusion
    /// overrides the load-time generator (verified <see cref="DotLLM.Core.Models.DiffusionConfig"/>
    /// defaults) is reused as-is. When any override is present a per-request generator is built over the
    /// same model/tokenizer with a <see cref="DotLLM.Core.Models.DiffusionConfig"/> patched from the
    /// overrides — every unset field falls back to the model default.
    /// </summary>
    internal static DiffusionTextGenerator ResolveDiffusionGenerator(
        DiffusionTextGenerator loadTime, ServerState state, DiffusionOptionsDto? overrides)
    {
        if (overrides is null ||
            (overrides.CanvasLength is null && overrides.MaxDenoisingSteps is null &&
             overrides.TemperatureMax is null && overrides.TemperatureMin is null))
        {
            return loadTime;
        }

        var baseConfig = state.Config!.DiffusionConfig!;
        var patched = baseConfig with
        {
            CanvasLength = overrides.CanvasLength is > 0 ? overrides.CanvasLength.Value : baseConfig.CanvasLength,
            MaxDenoisingSteps = overrides.MaxDenoisingSteps is > 0 ? overrides.MaxDenoisingSteps.Value : baseConfig.MaxDenoisingSteps,
            TemperatureMax = overrides.TemperatureMax ?? baseConfig.TemperatureMax,
            TemperatureMin = overrides.TemperatureMin ?? baseConfig.TemperatureMin,
        };
        return new DiffusionTextGenerator(state.Model!, state.Tokenizer!, sampler: null, patched);
    }

    /// <summary>
    /// Runs a diffusion generation, mapping the canvas-streaming hook to progressive decoded-text
    /// deltas (the SSE content fragments). Appends each newly-revealed suffix of the committed leading
    /// run to <paramref name="deltas"/>, per canvas (multi-canvas runs concatenate). Exposed at
    /// assembly scope so the routing/streaming mapping can be unit-tested without an HTTP pipeline.
    /// </summary>
    /// <returns>The completed diffusion result.</returns>
    internal static DiffusionResult RunDiffusionStreaming(
        DiffusionTextGenerator generator, string prompt, int targetLength, List<string> deltas)
    {
        string emittedForCanvas = string.Empty;
        int lastCanvasIndex = -1;

        return generator.Generate(prompt, targetLength: targetLength, onCanvasStep: snapshot =>
        {
            if (snapshot.CanvasIndex != lastCanvasIndex)
            {
                lastCanvasIndex = snapshot.CanvasIndex;
                emittedForCanvas = string.Empty;
            }

            // PartialText is the decoded leading contiguous run of committed tokens for this canvas.
            // Emit only its growth beyond what we already streamed for this canvas.
            string partial = snapshot.PartialText;
            if (partial.Length > emittedForCanvas.Length &&
                partial.StartsWith(emittedForCanvas, StringComparison.Ordinal))
            {
                deltas.Add(partial[emittedForCanvas.Length..]);
                emittedForCanvas = partial;
            }
        });
    }

    /// <summary>
    /// Non-streaming chat completion for a masked text-diffusion model. Runs the full denoise loop
    /// under the request gate and returns the final decoded text + usage. <c>max_tokens</c> maps to
    /// the overall diffusion target length.
    /// </summary>
    private static async Task HandleDiffusionNonStreamingAsync(
        ChatCompletionRequest request,
        DiffusionTextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        int targetLength,
        string requestId, string modelId,
        CancellationToken ct)
    {
        DiffusionResult? result = null;
        await state.ExecuteAsync(() =>
        {
            result = generator.Generate(prompt, targetLength: targetLength);
            return Task.CompletedTask;
        }, ct);

        var message = new ChatMessageDto { Role = "assistant", Content = result!.Text };

        var response = new ChatCompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChoiceDto
            {
                Index = 0,
                Message = message,
                FinishReason = RequestConverter.ToFinishReasonString(result.FinishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };

        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(result.PromptTokenCount + result.GeneratedTokenCount);

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response,
            ServerJsonContext.Default.ChatCompletionResponse, ct);
    }

    /// <summary>
    /// Streaming chat completion for a masked text-diffusion model. Maps the diffusion canvas hook to
    /// SSE deltas: each denoise step that grows the committed (leading, unmasked) text prefix emits the
    /// newly-decoded suffix as a content delta — analogous to HF's <c>TextDiffusionStreamer</c>. A final
    /// usage chunk + <c>[DONE]</c> close the stream.
    /// </summary>
    private static async Task HandleDiffusionStreamingAsync(
        ChatCompletionRequest request,
        DiffusionTextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        int targetLength,
        string requestId, string modelId,
        CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
        httpContext.Response.Headers.Connection = "keep-alive";

        var roleChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto { Delta = new ChatDeltaDto { Role = "assistant" } }],
        };
        await WriteSseChunk(httpContext, roleChunk, ct);

        // The canvas hook fires synchronously on the generation thread. Buffer the decoded-text deltas
        // it produces (see RunDiffusionStreaming) and flush them to the SSE stream after generation
        // completes — the denoise loop is a single blocking call.
        var deltas = new List<string>();

        DiffusionResult? result = null;
        await state.ExecuteAsync(() =>
        {
            result = RunDiffusionStreaming(generator, prompt, targetLength, deltas);
            return Task.CompletedTask;
        }, ct);

        int streamedChars = 0;
        foreach (var delta in deltas)
        {
            if (delta.Length == 0)
                continue;
            streamedChars += delta.Length;
            var contentChunk = new ChatCompletionChunk
            {
                Id = requestId,
                Model = modelId,
                Choices = [new ChatChunkChoiceDto { Delta = new ChatDeltaDto { Content = delta } }],
            };
            await WriteSseChunk(httpContext, contentChunk, ct);
        }

        // The canvas-prefix deltas only cover the leading contiguous committed run per canvas. If the
        // finished text contains more than was streamed progressively (interior positions committed
        // out of order), emit the remainder so the streamed text matches the non-streaming result.
        string finalText = result!.Text;
        if (finalText.Length > streamedChars && finalText.Length > 0)
        {
            var tailChunk = new ChatCompletionChunk
            {
                Id = requestId,
                Model = modelId,
                Choices = [new ChatChunkChoiceDto { Delta = new ChatDeltaDto { Content = finalText[streamedChars..] } }],
            };
            await WriteSseChunk(httpContext, tailChunk, ct);
        }

        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(result.PromptTokenCount + result.GeneratedTokenCount);

        var finalChunk = new ChatCompletionChunk
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChunkChoiceDto
            {
                Delta = new ChatDeltaDto(),
                FinishReason = RequestConverter.ToFinishReasonString(result.FinishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };
        await WriteSseChunk(httpContext, finalChunk, ct);

        // stream_options.include_usage (#450) — same closing shape as the autoregressive path.
        if (request.WantsUsageChunk)
        {
            await WriteSseChunk(httpContext, BuildUsageChunk(
                requestId, modelId, result.PromptTokenCount, result.GeneratedTokenCount), ct);
        }

        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }
}
