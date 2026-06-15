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
        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null)
        {
            httpContext.Response.StatusCode = 503;
            await httpContext.Response.WriteAsJsonAsync(
                new ErrorResponse { Error = "No model loaded" },
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
                new ErrorResponse { Error = validationError },
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
                    new ErrorResponse { Error = $"prefix_id '{request.PrefixId}' is not registered. POST /v1/prompt-cache/{request.PrefixId} first." },
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
                new ErrorResponse { Error = ex.Message },
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
                new ErrorResponse { Error = promptError },
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

        // Diffusion routing: when the loaded model is a masked text-diffusion model, generation runs
        // through DiffusionTextGenerator (canvas denoising) instead of the autoregressive TextGenerator.
        // AR models leave DiffusionGenerator null and fall through to the unchanged path below.
        if (state.DiffusionGenerator is { } diffusionGenerator)
        {
            // Per-request diffusion overrides (canvas length / steps / temperature schedule) build a
            // fresh generator over the same model with a tweaked DiffusionConfig; absent overrides reuse
            // the load-time generator (verified DiffusionConfig defaults). max_tokens → target length.
            var effective = ResolveDiffusionGenerator(diffusionGenerator, state, request.Diffusion);

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
                requestId, modelId, tools, adapter, ct);
        else
            await HandleNonStreamingAsync(request, generator, state, httpContext, prompt, options,
                requestId, modelId, tools, adapter, ct);
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
        DotLLM.Core.Lora.ILoraAdapter? adapter,
        CancellationToken ct)
    {
        InferenceResponse? result = null;

        // Route through the continuous-batch scheduler when it's the right shape for it: no LoRA
        // adapter, no logprobs capture. Multiple concurrent requests pipeline through one model
        // dispatch per scheduler iteration.
        if (state.Scheduler is { } scheduler && adapter is null && !options.Logprobs)
        {
            int[] promptIds = state.Tokenizer!.Encode(prompt);
            var inferenceRequest = new InferenceRequest
            {
                TokenIds = promptIds,
                Options = options,
            };
            result = await scheduler.EnqueueAsync(inferenceRequest, ct);
        }
        else
        {
            await state.ExecuteAsync(async () =>
            {
                result = generator.Generate(prompt, options, adapter: adapter);
            }, ct);
        }

        // Detect tool calls
        string text = result!.Text;
        ToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;

        if (state.ToolCallParser is not null && tools is { Length: > 0 })
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, state.ToolCallParser);
            text = enriched.Text;
            toolCalls = enriched.ToolCalls;
            finishReason = enriched.FinishReason;
        }

        // Strip stop sequence suffixes
        foreach (var seq in options.StopSequences)
        {
            if (text.EndsWith(seq, StringComparison.Ordinal))
            {
                text = text[..^seq.Length];
                break;
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

        var response = new ChatCompletionResponse
        {
            Id = requestId,
            Model = modelId,
            Choices = [new ChatChoiceDto
            {
                Index = 0,
                Message = message,
                Logprobs = logprobsDto,
                FinishReason = RequestConverter.ToFinishReasonString(finishReason),
            }],
            Usage = new UsageDto
            {
                PromptTokens = result.PromptTokenCount,
                CompletionTokens = result.GeneratedTokenCount,
                TotalTokens = result.PromptTokenCount + result.GeneratedTokenCount,
            },
        };

        // Report actuals to the rate-limit lease so unused token budget is refunded.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(result.PromptTokenCount + result.GeneratedTokenCount);

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response, ServerJsonContext.Default.ChatCompletionResponse, ct);
    }

    private static async Task HandleStreamingAsync(
        ChatCompletionRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string requestId, string modelId,
        ToolDefinition[]? tools,
        DotLLM.Core.Lora.ILoraAdapter? adapter,
        CancellationToken ct)
    {
        httpContext.Response.ContentType = "text/event-stream";
        httpContext.Response.Headers.CacheControl = "no-cache";
        httpContext.Response.Headers.Connection = "keep-alive";

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
        if (state.ToolCallParser is not null && tools is { Length: > 0 })
        {
            toolCalls = state.ToolCallParser.TryParse(text);
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
            Usage = new UsageDto
            {
                PromptTokens = promptTokens,
                CompletionTokens = completionTokens,
                TotalTokens = promptTokens + completionTokens,
            },
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

        // [DONE] sentinel
        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }

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

        await httpContext.Response.WriteAsync("data: [DONE]\n\n", ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }
}
