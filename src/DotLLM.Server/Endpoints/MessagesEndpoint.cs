using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Server.Models;
using DotLLM.Server.RateLimiting;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Anthropic-compatible Messages API: <c>POST /v1/messages</c>, non-streaming (JSON)
/// and streaming (named SSE events).
/// </summary>
/// <remarks>
/// <para>
/// This reshapes the Anthropic wire format onto the same engine pipeline that backs
/// <see cref="ChatCompletionEndpoint"/> — model residency, chat template, scheduler,
/// sampler and tool-call parser are all shared verbatim; only the request/response
/// shape differs. Reference: <c>https://docs.anthropic.com/en/api/messages</c>.
/// </para>
/// <para>
/// Fork-only feature (#448), completed by #449: <c>POST /v1/messages/count_tokens</c>, the
/// <c>anthropic-version</c>/<c>anthropic-beta</c> headers, a mid-stream <c>error</c> event and
/// input-side <c>thinking</c>/<c>redacted_thinking</c> blocks. dotLLM does not itself emit
/// extended-thinking output blocks — see <c>docs/ANTHROPIC_API.md</c>.
/// </para>
/// </remarks>
public static class MessagesEndpoint
{
    private static readonly string[] CommonStopSequences =
        ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>"];

    /// <summary>
    /// Maps <c>POST /v1/messages</c> and <c>POST /v1/messages/count_tokens</c>
    /// onto <paramref name="app"/>.
    /// </summary>
    /// <param name="app">The web application to map the routes on.</param>
    public static void Map(WebApplication app)
    {
        app.MapPost("/v1/messages", HandleAsync);
        app.MapPost("/v1/messages/count_tokens", HandleCountTokensAsync);
    }

    private static async Task HandleAsync(
        AnthropicMessagesRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        // anthropic-version / anthropic-beta are checked before anything else: a request pinned
        // to a version dotLLM does not implement must not load a model as a side effect (#449).
        var headerError = AnthropicHeaders.Validate(httpContext.Request.Headers);
        if (headerError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", headerError);
            return;
        }

        // (#369) Activate the requested model, mirroring the OpenAI surface: a cheap field-swap
        // when already resident, a lazy reload when it idled out, a fresh load otherwise.
        // Anthropic has no `keep_alive` request field, so the per-model override is left alone.
        var activationError = await state.EnsureActiveAsync(
            request.Model, keepAliveOverride: null, httpContext.RequestAborted);
        if (activationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", activationError);
            return;
        }

        if (!state.IsReady || state.Generator is null || state.ChatTemplate is null)
        {
            await WriteErrorAsync(httpContext, 503, "api_error", "No model loaded");
            return;
        }

        // Masked text-diffusion models generate through DiffusionTextGenerator, not the
        // autoregressive Generator (which ServerStartup still populates). Rather than silently
        // producing autoregressive output from a diffusion checkpoint, refuse — diffusion
        // routing for this surface is out of scope for #448 and tracked with #449.
        if (state.DiffusionGenerator is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error",
                "/v1/messages does not support masked text-diffusion models; use /v1/chat/completions");
            return;
        }

        var validationError = ValidateRequest(request);
        if (validationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", validationError);
            return;
        }

        var ct = httpContext.RequestAborted;
        var messageId = AnthropicConverter.GenerateMessageId();
        // Echo the model that actually served the request, not the requested alias — after
        // EnsureActiveAsync, Options.ModelId is the active model's key.
        var modelId = state.Options.ModelId;
        var generator = state.Generator;

        string prompt = BuildPrompt(request, state, out var tools);

        int maxTokens = request.MaxTokens ?? state.SamplingDefaults.MaxTokens;
        var promptError = RequestValidator.ValidatePromptLength(
            prompt, state.Tokenizer!, state.Config!.MaxSequenceLength,
            maxTokens, out int effectiveMaxTokens, out int promptTokenCount);
        if (promptError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", promptError);
            return;
        }

        var options = AnthropicConverter.ToInferenceOptions(request, CommonStopSequences,
            state.SamplingDefaults,
            new DotLLM.Core.Configuration.ThreadingConfig(state.Options.Threads, state.Options.DecodeThreads));
        options = options with { MaxTokens = effectiveMaxTokens };

        // tool_choice was parsed and then dropped on the floor (#449): the prompt was built with
        // the tools but nothing constrained or suppressed the model, so `{"type":"tool"}` was
        // indistinguishable from `auto` and `none` still let a tool call through.
        var toolChoice = AnthropicConverter.ParseToolChoice(request.ToolChoice);
        var effectiveParser = ApplyToolChoice(
            toolChoice, tools, state.ToolCallParser, ref options, out bool forcedToolCall);

        if (request.Stream)
            await HandleStreamingAsync(request, generator, state, httpContext, prompt, options,
                messageId, modelId, effectiveParser, forcedToolCall, promptTokenCount, ct);
        else
            await HandleNonStreamingAsync(request, generator, state, httpContext, prompt, options,
                messageId, modelId, effectiveParser, ct);
    }

    /// <summary>
    /// Delegates to <see cref="ToolChoiceBinder.Apply"/>. Kept so the Anthropic call sites and
    /// #449's tests read unchanged; the logic is shared with the OpenAI path (#456).
    /// </summary>
    internal static IToolCallParser? ApplyToolChoice(
        ToolChoice toolChoice,
        ToolDefinition[]? tools,
        IToolCallParser? modelParser,
        ref DotLLM.Core.Configuration.InferenceOptions options,
        out bool forcedToolCall)
        => ToolChoiceBinder.Apply(toolChoice, tools, modelParser, ref options, out forcedToolCall);

    private static async Task HandleNonStreamingAsync(
        AnthropicMessagesRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string messageId, string modelId,
        IToolCallParser? toolCallParser,
        CancellationToken ct)
    {
        InferenceResponse? result = null;

        // Route through the continuous-batch scheduler when one is running, same as the OpenAI
        // surface. This surface exposes neither LoRA adapters nor logprobs, so the extra guards
        // ChatCompletionEndpoint needs (adapter is null && !options.Logprobs) are unconditional here.
        if (state.Scheduler is { } scheduler)
        {
            int[] promptIds = state.Tokenizer!.Encode(prompt);
            result = await scheduler.EnqueueAsync(
                new InferenceRequest { TokenIds = promptIds, Options = options }, ct);
        }
        else
        {
            await state.ExecuteAsync(async () =>
            {
                result = generator.Generate(prompt, options);
            }, ct);
        }

        string text = result!.Text;
        ToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;

        if (toolCallParser is not null)
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, toolCallParser);
            text = enriched.Text;
            toolCalls = enriched.ToolCalls;
            finishReason = enriched.FinishReason;
        }

        // Determine whether a caller-supplied stop sequence ended generation, and strip it.
        bool matchedStopSequence = false;
        string? engineStopMatch = null;
        if (finishReason == FinishReason.Stop)
            text = StripAndDetectStopSequence(text, request.StopSequences, options.StopSequences,
                out matchedStopSequence, out engineStopMatch, result.MatchedStopSequence);

        AnthropicContentBlockDto[] content;
        string stopReason;
        if (toolCalls is { Length: > 0 })
        {
            content = AnthropicConverter.ToToolUseBlocks(toolCalls);
            stopReason = "tool_use";
        }
        else
        {
            content = [new AnthropicContentBlockDto { Type = "text", Text = text }];
            stopReason = AnthropicConverter.ToStopReason(finishReason, matchedStopSequence);
        }

        // Prefer what the engine reported; fall back to matching the raw text for callers that
        // report nothing.
        string? stopSequence = stopReason == "stop_sequence"
            ? engineStopMatch ?? MatchStopSequence(result.Text, request.StopSequences)
            : null;

        var response = new AnthropicMessageResponse
        {
            Id = messageId,
            Model = modelId,
            Content = content,
            StopReason = stopReason,
            StopSequence = stopSequence,
            Usage = new AnthropicUsageDto
            {
                InputTokens = result.PromptTokenCount,
                OutputTokens = result.GeneratedTokenCount,
            },
        };

        // Report actuals to the rate-limit lease so unused token budget is refunded. No-op when
        // rate limiting is off or the middleware did not meter this path (see the note on Map).
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(result.PromptTokenCount + result.GeneratedTokenCount);

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response,
            ServerJsonContext.Default.AnthropicMessageResponse, ct);
    }

    private static async Task HandleStreamingAsync(
        AnthropicMessagesRequest request,
        TextGenerator generator,
        ServerState state,
        HttpContext httpContext,
        string prompt,
        DotLLM.Core.Configuration.InferenceOptions options,
        string messageId, string modelId,
        IToolCallParser? toolCallParser,
        bool forcedToolCall,
        int promptTokenCount,
        CancellationToken ct)
        => await WriteMessageStreamAsync(
            httpContext,
            innerCt => generator.GenerateStreamingTokensAsync(prompt, options, innerCt),
            state.ExecuteAsync,
            toolCallParser,
            request.StopSequences,
            messageId, modelId, promptTokenCount, ct, forcedToolCall);

    /// <summary>
    /// Emits the Anthropic SSE event sequence for one streaming request:
    /// <c>message_start</c>, <c>content_block_start</c> + <c>ping</c>, a
    /// <c>content_block_delta</c> per generated token, <c>content_block_stop</c>,
    /// an optional start/delta/stop trio per detected <c>tool_use</c> block, then
    /// <c>message_delta</c> and <c>message_stop</c>.
    /// </summary>
    /// <remarks>
    /// The token source and the model-serialisation gate are injected rather than read
    /// from <see cref="ServerState"/> so the emitted event sequence can be asserted in
    /// unit tests without loading a model. <paramref name="execute"/> wraps only the
    /// generation loop, so <c>message_start</c> still reaches the client before the
    /// request queues behind the model lock.
    /// </remarks>
    internal static async Task WriteMessageStreamAsync(
        HttpContext httpContext,
        Func<CancellationToken, IAsyncEnumerable<GenerationToken>> tokenSource,
        Func<Func<Task>, CancellationToken, Task> execute,
        IToolCallParser? toolCallParser,
        string[]? requestStopSequences,
        string messageId,
        string modelId,
        int promptTokenCount,
        CancellationToken ct,
        bool forcedToolCall = false)
    {
        // No `Connection: keep-alive` — it is connection-specific and illegal over HTTP/2+.
        SseResponse.ApplyHeaders(httpContext);

        // message_start — input_tokens known up front from the prompt.
        var startMessage = new AnthropicMessageResponse
        {
            Id = messageId,
            Model = modelId,
            Content = [],
            StopReason = null,
            StopSequence = null,
            Usage = new AnthropicUsageDto { InputTokens = promptTokenCount, OutputTokens = 0 },
        };
        await WriteEventAsync(httpContext, "message_start",
            new AnthropicMessageStartEvent { Message = startMessage },
            ServerJsonContext.Default.AnthropicMessageStartEvent, ct);

        // Text content block opens at index 0.
        await WriteEventAsync(httpContext, "content_block_start",
            new AnthropicContentBlockStartEvent
            {
                Index = 0,
                ContentBlock = new AnthropicContentBlockDto { Type = "text", Text = "" },
            },
            ServerJsonContext.Default.AnthropicContentBlockStartEvent, ct);
        await WriteEventAsync(httpContext, "ping", new AnthropicPingEvent(),
            ServerJsonContext.Default.AnthropicPingEvent, ct);

        var sb = new StringBuilder();
        FinishReason finishReason = FinishReason.Length;
        int completionTokens = 0;

        // Tool-call markup must not ALSO go out as text_delta: the same payload would be
        // reported twice — once as text, once as the tool_use block emitted below — and an SDK's
        // text_stream would print raw JSON at the user. The accumulator holds text back from the
        // moment the parser recognises a tool call; prose emitted before that is genuine and
        // still streams. Under a forced tool_choice the whole completion is the call, so
        // suppression starts immediately. For `auto`, only a marker-based parser drives
        // suppression: GenericToolCallParser's heuristic fires on any `{ ... "name"`, which would
        // swallow ordinary prose that merely looks JSON-ish.
        var suppressor = toolCallParser is not null && !forcedToolCall && toolCallParser is not GenericToolCallParser
            ? new StreamingToolCallAccumulator(toolCallParser)
            : null;

        try
        {
            await execute(async () =>
            {
                await foreach (var token in tokenSource(ct))
                {
                    if (token.Text.Length > 0)
                    {
                        completionTokens++;
                        sb.Append(token.Text);
                        bool suppress = forcedToolCall || (suppressor?.Append(token.Text) ?? false);
                        if (!suppress)
                        {
                            await WriteEventAsync(httpContext, "content_block_delta",
                                new AnthropicContentBlockDeltaEvent
                                {
                                    Index = 0,
                                    Delta = new AnthropicStreamDeltaDto { Type = "text_delta", Text = token.Text },
                                },
                                ServerJsonContext.Default.AnthropicContentBlockDeltaEvent, ct);
                        }
                    }

                    if (token.FinishReason.HasValue)
                        finishReason = token.FinishReason.Value;
                }
            }, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // The headers and message_start are already on the wire, so there is no status code
            // left to set. The Anthropic stream protocol covers exactly this: a named `error`
            // event carrying the usual envelope, which the SDK turns back into an APIStatusError.
            // Without it the client sees a truncated stream and reports a parse/connection error
            // instead of the failure (#449).
            await WriteEventAsync(httpContext, "error",
                new AnthropicErrorResponse
                {
                    Error = new AnthropicErrorBody { Type = "api_error", Message = ex.Message },
                },
                ServerJsonContext.Default.AnthropicErrorResponse, ct);
            await httpContext.Response.Body.FlushAsync(ct);
            return;
        }

        // Close the text block.
        await WriteEventAsync(httpContext, "content_block_stop",
            new AnthropicContentBlockStopEvent { Index = 0 },
            ServerJsonContext.Default.AnthropicContentBlockStopEvent, ct);

        // Post-generation tool-call detection (mirrors the OpenAI streaming endpoint).
        string text = sb.ToString();
        ToolCall[]? toolCalls = null;
        if (toolCallParser is not null)
        {
            toolCalls = toolCallParser.TryParse(text);
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        // Emit detected tool calls as tool_use blocks after the text block.
        if (toolCalls is { Length: > 0 })
        {
            var blocks = AnthropicConverter.ToToolUseBlocks(toolCalls);
            for (int i = 0; i < blocks.Length; i++)
            {
                int index = i + 1;
                var block = blocks[i];
                await WriteEventAsync(httpContext, "content_block_start",
                    new AnthropicContentBlockStartEvent
                    {
                        Index = index,
                        ContentBlock = new AnthropicContentBlockDto
                        {
                            Type = "tool_use",
                            Id = block.Id,
                            Name = block.Name,
                            Input = AnthropicConverter.ParseInput("{}"),
                        },
                    },
                    ServerJsonContext.Default.AnthropicContentBlockStartEvent, ct);
                await WriteEventAsync(httpContext, "content_block_delta",
                    new AnthropicContentBlockDeltaEvent
                    {
                        Index = index,
                        Delta = new AnthropicStreamDeltaDto
                        {
                            Type = "input_json_delta",
                            PartialJson = block.Input?.GetRawText() ?? "{}",
                        },
                    },
                    ServerJsonContext.Default.AnthropicContentBlockDeltaEvent, ct);
                await WriteEventAsync(httpContext, "content_block_stop",
                    new AnthropicContentBlockStopEvent { Index = index },
                    ServerJsonContext.Default.AnthropicContentBlockStopEvent, ct);
            }
        }

        bool matchedStopSequence = false;
        if (finishReason == FinishReason.Stop)
            matchedStopSequence = MatchStopSequence(text, requestStopSequences) is not null;

        string stopReason = AnthropicConverter.ToStopReason(finishReason, matchedStopSequence);
        string? stopSequence = stopReason == "stop_sequence"
            ? MatchStopSequence(text, requestStopSequences)
            : null;

        // Report actuals to the rate-limit lease so unused token budget is refunded.
        RateLimitMiddleware.GetLease(httpContext)
            ?.ReportActualTokens(promptTokenCount + completionTokens);

        await WriteEventAsync(httpContext, "message_delta",
            new AnthropicMessageDeltaEvent
            {
                Delta = new AnthropicMessageDeltaBody { StopReason = stopReason, StopSequence = stopSequence },
                Usage = new AnthropicUsageDto { InputTokens = promptTokenCount, OutputTokens = completionTokens },
            },
            ServerJsonContext.Default.AnthropicMessageDeltaEvent, ct);

        await WriteEventAsync(httpContext, "message_stop", new AnthropicMessageStopEvent(),
            ServerJsonContext.Default.AnthropicMessageStopEvent, ct);
        await httpContext.Response.Body.FlushAsync(ct);
    }

    /// <summary>Validates the structural invariants of an Anthropic Messages request.</summary>
    /// <param name="request">The deserialized request body.</param>
    /// <returns>An error message, or <see langword="null"/> when the request is well-formed.</returns>
    /// <param name="requireMaxTokens">
    /// True on <c>/v1/messages</c>, where <c>max_tokens</c> is a required field; false on
    /// <c>/v1/messages/count_tokens</c>, whose request body has no <c>max_tokens</c> at all
    /// (see <c>MessageCountTokensParams</c> in the official SDK).
    /// </param>
    internal static string? ValidateRequest(AnthropicMessagesRequest request, bool requireMaxTokens = true)
    {
        if (request.Messages is null || request.Messages.Length == 0)
            return "messages: at least one message is required";

        if (request.Messages.Length > RequestValidator.MaxMessages)
            return $"messages: array exceeds maximum of {RequestValidator.MaxMessages}";

        // max_tokens is a required field of the Anthropic Messages API (unlike OpenAI's),
        // but it is absent from the count_tokens body — hence the flag.
        if (requireMaxTokens && !request.MaxTokens.HasValue)
            return "max_tokens: field required";
        if (request.MaxTokens.HasValue && request.MaxTokens.Value <= 0)
            return "max_tokens: must be a positive integer";

        // Roles and content kinds are checked here rather than left to the converter:
        // ToMessages passes `role` straight through to the chat template, so an unchecked
        // `system` (or arbitrary) role would let a caller inject a system turn mid-
        // conversation, and an unchecked content kind would silently flatten to an empty
        // message instead of surfacing the client's mistake as a 400.
        for (int i = 0; i < request.Messages.Length; i++)
        {
            var msg = request.Messages[i];
            if (msg is null)
                return $"messages[{i}]: must be an object";

            if (msg.Role is not ("user" or "assistant"))
                return $"messages[{i}].role: must be one of \"user\", \"assistant\"";

            if (msg.Content.ValueKind is not (JsonValueKind.String or JsonValueKind.Array))
                return $"messages[{i}].content: must be a string or an array of content blocks";

            if (msg.Content.ValueKind == JsonValueKind.Array)
            {
                string? blockError = ValidateContentBlocks(msg.Content, i);
                if (blockError is not null)
                    return blockError;
            }
        }

        // A forced tool that is not in `tools` can never be satisfied: the constraint has no
        // schema to build from, so the request would silently degrade to an ordinary completion.
        var toolChoice = AnthropicConverter.ParseToolChoice(request.ToolChoice);
        if (toolChoice is ToolChoice.Function fn &&
            (request.Tools is null || Array.FindIndex(request.Tools, t => t.Name == fn.Name) < 0))
        {
            return $"tool_choice.name: no tool named '{fn.Name}' was provided in tools";
        }

        return null;
    }

    /// <summary>
    /// Rejects content blocks this surface cannot represent.
    /// </summary>
    /// <remarks>
    /// The converter understands <c>text</c>, <c>tool_use</c> and <c>tool_result</c>, and
    /// deliberately drops <c>thinking</c>/<c>redacted_thinking</c> (they carry no prompt content
    /// dotLLM can replay). Anything else — <c>image</c>, <c>document</c>, a typo — would be
    /// silently dropped, and a dropped image means the model answers about a picture it never
    /// saw. The real API rejects an unknown block type, so dotLLM does too (#449).
    /// </remarks>
    private static string? ValidateContentBlocks(JsonElement content, int messageIndex)
    {
        int b = 0;
        foreach (var block in content.EnumerateArray())
        {
            if (block.ValueKind != JsonValueKind.Object)
                return $"messages[{messageIndex}].content[{b}]: must be an object";

            string? type = block.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String
                ? t.GetString()
                : null;
            if (type is null)
                return $"messages[{messageIndex}].content[{b}].type: field required";

            if (Array.IndexOf(SupportedContentBlockTypes, type) < 0)
                return $"messages[{messageIndex}].content[{b}].type: unsupported content block type '{type}'";

            b++;
        }
        return null;
    }

    // Input block types this surface accepts. thinking/redacted_thinking are accepted and then
    // dropped, so a client replaying an extended-thinking transcript is not rejected.
    private static readonly string[] SupportedContentBlockTypes =
        ["text", "tool_use", "tool_result", "thinking", "redacted_thinking"];

    /// <summary>
    /// Builds the prompt for an Anthropic request: message flattening, tool definitions and the
    /// chat template. Shared by <c>/v1/messages</c> and <c>/v1/messages/count_tokens</c> so the
    /// count the latter reports cannot drift from the prompt the former actually runs.
    /// </summary>
    private static string BuildPrompt(
        AnthropicMessagesRequest request, ServerState state, out ToolDefinition[]? tools)
    {
        var messages = AnthropicConverter.ToMessages(request);
        tools = AnthropicConverter.ToTools(request.Tools);
        return state.ChatTemplate!.Apply(messages, new ChatTemplateOptions
        {
            AddGenerationPrompt = true,
            Tools = tools,
        });
    }

    /// <summary>
    /// Handles <c>POST /v1/messages/count_tokens</c>: the number of input tokens the same body
    /// would consume on <c>POST /v1/messages</c>, without generating anything.
    /// </summary>
    /// <remarks>
    /// The count is <c>ITokenizer.CountTokens</c> over the templated prompt — the same value
    /// <see cref="RequestValidator.ValidatePromptLength"/> computes for the generating route, so
    /// <c>count_tokens(body).input_tokens</c> matches <c>messages.create(body).usage.input_tokens</c>.
    /// Unlike the generating route this needs only the tokenizer and the template, so the
    /// diffusion-model refusal does not apply.
    /// </remarks>
    internal static async Task HandleCountTokensAsync(
        AnthropicMessagesRequest request,
        ServerState state,
        HttpContext httpContext)
    {
        var headerError = AnthropicHeaders.Validate(httpContext.Request.Headers);
        if (headerError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", headerError);
            return;
        }

        var activationError = await state.EnsureActiveAsync(
            request.Model, keepAliveOverride: null, httpContext.RequestAborted);
        if (activationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", activationError);
            return;
        }

        if (!state.IsReady || state.ChatTemplate is null || state.Tokenizer is null)
        {
            await WriteErrorAsync(httpContext, 503, "api_error", "No model loaded");
            return;
        }

        var validationError = ValidateRequest(request, requireMaxTokens: false);
        if (validationError is not null)
        {
            await WriteErrorAsync(httpContext, 400, "invalid_request_error", validationError);
            return;
        }

        string prompt = BuildPrompt(request, state, out _);
        var response = new AnthropicCountTokensResponse { InputTokens = state.Tokenizer.CountTokens(prompt) };

        httpContext.Response.ContentType = "application/json";
        await JsonSerializer.SerializeAsync(httpContext.Response.Body, response,
            ServerJsonContext.Default.AnthropicCountTokensResponse, httpContext.RequestAborted);
    }

    /// <summary>
    /// Returns the request stop sequence that is a suffix of <paramref name="text"/>,
    /// or null if none matches.
    /// </summary>
    private static string? MatchStopSequence(string text, string[]? stopSequences)
    {
        if (stopSequences is null)
            return null;
        foreach (var seq in stopSequences)
        {
            if (!string.IsNullOrEmpty(seq) && text.EndsWith(seq, StringComparison.Ordinal))
                return seq;
        }
        return null;
    }

    /// <summary>
    /// Strips a trailing stop-sequence suffix from <paramref name="text"/> and reports
    /// whether a caller-supplied stop sequence matched.
    /// </summary>
    private static string StripAndDetectStopSequence(
        string text, string[]? requestStops, IReadOnlyList<string> allStops,
        out bool matchedRequestStop, out string? matched, string? engineMatch = null)
    {
        matchedRequestStop = false;
        matched = null;

        // The engine reports the stop string it matched (#459), and has already trimmed it out of
        // the text — so testing the text cannot find it, and this is the only reliable source.
        // The text-based passes below remain for callers that do not report one.
        if (engineMatch is not null)
        {
            matched = engineMatch;
            matchedRequestStop = requestStops is not null
                && Array.IndexOf(requestStops, engineMatch) >= 0;
            return text;
        }

        // Caller-supplied stop sequences are reported as "stop_sequence".
        string? requestMatch = MatchStopSequence(text, requestStops);
        if (requestMatch is not null)
        {
            matchedRequestStop = true;
            matched = requestMatch;
            return text[..^requestMatch.Length];
        }

        // Built-in/template stop sequences are stripped but reported as "end_turn".
        foreach (var seq in allStops)
        {
            if (text.EndsWith(seq, StringComparison.Ordinal))
            {
                matched = seq;
                return text[..^seq.Length];
            }
        }
        return text;
    }

    private static async Task WriteEventAsync<T>(
        HttpContext ctx, string eventName, T payload,
        System.Text.Json.Serialization.Metadata.JsonTypeInfo<T> typeInfo, CancellationToken ct)
    {
        await ctx.Response.WriteAsync($"event: {eventName}\n", ct);
        await ctx.Response.WriteAsync("data: ", ct);
        await JsonSerializer.SerializeAsync(ctx.Response.Body, payload, typeInfo, ct);
        await ctx.Response.WriteAsync("\n\n", ct);
        await ctx.Response.Body.FlushAsync(ct);
    }

    /// <summary>
    /// Writes an Anthropic error envelope (<c>{"type":"error","error":{...}}</c>) with the given
    /// HTTP status. Every failure path on this surface goes through here: an Anthropic SDK client
    /// parses this envelope and would throw on the OpenAI surface's <c>{"error":"..."}</c> shape.
    /// </summary>
    internal static async Task WriteErrorAsync(
        HttpContext ctx, int statusCode, string errorType, string message)
    {
        ctx.Response.StatusCode = statusCode;
        await ctx.Response.WriteAsJsonAsync(
            new AnthropicErrorResponse { Error = new AnthropicErrorBody { Type = errorType, Message = message } },
            ServerJsonContext.Default.AnthropicErrorResponse,
            contentType: null,
            ctx.RequestAborted);
    }
}
