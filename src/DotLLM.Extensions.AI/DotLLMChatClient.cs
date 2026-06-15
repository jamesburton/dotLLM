using System.Runtime.CompilerServices;
using System.Text;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using EngineToolCall = DotLLM.Tokenizers.ToolCall;

namespace DotLLM.Extensions.AI;

/// <summary>
/// A <see cref="IChatClient"/> implementation backed by the dotLLM inference engine.
/// This makes dotLLM a native, in-process backend for the Microsoft Agent Framework
/// and the broader <c>Microsoft.Extensions.AI</c> ecosystem: wrap a model and pass
/// the client to <c>chatClient.CreateAIAgent(...)</c>.
/// </summary>
/// <remarks>
/// The underlying <see cref="TextGenerator"/> is stateful and single-request; calls
/// are serialized through an internal gate (matching the dotLLM server). Streaming a
/// response holds the gate for the duration of the enumeration.
/// </remarks>
public sealed class DotLLMChatClient : IChatClient
{
    private readonly TextGenerator _generator;
    private readonly IChatTemplate _chatTemplate;
    private readonly IToolCallParser? _toolCallParser;
    private readonly string _modelId;
    private readonly ChatClientMetadata _metadata;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Creates a chat client over a loaded dotLLM model.</summary>
    /// <param name="generator">The engine text generator wired to the loaded model.</param>
    /// <param name="chatTemplate">The model's chat template (renders messages to a prompt).</param>
    /// <param name="modelId">Model id reported in responses and metadata.</param>
    /// <param name="toolCallParser">
    /// Optional model-specific tool-call parser. When supplied and the request carries
    /// tools, generated tool calls are surfaced as <see cref="FunctionCallContent"/>.
    /// </param>
    public DotLLMChatClient(
        TextGenerator generator,
        IChatTemplate chatTemplate,
        string modelId = "dotllm",
        IToolCallParser? toolCallParser = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        ArgumentNullException.ThrowIfNull(chatTemplate);
        _generator = generator;
        _chatTemplate = chatTemplate;
        _modelId = modelId;
        _toolCallParser = toolCallParser;
        _metadata = new ChatClientMetadata("dotLLM", providerUri: null, defaultModelId: modelId);
    }

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (prompt, inferenceOptions, tools) = Prepare(messages, options);

        InferenceResponse result;
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            result = await Task.Run(
                () => _generator.Generate(prompt, inferenceOptions), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }

        string text = result.Text;
        EngineToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;
        if (_toolCallParser is not null && tools is { Length: > 0 })
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, _toolCallParser);
            text = enriched.Text;
            toolCalls = enriched.ToolCalls;
            finishReason = enriched.FinishReason;
        }

        var responseMessage = new ChatMessage(
            ChatRole.Assistant, ChatClientMapping.ToResponseContents(text, toolCalls));

        return new ChatResponse(responseMessage)
        {
            ResponseId = Guid.NewGuid().ToString("N"),
            ModelId = options?.ModelId ?? _modelId,
            FinishReason = ChatClientMapping.ToChatFinishReason(finishReason),
            Usage = ChatClientMapping.ToUsageDetails(result.PromptTokenCount, result.GeneratedTokenCount),
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (prompt, inferenceOptions, tools) = Prepare(messages, options);

        string responseId = Guid.NewGuid().ToString("N");
        string modelId = options?.ModelId ?? _modelId;

        var accumulated = new StringBuilder();
        var finishReason = FinishReason.Length;

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var token in _generator
                .GenerateStreamingTokensAsync(prompt, inferenceOptions, cancellationToken).ConfigureAwait(false))
            {
                if (token.Text.Length > 0)
                {
                    accumulated.Append(token.Text);
                    yield return new ChatResponseUpdate
                    {
                        Role = ChatRole.Assistant,
                        Contents = [new TextContent(token.Text)],
                        ResponseId = responseId,
                        MessageId = responseId,
                        ModelId = modelId,
                    };
                }

                if (token.FinishReason.HasValue)
                    finishReason = token.FinishReason.Value;
            }
        }
        finally
        {
            _gate.Release();
        }

        // Post-generation tool-call detection (mirrors the server streaming endpoints).
        EngineToolCall[]? toolCalls = null;
        if (_toolCallParser is not null && tools is { Length: > 0 })
        {
            toolCalls = _toolCallParser.TryParse(accumulated.ToString());
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        var finalContents = toolCalls is { Length: > 0 }
            ? ChatClientMapping.ToResponseContents("", toolCalls)
            : [];

        yield return new ChatResponseUpdate
        {
            Role = ChatRole.Assistant,
            Contents = finalContents,
            ResponseId = responseId,
            MessageId = responseId,
            ModelId = modelId,
            FinishReason = ChatClientMapping.ToChatFinishReason(finishReason),
        };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
            return null;
        if (serviceType == typeof(ChatClientMetadata))
            return _metadata;
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <inheritdoc/>
    public void Dispose() => _gate.Dispose();

    private (string Prompt, DotLLM.Core.Configuration.InferenceOptions Options, ToolDefinition[]? Tools) Prepare(
        IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        var engineMessages = ChatClientMapping.ToEngineMessages(messages);
        var tools = ChatClientMapping.ToToolDefinitions(options);
        var inferenceOptions = ChatClientMapping.ToInferenceOptions(options);
        string prompt = _chatTemplate.Apply(
            engineMessages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });
        return (prompt, inferenceOptions, tools);
    }
}
