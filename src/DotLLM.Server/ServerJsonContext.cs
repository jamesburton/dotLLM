using System.Text.Json.Serialization;
using DotLLM.Server.Models;

namespace DotLLM.Server;

/// <summary>
/// Source-generated JSON serializer context for all server DTOs.
/// Eliminates reflection overhead and enables AOT compilation.
/// </summary>
[JsonSerializable(typeof(ChatCompletionRequest))]
[JsonSerializable(typeof(ChatCompletionResponse))]
[JsonSerializable(typeof(ChatCompletionChunk))]
[JsonSerializable(typeof(CompletionRequest))]
[JsonSerializable(typeof(CompletionResponse))]
[JsonSerializable(typeof(CompletionChunk))]
[JsonSerializable(typeof(TokenizeRequest))]
[JsonSerializable(typeof(TokenizeResponse))]
[JsonSerializable(typeof(DetokenizeRequest))]
[JsonSerializable(typeof(DetokenizeResponse))]
[JsonSerializable(typeof(ModelListResponse))]
[JsonSerializable(typeof(ModelInfoDto))]
[JsonSerializable(typeof(StreamOptionsDto))]
[JsonSerializable(typeof(PropsResponse))]
[JsonSerializable(typeof(SamplingDefaultsDto))]
[JsonSerializable(typeof(AvailableModelsResponse))]
[JsonSerializable(typeof(ModelLoadRequest))]
[JsonSerializable(typeof(ModelLoadResponse))]
[JsonSerializable(typeof(TimingsDto))]
[JsonSerializable(typeof(LogprobsDto))]
[JsonSerializable(typeof(ModelInspectResponse))]
[JsonSerializable(typeof(ErrorResponse))]
[JsonSerializable(typeof(ErrorDetail))]
[JsonSerializable(typeof(StatusResponse))]
[JsonSerializable(typeof(LoraLoadRequest))]
[JsonSerializable(typeof(LoraLoadResponse))]
[JsonSerializable(typeof(LoraListResponse))]
[JsonSerializable(typeof(PromptCacheRegisterRequest))]
[JsonSerializable(typeof(PromptCacheResponse))]
[JsonSerializable(typeof(PromptCacheStatsResponse))]
[JsonSerializable(typeof(SettingsDto))]
[JsonSerializable(typeof(SettingsUpdateRequest))]
[JsonSerializable(typeof(SettingsUpdateResponse))]
[JsonSerializable(typeof(ModelUnloadRequest))]
[JsonSerializable(typeof(ModelUnloadResponse))]
[JsonSerializable(typeof(ModelEnableRequest))]
[JsonSerializable(typeof(ModelEnableResponse))]
[JsonSerializable(typeof(DeviceListResponse))]
[JsonSerializable(typeof(BackendInfoDto))]
[JsonSerializable(typeof(DeviceInfoDto))]
[JsonSerializable(typeof(ModelPullRequest))]
[JsonSerializable(typeof(ModelPullJobDto))]
[JsonSerializable(typeof(ModelPullJobListResponse))]
// Anthropic Messages API (#448) — fork-only surface. The request type is listed so
// minimal-API body binding resolves it through the source-gen context (AOT-clean),
// same as ChatCompletionRequest.
[JsonSerializable(typeof(AnthropicMessagesRequest))]
[JsonSerializable(typeof(AnthropicMessageResponse))]
[JsonSerializable(typeof(AnthropicErrorResponse))]
[JsonSerializable(typeof(AnthropicMessageStartEvent))]
[JsonSerializable(typeof(AnthropicContentBlockStartEvent))]
[JsonSerializable(typeof(AnthropicContentBlockDeltaEvent))]
[JsonSerializable(typeof(AnthropicContentBlockStopEvent))]
[JsonSerializable(typeof(AnthropicMessageDeltaEvent))]
[JsonSerializable(typeof(AnthropicMessageStopEvent))]
[JsonSerializable(typeof(AnthropicPingEvent))]
[JsonSourceGenerationOptions(
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
internal partial class ServerJsonContext : JsonSerializerContext;
