using System.Text.Json.Serialization;

namespace DotLLM.Tray.Api;

/// <summary>
/// Source-generated serialization context for the tray's client-side DTOs.
/// </summary>
/// <remarks>
/// Source-generated rather than reflection-based so the tray stays trim-friendly and the
/// single-file publish carries no reflection surprise. Every DTO the client touches must be
/// registered here; a missing registration compiles fine and throws only at runtime, which is
/// exactly the failure mode #454's own test notes call out.
/// </remarks>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(TrayModelList))]
[JsonSerializable(typeof(TrayModelInfo))]
[JsonSerializable(typeof(TrayAvailableModelList))]
[JsonSerializable(typeof(TrayAvailableModel))]
[JsonSerializable(typeof(TraySettingsDto))]
[JsonSerializable(typeof(TraySettingsUpdate))]
[JsonSerializable(typeof(TraySettingsUpdateResult))]
[JsonSerializable(typeof(TrayUnloadRequest))]
[JsonSerializable(typeof(TrayUnloadResult))]
[JsonSerializable(typeof(TrayEnableRequest))]
[JsonSerializable(typeof(TrayEnableResult))]
[JsonSerializable(typeof(TrayLoadRequest))]
[JsonSerializable(typeof(TrayLoadResult))]
[JsonSerializable(typeof(TrayDeviceList))]
[JsonSerializable(typeof(TrayBackendInfo))]
[JsonSerializable(typeof(TrayDeviceInfo))]
[JsonSerializable(typeof(TrayPullRequest))]
[JsonSerializable(typeof(TrayPullJob))]
[JsonSerializable(typeof(TrayPullJobList))]
[JsonSerializable(typeof(TrayStatusResponse))]
[JsonSerializable(typeof(TrayErrorResponse))]
[JsonSerializable(typeof(TrayErrorDetail))]
public sealed partial class TrayJsonContext : JsonSerializerContext;
