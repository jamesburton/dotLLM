using System.Text.Json.Serialization;

namespace DotLLM.Server.Tuning;

/// <summary>Source-generated JSON for the tuning store. Compact and null-omitting: this exact form is what the checksum covers.</summary>
[JsonSourceGenerationOptions(WriteIndented = false, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(TuningResult))]
[JsonSerializable(typeof(TuningKey))]
[JsonSerializable(typeof(TunedSettings))]
[JsonSerializable(typeof(DeviceFingerprint))]
[JsonSerializable(typeof(PerfIdentity))]
internal sealed partial class TuningJsonContext : JsonSerializerContext
{
}
