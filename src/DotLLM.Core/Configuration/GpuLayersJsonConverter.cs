using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Core.Configuration;

/// <summary>
/// JSON form of a <c>gpu_layers</c> value (issue #1851): reads a number or a string (<c>"auto"</c>, <c>"all"</c>, a count),
/// validated exactly like the CLI's <c>--gpu-layers</c> (<see cref="GpuOffloadPlanner.TryParseGpuLayers"/>), so <c>-5</c> is an
/// error rather than "all layers"; writes <see cref="GpuOffloadPlanner.AutoGpuLayers"/> as <c>"auto"</c>, never the sentinel.
/// </summary>
public sealed class GpuLayersJsonConverter : JsonConverter<int?>
{
    /// <inheritdoc/>
    public override int? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        string? raw = reader.TokenType switch
        {
            JsonTokenType.Number => reader.TryGetInt64(out long n) ? n.ToString(CultureInfo.InvariantCulture) : "invalid",
            JsonTokenType.String => reader.GetString(),
            JsonTokenType.Null => null,
            _ => throw new JsonException("gpu_layers must be a number or \"auto\" / \"all\"."),
        };
        if (raw is null) return null;
        return GpuOffloadPlanner.TryParseGpuLayers(raw, out int? layers, out string? error)
            ? layers
            : throw new JsonException(error!.Replace("--gpu-layers", "gpu_layers", StringComparison.Ordinal));
    }

    /// <inheritdoc/>
    public override void Write(Utf8JsonWriter writer, int? value, JsonSerializerOptions options)
    {
        if (value is null) writer.WriteNullValue();
        else if (GpuOffloadPlanner.IsAuto(value)) writer.WriteStringValue("auto");
        else writer.WriteNumberValue(value.Value);
    }
}
