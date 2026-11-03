// SPDX-License-Identifier: MIT
using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace DotLLM.ChunkCache.Client;

/// <summary>
/// The response's <c>chunk_cache</c> report (spec 5.9) in MEAI types: <see cref="ChunkAnnotatingChatClient"/> puts it in
/// <c>AdditionalProperties["chunk_cache"]</c> of the <see cref="ChatResponse"/> (blocking) or of one
/// <see cref="ChatResponseUpdate"/> (streaming; <c>ToChatResponse</c> carries it to the message it builds, where
/// <see cref="TryGet(ChatResponse, out JsonObject?)"/> also looks).
/// </summary>
/// <remarks>The report is the server's JSON object as is (a <see cref="JsonObject"/>), so additive server fields survive.</remarks>
public static class ChunkCacheReport
{
    /// <summary>The key in <c>AdditionalProperties</c> and the JSON property name.</summary>
    public const string PropertyName = "chunk_cache";

    /// <summary>
    /// Reads the report from a response: its own <c>AdditionalProperties</c>, else those of its messages, last first
    /// (MEAI's <c>ToChatResponse</c> moves an update's <c>AdditionalProperties</c> onto the message it builds).
    /// </summary>
    public static bool TryGet(ChatResponse response, [NotNullWhen(true)] out JsonObject? report)
    {
        ArgumentNullException.ThrowIfNull(response);
        if (TryGet(response.AdditionalProperties, out report))
            return true;
        for (int i = response.Messages.Count - 1; i >= 0; i--)
        {
            if (TryGet(response.Messages[i].AdditionalProperties, out report))
                return true;
        }

        return false;
    }

    /// <summary>Reads the report from a streaming update.</summary>
    public static bool TryGet(ChatResponseUpdate update, [NotNullWhen(true)] out JsonObject? report)
    {
        ArgumentNullException.ThrowIfNull(update);
        return TryGet(update.AdditionalProperties, out report);
    }

    /// <summary>Attaches <paramref name="report"/> to <paramref name="response"/> (no-op for <see langword="null"/>).</summary>
    public static void Attach(ChatResponse response, JsonObject? report)
    {
        ArgumentNullException.ThrowIfNull(response);
        if (report is null) return;
        (response.AdditionalProperties ??= [])[PropertyName] = report;
    }

    internal static void Attach(ChatResponseUpdate update, JsonObject report) =>
        (update.AdditionalProperties ??= [])[PropertyName] = report;

    /// <summary>
    /// The top-level <c>chunk_cache</c> object of a JSON document (a blocking body or one SSE event's data), or
    /// <see langword="null"/> when there is none or the bytes are not a JSON object.
    /// </summary>
    internal static JsonObject? FromJson(ReadOnlySpan<byte> json)
    {
        try
        {
            var reader = new Utf8JsonReader(json, new JsonReaderOptions { MaxDepth = 64 });
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject) return null;
            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                bool match = reader.ValueTextEquals(PropertyName);
                if (!reader.Read()) return null;
                if (match)
                    return reader.TokenType == JsonTokenType.StartObject ? JsonNode.Parse(ref reader) as JsonObject : null;
                reader.Skip();
            }
        }
        catch (JsonException)
        {
        }

        return null;
    }

    private static bool TryGet(AdditionalPropertiesDictionary? properties, [NotNullWhen(true)] out JsonObject? report)
    {
        report = null;
        if (properties is null || !properties.TryGetValue(PropertyName, out object? value)) return false;
        report = value switch
        {
            JsonObject o => o,
            JsonElement { ValueKind: JsonValueKind.Object } e => JsonObject.Create(e),
            _ => null,
        };
        return report is not null;
    }
}
