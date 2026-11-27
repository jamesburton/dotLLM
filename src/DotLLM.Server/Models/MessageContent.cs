using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DotLLM.Server.Models;

/// <summary>One part of an OpenAI multimodal message: <c>text</c> or <c>image_url</c> (issue #2035).</summary>
/// <param name="Type"><c>"text"</c> or <c>"image_url"</c>.</param>
/// <param name="Text">The text of a text part.</param>
/// <param name="ImageUrl">The <c>image_url.url</c> of an image part (a <c>data:</c> URI).</param>
/// <param name="Chunk">
/// The part's inline <c>chunk</c> annotation (chunk-cache, docs/CHUNK_CACHE.md), the raw JSON object; null when absent. Read by
/// <c>ChunkAnnotationParser</c> with the same semantics as the Anthropic inline placement. Kept raw (a cloned element) so a malformed annotation
/// is that annotation's defect, not a binding failure. Not a DTO property: this type is read by <see cref="MessageContentConverter"/>, so the
/// source-generator initializer rule (CLAUDE.md, JSON DTO Rules) does not apply.
/// </param>
public sealed record MessageContentPart(string Type, string? Text, string? ImageUrl, JsonElement? Chunk = null);

/// <summary>
/// OpenAI message content: a plain string, or an array of content parts (text and images). Reads either; <see cref="Text"/> is the string form
/// (the concatenated text parts), so code that only handles text keeps working through the implicit conversions.
/// </summary>
[JsonConverter(typeof(MessageContentConverter))]
public sealed class MessageContent
{
    /// <summary>The plain string, or the text parts joined in order.</summary>
    public string Text { get; }

    /// <summary>The parts of an array content; null for a plain string.</summary>
    public IReadOnlyList<MessageContentPart>? Parts { get; }

    /// <summary>Plain-string content.</summary>
    public MessageContent(string text) { Text = text; }

    /// <summary>Array content.</summary>
    public MessageContent(IReadOnlyList<MessageContentPart> parts)
    {
        Parts = parts;
        var sb = new StringBuilder();
        foreach (var p in parts) if (p.Type == "text" && p.Text is not null) sb.Append(p.Text);
        Text = sb.ToString();
    }

    /// <summary>The <c>image_url</c> parts' URLs, in order.</summary>
    public IEnumerable<string> ImageUrls => Parts is null ? [] : Parts.Where(p => p.Type == "image_url" && p.ImageUrl is not null).Select(p => p.ImageUrl!);

    /// <summary>String form.</summary>
    public static implicit operator string?(MessageContent? c) => c?.Text;

    /// <summary>Plain-string content from a string.</summary>
    public static implicit operator MessageContent?(string? s) => s is null ? null : new MessageContent(s);

    /// <inheritdoc/>
    public override string ToString() => Text;
}

/// <summary>JSON converter for <see cref="MessageContent"/>: string, array of parts, or null.</summary>
public sealed class MessageContentConverter : JsonConverter<MessageContent>
{
    /// <inheritdoc/>
    public override MessageContent? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Null) return null;
        if (reader.TokenType == JsonTokenType.String) return new MessageContent(reader.GetString()!);
        if (reader.TokenType != JsonTokenType.StartArray) throw new JsonException("message content must be a string or an array of content parts");
        using var doc = JsonDocument.ParseValue(ref reader);
        var parts = new List<MessageContentPart>();
        foreach (var e in doc.RootElement.EnumerateArray())
        {
            string type = e.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString()! : "";
            switch (type)
            {
                case "text":
                    parts.Add(new MessageContentPart("text", e.TryGetProperty("text", out var tx) && tx.ValueKind == JsonValueKind.String ? tx.GetString() : "", null, ChunkOf(e)));
                    break;
                case "image_url":
                {
                    if (!e.TryGetProperty("image_url", out var iu)) throw new JsonException("image_url part without image_url");
                    string? url = iu.ValueKind == JsonValueKind.String ? iu.GetString()
                        : iu.TryGetProperty("url", out var u) && u.ValueKind == JsonValueKind.String ? u.GetString() : null;
                    if (url is null) throw new JsonException("image_url part without a url");
                    parts.Add(new MessageContentPart("image_url", null, url, ChunkOf(e)));
                    break;
                }
                default:
                    throw new JsonException($"unsupported content part type '{type}' (supported: text, image_url)");
            }
        }
        return new MessageContent(parts);
    }

    private static JsonElement? ChunkOf(JsonElement part) =>
        part.TryGetProperty("chunk", out var c) && c.ValueKind != JsonValueKind.Null ? c.Clone() : null;

    /// <inheritdoc/>
    public override void Write(Utf8JsonWriter writer, MessageContent value, JsonSerializerOptions options)
    {
        if (value.Parts is null) { writer.WriteStringValue(value.Text); return; }
        writer.WriteStartArray();
        foreach (var p in value.Parts)
        {
            writer.WriteStartObject();
            writer.WriteString("type", p.Type);
            if (p.Type == "text") writer.WriteString("text", p.Text);
            else { writer.WriteStartObject("image_url"); writer.WriteString("url", p.ImageUrl); writer.WriteEndObject(); }
            if (p.Chunk is { } chunk) { writer.WritePropertyName("chunk"); chunk.WriteTo(writer); }
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }
}
