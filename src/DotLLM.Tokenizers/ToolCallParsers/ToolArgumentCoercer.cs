using System.Buffers;
using System.Globalization;
using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Coerces parsed tool-call arguments to the types the tool's JSON Schema declares.
/// </summary>
/// <remarks>
/// Models emit loosely typed arguments: the XML family writes every value as text, Gemma writes
/// bare numbers where a schema says <c>string</c>, small Llama models quote numbers. The
/// declared type is the authority, so <c>"17"</c> becomes <c>17</c> for an <c>integer</c> parameter and
/// <c>12345</c> becomes <c>"12345"</c> for a <c>string</c> one. Anything that cannot be coerced is
/// left exactly as the model wrote it; this never throws and never drops an argument.
/// </remarks>
public static class ToolArgumentCoercer
{
    /// <summary>
    /// Returns <paramref name="calls"/> with each call's arguments coerced to its tool's parameter schema.
    /// Calls whose tool is unknown (or that carry no schema) are returned unchanged.
    /// </summary>
    /// <param name="calls">Parsed tool calls.</param>
    /// <param name="tools">The request's tool definitions; null or empty means no coercion.</param>
    public static ToolCall[] Coerce(ToolCall[] calls, IReadOnlyList<ToolDefinition>? tools)
    {
        if (tools is not { Count: > 0 })
            return calls;

        ToolCall[]? result = null;
        for (int i = 0; i < calls.Length; i++)
        {
            string coerced = CoerceArguments(calls[i].Arguments, FindSchema(tools, calls[i].FunctionName));
            if (!ReferenceEquals(coerced, calls[i].Arguments) && coerced != calls[i].Arguments)
            {
                result ??= (ToolCall[])calls.Clone();
                result[i] = calls[i] with { Arguments = coerced };
            }
        }

        return result ?? calls;
    }

    /// <summary>Finds the tool named <paramref name="name"/> in <paramref name="tools"/>.</summary>
    internal static ToolDefinition? Find(IReadOnlyList<ToolDefinition>? tools, string name)
    {
        if (tools is null)
            return null;
        for (int i = 0; i < tools.Count; i++)
            if (string.Equals(tools[i].Name, name, StringComparison.Ordinal))
                return tools[i];
        return null;
    }

    private static string? FindSchema(IReadOnlyList<ToolDefinition> tools, string name)
        => Find(tools, name)?.ParametersSchema;

    /// <summary>Coerces a JSON arguments object against a JSON Schema string; returns the input on any problem.</summary>
    internal static string CoerceArguments(string argumentsJson, string? schemaJson)
    {
        if (string.IsNullOrWhiteSpace(schemaJson) || string.IsNullOrWhiteSpace(argumentsJson))
            return argumentsJson;

        try
        {
            using var schemaDoc = JsonDocument.Parse(schemaJson);
            using var argsDoc = JsonDocument.Parse(argumentsJson);
            if (argsDoc.RootElement.ValueKind != JsonValueKind.Object)
                return argumentsJson;

            var buffer = new ArrayBufferWriter<byte>();
            using (var w = new Utf8JsonWriter(buffer))
                WriteCoerced(w, argsDoc.RootElement, schemaDoc.RootElement);
            return System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan);
        }
        catch (JsonException)
        {
            return argumentsJson;
        }
    }

    /// <summary>
    /// Writes <paramref name="value"/> to <paramref name="w"/> coerced to <paramref name="schema"/>.
    /// </summary>
    internal static void WriteCoerced(Utf8JsonWriter w, JsonElement value, JsonElement? schema)
    {
        if (schema is not { ValueKind: JsonValueKind.Object } s)
        {
            value.WriteTo(w);
            return;
        }

        var kinds = KindsOf(s);

        switch (value.ValueKind)
        {
            case JsonValueKind.String:
                WriteFromText(w, value.GetString()!, s, kinds, value);
                return;

            case JsonValueKind.Number:
                if (kinds.Has(Kind.String) && !kinds.Has(Kind.Number) && !kinds.Has(Kind.Integer))
                    w.WriteStringValue(value.GetRawText());
                else if (kinds.Has(Kind.Integer) && !kinds.Has(Kind.Number) && value.TryGetDouble(out double d)
                         && d == Math.Floor(d) && Math.Abs(d) < 9e15 && !value.TryGetInt64(out _))
                    w.WriteNumberValue((long)d); // 3.0 for an integer parameter
                else
                    value.WriteTo(w);
                return;

            case JsonValueKind.True:
            case JsonValueKind.False:
                if (kinds.Has(Kind.String) && !kinds.Has(Kind.Boolean))
                    w.WriteStringValue(value.GetRawText());
                else
                    value.WriteTo(w);
                return;

            case JsonValueKind.Object:
                if (s.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
                {
                    w.WriteStartObject();
                    foreach (var p in value.EnumerateObject())
                    {
                        w.WritePropertyName(p.Name);
                        JsonElement? sub = props.TryGetProperty(p.Name, out var ps) ? ps : null;
                        WriteCoerced(w, p.Value, sub);
                    }
                    w.WriteEndObject();
                }
                else
                {
                    value.WriteTo(w);
                }
                return;

            case JsonValueKind.Array:
                if (s.TryGetProperty("items", out var items) && items.ValueKind == JsonValueKind.Object)
                {
                    w.WriteStartArray();
                    foreach (var e in value.EnumerateArray())
                        WriteCoerced(w, e, items);
                    w.WriteEndArray();
                }
                else
                {
                    value.WriteTo(w);
                }
                return;

            default:
                value.WriteTo(w);
                return;
        }
    }

    /// <summary>
    /// Writes a raw text value (XML parameter body) typed by <paramref name="schema"/>. A declared
    /// string stays verbatim; a declared integer/number/boolean/object/array is parsed as JSON; with no
    /// usable schema the text is parsed as JSON when it is valid JSON and kept as a string otherwise
    /// (llama.cpp's heuristic).
    /// </summary>
    internal static void WriteTyped(Utf8JsonWriter w, string raw, JsonElement? schema)
    {
        if (schema is { ValueKind: JsonValueKind.Object } s)
        {
            var kinds = KindsOf(s);
            if (kinds.Has(Kind.String))
            {
                w.WriteStringValue(raw);
                return;
            }
            if (kinds.Any)
            {
                WriteFromText(w, raw, s, kinds, null);
                return;
            }
        }

        // Unknown type: JSON if it parses, else string.
        if (TryParseJson(raw.Trim(), out var doc))
        {
            using (doc)
                doc.RootElement.WriteTo(w);
        }
        else
        {
            w.WriteStringValue(raw);
        }
    }

    private static void WriteFromText(Utf8JsonWriter w, string text, JsonElement schema, Kinds kinds, JsonElement? original)
    {
        string t = text.Trim();

        if (kinds.Has(Kind.String))
        {
            w.WriteStringValue(text);
            return;
        }

        if (kinds.Has(Kind.Integer) && long.TryParse(t, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long l))
        {
            w.WriteNumberValue(l);
            return;
        }

        if ((kinds.Has(Kind.Number) || kinds.Has(Kind.Integer))
            && double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) && double.IsFinite(d))
        {
            if (kinds.Has(Kind.Number))
            {
                w.WriteNumberValue(d);
                return;
            }
            if (d == Math.Floor(d) && Math.Abs(d) < 9e15)
            {
                w.WriteNumberValue((long)d); // "3.0" for an integer parameter
                return;
            }
            // "3.5" for an integer parameter: keep what the model wrote.
        }

        if (kinds.Has(Kind.Boolean))
        {
            if (t.Equals("true", StringComparison.OrdinalIgnoreCase)) { w.WriteBooleanValue(true); return; }
            if (t.Equals("false", StringComparison.OrdinalIgnoreCase)) { w.WriteBooleanValue(false); return; }
        }

        if (kinds.Has(Kind.Null) && t.Equals("null", StringComparison.OrdinalIgnoreCase))
        {
            w.WriteNullValue();
            return;
        }

        if ((kinds.Has(Kind.Object) || kinds.Has(Kind.Array)) && TryParseJson(t, out var doc))
        {
            using (doc)
            {
                bool wantObj = kinds.Has(Kind.Object), wantArr = kinds.Has(Kind.Array);
                var vk = doc.RootElement.ValueKind;
                if ((vk == JsonValueKind.Object && wantObj) || (vk == JsonValueKind.Array && wantArr))
                {
                    WriteCoerced(w, doc.RootElement, schema);
                    return;
                }
            }
        }

        // Could not honour the declared type: leave the model's text alone.
        if (original is { } o)
            o.WriteTo(w);
        else if (TryParseJson(t, out var any))
        {
            using (any)
                any.RootElement.WriteTo(w);
        }
        else
            w.WriteStringValue(text);
    }

    private static bool TryParseJson(string text, out JsonDocument doc)
    {
        doc = null!;
        if (text.Length == 0)
            return false;
        try
        {
            doc = JsonDocument.Parse(text);
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    [Flags]
    private enum Kind
    {
        None = 0,
        String = 1,
        Integer = 2,
        Number = 4,
        Boolean = 8,
        Object = 16,
        Array = 32,
        Null = 64,
    }

    private readonly record struct Kinds(Kind Value)
    {
        public bool Any => Value != Kind.None;
        public bool Has(Kind k) => (Value & k) != 0;
    }

    private static Kinds KindsOf(JsonElement schema)
    {
        Kind k = Kind.None;
        Collect(schema, ref k, 0);
        return new Kinds(k);

        static void Collect(JsonElement s, ref Kind k, int depth)
        {
            if (s.ValueKind != JsonValueKind.Object || depth > 4)
                return;

            if (s.TryGetProperty("type", out var t))
            {
                if (t.ValueKind == JsonValueKind.String)
                    k |= Parse(t.GetString());
                else if (t.ValueKind == JsonValueKind.Array)
                    foreach (var e in t.EnumerateArray())
                        if (e.ValueKind == JsonValueKind.String)
                            k |= Parse(e.GetString());
            }

            foreach (string key in (ReadOnlySpan<string>)["anyOf", "oneOf"])
                if (s.TryGetProperty(key, out var alts) && alts.ValueKind == JsonValueKind.Array)
                    foreach (var a in alts.EnumerateArray())
                        Collect(a, ref k, depth + 1);
        }

        static Kind Parse(string? s) => s switch
        {
            "string" => Kind.String,
            "integer" => Kind.Integer,
            "number" => Kind.Number,
            "boolean" => Kind.Boolean,
            "object" => Kind.Object,
            "array" => Kind.Array,
            "null" => Kind.Null,
            _ => Kind.None,
        };
    }
}
