using System.Text.Json;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Shared JSON parsing logic for tool call extraction.
/// Handles both single-object and array formats, normalizes key names.
/// </summary>
internal static class ToolCallJsonHelper
{
    /// <summary>
    /// Parses a JSON string containing one or more tool calls.
    /// Accepts a single object (<c>{"name": "...", "arguments": {...}}</c>)
    /// or an array of such objects.
    /// </summary>
    /// <param name="json">Raw JSON string (may include leading/trailing whitespace).</param>
    /// <param name="idPrefix">Prefix for generated call IDs (e.g., "call").</param>
    /// <returns>Parsed tool calls, or null if the JSON is invalid or missing required fields.</returns>
    public static ToolCall[]? ParseToolCallJson(string json, string idPrefix = "call")
    {
        json = json.Trim();
        if (json.Length == 0)
            return null;

        // Try strict JSON first
        var result = TryParseStrict(json, idPrefix);
        if (result is not null)
            return result;

        // Fallback: fix common model quirks (unquoted keys, extra braces)
        string normalized = FixJavascriptStyleJson(json);
        if (normalized != json)
            return TryParseStrict(normalized, idPrefix);

        return null;
    }

    private static ToolCall[]? TryParseStrict(string json, string idPrefix)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            return root.ValueKind switch
            {
                JsonValueKind.Array => ParseArray(root, idPrefix),
                JsonValueKind.Object => ParseSingle(root, idPrefix, 0) is { } tc ? [tc] : null,
                _ => null
            };
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>
    /// Fixes common non-standard JSON patterns emitted by models:
    /// unquoted keys (<c>{name:</c> → <c>{"name":</c>), extra trailing braces.
    /// </summary>
    private static string FixJavascriptStyleJson(string json)
    {
        // Quote unquoted keys: {name: or ,name: → {"name": or ,"name":
        string result = Regex.Replace(json, @"(?<=[\{,])\s*(\w+)\s*:", " \"$1\":");

        // Strip doubled outer braces: {{...}} → {...}
        while (result.StartsWith("{{") && result.EndsWith("}}"))
            result = result[1..^1];

        return result;
    }

    /// <summary>
    /// Attempts to extract the first valid JSON object or array from a text string.
    /// Scans for '{' or '[' and tries to parse from that position.
    /// </summary>
    /// <param name="text">Text that may contain embedded JSON.</param>
    /// <param name="idPrefix">Prefix for generated call IDs.</param>
    /// <returns>Parsed tool calls, or null if no valid JSON found.</returns>
    public static ToolCall[]? ExtractAndParse(string text, string idPrefix = "call")
    {
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c != '{' && c != '[')
                continue;

            // Find matching close bracket
            string candidate = ExtractBalancedJson(text, i);
            if (candidate.Length == 0)
                continue;

            var result = ParseToolCallJson(candidate, idPrefix);
            if (result is { Length: > 0 })
                return result;
        }

        return null;
    }

    /// <summary>
    /// Collects every top-level JSON tool call in <paramref name="text"/> — one or several objects/arrays
    /// separated by whitespace, <c>;</c> or <c>,</c> (Llama 3.x emits parallel calls as
    /// <c>{...}; {...}</c> as well as a JSON array). Call ids are renumbered sequentially.
    /// </summary>
    /// <param name="text">Text following the tool-call marker.</param>
    /// <param name="requireWholeText">When true, anything other than separators between/after the values
    /// makes the result null (used when there is no marker, so prose that quotes JSON is not a call).</param>
    public static ToolCall[]? ParseAll(string text, bool requireWholeText)
    {
        var calls = new List<ToolCall>();
        int i = 0;
        while (i < text.Length)
        {
            char c = text[i];
            if (c is '{' or '[')
            {
                string candidate = ExtractBalancedJson(text, i);
                if (candidate.Length == 0)
                {
                    if (requireWholeText) return null;
                    i++;
                    continue;
                }

                var parsed = ParseToolCallJson(candidate, "call");
                if (parsed is { Length: > 0 })
                    calls.AddRange(parsed);
                else if (requireWholeText)
                    return null;
                i += candidate.Length;
            }
            else if (char.IsWhiteSpace(c) || c is ';' or ',')
            {
                i++;
            }
            else
            {
                if (requireWholeText) return null;
                i++;
            }
        }

        if (calls.Count == 0)
            return null;
        for (int k = 0; k < calls.Count; k++)
            if (calls[k].Id.StartsWith("call_", StringComparison.Ordinal))
                calls[k] = calls[k] with { Id = $"call_{k}" };
        return calls.ToArray();
    }

    private static ToolCall[]? ParseArray(JsonElement array, string idPrefix)
    {
        var calls = new List<ToolCall>();
        int index = 0;

        foreach (var element in array.EnumerateArray())
        {
            if (element.ValueKind != JsonValueKind.Object)
                continue;

            var call = ParseSingle(element, idPrefix, index);
            if (call is not null)
            {
                calls.Add(call);
                index++;
            }
        }

        return calls.Count > 0 ? calls.ToArray() : null;
    }

    private static ToolCall? ParseSingle(JsonElement obj, string idPrefix, int index)
    {
        // Extract function name. Canonical: {"name": "f", ...}. Seen from small Llama models:
        //   {"type":"function","function":"f","parameters":{...}}          (function = the name)
        //   {"type":"function","function":{"name":"f","arguments":{...}}}  (OpenAI envelope)
        JsonElement argsHolder = obj;
        string? functionName = null;
        if (obj.TryGetProperty("name", out var nameProp) && nameProp.ValueKind == JsonValueKind.String)
        {
            functionName = nameProp.GetString();
        }
        else if (obj.TryGetProperty("function", out var fnProp))
        {
            if (fnProp.ValueKind == JsonValueKind.String)
            {
                functionName = fnProp.GetString();
            }
            else if (fnProp.ValueKind == JsonValueKind.Object
                     && fnProp.TryGetProperty("name", out var innerName) && innerName.ValueKind == JsonValueKind.String)
            {
                functionName = innerName.GetString();
                // Arguments live inside the envelope unless the outer object carries them.
                if (!obj.TryGetProperty("arguments", out _) && !obj.TryGetProperty("parameters", out _))
                    argsHolder = fnProp;
            }
        }

        if (string.IsNullOrEmpty(functionName))
            return null;

        // Extract arguments — try "arguments" first, fall back to "parameters" (Llama convention)
        string arguments = "{}";
        if (argsHolder.TryGetProperty("arguments", out var argsProp))
        {
            arguments = SerializeValue(argsProp);
        }
        else if (argsHolder.TryGetProperty("parameters", out var paramsProp))
        {
            arguments = SerializeValue(paramsProp);
        }

        // Extract or generate ID
        string id = obj.TryGetProperty("id", out var idProp) && idProp.ValueKind == JsonValueKind.String
            ? idProp.GetString()!
            : $"{idPrefix}_{index}";

        return new ToolCall(id, functionName, arguments);
    }

    private static string SerializeValue(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.String)
        {
            // Some models emit arguments as a JSON string (double-serialized)
            string str = element.GetString()!;
            // Check if it's a JSON object string
            if (str.TrimStart().StartsWith('{'))
                return str;
            return element.GetRawText();
        }

        return element.GetRawText();
    }

    internal static string ExtractBalancedJson(string text, int start)
    {
        char open = text[start];
        char close = open == '{' ? '}' : ']';
        int depth = 0;
        bool inString = false;
        bool escaped = false;

        for (int i = start; i < text.Length; i++)
        {
            char c = text[i];

            if (escaped)
            {
                escaped = false;
                continue;
            }

            if (c == '\\' && inString)
            {
                escaped = true;
                continue;
            }

            if (c == '"')
            {
                inString = !inString;
                continue;
            }

            if (inString)
                continue;

            if (c == open)
                depth++;
            else if (c == close)
            {
                depth--;
                if (depth == 0)
                    return text[start..(i + 1)];
            }
        }

        return string.Empty; // unbalanced
    }
}
