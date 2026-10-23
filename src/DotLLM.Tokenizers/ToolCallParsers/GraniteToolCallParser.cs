using System.Text.Json;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses IBM Granite 3.x tool calls (#796): the model answers a tool request with the
/// <c>&lt;|tool_call|&gt;</c> special token followed by a JSON list of calls,
/// <c>&lt;|tool_call|&gt;[{"name": "get_weather", "arguments": {"city": "Paris"}}]</c>.
/// </summary>
/// <remarks>
/// <para>Mirrors llama.cpp's <c>common_chat_params_init_granite</c> (a <c>&lt;|tool_call|&gt;</c> prefix then a JSON
/// array; the turn then ends with <c>&lt;|end_of_text|&gt;</c>). Before this parser existed Granite fell through to
/// <see cref="GenericToolCallParser"/>: the call itself parsed (the JSON is bare), but the server's streaming
/// suppressor is deliberately off for the heuristic generic parser, so <c>&lt;|tool_call|&gt;</c> and the raw JSON
/// leaked into <c>delta.content</c> as well as arriving as <c>tool_calls</c>. A marker-based parser lets the
/// suppressor hold the call back from the moment the marker appears.</para>
/// <para>Granite-3.3-2b, greedy, regularly omits the <c>name</c> key when exactly one tool was offered
/// (<c>&lt;|tool_call|&gt;[{"arguments": {...}}]</c>); with the request's tools known and exactly one declared, such a
/// call is attributed to that tool rather than lost. With several tools a nameless call is ambiguous and is not
/// reported.</para>
/// </remarks>
public sealed class GraniteToolCallParser : IToolCallParser
{
    private const string Marker = "<|tool_call|>";

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText)
    {
        int m = generatedText.IndexOf(Marker, StringComparison.Ordinal);
        if (m < 0)
            return null;

        // The model may repeat the marker per call (<|tool_call|>{...}<|tool_call|>{...}); every JSON value
        // after the first marker counts, in order.
        return ToolCallJsonHelper.ParseAll(AfterMarker(generatedText, m), requireWholeText: false);
    }

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText, IReadOnlyList<ToolDefinition>? tools)
    {
        var calls = TryParse(generatedText);
        if (calls is null && tools is { Count: 1 })
            calls = TryParseNameless(generatedText, tools[0].Name);
        return calls is null ? null : ToolArgumentCoercer.Coerce(calls, tools);
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
        => text.Contains(Marker, StringComparison.Ordinal);

    private static string AfterMarker(string text, int markerIndex)
        => text[(markerIndex + Marker.Length)..].Replace(Marker, " ", StringComparison.Ordinal);

    /// <summary>Calls shaped <c>{"arguments": {...}}</c> / <c>{"parameters": {...}}</c> with no name, attributed to the only tool.</summary>
    private static ToolCall[]? TryParseNameless(string text, string toolName)
    {
        int m = text.IndexOf(Marker, StringComparison.Ordinal);
        if (m < 0)
            return null;

        string after = AfterMarker(text, m);
        var calls = new List<ToolCall>();
        int i = 0;
        while (i < after.Length)
        {
            if (after[i] is not ('{' or '['))
            {
                i++;
                continue;
            }

            string candidate = ToolCallJsonHelper.ExtractBalancedJson(after, i);
            if (candidate.Length == 0)
                return null;                       // truncated generation: never report half a call
            i += candidate.Length;

            try
            {
                using var doc = JsonDocument.Parse(candidate);
                var items = doc.RootElement.ValueKind == JsonValueKind.Array
                    ? doc.RootElement.EnumerateArray().ToArray()
                    : [doc.RootElement];
                foreach (var item in items)
                {
                    if (item.ValueKind != JsonValueKind.Object)
                        continue;
                    if (item.TryGetProperty("name", out _))
                        return null;               // a named call is the normal path's business, not ours
                    if (item.TryGetProperty("arguments", out var a) || item.TryGetProperty("parameters", out a))
                        calls.Add(new ToolCall($"call_{calls.Count}", toolName, a.GetRawText()));
                }
            }
            catch (JsonException)
            {
                return null;
            }
        }

        return calls.Count == 0 ? null : calls.ToArray();
    }
}
