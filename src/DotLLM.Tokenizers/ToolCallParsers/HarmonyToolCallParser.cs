using System.Text.Json;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.ToolCallParsers;

/// <summary>
/// Parses OpenAI Harmony tool calls (gpt-oss, #798):
/// <c>&lt;|channel|&gt;commentary to=functions.NAME &lt;|constrain|&gt;json&lt;|message|&gt;{"k":"v"}&lt;|call|&gt;</c>.
/// </summary>
/// <remarks>
/// <para>Follows llama.cpp's gpt-oss handling: the recipient (<c>to=functions.NAME</c>) may sit either after
/// the channel name or in the role header (<c>&lt;|start|&gt;assistant to=functions.NAME&lt;|channel|&gt;commentary json&lt;|message|&gt;</c>);
/// the content type (<c>&lt;|constrain|&gt;json</c> / <c>json</c>) is informational; several messages are parallel
/// calls; and the terminator is optional because <c>&lt;|call|&gt;</c> is an end-of-generation token and is not in
/// the decoded text. Messages whose recipient is not <c>functions.*</c> (the built-in <c>browser</c> / <c>python</c>
/// tools) are not reported, and a body that is not complete JSON is a truncated generation and is not reported
/// either (never execute half a call). Analysis and final messages in the same text are ignored.</para>
/// <para>The parser works on both the raw model output and the output of
/// <see cref="Reasoning.HarmonySplitter"/>, which passes tool segments through verbatim.</para>
/// </remarks>
public sealed class HarmonyToolCallParser : IToolCallParser
{
    private const string Message = "<|message|>";
    private const string Functions = "functions.";

    private static readonly string[] Terminators = ["<|end|>", "<|call|>", "<|return|>", "<|start|>", "<|channel|>"];
    private static readonly Regex Recipient = new(@"\bto=([^\s<]+)", RegexOptions.Compiled);

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText) => TryParse(generatedText, null);

    /// <inheritdoc/>
    public ToolCall[]? TryParse(string generatedText, IReadOnlyList<ToolDefinition>? tools)
    {
        if (string.IsNullOrEmpty(generatedText))
            return null;

        var calls = new List<ToolCall>();
        int pos = 0;
        while (true)
        {
            int msg = generatedText.IndexOf(Message, pos, StringComparison.Ordinal);
            if (msg < 0)
                break;

            // The header runs from the end of the previous message (or the last <|start|>) to <|message|>.
            string seg = generatedText.Substring(pos, msg - pos);
            int cut = 0;
            for (int i = 0; i < 3; i++)   // end / call / return terminate the previous message
            {
                int t = seg.LastIndexOf(Terminators[i], StringComparison.Ordinal);
                if (t >= 0)
                    cut = Math.Max(cut, t + Terminators[i].Length);
            }
            int st = seg.LastIndexOf("<|start|>", StringComparison.Ordinal);
            if (st >= cut)
                cut = st;
            int headerStart = pos + cut;

            string header = generatedText.Substring(headerStart, msg - headerStart);

            int bodyStart = msg + Message.Length;
            int bodyEnd = generatedText.Length;
            foreach (var term in Terminators)
            {
                int i = generatedText.IndexOf(term, bodyStart, StringComparison.Ordinal);
                if (i >= 0 && i < bodyEnd)
                {
                    bodyEnd = i;
                }
            }

            pos = bodyEnd;
            var m = Recipient.Match(header);
            if (!m.Success || !m.Groups[1].Value.StartsWith(Functions, StringComparison.Ordinal))
                continue;

            string name = m.Groups[1].Value[Functions.Length..];
            if (name.Length == 0)
                continue;

            string body = generatedText.Substring(bodyStart, bodyEnd - bodyStart).Trim();
            if (body.Length == 0)
                body = "{}";
            if (!IsCompleteJson(body))
                continue;      // truncated (or not JSON): never report half a call

            calls.Add(new ToolCall($"call_{calls.Count}", name, body));
        }

        if (calls.Count == 0)
            return null;
        return ToolArgumentCoercer.Coerce(calls.ToArray(), tools);
    }

    /// <inheritdoc/>
    public bool IsToolCallStart(string text)
    {
        int r = text.IndexOf("to=" + Functions, StringComparison.Ordinal);
        return r >= 0 && text.IndexOf(Message, r, StringComparison.Ordinal) >= 0;
    }

    private static bool IsCompleteJson(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.ValueKind == JsonValueKind.Object;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
