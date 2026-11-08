using System.Text.Json;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// A spec-compliant reader for the SSE bodies the endpoint tests capture (#1498).
/// </summary>
/// <remarks>
/// <para>The endpoints write SSE <b>comment</b> lines (<c>: waiting for prompt processing (10s)</c>, #1023/#1025) whenever a
/// request waits longer than the keep-alive interval for the gate or for its first token. Every test used to read the body with
/// <c>block["data: ".Length..]</c>, which turns such a comment into <c>ing for prompt processing (10s)</c> and hands it to
/// <see cref="JsonDocument.Parse(string, JsonDocumentOptions)"/>: <c>'i' is an invalid start of a value</c>. A stalled CI runner
/// (a first <c>Task.Yield</c> continuation more than 10 s late) was enough to trip it, which made the failure look like a
/// parser race.</para>
/// <para>This reader follows the event-stream spec instead: blocks are separated by a blank line, a line starting with
/// <c>:</c> is a comment and ignored, <c>event:</c> and <c>data:</c> are fields (one optional space after the colon), and
/// multiple <c>data:</c> lines of one block join with <c>\n</c>. A block without a <c>data:</c> field (a pure comment) yields
/// no frame.</para>
/// </remarks>
internal static class SseTestReader
{
    /// <summary>One SSE event: its <c>event:</c> name (null when absent, as on the OpenAI wire) and its <c>data:</c> payload.</summary>
    public readonly record struct Frame(string? Event, string Data);

    /// <summary>Every event in <paramref name="raw"/>, in order, comments dropped. <c>data: [DONE]</c> is kept.</summary>
    public static List<Frame> Frames(string raw)
    {
        var frames = new List<Frame>();
        foreach (string block in raw.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string? name = null;
            List<string>? data = null;
            foreach (string line in block.Split('\n'))
            {
                if (line.Length == 0 || line[0] == ':')
                    continue;                                   // comment (keep-alive) or stray blank
                int colon = line.IndexOf(':', StringComparison.Ordinal);
                string field = colon < 0 ? line : line[..colon];
                string value = colon < 0 ? "" : line[(colon + 1)..];
                if (value.StartsWith(' '))
                    value = value[1..];
                switch (field)
                {
                    case "event": name = value; break;
                    case "data": (data ??= []).Add(value); break;
                    default: break;                             // id:, retry:, unknown fields: ignored
                }
            }
            if (data is not null)
                frames.Add(new Frame(name, string.Join('\n', data)));
        }
        return frames;
    }

    /// <summary>The <c>data:</c> payloads of <paramref name="raw"/>, in order, with <c>[DONE]</c> and comments excluded.</summary>
    public static IEnumerable<string> Data(string raw)
        => Frames(raw).Select(f => f.Data).Where(d => d != "[DONE]");

    /// <summary>Every <c>data:</c> payload of <paramref name="raw"/> parsed as JSON (<c>[DONE]</c> and comments excluded).</summary>
    public static IEnumerable<JsonElement> Json(string raw)
        => Data(raw).Select(d => JsonDocument.Parse(d).RootElement.Clone());

    /// <summary>Anthropic-style frames: every event with its name and its JSON payload (comments excluded).</summary>
    public static IEnumerable<(string Event, JsonElement Data)> NamedJson(string raw)
        => Frames(raw).Select(f => (f.Event ?? throw new InvalidOperationException("frame without event: " + f.Data), JsonDocument.Parse(f.Data).RootElement.Clone()));

    /// <summary>True when <paramref name="raw"/> carries at least one SSE comment line (a keep-alive).</summary>
    public static bool HasComment(string raw)
        => raw.Split('\n').Any(l => l.StartsWith(':'));
}
