using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Accumulates streaming token text and detects tool call boundaries.
/// Used by callers of <see cref="TextGenerator.GenerateStreamingTokensAsync"/> to handle
/// tool calls during streaming without buffering all output.
/// </summary>
public sealed class StreamingToolCallAccumulator
{
    private readonly IToolCallParser _parser;
    private readonly StringBuilder _buffer = new();
    private readonly IReadOnlyList<string> _markers;
    private readonly int _longestMarker;
    private bool _toolCallDetected;
    private int _emitted;

    /// <summary>Whether a tool call start marker has been detected in the accumulated text.</summary>
    public bool IsToolCallInProgress => _toolCallDetected;

    /// <summary>Creates a new accumulator with the given parser.</summary>
    /// <param name="parser">The model-specific tool call parser.</param>
    public StreamingToolCallAccumulator(IToolCallParser parser)
    {
        _parser = parser;
        _markers = parser.StartMarkers;
        foreach (string m in _markers)
            _longestMarker = Math.Max(_longestMarker, m.Length);
    }

    /// <summary>Characters of the accumulated text already handed out by <see cref="AppendStreamable"/>.</summary>
    public int EmittedLength => _emitted;

    /// <summary>
    /// Appends answer text and returns the part that is safe to stream now (#1287): everything up to the first complete
    /// start marker of the parser, minus a trailing fragment that is a prefix of some marker and could still complete
    /// into one. From the first marker on (or once the parser's <see cref="IToolCallParser.IsToolCallStart"/> heuristic
    /// fires) nothing more is released. Text held back that never parses into a call is the caller's to deliver at the
    /// end (from <see cref="EmittedLength"/>).
    /// </summary>
    /// <param name="text">Incremental answer text.</param>
    /// <returns>The newly streamable text; empty when everything new is held back.</returns>
    public string AppendStreamable(string text)
    {
        _buffer.Append(text);
        if (_toolCallDetected)
            return string.Empty;

        string s = _buffer.ToString();

        // A marker can start at most longest-1 chars before what is already out (that tail was held back).
        int from = Math.Max(0, _emitted - _longestMarker);
        int cut = -1;
        foreach (string m in _markers)
        {
            int i = s.IndexOf(m, from, StringComparison.Ordinal);
            if (i >= 0 && (cut < 0 || i < cut))
                cut = i;
        }

        int safe;
        if (cut >= 0)
        {
            _toolCallDetected = true;
            safe = cut;
        }
        else if (_parser.IsToolCallStart(s))
        {
            // A heuristic call start with no literal marker (e.g. a bare pythonic call): stop where we are.
            _toolCallDetected = true;
            safe = _emitted;
        }
        else
        {
            safe = s.Length - MarkerPrefixTail(s);
        }

        if (safe <= _emitted)
            return string.Empty;
        string delta = s[_emitted..safe];
        _emitted = safe;
        return delta;
    }

    /// <summary>Length of the longest proper suffix of <paramref name="s"/> that is a prefix of some marker.</summary>
    private int MarkerPrefixTail(string s)
    {
        int best = 0;
        foreach (string m in _markers)
        {
            for (int len = Math.Min(m.Length - 1, s.Length); len > best; len--)
            {
                if (s.AsSpan(s.Length - len).SequenceEqual(m.AsSpan(0, len)))
                {
                    best = len;
                    break;
                }
            }
        }

        return best;
    }

    /// <summary>
    /// Appends new text and checks for tool call markers.
    /// </summary>
    /// <param name="text">Incremental text from streaming generation.</param>
    /// <returns>True if this text should be suppressed from user output (it's part of a tool call).</returns>
    public bool Append(string text)
    {
        _buffer.Append(text);
        if (!_toolCallDetected)
            _toolCallDetected = _parser.IsToolCallStart(_buffer.ToString());
        return _toolCallDetected;
    }

    /// <summary>
    /// Attempts to parse completed tool calls from the accumulated buffer.
    /// </summary>
    /// <returns>Parsed tool calls, or null if parsing fails or is incomplete.</returns>
    public ToolCall[]? TryParseCompleted(IReadOnlyList<ToolDefinition>? tools = null) => _parser.TryParse(_buffer.ToString(), tools);

    /// <summary>Gets the full accumulated text.</summary>
    public string GetAccumulatedText() => _buffer.ToString();

    /// <summary>Resets the accumulator for a new generation turn.</summary>
    public void Reset()
    {
        _buffer.Clear();
        _toolCallDetected = false;
        _emitted = 0;
    }
}
