using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DotLLM.Engine.Evaluation;

/// <summary>Totals for one scored document.</summary>
/// <param name="TokensScored">Targets scored: tokens 1..n-1 (the first token has no prediction and is never scored).</param>
/// <param name="NllSum">Sum of per-target NLL in nats.</param>
/// <param name="KlSum">Sum of per-position KL(teacher||model) in nats, or null without a reference.</param>
/// <param name="KlMode">"full" or "topk_tail_lb", or null without a reference.</param>
public readonly record struct DocumentScore(int TokensScored, double NllSum, double? KlSum, string? KlMode)
{
    /// <summary>Mean NLL per scored token, or null when nothing was scored.</summary>
    public double? NllMean => TokensScored > 0 ? NllSum / TokensScored : null;

    /// <summary>Mean KL per scored position, or null without a reference or scored positions.</summary>
    public double? KlMean => KlSum is { } k && TokensScored > 0 ? k / TokensScored : null;
}

/// <summary>One per-document result line.</summary>
/// <param name="Doc">0-based document index.</param>
/// <param name="NTokens">Tokens fed to the model (after BOS prepend and truncation).</param>
/// <param name="Score">Scored totals.</param>
public readonly record struct DocResult(int Doc, int NTokens, DocumentScore Score);

/// <summary>
/// JSONL readers/writers for <c>dotllm perplexity --docs-file</c> (#1706). Hand-rolled on <see cref="Utf8JsonReader"/> /
/// <see cref="Utf8JsonWriter"/> (no source-generated DTOs, so the init-initializer trap in CLAUDE.md cannot apply).
/// </summary>
public static class DocumentJsonl
{
    /// <summary>
    /// Parses one document line: a JSON array of non-negative integer token ids, e.g. <c>[1, 5, 9]</c>.
    /// </summary>
    /// <param name="line">One line of text.</param>
    /// <param name="lineNumber">1-based line number for error messages.</param>
    /// <exception cref="FormatException">The line is not a JSON array of non-negative 32-bit integers.</exception>
    public static int[] ParseDocumentLine(string line, int lineNumber = 0)
    {
        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(line);
            var reader = new Utf8JsonReader(bytes);
            if (!reader.Read() || reader.TokenType != JsonTokenType.StartArray)
                throw new FormatException("expected a JSON array of token ids");
            var ids = new List<int>();
            while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
            {
                if (reader.TokenType != JsonTokenType.Number || !reader.TryGetInt32(out int id) || id < 0)
                    throw new FormatException("token ids must be non-negative integers");
                ids.Add(id);
            }
            if (reader.TokenType != JsonTokenType.EndArray)
                throw new FormatException("unterminated array");
            if (reader.Read())
                throw new FormatException("trailing content after the array");
            return ids.ToArray();
        }
        catch (JsonException ex)
        {
            throw new FormatException($"docs line {lineNumber}: {ex.Message}", ex);
        }
        catch (FormatException ex) when (!ex.Message.StartsWith("docs line", StringComparison.Ordinal))
        {
            throw new FormatException($"docs line {lineNumber}: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Reads every document of a JSONL file. Blank lines are ignored and do not consume a document index
    /// (document <c>i</c> is the i-th non-blank line, 0-based).
    /// </summary>
    /// <param name="path">Path of the <c>docs.jsonl</c> file.</param>
    public static IEnumerable<int[]> ReadDocuments(string path)
    {
        int lineNo = 0;
        foreach (string line in File.ReadLines(path))
        {
            lineNo++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            yield return ParseDocumentLine(line, lineNo);
        }
    }

    /// <summary>Formats one result as a single JSON line (no trailing newline).</summary>
    /// <param name="r">The result.</param>
    public static string FormatResultLine(in DocResult r)
    {
        var buffer = new ArrayBufferWriter<byte>(256);
        using (var w = new Utf8JsonWriter(buffer))
        {
            w.WriteStartObject();
            w.WriteNumber("doc", r.Doc);
            w.WriteNumber("n_tokens", r.NTokens);
            w.WriteNumber("n_tokens_scored", r.Score.TokensScored);
            w.WriteNumber("nll_sum", r.Score.NllSum);
            if (r.Score.NllMean is { } m) w.WriteNumber("nll_mean", m); else w.WriteNull("nll_mean");
            if (r.Score.KlSum is { } ks)
            {
                w.WriteNumber("kl_sum", ks);
                if (r.Score.KlMean is { } km) w.WriteNumber("kl_mean", km); else w.WriteNull("kl_mean");
                w.WriteString("kl_mode", r.Score.KlMode);
            }
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    /// <summary>Parses a result line back (used by tests and downstream tooling).</summary>
    /// <param name="line">A line produced by <see cref="FormatResultLine"/>.</param>
    public static DocResult ParseResultLine(string line)
    {
        using var d = JsonDocument.Parse(line);
        var o = d.RootElement;
        double? kl = o.TryGetProperty("kl_sum", out var k) ? k.GetDouble() : null;
        string? mode = o.TryGetProperty("kl_mode", out var km) ? km.GetString() : null;
        return new DocResult(o.GetProperty("doc").GetInt32(), o.GetProperty("n_tokens").GetInt32(),
            new DocumentScore(o.GetProperty("n_tokens_scored").GetInt32(), o.GetProperty("nll_sum").GetDouble(), kl, mode));
    }
}
