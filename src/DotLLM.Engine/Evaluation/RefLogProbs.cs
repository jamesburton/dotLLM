using System.Buffers;
using System.Text;
using System.Text.Json;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// One document of a <c>--ref-logprobs</c> file: the reference ("teacher") model's log-probabilities at every scored
/// position of that document. Position <c>j</c> (0-based) is the distribution that predicts sequence token <c>j + 1</c>,
/// i.e. exactly the rows <c>dotllm perplexity --docs-file</c> scores, so a file has <c>n_tokens_scored</c> positions.
/// </summary>
public sealed class RefDocument
{
    private readonly float[][]? _full;
    private readonly int[][]? _ids;
    private readonly float[][]? _topLogProbs;
    private readonly double[]? _tail;

    private RefDocument(int doc, float[][] full)
    {
        Doc = doc; _full = full; Positions = full.Length; Mode = "full";
    }

    private RefDocument(int doc, int[][] ids, float[][] lps, double[] tail)
    {
        Doc = doc; _ids = ids; _topLogProbs = lps; _tail = tail; Positions = ids.Length; Mode = "topk_tail_lb";
    }

    /// <summary>0-based document index (must equal the docs-file index).</summary>
    public int Doc { get; }

    /// <summary>Number of positions in this reference document.</summary>
    public int Positions { get; }

    /// <summary>KL mode label: <c>full</c> (exact) or <c>topk_tail_lb</c> (lower bound, see <see cref="DocumentScoring.KlTopKTail"/>).</summary>
    public string Mode { get; }

    /// <summary>KL(teacher || model) at <paramref name="position"/> for the model's logits row, in nats.</summary>
    /// <param name="position">0-based scored position.</param>
    /// <param name="modelLogits">Model logits row predicting that position's target.</param>
    public double Kl(int position, ReadOnlySpan<float> modelLogits)
        => _full is not null
            ? DocumentScoring.KlFull(_full[position], modelLogits)
            : DocumentScoring.KlTopKTail(_ids![position], _topLogProbs![position], _tail![position], modelLogits);

    /// <summary>Builds a full-vocabulary reference document.</summary>
    public static RefDocument FromFull(int doc, float[][] logProbs) => new(doc, logProbs);

    /// <summary>Builds a top-K + tail reference document.</summary>
    public static RefDocument FromTopK(int doc, int[][] ids, float[][] logProbs, double[] tailLogProb)
        => new(doc, ids, logProbs, tailLogProb);

    /// <summary>Parses one reference line.</summary>
    /// <param name="line">JSON object, see <c>docs/PERPLEXITY.md</c> "--ref-logprobs format".</param>
    /// <param name="lineNumber">1-based line number for error messages.</param>
    public static RefDocument Parse(string line, int lineNumber = 0)
    {
        try
        {
            byte[] bytes = Encoding.UTF8.GetBytes(line);
            var r = new Utf8JsonReader(bytes);
            if (!r.Read() || r.TokenType != JsonTokenType.StartObject)
                throw new FormatException("expected a JSON object");
            int doc = -1;
            string? format = null;
            List<float[]>? logprobs = null;
            List<int[]>? ids = null;
            List<double>? tail = null;
            while (r.Read() && r.TokenType == JsonTokenType.PropertyName)
            {
                string name = r.GetString()!;
                r.Read();
                switch (name)
                {
                    case "doc": doc = r.GetInt32(); break;
                    case "format": format = r.GetString(); break;
                    case "logprobs": logprobs = ReadFloatMatrix(ref r); break;
                    case "ids": ids = ReadIntMatrix(ref r); break;
                    case "tail_logprob":
                        tail = [];
                        r.Read();
                        while (r.TokenType != JsonTokenType.EndArray)
                        {
                            tail.Add(r.TokenType == JsonTokenType.Null ? double.NegativeInfinity : r.GetDouble());
                            r.Read();
                        }
                        break;
                    default: r.Skip(); break;
                }
            }
            if (doc < 0) throw new FormatException("missing \"doc\"");
            switch (format)
            {
                case "full":
                    if (logprobs is null) throw new FormatException("format \"full\" needs \"logprobs\"");
                    return new RefDocument(doc, logprobs.ToArray());
                case "topk":
                    if (logprobs is null || ids is null) throw new FormatException("format \"topk\" needs \"ids\" and \"logprobs\"");
                    if (ids.Count != logprobs.Count) throw new FormatException("\"ids\" and \"logprobs\" must have one row per position");
                    var tailArr = new double[ids.Count];
                    for (int p = 0; p < ids.Count; p++)
                    {
                        if (ids[p].Length != logprobs[p].Length)
                            throw new FormatException($"position {p}: ids/logprobs length mismatch");
                        if (ids[p].Distinct().Count() != ids[p].Length)
                            throw new FormatException($"position {p}: duplicate token ids in top-K row");
                        if (tail is not null)
                        {
                            if (tail.Count != ids.Count) throw new FormatException("\"tail_logprob\" must have one entry per position");
                            tailArr[p] = tail[p];
                        }
                        else
                        {
                            double mass = 0;
                            foreach (float lp in logprobs[p]) mass += Math.Exp(lp);
                            double rest = 1.0 - mass;
                            tailArr[p] = rest > 1e-12 ? Math.Log(rest) : double.NegativeInfinity;
                        }
                    }
                    return new RefDocument(doc, ids.ToArray(), logprobs.ToArray(), tailArr);
                default:
                    throw new FormatException($"unknown format \"{format}\" (expected \"full\" or \"topk\")");
            }
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException or FormatException && !ex.Message.StartsWith("ref line", StringComparison.Ordinal))
        {
            throw new FormatException($"ref line {lineNumber}: {ex.Message}", ex);
        }
    }

    private static List<float[]> ReadFloatMatrix(ref Utf8JsonReader r)
    {
        var rows = new List<float[]>();
        if (r.TokenType != JsonTokenType.StartArray) throw new FormatException("expected an array of arrays");
        r.Read();
        while (r.TokenType != JsonTokenType.EndArray)
        {
            var row = new List<float>();
            r.Read();
            while (r.TokenType != JsonTokenType.EndArray)
            {
                row.Add(r.TokenType == JsonTokenType.Null ? float.NegativeInfinity : r.GetSingle());
                r.Read();
            }
            rows.Add(row.ToArray());
            r.Read();
        }
        return rows;
    }

    private static List<int[]> ReadIntMatrix(ref Utf8JsonReader r)
    {
        var rows = new List<int[]>();
        if (r.TokenType != JsonTokenType.StartArray) throw new FormatException("expected an array of arrays");
        r.Read();
        while (r.TokenType != JsonTokenType.EndArray)
        {
            var row = new List<int>();
            r.Read();
            while (r.TokenType != JsonTokenType.EndArray)
            {
                row.Add(r.GetInt32());
                r.Read();
            }
            rows.Add(row.ToArray());
            r.Read();
        }
        return rows;
    }

    /// <summary>Formats a full-vocabulary reference line (for tests / tooling).</summary>
    public static string FormatFullLine(int doc, IReadOnlyList<float[]> logProbs)
    {
        var buf = new ArrayBufferWriter<byte>(1024);
        using (var w = new Utf8JsonWriter(buf))
        {
            w.WriteStartObject();
            w.WriteNumber("doc", doc);
            w.WriteString("format", "full");
            w.WriteStartArray("logprobs");
            foreach (float[] row in logProbs)
            {
                w.WriteStartArray();
                foreach (float v in row)
                    if (float.IsNegativeInfinity(v)) w.WriteNullValue(); else w.WriteNumberValue(v);
                w.WriteEndArray();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buf.WrittenSpan);
    }

    /// <summary>Formats a top-K + tail reference line (for tests / tooling).</summary>
    public static string FormatTopKLine(int doc, IReadOnlyList<int[]> ids, IReadOnlyList<float[]> logProbs, IReadOnlyList<double>? tailLogProb)
    {
        var buf = new ArrayBufferWriter<byte>(1024);
        using (var w = new Utf8JsonWriter(buf))
        {
            w.WriteStartObject();
            w.WriteNumber("doc", doc);
            w.WriteString("format", "topk");
            w.WriteStartArray("ids");
            foreach (int[] row in ids) { w.WriteStartArray(); foreach (int v in row) w.WriteNumberValue(v); w.WriteEndArray(); }
            w.WriteEndArray();
            w.WriteStartArray("logprobs");
            foreach (float[] row in logProbs) { w.WriteStartArray(); foreach (float v in row) w.WriteNumberValue(v); w.WriteEndArray(); }
            w.WriteEndArray();
            if (tailLogProb is not null)
            {
                w.WriteStartArray("tail_logprob");
                foreach (double t in tailLogProb) if (double.IsNegativeInfinity(t)) w.WriteNullValue(); else w.WriteNumberValue(t);
                w.WriteEndArray();
            }
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buf.WrittenSpan);
    }
}

/// <summary>Streams a <c>--ref-logprobs</c> file in document order.</summary>
public sealed class RefLogProbsReader : IDisposable
{
    private readonly StreamReader _reader;
    private int _line;
    private int _doc;

    /// <summary>Opens <paramref name="path"/>.</summary>
    public RefLogProbsReader(string path) => _reader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete));

    /// <summary>Reads the next document, or <see langword="null"/> at end of file. Blank lines are skipped.</summary>
    public RefDocument? Next()
    {
        string? line;
        while ((line = _reader.ReadLine()) is not null)
        {
            _line++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            RefDocument d = RefDocument.Parse(line, _line);
            if (d.Doc != _doc)
                throw new FormatException($"ref line {_line}: expected doc {_doc} but the line says {d.Doc} (references must be in docs-file order, one per document).");
            _doc++;
            return d;
        }
        return null;
    }

    /// <inheritdoc/>
    public void Dispose() => _reader.Dispose();
}
