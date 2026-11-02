using System.Globalization;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>
/// E4c, tokenizer only (no weights run): how often a chunk fails the <b>text round trip</b> a wire server would do — its text,
/// tokenised alone with <see cref="ITokenizer.EncodeRaw"/>, gives ids different from the chunk's. Per chunker (<c>fixed</c> id
/// slices, <c>symbol</c>, <c>fixed-text</c>) and per chunk, two texts are checked: the decoded ids (what a client holding only
/// the ids would send) and the chunk's character range of the file (what a text-first client sends). Also checks the
/// text-first invariants on the real tokenizer (pieces concatenate to the file, every piece ≤ N, no lone surrogate) and prints
/// each chunker's chunk-id hash (the <c>--expect-hash</c> values).
/// </summary>
internal static class RoundTripCommand
{
    public static int Run(string modelPath, string corpusDir, string name, int chunkTokens, string outCsv, string[] chunkers)
    {
        var ci = CultureInfo.InvariantCulture;
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(modelPath);
        ITokenizer tok = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        bool append = File.Exists(outCsv);
        using var csv = new StreamWriter(outCsv, append);
        if (!append)
            csv.WriteLine("corpus,chunker,file,chunk,tokens,decode_fffd,decoded_rt_ok,decoded_len_diff,decoded_first_diff,decoded_last_diff,range_rt_ok");
        int violations = 0;
        foreach (string chunker in chunkers)
        {
            Corpus corpus = TasksRunner.LoadCorpus(corpusDir, chunker, chunkTokens, tok, null);
            int n = 0, decodedFail = 0, fffd = 0, rangeFail = 0, edgeOnly = 0;
            var failFiles = new HashSet<string>(StringComparer.Ordinal);
            foreach (SourceFile f in corpus.Files)
            {
                IReadOnlyList<Chunk> chunks = corpus.ChunksOf(f.Path);
                if (chunker == "fixed-text")
                {
                    // Invariants on the real tokenizer.
                    if (!string.Equals(string.Concat(chunks.Select(corpus.TextOf)), f.Text, StringComparison.Ordinal)) { violations++; Console.WriteLine($"  VIOLATION concat {f.Path}"); }
                    foreach (Chunk c in chunks)
                    {
                        string t = corpus.TextOf(c);
                        if (c.TokenIds.Length > chunkTokens || c.TokenIds.Length == 0) { violations++; Console.WriteLine($"  VIOLATION size {c.Id} {c.TokenIds.Length}"); }
                        if (char.IsLowSurrogate(t[0]) || char.IsHighSurrogate(t[^1])) { violations++; Console.WriteLine($"  VIOLATION surrogate {c.Id}"); }
                    }
                }
                for (int k = 0; k < chunks.Count; k++)
                {
                    Chunk c = chunks[k];
                    n++;
                    string decoded = tok.Decode(c.TokenIds);
                    int[] re = tok.EncodeRaw(decoded);
                    bool ok = re.AsSpan().SequenceEqual(c.TokenIds);
                    int firstDiff = -1, lastDiff = -1;
                    if (!ok)
                    {
                        decodedFail++;
                        failFiles.Add(f.Path);
                        int m = Math.Min(re.Length, c.TokenIds.Length);
                        firstDiff = 0;
                        while (firstDiff < m && re[firstDiff] == c.TokenIds[firstDiff]) firstDiff++;
                        lastDiff = 0;
                        while (lastDiff < m && re[^(lastDiff + 1)] == c.TokenIds[^(lastDiff + 1)]) lastDiff++;
                        // A mismatch confined to the first or last 3 ids of the chunk (a merge or a split character at a cut).
                        if (firstDiff >= m - 3 || lastDiff >= m - 3) edgeOnly++;
                    }
                    bool hasFffd = decoded.Contains('�', StringComparison.Ordinal) && !f.Text.Contains('�', StringComparison.Ordinal);
                    if (hasFffd) fffd++;
                    string range = corpus.TextOf(c);
                    bool rangeOk = tok.EncodeRaw(range).AsSpan().SequenceEqual(c.TokenIds);
                    if (!rangeOk) rangeFail++;
                    csv.WriteLine(string.Join(',', name, chunker, "\"" + f.Path + "\"", k.ToString(ci), c.TokenIds.Length.ToString(ci), hasFffd ? "1" : "0",
                        ok ? "1" : "0", (re.Length - c.TokenIds.Length).ToString(ci), firstDiff.ToString(ci), lastDiff.ToString(ci), rangeOk ? "1" : "0"));
                }
            }
            Console.WriteLine($"roundtrip {name} {corpus.Chunker.Name}: {n} chunks; decoded-text round trip fails {decodedFail} ({100.0 * decodedFail / n:F1} %; "
                + $"{failFiles.Count}/{corpus.Files.Count} files; U+FFFD in decoded text {fffd}; mismatch confined to the first/last 3 ids {edgeOnly}); "
                + $"file-range text round trip fails {rangeFail} ({100.0 * rangeFail / n:F1} %)");
        }
        Console.WriteLine(violations == 0 ? "text-first invariants: OK" : $"text-first invariants: {violations} VIOLATIONS");
        return violations == 0 ? 0 : 1;
    }
}
