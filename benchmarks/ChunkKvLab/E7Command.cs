using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace ChunkKvLab;

/// <summary>Options of <see cref="E7Command"/>.</summary>
internal sealed record E7Options(string ModelPath, string CorpusDir, int[] Seeds, int Edits, int ChunkTokens,
    string? ExpectSymbolHash, string? ExpectFixedHash, string OutCsv);

/// <summary>
/// E7 (incremental re-encode, tokenisation only): index a frozen corpus with the symbol and the fixed-size chunker
/// (<see cref="ChunkIndex"/>), then apply single-method edits one at a time to the pristine snapshot and count the tokens whose
/// chunk K/V would have to be re-encoded (chunks the index reports as added whose key the pristine store lacks), per chunker.
/// Each edit is reverted before the next, so edits are independent. No model forward runs: keys come from token ids and a
/// fixed producer label.
/// </summary>
/// <remarks>
/// An edit picks a method or constructor (uniformly, without replacement within a seed) that has at least one statement alone
/// on its line, one such statement, and one of three kinds: <c>insert</c> (a new statement line after it), <c>delete</c> (its
/// line removed) or <c>modify</c> (the last character of its first identifier of length 2 or more replaced, in place).
/// </remarks>
internal static class E7Command
{
    private const string InsertedStatement = "System.Diagnostics.Debug.WriteLine(\"e7 edit\");";
    private static readonly string[] Kinds = ["insert", "delete", "modify"];

    public static int Run(E7Options o)
    {
        var total = Stopwatch.StartNew();
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(o.ModelPath);
        ITokenizer tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Console.WriteLine($"model {o.ModelPath} (tokenizer only)");

        // Frozen-snapshot check: both chunkings must reproduce the corpus.md hashes.
        Corpus symbolCorpus = TasksRunner.LoadCorpus(o.CorpusDir, "symbol", o.ChunkTokens, tokenizer, o.ExpectSymbolHash);
        Corpus fixedCorpus = TasksRunner.LoadCorpus(o.CorpusDir, "fixed", o.ChunkTokens, tokenizer, o.ExpectFixedHash);
        IReadOnlyList<SourceFile> files = symbolCorpus.Files;

        var rope = new RopeParameters(128, 5_000_000f, RoPEType.NeoX);
        ChunkKeyFunction keyOf = ids => ChunkKey.Create("e7-sim", "gguf-embedded:" + Path.GetFileNameWithoutExtension(o.ModelPath), rope,
            "cpu", KvDtype.F32, ids);

        var arms = new[] { new Arm("symbol", symbolCorpus.Chunker, keyOf), new Arm("fixed", fixedCorpus.Chunker, keyOf) };
        foreach (Arm arm in arms)
        {
            var sw = Stopwatch.StartNew();
            IndexDiff build = arm.Index.Update(files, arm.Chunker);
            long tokens = build.AddedTokens;
            Corpus reference = arm.Name == "symbol" ? symbolCorpus : fixedCorpus;
            if (build.Added.Count != reference.Chunks.Count || tokens != reference.TotalTokens)
                throw new InvalidOperationException($"{arm.Name}: index has {build.Added.Count} chunks / {tokens} tokens, corpus {reference.Chunks.Count} / {reference.TotalTokens}");
            foreach (ChunkRef r in build.Added) arm.Store.Add(r.Key!.Value);
            Console.WriteLine($"{arm.Name}: index built in {sw.Elapsed.TotalSeconds:F1} s: {build.Added.Count} chunks, {tokens} tokens, "
                + $"{arm.Index.DistinctKeyCount} distinct keys; status missing {arm.Index.Status(arm.Store).MissingChunks}");
        }

        List<Site> sites = FindSites(files);
        Console.WriteLine($"edit sites: {sites.Count} methods/constructors with a statement alone on its line, in {sites.Select(s => s.File).Distinct().Count()} files");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(o.OutCsv))!);
        using var csv = new StreamWriter(o.OutCsv, append: false, new UTF8Encoding(false));
        var header = new StringBuilder("seed,edit,file,method,line,kind,method_tokens,file_tokens_before,file_tokens_after");
        foreach (Arm arm in arms)
            header.Append(CultureInfo.InvariantCulture, $",{arm.Name}_file_chunks,{arm.Name}_added,{arm.Name}_removed,{arm.Name}_miss_chunks,{arm.Name}_reencoded_tokens,{arm.Name}_update_ms");
        csv.WriteLine(header.ToString());

        foreach (int seed in o.Seeds)
        {
            var rng = new Random(seed);
            int[] order = Enumerable.Range(0, sites.Count).OrderBy(_ => rng.Next()).Take(o.Edits).ToArray();
            for (int e = 0; e < order.Length; e++)
            {
                Site site = sites[order[e]];
                string kind = Kinds[rng.Next(Kinds.Length)];
                (int start, int end) = site.Statements[rng.Next(site.Statements.Length)];
                SourceFile original = files[site.FileIndex];
                string edited = Apply(original.Text, kind, start, end);
                if (edited == original.Text) throw new InvalidOperationException($"edit {seed}/{e} did not change {site.File}");
                var editedFile = new SourceFile(original.Path, edited);

                int methodTokens = tokenizer.EncodeRaw(original.Text[site.Start..site.End]).Length;
                int before = tokenizer.EncodeRaw(original.Text).Length, after = tokenizer.EncodeRaw(edited).Length;
                var row = new StringBuilder();
                row.Append(CultureInfo.InvariantCulture, $"{seed},{e},{site.File},{site.Name},{site.Line},{kind},{methodTokens},{before},{after}");
                foreach (Arm arm in arms)
                {
                    var sw = Stopwatch.StartNew();
                    IndexDiff diff = arm.Index.Update([editedFile], arm.Chunker, removeMissingFiles: false);
                    double ms = sw.Elapsed.TotalMilliseconds;
                    List<ChunkRef> misses = diff.Added.Where(r => !arm.Store.Contains(r.Key!.Value)).ToList();
                    long reencoded = misses.Sum(r => (long)r.TokenCount!.Value);
                    int fileChunks = arm.Index.ChunksOf(original.Path).Count;
                    // Cross-check: status against the pristine store lists exactly these chunks.
                    StatusReport status = arm.Index.Status(arm.Store);
                    if (status.MissingChunks != misses.Count || status.MissingTokens != reencoded)
                        throw new InvalidOperationException($"{arm.Name} edit {seed}/{e}: status {status.MissingChunks}/{status.MissingTokens} vs diff {misses.Count}/{reencoded}");
                    row.Append(CultureInfo.InvariantCulture, $",{fileChunks},{diff.Added.Count},{diff.Removed.Count},{misses.Count},{reencoded},{ms:F1}");

                    // Revert: every chunk of the pristine file is back in the store.
                    IndexDiff revert = arm.Index.Update([original], arm.Chunker, removeMissingFiles: false);
                    if (revert.Added.Any(r => !arm.Store.Contains(r.Key!.Value)) || !arm.Index.Status(arm.Store).IsUpToDate)
                        throw new InvalidOperationException($"{arm.Name} edit {seed}/{e}: revert did not restore the pristine index");
                }
                csv.WriteLine(row.ToString());
            }
            csv.Flush();
            Console.WriteLine($"seed {seed}: {order.Length} edits done ({total.Elapsed.TotalSeconds:F0} s)");
        }
        Console.WriteLine($"done in {total.Elapsed.TotalSeconds:F1} s; rows in {o.OutCsv}");
        return 0;
    }

    private sealed class Arm(string name, IChunker chunker, ChunkKeyFunction keyOf)
    {
        public string Name { get; } = name;
        public IChunker Chunker { get; } = chunker;
        public ChunkIndex Index { get; } = new(keyOf);
        public KeySetStore Store { get; } = new();
    }

    /// <summary>A method or constructor with a block body; <see cref="Statements"/> are spans of statements alone on their line.</summary>
    private sealed record Site(int FileIndex, string File, string Name, int Line, int Start, int End, (int Start, int End)[] Statements);

    private static List<Site> FindSites(IReadOnlyList<SourceFile> files)
    {
        var sites = new List<Site>();
        for (int fi = 0; fi < files.Count; fi++)
        {
            SourceFile f = files[fi];
            SyntaxNode root = CSharpSyntaxTree.ParseText(f.Text, SymbolChunker.ParseOptions, f.Path).GetRoot();
            foreach (BaseMethodDeclarationSyntax m in root.DescendantNodes().OfType<BaseMethodDeclarationSyntax>())
            {
                if (m is not (MethodDeclarationSyntax or ConstructorDeclarationSyntax) || m.Body is null) continue;
                var stmts = new List<(int, int)>();
                foreach (StatementSyntax s in m.Body.DescendantNodes().OfType<StatementSyntax>())
                {
                    if (s is BlockSyntax) continue;
                    int lineStart = f.Text.LastIndexOf('\n', Math.Max(0, s.SpanStart - 1)) + 1;
                    int lineEnd = f.Text.IndexOf('\n', s.Span.End - 1);
                    if (lineEnd < 0) continue;
                    // Alone on its line: only whitespace before it on the line and nothing (but whitespace) after it.
                    if (!string.IsNullOrWhiteSpace(f.Text[lineStart..s.SpanStart]) && lineStart != s.SpanStart) continue;
                    if (!string.IsNullOrWhiteSpace(f.Text[s.Span.End..lineEnd]) && s.Span.End != lineEnd) continue;
                    if (f.Text.AsSpan(s.SpanStart, s.Span.Length).Contains('\n')) continue;
                    if (!s.DescendantTokens().Any(t => t.IsKind(SyntaxKind.IdentifierToken) && t.ValueText.Length >= 2)) continue;
                    stmts.Add((s.SpanStart, s.Span.End));
                }
                if (stmts.Count == 0) continue;
                string name = m is MethodDeclarationSyntax md ? md.Identifier.ValueText : ".ctor";
                int line = f.Text.AsSpan(0, m.SpanStart).Count('\n') + 1;
                sites.Add(new Site(fi, f.Path, name, line, m.FullSpan.Start, m.FullSpan.End, stmts.ToArray()));
            }
        }
        return sites;
    }

    private static string Apply(string text, string kind, int start, int end)
    {
        int lineStart = text.LastIndexOf('\n', Math.Max(0, start - 1)) + 1;
        int lineEnd = text.IndexOf('\n', end - 1) + 1; // past the '\n'
        string indent = text[lineStart..start];
        switch (kind)
        {
            case "insert":
                return text[..lineEnd] + indent + InsertedStatement + "\n" + text[lineEnd..];
            case "delete":
                return text[..lineStart] + text[lineEnd..];
            case "modify":
            {
                SyntaxToken id = SyntaxFactory.ParseStatement(text[start..end]).DescendantTokens()
                    .First(t => t.IsKind(SyntaxKind.IdentifierToken) && t.ValueText.Length >= 2);
                int pos = start + id.Span.End - 1;
                char c = text[pos];
                char replacement = c == 'q' ? 'x' : 'q';
                return text[..pos] + replacement + text[(pos + 1)..];
            }
            default:
                throw new ArgumentException(kind);
        }
    }

    /// <summary>In-memory key set standing in for the pristine store (no K/V).</summary>
    private sealed class KeySetStore : IChunkStore
    {
        private readonly HashSet<ChunkKey> _keys = [];
        public void Add(ChunkKey key) => _keys.Add(key);
        public bool TryGet(ChunkKey key, out ChunkKvEntry entry) { entry = null!; return false; }
        public void Put(ChunkKey key, ChunkKvEntry entry) => _keys.Add(key);
        public bool Contains(ChunkKey key) => _keys.Contains(key);
        public IEnumerable<ChunkKey> Keys() => _keys;
        public long TotalBytes => 0;
        public void Evict(ChunkKey key) => _keys.Remove(key);
    }
}
