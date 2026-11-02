using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;

namespace ChunkKvLab;

/// <summary>
/// Save→Load round trip of real <see cref="ChunkIndex"/>es (symbol and fixed) over a frozen corpus, tokenizer only: the
/// corpus hashes are checked first (so the chunkers reproduce the pinned chunking), then each index is built, saved under a
/// fresh store directory, loaded back, and compared chunk by chunk; reports JSON size, save/load time, non-ASCII files and
/// zero-length chunks. Exits non-zero on any mismatch or load note.
/// </summary>
internal static class IndexRoundTrip
{
    public static int Run(string modelPath, string corpusDir, int chunkTokens, string? expectSymbol, string? expectFixed, string storeDir)
    {
        if (Directory.Exists(storeDir) && Directory.EnumerateFileSystemEntries(storeDir).Any())
            throw new InvalidOperationException($"store directory {storeDir} is not empty");
        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(modelPath);
        ITokenizer tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var rope = new RopeParameters(128, 5_000_000f, RoPEType.NeoX);
        ChunkKeyFunction keyOf = ids => ChunkKey.Create("index-roundtrip", "gguf-embedded:" + Path.GetFileNameWithoutExtension(modelPath), rope,
            "cpu", KvDtype.F32, ids);
        int failures = 0;
        foreach ((string name, string? hash) in new[] { ("symbol", expectSymbol), ("fixed", expectFixed) })
        {
            Corpus corpus = TasksRunner.LoadCorpus(corpusDir, name, chunkTokens, tokenizer, hash);
            int nonAscii = corpus.Files.Count(f => f.Text.Any(c => c > 127));
            var index = new ChunkIndex(keyOf);
            var sw = Stopwatch.StartNew();
            index.Update(corpus.Files, corpus.Chunker); // checks tiling of every file
            double buildMs = sw.Elapsed.TotalMilliseconds;
            string path = ChunkIndex.DefaultPath(storeDir, corpus.Chunker.Name);
            sw.Restart();
            index.Save(path);
            double saveMs = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            ChunkIndex loaded = ChunkIndex.Load(path, keyOf);
            double loadMs = sw.Elapsed.TotalMilliseconds;
            int mismatches = corpus.Files.Count(f => !index.ChunksOf(f.Path).SequenceEqual(loaded.ChunksOf(f.Path)));
            int zeroLength = corpus.Files.Sum(f => index.ChunksOf(f.Path).Count(r => r.EndOffset == r.StartOffset));
            bool idsMatch = corpus.Chunks.Select(c => c.Id).SequenceEqual(corpus.Files.SelectMany(f => index.ChunksOf(f.Path)).Select(r => r.Id));
            Console.WriteLine($"{corpus.Chunker.Name}: {index.ChunkCount} chunks in {index.Paths.Count} files ({nonAscii} with non-ASCII text), "
                + $"{zeroLength} zero-length chunks; build {buildMs:F0} ms, save {saveMs:F0} ms, load {loadMs:F0} ms; "
                + $"JSON {new FileInfo(path).Length:N0} bytes; load note: {loaded.LoadNote ?? "none"}; "
                + $"files differing after load: {mismatches}; chunk ids equal to the corpus chunking: {idsMatch}");
            if (loaded.LoadNote is not null || mismatches > 0 || !idsMatch) failures++;
        }
        return failures == 0 ? 0 : 1;
    }
}
