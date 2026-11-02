using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;

namespace ChunkKvLab;

/// <summary>
/// Store size and speed checkpoint: encodes a few corpus chunks, puts them into a fresh <see cref="FileChunkStore"/>, reads
/// them back through a second store instance and reports file bytes, bytes per token per layer, and encode/put/get times.
/// </summary>
internal static class StoreProbe
{
    public static int Run(string modelPath, string corpusDir, string pattern, int chunks, int chunkTokens, int threads, string storeDir)
    {
        using LabModel lab = LabModel.Load(modelPath, threads);
        ContextBuilder corpus = ContextBuilder.FromDirectory(corpusDir, pattern, lab.Tokenizer, chunkTokens);
        var encoder = new ChunkEncoder(lab.Fingerprint);
        var rope = RopeParameters.FromConfig(lab.Config);
        string tokenizerId = lab.TokenizerId;
        string engine = ChunkFingerprint.EngineVersion; // build identity, logged only
        Console.WriteLine($"corpus {corpusDir}: {corpus.DocumentCount} documents, {corpus.Chunks.Count} chunks; chunk-id sha256 {corpus.ChunkIdHash}");
        Console.WriteLine($"store {storeDir}; key: backend cpu, dtype F32, kv-output version {ChunkReuseSchema.KvOutputVersion} (build {engine})");
        Console.WriteLine($"analytic F32 K+V bytes per token: {lab.KvBytesPerToken} over {lab.Geometry.LayerCount} layers "
            + $"= {lab.KvBytesPerToken / lab.Geometry.LayerCount} per token per layer");
        if (Directory.Exists(storeDir) && Directory.EnumerateFileSystemEntries(storeDir).Any())
            throw new InvalidOperationException($"store directory {storeDir} is not empty");

        var writer = new FileChunkStore(storeDir);
        Console.WriteLine("chunk,tokens,encode_ms,put_ms,get_ms,file_bytes,payload_bytes_per_token_per_layer,header_bytes,bit_exact");
        var rng = new Random(12345);
        for (int c = 0; c < chunks; c++)
        {
            int idx = rng.Next(corpus.Chunks.Count);
            int[] ids = corpus.Chunks[idx].TokenIds;
            var sw = Stopwatch.StartNew();
            ChunkKvEntry entry = encoder.Encode(lab.Model, ids);
            double encodeMs = sw.Elapsed.TotalMilliseconds;

            ChunkKey key = ChunkKey.Create(lab.Fingerprint, tokenizerId, rope, "cpu", KvDtype.F32, ids);
            sw.Restart();
            writer.Put(key, entry);
            double putMs = sw.Elapsed.TotalMilliseconds;

            var reader = new FileChunkStore(storeDir); // fresh instance: the read comes from disk (OS page cache may be warm)
            sw.Restart();
            bool hit = reader.TryGet(key, out ChunkKvEntry got);
            double getMs = sw.Elapsed.TotalMilliseconds;

            bool exact = hit && got.TokenIds.AsSpan().SequenceEqual(ids);
            for (int l = 0; exact && l < entry.KeysPerLayer.Length; l++)
            {
                exact = MemoryMarshal.AsBytes(entry.KeysPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(got.KeysPerLayer[l].AsSpan()))
                    && MemoryMarshal.AsBytes(entry.ValuesPerLayer[l].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(got.ValuesPerLayer[l].AsSpan()));
            }
            long fileBytes = new FileInfo(writer.PathOf(key)).Length;
            long kvBytes = 0;
            for (int l = 0; l < entry.KeysPerLayer.Length; l++) kvBytes += (entry.KeysPerLayer[l].Length + entry.ValuesPerLayer[l].Length) * 4L;
            long header = fileBytes - kvBytes - ids.Length * 4L;
            double perTokenLayer = (double)kvBytes / ids.Length / entry.KeysPerLayer.Length;
            Console.WriteLine($"{idx},{ids.Length},{encodeMs:F1},{putMs:F1},{getMs:F1},{fileBytes},{perTokenLayer:F1},{header},{(exact ? 1 : 0)}");
            if (!exact) return 1;
        }
        Console.WriteLine($"store total bytes {writer.TotalBytes}, entries {writer.Keys().Count()}");
        return 0;
    }
}
