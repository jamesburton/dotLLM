using System.Security.Cryptography;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// #1452 identity probe: <c>ovident gguf [chunks=2,5,8]</c>. Teacher-forces the <see cref="DecodeKl.Texts"/> sequences in N-row chunks twice - shared-expert overlap OFF
/// then ON, same process, same model - and compares every chunk's logits byte for byte (all rows) plus the greedy argmax of every row. Also checks the overlap counter
/// moved only in the ON pass. Exit code 0 = bit-identical.
/// </summary>
internal static class SharedOverlapIdent
{
    public static unsafe int Run(string[] args)
    {
        string path = args[1];
        var chunks = (args.Length > 2 ? args[2] : "2,5,8").Split(',').Select(int.Parse).ToArray();
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        var seqs = DecodeKl.Texts.Select(t => tk.Encode(t).ToArray()).ToList();
        using var dev = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        bool allRows = model.TrySetAllRowLogitsLimit(8);
        var core = model as VulkanQwen3MoeHybridTransformerModel;
        Console.WriteLine($"[ovident] {model.GetType().Name}, all-row logits: {allRows}, overlap-capable core: {core is not null}, {seqs.Sum(s => s.Length)} tokens");
        int vocab = cfg.VocabSize; int bad = 0;
        foreach (int chunk in chunks)
        {
            var perCall = new List<string>[] { new(), new(), new() }; var hashes = new string[3]; var argmaxes = new List<int>[] { new(), new(), new() };
            long[] counts = new long[3];
            for (int pass = 0; pass < 3; pass++)   // pass 0 = discarded warm-up (recurrent state leaks across sequences: both compared passes must start from the same state)
            {
                VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = pass == 2 && Environment.GetEnvironmentVariable("OVIDENT_CONTROL") != "1";
                long before = core?.MoePathCounts[(int)VulkanQwen3MoeHybridTransformerModel.MoePath.SharedOverlap] ?? 0;
                using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                foreach (var ids in seqs)
                {
                    model.ResetSequenceState(); using var cache = kvFactory(ids.Length + 8);
                    for (int i = 0; i < ids.Length; i += chunk)
                    {
                        int n = Math.Min(chunk, ids.Length - i);
                        using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), 0, cache);
                        int rows = (int)t.Shape[0];
                        var span = new ReadOnlySpan<float>((void*)t.DataPointer, rows * vocab);
                        h.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(span));
                        perCall[pass].Add(Convert.ToHexString(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(span)))[..8]);
                        for (int r = 0; r < rows; r++)
                        {
                            int best = 0; var row = span.Slice(r * vocab, vocab);
                            for (int v = 1; v < vocab; v++) if (row[v] > row[best]) best = v;
                            argmaxes[pass].Add(best);
                        }
                    }
                }
                hashes[pass] = Convert.ToHexString(h.GetHashAndReset());
                counts[pass] = (core?.MoePathCounts[(int)VulkanQwen3MoeHybridTransformerModel.MoePath.SharedOverlap] ?? 0) - before;
            }
            bool same = hashes[1] == hashes[2] && argmaxes[1].SequenceEqual(argmaxes[2]);
            bool counterOk = Environment.GetEnvironmentVariable("OVIDENT_CONTROL") == "1" || (counts[1] == 0 && counts[2] > 0);
            Console.WriteLine($"[ovident] chunk {chunk}: logits {(hashes[1] == hashes[2] ? "BIT-IDENTICAL" : "DIFFER")}, greedy {(argmaxes[1].SequenceEqual(argmaxes[2]) ? "identical" : "DIFFER")} ({argmaxes[1].Count} rows), overlap layers off/on = {counts[1]}/{counts[2]} {(counterOk ? "" : "COUNTER-WRONG")}");
            if (!same) { int fd = Enumerable.Range(0, perCall[1].Count).First(i => perCall[1][i] != perCall[2][i]); int nd = Enumerable.Range(0, perCall[1].Count).Count(i => perCall[1][i] != perCall[2][i]); Console.WriteLine($"[ovident]   first differing forward #{fd} of {perCall[1].Count}; {nd} differ"); }
            if (!same || !counterOk) bad++;
        }
        VulkanQwen3MoeHybridTransformerModel.SharedOverlapEnabled = true;
        (model as IDisposable)?.Dispose();
        return bad;
    }
}
