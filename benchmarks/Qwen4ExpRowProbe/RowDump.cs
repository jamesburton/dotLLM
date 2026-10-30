using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// Build-to-build identity probe (#996): <c>rowdump gguf outPrefix [chunks=1,2,3,4]</c>. Runs the <see cref="DecodeKl.Texts"/> sequences teacher-forced in chunks of N rows
/// through ANY Vulkan model class (it only uses <see cref="IModel"/>), writing every chunk's logits (all rows where the model supports all-row logits, else the last row)
/// to <c>outPrefix.cN.f32</c>. Compiled unchanged into a dev build and a branch build; the files are then compared byte for byte (`cmp`) - a model class the diff does not
/// reach must come out identical.
/// </summary>
internal static class RowDump
{
    public static unsafe void Run(string[] args)
    {
        string path = args[1], prefix = args[2];
        var chunks = (args.Length > 3 ? args[3] : "1,2,3,4").Split(',').Select(int.Parse).ToArray();
        using var gguf = GgufFile.Open(path);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        var seqs = DecodeKl.Texts.Select(t => tk.Encode(t).ToArray()).ToList();
        string? longPath = Environment.GetEnvironmentVariable("DOTLLM_SWEEP_LONG");
        if (!string.IsNullOrEmpty(longPath)) seqs.Add(tk.Encode(File.ReadAllText(longPath)).Take(int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_SWEEP_LONG_TOKENS"), out int lc) ? lc : 1000).ToArray());
        bool cpu = Environment.GetEnvironmentVariable("DOTLLM_ROWDUMP_CPU") == "1";   // CPU engine as the truth oracle (branch build only)
        VulkanDevice? dev = cpu ? null : VulkanDevice.Create();
        IModel model; Func<int, DotLLM.Core.Attention.IKvCache> kvFactory;
        if (cpu)
        {
            model = ModelLoader.CreateCpuModelFromGguf(gguf, cfg, new DotLLM.Core.Configuration.ThreadingConfig(0, 0));
            kvFactory = size => new DotLLM.Engine.KvCache.SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(cfg), size);
        }
        else (model, kvFactory) = VulkanModelLoader.CreateFromGguf(dev!, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        bool allRows = model.TrySetAllRowLogitsLimit(4);
        Console.WriteLine($"[rowdump] {cfg.Architecture} as {model.GetType().Name}, all-row logits: {allRows}, vocab {cfg.VocabSize}, {seqs.Sum(s => s.Length)} tokens");
        int vocab = cfg.VocabSize;
        foreach (int chunk in chunks)
        {
            using var fs = new FileStream($"{prefix}.c{chunk}.f32", FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
            foreach (var ids in seqs)
            {
                using var cache = kvFactory(ids.Length + 8);
                for (int i = 0; i < ids.Length; i += chunk)
                {
                    int n = Math.Min(chunk, ids.Length - i);
                    using var t = model.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), cpu ? -1 : 0, cache);
                    int rows = (int)t.Shape[0];
                    if (allRows) { if (rows != n) throw new InvalidOperationException($"rows {rows} != {n}"); fs.Write(new ReadOnlySpan<byte>((void*)t.DataPointer, n * vocab * 4)); }
                    else fs.Write(new ReadOnlySpan<byte>((byte*)t.DataPointer + (long)(rows - 1) * vocab * 4, vocab * 4));
                }
            }
            Console.WriteLine($"[rowdump] chunk {chunk} done, {fs.Length} bytes");
        }
        (model as IDisposable)?.Dispose();
        dev?.Dispose();
    }
}
