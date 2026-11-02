using System.Globalization;

namespace ChunkKvLab;

/// <summary>
/// Chunk-KV reuse lab (research harness, CPU).
/// <code>
/// ChunkKvLab e1 --model &lt;gguf&gt; --corpus &lt;dir&gt; --seeds 3 --out results.csv [--sizes 2048,4096,8192] [--pattern *.cs]
///               [--expect-hash &lt;sha256&gt;] [--first-seed 0] [--cross-check none|cpu|cpu+full] (cpu: with --backend vulkan, also CPU naive; +full: also CPU full prefill)
///               [--chunk 512] [--continuation 64] [--greedy 16] [--threads 0]
/// ChunkKvLab e2 --model &lt;gguf&gt; --corpus &lt;dir&gt; --out e2.csv [--sizes 2048,4096] [--budgets 5,10,15,20] [--selectors ...]
///               [--contiguous =a,b] [--query 16] [--noise 1] [--expect-hash &lt;sha256&gt;] [--seeds 3] [--first-seed 0]
/// ChunkKvLab corpus-info --model &lt;gguf&gt; --corpus &lt;dir&gt; [--chunker e1|symbol|fixed] [--chunk 512] [--expect-hash &lt;sha256&gt;]
/// ChunkKvLab tasks --model &lt;gguf&gt; --corpus &lt;dir&gt; --out &lt;prefix&gt; [--chunker symbol|fixed] [--chunk 512] [--count 100] [--seed 0]
///               [--context 2048] [--continuation 32] [--prefix 256] [--per-context 4] [--max-distractors n] [--validate 0|1]
///               [--max-minutes 0] [--expect-hash &lt;sha256&gt;]
/// ChunkKvLab graph-info --model &lt;gguf&gt; --corpus &lt;dir&gt; [--chunker symbol|fixed] [--chunk 512] [--repeat 3] [--plans 100] [--seed 0]
///               [--expect-hash &lt;sha256&gt;]
/// ChunkKvLab e3 --model &lt;gguf&gt; --corpus &lt;dir&gt; --name &lt;corpus name&gt; --out-dir &lt;dir&gt; [--chunker symbol|fixed] [--chunk 512]
///               [--seed 0] [--count 100] [--context 2048] [--contexts 0 (all)] [--kept &lt;validation.csv&gt; --plans-jsonl &lt;plans&gt;]
///               [--query 1] [--raw 0] [--draws 20] [--max-minutes 0] [--chunk-cache-tokens 40000] [--expect-hash &lt;sha256&gt;]
/// ChunkKvLab e4 --model &lt;gguf&gt; --corpus &lt;dir&gt; --name &lt;corpus name&gt; --out &lt;csv&gt; [--chunker symbol|fixed] [--chunk 512] [--seed 0]
///               [--mode kept|unfiltered] [--kept &lt;validation.csv&gt;] [--plans-jsonl &lt;plans&gt;] [--contexts 0 (all)] [--budgets 5,10,15,20]
///               [--arms ...] [--light-arms a,b --light-budgets 5,20] [--window 16] [--shard 0/1] [--max-minutes 0] [--expect-hash &lt;sha256&gt;]
/// ChunkKvLab rope-precision --model &lt;gguf&gt; --corpus &lt;dir&gt; --out rope.csv [--deltas 1024,4096,8192] [--chunks 3]
/// ChunkKvLab backend-check --model &lt;gguf&gt; [--tokens 64]
/// ChunkKvLab floor --model &lt;gguf&gt; --names a,b --corpus-dirs d1,d2 --hashes h1,h2 --contexts 7,6 --kept-dir &lt;experiments/corpus&gt;
///               --out-dir &lt;dir&gt; [--label 2k] [--mode kept|unfiltered] [--context 2048] [--ref-block 512] [--blocks 2048,96,64,8,1]
///               (loads CPU and Vulkan itself; no --backend)
/// </code>
/// Global option on every model-loading command: <c>--backend cpu|vulkan</c> (default cpu). On vulkan the model and the chunk
/// backend share one device and all-row logits are enabled on demand (<c>TrySetAllRowLogitsLimit</c>); the experiment runners
/// other than e1 still need the CPU <c>SimpleKvCache</c> and refuse vulkan (e1 runs on either backend since Phase G Task 3).
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("usage: ChunkKvLab e1|rope-precision --model <gguf> --corpus <dir> --out <csv> [options]");
            return 2;
        }
        var opts = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 1; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
            {
                Console.Error.WriteLine($"bad argument '{args[i]}'");
                return 2;
            }
            opts[args[i][2..]] = args[++i];
        }

        string Get(string key, string? fallback = null) =>
            opts.TryGetValue(key, out string? v) ? v : fallback ?? throw new ArgumentException($"--{key} is required");
        int Int(string key, int fallback) => opts.TryGetValue(key, out string? v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;
        int[] Ints(string key, string fallback) =>
            Get(key, fallback).Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();

        if (args[0] == "floor" && opts.ContainsKey("backend"))
        {
            Console.Error.WriteLine("'floor' loads both the CPU and the Vulkan model itself; drop --backend.");
            return 2;
        }
        string backend = Get("backend", DotLLM.ChunkReuse.Backends.ChunkBackendTags.Cpu);
        if (backend is not (DotLLM.ChunkReuse.Backends.ChunkBackendTags.Cpu or DotLLM.ChunkReuse.Backends.ChunkBackendTags.Vulkan))
        {
            Console.Error.WriteLine($"--backend must be cpu or vulkan; got '{backend}'");
            return 2;
        }
        LabModel.SelectedBackend = backend;
        if (backend != DotLLM.ChunkReuse.Backends.ChunkBackendTags.Cpu && args[0] is not ("backend-check" or "corpus-info" or "graph-info" or "e1"))
        {
            // Fail before loading weights: the experiment runners still drive SimpleKvCache directly.
            Console.Error.WriteLine($"'{args[0]}' runs on the CPU backend only (not yet ported to the chunk backend seam); --backend {backend} is accepted by e1 and backend-check.");
            return 2;
        }

        switch (args[0])
        {
            case "e1":
                return E1Runner.Run(new E1Options(
                    Get("model"), Get("corpus"), Get("pattern", "*.cs"), Int("seeds", 3), Ints("sizes", "2048,4096,8192"),
                    Int("chunk", 512), Int("continuation", 64), Int("greedy", 16), Int("threads", 0), Get("out"),
                    opts.GetValueOrDefault("expect-hash"), Get("cross-check", "none") switch
                    {
                        "none" => false,
                        "cpu" or "cpu+full" => true,
                        string x => throw new ArgumentException($"--cross-check must be none, cpu or cpu+full; got '{x}'"),
                    }, Int("first-seed", 0), Get("cross-check", "none") == "cpu+full"));
            case "e2":
            {
                // "--contiguous =" (value after '=') lets a script pass an empty list.
                string[] List(string key, string fallback) =>
                    Get(key, fallback).TrimStart('=').Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                return E2Runner.Run(new E2Options(
                    Get("model"), Get("corpus"), Get("pattern", "*.cs"), Int("seeds", 3), Ints("sizes", "2048,4096"),
                    Ints("budgets", "5,10,15,20"),
                    List("selectors", "first_k,chunk_starts,deviation,deviation_oracle,random,query,combined"),
                    List("contiguous", "="), Int("chunk", 512), Int("continuation", 64), Int("query", 16), Int("greedy", 16),
                    Int("threads", 0), Get("out"), opts.GetValueOrDefault("expect-hash"), Int("noise", 1) != 0, Int("first-seed", 0)));
            }
            case "rope-precision":
                return RopePrecision.Run(Get("model"), Get("corpus"), Get("pattern", "*.cs"), Ints("deltas", "1024,4096,8192"),
                    Int("chunks", 3), Int("chunk", 512), Int("threads", 0), Get("out"));
            case "corpus-info":
            {
                // Tokenizer only (no weights are run): document/chunk counts and the chunk-id hash of a corpus directory.
                // --chunker e1 (default): the E1/E2 ContextBuilder (all files, 512-token chunks, tails dropped).
                // --chunker symbol|fixed: the Task 8 code corpus (tests/generated/bin/obj excluded, tails kept).
                using var gguf = DotLLM.Models.Gguf.GgufFile.Open(Get("model"));
                var tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
                string chunker = Get("chunker", "e1");
                if (chunker != "e1")
                {
                    TasksRunner.LoadCorpus(Get("corpus"), chunker, Int("chunk", 512), tokenizer, opts.GetValueOrDefault("expect-hash"));
                    return 0;
                }
                var corpus = ContextBuilder.FromDirectory(Get("corpus"), Get("pattern", "*.cs"), tokenizer, Int("chunk", 512));
                Console.WriteLine($"corpus {Get("corpus")}: {corpus.DocumentCount} documents, {corpus.Chunks.Count} chunks; chunk-id sha256 {corpus.ChunkIdHash}");
                return 0;
            }
            case "graph-info":
                return GraphRunner.Run(new GraphOptions(Get("model"), Get("corpus"), Get("chunker", "symbol"), Int("chunk", 512),
                    Int("repeat", 3), Int("plans", 100), Int("seed", 0), opts.GetValueOrDefault("expect-hash")));
            case "e3":
                return E3Command.Run(new E3Options(
                    Get("model"), Get("corpus"), Get("name"), Get("chunker", "symbol"), Int("chunk", 512), opts.GetValueOrDefault("expect-hash"),
                    Int("seed", 0), Int("count", 100),
                    new DotLLM.ChunkReuse.CodeCorpus.TaskGeneratorOptions
                    {
                        ContextTokens = Int("context", 2048), ContinuationTokens = Int("continuation", 32), PrefixTokens = Int("prefix", 256),
                        TasksPerContext = Int("per-context", 4),
                    },
                    Int("contexts", 0), opts.GetValueOrDefault("plans-jsonl"), opts.GetValueOrDefault("kept"), Int("query", 1) != 0,
                    Int("raw", 0), Int("draws", 20), double.Parse(Get("max-minutes", "0"), CultureInfo.InvariantCulture), Int("threads", 0),
                    Get("out-dir"), Int("chunk-cache-tokens", 40000)));
            case "e4":
            {
                string[] List(string key, string fallback) =>
                    Get(key, fallback).TrimStart('=').Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                string[] shard = Get("shard", "0/1").Split('/');
                return E4Command.Run(new E4Options(
                    Get("model"), Get("corpus"), Get("name"), Get("chunker", "symbol"), Int("chunk", 512), opts.GetValueOrDefault("expect-hash"),
                    Int("seed", 0), Int("count", 100),
                    new DotLLM.ChunkReuse.CodeCorpus.TaskGeneratorOptions
                    {
                        ContextTokens = Int("context", 2048), ContinuationTokens = Int("continuation", 32), PrefixTokens = Int("prefix", 256),
                        TasksPerContext = Int("per-context", 4),
                    },
                    Get("mode", "kept"), opts.GetValueOrDefault("kept"), opts.GetValueOrDefault("plans-jsonl"), Int("contexts", 0),
                    Ints("budgets", "5,10,15,20"),
                    List("arms", "drop_required,oracle_chunk,oracle_def,first_k,chunk_starts,deviation,random,query,graph,hybrid_query,hybrid_dev,oracle_def_fill"),
                    List("light-arms", "="), Ints("light-budgets", "5,20"), Int("window", 16),
                    double.Parse(Get("max-minutes", "0"), CultureInfo.InvariantCulture), Int("threads", 0), Get("out"),
                    Int("chunk-cache-tokens", 40000), int.Parse(shard[0], CultureInfo.InvariantCulture), int.Parse(shard[1], CultureInfo.InvariantCulture)));
            }
            case "tasks":
                return TasksRunner.Run(new TasksOptions(
                    Get("model"), Get("corpus"), Get("chunker", "symbol"), Int("chunk", 512), Int("count", 100), Int("seed", 0),
                    new DotLLM.ChunkReuse.CodeCorpus.TaskGeneratorOptions
                    {
                        ContextTokens = Int("context", 2048), ContinuationTokens = Int("continuation", 32), PrefixTokens = Int("prefix", 256),
                        TasksPerContext = Int("per-context", 4), MaxDistractors = Int("max-distractors", int.MaxValue),
                    },
                    Int("validate", 0) != 0, double.Parse(Get("max-minutes", "0"), CultureInfo.InvariantCulture), Int("threads", 0),
                    Get("out"), opts.GetValueOrDefault("expect-hash")));
            case "floor":
            {
                string[] List(string key) => Get(key).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                return FloorCommand.Run(new FloorOptions(
                    Get("model"), List("names"), List("corpus-dirs"), List("hashes"), Ints("contexts", ""), Get("kept-dir"), Get("mode", "kept"),
                    Int("context", 2048), Int("ref-block", 512), Ints("blocks", "2048,96,64,8,1"), Int("threads", 0), Get("label", "2k"),
                    Get("out-dir")));
            }
            case "backend-check":
                return BackendCheck.Run(Get("model"), Int("tokens", 64), Int("threads", 0));
            default:
                Console.Error.WriteLine($"unknown command '{args[0]}'");
                return 2;
        }
    }
}
