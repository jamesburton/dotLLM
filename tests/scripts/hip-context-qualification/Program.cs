using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;

namespace DotLLM.HipContextQualification;

internal static class Program
{
    private const string ModelHash = "5157CA60744D21631818364854AC8E4452E1B8022D2AB4C8A2F9CDA2344AFB30";
    private const string CorpusHash = "01B38EA4C710A84BC18D0BD41271A5A1A92B94E97B2812F4DECE97D4A694725E";
    private const string TokenHash = "7C043C12B9A873068520D7D7A4E3AB3B89A0F6AE9BF456BFBE53D3B61FDA4F37";
    private const int Window = 128;
    private static readonly int[] Contexts = [1024, 2048, 4096, 8192];
    private static readonly string[] GpuComparisons = ["hipF16VsCpuF32", "hipQ8VsCpuF32", "hipQ8VsCpuQ8", "hipQ8VsHipF16", "cpuQ8VsCpuF32"];
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private static StreamWriter? _log;

    private sealed record Options(string Model, string Corpus, string Output, int MaxContext, bool CpuOnly);
    private sealed record TopLogit(int Token, float Value);
    private sealed record MetricResult(int Scored, int Top1Matches, double MaxAbs, double MaxRelativeL2, double MaxKl,
        double PplReference, double PplActual, double PplDeltaPercent, bool GatePassed);

    private sealed class Metrics(string name)
    {
        private int _scored, _topMatches;
        private double _maxAbs, _maxL2, _maxKl, _nllReference, _nllActual;
        public bool Pass => _scored > 0 && _maxL2 < .02 && _maxKl < .002;
        public MetricResult Result => new(_scored, _topMatches, _maxAbs, _maxL2, _maxKl,
            Math.Exp(_nllReference / _scored), Math.Exp(_nllActual / _scored),
            100 * (Math.Exp((_nllActual - _nllReference) / _scored) - 1), Pass);

        public void Add(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual, int target, int position, bool rotate = false)
        {
            Require(reference.Length == actual.Length && (uint)target < (uint)reference.Length, "Invalid logit rows");
            double squaredError = 0, squaredNorm = 0, sumReference = 0, sumActual = 0, kl = 0, maxAbs = 0;
            float maxReference = float.NegativeInfinity, maxActual = float.NegativeInfinity;
            int bestReference = 0, bestActual = 0;
            for (int i = 0; i < reference.Length; i++)
            {
                float e = reference[i], a = actual[(i + (rotate ? 1 : 0)) % actual.Length];
                Require(float.IsFinite(e) && float.IsFinite(a), "Non-finite logits");
                if (e > maxReference) { maxReference = e; bestReference = i; }
                if (a > maxActual) { maxActual = a; bestActual = i; }
                maxAbs = Math.Max(maxAbs, Math.Abs(e - a));
                squaredError += (e - a) * (double)(e - a); squaredNorm += e * (double)e;
            }
            Require(squaredNorm > 0, "Zero-norm reference logits");
            for (int i = 0; i < reference.Length; i++)
            {
                sumReference += Math.Exp(reference[i] - maxReference);
                sumActual += Math.Exp(actual[(i + (rotate ? 1 : 0)) % actual.Length] - maxActual);
            }
            double logReference = Math.Log(sumReference) + maxReference, logActual = Math.Log(sumActual) + maxActual;
            for (int i = 0; i < reference.Length; i++)
                kl += Math.Exp(reference[i] - logReference) * (reference[i] - logReference - actual[(i + (rotate ? 1 : 0)) % actual.Length] + logActual);
            double relativeL2 = Math.Sqrt(squaredError / squaredNorm);
            double nllReference = logReference - reference[target], nllActual = logActual - actual[(target + (rotate ? 1 : 0)) % actual.Length];
            Require(double.IsFinite(relativeL2) && double.IsFinite(kl) && double.IsFinite(nllReference) && double.IsFinite(nllActual), "Non-finite metrics");
            _maxAbs = Math.Max(_maxAbs, maxAbs); _maxL2 = Math.Max(_maxL2, relativeL2); _maxKl = Math.Max(_maxKl, kl);
            _nllReference += nllReference; _nllActual += nllActual;
            _scored++; if (bestReference == bestActual) _topMatches++;
            if (!rotate)
            {
                var referenceTop = Top(reference); var actualTop = Top(actual);
                Log(new { kind = "row", comparison = name, position, target, maxAbs, relativeL2, kl, nllReference, nllActual,
                    referenceTop, actualTop, referenceMargin = referenceTop[0].Value - referenceTop[1].Value,
                    actualMargin = actualTop[0].Value - actualTop[1].Value });
            }
        }
    }

    private static TopLogit[] Top(ReadOnlySpan<float> row)
    {
        TopLogit[] result = [new(-1, float.NegativeInfinity), new(-1, float.NegativeInfinity), new(-1, float.NegativeInfinity)];
        for (int i = 0; i < row.Length; i++)
            for (int k = 0; k < result.Length; k++)
                if (row[i] > result[k].Value)
                {
                    for (int j = result.Length - 1; j > k; j--) result[j] = result[j - 1];
                    result[k] = new(i, row[i]); break;
                }
        return result;
    }

    private static int Main(string[] args)
    {
        try
        {
            if (args is ["--help"])
            {
                Console.WriteLine("--model <external SmolLM2-135M F16 GGUF> --corpus <external Gutenberg Alice11 UTF8> --output <fresh .docs or external directory> [--max-context 1024|2048|4096|8192] [--cpu-only]\nvalidate --results <output/results.jsonl>\nExit 0: all original quality gates pass (or artifact validation passes); 10: completed with numerical gate failures; 2: runtime/provenance/harness failure. GPU Q8 requires user-set DOTLLM_HIP_ALLOW_Q8_KV=1. No fixture downloads or acknowledgement changes are performed.");
                return 0;
            }
            if (args is ["validate", "--results", var results])
            {
                var records = ReadRecords(results);
                Validate(records, Path.GetDirectoryName(Path.GetFullPath(results))!);
                var mutated = records.Select(r => (JsonObject)r.DeepClone()).ToList();
                var last = mutated.Last(r => Text(r, "kind") is "checkpoint");
                last["cpuF32Length"] = Number(last, "context") - 1;
                bool killed = false;
                try { Validate(mutated, Path.GetDirectoryName(Path.GetFullPath(results))!); }
                catch (InvalidDataException e) when (e.Message is "Cache did not reach real context") { killed = true; }
                Require(killed, "Shortened-context artifact mutant survived");
                Console.WriteLine(JsonSerializer.Serialize(new { artifactChecksPassed = true, shortenedContextMutantKilled = true }, Json));
                return 0;
            }
            var options = Parse(args);
            Directory.CreateDirectory(options.Output);
            string logPath = Path.Combine(options.Output, "results.jsonl");
            using var log = new StreamWriter(new FileStream(logPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
            _log = log;
            try
            {
                bool allPass = Run(options);
                int exitCode = allPass ? 0 : 10;
                Log(new { kind = "summary", completed = true, allOriginalGatesPassed = allPass, negativeControlKilled = true,
                    lifecycleComplete = true, scoredPerComparison = Contexts.Count(c => c <= options.MaxContext) * Window, exitCode });
                Validate(ReadRecords(logPath), options.Output);
                return exitCode;
            }
            catch (Exception e) { Log(new { kind = "error", message = e.Message, exitCode = 2 }); throw; }
            finally { _log = null; }
        }
        catch (Exception e) { Console.Error.WriteLine($"Qualification failed: {e.Message}"); return 2; }
    }

    private static Options Parse(string[] args)
    {
        string? model = null, corpus = null, output = null;
        int maxContext = 8192; bool cpuOnly = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] is "--cpu-only") { cpuOnly = true; continue; }
            Require(i + 1 < args.Length, "Missing option value; use --help");
            switch (args[i])
            {
                case "--model": model = Path.GetFullPath(args[++i]); break;
                case "--corpus": corpus = Path.GetFullPath(args[++i]); break;
                case "--output": output = Path.GetFullPath(args[++i]); break;
                case "--max-context": maxContext = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                default: throw new ArgumentException($"Unknown option {args[i]}; use --help", nameof(args));
            }
        }
        Require(model is not null && corpus is not null && output is not null, "Explicit --model, --corpus and --output paths are required");
        Require(Contexts.Contains(maxContext), "Context must be 1024, 2048, 4096 or 8192");
        string? repository = FindRepository(Environment.CurrentDirectory) ?? FindRepository(AppContext.BaseDirectory);
        if (repository is not null)
        {
            Require(!Inside(model!, repository) && !Inside(corpus!, repository), "Model and corpus must remain outside the repository");
            if (Inside(output!, repository)) Require(Inside(output!, Path.Combine(repository, ".docs")), "In-repository results must be under ignored .docs");
        }
        if (!cpuOnly) Require(Environment.GetEnvironmentVariable("DOTLLM_HIP_ALLOW_Q8_KV") is "1",
            "GPU qualification includes experimental Q8 KV: explicitly set DOTLLM_HIP_ALLOW_Q8_KV=1 to acknowledge its failed original parity gates");
        return new(model!, corpus!, output!, maxContext, cpuOnly);
    }

    private static string? FindRepository(string path)
    {
        for (DirectoryInfo? directory = new(path); directory is not null; directory = directory.Parent)
            if (Directory.Exists(Path.Combine(directory.FullName, ".git")) || File.Exists(Path.Combine(directory.FullName, ".git"))) return directory.FullName;
        return null;
    }
    private static bool Inside(string path, string directory)
    {
        string relative = Path.GetRelativePath(directory, path);
        return !Path.IsPathRooted(relative) && relative is not ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal);
    }
    private static string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
    private static string HashTokens(int[] tokens) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(tokens.AsSpan())));
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static void Log(object value) { string json = JsonSerializer.Serialize(value, Json); _log!.WriteLine(json); Console.WriteLine(json); }
    private static unsafe ReadOnlySpan<float> Row(ITensor tensor, int vocab) => new((void*)tensor.DataPointer, vocab);
    private static long FreeBytes() { HipDriverApi.hipMemGetInfo(out nuint free, out _).ThrowOnError(); return (long)free; }

    private static bool Run(Options options)
    {
        Require(BitConverter.IsLittleEndian, "Pinned token hash requires a little-endian process");
        Require(string.Equals(HashFile(options.Model), ModelHash, StringComparison.Ordinal), "Pinned model SHA256 differs");
        Require(string.Equals(HashFile(options.Corpus), CorpusHash, StringComparison.Ordinal), "Pinned corpus SHA256 differs");
        using var file = GgufFile.Open(options.Model);
        var config = GgufModelConfigExtractor.Extract(file.Metadata);
        Require(config.MaxSequenceLength >= options.MaxContext && config.HiddenSize == 576 && config.NumLayers == 30
            && config.NumAttentionHeads == 9 && config.NumKvHeads == 3 && config.HeadDim == 64 && config.VocabSize == 49152, "Pinned model geometry differs");
        string corpus = Encoding.UTF8.GetString(File.ReadAllBytes(options.Corpus)).Replace("\r\n", "\n");
        int contents = corpus.IndexOf("CHAPTER I.", StringComparison.Ordinal);
        int start = contents < 0 ? -1 : corpus.IndexOf("CHAPTER I.", contents + 1, StringComparison.Ordinal);
        Require(start == 1467, "Pinned chapter offset differs");
        int[] tokens = GgufTokenizerFactory.Load(file.Metadata).Encode(corpus.Substring(start, Math.Min(60000, corpus.Length - start))).Take(8193).ToArray();
        Require(tokens.Length == 8193 && string.Equals(HashTokens(tokens), TokenHash, StringComparison.Ordinal), "Pinned shared token stream differs");
        using (var tokenFile = new FileStream(Path.Combine(options.Output, "tokens.json"), FileMode.CreateNew)) JsonSerializer.Serialize(tokenFile, tokens);
        int[] boundaries = Contexts.Where(c => c <= options.MaxContext).ToArray();
        var assemblies = new[] { typeof(HipTransformerModel).Assembly, typeof(TransformerModel).Assembly, typeof(SimpleKvCache).Assembly,
            typeof(ThreadingConfig).Assembly, typeof(Dequantize).Assembly, typeof(ITokenizer).Assembly, typeof(Program).Assembly };
        Log(new { kind = "protocol", version = 1, mode = options.CpuOnly ? "cpu-only" : "hip", hipSkipped = options.CpuOnly,
            model = options.Model, corpus = options.Corpus, modelSha256 = ModelHash, corpusSha256 = CorpusHash, tokenSha256 = TokenHash,
            maxContext = options.MaxContext, window = Window, prefillBatch = 32, boundaries, chapterStartCharacter = start,
            tokenizer = "shared GGUF token IDs; CRLF normalized to LF; no added BOS", config.HiddenSize, config.NumLayers,
            config.NumAttentionHeads, config.NumKvHeads, config.HeadDim, config.VocabSize, config.RoPEConfig,
            relativeL2Gate = .02, klGate = .002, runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
            processArchitecture = RuntimeInformation.ProcessArchitecture.ToString(), cpuThreads = 4,
            managedAssemblies = assemblies.Select(a => new { name = a.GetName().Name,
                informationalVersion = a.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion, sha256 = HashFile(a.Location) }),
            nativeCodeObjectSha256 = options.CpuOnly ? null : HashFile(Path.Combine(AppContext.BaseDirectory, "co", "dense_f16.co")) });
        using var cpu = TransformerModel.LoadFromGguf(file, config with { MaxSequenceLength = options.MaxContext }, new ThreadingConfig(4, EnableCallerPinning: false));
        using var cpuF32 = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, options.MaxContext);
        using var cpuQ8 = new QuantizedKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, options.MaxContext, KvCacheDType.Q8_0, KvCacheDType.Q8_0, windowSize: 0);
        using var hip = options.CpuOnly ? null : HipTransformerModel.LoadFromGguf(file, config);
        using var hipF16 = hip?.CreateKvCache(options.MaxContext);
        using var hipQ8 = hip?.CreateKvCache(options.MaxContext, keyType: KvCacheDType.Q8_0, valueType: KvCacheDType.Q8_0);
        long initialFree = options.CpuOnly ? 0 : FreeBytes();
        Require(cpuF32.CurrentLength == 0 && cpuQ8.CurrentLength == 0 && (hipF16?.AllocatedBytes ?? 0) == 0 && (hipQ8?.AllocatedBytes ?? 0) == 0, "Caches must begin empty");
        Log(new { kind = "memory", at = "empty", cpuF32Length = 0, cpuQ8Length = 0, hipSkipped = options.CpuOnly,
            hipF16Bytes = hipF16?.AllocatedBytes, hipQ8Bytes = hipQ8?.AllocatedBytes, freeDeviceBytes = options.CpuOnly ? (long?)null : initialFree,
            storedWeightBytes = hip?.StoredWeightBytes, weightMemoryBytes = hip?.WeightMemoryBytes, computeMemoryBytes = hip?.ComputeMemoryBytes });
        var stopwatch = Stopwatch.StartNew();
        double[] timings = new double[4];
        bool allPass = true, negativeControlDone = false;
        int position = 0;
        string[] comparisons = options.CpuOnly ? ["cpuQ8VsCpuF32"] : GpuComparisons;
        foreach (int boundary in boundaries)
        {
            var metrics = comparisons.ToDictionary(name => name, name => new Metrics(name), StringComparer.Ordinal);
            while (position < boundary)
            {
                bool score = position >= boundary - Window;
                int count = score ? 1 : Math.Min(32, boundary - Window - position);
                var ids = tokens.AsSpan(position, count); int[] positions = Enumerable.Range(position, count).ToArray();
                long tick = Stopwatch.GetTimestamp();
                using var expected = cpu.Forward(ids, positions, -1, cpuF32, true);
                timings[0] += Stopwatch.GetElapsedTime(tick).TotalMilliseconds;
                tick = Stopwatch.GetTimestamp();
                using var expectedQ8 = cpu.Forward(ids, positions, -1, cpuQ8, true);
                timings[1] += Stopwatch.GetElapsedTime(tick).TotalMilliseconds;
                tick = Stopwatch.GetTimestamp();
                using var actualF16 = hip?.Forward(ids, positions, 0, hipF16, true);
                if (hip is not null) timings[2] += Stopwatch.GetElapsedTime(tick).TotalMilliseconds;
                tick = Stopwatch.GetTimestamp();
                using var actualQ8 = hip?.Forward(ids, positions, 0, hipQ8, true);
                if (hip is not null) timings[3] += Stopwatch.GetElapsedTime(tick).TotalMilliseconds;
                if (score)
                {
                    int target = tokens[position + 1], vocab = config.VocabSize;
                    metrics["cpuQ8VsCpuF32"].Add(Row(expected, vocab), Row(expectedQ8, vocab), target, position);
                    if (!options.CpuOnly)
                    {
                        metrics["hipF16VsCpuF32"].Add(Row(expected, vocab), Row(actualF16!, vocab), target, position);
                        metrics["hipQ8VsCpuF32"].Add(Row(expected, vocab), Row(actualQ8!, vocab), target, position);
                        metrics["hipQ8VsCpuQ8"].Add(Row(expectedQ8, vocab), Row(actualQ8!, vocab), target, position);
                        metrics["hipQ8VsHipF16"].Add(Row(actualF16!, vocab), Row(actualQ8!, vocab), target, position);
                    }
                    if (!negativeControlDone)
                    {
                        var wrong = new Metrics("negativeControl");
                        wrong.Add(Row(expected, vocab), Row(actualF16 ?? expectedQ8, vocab), target, position, rotate: true);
                        Log(new { kind = "negativeControl", position, substitution = "rotate actual vocabulary logits by one on a real scored token", result = wrong.Result });
                        Require(!wrong.Pass, "Numerical substitution survived original gates"); negativeControlDone = true;
                    }
                }
                position += count;
                if (position % 256 == 0) Log(new { kind = "progress", position, seconds = stopwatch.Elapsed.TotalSeconds, timingsMs = timings });
            }
            Require(cpuF32.CurrentLength == boundary && cpuQ8.CurrentLength == boundary
                && (hipF16 is null || hipF16.CurrentLength == boundary) && (hipQ8 is null || hipQ8.CurrentLength == boundary), "Cache did not reach real context");
            long? free = options.CpuOnly ? null : FreeBytes();
            Log(new { kind = "checkpoint", context = boundary, firstScoredInputPosition = boundary - Window,
                lastScoredInputPosition = boundary - 1, lastTargetPosition = boundary, cpuF32Length = cpuF32.CurrentLength, cpuQ8Length = cpuQ8.CurrentLength,
                hipF16Length = hipF16?.CurrentLength, hipQ8Length = hipQ8?.CurrentLength, metrics = metrics.ToDictionary(k => k.Key, k => k.Value.Result, StringComparer.Ordinal),
                hipF16Bytes = hipF16?.AllocatedBytes, hipQ8Bytes = hipQ8?.AllocatedBytes, hipF16Pages = hipF16?.AllocatedPages, hipQ8Pages = hipQ8?.AllocatedPages,
                freeDeviceBytes = free, driverGrowthBytes = free is null ? null : initialFree - free,
                cpuF32PopulatedPayloadBytes = 2L * config.NumLayers * config.NumKvHeads * config.HeadDim * boundary * 4,
                cpuQ8PopulatedPayloadBytes = 2L * config.NumLayers * config.NumKvHeads * config.HeadDim / 32 * 34 * boundary,
                seconds = stopwatch.Elapsed.TotalSeconds, timingsMs = timings });
            allPass &= metrics.Values.All(m => m.Pass);
        }
        Require(negativeControlDone, "No numerical substitution was exercised");
        cpuF32.Rollback(0); cpuQ8.Rollback(0); hipF16?.Rollback(0); hipQ8?.Rollback(0);
        Require(cpuF32.CurrentLength == 0 && cpuQ8.CurrentLength == 0 && (hipF16?.AllocatedBytes ?? 0) == 0 && (hipQ8?.AllocatedBytes ?? 0) == 0, "KV reset failed");
        Log(new { kind = "memory", at = "reset", cpuF32Length = cpuF32.CurrentLength, cpuQ8Length = cpuQ8.CurrentLength,
            hipF16Bytes = hipF16?.AllocatedBytes, hipQ8Bytes = hipQ8?.AllocatedBytes, freeDeviceBytes = options.CpuOnly ? (long?)null : FreeBytes() });
        hipF16?.Dispose(); hipQ8?.Dispose(); hip?.Dispose();
        Log(new { kind = "memory", at = "disposed", hipSkipped = options.CpuOnly, weightMemoryBytes = hip?.WeightMemoryBytes,
            computeMemoryBytes = hip?.ComputeMemoryBytes, freeDeviceBytes = options.CpuOnly ? (long?)null : FreeBytes() });
        return allPass;
    }

    private static List<JsonObject> ReadRecords(string path)
    {
        using var reader = new StreamReader(new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
        var records = new List<JsonObject>();
        while (reader.ReadLine() is { } line)
            records.Add(JsonNode.Parse(line) as JsonObject ?? throw new InvalidDataException("Expected one JSON object per line"));
        return records;
    }
    private static string Text(JsonObject record, string key) => record[key]?.GetValue<string>() ?? throw new InvalidDataException($"Missing {key}");
    private static int Number(JsonObject record, string key) => record[key]?.GetValue<int>() ?? throw new InvalidDataException($"Missing {key}");
    private static long Bytes(JsonObject record, string key) => record[key]?.GetValue<long>() ?? throw new InvalidDataException($"Missing {key}");
    private static double Real(JsonObject record, string key)
    {
        double value = record[key]?.GetValue<double>() ?? throw new InvalidDataException($"Missing {key}");
        Require(double.IsFinite(value), $"Non-finite {key}"); return value;
    }
    private static bool Flag(JsonObject record, string key) => record[key]?.GetValue<bool>() ?? throw new InvalidDataException($"Missing {key}");
    private static JsonObject Single(List<JsonObject> records, string kind, string? at = null) => records.Single(r => string.Equals(Text(r, "kind"), kind, StringComparison.Ordinal)
        && (at is null || string.Equals(Text(r, "at"), at, StringComparison.Ordinal)));
    private static void Near(double expected, double actual, string message) => Require(Math.Abs(expected - actual) <= 1e-10 * Math.Max(1, Math.Abs(expected)), message);

    private static void Validate(List<JsonObject> records, string output)
    {
        Require(!records.Any(r => Text(r, "kind") is "error"), "Artifact records runtime failure");
        var protocol = Single(records, "protocol");
        Require(Number(protocol, "version") == 1 && string.Equals(Text(protocol, "modelSha256"), ModelHash, StringComparison.Ordinal)
            && string.Equals(Text(protocol, "corpusSha256"), CorpusHash, StringComparison.Ordinal)
            && string.Equals(Text(protocol, "tokenSha256"), TokenHash, StringComparison.Ordinal), "Pinned provenance differs");
        Require(Real(protocol, "relativeL2Gate") == .02 && Real(protocol, "klGate") == .002 && Number(protocol, "window") == Window
            && Number(protocol, "prefillBatch") == 32 && Number(protocol, "chapterStartCharacter") == 1467, "Original protocol changed");
        int maxContext = Number(protocol, "maxContext"); Require(Contexts.Contains(maxContext), "Invalid maximum context");
        bool cpuOnly = Text(protocol, "mode") is "cpu-only";
        Require((Text(protocol, "mode") is "cpu-only" or "hip") && Flag(protocol, "hipSkipped") == cpuOnly, "Wrong execution mode");
        Require(Number(protocol, "hiddenSize") == 576 && Number(protocol, "numLayers") == 30 && Number(protocol, "numAttentionHeads") == 9
            && Number(protocol, "numKvHeads") == 3 && Number(protocol, "headDim") == 64 && Number(protocol, "vocabSize") == 49152, "Pinned geometry differs");
        var assemblies = protocol["managedAssemblies"]!.AsArray();
        Require(assemblies.Count == 7 && assemblies.All(a => a is JsonObject assembly && IsHash(Text(assembly, "sha256"))
            && !string.IsNullOrEmpty(Text(assembly, "informationalVersion"))), "Managed assembly provenance missing");
        Require(cpuOnly ? protocol["nativeCodeObjectSha256"] is null : IsHash(Text(protocol, "nativeCodeObjectSha256")), "Native code-object provenance differs");
        int[] tokens = JsonSerializer.Deserialize<int[]>(File.ReadAllText(Path.Combine(output, "tokens.json"))) ?? [];
        Require(tokens.Length == 8193 && string.Equals(HashTokens(tokens), TokenHash, StringComparison.Ordinal), "Pinned token artifact differs");
        int[] expectedContexts = Contexts.Where(c => c <= maxContext).ToArray();
        Require(protocol["boundaries"]!.AsArray().Select(n => n!.GetValue<int>()).SequenceEqual(expectedContexts), "Wrong protocol boundaries");
        var checkpoints = records.Where(r => Text(r, "kind") is "checkpoint").ToArray();
        Require(checkpoints.Select(r => Number(r, "context")).SequenceEqual(expectedContexts), "Missing actual context checkpoints");
        string[] names = cpuOnly ? ["cpuQ8VsCpuF32"] : GpuComparisons;
        var allRows = records.Where(r => Text(r, "kind") is "row").ToArray();
        Require(allRows.Length == expectedContexts.Length * Window * names.Length, "Wrong total diagnostic row count");
        bool allPass = true;
        foreach (var checkpoint in checkpoints)
        {
            int context = Number(checkpoint, "context");
            Require(Number(checkpoint, "cpuF32Length") == context && Number(checkpoint, "cpuQ8Length") == context
                && (cpuOnly || Number(checkpoint, "hipF16Length") == context && Number(checkpoint, "hipQ8Length") == context), "Cache did not reach real context");
            Require(Number(checkpoint, "firstScoredInputPosition") == context - Window && Number(checkpoint, "lastScoredInputPosition") == context - 1
                && Number(checkpoint, "lastTargetPosition") == context, "Wrong scoring window");
            foreach (string name in names)
            {
                var metric = checkpoint["metrics"]?[name]?.AsObject() ?? throw new InvalidDataException("Missing comparison");
                var rows = allRows.Where(r => string.Equals(Text(r, "comparison"), name, StringComparison.Ordinal)
                    && Number(r, "position") >= context - Window && Number(r, "position") < context).ToArray();
                Require(rows.Select(r => Number(r, "position")).SequenceEqual(Enumerable.Range(context - Window, Window)), "Wrong scored positions");
                Require(rows.All(r => Number(r, "target") == tokens[Number(r, "position") + 1]), "Wrong target token");
                Require(Number(metric, "scored") == Window, "Wrong scored count");
                Near(rows.Max(r => Real(r, "relativeL2")), Real(metric, "maxRelativeL2"), "L2 aggregate differs");
                Near(Math.Max(0, rows.Max(r => Real(r, "kl"))), Real(metric, "maxKl"), "KL aggregate differs");
                Near(rows.Max(r => Real(r, "maxAbs")), Real(metric, "maxAbs"), "Absolute-error aggregate differs");
                Near(Math.Exp(rows.Sum(r => Real(r, "nllReference")) / Window), Real(metric, "pplReference"), "Reference PPL aggregate differs");
                Near(Math.Exp(rows.Sum(r => Real(r, "nllActual")) / Window), Real(metric, "pplActual"), "Actual PPL aggregate differs");
                Near(100 * (Math.Exp(rows.Sum(r => Real(r, "nllActual") - Real(r, "nllReference")) / Window) - 1), Real(metric, "pplDeltaPercent"), "PPL delta aggregate differs");
                int matches = rows.Count(r => r["referenceTop"]![0]!["token"]!.GetValue<int>() == r["actualTop"]![0]!["token"]!.GetValue<int>());
                Require(Number(metric, "top1Matches") == matches, "Top1 aggregate differs");
                bool pass = Real(metric, "maxRelativeL2") < .02 && Real(metric, "maxKl") < .002;
                Require(Flag(metric, "gatePassed") == pass, "Original numerical gate changed"); allPass &= pass;
            }
            if (!cpuOnly)
            {
                long tableBytes = context / 16 * 2L * 30 * 8;
                Require(context * 23040L + tableBytes == Bytes(checkpoint, "hipF16Bytes"), "Owned F16 KV geometry differs");
                Require(context * 12240L + tableBytes == Bytes(checkpoint, "hipQ8Bytes"), "Owned Q8 KV geometry differs");
                Require(Number(checkpoint, "hipF16Pages") == context / 16 && Number(checkpoint, "hipQ8Pages") == context / 16, "Wrong page count");
            }
            Require(Bytes(checkpoint, "cpuF32PopulatedPayloadBytes") == context * 46080L
                && Bytes(checkpoint, "cpuQ8PopulatedPayloadBytes") == context * 12240L, "CPU populated KV payload differs");
        }
        var empty = Single(records, "memory", "empty"); var reset = Single(records, "memory", "reset"); var disposed = Single(records, "memory", "disposed");
        Require(Flag(empty, "hipSkipped") == cpuOnly && Flag(disposed, "hipSkipped") == cpuOnly, "HIP skip provenance differs");
        Require(Number(empty, "cpuF32Length") == 0 && Number(empty, "cpuQ8Length") == 0 && Number(reset, "cpuF32Length") == 0 && Number(reset, "cpuQ8Length") == 0, "CPU cache lifecycle differs");
        if (!cpuOnly) Require(Real(empty, "hipF16Bytes") == 0 && Real(empty, "hipQ8Bytes") == 0 && Real(reset, "hipF16Bytes") == 0
            && Real(reset, "hipQ8Bytes") == 0 && Real(disposed, "weightMemoryBytes") == 0 && Real(disposed, "computeMemoryBytes") == 0, "Owned GPU allocations not released");
        var negative = Single(records, "negativeControl")["result"]!.AsObject();
        Require(Number(negative, "scored") == 1 && !Flag(negative, "gatePassed") && (Real(negative, "maxRelativeL2") >= .02 || Real(negative, "maxKl") >= .002), "Numerical substitution survived");
        var summary = Single(records, "summary");
        Require(Flag(summary, "completed") && Flag(summary, "lifecycleComplete") && Flag(summary, "negativeControlKilled")
            && Number(summary, "scoredPerComparison") == expectedContexts.Length * Window && Flag(summary, "allOriginalGatesPassed") == allPass
            && Number(summary, "exitCode") == (allPass ? 0 : 10), "Incomplete or inconsistent run summary");
    }
    private static bool IsHash(string value) => value.Length == 64 && value.All(Uri.IsHexDigit);
}
