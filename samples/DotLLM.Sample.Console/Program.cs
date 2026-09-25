using DotLLM.Core.Configuration;
using DotLLM.Core.Backends;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: dotLLM.Sample.Console <model.gguf> [prompt] [--greedy] [--max N] [--threads N]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  prompt      Text prompt (default: \"The capital of France is\")");
    Console.Error.WriteLine("  --greedy    Disable temperature/top-k/top-p — argmax sampling for parity tests");
    Console.Error.WriteLine("  --max N     Max tokens to generate (default 128)");
    Console.Error.WriteLine("  --threads N Number of CPU compute threads (default 1 = single-threaded).");
    Console.Error.WriteLine("              Currently honoured for Qwen3MoeHybrid only; other architectures");
    Console.Error.WriteLine("              still go through the IModelArchitecture factory.");
    return 1;
}

string modelPath = args[0];
bool greedy = args.Contains("--greedy");
int maxTokens = 128;
int threadCount = 1;
for (int i = 0; i < args.Length - 1; i++)
{
    if (args[i] == "--max" && int.TryParse(args[i + 1], out var v)) maxTokens = v;
    if (args[i] == "--threads" && int.TryParse(args[i + 1], out var t)) threadCount = t;
}
string prompt = args.Length > 1 && !args[1].StartsWith("--")
    ? string.Join(' ', args.Skip(1).TakeWhile(a => !a.StartsWith("--")))
    : "The capital of France is";

bool inspect = args.Contains("--inspect");

Console.WriteLine($"Loading model: {modelPath}");
using var gguf = GgufFile.Open(modelPath);

if (inspect)
{
    // Dump per-tensor metadata grouped by quantization type — used to scope quantized-kernel work.
    Console.WriteLine($"Total tensors: {gguf.Tensors.Count}");
    var byType = gguf.Tensors.GroupBy(t => t.QuantizationType)
        .OrderBy(g => g.Key.ToString())
        .Select(g => (Type: g.Key, Count: g.Count(), Sample: g.First().Name))
        .ToList();
    Console.WriteLine();
    Console.WriteLine($"{"Type",-12} {"Count",6}  Sample tensor");
    foreach (var (type, count, sample) in byType)
        Console.WriteLine($"{type,-12} {count,6}  {sample}");

    // Drill down: what quant types do the MoE expert tensors use?
    Console.WriteLine();
    Console.WriteLine("=== MoE expert tensor quant types (filtered by name) ===");
    string[] moePatterns = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
                            "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"];
    foreach (var pat in moePatterns)
    {
        var matching = gguf.Tensors.Where(t => t.Name.Contains(pat)).ToList();
        if (matching.Count == 0) { Console.WriteLine($"{pat}: (none)"); continue; }
        var qts = matching.GroupBy(t => t.QuantizationType)
            .Select(g => $"{g.Key}×{g.Count()}")
            .ToList();
        var first = matching.First();
        Console.WriteLine($"{pat,-18} {matching.Count} tensors, shape={first.Shape}, types: {string.Join(", ", qts)}");
    }
    return 0;
}

var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

// Threading: the IModelArchitecture factory does not yet expose ThreadingConfig — for the
// Qwen3MoeHybrid architecture (qwen35moe) we therefore dispatch directly so the user can
// opt into parallel compute with --threads N. All other architectures continue through
// the factory (single-threaded) until threading is plumbed into IModelArchitecture.
IDisposable LoadModel()
{
    if (config.Architecture == Architecture.Qwen3MoeHybrid && threadCount > 1)
    {
        var threading = new ThreadingConfig(ThreadCount: threadCount);
        Console.WriteLine($"Threading: {threadCount} compute threads (Qwen3MoeHybrid)");
        return Qwen3MoeHybridTransformerModel.LoadFromGguf(gguf, config, threading);
    }
    var factory = new TransformerArchitecture(gguf);
    return (IDisposable)factory.CreateModel(config, backend: null!);
}
using var model = LoadModel();
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

Console.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.VocabSize} vocab");
Console.WriteLine($"Prompt: \"{prompt}\"");
Console.WriteLine();

var generator = new TextGenerator((DotLLM.Core.Models.IModel)model, tokenizer);

// --- Composable sampling pipeline ---
// Greedy = TopK(1) → deterministic argmax (effective regardless of any preceding temperature).
var samplerSteps = greedy
    ? new DotLLM.Core.Sampling.ISamplerStep[] { new TopKSampler(1) }
    : new DotLLM.Core.Sampling.ISamplerStep[]
    {
        new TemperatureSampler(0.8f),
        new TopKSampler(40),
        new TopPSampler(0.95f),
        new MinPSampler(0.05f)
    };
var options = new InferenceOptions
{
    SamplerSteps = samplerSteps,
    StopConditions =
    [
        new EosStopCondition(tokenizer.EosTokenId),
        new MaxTokensStopCondition(maxTokens)
    ],
    Seed = 42,
    MaxTokens = maxTokens
};

// --- Streaming generation via IAsyncEnumerable ---
Console.Write(prompt);

InferenceTimings timings = default;
int tokenCount = 0;

await foreach (var token in generator.GenerateStreamingTokensAsync(prompt, options))
{
    Console.Write(token.Text);
    tokenCount++;
    if (token.Timings.HasValue)
        timings = token.Timings.Value;
}

Console.WriteLine();
Console.WriteLine();
Console.WriteLine($"[Prompt tokens: {timings.PrefillTokenCount}, Generated: {tokenCount}, " +
    $"Decode steps: {timings.DecodeTokenCount}]");
Console.WriteLine($"[Prefill: {timings.PrefillTimeMs:F1} ms ({timings.PrefillTokensPerSec:F1} tok/s), " +
    $"Decode: {timings.DecodeTimeMs:F1} ms ({timings.DecodeTokensPerSec:F1} tok/s), " +
    $"Sampling: {timings.SamplingTimeMs:F1} ms]");

return 0;
