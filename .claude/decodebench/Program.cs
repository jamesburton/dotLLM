using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;

// Decode launch-overhead measurement harness.
//
// Usage:
//   DecodeBench cuda  <gguf>   -> CUDA decode tok/s, graph ON vs OFF interleaved (per-instance toggle).
//   DecodeBench vulkan <gguf>  -> Vulkan decode tok/s + record-vs-submit split (DOTLLM_VULKAN_PROFILE_SUBMIT).
//
// Drives model.Forward directly (greedy argmax on host) so the per-token
// workload is byte-identical across modes. Prefill once, then time N decode
// tokens; interleave modes in one warmed process, report min-of-blocks.

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: DecodeBench <cuda|vulkan> <gguf-path> [decodeTokens] [blocks]");
    return 1;
}

string backend = args[0].ToLowerInvariant();
string modelPath = args[1];
int decodeTokens = args.Length > 2 ? int.Parse(args[2]) : 128;
int blocks = args.Length > 3 ? int.Parse(args[3]) : 6;

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"model not found: {modelPath}");
    return 1;
}

var gguf = GgufFile.Open(modelPath);
var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
Console.WriteLine($"Model: {Path.GetFileName(modelPath)}  layers={config.NumLayers} hidden={config.HiddenSize} vocab={config.VocabSize}");

// Short fixed prompt token ids (no tokenizer dependency on chat template).
// Just a handful of valid ids; content is irrelevant for timing.
int[] promptTokens = { 1, 2, 3, 4, 5, 6, 7, 8 };

if (backend == "cuda")
    return RunCuda(gguf, config, promptTokens, decodeTokens, blocks);
if (backend == "vulkan")
    return RunVulkan(gguf, config, promptTokens, decodeTokens, blocks);

Console.Error.WriteLine($"unknown backend: {backend}");
return 1;

// ───────────────────────── CUDA ─────────────────────────
static int RunCuda(GgufFile gguf, ModelConfig config,
    int[] promptTokens, int decodeTokens, int blocks)
{
    var model = DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, 0);
    Console.WriteLine($"CUDA loaded. Default UseGraphCapture={model.UseGraphCapture}");
    Console.WriteLine($"CUDA DisableGraphCapture(static)={DotLLM.Cuda.CudaTransformerModel.DisableGraphCapture}");

    // Warm both modes once each (JIT + first-time graph capture).
    Warm(() => model.UseGraphCapture = true, model, gguf, config, promptTokens, decodeTokens / 2);
    Warm(() => model.UseGraphCapture = false, model, gguf, config, promptTokens, decodeTokens / 2);

    var onMs = new List<double>();
    var offMs = new List<double>();
    for (int b = 0; b < blocks; b++)
    {
        model.UseGraphCapture = true;
        onMs.Add(MeasureDecode(model, gguf, config, promptTokens, decodeTokens));
        model.UseGraphCapture = false;
        offMs.Add(MeasureDecode(model, gguf, config, promptTokens, decodeTokens));
    }

    double onMin = onMs.Min(), offMin = offMs.Min();
    Console.WriteLine();
    Console.WriteLine($"=== CUDA decode {decodeTokens} tok, {blocks} blocks (min-of-block) ===");
    Report("graph ON ", onMin, decodeTokens);
    Report("graph OFF", offMin, decodeTokens);
    double onTps = decodeTokens / (onMin / 1000.0), offTps = decodeTokens / (offMin / 1000.0);
    Console.WriteLine($"speedup (ON vs OFF): {onTps / offTps:F3}x   ({onTps - offTps:+0.0;-0.0} tok/s)");
    Console.WriteLine($"per-token ON={onMin / decodeTokens:F3}ms OFF={offMin / decodeTokens:F3}ms  saved/token={(offMin - onMin) / decodeTokens:F3}ms");
    model.Dispose();
    return 0;
}

// ───────────────────────── Vulkan ─────────────────────────
static int RunVulkan(GgufFile gguf, ModelConfig config,
    int[] promptTokens, int decodeTokens, int blocks)
{
    string spvDir = Path.Combine(AppContext.BaseDirectory, "spv");
    var model = DotLLM.Vulkan.VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
    Console.WriteLine("Vulkan loaded.");

    Warm(null, model, gguf, config, promptTokens, 8);

    var ms = new List<double>();
    DotLLM.Vulkan.VulkanTransformerModel.ResetProfile();
    for (int b = 0; b < blocks; b++)
    {
        ms.Add(MeasureDecode(model, gguf, config, promptTokens, decodeTokens));
        // Brief idle between blocks to avoid the Arc iGPU TDR watchdog tripping
        // on sustained back-to-back submit/wait load (DEVICE_LOST).
        System.Threading.Thread.Sleep(150);
    }

    double min = ms.Min();
    Console.WriteLine();
    Console.WriteLine($"=== Vulkan decode {decodeTokens} tok, {blocks} blocks (min-of-block) ===");
    Report("decode   ", min, decodeTokens);

    long fwd = DotLLM.Vulkan.VulkanTransformerModel.ProfileForwardCount;
    if (fwd > 0)
    {
        double recMs = DotLLM.Vulkan.VulkanTransformerModel.ProfileRecordTicks * 1000.0 / Stopwatch.Frequency / fwd;
        double subMs = DotLLM.Vulkan.VulkanTransformerModel.ProfileSubmitTicks * 1000.0 / Stopwatch.Frequency / fwd;
        double subWaitMs = DotLLM.Vulkan.VulkanTransformerModel.ProfileSubmitWaitTicks * 1000.0 / Stopwatch.Frequency / fwd;
        double d2hMs = subMs - subWaitMs;
        Console.WriteLine();
        Console.WriteLine($"=== Vulkan per-token CPU record vs submit split (n={fwd} forwards) ===");
        Console.WriteLine($"  record (CPU vkCmd*)    : {recMs:F3} ms/token");
        Console.WriteLine($"  submit+fence-wait (GPU): {subWaitMs:F3} ms/token");
        Console.WriteLine($"  logits D2H + softcap   : {d2hMs:F3} ms/token");
        Console.WriteLine($"  record fraction        : {recMs / (recMs + subMs) * 100:F1}%");
        Console.WriteLine($"  -> record-once upper-bound saving if record went to ~0: {recMs:F3} ms/token");
    }
    model.Dispose();
    return 0;
}

// ───────────────────────── shared ─────────────────────────
static void Warm(Action? setMode, IModel model, GgufFile gguf,
    ModelConfig config, int[] promptTokens, int decodeTokens)
{
    setMode?.Invoke();
    MeasureDecode(model, gguf, config, promptTokens, decodeTokens);
}

static unsafe double MeasureDecode(IModel model, GgufFile gguf,
    ModelConfig config, int[] promptTokens, int decodeTokens)
{
    int maxSeq = promptTokens.Length + decodeTokens + 8;
    IKvCacheFactory factory = KvCacheFactory.For(model);
    var kv = factory.Create(maxSeq);

    // Prefill the prompt (single Forward over all prompt tokens).
    var prefillPos = new int[promptTokens.Length];
    for (int i = 0; i < prefillPos.Length; i++) prefillPos[i] = i;
    int nextTok;
    using (var logits = model.Forward(promptTokens, prefillPos, 0, kv))
        nextTok = Argmax(logits, config.VocabSize);

    int pos = promptTokens.Length;
    Span<int> one = stackalloc int[1];
    Span<int> onePos = stackalloc int[1];

    var sw = Stopwatch.StartNew();
    for (int t = 0; t < decodeTokens; t++)
    {
        one[0] = nextTok;
        onePos[0] = pos++;
        using var logits = model.Forward(one, onePos, 0, kv);
        nextTok = Argmax(logits, config.VocabSize);
    }
    sw.Stop();
    kv.Dispose();
    return sw.Elapsed.TotalMilliseconds;
}

static unsafe int Argmax(ITensor logits, int vocab)
{
    float* p = (float*)logits.DataPointer;
    int best = 0; float bv = p[0];
    for (int i = 1; i < vocab; i++) { if (p[i] > bv) { bv = p[i]; best = i; } }
    // Clamp to a safe non-special id to keep decode going on degenerate logits.
    if (best <= 0) best = 5;
    return best;
}

static void Report(string label, double ms, int tokens)
    => Console.WriteLine($"  {label}: {ms:F2} ms total  {tokens / (ms / 1000.0):F2} tok/s  {ms / tokens:F3} ms/token");

interface IKvCacheFactory { IKvCache Create(int maxSeq); }

static class KvCacheFactory
{
    public static IKvCacheFactory For(IModel model) => model switch
    {
        DotLLM.Cuda.CudaTransformerModel c => new CudaFac(c),
        DotLLM.Vulkan.VulkanTransformerModel v => new VkFac(v),
        _ => throw new NotSupportedException(model.GetType().Name)
    };

    sealed class CudaFac(DotLLM.Cuda.CudaTransformerModel m) : IKvCacheFactory
    { public IKvCache Create(int n) => m.CreateKvCache(n); }
    sealed class VkFac(DotLLM.Vulkan.VulkanTransformerModel m) : IKvCacheFactory
    { public IKvCache Create(int n) => m.CreateKvCache(n); }
}
