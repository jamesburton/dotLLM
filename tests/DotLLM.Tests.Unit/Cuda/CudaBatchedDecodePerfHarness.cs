using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in timing harness for issue #251 — compares wall-clock decode throughput of the
/// per-sequence <c>Forward</c> loop (the interface's default <c>ForwardBatch</c> fallback,
/// i.e. what CUDA did before this issue) against <see cref="CudaTransformerModel.ForwardBatch"/>'s
/// deferred-host-sync implementation, for the same N concurrent decoding sequences.
/// </summary>
/// <remarks>
/// Correctness is covered by <c>CudaBatchedDecodeTests</c>; this harness only reports timing and
/// does not gate on an absolute speedup threshold (GPU wall-clock timing is noisy across
/// machines/driver versions/thermal state) — it prints both wall times and the ratio so a human
/// can judge whether the deferred-sync design is winning as expected. Set
/// <c>DOTLLM_CUDA_PERF=1</c> to run.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaBatchedDecodePerfHarness
{
    private readonly ITestOutputHelper _output;

    public CudaBatchedDecodePerfHarness(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts =
    {
        "The capital of France is Paris. The capital of Germany is",
        "Once upon a time there was a",
        "2 + 2 = 4. 3 + 3 =",
        "The quick brown fox jumps over the",
    };

    [SkippableFact]
    public unsafe void MeasureBatchedVsPerSequenceDecodeThroughput()
    {
        Skip.IfNot(
            string.Equals(Environment.GetEnvironmentVariable("DOTLLM_CUDA_PERF"), "1", StringComparison.Ordinal),
            "DOTLLM_CUDA_PERF=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int n = Prompts.Length;
        var promptTokens = new int[n][];
        for (int i = 0; i < n; i++) promptTokens[i] = tokenizer.Encode(Prompts[i]);
        int[] promptLens = new int[n];
        for (int i = 0; i < n; i++) promptLens[i] = promptTokens[i].Length;

        const int warmupSteps = 4;
        const int decodeSteps = 32;

        double perSeqMs = RunPerSequenceLoop(gguf, config, promptTokens, promptLens, warmupSteps, decodeSteps);
        double batchedMs = RunForwardBatch(gguf, config, promptTokens, promptLens, warmupSteps, decodeSteps);

        _output.WriteLine("=== summary ===");
        _output.WriteLine($"sequences={n} decode_steps={decodeSteps} (warmup={warmupSteps} excluded)");
        _output.WriteLine($"per_sequence_loop_total_ms={perSeqMs:F2}");
        _output.WriteLine($"forward_batch_total_ms={batchedMs:F2}");
        _output.WriteLine($"speedup(per_seq/batched)={(batchedMs > 0 ? perSeqMs / batchedMs : 0):F3}x");
    }

    private static double RunPerSequenceLoop(
        GgufFile gguf, DotLLM.Core.Models.ModelConfig config, int[][] promptTokens, int[] promptLens,
        int warmupSteps, int decodeSteps)
    {
        int n = promptTokens.Length;
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        var caches = new CudaKvCache[n];
        var curTok = new int[n];
        for (int i = 0; i < n; i++)
        {
            caches[i] = model.CreateKvCache(promptLens[i] + warmupSteps + decodeSteps + 8);
            using var prefillLogits = model.Forward(promptTokens[i], Positions(promptLens[i]), 0, caches[i]);
            curTok[i] = Argmax(prefillLogits);
        }

        for (int step = 0; step < warmupSteps; step++)
            for (int i = 0; i < n; i++)
            {
                var tokBuf = new[] { curTok[i] };
                var posBuf = new[] { promptLens[i] + step };
                using var t = model.Forward(tokBuf, posBuf, 0, caches[i]);
                curTok[i] = Argmax(t);
            }

        var sw = Stopwatch.StartNew();
        for (int step = 0; step < decodeSteps; step++)
            for (int i = 0; i < n; i++)
            {
                var tokBuf = new[] { curTok[i] };
                var posBuf = new[] { promptLens[i] + warmupSteps + step };
                using var t = model.Forward(tokBuf, posBuf, 0, caches[i]);
                curTok[i] = Argmax(t);
            }
        sw.Stop();

        foreach (var c in caches) c.Dispose();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static double RunForwardBatch(
        GgufFile gguf, DotLLM.Core.Models.ModelConfig config, int[][] promptTokens, int[] promptLens,
        int warmupSteps, int decodeSteps)
    {
        int n = promptTokens.Length;
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        var caches = new CudaKvCache[n];
        var curTok = new int[n];
        for (int i = 0; i < n; i++)
        {
            caches[i] = model.CreateKvCache(promptLens[i] + warmupSteps + decodeSteps + 8);
            using var prefillLogits = model.Forward(promptTokens[i], Positions(promptLens[i]), 0, caches[i]);
            curTok[i] = Argmax(prefillLogits);
        }

        for (int step = 0; step < warmupSteps; step++)
            RunOneBatchedStep(model, caches, curTok, promptLens, step);

        var sw = Stopwatch.StartNew();
        for (int step = 0; step < decodeSteps; step++)
            RunOneBatchedStep(model, caches, curTok, promptLens, warmupSteps + step);
        sw.Stop();

        foreach (var c in caches) c.Dispose();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static void RunOneBatchedStep(
        CudaTransformerModel model, CudaKvCache[] caches, int[] curTok, int[] promptLens, int step)
    {
        int n = caches.Length;
        var requests = new SequenceForwardRequest[n];
        for (int i = 0; i < n; i++)
        {
            requests[i] = new SequenceForwardRequest
            {
                TokenIds = new[] { curTok[i] },
                Positions = new[] { promptLens[i] + step },
                KvCache = caches[i],
            };
        }

        var results = model.ForwardBatch(requests, 0);
        for (int i = 0; i < n; i++)
        {
            curTok[i] = Argmax(results[i]);
            results[i].Dispose();
        }
    }

    private static int[] Positions(int count)
    {
        int[] positions = new int[count];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;
        return positions;
    }

    private static unsafe int Argmax(DotLLM.Core.Tensors.ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0;
        float best = span[0];
        for (int i = 1; i < n; i++)
        {
            if (span[i] > best) { best = span[i]; idx = i; }
        }
        return idx;
    }
}
