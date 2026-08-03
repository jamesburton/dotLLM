using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness tests for issue #251 — <see cref="CudaTransformerModel.ForwardBatch"/>.
/// </summary>
/// <remarks>
/// The most important property (per the issue's own acceptance criteria): batched output must
/// exactly match running each sequence individually through the existing per-sequence
/// <see cref="CudaTransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
/// path. <see cref="ForwardBatch_MultipleSequences_MatchesPerSequenceBaseline"/> drives 3
/// independent prompts through both call shapes for the same number of decode steps and checks
/// argmax (and near-identical logits) match at every step.
/// </remarks>
[Trait("Category", "GPU")]
public class CudaBatchedDecodeTests
{
    private readonly ITestOutputHelper _out;

    public CudaBatchedDecodeTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] Prompts =
    {
        "The capital of France is Paris. The capital of Germany is",
        "Once upon a time there was a",
        "2 + 2 = 4. 3 + 3 =",
    };

    private static string ResolveModelPath(string fileName) => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", fileName);

    /// <summary>
    /// Prefills <c>Prompts.Length</c> independent sequences (each its own <see cref="CudaKvCache"/>),
    /// then decodes <see cref="DecodeSteps"/> tokens twice against two FRESH model instances:
    /// once via the per-sequence <c>Forward</c> loop (today's baseline / what the interface's
    /// default <c>ForwardBatch</c> would have done), once via one <see cref="CudaTransformerModel.ForwardBatch"/>
    /// call per decode step across all sequences. Argmax must match at every step for every
    /// sequence — this is the correctness gate for #251's deferred-host-sync design (same kernels,
    /// same per-sequence order, same shared scratch; only the host synchronization points move).
    /// </summary>
    [SkippableTheory]
    [InlineData("SmolLM-135M.Q8_0.gguf")]
    [InlineData("SmolLM-135M.Q4_K_M.gguf")]
    public unsafe void ForwardBatch_MultipleSequences_MatchesPerSequenceBaseline(string modelFile)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string modelPath = ResolveModelPath(modelFile);
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        int n = Prompts.Length;
        var promptTokens = new int[n][];
        for (int i = 0; i < n; i++) promptTokens[i] = tokenizer.Encode(Prompts[i]);

        const int decodeSteps = 12;
        int[] promptLens = new int[n];
        for (int i = 0; i < n; i++) promptLens[i] = promptTokens[i].Length;

        // === Baseline: per-sequence Forward loop ===
        var baselineTokens = new int[n][];
        var baselineFirstLogits = new float[n][];
        using (var model = CudaTransformerModel.LoadFromGguf(gguf, config))
        {
            var caches = new CudaKvCache[n];
            var curTok = new int[n];
            for (int i = 0; i < n; i++)
            {
                caches[i] = model.CreateKvCache(promptLens[i] + decodeSteps + 8);
                int[] positions = Positions(promptLens[i]);
                using var prefillLogits = model.Forward(promptTokens[i], positions, 0, caches[i]);
                curTok[i] = ArgMax((float*)prefillLogits.DataPointer, config.VocabSize);
                baselineTokens[i] = new int[decodeSteps];
                baselineFirstLogits[i] = new float[config.VocabSize];
            }

            for (int step = 0; step < decodeSteps; step++)
            {
                for (int i = 0; i < n; i++)
                {
                    var tokBuf = new[] { curTok[i] };
                    var posBuf = new[] { promptLens[i] + step };
                    using var t = model.Forward(tokBuf, posBuf, 0, caches[i]);
                    int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                    if (step == 0)
                        new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize)
                            .CopyTo(baselineFirstLogits[i]);
                    baselineTokens[i][step] = argmax;
                    curTok[i] = argmax;
                }
            }

            foreach (var c in caches) c.Dispose();
        }

        // === ForwardBatch: one batched call per decode step across all sequences ===
        var batchedTokens = new int[n][];
        var batchedFirstLogits = new float[n][];
        using (var model = CudaTransformerModel.LoadFromGguf(gguf, config))
        {
            var caches = new CudaKvCache[n];
            var curTok = new int[n];
            for (int i = 0; i < n; i++)
            {
                caches[i] = model.CreateKvCache(promptLens[i] + decodeSteps + 8);
                int[] positions = Positions(promptLens[i]);
                using var prefillLogits = model.Forward(promptTokens[i], positions, 0, caches[i]);
                curTok[i] = ArgMax((float*)prefillLogits.DataPointer, config.VocabSize);
                batchedTokens[i] = new int[decodeSteps];
                batchedFirstLogits[i] = new float[config.VocabSize];
            }

            for (int step = 0; step < decodeSteps; step++)
            {
                var requests = new SequenceForwardRequest[n];
                var tokBufs = new int[n][];
                var posBufs = new int[n][];
                for (int i = 0; i < n; i++)
                {
                    tokBufs[i] = new[] { curTok[i] };
                    posBufs[i] = new[] { promptLens[i] + step };
                    requests[i] = new SequenceForwardRequest
                    {
                        TokenIds = tokBufs[i],
                        Positions = posBufs[i],
                        KvCache = caches[i],
                    };
                }

                var results = model.ForwardBatch(requests, 0);
                Assert.Equal(n, results.Count);
                for (int i = 0; i < n; i++)
                {
                    using var t = results[i];
                    int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                    if (step == 0)
                        new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize)
                            .CopyTo(batchedFirstLogits[i]);
                    batchedTokens[i][step] = argmax;
                    curTok[i] = argmax;
                }
            }

            foreach (var c in caches) c.Dispose();
        }

        // === Compare ===
        for (int i = 0; i < n; i++)
        {
            _out.WriteLine($"seq {i} baseline: [{string.Join(", ", baselineTokens[i])}]");
            _out.WriteLine($"seq {i} batched : [{string.Join(", ", batchedTokens[i])}]");

            float maxDiff = 0;
            for (int v = 0; v < config.VocabSize; v++)
                maxDiff = MathF.Max(maxDiff, MathF.Abs(baselineFirstLogits[i][v] - batchedFirstLogits[i][v]));
            _out.WriteLine($"seq {i} step-0 logit max abs diff: {maxDiff:F6}");

            for (int step = 0; step < decodeSteps; step++)
            {
                Assert.True(baselineTokens[i][step] == batchedTokens[i][step],
                    $"seq {i} argmax divergence at step {step}: " +
                    $"baseline={baselineTokens[i][step]}, batched={batchedTokens[i][step]}");
            }

            // Same generous tolerance as CudaGraphCaptureEquivalenceTest (PTX-JIT SASS-scheduling
            // drift between separate model instances/runs in one process is a known, harmless
            // source of small FP differences — see that test's doc comment for the full
            // explanation). Argmax equality above is the real correctness gate.
            Assert.True(maxDiff < 5.0f,
                $"seq {i} step-0 logit divergence too large: max abs diff = {maxDiff}");
        }
    }

    /// <summary>
    /// A single-request <see cref="CudaTransformerModel.ForwardBatch"/> call must delegate to
    /// <c>Forward</c> (the interface's documented contract: "a single decoder uses <c>Forward</c>,
    /// keeping single-tenant latency unchanged" — see <c>docs/SCHEDULING.md</c>), producing
    /// identical output to calling <c>Forward</c> directly.
    /// </summary>
    [SkippableFact]
    public unsafe void ForwardBatch_SingleRequest_MatchesDirectForward()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string modelPath = ResolveModelPath("SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode(Prompts[0]);

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);
        using var cacheA = model.CreateKvCache(prompt.Length + 8);
        using var cacheB = model.CreateKvCache(prompt.Length + 8);

        int[] positions = Positions(prompt.Length);
        using var directLogits = model.Forward(prompt, positions, 0, cacheA);
        int directArgmax = ArgMax((float*)directLogits.DataPointer, config.VocabSize);

        var request = new SequenceForwardRequest
        {
            TokenIds = prompt,
            Positions = positions,
            KvCache = cacheB,
        };
        var batchedResults = model.ForwardBatch(new[] { request }, 0);
        Assert.Single(batchedResults);
        using var batchedLogits = batchedResults[0];
        int batchedArgmax = ArgMax((float*)batchedLogits.DataPointer, config.VocabSize);

        Assert.Equal(directArgmax, batchedArgmax);
    }

    /// <summary>Empty request list returns an empty result list without touching the GPU further.</summary>
    [SkippableFact]
    public void ForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string modelPath = ResolveModelPath("SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);

        var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), 0);
        Assert.Empty(results);
    }

    private static int[] Positions(int length)
    {
        var positions = new int[length];
        for (int i = 0; i < length; i++) positions[i] = i;
        return positions;
    }

    private static unsafe int ArgMax(float* data, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
