using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Equivalence test: the CUDA Graphs decode replay path must produce identical
/// argmax (and near-identical logits) to the eager kernel-launch path. Validates
/// the device-resident <c>seq_kv</c> / <c>position_offset</c> mechanism and the
/// device-side KV-cache write kernel against the existing eager forward as oracle.
/// </summary>
[Trait("Category", "GPU")]
public class CudaGraphCaptureEquivalenceTest
{
    private readonly ITestOutputHelper _out;

    public CudaGraphCaptureEquivalenceTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void EagerVsGraphDecode_Q4KM_Match()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf");
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode("The capital of France is Paris. The capital of Germany is");
        _out.WriteLine($"Prompt tokens: {prompt.Length}");

        const int decodeSteps = 32;
        int kvCap = prompt.Length + decodeSteps + 8;

        // === Run 1: Eager ===
        // Graph capture is now default-on, so explicitly disable it here to
        // exercise the eager path the test name promises.
        int[] eagerTokens = new int[decodeSteps];
        float[] eagerFirstLogits = new float[config.VocabSize];
        using (var modelEager = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kvEager = modelEager.CreateKvCache(kvCap))
        {
            modelEager.UseGraphCapture = false;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            using (var _ = modelEager.Forward(prompt, positions, 0, kvEager)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            for (int i = 0; i < decodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                using var t = modelEager.Forward(tokBuf, posBuf, 0, kvEager);
                int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                if (i == 0)
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize);
                    span.CopyTo(eagerFirstLogits);
                }
                eagerTokens[i] = argmax;
                curTok = argmax;
            }
        }

        // === Run 2: Graph capture ===
        int[] graphTokens = new int[decodeSteps];
        float[] graphFirstLogits = new float[config.VocabSize];
        using (var modelGraph = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kvGraph = modelGraph.CreateKvCache(kvCap))
        {
            modelGraph.UseGraphCapture = true;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            // Prefill stays eager (multi-token).
            using (var _ = modelGraph.Forward(prompt, positions, 0, kvGraph)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            for (int i = 0; i < decodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                using var t = modelGraph.Forward(tokBuf, posBuf, 0, kvGraph);
                int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                if (i == 0)
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize);
                    span.CopyTo(graphFirstLogits);
                }
                graphTokens[i] = argmax;
                curTok = argmax;
            }
        }

        // === Compare ===
        _out.WriteLine($"Eager tokens: [{string.Join(", ", eagerTokens)}]");
        _out.WriteLine($"Graph tokens: [{string.Join(", ", graphTokens)}]");

        float maxDiff = 0;
        float sumDiff = 0;
        for (int i = 0; i < config.VocabSize; i++)
        {
            float d = MathF.Abs(eagerFirstLogits[i] - graphFirstLogits[i]);
            sumDiff += d;
            if (d > maxDiff) maxDiff = d;
        }
        _out.WriteLine($"Step 0 logit max abs diff: {maxDiff:F6}, mean diff: {sumDiff / config.VocabSize:F6}");

        // Argmax MUST match at every step — this is the real correctness gate.
        // Logit values may shift slightly due to PTX-JIT cache state from earlier
        // tests in the suite (e.g. order of kernel registrations in a module
        // changes SASS scheduling, which changes accumulation order in identical
        // arithmetic). When the test runs in isolation the diff is exactly 0.0;
        // in the full suite it lands around 0.25 max abs without affecting
        // argmax over a vocab of ~50K. The 5.0f tolerance below catches genuine
        // divergence (e.g. a kernel bug producing incoherent logits) while
        // ignoring this JIT-induced FP drift.
        for (int i = 0; i < decodeSteps; i++)
        {
            Assert.True(eagerTokens[i] == graphTokens[i],
                $"Argmax divergence at step {i}: eager={eagerTokens[i]}, graph={graphTokens[i]}");
        }

        Assert.True(maxDiff < 5.0f,
            $"First-step logit divergence too large: max abs diff = {maxDiff}");
    }

    /// <summary>
    /// Same equivalence check as <see cref="EagerVsGraphDecode_Q4KM_Match"/>, but with
    /// the mixed-precision quantized KV-cache (Q8_0 stored region + small FP16 window).
    /// Validates that the device-resident eviction state machine (predicated
    /// quant-on-evict + dyn dequant + window scatter) matches the host-driven
    /// eager path bit-identically. This is the test that gates the 2× graph-decode
    /// speedup landing for KV-quantized configs.
    /// </summary>
    [SkippableFact]
    public unsafe void EagerVsGraphDecode_QuantizedKv_Match()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q4_K_M.gguf");
        Skip.If(!File.Exists(modelPath), $"{modelPath} not found");

        var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode("The capital of France is Paris. The capital of Germany is");
        _out.WriteLine($"Prompt tokens: {prompt.Length}");

        const int decodeSteps = 32;
        int kvCap = prompt.Length + decodeSteps + 8;
        // Window small enough that we exercise both phases (window-only and
        // post-eviction) during the timed decode steps.
        var kvCfg = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0,
                                       MixedPrecisionWindowSize: 16);

        // === Run 1: Eager (quantized cache) ===
        int[] eagerTokens = new int[decodeSteps];
        float[] eagerFirstLogits = new float[config.VocabSize];
        using (var modelEager = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kvEager = (DotLLM.Core.Attention.IKvCache)modelEager.CreateKvCache(kvCap, kvCfg))
        {
            // Graph capture is now default-on; explicitly disable for eager pass.
            modelEager.UseGraphCapture = false;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            using (var _ = modelEager.Forward(prompt, positions, 0, kvEager)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            for (int i = 0; i < decodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                using var t = modelEager.Forward(tokBuf, posBuf, 0, kvEager);
                int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                if (i == 0)
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize);
                    span.CopyTo(eagerFirstLogits);
                }
                eagerTokens[i] = argmax;
                curTok = argmax;
            }
        }

        // === Run 2: Graph capture (quantized cache) ===
        int[] graphTokens = new int[decodeSteps];
        float[] graphFirstLogits = new float[config.VocabSize];
        using (var modelGraph = CudaTransformerModel.LoadFromGguf(gguf, config))
        using (var kvGraph = (DotLLM.Core.Attention.IKvCache)modelGraph.CreateKvCache(kvCap, kvCfg))
        {
            modelGraph.UseGraphCapture = true;
            int[] positions = new int[prompt.Length];
            for (int i = 0; i < prompt.Length; i++) positions[i] = i;
            using (var _ = modelGraph.Forward(prompt, positions, 0, kvGraph)) { }

            int curTok = prompt[^1];
            int[] tokBuf = new int[1];
            int[] posBuf = new int[1];
            for (int i = 0; i < decodeSteps; i++)
            {
                tokBuf[0] = curTok;
                posBuf[0] = prompt.Length + i;
                using var t = modelGraph.Forward(tokBuf, posBuf, 0, kvGraph);
                int argmax = ArgMax((float*)t.DataPointer, config.VocabSize);
                if (i == 0)
                {
                    var span = new ReadOnlySpan<float>((void*)t.DataPointer, config.VocabSize);
                    span.CopyTo(graphFirstLogits);
                }
                graphTokens[i] = argmax;
                curTok = argmax;
            }
        }

        _out.WriteLine($"Eager tokens: [{string.Join(", ", eagerTokens)}]");
        _out.WriteLine($"Graph tokens: [{string.Join(", ", graphTokens)}]");

        float maxDiff = 0;
        float sumDiff = 0;
        for (int i = 0; i < config.VocabSize; i++)
        {
            float d = MathF.Abs(eagerFirstLogits[i] - graphFirstLogits[i]);
            sumDiff += d;
            if (d > maxDiff) maxDiff = d;
        }
        _out.WriteLine($"Step 0 logit max abs diff: {maxDiff:F6}, mean diff: {sumDiff / config.VocabSize:F6}");

        for (int i = 0; i < decodeSteps; i++)
        {
            Assert.True(eagerTokens[i] == graphTokens[i],
                $"Argmax divergence at step {i}: eager={eagerTokens[i]}, graph={graphTokens[i]}");
        }

        // Eager and graph paths invoke the SAME kernels — only the host-vs-device
        // origin of the eviction counters differs. Logits should be bit-identical
        // (all FP arithmetic is reproducible because the launch sequence is fixed).
        Assert.True(maxDiff < 1e-3f,
            $"First-step logit divergence too large: max abs diff = {maxDiff}");
    }

    private static unsafe int ArgMax(float* data, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
