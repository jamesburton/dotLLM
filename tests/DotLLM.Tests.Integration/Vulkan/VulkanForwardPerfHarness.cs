using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Non-asserting timing harness for the Vulkan forward pass on SmolLM-135M.
/// Gated by <c>DOTLLM_VULKAN_PERF=1</c> so it does not add run time to the
/// default test sweep; invoked manually from the perf wave.
/// </summary>
/// <remarks>
/// <para>
/// Runs one prefill (≈10 tokens) + N decode steps (default 32) on a warmed-up
/// <see cref="VulkanTransformerModel"/> and prints per-step wall time via
/// <see cref="Stopwatch"/>. The parity test
/// <see cref="VulkanTransformerModelTests.VulkanForward_MatchesCpuReference_OnEightDecodeSteps"/>
/// remains the correctness oracle — this harness only measures latency.
/// </para>
/// <para>
/// Env vars:
/// <list type="bullet">
///   <item><c>DOTLLM_VULKAN_PERF=1</c> — required to run.</item>
///   <item><c>DOTLLM_VULKAN_PERF_DECODE_STEPS</c> — override decode step count (default 32).</item>
///   <item><c>DOTLLM_VULKAN_PERF_WARMUP</c> — warm-up decode steps that are timed but reported separately (default 4).</item>
/// </list>
/// </para>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public class VulkanForwardPerfHarness
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanForwardPerfHarness(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void MeasureDecodeLatency()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF") == "1",
            "DOTLLM_VULKAN_PERF=1 not set.");
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();

        int warmupSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_WARMUP", 4);
        int decodeSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_DECODE_STEPS", 32);

        // DOTLLM_VULKAN_PERF_MODEL overrides the SmolLM-135M fixture with any
        // GGUF path so the harness can profile larger / differently-quantized
        // models (e.g. Q4_K) for the decode-GEMV optimisation work.
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF_MODEL") is { Length: > 0 } mp
            ? mp : _fixture.FilePath;
        _output.WriteLine($"model={modelPath}");
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

        int[] prompt = tokenizer.Encode("The capital of France is").ToArray();
        Assert.NotEmpty(prompt);

        using var cache = model.CreateKvCache(maxSeqLen: 256);

        int[] positions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) positions[i] = i;

        // Prefill
        var prefillSw = Stopwatch.StartNew();
        int nextToken;
        using (var logits = model.Forward(prompt, positions, deviceId: -1, cache))
        {
            prefillSw.Stop();
            nextToken = Argmax(logits);
        }
        _output.WriteLine($"prefill_len={prompt.Length} prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F2}");

        int nextPos = prompt.Length;

        // Warm-up decodes — report separately so JIT / driver shader compile cost
        // does not leak into the steady-state numbers.
        var warmupTotal = 0.0;
        for (int i = 0; i < warmupSteps; i++)
        {
            int[] single = { nextToken };
            int[] pos = { nextPos };
            var sw = Stopwatch.StartNew();
            using (var logits = model.Forward(single, pos, deviceId: -1, cache))
            {
                sw.Stop();
                nextToken = Argmax(logits);
            }
            nextPos++;
            warmupTotal += sw.Elapsed.TotalMilliseconds;
            _output.WriteLine($"warmup[{i}]_ms={sw.Elapsed.TotalMilliseconds:F2}");
        }
        _output.WriteLine($"warmup_avg_ms={(warmupSteps == 0 ? 0.0 : warmupTotal / warmupSteps):F2}");

        // Steady-state decodes.
        double decodeTotal = 0.0;
        double decodeMin = double.PositiveInfinity;
        double decodeMax = 0.0;
        for (int i = 0; i < decodeSteps; i++)
        {
            int[] single = { nextToken };
            int[] pos = { nextPos };
            var sw = Stopwatch.StartNew();
            using (var logits = model.Forward(single, pos, deviceId: -1, cache))
            {
                sw.Stop();
                nextToken = Argmax(logits);
            }
            nextPos++;
            double ms = sw.Elapsed.TotalMilliseconds;
            decodeTotal += ms;
            if (ms < decodeMin) decodeMin = ms;
            if (ms > decodeMax) decodeMax = ms;
            _output.WriteLine($"decode[{i}]_ms={ms:F2}");
        }
        double decodeAvg = decodeSteps == 0 ? 0.0 : decodeTotal / decodeSteps;
        double tokPerSec = decodeAvg > 0 ? 1000.0 / decodeAvg : 0.0;

        _output.WriteLine($"=== summary ===");
        _output.WriteLine($"decode_steps={decodeSteps}");
        _output.WriteLine($"decode_avg_ms={decodeAvg:F2}");
        _output.WriteLine($"decode_min_ms={decodeMin:F2}");
        _output.WriteLine($"decode_max_ms={decodeMax:F2}");
        _output.WriteLine($"decode_tok_per_sec={tokPerSec:F2}");
    }

    /// <summary>
    /// Isolated bandwidth microbench for the Q4_K MMVQ decode GEMV. Allocates a
    /// single large device-local Q4_K weight blob, quantizes one random
    /// activation to Q8_1, then times <c>iters</c> back-to-back dispatches in one
    /// submit (no inter-dispatch barrier — they all read the immutable weights so
    /// the GPU can overlap them, exposing the kernel's true streaming bandwidth
    /// independent of the full forward pass's dispatch/barrier serialization).
    /// Enable with DOTLLM_VULKAN_MMVQ_BW=1.
    /// </summary>
    [SkippableFact]
    public void MeasureMmvqBandwidth()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MMVQ_BW") == "1",
            "DOTLLM_VULKAN_MMVQ_BW=1 not set.");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string spvDir = ResolveSpvDir();

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");

        // FFN-down-ish shape: M=4096, K=14336 → ~28 MiB of Q4_K weights per matmul.
        int m = ParseEnvInt("DOTLLM_VULKAN_MMVQ_BW_M", 4096);
        int k = ParseEnvInt("DOTLLM_VULKAN_MMVQ_BW_K", 14336);
        int iters = ParseEnvInt("DOTLLM_VULKAN_MMVQ_BW_ITERS", 50);

        int blocksPerRow = k / 256;
        long rowBytes = (long)blocksPerRow * 144;
        long wBytes = (long)m * rowBytes;
        _output.WriteLine($"mmvq_bw: M={m} K={k} weight_MiB={wBytes / 1048576.0:F1} iters={iters}");

        var rng = new Random(1);
        byte[] w = new byte[wBytes];
        rng.NextBytes(w);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using var quant = DotLLM.Vulkan.Kernels.QuantizeQ8_1Kernel.TryCreate(device, spvDir)!;
        using var mmvq = DotLLM.Vulkan.Kernels.MatMulQ4KMmvqKernel.TryCreate(device, spvDir)!;

        long wBufBytes = (wBytes + 3) & ~3L;
        using var bufW = device.Allocate(wBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(DotLLM.Vulkan.Kernels.QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(DotLLM.Vulkan.Kernels.QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);

        // Quantize once.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            ctx.SubmitAndWait();
        }

        // Warmup + timed runs. Each run records `iters` MMVQ dispatches into one
        // command buffer (no barrier between them) and submits once.
        void RunBatch()
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            for (int i = 0; i < iters; i++)
                mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
            ctx.SubmitAndWait();
        }

        RunBatch(); // warmup (driver pipeline compile)
        RunBatch();

        const int batches = 5;
        var sw = Stopwatch.StartNew();
        for (int b = 0; b < batches; b++) RunBatch();
        sw.Stop();

        double totalDispatches = (double)batches * iters;
        double msPerDispatch = sw.Elapsed.TotalMilliseconds / totalDispatches;
        double gbPerSec = wBytes / (msPerDispatch / 1000.0) / 1e9;
        _output.WriteLine($"mmvq_bw_ms_per_dispatch={msPerDispatch:F3}");
        _output.WriteLine($"mmvq_bw_gb_per_sec={gbPerSec:F1}");

        // Serialized variant: ONE dispatch per submit (mirrors the forward pass,
        // where a full barrier + submit boundary sits around every GEMV). Exposes
        // the per-dispatch latency tax the batched number hides.
        void RunSingle()
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
            ctx.SubmitAndWait();
        }
        RunSingle();
        const int singles = 100;
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < singles; i++) RunSingle();
        sw2.Stop();
        double msSingle = sw2.Elapsed.TotalMilliseconds / singles;
        double gbSingle = wBytes / (msSingle / 1000.0) / 1e9;
        _output.WriteLine($"mmvq_bw_serialized_ms_per_dispatch={msSingle:F3}");
        _output.WriteLine($"mmvq_bw_serialized_gb_per_sec={gbSingle:F1}");
    }

    private static unsafe int Argmax(ITensor logits)
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

    private static int ParseEnvInt(string key, int fallback)
    {
        string? v = Environment.GetEnvironmentVariable(key);
        return int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
