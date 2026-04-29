using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in timing harness for CUDA real-GGUF forward latency.
/// Correctness remains covered by parity tests; this harness only reports load,
/// prefill, warm-up decode, and steady-state decode timings.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaForwardPerfHarness
{
    private readonly ITestOutputHelper _output;

    public CudaForwardPerfHarness(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public void MeasureDecodeLatency()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_CUDA_PERF") == "1",
            "DOTLLM_CUDA_PERF=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? modelPath = ResolveIQuantFixturePath();
        Skip.If(modelPath is null,
            "CUDA IQ4 GGUF fixture not found. Set DOTLLM_CUDA_PERF_GGUF, DOTLLM_CUDA_IQ4_GGUF, "
            + "DOTLLM_IQ4_XS_GGUF_PATH, or DOTLLM_IQ4_NL_GGUF_PATH.");

        int warmupSteps = ParseEnvInt("DOTLLM_CUDA_PERF_WARMUP", 4);
        int decodeSteps = ParseEnvInt("DOTLLM_CUDA_PERF_DECODE_STEPS", 32);
        string ptxDir = ResolvePtxDir();

        _output.WriteLine($"model={modelPath}");
        _output.WriteLine($"ptx_dir={ptxDir}");

        using var gguf = GgufFile.Open(modelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var loadSw = Stopwatch.StartNew();
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        loadSw.Stop();
        _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

        int[] prompt = tokenizer.Encode("The capital of France is").ToArray();
        Assert.NotEmpty(prompt);
        if (prompt.Length > 16)
            prompt = prompt[..16];

        using var cache = model.CreateKvCache(maxSeqLen: prompt.Length + warmupSteps + decodeSteps + 8);
        int[] positions = Positions(prompt.Length);

        int nextToken;
        var prefillSw = Stopwatch.StartNew();
        using (ITensor logits = model.Forward(prompt, positions, deviceId: 0, cache))
        {
            prefillSw.Stop();
            nextToken = Argmax(logits);
        }
        _output.WriteLine($"prefill_len={prompt.Length} prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F2}");

        int nextPos = prompt.Length;
        double warmupTotal = 0.0;
        for (int i = 0; i < warmupSteps; i++)
        {
            double ms = DecodeOne(model, cache, nextToken, nextPos, out nextToken);
            nextPos++;
            warmupTotal += ms;
            _output.WriteLine($"warmup[{i}]_ms={ms:F2}");
        }
        _output.WriteLine($"warmup_avg_ms={(warmupSteps == 0 ? 0.0 : warmupTotal / warmupSteps):F2}");

        double decodeTotal = 0.0;
        double decodeMin = double.PositiveInfinity;
        double decodeMax = 0.0;
        for (int i = 0; i < decodeSteps; i++)
        {
            double ms = DecodeOne(model, cache, nextToken, nextPos, out nextToken);
            nextPos++;
            decodeTotal += ms;
            if (ms < decodeMin) decodeMin = ms;
            if (ms > decodeMax) decodeMax = ms;
            _output.WriteLine($"decode[{i}]_ms={ms:F2}");
        }

        double decodeAvg = decodeSteps == 0 ? 0.0 : decodeTotal / decodeSteps;
        double tokPerSec = decodeAvg > 0 ? 1000.0 / decodeAvg : 0.0;

        _output.WriteLine("=== summary ===");
        _output.WriteLine($"decode_steps={decodeSteps}");
        _output.WriteLine($"decode_avg_ms={decodeAvg:F2}");
        _output.WriteLine($"decode_min_ms={decodeMin:F2}");
        _output.WriteLine($"decode_max_ms={decodeMax:F2}");
        _output.WriteLine($"decode_tok_per_sec={tokPerSec:F2}");
    }

    private static double DecodeOne(
        CudaTransformerModel model,
        CudaKvCache cache,
        int token,
        int position,
        out int nextToken)
    {
        int[] single = [token];
        int[] positions = [position];
        var sw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(single, positions, deviceId: 0, cache);
        sw.Stop();
        nextToken = Argmax(logits);
        return sw.Elapsed.TotalMilliseconds;
    }

    private static int ParseEnvInt(string key, int fallback)
    {
        string? value = Environment.GetEnvironmentVariable(key);
        return int.TryParse(value, out int n) && n >= 0 ? n : fallback;
    }

    private static string? ResolveIQuantFixturePath()
    {
        string[] envVars =
        [
            "DOTLLM_CUDA_PERF_GGUF",
            "DOTLLM_CUDA_IQ4_GGUF",
            "DOTLLM_IQ4_XS_GGUF_PATH",
            "DOTLLM_IQ4_NL_GGUF_PATH",
        ];

        foreach (string envVar in envVars)
        {
            string? envPath = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
                return Path.GetFullPath(envPath);
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] relativeCandidates =
        [
            Path.Combine("bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"),
            Path.Combine("bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-IQ4_NL.gguf"),
            Path.Combine("bartowski", "Qwen2.5-7B-Instruct-GGUF",
                "Qwen2.5-7B-Instruct-IQ4_XS.gguf"),
            Path.Combine("bartowski", "Qwen2.5-7B-Instruct-GGUF",
                "Qwen2.5-7B-Instruct-IQ4_NL.gguf"),
        ];

        string[] roots =
        [
            Path.Combine(home, ".dotllm", "models"),
            Path.Combine(home, ".dotllm", "test-cache"),
        ];

        foreach (string root in roots)
        {
            foreach (string relative in relativeCandidates)
            {
                string candidate = Path.Combine(root, relative);
                if (File.Exists(candidate))
                    return Path.GetFullPath(candidate);
            }
        }

        return null;
    }

    private static string ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            string candidate = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;

            dir = Path.GetDirectoryName(dir);
        }

        return Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
    }

    private static int[] Positions(int count)
    {
        int[] positions = new int[count];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = i;
        return positions;
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0;
        float best = span[0];
        for (int i = 1; i < n; i++)
        {
            if (span[i] > best)
            {
                best = span[i];
                idx = i;
            }
        }
        return idx;
    }
}
