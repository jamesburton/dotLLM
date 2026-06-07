using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

[Trait("Category", "GPU")]
public sealed class CudaRealGgufIQuantSmokeTests
{
    private const string ModelPathEnvVar = "DOTLLM_CUDA_IQ4_GGUF";
    private const string Iq4NlModelPathEnvVar = "DOTLLM_CUDA_IQ4_NL_GGUF";
    private const string Iq4NlAliasModelPathEnvVar = "DOTLLM_IQ4_NL_GGUF_PATH";

    private readonly ITestOutputHelper _output;

    public CudaRealGgufIQuantSmokeTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public unsafe void IQ4RealGguf_LoadsAndRunsCudaPrefillAndDecode()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? ggufPath = ResolveIQuantFixturePath();
        Skip.If(ggufPath is null,
            "IQ4_XS/IQ4_NL GGUF fixture not found. Set DOTLLM_CUDA_IQ4_GGUF, or place "
            + "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf under "
            + "~/.dotllm/models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/.");

        RunRealGgufPrefillAndDecode(ggufPath);
    }

    [SkippableFact]
    public unsafe void IQ4NLRealGguf_LoadsAndRunsCudaPrefillAndDecode()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string? ggufPath = ResolveIq4NlFixturePath();
        Skip.If(ggufPath is null,
            $"IQ4_NL GGUF fixture not found. Set {Iq4NlModelPathEnvVar} or {Iq4NlAliasModelPathEnvVar}, "
            + "or place an IQ4_NL fixture under ~/.dotllm/models or ~/.dotllm/test-cache.");

        RunRealGgufPrefillAndDecode(ggufPath);
    }

    private unsafe void RunRealGgufPrefillAndDecode(string ggufPath)
    {
        string ptxDir = ResolvePtxDir();
        _output.WriteLine($"GGUF: {ggufPath}");
        _output.WriteLine($"PTX dir: {ptxDir}");

        using var gguf = GgufFile.Open(ggufPath);
        Assert.Contains(gguf.TensorsByName.Values, t =>
            t.QuantizationType is QuantizationType.IQ4_XS or QuantizationType.IQ4_NL);

        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        _output.WriteLine(
            $"Config: arch={config.Architecture} layers={config.NumLayers} hidden={config.HiddenSize} "
            + $"heads={config.NumAttentionHeads}/{config.NumKvHeads} vocab={config.VocabSize}");

        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] promptTokens = tokenizer.Encode("The capital of France is");
        Skip.If(promptTokens.Length == 0, "GGUF tokenizer returned no tokens for the smoke prompt.");
        if (promptTokens.Length > 8)
            promptTokens = promptTokens[..8];

        int[] prefillPositions = Positions(promptTokens.Length, start: 0);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kvCache = model.CreateKvCache(maxSeqLen: 32);

        using ITensor prefillLogits = model.Forward(promptTokens, prefillPositions, deviceId: 0, kvCache);
        AssertLogitsAreFinite(prefillLogits, config.VocabSize, "prefill");

        int nextToken = ArgMax(prefillLogits, config.VocabSize);
        using ITensor decodeLogits = model.Forward(
            [nextToken],
            [promptTokens.Length],
            deviceId: 0,
            kvCache);
        AssertLogitsAreFinite(decodeLogits, config.VocabSize, "decode");
    }

    private static string? ResolveIq4NlFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(Iq4NlModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        envPath = Environment.GetEnvironmentVariable(Iq4NlAliasModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] relativeCandidates =
        [
            Path.Combine("bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-IQ4_NL.gguf"),
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
                    return candidate;
            }
        }

        return null;
    }

    private static string? ResolveIQuantFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string modelsRoot = Path.Combine(home, ".dotllm", "models");
        string testCacheRoot = Path.Combine(home, ".dotllm", "test-cache");

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

        foreach (string relative in relativeCandidates)
        {
            string cliPath = Path.Combine(modelsRoot, relative);
            if (File.Exists(cliPath))
                return cliPath;

            string testCachePath = Path.Combine(testCacheRoot, relative);
            if (File.Exists(testCachePath))
                return testCachePath;
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

    private static int[] Positions(int count, int start)
    {
        int[] positions = new int[count];
        for (int i = 0; i < positions.Length; i++)
            positions[i] = start + i;
        return positions;
    }

    private static unsafe void AssertLogitsAreFinite(ITensor logits, int vocabSize, string step)
    {
        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(vocabSize, logits.Shape[1]);

        var values = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int finiteCount = 0;
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;

        for (int i = 0; i < values.Length; i++)
        {
            float value = values[i];
            if (!float.IsFinite(value))
                continue;

            finiteCount++;
            if (value < min) min = value;
            if (value > max) max = value;
        }

        Assert.Equal(vocabSize, finiteCount);
        Assert.True(max > min, $"{step} logits are finite but flat: min={min:G9}, max={max:G9}.");
    }

    private static unsafe int ArgMax(ITensor logits, int vocabSize)
    {
        var values = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
                best = i;
        }

        return best;
    }
}
