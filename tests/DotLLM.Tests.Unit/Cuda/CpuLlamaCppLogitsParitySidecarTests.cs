using System.Diagnostics;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only llama.cpp logits parity scaffold for real IQ4_XS GGUF fixtures.
/// Used to classify divergence: if CPU matches llama.cpp but CUDA diverges,
/// the problem is CUDA-specific (dequant/GEMV/MMQ/dispatch). If CPU diverges too,
/// the problem is in the model loader, GGUF tensor mapping, or CPU dequant oracle.
/// </summary>
[Trait("Category", "CPU")]
public sealed class CpuLlamaCppLogitsParitySidecarTests
{
    private const string ModelPathEnvVar = "DOTLLM_IQ4_XS_GGUF_PATH";
    private const string SharedIQuantModelPathEnvVar = "DOTLLM_CUDA_IQ4_GGUF";
    private const string SidecarPathEnvVar = "DOTLLM_LLAMA_CPP_IQ4_XS_SIDECAR_PATH";
    private const string DefaultSidecarRelativePath =
        "tests/DotLLM.Tests.Unit/Cuda/References/llamacpp/iq4-xs-logits-sidecar.json";

    private readonly ITestOutputHelper _output;

    public CpuLlamaCppLogitsParitySidecarTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public unsafe void IQ4XS_LastTokenLogits_MatchLlamaCppSidecar_Cpu()
    {
        string sidecarPath = ResolveSidecarPath();
        Skip.If(!File.Exists(sidecarPath),
            $"llama.cpp IQ4_XS logits sidecar not found at {sidecarPath}. "
            + $"Set {SidecarPathEnvVar} or generate the default sidecar; see "
            + "tests/DotLLM.Tests.Unit/Cuda/References/llamacpp/README.md.");

        var sidecar = LoadSidecar(sidecarPath);
        Assert.Equal("llama.cpp-logits-v1", sidecar.Schema);
        Assert.Equal("IQ4_XS", sidecar.Quantization);
        Assert.NotEmpty(sidecar.InputIds);
        Assert.NotEmpty(sidecar.ReferenceLogits);

        string? modelPath = ResolveModelPath(sidecar.ModelPath);
        Skip.If(modelPath is null,
            $"IQ4_XS GGUF fixture not found. Set {ModelPathEnvVar} or {SharedIQuantModelPathEnvVar} "
            + "to the same GGUF used to generate the sidecar. Do not commit the GGUF model file.");

        _output.WriteLine($"Model:   {modelPath}");
        _output.WriteLine($"Sidecar: {sidecarPath}");

        using var gguf = GgufFile.Open(modelPath!);
        Skip.If(!ContainsQuantization(gguf, QuantizationType.IQ4_XS),
            $"GGUF at {modelPath} does not contain IQ4_XS tensors; set {ModelPathEnvVar} "
            + "to the real IQ4_XS fixture used for the sidecar.");

        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(sidecar.VocabSize, config.VocabSize);
        Assert.Equal(sidecar.VocabSize, sidecar.ReferenceLogits.Length);

        int[] positions = new int[sidecar.InputIds.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        var loadWatch = Stopwatch.StartNew();
        var model = TransformerModel.LoadFromGguf(gguf, config);
        loadWatch.Stop();
        _output.WriteLine(
            $"CPU load: {loadWatch.Elapsed.TotalMilliseconds:F1} ms "
            + $"(arch={config.Architecture}, layers={config.NumLayers}, hidden={config.HiddenSize}, vocab={config.VocabSize})");

        try
        {
            var forwardWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(sidecar.InputIds, positions, deviceId: 0);
            forwardWatch.Stop();
            _output.WriteLine(
                $"CPU forward ({forwardWatch.Elapsed.TotalSeconds:F3} s): "
                + $"shape=[{logits.Shape[0]}, {logits.Shape[1]}]");

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(sidecar.InputIds.Length, logits.Shape[0]);  // seqLen = input IDs count
            Assert.Equal(sidecar.VocabSize, logits.Shape[1]);

            // Extract last token logits [vocab_size]
            // Logits are in row-major order: logits[seq][vocab]
            int lastTokenIndex = logits.Shape[0] - 1;
            nint lastTokenPtr = logits.DataPointer + (lastTokenIndex * sidecar.VocabSize * sizeof(float));
            var ours = new ReadOnlySpan<float>((void*)lastTokenPtr, sidecar.VocabSize);
            CompareLogits(ours, sidecar);
        }
        finally
        {
            model.Dispose();
        }
    }

    private void CompareLogits(ReadOnlySpan<float> ours, LlamaCppLogitsSidecar sidecar)
    {
        int oursArgmax = Argmax(ours);
        int refArgmax = Argmax(sidecar.ReferenceLogits);

        double sumAbs = 0;
        float maxAbs = 0;
        int maxAbsIndex = 0;
        for (int i = 0; i < sidecar.ReferenceLogits.Length; i++)
        {
            float diff = MathF.Abs(ours[i] - sidecar.ReferenceLogits[i]);
            sumAbs += diff;
            if (diff > maxAbs)
            {
                maxAbs = diff;
                maxAbsIndex = i;
            }
        }

        double meanAbs = sumAbs / sidecar.ReferenceLogits.Length;
        _output.WriteLine(
            $"argmax: dotLLM={oursArgmax} ({ours[oursArgmax]:F4}) "
            + $"llama.cpp={refArgmax} ({sidecar.ReferenceLogits[refArgmax]:F4})");
        _output.WriteLine(
            $"drift: max_abs={maxAbs:F6} at token {maxAbsIndex}, mean_abs={meanAbs:F6}");

        Assert.Equal(sidecar.ArgmaxTokenId, refArgmax);
        Assert.Equal(sidecar.ArgmaxTokenId, oursArgmax);
        Assert.True(maxAbs <= sidecar.MaxAbsTolerance,
            $"max_abs={maxAbs:F6} exceeds sidecar tolerance {sidecar.MaxAbsTolerance:F6}.");
        Assert.True(meanAbs <= sidecar.MeanAbsTolerance,
            $"mean_abs={meanAbs:F6} exceeds sidecar tolerance {sidecar.MeanAbsTolerance:F6}.");
    }

    private static int Argmax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestValue = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestValue)
            {
                best = i;
                bestValue = values[i];
            }
        }
        return best;
    }

    private static bool ContainsQuantization(GgufFile gguf, QuantizationType quantization)
    {
        foreach (var tensor in gguf.Tensors)
        {
            if (tensor.QuantizationType == quantization)
                return true;
        }
        return false;
    }

    private static string? ResolveModelPath(string? sidecarModelPath)
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return Path.GetFullPath(envPath);

        string? sharedEnvPath = Environment.GetEnvironmentVariable(SharedIQuantModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(sharedEnvPath) && File.Exists(sharedEnvPath))
            return Path.GetFullPath(sharedEnvPath);

        if (!string.IsNullOrWhiteSpace(sidecarModelPath) && File.Exists(sidecarModelPath))
            return Path.GetFullPath(sidecarModelPath);

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
                "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"),
            Path.Combine(home, ".dotllm", "models", "bartowski", "Qwen2.5-7B-Instruct-GGUF",
                "Qwen2.5-7B-Instruct-IQ4_XS.gguf"),
            Path.Combine(home, ".dotllm", "test-cache", "bartowski", "Qwen2.5-7B-Instruct-GGUF",
                "Qwen2.5-7B-Instruct-IQ4_XS.gguf"),
        ];

        foreach (string candidate in candidates)
        {
            if (File.Exists(candidate))
                return Path.GetFullPath(candidate);
        }

        return null;
    }

    private static string ResolveSidecarPath()
    {
        string? envPath = Environment.GetEnvironmentVariable(SidecarPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath))
            return Path.GetFullPath(envPath);

        string? repoRoot = FindRepoRoot();
        return repoRoot is null
            ? Path.GetFullPath(DefaultSidecarRelativePath)
            : Path.Combine(repoRoot, DefaultSidecarRelativePath);
    }

    private static string? FindRepoRoot()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            if (File.Exists(Path.Combine(dir, "dotLLM.slnx")))
                return dir;
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private static LlamaCppLogitsSidecar LoadSidecar(string path)
    {
        using FileStream fs = File.OpenRead(path);
        using var doc = JsonDocument.Parse(fs, new JsonDocumentOptions
        {
            AllowTrailingCommas = true,
            CommentHandling = JsonCommentHandling.Skip,
        });

        JsonElement root = doc.RootElement;
        string schema = RequiredString(root, "schema");
        string quantization = RequiredString(root, "quantization");
        string? modelPath = OptionalString(root, "model_path");
        string prompt = OptionalString(root, "prompt") ?? "";
        int vocabSize = RequiredInt(root, "vocab_size");
        int argmaxTokenId = RequiredInt(root, "argmax_token_id");
        float maxAbsTolerance = OptionalFloat(root, "max_abs_tolerance") ?? 2.0f;
        double meanAbsTolerance = OptionalDouble(root, "mean_abs_tolerance") ?? 0.25;

        int[] inputIds = ReadIntArray(root.GetProperty("input_ids"));
        float[] logits = ReadFloatArray(root.GetProperty("last_token_logits"));

        return new LlamaCppLogitsSidecar(
            schema,
            quantization,
            modelPath,
            prompt,
            inputIds,
            vocabSize,
            logits,
            argmaxTokenId,
            maxAbsTolerance,
            meanAbsTolerance);
    }

    private static string RequiredString(JsonElement root, string propertyName)
        => root.GetProperty(propertyName).GetString()
            ?? throw new InvalidDataException($"Sidecar property '{propertyName}' must be a string.");

    private static string? OptionalString(JsonElement root, string propertyName)
        => root.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static int RequiredInt(JsonElement root, string propertyName)
        => root.GetProperty(propertyName).GetInt32();

    private static float? OptionalFloat(JsonElement root, string propertyName)
        => root.TryGetProperty(propertyName, out var value) ? (float)value.GetDouble() : null;

    private static double? OptionalDouble(JsonElement root, string propertyName)
        => root.TryGetProperty(propertyName, out var value) ? value.GetDouble() : null;

    private static int[] ReadIntArray(JsonElement element)
    {
        int[] values = new int[element.GetArrayLength()];
        int i = 0;
        foreach (var value in element.EnumerateArray())
            values[i++] = value.GetInt32();
        return values;
    }

    private static float[] ReadFloatArray(JsonElement element)
    {
        float[] values = new float[element.GetArrayLength()];
        int i = 0;
        foreach (var value in element.EnumerateArray())
            values[i++] = (float)value.GetDouble();
        return values;
    }

    private sealed record LlamaCppLogitsSidecar(
        string Schema,
        string Quantization,
        string? ModelPath,
        string Prompt,
        int[] InputIds,
        int VocabSize,
        float[] ReferenceLogits,
        int ArgmaxTokenId,
        float MaxAbsTolerance,
        double MeanAbsTolerance);
}
