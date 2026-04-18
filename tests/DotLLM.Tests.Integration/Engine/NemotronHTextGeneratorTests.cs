using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// End-to-end smoke test for the Nemotron-H hybrid SSM + Transformer model.
/// Requires a local GGUF checkpoint (e.g. NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf,
/// ~2.7 GB) referenced via the <c>DOTLLM_NEMOTRON_H_GGUF</c> environment variable.
/// Skips when the env var is unset or the file is missing — CI runs don't pull
/// multi-gig checkpoints.
/// </summary>
public class NemotronHTextGeneratorTests
{
    private const string ModelPathEnvVar = "DOTLLM_NEMOTRON_H_GGUF";

    private static string? TryResolveModelPath()
    {
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return null;
        return path;
    }

    [SkippableFact]
    public void Config_DetectsNemotronHArchitecture()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to a Nemotron-H GGUF to run this test.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.NemotronH, config.Architecture);
        Assert.NotNull(config.HybridLayout);
        Assert.NotNull(config.SsmConfig);
    }

    [SkippableFact]
    public void Forward_SinglePrefill_ProducesFiniteLogits()
    {
        string? path = TryResolveModelPath();
        Skip.If(path is null, $"Set {ModelPathEnvVar} to a Nemotron-H GGUF to run this test.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config);

        int[] tokenIds = [1, 2, 3, 4, 5];
        int[] positions = [0, 1, 2, 3, 4];

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: 0);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);

        int totalElems = tokenIds.Length * config.VocabSize;
        ReadOnlySpan<float> logitSpan;
        unsafe
        {
            logitSpan = new ReadOnlySpan<float>((void*)logits.DataPointer, totalElems);
        }

        // Non-NaN / non-Inf over the final token's row (the one that matters for next-token sampling).
        int lastRow = tokenIds.Length - 1;
        var lastLogits = logitSpan.Slice(lastRow * config.VocabSize, config.VocabSize);
        for (int i = 0; i < lastLogits.Length; i++)
        {
            Assert.True(float.IsFinite(lastLogits[i]),
                $"Logit[{lastRow},{i}] = {lastLogits[i]} is non-finite.");
        }

        // Argmax over last row must land in the vocab range (trivially true for finite logits, but
        // catches the pathological case where the whole row is constant — a common symptom of a
        // broken sub-layer that zeros its output).
        int argmax = 0;
        float best = lastLogits[0];
        for (int i = 1; i < lastLogits.Length; i++)
        {
            if (lastLogits[i] > best)
            {
                best = lastLogits[i];
                argmax = i;
            }
        }
        Assert.InRange(argmax, 0, config.VocabSize - 1);
        Assert.True(TensorPrimitives.Max(lastLogits) > TensorPrimitives.Min(lastLogits),
            "Last-token logits are constant — likely a broken sub-layer.");
    }
}
