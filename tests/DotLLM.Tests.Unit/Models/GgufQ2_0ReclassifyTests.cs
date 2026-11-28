using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// GGUF type id 42 is both PrismML's PQ2_0 (34 B / 128) and upstream ggml's Q2_0 (18 B / 64); the reader tells them apart by on-disk extent
/// (#823, #2074). Both directions are pinned, because a reader that always answered PQ2_0 (or always Q2_0) would still pass one of them.
/// </summary>
public sealed class GgufQ2_0ReclassifyTests
{
    [Fact]
    public void ExtentDecides_Q2_0_Versus_PQ2_0()
    {
        // 16384 elements: Q2_0 needs 4608 B, PQ2_0 needs 4352 B; alignment 32.
        var shape = new TensorShape(16384);
        var list = new List<GgufTensorDescriptor>
        {
            new("upstream", shape, QuantizationType.PQ2_0, 0),            // 4608 B to the next offset -> Q2_0
            new("prism", shape, QuantizationType.PQ2_0, 4608),            // 4352 B to the next offset -> stays PQ2_0
            new("tiny", new TensorShape(1024), QuantizationType.PQ2_0, 8960),   // < 2048 elements: ambiguous, left alone
        };
        GgufReader.ReclassifyUpstreamQ2_0(list, 32, 8960 + 288);
        Assert.Equal(QuantizationType.Q2_0, list[0].QuantizationType);
        Assert.Equal(QuantizationType.PQ2_0, list[1].QuantizationType);
        Assert.Equal(QuantizationType.PQ2_0, list[2].QuantizationType);
    }

    [SkippableFact]
    public void RealBonsai25Q2_0File_RelabelsEveryType42Tensor()
    {
        string root = Environment.GetEnvironmentVariable("HF_HUB_CACHE")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "hub");
        string dir = Path.Combine(root, "models--prism-ml--Ternary-Bonsai-2.5-27B-GGUF", "snapshots");
        string? file = Directory.Exists(dir)
            ? Directory.EnumerateFiles(dir, "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf", SearchOption.AllDirectories).FirstOrDefault()
            : null;
        Skip.If(file is null, "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf is not in the HF cache");
        using var model = GgufFile.Open(file!);
        Assert.Equal(402, model.Tensors.Count(t => t.QuantizationType == QuantizationType.Q2_0));
        Assert.DoesNotContain(model.Tensors, t => t.QuantizationType == QuantizationType.PQ2_0);
        Assert.Equal(QuantizationType.Q2_0, model.TensorsByName["token_embd.weight"].QuantizationType);
        Assert.Equal(QuantizationType.Q2_0, model.TensorsByName["output.weight"].QuantizationType);
    }
}
