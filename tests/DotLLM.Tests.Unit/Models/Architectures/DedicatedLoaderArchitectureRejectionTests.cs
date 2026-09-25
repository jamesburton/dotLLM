using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #324. Architectures whose GGUF tensor naming is NOT dense Llama-style must be
/// rejected at the shared dense-load choke point with an error that names the architecture
/// and the dispatcher to use — not with the <c>KeyNotFoundException</c> on
/// <c>blk.0.attn_output.weight</c> that the load would otherwise die on several hundred
/// lines later.
/// </summary>
/// <remarks>
/// These run against a synthetic <see cref="ModelConfig"/>: no GGUF file, no GPU, no
/// fixture download. The guard is deliberately placed before the first tensor lookup in
/// <c>TransformerWeights.LoadFromGguf</c>, which is exactly what makes that possible.
/// </remarks>
public sealed class DedicatedLoaderArchitectureRejectionTests
{
    private static ModelConfig ConfigFor(Architecture arch) => new()
    {
        Architecture = arch,
        VocabSize = 32,
        HiddenSize = 8,
        IntermediateSize = 16,
        NumLayers = 1,
        NumAttentionHeads = 2,
        NumKvHeads = 2,
        HeadDim = 4,
        MaxSequenceLength = 16,
        NormEpsilon = 1e-5f,
    };

    [Theory]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.Qwen3MoeHybrid)]
    [InlineData(Architecture.Qwen3HybridDense)]
    [InlineData(Architecture.Mamba3)]
    public void DedicatedLoaderArchitectures_AreRejectedWithAnActionableMessage(Architecture arch)
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => TransformerWeights.ThrowIfArchitectureNeedsDedicatedLoader(ConfigFor(arch)));

        // Names the architecture, so the reader knows which model tripped it.
        Assert.Contains(arch.ToString(), ex.Message, StringComparison.Ordinal);

        // Names the way out on every backend — the whole point of the guard is that the
        // caller should not have to know which of these exists.
        Assert.Contains("ModelLoader.CreateCpuModelFromGguf", ex.Message, StringComparison.Ordinal);
        Assert.Contains("VulkanModelLoader.CreateFromGguf", ex.Message, StringComparison.Ordinal);
        Assert.Contains("CudaModelLoader.CreateFromGguf", ex.Message, StringComparison.Ordinal);

        // And pre-empts the misleading symptom by name, so a search for the old error
        // lands on the real explanation.
        Assert.Contains("blk.0.attn_output.weight", ex.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(Architecture.Llama)]
    [InlineData(Architecture.Mistral)]
    [InlineData(Architecture.Qwen)]
    [InlineData(Architecture.QwenMoe)]
    [InlineData(Architecture.DeepSeekV2)]
    public void DenseArchitectures_AreNotRejected(Architecture arch)
    {
        // Guard must be inert for everything that genuinely uses dense tensor naming;
        // a false positive here would break every currently-loading model.
        TransformerWeights.ThrowIfArchitectureNeedsDedicatedLoader(ConfigFor(arch));
    }
}
