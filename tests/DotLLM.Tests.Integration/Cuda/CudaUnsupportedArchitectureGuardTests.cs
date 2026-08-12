using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Validates that <see cref="CudaModelLoader.CreateFromGguf"/> fails loudly and specifically for
/// architectures it has no dedicated CUDA loader for, instead of silently falling through to the
/// generic <see cref="CudaTransformerModel"/> — which would either throw a confusing
/// tensor-not-found error deep inside the generic loader (the #259 failure mode), or worse,
/// silently produce wrong output on whichever tensors happen to coincide (the gpt-oss MoE case).
/// </summary>
/// <remarks>
/// The guard fires purely on <c>config.Architecture</c>, before any GGUF tensor is read, so any
/// real GGUF file works as the <c>gguf</c> argument regardless of its own declared architecture —
/// only the manually-constructed <see cref="ModelConfig"/> passed alongside it needs to claim the
/// architecture under test.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaUnsupportedArchitectureGuardTests
{
    [SkippableTheory]
    [InlineData(Architecture.Mamba3)]
    [InlineData(Architecture.NemotronH)]
    public void CreateFromGguf_UnsupportedArchitecture_ThrowsNotSupportedInsteadOfSilentFallthrough(
        Architecture architecture)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("any real GGUF (content unused by this test)"));

        using var gguf = GgufFile.Open(fixture.Path!);
        var realConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var config = realConfig with { Architecture = architecture };

        var ex = Assert.Throws<NotSupportedException>(
            () => CudaModelLoader.CreateFromGguf(gguf, config));

        Assert.DoesNotContain("attn_output.weight", ex.Message, StringComparison.Ordinal);
        Assert.Contains(architecture.ToString(), ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
