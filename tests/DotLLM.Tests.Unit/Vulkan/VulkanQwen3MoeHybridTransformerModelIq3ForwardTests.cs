using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity tests for the Vulkan Qwen3MoeHybrid forward path with IQ3_S and
/// IQ3_XXS projection-weight upload. Audit finding H3: commit 48d65fe added IQ3_S /
/// IQ3_XXS case branches to <see cref="VulkanQwen3MoeHybridTransformerModel"/>'s
/// dispatch — this is the test that would prove the dispatch + upload pipeline is
/// correctly wired for IQ3, not just the kernels in isolation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Status: skip-gated on synthetic-fixture scaffolding only.</b> Both upload-path
/// prerequisites are now in tree (audit-H3 follow-up): the
/// <c>KeepIq3XxsOnDevice</c> / <c>KeepIq3SOnDevice</c> predicates landed on
/// <see cref="VulkanQwen3MoeHybridWeights"/> alongside the Mamba3 + NemotronH
/// hosts, and <see cref="VulkanQwen3MoeHybridTransformerModel"/> now exposes a
/// <c>BuildFromPrebuiltWeights</c> factory mirroring the
/// <see cref="VulkanNemotronHTransformerModel"/> equivalent. The remaining gap
/// is the test-side fixture builder.
/// </para>
/// <para>
/// Synthesising a Qwen3MoeHybrid IQ3 fixture is non-trivial — layers carry
/// GDN-recurrence tensors (ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt.bias,
/// ssm_norm, ssm_out, attn_qkv fused projection, attn_gate) plus a full sparse
/// MoE FFN (router gate, per-routed-expert W1/W2/W3 banks in 3D-stacked layout,
/// optional shared-expert branch). Forty-plus tensors per layer; the fixture
/// builder is the only piece blocking auto-promotion of these tests from skip
/// to passing.
/// </para>
/// <para>
/// <b>Next step:</b> see <c>.planning/notes/iq3-per-host-parity-deferred.md</c>.
/// Flesh out this skip-gated class with a <c>NemotronH</c>-style fixture builder
/// that quantises every matmul-target projection to IQ3 via
/// <see cref="Iq3Fixture"/>, hands the resulting <see cref="Qwen3MoeLayerWeights"/>
/// array to both <see cref="Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights"/>
/// and <see cref="VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights"/>,
/// and asserts CPU-vs-Vulkan logit parity at IQ3 tolerance (abs 1e-1 / rel 1e-1).
/// Estimate: 1-2 hours once a developer is on the Strix Halo host.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridTransformerModelIq3ForwardTests
{
    private const string DeferReason =
        "Qwen3MoeHybrid IQ3 host parity: both upload-path prerequisites now in tree " +
        "(KeepIq3* predicates on VulkanQwen3MoeHybridWeights + BuildFromPrebuiltWeights " +
        "factory on VulkanQwen3MoeHybridTransformerModel). The remaining gap is the " +
        "test-side synthetic-fixture builder for the hybrid 40+-tensor-per-layer layout " +
        "(GDN recurrence + sparse MoE FFN). " +
        "See .planning/notes/iq3-per-host-parity-deferred.md for the concrete next-step.";

    [SkippableFact]
    public void Forward_IQ3_XXS_Prefill_FiniteLogits()
    {
        Skip.If(true, DeferReason);
    }

    [SkippableFact]
    public void Forward_IQ3_S_Prefill_FiniteLogits()
    {
        Skip.If(true, DeferReason);
    }

    [SkippableFact]
    public void Forward_IQ3_XXS_VsCpuOracle_LogitsMatch()
    {
        Skip.If(true, DeferReason);
    }

    [SkippableFact]
    public void Forward_IQ3_S_VsCpuOracle_LogitsMatch()
    {
        Skip.If(true, DeferReason);
    }
}
