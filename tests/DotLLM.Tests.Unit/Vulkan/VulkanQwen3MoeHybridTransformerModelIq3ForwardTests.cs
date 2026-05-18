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
/// <b>Status: deferred (Option C from the task statement).</b>
/// </para>
/// <para>
/// Two prerequisites are missing for this test to exercise the IQ3 dispatch end-to-end:
/// </para>
/// <list type="number">
///   <item>
///     <description>
///       <see cref="VulkanQwen3MoeHybridTransformerModel"/> only exposes
///       <c>BuildFromGguf</c> — there is no <c>BuildFromPrebuiltWeights</c>
///       entrypoint that the CPU-side <see cref="Qwen3MoeHybridTransformerModel"/>
///       already has. The Mamba3 / dense / NemotronH IQ3 forward tests in this folder
///       all use <c>BuildFromPrebuiltWeights</c> on both backends to skip the GGUF
///       loader. Synthesising a Qwen3MoeHybrid IQ3 GGUF via the existing
///       <c>GgufTestData</c> writer is non-trivial — Qwen3MoeHybrid layers carry
///       GDN-recurrence tensors (ssm_a, ssm_alpha, ssm_beta, ssm_conv1d,
///       ssm_dt.bias, ssm_norm, ssm_out, attn_qkv fused projection, attn_gate)
///       plus a full sparse MoE FFN (router gate, per-routed-expert W1/W2/W3 banks
///       in 3D-stacked layout, optional shared-expert branch). Forty-plus tensors
///       per layer; the GGUF-writer wiring alone is a session of work.
///     </description>
///   </item>
///   <item>
///     <description>
///       <see cref="VulkanQwen3MoeHybridWeights"/>'s upload predicate
///       <c>DeviceQuantTypeFor</c> currently does NOT recognise
///       <see cref="QuantizationType.IQ3_S"/> or <see cref="QuantizationType.IQ3_XXS"/>
///       — the same upload-path gap surfaced by the Mamba3 and NemotronH IQ3
///       forward tests (both Skip-gated for the same reason). Until
///       <c>KeepIq3XxsOnDevice</c> / <c>KeepIq3SOnDevice</c> predicates are added
///       to <see cref="VulkanQwen3MoeHybridWeights"/> (mirroring the dense host's
///       <see cref="VulkanWeights"/>), the IQ3 dispatch arm in
///       <see cref="VulkanQwen3MoeHybridTransformerModel"/> is unreachable
///       regardless of what the test feeds it.
///     </description>
///   </item>
/// </list>
/// <para>
/// <b>Next step:</b> see <c>.planning/notes/iq3-per-host-parity-deferred.md</c>. The
/// precise prescription is (1) add the Iq3 keep-on-device predicates to
/// <see cref="VulkanQwen3MoeHybridWeights"/>; (2) add a
/// <c>BuildFromPrebuiltWeights</c> factory on
/// <see cref="VulkanQwen3MoeHybridTransformerModel"/> (a 30-line mirror of the
/// dense / NemotronH equivalents); (3) flesh out this skip-gated class with an
/// <c>NemotronH</c>-style fixture builder that quantises every projection in the
/// fixture to IQ3 via <see cref="Iq3Fixture"/>. Estimate: half a session.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridTransformerModelIq3ForwardTests
{
    private const string DeferReason =
        "Qwen3MoeHybrid IQ3 host parity is deferred (audit H3 Option C). " +
        "Two prerequisites are missing: " +
        "(1) VulkanQwen3MoeHybridTransformerModel has no BuildFromPrebuiltWeights factory " +
        "(only BuildFromGguf), and a synthetic IQ3 GGUF for the hybrid 40+-tensor-per-layer " +
        "layout is half a session of GGUF-writer scaffolding; " +
        "(2) VulkanQwen3MoeHybridWeights.DeviceQuantTypeFor does not yet list IQ3_S / IQ3_XXS, " +
        "so the IQ3 dispatch arm wired in commit 48d65fe is unreachable until the upload-path " +
        "predicate is added (mirroring the dense host's VulkanWeights). " +
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
