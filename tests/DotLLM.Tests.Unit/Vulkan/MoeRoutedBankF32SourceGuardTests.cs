using System;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Guards the routed-expert F32 upload fallback against a <b>null host source</b> (#427).
/// </summary>
/// <remarks>
/// <para>
/// <b>The defect.</b> <c>UploadRoutedBankWhole</c>'s F32 branch reads
/// <c>f32Experts[e]</c> once per expert. For a <b>non-MLA</b> quant-expert model whose routed
/// bank type has no Vulkan kernel (MXFP4 / Q4_0 / Q4_1 — #344 units 2-4), the loader
/// <c>LoadQuantExpertMoeLayer</c> deliberately leaves <c>W1</c>/<c>W2</c>/<c>W3</c> as
/// <b>null placeholders</b> and never allocates host F32. So the fallback has nothing to read.
/// </para>
/// <para>
/// <b>Why this is worth a guard rather than a footnote.</b> Reading a zero pointer here does not
/// crash — it uploads a null/garbage matrix that is then silently multiplied into the forward
/// pass. Silent corruption is the failure class this repo keeps paying for (#373, #392, #261,
/// #416), and it is invisible in exactly the models most likely to hit it.
/// </para>
/// <para>
/// <b>Why the existing preflight does not cover it.</b> <c>PlanMoeF32HostDequant</c> /
/// <c>ResolveMoeBankResidency</c> both early-return unless <c>config.MlaConfig is not null</c>,
/// so the DeepSeek/MLA family is protected and every other MoE model is not. That asymmetry is
/// the whole of this issue.
/// </para>
/// <para>
/// <b>Discrimination (per #417).</b> These cases were demonstrated to fail before the guard
/// existed: the null-source cases returned normally, which is precisely the silent-corruption
/// behaviour. They are not shape-only assertions — the negative cases below pin that a
/// <i>valid</i> F32 source and every non-F32 type are still accepted, so a guard that simply
/// threw unconditionally would fail this class.
/// </para>
/// </remarks>
public sealed class MoeRoutedBankF32SourceGuardTests
{
    private const string Bank = "blk.3.ffn_gate_exps.weight";

    /// <summary>A null array is the <c>LoadQuantExpertMoeLayer</c> shape — no host F32 at all.</summary>
    [Fact]
    public void F32Fallback_WithNullExpertArray_ThrowsNamingTheBank()
    {
        var ex = Assert.Throws<NotSupportedException>(
            () => VulkanWeights.ValidateRoutedBankF32Source(
                QuantizationType.F32, f32Experts: null, numE: 8, bankName: Bank));

        // The message has to be actionable at the point of failure: which bank, and why.
        Assert.Contains(Bank, ex.Message, StringComparison.Ordinal);
        Assert.Contains("#344", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// A populated array with one null slot — the partial case. Worth separating from the
    /// all-null case because an <c>is null</c> check on the array alone would pass this and
    /// still corrupt exactly one expert, which is harder to notice than all of them.
    /// </summary>
    [Fact]
    public void F32Fallback_WithOneNullExpertSlot_ThrowsNamingTheExpert()
    {
        var experts = new nint[4];
        for (int e = 0; e < experts.Length; e++) experts[e] = 0x1000 + e;
        experts[2] = 0;

        var ex = Assert.Throws<NotSupportedException>(
            () => VulkanWeights.ValidateRoutedBankF32Source(
                QuantizationType.F32, experts, numE: 4, bankName: Bank));

        Assert.Contains(Bank, ex.Message, StringComparison.Ordinal);
        Assert.Contains("2", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>An array shorter than the expert count would read out of bounds.</summary>
    [Fact]
    public void F32Fallback_WithShortExpertArray_Throws()
    {
        var experts = new nint[2];
        for (int e = 0; e < experts.Length; e++) experts[e] = 0x1000 + e;

        Assert.Throws<NotSupportedException>(
            () => VulkanWeights.ValidateRoutedBankF32Source(
                QuantizationType.F32, experts, numE: 8, bankName: Bank));
    }

    /// <summary>
    /// NEGATIVE CONTROL: a fully-populated F32 source is the legitimate fallback and must still
    /// be accepted. Without this, a guard that threw unconditionally would pass the cases above.
    /// </summary>
    [Fact]
    public void F32Fallback_WithValidExpertPointers_DoesNotThrow()
    {
        var experts = new nint[8];
        for (int e = 0; e < experts.Length; e++) experts[e] = 0x1000 + e;

        VulkanWeights.ValidateRoutedBankF32Source(QuantizationType.F32, experts, numE: 8, bankName: Bank);
    }

    /// <summary>
    /// NEGATIVE CONTROL: non-F32 banks never read <c>f32Experts</c> — they upload the raw
    /// contiguous GGUF range — so a null array is legitimate there and must not throw. This is
    /// the common case (every kept-native Q8_0 / Q4_K bank), so a guard that got it wrong would
    /// break far more than it fixed.
    /// </summary>
    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q6_K)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void NonF32Bank_WithNullExpertArray_DoesNotThrow(QuantizationType qt)
    {
        VulkanWeights.ValidateRoutedBankF32Source(qt, f32Experts: null, numE: 8, bankName: Bank);
    }
}
