using System.Reflection;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// CPU-vs-Vulkan forward parity for the routed-MoE K-quant raw-view path added by #191,
/// plus preflight-check coverage for <c>VulkanWeights.CanSkipMoeF32HostDequant</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Two fixtures, two purposes.</b> This file uses TWO different synthetic GGUF
/// fixtures because the two things #191 touches live on different sides of an
/// UNRELATED, pre-existing gap in test coverage discovered while writing this suite:
/// </para>
/// <list type="bullet">
///   <item>
///     <b>Mixtral-style (dense GQA attention + MoE FFN)</b> — used for the full
///     CPU-vs-Vulkan <see cref="Forward_RoutedKQuant_MatchesCpuReference"/> parity test.
///     The CPU oracle here is <c>TransformerWeights.LoadQuantExpertMoeLayer</c> →
///     <c>DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp</c>, which ALWAYS reads routed experts
///     straight from the raw GGUF quant view (no F32 host dequant, no
///     <c>skipF32MoeDequant</c> flag involved) and already has explicit Q4_K/Q5_K/Q6_K
///     cases — a solid, pre-existing ground truth. The Vulkan side resolves the SAME
///     raw view via the newly-extended <c>VulkanWeights.MoeRoutedRawDeviceQuantType</c>
///     and dispatches through the (previously Gemma4-only-gated)
///     <c>MoeIndexedMatmulQ4_KF32Kernel</c>/<c>Q5_KF32Kernel</c>/<c>Q6_KF32Kernel</c>.
///   </item>
///   <item>
///     <b>DeepSeek-V2-style (MLA + MoE)</b> — used ONLY for the
///     <c>CanSkipMoeF32HostDequant_*</c> preflight tests (item 4), which never call
///     <c>.Forward()</c>. <c>skipF32MoeDequant</c> exclusively threads through
///     <c>TransformerWeights.LoadMlaLayer</c> → <c>LoadDeepSeekMoeLayer</c>, so testing
///     the preflight gate needs an MLA+MoE config specifically.
///   </item>
/// </list>
/// <para>
/// <b>Why not one full MLA+MoE forward test?</b> A synthetic DeepSeek-V2-style GGUF
/// with a full <c>.Forward()</c> call (CPU OR Vulkan — the crash reproduces on the CPU
/// side alone, before Vulkan is even reached) reliably crashes the test host process,
/// reproducible even with a plain-F32 (no quantization at all) 1-layer dense-only MLA
/// fixture at <c>HiddenSize=256</c>. This is a pre-existing bug in the CPU MLA forward
/// path unrelated to #191 (no existing test exercises a full forward on any synthetic
/// MLA GGUF — <c>DeepSeekV2GgufLoadTests</c> only checks weight loading, and
/// <c>VulkanTransformerModelMlaForwardTests</c> only exercises a dense, non-GGUF,
/// non-MoE safetensors-loaded MLA model at <c>HiddenSize=16</c>). Chasing that crash is
/// out of scope for this issue; the Mixtral-style fixture below sidesteps MLA entirely
/// while still exercising the exact same <c>VulkanWeights.UploadMoeLayer</c> /
/// <c>MoeRoutedRawDeviceQuantType</c> / <c>RecordMoeIndexedMatmul</c> code #191 changes
/// (that code path does not care whether the model is MLA or dense-attention).
/// </para>
/// <para>
/// Reference patterns: <c>VulkanQwen3MoeMoeUploadQ4KResidentTests</c> (upload/resolution
/// level parity — the Q4_K/Q5_K/Q6_K indexed-matmul kernels themselves are already
/// proven correct there and in the Q5_K/Q6_K sibling files) and
/// <c>VulkanTransformerModelMoeQ8_0ForwardTests</c> (full CPU-vs-Vulkan forward parity
/// for the generic, non-hybrid MoE path — mirrored here for K-quant instead of Q8_0).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeKQuantRoutedForwardTests : IDisposable
{
    // ── Mixtral-style (dense attention + MoE) fixture — full forward parity ──────────
    // Both HiddenSize (gate/up contraction) and MoeIntermediate (down contraction) must
    // be multiples of 256 (Q4_K/Q5_K/Q6_K super-block size).
    private const int MxHiddenSize = 256;
    private const int MxNumLayers = 1;         // single MoE layer (decoder_sparse_step=1 default → MoE)
    private const int MxNumHeads = 4;
    private const int MxNumKvHeads = 2;        // GQA
    private const int MxHeadDim = 64;          // NumHeads * HeadDim == HiddenSize
    private const int MxVocabSize = 8;
    private const int MxMoeIntermediate = 256;
    private const int MxNumExperts = 2;
    private const int MxNumExpertsPerTok = 2;  // both experts always active — deterministic routing

    // ── DeepSeek-V2-style (MLA + MoE) fixture — preflight-only, no forward call ──────
    private const int DsHiddenSize = 256;
    private const int DsNumLayers = 2;         // layer 0 dense, layer 1 MoE
    private const int DsNumHeads = 2;
    private const int DsVocabSize = 8;
    private const int DsQkNope = 4;
    private const int DsQkRope = 4;            // RoPE pairs — must be even
    private const int DsVHead = 4;
    private const int DsKvLoraRank = 8;
    private const int DsIntermediateSize = 32; // dense-layer FFN width — no alignment need
    private const int DsMoeIntermediate = 256;
    private const int DsNumExperts = 2;
    private const int DsNumExpertsPerTok = 2;
    private const int DsLeadingDenseBlocks = 1;

    // Looser than the repo's standard single-kernel-dispatch parity bar (5e-3 abs /
    // 1e-3 rel, used by e.g. the Q4K/Q5K/Q6K resident-upload tests that compare ONE
    // matmul dispatch in isolation). This test chains a FULL forward — RoPE, GQA
    // attention softmax, RMSNorm ×2, router softmax, gate/up/down K-quant matmuls,
    // SwiGLU, residual adds — through 3 causal positions. CPU (on-the-fly per-row
    // dequant in DotLLM.Cpu.Kernels.MoeQuantSwiGluMlp) and Vulkan (workgroup-tree
    // reduce in the indexed-matmul shader) both read the exact SAME quantized bytes
    // but accumulate their dot products in a different order; that FP noise compounds
    // through every subsequent nonlinear op (softmax, RMSNorm's rsqrt, SwiGLU) and
    // across causal positions (each token's residual stream carries the prior tokens'
    // rounding forward). Empirically: single-token drift is ~1.05-1.5x the tight
    // single-kernel bar; full 3-token drift was measured up to ~6e-2 absolute, spread
    // evenly across every logit column (not concentrated in one bank/column, which is
    // what a functional dispatch bug would look like) and shrinking sharply with fewer
    // chained positions — the signature of compounding FP noise, not a bug. This bar
    // has ~30% headroom over the worst measured run across Q4_K/Q5_K/Q6_K.
    private const float AbsTol = 8e-2f;
    private const float RelTol = 8e-3f;

    private readonly List<string> _tempFiles = new();

    public void Dispose()
    {
        foreach (string path in _tempFiles)
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, 42)]
    [InlineData(QuantizationType.Q4_K, 43)]
    [InlineData(QuantizationType.Q5_K, 7)]
    [InlineData(QuantizationType.Q5_K, 8)]
    [InlineData(QuantizationType.Q6_K, 101)]
    [InlineData(QuantizationType.Q6_K, 102)]
    // #407: the routed legacy/i-quant pair. End-to-end reachability — the CPU side
    // decodes these via MoeQuantSwiGluMlp's GemvDequantRows fallback (the trusted scalar
    // dequant oracle), the Vulkan side via the new moe_indexed kernels. Bit-level layout
    // correctness is proven separately on real GGUF bytes in
    // RealGgufMoeIndexedRoutedBankParityTests; this asserts the dispatch is wired.
    [InlineData(QuantizationType.Q5_0, 407)]
    [InlineData(QuantizationType.Q5_0, 408)]
    [InlineData(QuantizationType.IQ4_NL, 409)]
    [InlineData(QuantizationType.IQ4_NL, 410)]
    public void Forward_RoutedKQuant_MatchesCpuReference(QuantizationType routedQuantType, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = WriteMixtralFixture(routedQuantType, seed);

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];

        // ── CPU oracle: MoeQuantSwiGluMlp reads the raw K-quant bytes directly
        //    (already has explicit Q4_K/Q5_K/Q6_K support — no change from #191). ──
        float[] cpuLogits;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

            // Sanity: the fixture actually carries a raw K-quant view of the requested
            // type on the MoE layer, and the CPU loader routed it through the
            // quant-experts path (not a hidden F32 upcast) — otherwise this test would
            // silently prove nothing about the new resolver/dispatch.
            var moe = cpuWeights.Layers[0].Moe;
            Assert.NotNull(moe);
            Assert.True(moe!.UseQuantExperts);
            Assert.True(moe.HasRawQuantView);
            Assert.Equal(routedQuantType, moe.GateExpsRawQt);
            Assert.Equal(routedQuantType, moe.UpExpsRawQt);
            Assert.Equal(routedQuantType, moe.DownExpsRawQt);

            using var cpuModel = TransformerModel.BuildFromPrebuiltWeights(cpuWeights, config);
            using ITensor logits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan: resolves the same raw view to the K-quant device type and
        //    dispatches through the new indexed-matmul kernel. ──
        float[] vkLogits;
        QuantizationType resolvedW1Qt, resolvedW2Qt, resolvedW3Qt;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var vkModel = VulkanTransformerModel.LoadFromGguf(device, gguf, config, spvDir);

            (resolvedW1Qt, resolvedW2Qt, resolvedW3Qt) = ReadResolvedMoeBankQuantTypes(vkModel, layerIndex: 0);

            using ITensor logits = vkModel.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(MxVocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        // The whole point of #191: the routed banks must resolve to the K-quant type,
        // NOT fall back to F32 — proves the new dispatch wiring was actually exercised.
        Assert.Equal(routedQuantType, resolvedW1Qt);
        Assert.Equal(routedQuantType, resolvedW2Qt);
        Assert.Equal(routedQuantType, resolvedW3Qt);

        int lastRow = tokenIds.Length - 1;
        for (int c = 0; c < MxVocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * MxVocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"quant={routedQuantType}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// Preflight (#191 item 4) must recognize a fully K-quant-routed DeepSeek-V2-style
    /// (MLA + MoE) fixture as safe for <c>skipF32MoeDequant: true</c> — every routed
    /// bank on the (single) MoE layer resolves to the requested K-quant type, none fall
    /// back to F32. Does not call <c>.Forward()</c> — see the type doc for why.
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, 42)]
    [InlineData(QuantizationType.Q5_K, 7)]
    [InlineData(QuantizationType.Q6_K, 101)]
    public void CanSkipMoeF32HostDequant_TrueForFullyKQuantRoutedFixture(QuantizationType routedQuantType, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        string path = WriteDeepSeekV2Fixture(routedQuantType, seed);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        bool canSkip = InvokeCanSkipMoeF32HostDequant(device, gguf, config);
        Assert.True(canSkip);
    }

    /// <summary>
    /// Negative case: a non-MLA config (skipF32MoeDequant never threads through for
    /// non-MLA loaders) must always report <c>false</c>, regardless of quant type —
    /// guards the preflight's architecture gate.
    /// </summary>
    [SkippableFact]
    public void CanSkipMoeF32HostDequant_FalseWithoutMlaConfig()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        string path = WriteDeepSeekV2Fixture(QuantizationType.Q4_K, seed: 1);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata) with { MlaConfig = null };
        using var device = VulkanDevice.Create();

        bool canSkip = InvokeCanSkipMoeF32HostDequant(device, gguf, config);
        Assert.False(canSkip);
    }

    // VulkanWeights is `internal` (InternalsVisibleTo covers this test assembly), so
    // CanSkipMoeF32HostDequant can be called directly — no reflection needed.
    private static bool InvokeCanSkipMoeF32HostDequant(VulkanDevice device, GgufFile gguf, ModelConfig config)
        => VulkanWeights.CanSkipMoeF32HostDequant(device, gguf, config);

    /// <summary>
    /// #327's actual scenario, and the gap a prior review found: the OTHER preflight tests
    /// in this file only exercise fully-K-quant-resident fixtures (all three banks resolve
    /// the same way), so they never distinguish per-bank resolution
    /// (<see cref="VulkanWeights.ResolveMoeBankResidency"/>) from the OLD model-global
    /// behavior (<c>if (w1Qt==F32 || w2Qt==F32 || w3Qt==F32) return false</c> collapsed
    /// onto every bank). This fixture mixes quant types on ONE layer — gate/up Q4_K
    /// (resident-capable), down Q4_1 (no MoE-indexed Vulkan kernel in this worktree yet) — so
    /// gate/up staying independently resident is only observable if the per-bank aggregation
    /// is real. Reverting <c>ResolveMoeBankResidency</c> to AND across all three banks makes
    /// this test FAIL (verified manually — see task-2-report.md fix-round-1 section).
    /// </summary>
    /// <remarks>
    /// The blocking bank was Q5_0 until #407 gave Q5_0 (and IQ4_NL) a MoE-indexed kernel
    /// plus resolver wiring; it is now Q4_1, which is still #344 Unit 4 and therefore
    /// still has no routed kernel. Swapping the type — rather than weakening the assertion
    /// — is what keeps this test discriminating: it must always name a type the resolver
    /// genuinely cannot keep resident.
    /// </remarks>
    [SkippableFact]
    public void ResolveMoeBankResidency_IsPerBank_PartiallyResidentLayerKeepsGateUpResident()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        string path = WriteDeepSeekV2MixedResidencyFixture();
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        var residency = VulkanWeights.ResolveMoeBankResidency(device, gguf, config);

        // Layer DsLeadingDenseBlocks (index 1) is the fixture's single MoE layer.
        Assert.True(residency.TryGetValue(DsLeadingDenseBlocks, out var bank));
        Assert.True(bank.Gate, "ffn_gate_exps (Q4_K, 256-aligned) must resolve resident.");
        Assert.True(bank.Up, "ffn_up_exps (Q4_K, 256-aligned) must resolve resident.");
        Assert.False(bank.Down, "ffn_down_exps (Q4_1) has no MoE-indexed Vulkan kernel yet — must NOT resolve resident.");

        // The model-wide preflight still correctly reports false overall (down blocks the
        // full-skip decision) — the per-bank signal above is what #327 adds beyond this.
        Assert.False(VulkanWeights.CanSkipMoeF32HostDequant(device, gguf, config));
    }

    /// <summary>
    /// The #326 preflight's footprint must equal what the #327 per-bank load will ACTUALLY
    /// allocate. These two landed on separate branches — #326 wrote its accounting while the
    /// skip was still model-global ("all 78 banks get F32'd"), and #327 then made the skip
    /// per-bank without the accounting following it. Charging every bank of every MoE layer
    /// over-reports the footprint, which matters because
    /// <see cref="VulkanWeights.ThrowIfMoeF32HostDequantUnaffordable"/> turns that number into
    /// a hard refusal — an over-report can refuse a load the per-bank path just made fit.
    /// </summary>
    /// <remarks>
    /// Discriminating by construction: this fixture's single MoE layer has gate/up resident
    /// (Q4_K) and only down falling back (Q5_0), so the correct answer is ONE bank's worth of
    /// F32 and the old all-or-nothing answer is THREE — a 3x margin, not a tolerance question.
    /// </remarks>
    [SkippableFact]
    public void PlanMoeF32HostDequant_ChargesOnlyTheBanksThatActuallyFallBack()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        string path = WriteDeepSeekV2MixedResidencyFixture();
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        var plan = VulkanWeights.PlanMoeF32HostDequant(device, gguf, config);

        // One expert bank, F32: numExperts * sizeof(float) * moeIntermediate * hiddenSize.
        const long OneBankF32Bytes = (long)DsNumExperts * sizeof(float) * DsMoeIntermediate * DsHiddenSize;

        Assert.False(plan.CanSkip);
        Assert.Equal(3, plan.TotalBanks);
        var fallback = Assert.Single(plan.Fallbacks);
        Assert.Equal("ffn_down_exps.weight", fallback.Bank);
        Assert.Equal(QuantizationType.Q4_1, fallback.Quant);

        Assert.Equal(OneBankF32Bytes, plan.HostF32Bytes);
        Assert.NotEqual(3 * OneBankF32Bytes, plan.HostF32Bytes); // the pre-#327 all-or-nothing answer
    }

    /// <summary>
    /// #407 reachability gate: a Q5_0 or IQ4_NL routed expert bank must now resolve
    /// DEVICE-RESIDENT rather than falling back to the host F32 dequant.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the "prove routing by observation, not capability flags" requirement
    /// (#344 rule 7) at the resolver level: <c>ResolveMoeBankResidency</c> is the SAME
    /// predicate <c>UploadMoeLayer</c> uses to choose the upload form, so a true here
    /// means the packed bytes go to the device and <c>RecordMoeIndexedMatmul</c>
    /// dispatches the new kernel. Bit-level correctness of that kernel is proven
    /// separately, against real llama.cpp GGUF bytes, in
    /// <c>RealGgufMoeIndexedRoutedBankParityTests</c> — a shipped kernel that is never
    /// routed to and a routed kernel that decodes wrongly are different failures and
    /// need different tests.
    /// </para>
    /// <para>
    /// Bytes here are random-but-correctly-sized: the resolver inspects only the GGUF
    /// descriptor's quant type and shape and never dereferences them, and no
    /// <c>.Forward()</c> is called.
    /// </para>
    /// </remarks>
    /// <param name="downQuantType">Routed down-bank quant type under test.</param>
    [SkippableTheory]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.IQ4_NL)]
    public void ResolveMoeBankResidency_KeepsQ5_0AndIq4NlRoutedBanksResident(QuantizationType downQuantType)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);

        string path = WriteDeepSeekV2MixedResidencyFixture(downQuantType);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();

        // Guard against the fixture silently not carrying the type under test (#344 rule 4).
        var downDescriptor = Assert.Single(gguf.Tensors,
            t => t.Name == $"blk.{DsLeadingDenseBlocks}.ffn_down_exps.weight");
        Assert.Equal(downQuantType, downDescriptor.QuantizationType);

        var residency = VulkanWeights.ResolveMoeBankResidency(device, gguf, config);
        Assert.True(residency.TryGetValue(DsLeadingDenseBlocks, out var bank));
        Assert.True(bank.Gate, "ffn_gate_exps (Q4_K, 256-aligned) must resolve resident.");
        Assert.True(bank.Up, "ffn_up_exps (Q4_K, 256-aligned) must resolve resident.");
        Assert.True(bank.Down,
            $"ffn_down_exps ({downQuantType}) must resolve resident after #407 — without the "
            + "MoeRoutedRawDeviceQuantType wiring the new kernel is unreachable and every "
            + "expert bank expands to F32.");

        // With all three banks resident the whole-model preflight can skip the host
        // F32 dequant entirely — the memory outcome #407 exists for.
        var plan = VulkanWeights.PlanMoeF32HostDequant(device, gguf, config);
        Assert.True(plan.CanSkip);
        Assert.Empty(plan.Fallbacks);
        Assert.Equal(0, plan.HostF32Bytes);
    }

    /// <summary>
    /// Reflects into the Vulkan model's PRIVATE <c>_weights</c> field (the only part not
    /// reachable via InternalsVisibleTo) to read the on-device quant type each routed
    /// bank (gate/down/up) actually resolved to for the given layer — the same fields
    /// <c>RecordMoeIndexedMatmul</c> dispatches on.
    /// </summary>
    private static (QuantizationType w1, QuantizationType w2, QuantizationType w3) ReadResolvedMoeBankQuantTypes(
        VulkanTransformerModel model, int layerIndex)
    {
        var weightsField = typeof(VulkanTransformerModel).GetField("_weights",
            BindingFlags.Instance | BindingFlags.NonPublic);
        var weights = (VulkanWeights)weightsField!.GetValue(model)!;
        var moe = weights.Layers[layerIndex].Moe;
        Assert.NotNull(moe);
        return (moe!.Value.W1DeviceQuantType, moe.Value.W2DeviceQuantType, moe.Value.W3DeviceQuantType);
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Writes a synthetic minimal Mixtral-shaped GGUF (standard dense GQA attention,
    /// one all-MoE layer, no shared experts) to a temp file. Every tensor is F32 EXCEPT
    /// the three routed-expert banks (<c>ffn_gate_exps</c>/<c>ffn_up_exps</c>/
    /// <c>ffn_down_exps</c>), which are real <paramref name="routedQuantType"/>-quantized
    /// bytes — the exact on-disk shape a real <c>*_K_M</c>-quantized Mixtral/DeepSeek
    /// -family checkpoint carries.
    /// </summary>
    private string WriteMixtralFixture(QuantizationType routedQuantType, int seed)
    {
        var b = new GgufTestData(version: 3);
        var rng = new Random(seed);
        int qOut = MxNumHeads * MxHeadDim;
        int kvOut = MxNumKvHeads * MxHeadDim;

        b.AddString("general.architecture", "mixtral");
        b.AddUInt32("mixtral.embedding_length", (uint)MxHiddenSize);
        b.AddUInt32("mixtral.block_count", (uint)MxNumLayers);
        b.AddUInt32("mixtral.attention.head_count", (uint)MxNumHeads);
        b.AddUInt32("mixtral.attention.head_count_kv", (uint)MxNumKvHeads);
        b.AddUInt32("mixtral.attention.key_length", (uint)MxHeadDim);
        b.AddUInt32("mixtral.context_length", 16);
        b.AddFloat32("mixtral.attention.layer_norm_rms_epsilon", 1e-5f);
        b.AddUInt32("mixtral.vocab_size", (uint)MxVocabSize);
        b.AddFloat32("mixtral.rope.freq_base", 10000.0f);
        b.AddUInt32("mixtral.rope.dimension_count", (uint)MxHeadDim);
        b.AddUInt32("mixtral.expert_count", (uint)MxNumExperts);
        b.AddUInt32("mixtral.expert_used_count", (uint)MxNumExpertsPerTok);
        b.AddUInt32("mixtral.expert_feed_forward_length", (uint)MxMoeIntermediate);

        AddF32Tensor(b, "token_embd.weight", [MxHiddenSize, MxVocabSize], rng);
        AddF32Tensor(b, "output_norm.weight", [MxHiddenSize], rng, center: 1.0f, jitter: 0.05f);
        AddF32Tensor(b, "output.weight", [MxHiddenSize, MxVocabSize], rng);

        for (int i = 0; i < MxNumLayers; i++)
        {
            string p = $"blk.{i}";
            AddF32Tensor(b, $"{p}.attn_norm.weight", [MxHiddenSize], rng, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.ffn_norm.weight", [MxHiddenSize], rng, center: 1.0f, jitter: 0.05f);

            AddF32Tensor(b, $"{p}.attn_q.weight", [MxHiddenSize, qOut], rng);
            AddF32Tensor(b, $"{p}.attn_k.weight", [MxHiddenSize, kvOut], rng);
            AddF32Tensor(b, $"{p}.attn_v.weight", [MxHiddenSize, kvOut], rng);
            AddF32Tensor(b, $"{p}.attn_output.weight", [qOut, MxHiddenSize], rng);

            AddF32Tensor(b, $"{p}.ffn_gate_inp.weight", [MxHiddenSize, MxNumExperts], rng);
            AddExpertBankQuant(b, $"{p}.ffn_gate_exps.weight",
                k: MxHiddenSize, m: MxMoeIntermediate, numExperts: MxNumExperts, routedQuantType, rng);
            AddExpertBankQuant(b, $"{p}.ffn_up_exps.weight",
                k: MxHiddenSize, m: MxMoeIntermediate, numExperts: MxNumExperts, routedQuantType, rng);
            AddExpertBankQuant(b, $"{p}.ffn_down_exps.weight",
                k: MxMoeIntermediate, m: MxHiddenSize, numExperts: MxNumExperts, routedQuantType, rng);
        }

        string path = b.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    /// <summary>
    /// Writes a synthetic minimal DeepSeek-V2-Lite-shaped GGUF (monolithic-Q MLA,
    /// layer 0 dense / layer 1 MoE) to a temp file, mirroring
    /// <c>DeepSeekV2GgufLoadTests.WriteFixture</c>'s recipe (F32 there) with the three
    /// routed-expert banks swapped for real <paramref name="routedQuantType"/>-quantized
    /// bytes. Used ONLY by the <c>CanSkipMoeF32HostDequant_*</c> preflight tests, which
    /// never call <c>.Forward()</c> — see the type doc for why a full MLA forward isn't
    /// exercised here.
    /// </summary>
    private string WriteDeepSeekV2Fixture(QuantizationType routedQuantType, int seed)
    {
        var b = new GgufTestData(version: 3);
        var rng = new Random(seed);

        int qkHead = DsQkNope + DsQkRope;
        int qTotal = DsNumHeads * qkHead;
        int kvAOut = DsKvLoraRank + DsQkRope;
        int kvBOut = DsNumHeads * (DsQkNope + DsVHead);
        int oInput = DsNumHeads * DsVHead;

        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", (uint)DsHiddenSize);
        b.AddUInt32("deepseek2.block_count", (uint)DsNumLayers);
        b.AddUInt32("deepseek2.feed_forward_length", (uint)DsIntermediateSize);
        b.AddUInt32("deepseek2.attention.head_count", (uint)DsNumHeads);
        b.AddUInt32("deepseek2.attention.head_count_kv", (uint)DsNumHeads);
        b.AddUInt32("deepseek2.context_length", 16);
        b.AddFloat32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        b.AddUInt32("deepseek2.vocab_size", (uint)DsVocabSize);
        b.AddFloat32("deepseek2.rope.freq_base", 10000.0f);
        b.AddUInt32("deepseek2.rope.dimension_count", (uint)DsQkRope);

        b.AddUInt32("deepseek2.attention.q_lora_rank", 0);   // monolithic Q (V2-Lite convention)
        b.AddUInt32("deepseek2.attention.kv_lora_rank", (uint)DsKvLoraRank);
        b.AddUInt32("deepseek2.attention.key_length", (uint)(DsQkNope + DsQkRope));
        b.AddUInt32("deepseek2.attention.value_length", (uint)DsVHead);

        b.AddUInt32("deepseek2.expert_count", (uint)DsNumExperts);
        b.AddUInt32("deepseek2.expert_used_count", (uint)DsNumExpertsPerTok);
        b.AddUInt32("deepseek2.expert_shared_count", 0);
        b.AddUInt32("deepseek2.expert_feed_forward_length", (uint)DsMoeIntermediate);
        b.AddUInt32("deepseek2.leading_dense_block_count", (uint)DsLeadingDenseBlocks);

        AddF32Tensor(b, "token_embd.weight", [DsHiddenSize, DsVocabSize], rng);
        AddF32Tensor(b, "output_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);
        AddF32Tensor(b, "output.weight", [DsHiddenSize, DsVocabSize], rng);

        for (int i = 0; i < DsNumLayers; i++)
        {
            string p = $"blk.{i}";

            AddF32Tensor(b, $"{p}.attn_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.ffn_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);

            AddF32Tensor(b, $"{p}.attn_q.weight", [DsHiddenSize, qTotal], rng);
            AddF32Tensor(b, $"{p}.attn_kv_a_mqa.weight", [DsHiddenSize, kvAOut], rng);
            AddF32Tensor(b, $"{p}.attn_kv_a_norm.weight", [DsKvLoraRank], rng, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.attn_kv_b.weight", [DsKvLoraRank, kvBOut], rng);
            AddF32Tensor(b, $"{p}.attn_output.weight", [oInput, DsHiddenSize], rng);

            if (i < DsLeadingDenseBlocks)
            {
                AddF32Tensor(b, $"{p}.ffn_gate.weight", [DsHiddenSize, DsIntermediateSize], rng);
                AddF32Tensor(b, $"{p}.ffn_up.weight", [DsHiddenSize, DsIntermediateSize], rng);
                AddF32Tensor(b, $"{p}.ffn_down.weight", [DsIntermediateSize, DsHiddenSize], rng);
            }
            else
            {
                AddF32Tensor(b, $"{p}.ffn_gate_inp.weight", [DsHiddenSize, DsNumExperts], rng);
                AddExpertBankQuant(b, $"{p}.ffn_gate_exps.weight",
                    k: DsHiddenSize, m: DsMoeIntermediate, numExperts: DsNumExperts, routedQuantType, rng);
                AddExpertBankQuant(b, $"{p}.ffn_up_exps.weight",
                    k: DsHiddenSize, m: DsMoeIntermediate, numExperts: DsNumExperts, routedQuantType, rng);
                AddExpertBankQuant(b, $"{p}.ffn_down_exps.weight",
                    k: DsMoeIntermediate, m: DsHiddenSize, numExperts: DsNumExperts, routedQuantType, rng);
            }
        }

        string path = b.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    /// <summary>
    /// Sibling of <see cref="WriteDeepSeekV2Fixture"/> that mixes routed-bank quant types on
    /// the ONE MoE layer instead of using the same type for all three: gate/up are Q4_K
    /// (resident-capable — 256-aligned on <see cref="DsHiddenSize"/>), down is Q5_0 (no
    /// MoE-indexed Vulkan kernel exists in this worktree yet, so it must NOT resolve resident). This is
    /// the #327 motivating shape (DeepSeek-V2-Lite Q4_K_M ships mixed-quant routed banks) and
    /// is what distinguishes true per-bank resolution from the old model-global AND — a test
    /// built only on <see cref="WriteDeepSeekV2Fixture"/>'s uniform-type fixtures cannot tell
    /// the two apart. Down's bytes are never dequantized by the tests that consume this
    /// fixture (preflight-only, no <c>.Forward()</c>), so they are random-but-correctly-sized
    /// rather than a faithful Q5_0 quantization.
    /// </summary>
    /// <param name="downQuantType">
    /// Quant type for the routed <c>ffn_down_exps</c> bank. Defaults to Q4_1, the type
    /// the per-bank-residency tests need (still no routed kernel); the #407 tests pass
    /// Q5_0 / IQ4_NL to assert the opposite outcome on the same fixture shape.
    /// </param>
    private string WriteDeepSeekV2MixedResidencyFixture(
        QuantizationType downQuantType = QuantizationType.Q4_1)
    {
        var b = new GgufTestData(version: 3);
        var rng = new Random(0x327);

        int qkHead = DsQkNope + DsQkRope;
        int qTotal = DsNumHeads * qkHead;
        int kvAOut = DsKvLoraRank + DsQkRope;
        int kvBOut = DsNumHeads * (DsQkNope + DsVHead);
        int oInput = DsNumHeads * DsVHead;

        b.AddString("general.architecture", "deepseek2");
        b.AddUInt32("deepseek2.embedding_length", (uint)DsHiddenSize);
        b.AddUInt32("deepseek2.block_count", (uint)DsNumLayers);
        b.AddUInt32("deepseek2.feed_forward_length", (uint)DsIntermediateSize);
        b.AddUInt32("deepseek2.attention.head_count", (uint)DsNumHeads);
        b.AddUInt32("deepseek2.attention.head_count_kv", (uint)DsNumHeads);
        b.AddUInt32("deepseek2.context_length", 16);
        b.AddFloat32("deepseek2.attention.layer_norm_rms_epsilon", 1e-6f);
        b.AddUInt32("deepseek2.vocab_size", (uint)DsVocabSize);
        b.AddFloat32("deepseek2.rope.freq_base", 10000.0f);
        b.AddUInt32("deepseek2.rope.dimension_count", (uint)DsQkRope);

        b.AddUInt32("deepseek2.attention.q_lora_rank", 0);
        b.AddUInt32("deepseek2.attention.kv_lora_rank", (uint)DsKvLoraRank);
        b.AddUInt32("deepseek2.attention.key_length", (uint)(DsQkNope + DsQkRope));
        b.AddUInt32("deepseek2.attention.value_length", (uint)DsVHead);

        b.AddUInt32("deepseek2.expert_count", (uint)DsNumExperts);
        b.AddUInt32("deepseek2.expert_used_count", (uint)DsNumExpertsPerTok);
        b.AddUInt32("deepseek2.expert_shared_count", 0);
        b.AddUInt32("deepseek2.expert_feed_forward_length", (uint)DsMoeIntermediate);
        b.AddUInt32("deepseek2.leading_dense_block_count", (uint)DsLeadingDenseBlocks);

        AddF32Tensor(b, "token_embd.weight", [DsHiddenSize, DsVocabSize], rng);
        AddF32Tensor(b, "output_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);
        AddF32Tensor(b, "output.weight", [DsHiddenSize, DsVocabSize], rng);

        for (int i = 0; i < DsNumLayers; i++)
        {
            string p = $"blk.{i}";

            AddF32Tensor(b, $"{p}.attn_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.ffn_norm.weight", [DsHiddenSize], rng, center: 1.0f, jitter: 0.05f);

            AddF32Tensor(b, $"{p}.attn_q.weight", [DsHiddenSize, qTotal], rng);
            AddF32Tensor(b, $"{p}.attn_kv_a_mqa.weight", [DsHiddenSize, kvAOut], rng);
            AddF32Tensor(b, $"{p}.attn_kv_a_norm.weight", [DsKvLoraRank], rng, center: 1.0f, jitter: 0.05f);
            AddF32Tensor(b, $"{p}.attn_kv_b.weight", [DsKvLoraRank, kvBOut], rng);
            AddF32Tensor(b, $"{p}.attn_output.weight", [oInput, DsHiddenSize], rng);

            if (i < DsLeadingDenseBlocks)
            {
                AddF32Tensor(b, $"{p}.ffn_gate.weight", [DsHiddenSize, DsIntermediateSize], rng);
                AddF32Tensor(b, $"{p}.ffn_up.weight", [DsHiddenSize, DsIntermediateSize], rng);
                AddF32Tensor(b, $"{p}.ffn_down.weight", [DsIntermediateSize, DsHiddenSize], rng);
            }
            else
            {
                AddF32Tensor(b, $"{p}.ffn_gate_inp.weight", [DsHiddenSize, DsNumExperts], rng);
                AddExpertBankQuant(b, $"{p}.ffn_gate_exps.weight",
                    k: DsHiddenSize, m: DsMoeIntermediate, numExperts: DsNumExperts, QuantizationType.Q4_K, rng);
                AddExpertBankQuant(b, $"{p}.ffn_up_exps.weight",
                    k: DsHiddenSize, m: DsMoeIntermediate, numExperts: DsNumExperts, QuantizationType.Q4_K, rng);
                AddExpertBankRawLegacy(b, $"{p}.ffn_down_exps.weight",
                    k: DsMoeIntermediate, m: DsHiddenSize, numExperts: DsNumExperts, downQuantType, rng);
            }
        }

        string path = b.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    /// <summary>
    /// Writes one fused-experts tensor as <paramref name="qt"/>-tagged bytes of the
    /// CORRECT size for that 32-element legacy/i-quant block format, but with random
    /// content — sufficient for <see cref="VulkanWeights.ResolveMoeBankResidency"/> /
    /// <c>MoeRoutedRawDeviceQuantType</c>, which only inspect the descriptor's quant type
    /// and shape and never dereference the raw bytes. Not a faithful quantization — do
    /// not use with a <c>.Forward()</c> test.
    /// <para>
    /// The mixed-residency fixture used Q5_0 here until #407 gave Q5_0 a routed
    /// (MoE-indexed) kernel and resolver wiring; the default is now Q4_1, which is #344
    /// Unit 4 and still has none — the property that fixture actually needs.
    /// </para>
    /// </summary>
    private static void AddExpertBankRawLegacy(
        GgufTestData b, string name, int k, int m, int numExperts, QuantizationType qt, Random rng)
    {
        const int groupSize = 32;
        int blockBytes = qt switch
        {
            QuantizationType.Q4_1 => 20,
            QuantizationType.Q5_0 => 22,
            QuantizationType.IQ4_NL => 18,
            _ => throw new NotSupportedException($"Unsupported legacy routed quant type: {qt}"),
        };
        if (k % groupSize != 0)
            throw new ArgumentException($"k={k} must be a multiple of {groupSize} for {qt}.", nameof(k));

        int rowBytes = (k / groupSize) * blockBytes;
        var all = new byte[(long)numExperts * m * rowBytes];
        rng.NextBytes(all);
        b.AddTensor(name, [k, m, numExperts], (uint)qt, all);
    }

    /// <summary>
    /// Writes one fused-experts tensor ([<paramref name="k"/>, <paramref name="m"/>,
    /// <paramref name="numExperts"/>] — K innermost, GGUF convention) as REAL
    /// <paramref name="qt"/>-quantized bytes: each expert's [<paramref name="m"/>,
    /// <paramref name="k"/>] slab is generated as random F32 then quantized via the
    /// matching test fixture quantizer, matching llama.cpp's on-disk block layout
    /// byte-for-byte (verified elsewhere by the Q4K/Q5K/Q6K resident-upload tests).
    /// </summary>
    private static void AddExpertBankQuant(
        GgufTestData b, string name, int k, int m, int numExperts, QuantizationType qt, Random rng)
    {
        int rowBytes = qt switch
        {
            QuantizationType.Q4_K => (k / Q4KFixture.Q4KGroupSize) * Q4KFixture.Q4KBlockBytes,
            QuantizationType.Q5_K => (k / Q5KFixture.Q5KGroupSize) * Q5KFixture.Q5KBlockBytes,
            QuantizationType.Q6_K => (k / Q6KFixture.Q6KGroupSize) * Q6KFixture.Q6KBlockBytes,
            // #407: 32-element legacy / non-linear formats.
            QuantizationType.Q5_0 => (k / 32) * 22,
            QuantizationType.IQ4_NL => (k / 32) * 18,
            _ => throw new NotSupportedException($"Unsupported routed quant type for this fixture: {qt}"),
        };
        var all = new byte[(long)numExperts * m * rowBytes];
        for (int e = 0; e < numExperts; e++)
        {
            float[] f32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
            byte[] q = qt switch
            {
                QuantizationType.Q4_K => Q4KFixture.QuantizeRows(f32, m, k),
                QuantizationType.Q5_K => Q5KFixture.QuantizeRows(f32, m, k),
                QuantizationType.Q6_K => Q6KFixture.QuantizeRows(f32, m, k),
                // Q5_0 goes through the PRODUCTION encoder (DotLLM.Cpu.Kernels.Quantize)
                // rather than a test-local one, so the fixture cannot silently share a
                // layout bug with a hand-written encoder the way Q3_K's did (#311).
                QuantizationType.Q5_0 => QuantizeRowsWithCpu(f32, m, k, QuantizationType.Q5_0, rowBytes),
                QuantizationType.IQ4_NL => Iq4Fixture.QuantizeRowsIq4Nl(f32, m, k),
                _ => throw new NotSupportedException(),
            };
            Buffer.BlockCopy(q, 0, all, e * m * rowBytes, q.Length);
        }
        b.AddTensor(name, [k, m, numExperts], (uint)qt, all);
    }

    /// <summary>
    /// Row-wise quantization through the production CPU encoder
    /// (<c>DotLLM.Cpu.Kernels.Quantize.FromFloat32</c>), used for the formats that have
    /// no dedicated test fixture quantizer.
    /// </summary>
    private static byte[] QuantizeRowsWithCpu(float[] src, int m, int k, QuantizationType qt, int rowBytes)
    {
        var dest = new byte[(long)m * rowBytes];
        for (int row = 0; row < m; row++)
        {
            DotLLM.Cpu.Kernels.Quantize.FromFloat32(
                src.AsSpan(row * k, k), k, qt,
                dest.AsSpan(row * rowBytes, rowBytes));
        }
        return dest;
    }

    private static void AddF32Tensor(GgufTestData b, string name, int[] shape, Random rng,
        float amplitude = 0.1f, float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        foreach (int d in shape) n *= d;
        byte[] bytes = new byte[n * sizeof(float)];
        for (long i = 0; i < n; i++)
        {
            float raw = (float)(rng.NextDouble() * 2.0 - 1.0);
            float v = jitter > 0f ? center + jitter * raw : amplitude * raw;
            System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(
                bytes.AsSpan((int)(i * sizeof(float)), sizeof(float)), v);
        }
        b.AddTensor(name, shape, quantType: 0, bytes);
    }
}
