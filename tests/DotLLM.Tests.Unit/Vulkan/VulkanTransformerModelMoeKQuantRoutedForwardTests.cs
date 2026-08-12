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
    /// (resident-capable), down Q5_0 (no MoE-indexed Vulkan kernel in this worktree yet) — so gate/up
    /// staying independently resident is only observable if the per-bank aggregation is
    /// real. Reverting <c>ResolveMoeBankResidency</c> to AND across all three banks makes
    /// this test FAIL (verified manually — see task-2-report.md fix-round-1 section).
    /// </summary>
    /// <remarks>
    /// "No Vulkan kernel" below means no MoE-INDEXED Q5_0 kernel. #344 added a dense
    /// (non-routed) Vulkan Q5_0 GEMM/GEMV kernel — <see cref="MoeRoutedRawDeviceQuantType"/>
    /// deliberately does not extend to it, since MoE-indexed dispatch is a distinct kernel
    /// family from dense matmul (see <see cref="VulkanWeights.CanKeepBankResident"/>'s XML).
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
        Assert.False(bank.Down, "ffn_down_exps (Q5_0) has no MoE-indexed Vulkan kernel yet — must NOT resolve resident.");

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
        Assert.Equal(QuantizationType.Q5_0, fallback.Quant);

        Assert.Equal(OneBankF32Bytes, plan.HostF32Bytes);
        Assert.NotEqual(3 * OneBankF32Bytes, plan.HostF32Bytes); // the pre-#327 all-or-nothing answer
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
    private string WriteDeepSeekV2MixedResidencyFixture()
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
                AddExpertBankRawQ5_0(b, $"{p}.ffn_down_exps.weight",
                    k: DsMoeIntermediate, m: DsHiddenSize, numExperts: DsNumExperts, rng);
            }
        }

        string path = b.WriteToTempFile();
        _tempFiles.Add(path);
        return path;
    }

    /// <summary>
    /// Writes one fused-experts tensor as Q5_0-tagged bytes of the CORRECT size (22 bytes
    /// per 32-element block) but random content — sufficient for
    /// <see cref="VulkanWeights.ResolveMoeBankResidency"/>/<c>MoeRoutedRawDeviceQuantType</c>,
    /// which only inspect the descriptor's quant type and shape, never dereference the raw
    /// bytes. Not a faithful Q5_0 quantization — do not use with a <c>.Forward()</c> test.
    /// </summary>
    private static void AddExpertBankRawQ5_0(GgufTestData b, string name, int k, int m, int numExperts, Random rng)
    {
        const int q5_0GroupSize = 32;
        const int q5_0BlockBytes = 22;
        if (k % q5_0GroupSize != 0)
            throw new ArgumentException($"k={k} must be a multiple of {q5_0GroupSize} for Q5_0.", nameof(k));

        int rowBytes = (k / q5_0GroupSize) * q5_0BlockBytes;
        var all = new byte[(long)numExperts * m * rowBytes];
        rng.NextBytes(all);
        b.AddTensor(name, [k, m, numExperts], (uint)QuantizationType.Q5_0, all);
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
                _ => throw new NotSupportedException(),
            };
            Buffer.BlockCopy(q, 0, all, e * m * rowBytes, q.Length);
        }
        b.AddTensor(name, [k, m, numExperts], (uint)qt, all);
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
