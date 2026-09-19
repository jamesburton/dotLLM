using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Vulkan NemotronH hybrid forward pass against the CPU
/// reference. Builds a synthetic mini NemotronH "model" with all-F32 random projections
/// owned in unmanaged memory by the test fixture, loads it on both backends, runs
/// <c>Forward</c>, and asserts the last-token logits row matches.
/// </summary>
/// <remarks>
/// All projection weights are F32, sidestepping quant-handling complexity for this first
/// cut. Drift comes from F32 reduction-order differences between the CPU and Vulkan
/// matmul / scan / softmax pipelines; tolerance is abs 5e-3 / rel 1e-3 on last-token
/// logits, matching the MoE/MLA forward parity tests.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHTransformerModelForwardTests
{
    private const int HiddenSize = 16;
    private const int VocabSize = 8;
    private const int HeadDim = 8;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int IntermediateSize = 24;
    private const int DInner = 16;
    private const int DConv = 4;
    private const int DState = 8;
    private const int NGroup = 2;
    private const int NHead = 2;       // d_inner / n_head = 8 = head_dim_ssm
    private const int MaxSeqLen = 16;

    // Q8_0 quant-fixture dimensions — every contraction axis is a multiple of the
    // Q8_0 group size (32), so every projection lands on the matmul_q8_0 fast path
    // when the source is quantised. Bigger hidden / intermediate / d_inner; bigger
    // attention NumHeads so the attention O projection's contraction axis
    // (NumHeads * HeadDim) is also a multiple of 32. NumKvHeads stays at 2 (GQA
    // repeat factor 2).
    private const int Q8HiddenSize = 32;
    private const int Q8IntermediateSize = 32;
    private const int Q8DInner = 32;
    // d_inner / n_head = 8, matches the F32 case so head-dim-sensitive math is
    // identical.
    private const int Q8NHead = 4;
    // Q attention heads — NumHeads * HeadDim = 4 * 8 = 32 ⇒ Q8_0-friendly o_proj.
    private const int Q8NumAttentionHeads = 4;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void Forward_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 3, seed: 7);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 13);
    }

    // ── Q8_0 projection-weight parity tests ──────────────────────────────────────────
    // Same model shapes as the F32 cases above, but with HiddenSize / IntermediateSize /
    // DInner all bumped to Q8_0-compatible (multiple-of-32) dims so every projection's
    // contraction axis satisfies the Q8_0 group-size requirement and lands on the
    // matmul_q8_0 / matmul_q8_0_gemm fast paths. Both backends consume identical Q8_0
    // bytes (the fixture quantises F32 → Q8_0 once via MatMul.QuantizeF32ToQ8_0 and hands
    // the same byte buffer to CPU and Vulkan), so the only drift source is reduction-
    // order noise in the kernels — same 5e-3 / 1e-3 tolerance as the F32 cases.

    [SkippableFact]
    public void Forward_Q8_0_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 142, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 3, seed: 107, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 113, quantize: true);
    }

    // ── Lazy KV-cache growth (issue #437) ────────────────────────────────────────────
    // The KV buffers no longer allocate maxSeqLen eagerly; they start at a small capacity
    // and are reallocated by VulkanNemotronHKvCache.EnsureCapacity, which the model calls
    // before it opens its command buffer.
    //
    // Why the cases below are shaped the way they are:
    //   * Reallocation frees Vulkan buffer handles, and the driver RECYCLES those handles.
    //     The kernels' DescriptorSetCache is keyed on the handle, so a stale set can be
    //     re-bound to a freed buffer and the decode goes correct-then-silently-ZEROS.
    //     A single growth cannot catch that: the first growth's freed handles have not yet
    //     been recycled into anything the cache would collide with. These tests therefore
    //     decode far enough to grow TWICE (2 -> 4 -> 8 rows) and assert on the steps AFTER
    //     the second growth, where the recycled handles are live.
    //   * The oracle is an eagerly-allocated cache (initialCapacity == maxSeqLen) driven by
    //     the SAME model instance shapes, so the comparison is BIT-EXACT: identical kernels,
    //     identical dispatch order, identical reduction order. Any tolerance here would hide
    //     exactly the zeros this is hunting for.

    [SkippableFact]
    public void Decode_GrowingKvCache_MatchesEagerCache_AcrossTwoGrowths()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertGrowingKvCacheMatchesEager(kinds, seed: 437, decodeSteps: 30, initialCapacity: 1);
    }

    [SkippableFact]
    public void Decode_GrowingKvCache_MatchesEagerCache_TwoAttentionLayers()
    {
        // Two attention layers => two KV slots, so a growth that mis-wires one slot's
        // buffer (or leaves one descriptor set stale) diverges even if the other is fine.
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Attention };
        AssertGrowingKvCacheMatchesEager(kinds, seed: 4371, decodeSteps: 30, initialCapacity: 1);
    }

    [SkippableFact]
    public void Prefill_ThenDecode_GrowingKvCache_MatchesEagerCache()
    {
        // Prefill overshoots the initial capacity in one step (positions 0..3 against a
        // capacity of 2), exercising the multi-doubling loop, then decode grows again.
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertGrowingKvCacheMatchesEager(kinds, seed: 4372, decodeSteps: 24, initialCapacity: 1, prefillLen: 4);
    }

    // ── F16 KV store (issue #437) ────────────────────────────────────────────────────
    // Opt-in; F32 stays the default until F16 has been measured. The store is half the size and
    // the attention kernels read it natively through the _kvf16 shader variants, so the bytes
    // moved per decode step halve too. That is the whole reason this is NOT a dequantise-into-
    // scratch design: dequantising would read the F16 store, write an F32 scratch and then read
    // the scratch again — 2.5x the fp32 baseline's traffic on a path that is bandwidth-bound.

    [SkippableFact]
    public void Decode_F16KvCache_GrowingMatchesEager_AcrossTwoGrowths()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertGrowingKvCacheMatchesEager(kinds, seed: 4375, decodeSteps: 30, initialCapacity: 1,
            dataType: VulkanKvDataType.F16);
    }

    [SkippableFact]
    public void Prefill_ThenDecode_F16KvCache_GrowingMatchesEager()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Attention };
        AssertGrowingKvCacheMatchesEager(kinds, seed: 4376, decodeSteps: 24, initialCapacity: 1, prefillLen: 4,
            dataType: VulkanKvDataType.F16);
    }

    /// <summary>
    /// F16 KV against the F32 path, same prompt and same weights.
    /// </summary>
    /// <remarks>
    /// <para><b>Tolerance basis.</b> The only difference is that each cached K/V element is rounded
    /// to binary16, whose unit roundoff is 2^-11 ~= 4.9e-4 relative. That enters twice: through the
    /// Q.K scores, where a headDim-wide dot product of random-signed terms loses roughly
    /// sqrt(headDim) to cancellation (sum |q_d k_d| is ~sqrt(d) times the result, so the relative
    /// error is amplified by that factor), and through the weighted-V accumulation, where the
    /// softmax weights form a convex combination so the error stays ~2^-11. Propagating the score
    /// error through softmax and the remaining layers puts an abs 1e-2 / rel 1e-2 bound on the final
    /// logits at the right order — about 20x the fp32 reduction-order noise the CPU-parity tests
    /// above already tolerate (5e-3 / 1e-3), and far below the ~1 scale of the logits themselves.
    /// It is a bound on THIS fixture, not a quality claim: perplexity on a real model is a separate
    /// measurement and has not been done.</para>
    ///
    /// <para><b>Why maxDiff must be non-zero.</b> A flag or a capability check proves nothing about
    /// which shader ran. Asserting that the F16 result DIFFERS from the F32 one proves the packed
    /// store and the _kvf16 kernels were exercised — had the plumbing fallen back to the fp32 path,
    /// the outputs would be bit-identical and this test would pass vacuously.</para>
    ///
    /// <para><b>Range.</b> binary16 saturates at 65504. The test checks the cached magnitudes stay
    /// well inside that, so a tolerance pass cannot be masking an overflow to infinity.</para>
    /// </remarks>
    [SkippableFact]
    public void Decode_F16KvCache_MatchesF32Path_WithinF16RoundingBound()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var layerKinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Attention };
        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed: 4377, quantize: false);
        var config = fixture.Config;
        int vocabSize = config.VocabSize;
        int hiddenSize = config.HiddenSize;
        const int prefillLen = 4, decodeSteps = 8;
        int maxSeqLen = prefillLen + decodeSteps;

        float[][] Run(VulkanKvDataType dt)
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            using var kv = model.CreateKvCache(maxSeqLen, maxSeqLen, dt);
            using var ssm = model.CreateSsmState();
            Assert.Equal(dt, kv.DataType);

            var perStep = new List<float[]>();
            int pos = 0;
            int[] pTok = new int[prefillLen], pPos = new int[prefillLen];
            for (int i = 0; i < prefillLen; i++) { pTok[i] = i % vocabSize; pPos[i] = pos++; }
            using (ITensor logits = model.Forward(pTok, pPos, deviceId: -1, kv, ssm))
                perStep.Add(CopyLogits(logits));

            for (int st = 0; st < decodeSteps; st++)
            {
                int[] tok = { (st * 5 + 2) % vocabSize };
                int[] posArr = { pos++ };
                using ITensor logits = model.Forward(tok, posArr, deviceId: -1, kv, ssm);
                perStep.Add(CopyLogits(logits));
            }

            // Cached magnitudes must stay inside the binary16 range, or a "within tolerance"
            // result could be hiding a saturation to infinity.
            int live = kv.CurrentLength * config.NumKvHeads * config.HeadDim;
            float[] cachedK = new float[live];
            if (dt == VulkanKvDataType.F16) kv.DownloadKeysF16ForTest(0, cachedK);
            else kv.DownloadKeysForTest(0, cachedK);
            float maxAbs = 0f;
            foreach (float x in cachedK) maxAbs = MathF.Max(maxAbs, MathF.Abs(x));
            Assert.True(maxAbs > 0f, "cached K is all zeros — the cache was never written");
            Assert.True(maxAbs < 1000f,
                $"cached |K| max {maxAbs} is uncomfortably close to the binary16 limit 65504");

            return perStep.ToArray();
        }

        float[][] f32 = Run(VulkanKvDataType.F32);
        float[][] f16 = Run(VulkanKvDataType.F16);

        const float absTol = 1e-2f, relTol = 1e-2f;
        float maxDiff = 0f;
        for (int step = 0; step < f32.Length; step++)
        {
            for (int c = 0; c < vocabSize; c++)
            {
                float a = f32[step][c], b = f16[step][c];
                float diff = MathF.Abs(a - b);
                maxDiff = MathF.Max(maxDiff, diff);
                float bar = absTol + relTol * MathF.Abs(a);
                Assert.True(diff <= bar,
                    $"step={step} (0 = prefill), col={c}: f32={a:F6} vs f16={b:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }

        Assert.True(maxDiff > 0f,
            "F16 and F32 logits are bit-identical, so the F16 store / _kvf16 kernels did not actually "
            + "run — the plumbing fell back to the fp32 path and this test would pass vacuously.");
    }

    /// <summary>
    /// The opt-in must be reachable from the shipping call path, not just from a test that names
    /// the dtype. The CLI creates caches through the one-argument <c>CreateKvCache(maxSeqLen)</c>,
    /// so that overload is what has to honour <c>DOTLLM_VK_KV_F16</c> — otherwise the "option"
    /// exists only in code nobody calls.
    /// </summary>
    [SkippableFact]
    public void DefaultCreateKvCache_HonoursF16EnvironmentOptIn()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var layerKinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed: 4378, quantize: false);
        using var device = VulkanDevice.Create();
        using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
            device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, fixture.OutputQuantType, fixture.Config.VocabSize,
            fixture.Config.HiddenSize, fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);

        string? previous = Environment.GetEnvironmentVariable("DOTLLM_VK_KV_F16");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_KV_F16", null);
            Assert.Equal(VulkanKvDataType.F32, VulkanNemotronHTransformerModel.DefaultKvDataType);
            using (var def = model.CreateKvCache(MaxSeqLen))
                Assert.Equal(VulkanKvDataType.F32, def.DataType);

            Environment.SetEnvironmentVariable("DOTLLM_VK_KV_F16", "1");
            Assert.Equal(VulkanKvDataType.F16, VulkanNemotronHTransformerModel.DefaultKvDataType);
            using (var opted = model.CreateKvCache(MaxSeqLen))
            {
                Assert.Equal(VulkanKvDataType.F16, opted.DataType);
                Assert.NotNull(opted.F16Kernels);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_KV_F16", previous);
        }
    }

    [SkippableFact]
    public void F16Cache_HalvesAllocatedBytes()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernels = VulkanKvF16Kernels.Create(device, spvDir);

        var slotMap = new[] { 0, -1 };
        using var f32 = new VulkanNemotronHKvCache(device, slotMap, 1, numKvHeads: 4, headDim: 256,
            maxSeqLen: 1024, initialCapacity: 1024);
        using var f16 = new VulkanNemotronHKvCache(device, slotMap, 1, numKvHeads: 4, headDim: 256,
            maxSeqLen: 1024, initialCapacity: 1024, VulkanKvDataType.F16, kernels, ownsKernels: false);

        Assert.Equal(f32.AllocatedBytes / 2, f16.AllocatedBytes);
        Assert.True(f16.IsF16);
        Assert.False(f32.IsF16);
        Assert.NotNull(f16.F16Kernels);
        Assert.Null(f32.F16Kernels);
    }

    [SkippableFact]
    public void RecordUpdate_BeyondCapacity_Throws()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();

        // Layer 0 is attention, layer 1 is not — slot map mirrors the model's construction.
        var cache = new VulkanNemotronHKvCache(device, new[] { 0, -1 }, attentionLayerCount: 1,
            numKvHeads: 2, headDim: 8, maxSeqLen: 64, initialCapacity: 4);
        using (cache)
        {
            Assert.Equal(4, cache.Capacity);
            Assert.Equal(64, cache.MaxLength);

            // EnsureCapacity is the documented pre-recording step; growth is by doubling,
            // and it must report "reallocated" so the caller invalidates descriptor caches.
            Assert.False(cache.EnsureCapacity(4));
            Assert.True(cache.EnsureCapacity(5));
            Assert.Equal(8, cache.Capacity);
            Assert.True(cache.EnsureCapacity(20));
            Assert.Equal(32, cache.Capacity);
            Assert.Throws<ArgumentOutOfRangeException>(() => cache.EnsureCapacity(65));

            Assert.Equal(4, VulkanNemotronHKvCache.RequiredLengthFor(new[] { 0, 1, 2, 3 }));
            Assert.Equal(18, VulkanNemotronHKvCache.RequiredLengthFor(new[] { 17 }));
        }
    }

    private void AssertGrowingKvCacheMatchesEager(
        HybridLayerKind[] layerKinds, int seed, int decodeSteps, int initialCapacity, int prefillLen = 1,
        VulkanKvDataType dataType = VulkanKvDataType.F32)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed, quantize: false);
        var config = fixture.Config;
        int vocabSize = config.VocabSize;
        int hiddenSize = config.HiddenSize;
        int maxSeqLen = prefillLen + decodeSteps;

        float[][] Run(bool growing)
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir);
            using var kv = growing
                ? model.CreateKvCache(maxSeqLen, initialCapacity, dataType)
                : model.CreateKvCache(maxSeqLen, maxSeqLen, dataType);   // pre-#437 eager allocation
            Assert.Equal(dataType, kv.DataType);
            using var ssm = model.CreateSsmState();

            Assert.Equal(growing ? Math.Min(initialCapacity, maxSeqLen) : maxSeqLen, kv.Capacity);

            var perStep = new List<float[]>();
            int pos = 0;

            int[] prefillTokens = new int[prefillLen];
            int[] prefillPositions = new int[prefillLen];
            for (int i = 0; i < prefillLen; i++) { prefillTokens[i] = i % vocabSize; prefillPositions[i] = pos++; }
            using (ITensor logits = model.Forward(prefillTokens, prefillPositions, deviceId: -1, kv, ssm))
                perStep.Add(CopyLogits(logits));

            for (int s = 0; s < decodeSteps; s++)
            {
                int[] tok = { (s * 3 + 1) % vocabSize };
                int[] posArr = { pos++ };
                using ITensor logits = model.Forward(tok, posArr, deviceId: -1, kv, ssm);
                perStep.Add(CopyLogits(logits));
            }

            if (growing)
            {
                // The point of the test: capacity must actually have moved at least twice.
                Assert.True(kv.Capacity >= initialCapacity * 4,
                    $"expected at least two growths from {initialCapacity}, ended at {kv.Capacity}");
                Assert.True(kv.Capacity <= maxSeqLen);
            }
            return perStep.ToArray();
        }

        float[][] eager = Run(growing: false);
        float[][] grown = Run(growing: true);

        Assert.Equal(eager.Length, grown.Length);
        for (int step = 0; step < eager.Length; step++)
        {
            for (int c = 0; c < vocabSize; c++)
            {
                // Bit-exact: same kernels, same order, only the buffer allocation differs.
                Assert.True(eager[step][c].Equals(grown[step][c]),
                    $"layers={string.Join(',', layerKinds)}, step={step} (0 = prefill), col={c}: "
                    + $"eager={eager[step][c]:R} vs grown={grown[step][c]:R}. A step that is correct early "
                    + "and zero later is the descriptor-cache handle-aliasing failure (issue #437).");
            }
        }

        // Explicitly assert the post-second-growth tail is not silently zeroed — the
        // handle-aliasing symptom would otherwise have to be inferred from the diff above.
        float[] last = grown[^1];
        Assert.Contains(last, v => v != 0f);
    }

    private void AssertVulkanMatchesCpu(
        HybridLayerKind[] layerKinds, int seqLen, int seed, bool quantize = false)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed, quantize);
        var config = fixture.Config;

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                config, fixture.Layers, fixture.OutputNormWeight,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(vocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < vocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * vocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"layers={string.Join(',', layerKinds)}, seqLen={seqLen}, quant={quantize}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a randomly-generated NemotronH "model" in unmanaged memory: every projection
    /// weight (token embed, output, plus per-layer ssm/attn/ffn projections) is allocated
    /// 64-byte-aligned via <see cref="NativeMemory.AlignedAlloc"/>. Both the CPU and Vulkan
    /// model factories take the same <see cref="NemotronHLayerWeights"/> array so the
    /// backends consume identical bytes.
    /// </summary>
    /// <remarks>
    /// When <c>quantize=true</c> every projection (SSM in/out, attn Q/K/V/O, FFN up/down,
    /// LM head) is generated as F32, then quantised in place to Q8_0 via
    /// <see cref="MatMul.QuantizeF32ToQ8_0"/> and the resulting Q8_0 byte buffer is what
    /// both backends consume. The quant-mode fixture uses the larger
    /// <c>Q8HiddenSize</c> / <c>Q8IntermediateSize</c> / <c>Q8DInner</c> dims so every
    /// contraction axis is a multiple of 32 — a hard requirement of the Q8_0 matmul kernels.
    /// Token embedding stays F32 in both modes (the embedding gather is byte-offset
    /// vkCmdCopyBuffer, which only works with a contiguous F32 layout — same convention
    /// as <c>VulkanWeights</c>).
    /// </remarks>
    private sealed unsafe class NemotronHFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;
        public QuantizationType OutputQuantType;

        // Active fixture dimensions (depend on quantize flag).
        private int _hiddenSize;
        private int _intermediateSize;
        private int _dInner;
        private int _nHead;
        private int _numAttentionHeads;
        private bool _quantize;

        public static NemotronHFixtureBuilder Build(
            HybridLayerKind[] layerKinds, int seed, bool quantize = false)
        {
            var b = new NemotronHFixtureBuilder();
            b.BuildInternal(layerKinds, seed, quantize);
            return b;
        }

        private void BuildInternal(HybridLayerKind[] layerKinds, int seed, bool quantize)
        {
            int numLayers = layerKinds.Length;
            var rng = new Random(seed);

            _quantize = quantize;
            _hiddenSize = quantize ? Q8HiddenSize : HiddenSize;
            _intermediateSize = quantize ? Q8IntermediateSize : IntermediateSize;
            _dInner = quantize ? Q8DInner : DInner;
            _nHead = quantize ? Q8NHead : NHead;
            _numAttentionHeads = quantize ? Q8NumAttentionHeads : NumHeads;
            OutputQuantType = quantize ? QuantizationType.Q8_0 : QuantizationType.F32;

            // Per-layer kinds: hard-code per-layer KV head count and FFN length to match.
            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = layerKinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = layerKinds[i] == HybridLayerKind.Ffn ? _intermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = layerKinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLength,
            };

            var ssmConfig = new MambaSsmConfig(
                DConv: DConv, DInner: _dInner, DState: DState, NGroup: NGroup, NHead: _nHead);

            Config = new ModelConfig
            {
                Architecture = Architecture.NemotronH,
                VocabSize = VocabSize,
                HiddenSize = _hiddenSize,
                IntermediateSize = _intermediateSize,
                NumLayers = numLayers,
                NumAttentionHeads = _numAttentionHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                HybridLayout = layout,
                SsmConfig = ssmConfig,
                ChatTemplate = null,
            };

            // Token embedding [vocab, hidden] — always F32 (see class remarks).
            TokenEmbedPtr = AllocAndFillUniform(VocabSize * _hiddenSize, rng, amplitude: 0.05f);

            // Output norm + LM head.
            OutputNormWeight = FillNormVec(_hiddenSize, rng);
            // LM head: contraction axis is hiddenSize. In quant mode we Q8-quantise it.
            OutputWeightPtr = AllocProjection(VocabSize, _hiddenSize, rng, OutputQuantType);

            // Per-layer weights.
            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float[] attnNorm = FillNormVec(_hiddenSize, rng);
                switch (layerKinds[i])
                {
                    case HybridLayerKind.Ssm:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Ssm = BuildSsm(rng, ssmConfig),
                        };
                        break;
                    case HybridLayerKind.Attention:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Attention = BuildAttn(rng, NumKvHeads),
                        };
                        break;
                    case HybridLayerKind.Ffn:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Ffn = BuildFfn(rng, _intermediateSize),
                        };
                        break;
                    default:
                        throw new InvalidOperationException();
                }
            }
        }

        private NemotronHSsmWeights BuildSsm(Random rng, MambaSsmConfig ssm)
        {
            int convDim = ssm.ConvDim;
            int inProjDim = ssm.InputProjectionDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;

            // ssm_in: contraction axis is hiddenSize, output axis is in_proj_dim.
            nint inWeight = AllocProjection(inProjDim, _hiddenSize, rng, qt);
            // ssm_out: contraction axis is d_inner, output axis is hiddenSize.
            nint outWeight = AllocProjection(_hiddenSize, ssm.DInner, rng, qt);

            float[] conv1dWeight = FillRandom(ssm.DConv * convDim, rng, amplitude: 0.1f);
            float[] conv1dBias = FillRandom(convDim, rng, amplitude: 0.1f);
            // A: forced negative so exp(dt*A) decays.
            float[] a = NegativeRandom(ssm.NHead, rng);
            float[] d = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] dtBias = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            // ssm_norm.weight is [d_inner] (group-wise gain).
            float[] normWeight = FillNormVec(ssm.DInner, rng);

            return new NemotronHSsmWeights
            {
                InWeight = inWeight,
                InQuantType = qt,
                InInputDim = _hiddenSize,
                InOutputDim = inProjDim,
                Conv1dWeight = conv1dWeight,
                Conv1dBias = conv1dBias,
                A = a,
                D = d,
                DtBias = dtBias,
                NormWeight = normWeight,
                OutWeight = outWeight,
                OutQuantType = qt,
                OutInputDim = ssm.DInner,
                OutOutputDim = _hiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = _numAttentionHeads * HeadDim;
            int kvOut = numKvHeads * HeadDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHAttentionWeights
            {
                QWeight = AllocProjection(qOut, _hiddenSize, rng, qt),
                QQuantType = qt,
                QInputDim = _hiddenSize, QOutputDim = qOut,
                KWeight = AllocProjection(kvOut, _hiddenSize, rng, qt),
                KQuantType = qt,
                KInputDim = _hiddenSize, KOutputDim = kvOut,
                VWeight = AllocProjection(kvOut, _hiddenSize, rng, qt),
                VQuantType = qt,
                VInputDim = _hiddenSize, VOutputDim = kvOut,
                OWeight = AllocProjection(_hiddenSize, qOut, rng, qt),
                OQuantType = qt,
                OInputDim = qOut, OOutputDim = _hiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediateSize)
        {
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHFfnWeights
            {
                UpWeight = AllocProjection(intermediateSize, _hiddenSize, rng, qt),
                UpQuantType = qt,
                UpInputDim = _hiddenSize, UpOutputDim = intermediateSize,
                DownWeight = AllocProjection(_hiddenSize, intermediateSize, rng, qt),
                DownQuantType = qt,
                DownInputDim = intermediateSize, DownOutputDim = _hiddenSize,
                IntermediateSize = intermediateSize,
            };
        }

        /// <summary>
        /// Allocates a projection matrix [outputDim, inputDim] in the requested quant
        /// format. F32 → contiguous F32 row-major. Q8_0 → contiguous Q8_0 row-stride
        /// (rowBytes = (inputDim/32)*34) row-major, generated by quantising a freshly
        /// drawn F32 row in scratch space then writing the Q8_0 bytes to the final
        /// allocation. Both CPU and Vulkan paths consume the same bytes.
        /// </summary>
        private nint AllocProjection(int outputDim, int inputDim, Random rng, QuantizationType qt)
        {
            if (qt == QuantizationType.F32)
            {
                return AllocAndFillUniform(outputDim * inputDim, rng, amplitude: 0.05f);
            }
            if (qt != QuantizationType.Q8_0)
                throw new NotSupportedException($"AllocProjection only supports F32/Q8_0, got {qt}.");
            if ((inputDim % 32) != 0)
                throw new InvalidOperationException(
                    $"Q8_0 requires inputDim multiple of 32 (got {inputDim}). Bump fixture dims.");

            int rowBytes = (inputDim / 32) * 34;
            long totalBytes = (long)rowBytes * outputDim;
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
            _allocs.Add(dst);

            // Per-row scratch — quantise one row at a time so we don't stage the whole
            // matrix in F32 unnecessarily.
            float[] rowScratch = new float[inputDim];
            for (int row = 0; row < outputDim; row++)
            {
                for (int j = 0; j < inputDim; j++)
                    rowScratch[j] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
                fixed (float* srcPtr = rowScratch)
                {
                    byte* rowDst = (byte*)dst + (long)row * rowBytes;
                    MatMul.QuantizeF32ToQ8_0(srcPtr, rowDst, inputDim);
                }
            }
            return dst;
        }

        private nint AllocAndFillUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++)
            {
                dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            }
            return ptr;
        }

        private static float[] FillRandom(int count, Random rng, float amplitude)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return arr;
        }

        private static float[] FillNormVec(int count, Random rng)
        {
            // Centred at 1.0 with small jitter, like RMSNorm gamma weights.
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            // A forced negative so the SSM scan exp(dt*A) decays.
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
            {
                NativeMemory.AlignedFree((void*)p);
            }
            _allocs.Clear();
        }
    }
}
