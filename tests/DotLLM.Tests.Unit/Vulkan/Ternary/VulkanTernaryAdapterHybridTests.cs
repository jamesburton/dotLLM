using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Fixture = DotLLM.Tests.Unit.Vulkan.VulkanHybridLegacyQuantForwardTests.LegacyQuantFixture;

namespace DotLLM.Tests.Unit.Vulkan.Ternary;

/// <summary>
/// Device-side ternary residual adapters on the Vulkan qwen35 hybrid-dense model (#1711): attached to
/// <see cref="VulkanQwen3HybridDenseTransformerModel"/> the adapter steers the logits the way the CPU hook steers the CPU hybrid on the same bytes
/// (the steering delta agrees, not just the base logits), detaching restores the pre-attach logits bit for bit, and the hook survives a scratch
/// re-allocation (descriptor-cache invalidation) and the fused batch forward. Synthetic 2-layer hybrid (layer 0 Gated-DeltaNet, layer 1 attention).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanTernaryAdapterHybridTests(ITestOutputHelper output)
{
    private const int Vocab = 8, KvHeads = 2, HeadDim = 64, MaxSeq = 16;
    private const QuantizationType Qt = QuantizationType.Q4_1;
    private static readonly int[] Ids = [1, 5, 2, 7, 3, 0];

    private static float[] Copy(ITensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.Shape.ElementCount).ToArray();

    private static double MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double m = 0; for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m;
    }

    private static void AssertBits(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(a[i]) == BitConverter.SingleToInt32Bits(b[i]), $"{what}: element {i} {a[i]} != {b[i]}");
    }

    private static float[] CpuRun(Qwen3HybridDenseTransformerModel m, int[] ids)
    {
        m.ResetSequenceState();
        using var kv = new SimpleKvCache(m.AttentionLayerCount, KvHeads, HeadDim, MaxSeq);
        using ITensor t = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        return Copy(t);
    }

    private static float[] VkRun(VulkanQwen3HybridDenseTransformerModel m, int[] ids)
    {
        using var kv = m.CreateKvCache(MaxSeq);
        m.ResetSequenceState();
        using ITensor t = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        return Copy(t);
    }

    [SkippableFact]
    public void HybridDense_AdapterSteersLikeCpu_DetachIsBitExact_SurvivesScratchRegrowth()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        using var fx = Fixture.Build(1711, Qt);
        int d = fx.DenseConfig.HiddenSize;
        var pkgs = new[]
        {
            VulkanTernaryAdapterKernelTests.Synth(8, d, 77, 3f, 0f, layer: 0),   // after the Gated-DeltaNet layer
            VulkanTernaryAdapterKernelTests.Synth(8, d, 78, 3f, 0f, layer: 1),   // after the (final) full-attention layer
        };

        using var cpu = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32, fx.OutputWeightPtr, Qt, Vocab, d);
        using var device = VulkanDevice.Create();
        using var vk = VulkanQwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            device, fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.OutputWeightPtr, Qt, Vocab, d, fx.TokenEmbedPtr, QuantizationType.F32, spv);

        // A one-token forward first so the prefill below grows the scratch buffers with the modifier already attached.
        using var mod = VulkanTernaryResidualModifier.Create(device, spv, pkgs);
        mod.ValidateFor(d, fx.DenseConfig.NumLayers);
        vk.ResidualModifier = mod;
        _ = VkRun(vk, [4]);
        vk.ResidualModifier = null;

        float[] vkBase = VkRun(vk, Ids);
        vk.ResidualModifier = mod;
        float[] vkWith = VkRun(vk, Ids);
        vk.ResidualModifier = null;
        AssertBits(vkBase, VkRun(vk, Ids), "vulkan detach restores the pre-attach logits");

        float[] cpuBase = CpuRun(cpu, Ids);
        using var hook = new TernaryResidualHook(d);
        foreach (var p in pkgs) hook.Add(p);
        cpu.ResidualModifier = hook;
        float[] cpuWith = CpuRun(cpu, Ids);
        cpu.ResidualModifier = null;

        double effectCpu = MaxDiff(cpuBase, cpuWith), effectVk = MaxDiff(vkBase, vkWith), noise = MaxDiff(cpuBase, vkBase);
        output.WriteLine($"max|cpu_base-vk_base|={noise:E3}  adapter effect cpu={effectCpu:E3} vk={effectVk:E3}  max|cpu_with-vk_with|={MaxDiff(cpuWith, vkWith):E3}");
        Assert.True(effectVk > 1e-3, $"adapter did not steer the Vulkan hybrid (effect {effectVk:E3}); hook not firing?");
        Assert.True(effectCpu > 20 * noise, $"control arm insensitive: cpu effect {effectCpu:E3} vs backend noise {noise:E3}");
        // The steering DELTA must agree between backends (so a hook that fires at the wrong layer, or not on every row, cannot pass on base parity alone).
        var dCpu = new float[cpuBase.Length]; var dVk = new float[cpuBase.Length];
        for (int i = 0; i < dCpu.Length; i++) { dCpu[i] = cpuWith[i] - cpuBase[i]; dVk[i] = vkWith[i] - vkBase[i]; }
        double deltaErr = MaxDiff(dCpu, dVk);
        output.WriteLine($"steering delta max|d_cpu-d_vk|={deltaErr:E3} (effect {effectCpu:E3})");
        Assert.True(deltaErr < 0.25 * effectCpu, $"steering delta differs between backends: {deltaErr:E3} vs effect {effectCpu:E3}");
    }

    [SkippableFact]
    public void HybridDense_AdapterAtEachLayerKind_AndSingleTokenDecode_MatchCpuDelta()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        using var fx = Fixture.Build(1715, Qt);
        int d = fx.DenseConfig.HiddenSize;
        using var cpu = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32, fx.OutputWeightPtr, Qt, Vocab, d);
        using var device = VulkanDevice.Create();
        using var vk = VulkanQwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            device, fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.OutputWeightPtr, Qt, Vocab, d, fx.TokenEmbedPtr, QuantizationType.F32, spv);
        foreach (int layer in new[] { 0, 1 })
            foreach (var ids in new[] { Ids, new[] { 6 } })
            {
                var pkg = VulkanTernaryAdapterKernelTests.Synth(8, d, 90 + (ulong)layer, 3f, 0f, layer);
                using var mod = VulkanTernaryResidualModifier.Create(device, spv, [pkg]);
                using var hook = new TernaryResidualHook(d);
                hook.Add(pkg);
                float[] vb = VkRun(vk, ids), cb = CpuRun(cpu, ids);
                vk.ResidualModifier = mod; cpu.ResidualModifier = hook;
                float[] vw = VkRun(vk, ids), cw = CpuRun(cpu, ids);
                vk.ResidualModifier = null; cpu.ResidualModifier = null;
                double effect = MaxDiff(cb, cw), noise = MaxDiff(cb, vb);
                var dc = cw.Zip(cb, (a, b) => a - b).ToArray(); var dv = vw.Zip(vb, (a, b) => a - b).ToArray();
                double err = MaxDiff(dc, dv);
                output.WriteLine($"layer {layer} n={ids.Length}: effect cpu={effect:E3} vk={MaxDiff(vb, vw):E3} noise={noise:E3} delta-err={err:E3}");
                Assert.True(MaxDiff(vb, vw) > 1e-3, $"layer {layer} n={ids.Length}: no Vulkan steering");
                Assert.True(err < 0.25 * effect, $"layer {layer} n={ids.Length}: steering delta mismatch {err:E3} vs effect {effect:E3}");
            }
    }

    [SkippableFact]
    public void HybridDense_FusedBatchForward_AppliesTheAdapterToEverySequence()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spv);
        using var fx = Fixture.Build(1716, Qt);
        int d = fx.DenseConfig.HiddenSize;
        using var device = VulkanDevice.Create();
        using var vk = VulkanQwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            device, fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.OutputWeightPtr, Qt, Vocab, d, fx.TokenEmbedPtr, QuantizationType.F32, spv);
        var pkg = VulkanTernaryAdapterKernelTests.Synth(8, d, 95, 3f, 0f, layer: 0);
        using var mod = VulkanTernaryResidualModifier.Create(device, spv, [pkg]);

        int[][] seqs = [Ids, [3, 3, 1], [6, 2, 5, 4]];
        float[][] Batch()
        {
            var kvs = seqs.Select(_ => vk.CreateKvCache(MaxSeq)).ToArray();
            var gdn = seqs.Select(_ => (IGdnState)vk.CreateSequenceState()!).ToArray();
            try
            {
                var reqs = seqs.Select((s, i) => new SequenceForwardRequest
                {
                    TokenIds = s, Positions = Enumerable.Range(0, s.Length).ToArray(), KvCache = kvs[i], GdnState = gdn[i],
                }).ToArray();
                var outs = vk.ForwardBatch(reqs, -1);
                try { return outs.Select(Copy).ToArray(); }
                finally { foreach (var o in outs) o.Dispose(); }
            }
            finally { foreach (var k in kvs) k.Dispose(); foreach (var g in gdn) (g as IDisposable)?.Dispose(); }
        }

        float[][] baseB = Batch();
        vk.ResidualModifier = mod;
        float[][] withB = Batch();
        vk.ResidualModifier = null;
        float[] single = VkRun(vk, Ids);
        vk.ResidualModifier = mod;
        float[] singleWith = VkRun(vk, Ids);
        vk.ResidualModifier = null;
        for (int i = 0; i < seqs.Length; i++)
        {
            double effect = MaxDiff(baseB[i], withB[i]);
            output.WriteLine($"batch seq{i} (n={seqs[i].Length}): adapter effect {effect:E3}");
            Assert.True(effect > 1e-3, $"batch sequence {i} was not steered");
        }
        // The fused batch and the single-sequence forward must steer sequence 0 alike (same delta within backend noise).
        var dB = withB[0].Zip(baseB[0], (a, b) => a - b).ToArray();
        var dS = singleWith.Zip(single, (a, b) => a - b).ToArray();
        double dErr = MaxDiff(dB, dS);
        output.WriteLine($"batch-vs-single steering delta err {dErr:E3}");
        Assert.True(dErr < 0.25 * MaxDiff(single, singleWith), $"batch steering delta err {dErr:E3}");
    }
}
