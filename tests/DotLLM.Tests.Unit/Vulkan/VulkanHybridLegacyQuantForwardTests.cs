using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity for the #921 keep-packed arms: a synthetic 2-layer hybrid model (one GDN layer, one full-attention layer) whose
/// projections are Q4_0 / Q4_1 / Q5_0 / Q5_1, run through BOTH hybrid hosts (MoE-hybrid with F32 routed experts; hybrid-dense with a
/// dense SwiGLU FFN) against the CPU model on the same bytes. Discriminators, in order: (1) the loader must report the packed type on
/// device (a loader mutant that widens fails here, not in the tolerance), (2) the host must have run the legacy-quant arm
/// (<c>LegacyQuantDispatchCount</c>; a dispatch mutant routed to another case fails here or by parity), (3) logits within tolerance of the
/// CPU (a wrong block layout, e.g. the Q5_0 GEMV on Q5_1 bytes, is off by O(1)). The MMVQ decode arm is proven to have run by the
/// MMVQ counter where the device has integer dot product.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanHybridLegacyQuantForwardTests(ITestOutputHelper output)
{
    private const int HiddenSize = 256;
    private const int VocabSize = 8;
    private const int NumAttentionHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 64;
    private const int RopeDim = 32;
    private const int MaxSeqLen = 16;
    private const int NVHead = 8;
    private const int NKHead = 4;
    private const int DState = 32;
    private const int DConv = 4;
    private const int MoeIntermediate = 32;
    private const int SharedIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;
    private const int DenseIntermediate = 96;   // multiple of 32: the dense down projection contracts along it

    // Both backends read identical quantised bytes, so the divergence is kernel arithmetic: the CPU's packed x Q8_1 prefill and the Vulkan
    // dp4a decode round activations to Q8_1 per 32-block, the coopmat prefill holds the dequantised weight in F16. Measured on Strix
    // (seed-fixed): hybrid-dense <= 2.6e-2 abs, MoE-hybrid <= 7.6e-2 at 4 rows (Q4_0; the others <= 1.8e-2) (the top-2-of-4 routing amplifies a logit nudge into an
    // expert flip). A mis-decoded block layout is off by O(1) on logits of O(1) (the Q5_0-on-Q5_1 mutant: max diff 0.6-1.2).
    private const float DenseAbsTol = 3e-2f, DenseRelTol = 3e-2f;
    private const float MoeAbsTol = 1e-1f, MoeRelTol = 5e-2f;

    public static IEnumerable<object[]> Cases()
    {
        foreach (var qt in new[] { QuantizationType.Q4_0, QuantizationType.Q4_1, QuantizationType.Q5_0, QuantizationType.Q5_1 })
            foreach (int seqLen in new[] { 1, 4 })
                yield return [qt, seqLen];
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void MoeHybrid_VulkanMatchesCpu(QuantizationType qt, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.If(qt == QuantizationType.Q4_0 && !VulkanQwen3MoeMoeUpload.Q4_0NativeEnabled, "DOTLLM_VK_Q4_0_NATIVE=0");
        using var fx = LegacyQuantFixture.Build(seed: 921 + (int)qt * 7 + seqLen, qt);
        int[] ids = Enumerable.Range(0, seqLen).Select(i => i % VocabSize).ToArray();
        int[] pos = Enumerable.Range(0, seqLen).ToArray();

        float[] cpu;
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fx.MoeConfig, fx.MoeLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32,
            fx.OutputWeightPtr, qt, VocabSize, HiddenSize))
        using (var kv = new DotLLM.Engine.KvCache.SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(ids, pos, deviceId: -1, kv))
            cpu = Copy(logits);

        float[] vk;
        long dispatches, mmvq;
        bool hasDp4a;
        using (var device = VulkanDevice.Create())
        {
            hasDp4a = device.HasIntegerDotProduct;
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                device, fx.MoeConfig, fx.MoeLayers, fx.OutputNormWeight, fx.OutputWeightPtr, qt, VocabSize, HiddenSize,
                fx.TokenEmbedPtr, QuantizationType.F32, spvDir);
            Assert.Equal(qt, model.Weights.OutputDeviceQuantType);   // (1) kept packed, not widened
            using var kv = model.CreateKvCache(MaxSeqLen);
            using ITensor logits = model.Forward(ids, pos, deviceId: -1, kv);
            vk = Copy(logits);
            dispatches = model.LegacyQuantDispatchCount;
            mmvq = model.LegacyQuantMmvqDispatchCount;
        }

        Compare(cpu, vk, $"moe-hybrid {qt} seqLen={seqLen}", MoeAbsTol, MoeRelTol);
        // (2) 2 layers x (GDN qkv/gate/alpha/beta/out = 5 | attn q/k/v/o = 4) + lm_head: at least 10 legacy-quant dispatches per forward.
        Assert.True(dispatches >= 10, $"legacy-quant arm ran {dispatches} times; expected >= 10");
        if (seqLen == 1 && qt is QuantizationType.Q4_1 or QuantizationType.Q4_0 && hasDp4a)
            Assert.True(mmvq > 0, $"the {qt} decode went to the F32-in GEMV, not the dp4a MMVQ");
        output.WriteLine($"moe-hybrid {qt} seqLen={seqLen}: legacy dispatches {dispatches}, mmvq {mmvq}");
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void HybridDense_VulkanMatchesCpu(QuantizationType qt, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.If(qt == QuantizationType.Q4_0 && !VulkanQwen3MoeMoeUpload.Q4_0NativeEnabled, "DOTLLM_VK_Q4_0_NATIVE=0");
        using var fx = LegacyQuantFixture.Build(seed: 1921 + (int)qt * 7 + seqLen, qt);
        int[] ids = Enumerable.Range(0, seqLen).Select(i => i % VocabSize).ToArray();
        int[] pos = Enumerable.Range(0, seqLen).ToArray();

        float[] cpu;
        using (var model = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32,
            fx.OutputWeightPtr, qt, VocabSize, HiddenSize))
        using (var kv = new DotLLM.Engine.KvCache.SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(ids, pos, deviceId: -1, kv))
            cpu = Copy(logits);

        float[] vk;
        long dispatches, mmvq;
        bool hasDp4a;
        using (var device = VulkanDevice.Create())
        {
            hasDp4a = device.HasIntegerDotProduct;
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
                device, fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.OutputWeightPtr, qt, VocabSize, HiddenSize,
                fx.TokenEmbedPtr, QuantizationType.F32, spvDir);
            Assert.Equal(qt, model.Weights.OutputDeviceQuantType);
            Assert.Equal(qt, model.Weights.Layers[0].Ffn.DownDeviceQuantType);
            using var kv = model.CreateKvCache(MaxSeqLen);
            using ITensor logits = model.Forward(ids, pos, deviceId: -1, kv);
            vk = Copy(logits);
            dispatches = model.LegacyQuantDispatchCount;
            mmvq = model.LegacyQuantMmvqDispatchCount;
        }

        Compare(cpu, vk, $"hybrid-dense {qt} seqLen={seqLen}", DenseAbsTol, DenseRelTol);
        // 2 layers x (5 or 4 mixer projections + 3 FFN) + lm_head >= 16.
        Assert.True(dispatches >= 16, $"legacy-quant arm ran {dispatches} times; expected >= 16");
        if (seqLen == 1 && qt is QuantizationType.Q4_1 or QuantizationType.Q4_0 && hasDp4a)
            Assert.True(mmvq > 0, $"the {qt} decode went to the F32-in GEMV, not the dp4a MMVQ");
        output.WriteLine($"hybrid-dense {qt} seqLen={seqLen}: legacy dispatches {dispatches}, mmvq {mmvq}");
    }

    private void Compare(float[] cpu, float[] vk, string tag, float absTol, float relTol)
    {
        Assert.Equal(cpu.Length, vk.Length);
        double worst = 0, scale = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(float.IsFinite(cpu[i]) && float.IsFinite(vk[i]), $"{tag}: non-finite logit at {i}: cpu {cpu[i]} vk {vk[i]}");
            worst = Math.Max(worst, Math.Abs(cpu[i] - vk[i]));
            scale = Math.Max(scale, Math.Abs(cpu[i]));
        }
        output.WriteLine($"{tag}: max|cpu-vk|={worst:E3} max|cpu|={scale:E3}");
        for (int i = 0; i < cpu.Length; i++)
        {
            float diff = MathF.Abs(cpu[i] - vk[i]);
            float bar = absTol + relTol * MathF.Abs(cpu[i]);
            Assert.True(diff <= bar, $"{tag} col={i % VocabSize}: cpu={cpu[i]:F6} vk={vk[i]:F6} |diff|={diff:E3} > {bar:E3}");
        }
        double var = 0, mean = vk.Average();
        foreach (float v in vk) var += (v - mean) * (v - mean);
        Assert.True(var / vk.Length > 1e-12, $"{tag}: degenerate (constant) logits");
    }

    private static unsafe float[] Copy(ITensor t)
    {
        int n = checked((int)t.Shape.ElementCount);
        var f = new float[n];
        new ReadOnlySpan<float>((void*)t.DataPointer, n).CopyTo(f);
        return f;
    }

    /// <summary>Random 2-layer hybrid weights (layer 0 GDN, layer 1 full attention) with every projection in one legacy quant.</summary>
    private sealed unsafe class LegacyQuantFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        private QuantizationType _qt;
        public ModelConfig MoeConfig = null!;
        public ModelConfig DenseConfig = null!;
        public Qwen3MoeLayerWeights[] MoeLayers = null!;
        public Qwen3HybridDenseLayerWeights[] DenseLayers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static LegacyQuantFixture Build(int seed, QuantizationType qt)
        {
            var f = new LegacyQuantFixture { _qt = qt };
            f.BuildInternal(new Random(seed));
            return f;
        }

        private void BuildInternal(Random rng)
        {
            HybridLayerKind[] kinds = [HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention];
            var gdn = new GatedDeltaNetConfig(FullAttnInterval: 2, NVHead: NVHead, NKHead: NKHead, DState: DState, DInner: NVHead * DState, DConv: DConv);
            ModelConfig Cfg(Architecture arch, HybridLayerLayout layout, MoeConfig? moe, int inter) => new()
            {
                Architecture = arch, VocabSize = VocabSize, HiddenSize = HiddenSize, IntermediateSize = inter, NumLayers = 2,
                NumAttentionHeads = NumAttentionHeads, NumKvHeads = NumKvHeads, HeadDim = HeadDim, MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA, PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: RoPEType.NeoX),
                ActivationFunction = ActivationFunction.SiLU, NormType = NormType.RMSNorm, NormEpsilon = 1e-5f, TiedEmbeddings = false,
                HybridLayout = layout, GdnConfig = gdn, Moe = moe, ChatTemplate = null,
            };
            MoeConfig = Cfg(Architecture.Qwen3MoeHybrid,
                new HybridLayerLayout { LayerKind = kinds, HeadCountKv = [0, NumKvHeads], FeedForwardLength = [0, 0] },
                new MoeConfig
                {
                    NumExperts = NumExperts, NumExpertsPerTok = NumExpertsPerTok, MoeIntermediateSize = MoeIntermediate, NormTopKProb = true,
                    SharedExpertIntermediateSize = SharedIntermediate, NumSharedExperts = 1, HasSharedExpertGate = true, DecoderSparseStep = 1,
                }, inter: 0);
            DenseConfig = Cfg(Architecture.Qwen3HybridDense,
                new HybridLayerLayout { LayerKind = kinds, HeadCountKv = [0, NumKvHeads], FeedForwardLength = [DenseIntermediate, DenseIntermediate] },
                null, inter: DenseIntermediate);

            TokenEmbedPtr = AllocF32(VocabSize * HiddenSize, rng, 0.05f);
            OutputNormWeight = Norm(HiddenSize, rng);
            OutputWeightPtr = AllocProjection(VocabSize, HiddenSize, rng);

            var gdnW = BuildGdn(rng);
            var attnW = BuildFullAttn(rng);
            float[] n0 = Norm(HiddenSize, rng), n1 = Norm(HiddenSize, rng), n2 = Norm(HiddenSize, rng), n3 = Norm(HiddenSize, rng);
            MoeLayers =
            [
                new Qwen3MoeLayerWeights { AttnNormWeight = n0, PostAttnNormWeight = n1, Gdn = gdnW, FullAttn = null, Moe = BuildMoe(rng) },
                new Qwen3MoeLayerWeights { AttnNormWeight = n2, PostAttnNormWeight = n3, Gdn = null, FullAttn = attnW, Moe = BuildMoe(rng) },
            ];
            DenseLayers =
            [
                DenseLayer(n0, n1, gdnW, null, rng),
                DenseLayer(n2, n3, null, attnW, rng),
            ];
        }

        private Qwen3HybridDenseLayerWeights DenseLayer(float[] n0, float[] n1, GdnTokenMixingWeights? gdn, Qwen3FullAttnWeights? attn, Random rng) => new()
        {
            AttnNormWeight = n0, PostAttnNormWeight = n1, Gdn = gdn, FullAttn = attn,
            GateWeight = AllocProjection(DenseIntermediate, HiddenSize, rng), GateQuantType = _qt, GateInputDim = HiddenSize, GateOutputDim = DenseIntermediate,
            UpWeight = AllocProjection(DenseIntermediate, HiddenSize, rng), UpQuantType = _qt, UpInputDim = HiddenSize, UpOutputDim = DenseIntermediate,
            DownWeight = AllocProjection(HiddenSize, DenseIntermediate, rng), DownQuantType = _qt, DownInputDim = DenseIntermediate, DownOutputDim = HiddenSize,
        };

        private GdnTokenMixingWeights BuildGdn(Random rng)
        {
            int convDim = (2 * NKHead + NVHead) * DState, kDim = NKHead * DState, vDim = NVHead * DState, qkvOut = 2 * kDim + vDim;
            return new GdnTokenMixingWeights
            {
                QkvWeight = AllocProjection(qkvOut, HiddenSize, rng), QkvQuantType = _qt, QkvInputDim = HiddenSize, QkvOutputDim = qkvOut,
                GateWeight = AllocProjection(vDim, HiddenSize, rng), GateQuantType = _qt, GateInputDim = HiddenSize, GateOutputDim = vDim,
                A = Fill(NVHead, rng, 0.5f, -0.6f),
                AlphaWeight = AllocProjection(NVHead, HiddenSize, rng), AlphaQuantType = _qt, AlphaInputDim = HiddenSize, AlphaOutputDim = NVHead,
                BetaWeight = AllocProjection(NVHead, HiddenSize, rng), BetaQuantType = _qt, BetaInputDim = HiddenSize, BetaOutputDim = NVHead,
                Conv1dWeight = Fill(DConv * convDim, rng, 0.1f), Conv1dBias = new float[convDim], DtBias = Fill(NVHead, rng, 0.1f),
                SsmNormWeight = Norm(DState, rng),
                OutWeight = AllocProjection(HiddenSize, vDim, rng), OutQuantType = _qt, OutInputDim = vDim, OutOutputDim = HiddenSize,
            };
        }

        private Qwen3FullAttnWeights BuildFullAttn(Random rng)
        {
            int qOut = 2 * NumAttentionHeads * HeadDim, kvOut = NumKvHeads * HeadDim, oIn = NumAttentionHeads * HeadDim;
            return new Qwen3FullAttnWeights
            {
                QWeight = AllocProjection(qOut, HiddenSize, rng), QQuantType = _qt, QInputDim = HiddenSize, QOutputDim = qOut,
                KWeight = AllocProjection(kvOut, HiddenSize, rng), KQuantType = _qt, KInputDim = HiddenSize, KOutputDim = kvOut,
                VWeight = AllocProjection(kvOut, HiddenSize, rng), VQuantType = _qt, VInputDim = HiddenSize, VOutputDim = kvOut,
                OWeight = AllocProjection(HiddenSize, oIn, rng), OQuantType = _qt, OInputDim = oIn, OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads, QNormWeight = Norm(HeadDim, rng), KNormWeight = Norm(HeadDim, rng),
            };
        }

        private MoeLayerWeights BuildMoe(Random rng)
        {
            var w1 = new nint[NumExperts]; var w2 = new nint[NumExperts]; var w3 = new nint[NumExperts];
            for (int e = 0; e < NumExperts; e++)
            {
                w1[e] = AllocF32(MoeIntermediate * HiddenSize, rng, 0.05f);
                w2[e] = AllocF32(HiddenSize * MoeIntermediate, rng, 0.05f);
                w3[e] = AllocF32(MoeIntermediate * HiddenSize, rng, 0.05f);
            }
            nint[] sharedGate = [AllocF32(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedUp = [AllocF32(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedDown = [AllocF32(HiddenSize * SharedIntermediate, rng, 0.05f)];
            return new MoeLayerWeights(
                gate: Fill(NumExperts * HiddenSize, rng, 0.05f),
                w1: w1, w2: w2, w3: w3,
                numExperts: NumExperts, numExpertsPerTok: NumExpertsPerTok, hiddenSize: HiddenSize, intermediateSize: MoeIntermediate,
                normTopKProb: true,
                sharedGateProj: sharedGate, sharedUpProj: sharedUp, sharedDownProj: sharedDown,
                sharedIntermediateSize: SharedIntermediate, sharedExpertGate: Fill(HiddenSize, rng, 0.05f));
        }

        /// <summary>[outputDim, inputDim] random F32 rows quantised to the fixture's legacy type; inputDim % 32 == 0.</summary>
        private nint AllocProjection(int outputDim, int inputDim, Random rng)
        {
            float[] src = Fill(outputDim * inputDim, rng, 0.05f);
            byte[] q = Encode(src, _qt);
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)q.Length, 64);
            _allocs.Add(dst);
            q.CopyTo(new Span<byte>((void*)dst, q.Length));
            return dst;
        }

        internal static byte[] Encode(float[] src, QuantizationType qt)
        {
            if (qt is QuantizationType.Q5_0 or QuantizationType.Q5_1) return Quantize.FromFloat32(src, src.Length, qt);
            int blocks = src.Length / 32;
            int bb = qt == QuantizationType.Q4_0 ? 18 : 20;
            var dst = new byte[blocks * bb];
            for (int b = 0; b < blocks; b++)
            {
                var x = src.AsSpan(b * 32, 32);
                int o = b * bb;
                if (qt == QuantizationType.Q4_0)
                {
                    float amax = 0, max = 0;
                    foreach (float v in x) if (MathF.Abs(v) > amax) { amax = MathF.Abs(v); max = v; }
                    float d = max / -8f, id = d != 0 ? 1f / d : 0f;
                    BitConverter.TryWriteBytes(dst.AsSpan(o, 2), (Half)d);
                    d = (float)(Half)d; id = d != 0 ? 1f / d : 0f;
                    for (int j = 0; j < 16; j++)
                    {
                        int lo = Math.Clamp((int)MathF.Round(x[j] * id) + 8, 0, 15);
                        int hi = Math.Clamp((int)MathF.Round(x[j + 16] * id) + 8, 0, 15);
                        dst[o + 2 + j] = (byte)(lo | (hi << 4));
                    }
                }
                else
                {
                    float min = float.MaxValue, max = float.MinValue;
                    foreach (float v in x) { min = MathF.Min(min, v); max = MathF.Max(max, v); }
                    float d = (max - min) / 15f;
                    BitConverter.TryWriteBytes(dst.AsSpan(o, 2), (Half)d);
                    BitConverter.TryWriteBytes(dst.AsSpan(o + 2, 2), (Half)min);
                    d = (float)(Half)d; min = (float)(Half)min;
                    float id = d != 0 ? 1f / d : 0f;
                    for (int j = 0; j < 16; j++)
                    {
                        int lo = Math.Clamp((int)MathF.Round((x[j] - min) * id), 0, 15);
                        int hi = Math.Clamp((int)MathF.Round((x[j + 16] - min) * id), 0, 15);
                        dst[o + 4 + j] = (byte)(lo | (hi << 4));
                    }
                }
            }
            return dst;
        }

        private nint AllocF32(int count, Random rng, float amp)
        {
            nint p = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(p);
            float* d = (float*)p;
            for (int i = 0; i < count; i++) d[i] = ((float)rng.NextDouble() * 2f - 1f) * amp;
            return p;
        }

        private static float[] Fill(int n, Random rng, float amp, float offset = 0f)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = offset + ((float)rng.NextDouble() * 2f - 1f) * amp;
            return a;
        }

        private static float[] Norm(int n, Random rng) => Fill(n, rng, 0.05f, 1.0f);

        public void Dispose()
        {
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
