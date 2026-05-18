using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity tests for the Phase 5f-mirror NemotronH <c>ForwardBatch</c> override
/// against the per-seq <c>Forward</c> loop. Builds a tiny synthetic NemotronH hybrid model
/// (mixed SSM/Attention/FFN kinds, F32-only) and asserts that batched logits match per-seq
/// <c>Forward</c> within the Vulkan F32 envelope (5e-3 abs / 1e-3 rel — same bar as the
/// existing <see cref="VulkanNemotronHTransformerModelForwardTests"/>).
/// </summary>
/// <remarks>
/// <para>
/// The NemotronH batched path runs the full per-layer loop per-sequence (Mamba2 SSM state is
/// per-token recurrent) and only fuses the terminal RMSNorm + lm_head into a single
/// <c>matmul(W, [N_simple, hidden])</c> dispatch. The reduction-order drift between the
/// per-seq <c>matmul(W, [1, hidden])</c> and the batched <c>matmul(W, [N_simple, hidden])</c>
/// is bounded by the kernel's per-thread accumulation order — both dispatch the same
/// per-output-cell dot-product so drift stays well within the documented Vulkan envelope.
/// </para>
/// <para>
/// Covers four cases:
/// <list type="number">
///   <item>Empty request list — <c>ForwardBatch</c> returns empty.</item>
///   <item>Single sequence — must equal <c>Forward</c> exactly.</item>
///   <item>Two sequences with different prompts (mixed prefill lengths).</item>
///   <item>Mixed-length decode + prefill across 4 sequences (KvCache pre-seeded).</item>
/// </list>
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHTransformerModelForwardBatchTests
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
    private const int NHead = 2;
    private const int MaxSeqLen = 16;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void VulkanNemotronHForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        using var fixture = NemotronHForwardBatchFixture.Build(kinds, seed: 71);
        using var device = VulkanDevice.Create();
        using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
            device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);

        var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1);
        Assert.NotNull(results);
        Assert.Empty(results);
    }

    [SkippableFact]
    public void VulkanNemotronHForwardBatch_SingleSeq_EqualsForward()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        using var fixture = NemotronHForwardBatchFixture.Build(kinds, seed: 17);

        int[] tokenIds = [1, 3, 5];
        int[] positions = [0, 1, 2];

        // Reference: per-seq Forward.
        float[] reference;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            using var kv = model.CreateKvCache(MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache: kv);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            reference = CopyLogits(logits);
        }

        // Under test: ForwardBatch with one request — must equal Forward exactly (no batching
        // benefit; the override falls back to direct Forward for N == 1).
        float[] batched;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            using var kv = model.CreateKvCache(MaxSeqLen);
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokenIds.AsMemory(),
                    Positions = positions.AsMemory(),
                    KvCache = kv,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Single(results);
                Assert.Equal(1, results[0].Shape[0]);
                Assert.Equal(VocabSize, results[0].Shape[1]);
                batched = CopyLogits(results[0]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        AssertLogitsClose(reference, batched, "Single-seq batch");
    }

    [SkippableFact]
    public void VulkanNemotronHForwardBatch_TwoSeqs_DifferentPrompts_MatchesPerSeqLoop()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        using var fixture = NemotronHForwardBatchFixture.Build(kinds, seed: 31);

        int[] tokensA = [2, 4, 6];
        int[] positionsA = [0, 1, 2];
        int[] tokensB = [1, 3, 5, 7, 2];
        int[] positionsB = [0, 1, 2, 3, 4];

        // Reference: per-seq Forward with independent caches.
        float[] referenceA, referenceB;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            using var kvA = model.CreateKvCache(MaxSeqLen);
            using ITensor logitsA = model.Forward(tokensA, positionsA, deviceId: -1, kvCache: kvA);
            referenceA = CopyLogits(logitsA);

            using var kvB = model.CreateKvCache(MaxSeqLen);
            using ITensor logitsB = model.Forward(tokensB, positionsB, deviceId: -1, kvCache: kvB);
            referenceB = CopyLogits(logitsB);
        }

        // ForwardBatch with both requests on a fresh model.
        float[] batchedA, batchedB;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            using var kvA = model.CreateKvCache(MaxSeqLen);
            using var kvB = model.CreateKvCache(MaxSeqLen);
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(), KvCache = kvA,
                },
                new SequenceForwardRequest
                {
                    TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = kvB,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Equal(2, results.Count);
                Assert.Equal(1, results[0].Shape[0]);
                Assert.Equal(VocabSize, results[0].Shape[1]);
                Assert.Equal(1, results[1].Shape[0]);
                Assert.Equal(VocabSize, results[1].Shape[1]);
                batchedA = CopyLogits(results[0]);
                batchedB = CopyLogits(results[1]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        AssertLogitsClose(referenceA, batchedA, "Two-seq batch [A]");
        AssertLogitsClose(referenceB, batchedB, "Two-seq batch [B]");
    }

    [SkippableFact]
    public void VulkanNemotronHForwardBatch_FourSeqs_MixedPrefillDecode_MatchesPerSeqLoop()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        using var fixture = NemotronHForwardBatchFixture.Build(kinds, seed: 49);

        // Mixed prefill / decode-step seqs. Decode seqs (N=1) start at non-zero positions
        // — the KvCache and SSM state for those must be primed by an initial warm-up
        // prefill (mirroring how the scheduler primes a seq before the first decode step).
        int[][] tokens =
        [
            [1, 3, 5],         // S0: prefill, 3 tokens
            [7],               // S1: decode, 1 token at position 4
            [2, 4, 6, 1, 3],   // S2: prefill, 5 tokens
            [5, 6],            // S3: prefill, 2 tokens
        ];
        int[][] positions =
        [
            [0, 1, 2],
            [4],
            [0, 1, 2, 3, 4],
            [0, 1],
        ];

        // Per-seq reference. The decode seq (S1) needs a warm-up Forward to prime its
        // KvCache and SSM-state-cache; mirror that ordering on both sides.
        float[][] referenceLogits = new float[tokens.Length][];
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            for (int s = 0; s < tokens.Length; s++)
            {
                using var kv = model.CreateKvCache(MaxSeqLen);
                int firstPos = positions[s][0];
                if (firstPos > 0)
                {
                    int[] warmupTokens = new int[firstPos];
                    int[] warmupPositions = new int[firstPos];
                    for (int t = 0; t < firstPos; t++)
                    {
                        warmupTokens[t] = (t * 5 + s) % VocabSize;
                        warmupPositions[t] = t;
                    }
                    using (model.Forward(warmupTokens, warmupPositions, deviceId: -1, kvCache: kv)) { }
                }
                using ITensor logits = model.Forward(tokens[s], positions[s], deviceId: -1, kvCache: kv);
                referenceLogits[s] = CopyLogits(logits);
            }
        }

        // Same prep but using ForwardBatch on a fresh model.
        float[][] batchedLogits = new float[tokens.Length][];
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32, spvDir);
            var caches = new VulkanNemotronHKvCache[tokens.Length];
            try
            {
                for (int s = 0; s < tokens.Length; s++)
                {
                    caches[s] = model.CreateKvCache(MaxSeqLen);
                    int firstPos = positions[s][0];
                    if (firstPos > 0)
                    {
                        int[] warmupTokens = new int[firstPos];
                        int[] warmupPositions = new int[firstPos];
                        for (int t = 0; t < firstPos; t++)
                        {
                            warmupTokens[t] = (t * 5 + s) % VocabSize;
                            warmupPositions[t] = t;
                        }
                        using (model.Forward(warmupTokens, warmupPositions, deviceId: -1, kvCache: caches[s])) { }
                    }
                }

                var requests = new SequenceForwardRequest[tokens.Length];
                for (int s = 0; s < tokens.Length; s++)
                {
                    requests[s] = new SequenceForwardRequest
                    {
                        TokenIds = tokens[s].AsMemory(),
                        Positions = positions[s].AsMemory(),
                        KvCache = caches[s],
                    };
                }

                var results = model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    Assert.Equal(tokens.Length, results.Count);
                    for (int s = 0; s < tokens.Length; s++)
                    {
                        Assert.Equal(1, results[s].Shape[0]);
                        Assert.Equal(VocabSize, results[s].Shape[1]);
                        batchedLogits[s] = CopyLogits(results[s]);
                    }
                }
                finally { foreach (var t in results) t.Dispose(); }
            }
            finally
            {
                for (int s = 0; s < tokens.Length; s++) caches[s]?.Dispose();
            }
        }

        for (int s = 0; s < tokens.Length; s++)
        {
            AssertLogitsClose(referenceLogits[s], batchedLogits[s], $"Four-seq mixed [seq {s}, N={tokens[s].Length}]");
        }
    }

    private static void AssertLogitsClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        Assert.Equal(VocabSize, reference.Length);
        for (int c = 0; c < VocabSize; c++)
        {
            float r = reference[c];
            float a = actual[c];
            float diff = MathF.Abs(r - a);
            float bar = AbsTol + RelTol * MathF.Abs(r);
            Assert.True(diff <= bar,
                $"{label}: col={c}: reference={r:F6} vs actual={a:F6} (|diff|={diff:E3} > {bar:E3})");
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
    /// Owns a randomly-generated NemotronH "model" in unmanaged memory for the batched
    /// parity tests. F32-only (the batched path's correctness contract is independent of
    /// the projection quant type — the existing
    /// <see cref="VulkanNemotronHTransformerModelForwardTests"/> already pins Q8_0 parity).
    /// </summary>
    private sealed unsafe class NemotronHForwardBatchFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static NemotronHForwardBatchFixture Build(HybridLayerKind[] layerKinds, int seed)
        {
            var f = new NemotronHForwardBatchFixture();
            f.BuildInternal(layerKinds, seed);
            return f;
        }

        private void BuildInternal(HybridLayerKind[] layerKinds, int seed)
        {
            int numLayers = layerKinds.Length;
            var rng = new Random(seed);

            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = layerKinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = layerKinds[i] == HybridLayerKind.Ffn ? IntermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = layerKinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLength,
            };

            var ssmConfig = new MambaSsmConfig(
                DConv: DConv, DInner: DInner, DState: DState, NGroup: NGroup, NHead: NHead);

            Config = new ModelConfig
            {
                Architecture = Architecture.NemotronH,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = IntermediateSize,
                NumLayers = numLayers,
                NumAttentionHeads = NumHeads,
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

            TokenEmbedPtr = AllocAndFill(VocabSize * HiddenSize, rng, amplitude: 0.05f);
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocAndFill(VocabSize * HiddenSize, rng, amplitude: 0.05f);

            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float[] attnNorm = FillNormVec(HiddenSize, rng);
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
                            Ffn = BuildFfn(rng, IntermediateSize),
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
            nint inWeight = AllocAndFill(inProjDim * HiddenSize, rng, amplitude: 0.05f);
            nint outWeight = AllocAndFill(HiddenSize * ssm.DInner, rng, amplitude: 0.05f);
            float[] conv1dWeight = FillRandom(ssm.DConv * convDim, rng, amplitude: 0.1f);
            float[] conv1dBias = FillRandom(convDim, rng, amplitude: 0.1f);
            // A: forced negative so exp(dt*A) decays.
            float[] a = new float[ssm.NHead];
            for (int i = 0; i < a.Length; i++) a[i] = -((float)rng.NextDouble() * 0.1f + 0.05f);
            float[] d = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] dtBias = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] normWeight = FillNormVec(ssm.DInner, rng);
            return new NemotronHSsmWeights
            {
                InWeight = inWeight,
                InQuantType = QuantizationType.F32,
                InInputDim = HiddenSize,
                InOutputDim = inProjDim,
                Conv1dWeight = conv1dWeight,
                Conv1dBias = conv1dBias,
                A = a,
                D = d,
                DtBias = dtBias,
                NormWeight = normWeight,
                OutWeight = outWeight,
                OutQuantType = QuantizationType.F32,
                OutInputDim = ssm.DInner,
                OutOutputDim = HiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = NumHeads * HeadDim;
            int kvOut = numKvHeads * HeadDim;
            return new NemotronHAttentionWeights
            {
                QWeight = AllocAndFill(qOut * HiddenSize, rng, amplitude: 0.05f),
                QQuantType = QuantizationType.F32, QInputDim = HiddenSize, QOutputDim = qOut,
                KWeight = AllocAndFill(kvOut * HiddenSize, rng, amplitude: 0.05f),
                KQuantType = QuantizationType.F32, KInputDim = HiddenSize, KOutputDim = kvOut,
                VWeight = AllocAndFill(kvOut * HiddenSize, rng, amplitude: 0.05f),
                VQuantType = QuantizationType.F32, VInputDim = HiddenSize, VOutputDim = kvOut,
                OWeight = AllocAndFill(HiddenSize * qOut, rng, amplitude: 0.05f),
                OQuantType = QuantizationType.F32, OInputDim = qOut, OOutputDim = HiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediate)
        {
            return new NemotronHFfnWeights
            {
                UpWeight = AllocAndFill(intermediate * HiddenSize, rng, amplitude: 0.05f),
                UpQuantType = QuantizationType.F32, UpInputDim = HiddenSize, UpOutputDim = intermediate,
                DownWeight = AllocAndFill(HiddenSize * intermediate, rng, amplitude: 0.05f),
                DownQuantType = QuantizationType.F32, DownInputDim = intermediate, DownOutputDim = HiddenSize,
                IntermediateSize = intermediate,
            };
        }

        private static float[] FillRandom(int n, Random rng, float amplitude)
        {
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * amplitude);
            return arr;
        }

        private static float[] FillNormVec(int n, Random rng)
        {
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = 1.0f + (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            return arr;
        }

        private nint AllocAndFill(int n, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(n * sizeof(float)), 64);
            _allocs.Add(ptr);
            unsafe
            {
                var dst = new Span<float>((void*)ptr, n);
                for (int i = 0; i < n; i++) dst[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * amplitude);
            }
            return ptr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
            {
                unsafe { NativeMemory.AlignedFree((void*)p); }
            }
            _allocs.Clear();
        }
    }
}
