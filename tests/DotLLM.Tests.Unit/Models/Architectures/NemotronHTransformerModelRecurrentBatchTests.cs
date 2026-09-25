using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using Architecture = DotLLM.Core.Configuration.Architecture;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// CPU Nemotron-H per-sequence SSM-state threading (recurrent batched decode/prefill). Mirrors the
/// Mamba-3 / Qwen3-MoE pattern: the scheduler allocates one <see cref="ISsmState"/> per sequence and
/// threads it through <c>ForwardBatch</c>. The discriminating test runs concurrent <em>decode</em>
/// across two sequences — where a shared model-owned <c>_ssmCache</c> would corrupt the recurrent
/// state — and asserts per-seq parity. Uses a tiny synthetic SSM/FFN hybrid (no attention ⇒ no KV).
/// </summary>
public sealed unsafe class NemotronHTransformerModelRecurrentBatchTests
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

    [Fact]
    public void Flags_And_Factory_ExposeThreadedSsmState()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Ssm };
        using var fix = NemotronHFixture.Build(kinds, seed: 7);
        using var model = Build(fix);

        Assert.True(model.RequiresPerSequenceState);
        Assert.True(model.SupportsThreadedSequenceState);

        using var state = model.CreateSequenceState();
        var ssm = Assert.IsAssignableFrom<ISsmState>(state);
        Assert.Equal(2, ssm.NumSsmLayers); // two SSM layers in the layout
        Assert.IsType<SsmStateCache>(state);
    }

    [Fact]
    public void ForwardBatch_TwoSeqsWithNullSsmState_Throws()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        using var fix = NemotronHFixture.Build(kinds, seed: 11);
        using var model = Build(fix);

        int[] tokensA = [2, 4, 6];
        int[] posA = [0, 1, 2];
        var requests = new[]
        {
            new SequenceForwardRequest { TokenIds = tokensA.AsMemory(), Positions = posA.AsMemory(), KvCache = null! },
            new SequenceForwardRequest { TokenIds = tokensA.AsMemory(), Positions = posA.AsMemory(), KvCache = null! },
        };
        Assert.Throws<ArgumentException>(() => model.ForwardBatch(requests, deviceId: -1));
    }

    [Fact]
    public void ForwardBatch_ConcurrentDecode_ThreadsPerSeqState_MatchesReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Ssm };
        using var fix = NemotronHFixture.Build(kinds, seed: 23);

        int[] tokensA = [2, 4, 6];
        int[] posA = [0, 1, 2];
        int[] tokensB = [1, 3, 5, 7];
        int[] posB = [0, 1, 2, 3];
        int[] decTokA = [3]; int[] decPosA = [3];
        int[] decTokB = [2]; int[] decPosB = [4];

        // Reference: each sequence prefilled + decoded alone on its own fresh model + state.
        float[] refA, refB;
        using (var model = Build(fix))
        {
            using var s = (SsmStateCache)model.CreateSequenceState()!;
            model.Forward(tokensA, posA, -1, null, s).Dispose();
            using var lg = model.Forward(decTokA, decPosA, -1, null, s);
            refA = CopyLastRow(lg);
        }
        using (var model = Build(fix))
        {
            using var s = (SsmStateCache)model.CreateSequenceState()!;
            model.Forward(tokensB, posB, -1, null, s).Dispose();
            using var lg = model.Forward(decTokB, decPosB, -1, null, s);
            refB = CopyLastRow(lg);
        }

        // Batched: one model, two per-seq states, interleaved prefill then a fused decode ForwardBatch.
        float[] batchA, batchB;
        using (var model = Build(fix))
        {
            using var sA = (SsmStateCache)model.CreateSequenceState()!;
            using var sB = (SsmStateCache)model.CreateSequenceState()!;
            model.Forward(tokensA, posA, -1, null, sA).Dispose();
            model.Forward(tokensB, posB, -1, null, sB).Dispose();

            var requests = new[]
            {
                new SequenceForwardRequest { TokenIds = decTokA.AsMemory(), Positions = decPosA.AsMemory(), KvCache = null!, SsmState = sA },
                new SequenceForwardRequest { TokenIds = decTokB.AsMemory(), Positions = decPosB.AsMemory(), KvCache = null!, SsmState = sB },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                batchA = CopyLastRow(results[0]);
                batchB = CopyLastRow(results[1]);
            }
            finally { foreach (var t in results) t.Dispose(); }
        }

        // Per-seq threading ⇒ the batched concurrent decode equals the isolated reference. A shared
        // _ssmCache would let seq A's decode pollute seq B's SSM state and diverge here.
        AssertClose(refA, batchA, "seqA");
        AssertClose(refB, batchB, "seqB");
    }

    // ── Helpers ──

    private static NemotronHTransformerModel Build(NemotronHFixture f)
        => NemotronHTransformerModel.BuildFromPrebuiltWeights(
            f.Config, f.Layers, f.OutputNormWeight,
            f.TokenEmbedPtr, QuantizationType.F32,
            f.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize);

    private static float[] CopyLastRow(ITensor logits)
    {
        int rows = logits.Shape[0];
        var outp = new float[VocabSize];
        new Span<float>((float*)logits.DataPointer + (long)(rows - 1) * VocabSize, VocabSize).CopyTo(outp);
        return outp;
    }

    private static void AssertClose(float[] expected, float[] actual, string tag)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f,
                $"{tag}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    /// <summary>Tiny synthetic Nemotron-H weights (SSM/FFN/Attention kinds, F32). Owns the unmanaged
    /// weight buffers; disposed at end of test.</summary>
    private sealed class NemotronHFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static NemotronHFixture Build(HybridLayerKind[] kinds, int seed)
        {
            var f = new NemotronHFixture();
            f.BuildInternal(kinds, seed);
            return f;
        }

        private void BuildInternal(HybridLayerKind[] kinds, int seed)
        {
            int numLayers = kinds.Length;
            var rng = new Random(seed);

            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = kinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = kinds[i] == HybridLayerKind.Ffn ? IntermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = kinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLength,
            };
            var ssm = new MambaSsmConfig(DConv: DConv, DInner: DInner, DState: DState, NGroup: NGroup, NHead: NHead);

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
                SsmConfig = ssm,
                ChatTemplate = null,
            };

            TokenEmbedPtr = Alloc(VocabSize * HiddenSize, rng, 0.05f);
            OutputNormWeight = Norm(HiddenSize, rng);
            OutputWeightPtr = Alloc(VocabSize * HiddenSize, rng, 0.05f);

            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                Layers[i] = kinds[i] switch
                {
                    HybridLayerKind.Ssm => new NemotronHLayerWeights { AttnNormWeight = Norm(HiddenSize, rng), Ssm = BuildSsm(rng, ssm) },
                    HybridLayerKind.Attention => new NemotronHLayerWeights { AttnNormWeight = Norm(HiddenSize, rng), Attention = BuildAttn(rng) },
                    HybridLayerKind.Ffn => new NemotronHLayerWeights { AttnNormWeight = Norm(HiddenSize, rng), Ffn = BuildFfn(rng) },
                    _ => throw new InvalidOperationException(),
                };
            }
        }

        private NemotronHSsmWeights BuildSsm(Random rng, MambaSsmConfig ssm)
        {
            int convDim = ssm.ConvDim;
            int inProjDim = ssm.InputProjectionDim;
            var a = new float[ssm.NHead];
            for (int i = 0; i < a.Length; i++) a[i] = -((float)rng.NextDouble() * 0.1f + 0.05f); // negative ⇒ decay
            return new NemotronHSsmWeights
            {
                InWeight = Alloc(inProjDim * HiddenSize, rng, 0.05f),
                InQuantType = QuantizationType.F32, InInputDim = HiddenSize, InOutputDim = inProjDim,
                Conv1dWeight = Rand(ssm.DConv * convDim, rng, 0.1f),
                Conv1dBias = Rand(convDim, rng, 0.1f),
                A = a,
                D = Rand(ssm.NHead, rng, 0.1f),
                DtBias = Rand(ssm.NHead, rng, 0.1f),
                NormWeight = Norm(ssm.DInner, rng),
                OutWeight = Alloc(HiddenSize * ssm.DInner, rng, 0.05f),
                OutQuantType = QuantizationType.F32, OutInputDim = ssm.DInner, OutOutputDim = HiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng)
        {
            int qOut = NumHeads * HeadDim, kvOut = NumKvHeads * HeadDim;
            return new NemotronHAttentionWeights
            {
                QWeight = Alloc(qOut * HiddenSize, rng, 0.05f), QQuantType = QuantizationType.F32, QInputDim = HiddenSize, QOutputDim = qOut,
                KWeight = Alloc(kvOut * HiddenSize, rng, 0.05f), KQuantType = QuantizationType.F32, KInputDim = HiddenSize, KOutputDim = kvOut,
                VWeight = Alloc(kvOut * HiddenSize, rng, 0.05f), VQuantType = QuantizationType.F32, VInputDim = HiddenSize, VOutputDim = kvOut,
                OWeight = Alloc(HiddenSize * qOut, rng, 0.05f), OQuantType = QuantizationType.F32, OInputDim = qOut, OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng) => new()
        {
            UpWeight = Alloc(IntermediateSize * HiddenSize, rng, 0.05f), UpQuantType = QuantizationType.F32, UpInputDim = HiddenSize, UpOutputDim = IntermediateSize,
            DownWeight = Alloc(HiddenSize * IntermediateSize, rng, 0.05f), DownQuantType = QuantizationType.F32, DownInputDim = IntermediateSize, DownOutputDim = HiddenSize,
            IntermediateSize = IntermediateSize,
        };

        private static float[] Rand(int n, Random rng, float amp)
        {
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * amp);
            return arr;
        }

        private static float[] Norm(int n, Random rng)
        {
            var arr = new float[n];
            for (int i = 0; i < n; i++) arr[i] = 1.0f + (float)((rng.NextDouble() * 2.0 - 1.0) * 0.05);
            return arr;
        }

        private nint Alloc(int n, Random rng, float amp)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(n * sizeof(float)), 64);
            _allocs.Add(ptr);
            var dst = new Span<float>((void*)ptr, n);
            for (int i = 0; i < n; i++) dst[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * amp);
            return ptr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
