using System.Linq;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end parity test for the CUDA NemotronH hybrid forward pass against the CPU reference.
/// Part 1 (this class's fixture + 6 cacheless test cases) is a near-verbatim port of
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanNemotronHTransformerModelForwardTests"/> — same
/// synthetic-model construction, same tolerance. Part 2 (bottom of the file) adds cached
/// prefill + decode coverage that exercises <see cref="CudaNemotronHKvCache"/> and
/// <see cref="CudaNemotronHSsmStateCache"/>, which the cacheless cases never touch.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaNemotronHTransformerModelForwardTests
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

    // Q8_0 quant-fixture dimensions — every contraction axis is a multiple of 32.
    private const int Q8HiddenSize = 32;
    private const int Q8IntermediateSize = 32;
    private const int Q8DInner = 32;
    private const int Q8NHead = 4;
    private const int Q8NumAttentionHeads = 4;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableFact]
    public void Forward_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 3, seed: 7);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 13);
    }

    [SkippableFact]
    public void Forward_Q8_0_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 142, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertCudaMatchesCpu(kinds, seqLen: 3, seed: 107, quantize: true);
    }

    [SkippableFact]
    public void Forward_Q8_0_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertCudaMatchesCpu(kinds, seqLen: 1, seed: 113, quantize: true);
    }

    private void AssertCudaMatchesCpu(
        HybridLayerKind[] layerKinds, int seqLen, int seed, bool quantize = false)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed, quantize);
        var config = fixture.Config;

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        float[] cpuLogits;
        using (var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] cudaLogits;
        using (var model = CudaNemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            deviceId: 0, ptxDir))
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(seqLen, logits.Shape[0]);
            Assert.Equal(vocabSize, logits.Shape[1]);
            cudaLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < vocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * vocabSize + c];
            float cuda = cudaLogits[lastRow * vocabSize + c];
            float diff = MathF.Abs(cpu - cuda);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"layers={string.Join(',', layerKinds)}, seqLen={seqLen}, quant={quantize}, col={c}: " +
                $"cpu={cpu:F6} vs cuda={cuda:F6} (|diff|={diff:E3} > {bar:E3})");
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
    /// Runs a real cached prefill (3 tokens) followed by 3 cached decode steps on BOTH backends,
    /// using a real <see cref="IKvCache"/> (attention) and the model's own recurrent SSM state
    /// (not the cacheless <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/>
    /// overload every other test in this file uses). This is the only test in this plan that
    /// exercises <see cref="CudaNemotronHKvCache"/> and <see cref="CudaNemotronHSsmStateCache"/>
    /// end-to-end, together, the way a real generation loop drives the model.
    /// </summary>
    [SkippableFact]
    public void Forward_CachedPrefillThenDecode_MatchesCpuReference()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        var kinds = new[]
        {
            HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Attention,
        };
        using var fixture = NemotronHFixtureBuilder.Build(kinds, seed: 271);
        var config = fixture.Config;
        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;
        int attentionLayerCount = kinds.Count(k => k == HybridLayerKind.Attention);

        int[] promptIds = { 1, 2, 3 };
        int[] promptPositions = { 0, 1, 2 };
        int[] decodeTokens = { 4, 5, 6 };

        var cpuSteps = new List<float[]>();
        using (var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        using (var kvCache = new SimpleKvCache(attentionLayerCount, config.NumKvHeads, config.HeadDim, MaxSeqLen))
        {
            model.ResetSequenceState();
            using (ITensor logits = model.Forward(promptIds, promptPositions, deviceId: -1, kvCache))
                cpuSteps.Add(CopyLogits(logits));
            int pos = promptIds.Length;
            foreach (int tok in decodeTokens)
            {
                using ITensor logits = model.Forward(new[] { tok }, new[] { pos }, deviceId: -1, kvCache);
                cpuSteps.Add(CopyLogits(logits));
                pos++;
            }
        }

        var cudaSteps = new List<float[]>();
        using (var model = CudaNemotronHTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            deviceId: 0, ptxDir))
        using (var kvCache = model.CreateKvCache(MaxSeqLen))
        {
            model.ResetSequenceState();
            using (ITensor logits = model.Forward(promptIds, promptPositions, deviceId: -1, kvCache))
                cudaSteps.Add(CopyLogits(logits));
            int pos = promptIds.Length;
            foreach (int tok in decodeTokens)
            {
                using ITensor logits = model.Forward(new[] { tok }, new[] { pos }, deviceId: -1, kvCache);
                cudaSteps.Add(CopyLogits(logits));
                pos++;
            }
        }

        Assert.Equal(cpuSteps.Count, cudaSteps.Count);
        for (int step = 0; step < cpuSteps.Count; step++)
        {
            int rows = step == 0 ? promptIds.Length : 1;
            int lastRow = rows - 1;
            for (int c = 0; c < vocabSize; c++)
            {
                float cpu = cpuSteps[step][lastRow * vocabSize + c];
                float cuda = cudaSteps[step][lastRow * vocabSize + c];
                float diff = MathF.Abs(cpu - cuda);
                float bar = AbsTol + RelTol * MathF.Abs(cpu);
                Assert.True(diff <= bar,
                    $"step={step} col={c}: cpu={cpu:F6} vs cuda={cuda:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
    }

    /// <summary>Owns a randomly-generated NemotronH "model" in unmanaged memory. Verbatim port of
    /// <c>VulkanNemotronHTransformerModelForwardTests.NemotronHFixtureBuilder</c> — see that class
    /// for the full design rationale (identical bytes fed to both backends, Q8_0-friendly dims in
    /// quant mode, F32-only token embedding).</summary>
    private sealed unsafe class NemotronHFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;
        public QuantizationType OutputQuantType;

        private int _hiddenSize, _intermediateSize, _dInner, _nHead, _numAttentionHeads;
        private bool _quantize;

        public static NemotronHFixtureBuilder Build(HybridLayerKind[] layerKinds, int seed, bool quantize = false)
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

            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = layerKinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = layerKinds[i] == HybridLayerKind.Ffn ? _intermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = layerKinds, HeadCountKv = headCountKv, FeedForwardLength = ffnLength,
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

            TokenEmbedPtr = AllocAndFillUniform(VocabSize * _hiddenSize, rng, amplitude: 0.05f);
            OutputNormWeight = FillNormVec(_hiddenSize, rng);
            OutputWeightPtr = AllocProjection(VocabSize, _hiddenSize, rng, OutputQuantType);

            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float[] attnNorm = FillNormVec(_hiddenSize, rng);
                Layers[i] = layerKinds[i] switch
                {
                    HybridLayerKind.Ssm => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Ssm = BuildSsm(rng, ssmConfig) },
                    HybridLayerKind.Attention => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Attention = BuildAttn(rng, NumKvHeads) },
                    HybridLayerKind.Ffn => new NemotronHLayerWeights { AttnNormWeight = attnNorm, Ffn = BuildFfn(rng, _intermediateSize) },
                    _ => throw new InvalidOperationException(),
                };
            }
        }

        private NemotronHSsmWeights BuildSsm(Random rng, MambaSsmConfig ssm)
        {
            int convDim = ssm.ConvDim;
            int inProjDim = ssm.InputProjectionDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHSsmWeights
            {
                InWeight = AllocProjection(inProjDim, _hiddenSize, rng, qt), InQuantType = qt,
                InInputDim = _hiddenSize, InOutputDim = inProjDim,
                Conv1dWeight = FillRandom(ssm.DConv * convDim, rng, 0.1f),
                Conv1dBias = FillRandom(convDim, rng, 0.1f),
                A = NegativeRandom(ssm.NHead, rng),
                D = FillRandom(ssm.NHead, rng, 0.1f),
                DtBias = FillRandom(ssm.NHead, rng, 0.1f),
                NormWeight = FillNormVec(ssm.DInner, rng),
                OutWeight = AllocProjection(_hiddenSize, ssm.DInner, rng, qt), OutQuantType = qt,
                OutInputDim = ssm.DInner, OutOutputDim = _hiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = _numAttentionHeads * HeadDim;
            int kvOut = numKvHeads * HeadDim;
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHAttentionWeights
            {
                QWeight = AllocProjection(qOut, _hiddenSize, rng, qt), QQuantType = qt, QInputDim = _hiddenSize, QOutputDim = qOut,
                KWeight = AllocProjection(kvOut, _hiddenSize, rng, qt), KQuantType = qt, KInputDim = _hiddenSize, KOutputDim = kvOut,
                VWeight = AllocProjection(kvOut, _hiddenSize, rng, qt), VQuantType = qt, VInputDim = _hiddenSize, VOutputDim = kvOut,
                OWeight = AllocProjection(_hiddenSize, qOut, rng, qt), OQuantType = qt, OInputDim = qOut, OOutputDim = _hiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediateSize)
        {
            QuantizationType qt = _quantize ? QuantizationType.Q8_0 : QuantizationType.F32;
            return new NemotronHFfnWeights
            {
                UpWeight = AllocProjection(intermediateSize, _hiddenSize, rng, qt), UpQuantType = qt, UpInputDim = _hiddenSize, UpOutputDim = intermediateSize,
                DownWeight = AllocProjection(_hiddenSize, intermediateSize, rng, qt), DownQuantType = qt, DownInputDim = intermediateSize, DownOutputDim = _hiddenSize,
                IntermediateSize = intermediateSize,
            };
        }

        private nint AllocProjection(int outputDim, int inputDim, Random rng, QuantizationType qt)
        {
            if (qt == QuantizationType.F32) return AllocAndFillUniform(outputDim * inputDim, rng, amplitude: 0.05f);
            if (qt != QuantizationType.Q8_0) throw new NotSupportedException($"AllocProjection only supports F32/Q8_0, got {qt}.");
            if ((inputDim % 32) != 0) throw new InvalidOperationException($"Q8_0 requires inputDim multiple of 32 (got {inputDim}).");

            int rowBytes = (inputDim / 32) * 34;
            long totalBytes = (long)rowBytes * outputDim;
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
            _allocs.Add(dst);

            float[] rowScratch = new float[inputDim];
            for (int row = 0; row < outputDim; row++)
            {
                for (int j = 0; j < inputDim; j++) rowScratch[j] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
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
            for (int i = 0; i < count; i++) dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return ptr;
        }

        private static float[] FillRandom(int count, Random rng, float amplitude)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return arr;
        }

        private static float[] FillNormVec(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
