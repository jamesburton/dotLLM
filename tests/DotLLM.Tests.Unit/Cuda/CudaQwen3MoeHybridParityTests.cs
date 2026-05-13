using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-vs-CUDA last-token-logits parity tests for
/// <see cref="CudaQwen3MoeHybridTransformerModel"/>. These tests are the first end-to-end
/// coverage of the new CUDA forward path (CudaMoeFfn integration, F32 KV cache sidecar,
/// fused-op kernels) and they pin the CUDA model's last-token logits to the
/// <see cref="Qwen3MoeHybridTransformerModel"/> oracle on a synthetic fixture small enough
/// to fit in &lt;100 MB on any CUDA card (hidden=32, 4 experts, 2 layers).
/// </summary>
/// <remarks>
/// <para>
/// <b>Tolerance.</b> Both paths run F32 throughout; numerical drift comes only from the
/// order of GPU reductions (cuBLAS vs scalar SIMD CPU). CUDA's precise <c>expf</c> is
/// ≤1 ULP correctly rounded, so an absolute tolerance of <c>1e-4</c> with a
/// <c>1e-3</c> relative bound is comfortable. Divergence above this band indicates a
/// real kernel bug, not precision drift.
/// </para>
/// <para>
/// <b>Skip behaviour.</b> Each test skips cleanly when no CUDA driver is present or the
/// PTX directory cannot be located. The skip predicate mirrors the pattern used in
/// <c>CudaQwen3MoeHybridElementwiseKernelTests</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class CudaQwen3MoeHybridParityTests
{
    private readonly ITestOutputHelper _out;
    public CudaQwen3MoeHybridParityTests(ITestOutputHelper output) => _out = output;

    // ── Fixture shape (kept identical to CPU tests for cross-checking) ──────────
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 8;
    private const int MaxSeqLen = 8;
    private const int MoeIntermediate = 32;
    private const int SharedIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;
    private const int NVHead = 2;
    private const int NKHead = 1;
    private const int DState = 8;
    private const int DConv = 4;
    private const int DInner = NVHead * DState;

    // Same tolerance band on all four tests.
    private const float AbsTol = 1e-4f;
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

    /// <summary>
    /// Locate the PTX directory next to the test assembly (csproj copies
    /// <c>native/ptx/*.ptx</c> into the test output) or fall back to the repo's
    /// canonical <c>native/ptx/</c>.
    /// </summary>
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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    // ─── Test 1: GDN-only prefill ───────────────────────────────────────────────

    /// <summary>
    /// Prefill parity on a GDN-only model. Exercises the CudaMoeFfn dispatcher (with shared
    /// expert), the four fused-op kernels via the GDN body, and the per-token GDN scan —
    /// no full-attention path, no KV cache. A miscalibrated MoE dispatch or a wrong
    /// shared-expert wiring will diverge by more than the 1e-3 relative tolerance.
    /// </summary>
    [SkippableFact]
    public void CudaForward_GdnOnly_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = CudaQwen3MoeHybridFixture.Build(seed: 11, gdnOnly: true);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions, useKvCache: false);
        float[] cudaLast = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!, useKvCache: false);

        AssertLogitsMatch(cpuLast, cudaLast);
    }

    // ─── Test 2: Mixed (GDN + full-attn) prefill ────────────────────────────────

    /// <summary>
    /// Prefill parity on the default mixed-layer model (layer 0 GDN, layer 1 full-attn).
    /// Adds the Q+Gate fused projection, QK-norm, partial-rotary RoPE, GQA SDPA, and
    /// post-attention sigmoid-gate elementwise mul on top of the GDN-only test. The KV
    /// cache is non-null but used as the prefill anchor — exercises the F32 KV cache
    /// write path without the decode read path.
    /// </summary>
    [SkippableFact]
    public void CudaForward_Mixed_GdnAndFullAttn_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [5, 0, 3, 7];
        int[] positions = [0, 1, 2, 3];

        using var fixture = CudaQwen3MoeHybridFixture.Build(seed: 23);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions, useKvCache: true);
        float[] cudaLast = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!, useKvCache: true);

        AssertLogitsMatch(cpuLast, cudaLast);
    }

    // ─── Diagnostic 2b: Mixed prefill WITHOUT KV cache ──────────────────────────

    /// <summary>
    /// Diagnostic variant of Test 2: same mixed-layer model and tokens but with
    /// <c>kvCache: null</c> on both backends. The CUDA full-attn body takes the
    /// no-cache fast path (line ~982 of CudaQwen3MoeHybridTransformerModel.cs, walks
    /// the freshly-projected K/V directly), bypassing the F32 KV cache write/read
    /// path entirely. Use this to disambiguate whether the failure in Test 2 lives
    /// in the F32 KV cache sidecar or in the full-attn body itself.
    /// </summary>
    [SkippableFact]
    public void CudaForward_Mixed_GdnAndFullAttn_PrefillNoKvCache_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [5, 0, 3, 7];
        int[] positions = [0, 1, 2, 3];

        using var fixture = CudaQwen3MoeHybridFixture.Build(seed: 23);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions, useKvCache: false);
        float[] cudaLast = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!, useKvCache: false);

        AssertLogitsMatch(cpuLast, cudaLast);
    }

    // ─── Test 3: Shared-expert gate perturbation ───────────────────────────────

    /// <summary>
    /// Confirms the CUDA shared-expert sidecar is actually wired into the forward path:
    /// perturbing the shared-expert sigmoid-gate amplitude must change the logits. If the
    /// shared-expert MLP is skipped (a wiring bug in <c>ForwardSharedExpertF32</c>'s call
    /// site or in <c>EnsureSharedExpertScratch</c>), the baseline and perturbed runs
    /// would be bit-identical — failing this test.
    /// </summary>
    [SkippableFact]
    public void CudaForward_SharedExpertGate_PerturbationChangesLogits()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];

        float[] baseLogits;
        float[] perturbedLogits;
        using (var fixture = CudaQwen3MoeHybridFixture.Build(seed: 99))
        {
            baseLogits = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!, useKvCache: true);
        }
        using (var fixture = CudaQwen3MoeHybridFixture.Build(seed: 99, sharedGateAmplitude: 5f))
        {
            perturbedLogits = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!, useKvCache: true);
        }

        bool different = false;
        for (int i = 0; i < baseLogits.Length; i++)
        {
            if (MathF.Abs(baseLogits[i] - perturbedLogits[i]) > 1e-5f)
            {
                different = true;
                break;
            }
        }
        Assert.True(different,
            "perturbing the shared-expert gate amplitude did not change the CUDA logits — " +
            "the shared-expert path is not being executed.");
    }

    // ─── Test 4: KV cache decode parity ─────────────────────────────────────────

    /// <summary>
    /// Exercises the new model-private F32 KV cache: run a 3-token prefill, then a
    /// 1-token decode step, on BOTH CPU and CUDA — the per-step decode logits must match
    /// across backends. Diverging here points at the F32 KV cache write/read path the
    /// perf agent shipped.
    /// </summary>
    /// <remarks>
    /// We compare the decode-step row across backends, not "prefill-N+1 vs prefill-N
    /// then decode-1 on the same backend". The latter is the CPU's
    /// <c>Forward_Mixed_PrefillVsIncremental</c> contract and is already covered; here we
    /// pin CPU and CUDA to each other given the same prefill+decode sequence.
    /// </remarks>
    [SkippableFact]
    public void CudaForward_KvCache_Decode_MatchesPrefill()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] prefillTokens = [5, 0, 3];
        int[] prefillPositions = [0, 1, 2];
        int[] decodeTokens = [7];
        int[] decodePositions = [3];

        using var fixture = CudaQwen3MoeHybridFixture.Build(seed: 31);

        float[] cpuDecode;
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        {
            using (ITensor prefill = model.Forward(prefillTokens, prefillPositions, deviceId: -1, kv))
            {
                // Prefill output discarded; we only care about the decode-step logits.
            }
            using ITensor decode = model.Forward(decodeTokens, decodePositions, deviceId: -1, kv);
            Assert.Equal(1, decode.Shape[0]);
            Assert.Equal(VocabSize, decode.Shape[1]);
            cpuDecode = new ReadOnlySpan<float>((void*)decode.DataPointer, VocabSize).ToArray();
        }

        float[] cudaDecode;
        using (var model = CudaQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
            deviceId: 0, ptxDir: ptxDir!))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        {
            using (ITensor prefill = model.Forward(prefillTokens, prefillPositions, deviceId: -1, kv))
            {
                // Prefill output discarded.
            }
            using ITensor decode = model.Forward(decodeTokens, decodePositions, deviceId: -1, kv);
            Assert.Equal(1, decode.Shape[0]);
            Assert.Equal(VocabSize, decode.Shape[1]);
            cudaDecode = new ReadOnlySpan<float>((void*)decode.DataPointer, VocabSize).ToArray();
        }

        AssertLogitsMatch(cpuDecode, cudaDecode);
    }

    // ─── Shared helpers ─────────────────────────────────────────────────────────

    private static float[] RunCpuPrefillLastRow(
        CudaQwen3MoeHybridFixture fixture, int[] tokenIds, int[] positions, bool useKvCache)
    {
        using var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize);
        if (useKvCache)
        {
            using var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            return CopyLastRow(logits, tokenIds.Length);
        }
        else
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            return CopyLastRow(logits, tokenIds.Length);
        }
    }

    private static float[] RunCudaPrefillLastRow(
        CudaQwen3MoeHybridFixture fixture, int[] tokenIds, int[] positions, string ptxDir, bool useKvCache)
    {
        using var model = CudaQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
            deviceId: 0, ptxDir: ptxDir);
        if (useKvCache)
        {
            using var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            return CopyLastRow(logits, tokenIds.Length);
        }
        else
        {
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            return CopyLastRow(logits, tokenIds.Length);
        }
    }

    private static float[] CopyLastRow(ITensor logits, int seqLen)
    {
        Assert.Equal(seqLen, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * VocabSize);
        return span.Slice((seqLen - 1) * VocabSize, VocabSize).ToArray();
    }

    private void AssertLogitsMatch(float[] cpu, float[] cuda)
    {
        Assert.Equal(cpu.Length, cuda.Length);

        // Emit the full row pair for diagnostic context — these tests are diagnostic-first
        // and the deltas matter when triaging a divergence.
        _out.WriteLine("col | cpu        | cuda       | |diff|");
        _out.WriteLine("----+------------+------------+----------");
        for (int c = 0; c < cpu.Length; c++)
        {
            _out.WriteLine($"{c,3} | {cpu[c],10:F6} | {cuda[c],10:F6} | {MathF.Abs(cpu[c] - cuda[c]):E3}");
        }

        for (int c = 0; c < cpu.Length; c++)
        {
            float pref = cpu[c];
            float incr = cuda[c];
            Assert.True(float.IsFinite(pref), $"col={c}: cpu logit non-finite: {pref}");
            Assert.True(float.IsFinite(incr), $"col={c}: cuda logit non-finite: {incr}");
            float diff = MathF.Abs(pref - incr);
            float bar = AbsTol + RelTol * MathF.Abs(pref);
            Assert.True(diff <= bar,
                $"col={c}: cpu={pref:F6} vs cuda={incr:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    //  Fixture
    // ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Synthetic Qwen3MoeHybrid weight fixture in unmanaged memory. Mirrors the private
    /// fixture in <c>Qwen3MoeHybridTransformerModelTests</c> but is internally visible to
    /// keep the parity tests fully self-contained without disturbing the CPU test file.
    /// </summary>
    /// <remarks>
    /// All projections are F32, 64-byte-aligned via <see cref="NativeMemory.AlignedAlloc"/>.
    /// The fixture owns every <see cref="nint"/> allocation and frees them in
    /// <see cref="Dispose"/>; both CPU and CUDA <c>BuildFromPrebuiltWeights</c> contracts
    /// state the caller retains ownership of the input pointers, so both models can be
    /// constructed and disposed inside this fixture's lifetime.
    /// </remarks>
    private sealed unsafe class CudaQwen3MoeHybridFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public Qwen3MoeLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static CudaQwen3MoeHybridFixture Build(
            int seed, float sharedGateAmplitude = 0.05f, bool gdnOnly = false)
        {
            var b = new CudaQwen3MoeHybridFixture();
            b.BuildInternal(seed, sharedGateAmplitude, gdnOnly);
            return b;
        }

        private void BuildInternal(int seed, float sharedGateAmplitude, bool gdnOnly)
        {
            var rng = new Random(seed);

            HybridLayerKind[] kinds = gdnOnly
                ? [HybridLayerKind.GatedDeltaNet, HybridLayerKind.GatedDeltaNet]
                : [HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention];
            int[] headCountKv = gdnOnly ? [0, 0] : [0, NumKvHeads];
            int[] ffnLen = [0, 0];

            var layout = new HybridLayerLayout
            {
                LayerKind = kinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLen,
            };

            var gdnConfig = new GatedDeltaNetConfig(
                FullAttnInterval: 2,
                NVHead: NVHead,
                NKHead: NKHead,
                DState: DState,
                DInner: DInner,
                DConv: DConv);

            var moeConfig = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = NumExpertsPerTok,
                MoeIntermediateSize = MoeIntermediate,
                NormTopKProb = true,
                SharedExpertIntermediateSize = SharedIntermediate,
                NumSharedExperts = 1,
                HasSharedExpertGate = true,
                DecoderSparseStep = 1,
            };

            Config = new ModelConfig
            {
                Architecture = Architecture.Qwen3MoeHybrid,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = 0,
                NumLayers = 2,
                NumAttentionHeads = NumAttentionHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: RoPEType.NeoX),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                HybridLayout = layout,
                GdnConfig = gdnConfig,
                Moe = moeConfig,
                ChatTemplate = null,
            };

            TokenEmbedPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            Layers = new Qwen3MoeLayerWeights[2];

            Layers[0] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                Gdn = BuildGdn(rng),
                FullAttn = null,
                Moe = BuildMoe(rng, sharedGateAmplitude),
            };

            Layers[1] = gdnOnly
                ? new Qwen3MoeLayerWeights
                {
                    AttnNormWeight = FillNormVec(HiddenSize, rng),
                    PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                    Gdn = BuildGdn(rng),
                    FullAttn = null,
                    Moe = BuildMoe(rng, sharedGateAmplitude),
                }
                : new Qwen3MoeLayerWeights
                {
                    AttnNormWeight = FillNormVec(HiddenSize, rng),
                    PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                    Gdn = null,
                    FullAttn = BuildFullAttn(rng),
                    Moe = BuildMoe(rng, sharedGateAmplitude),
                };
        }

        private GdnTokenMixingWeights BuildGdn(Random rng)
        {
            int convDim = (2 * NKHead + NVHead) * DState;
            int gdnKDim = NKHead * DState;
            int gdnVDim = NVHead * DState;
            int qkvOut = 2 * gdnKDim + gdnVDim;

            return new GdnTokenMixingWeights
            {
                QkvWeight = AllocFloatsUniform(HiddenSize * qkvOut, rng, 0.05f),
                QkvQuantType = QuantizationType.F32,
                QkvInputDim = HiddenSize,
                QkvOutputDim = qkvOut,
                GateWeight = AllocFloatsUniform(HiddenSize * gdnVDim, rng, 0.05f),
                GateQuantType = QuantizationType.F32,
                GateInputDim = HiddenSize,
                GateOutputDim = gdnVDim,
                A = NegativeRandom(NVHead, rng),
                AlphaWeight = AllocFloatsUniform(HiddenSize * NVHead, rng, 0.05f),
                AlphaQuantType = QuantizationType.F32,
                AlphaInputDim = HiddenSize,
                AlphaOutputDim = NVHead,
                BetaWeight = AllocFloatsUniform(HiddenSize * NVHead, rng, 0.05f),
                BetaQuantType = QuantizationType.F32,
                BetaInputDim = HiddenSize,
                BetaOutputDim = NVHead,
                Conv1dWeight = FillRandom(DConv * convDim, rng, 0.1f),
                Conv1dBias = new float[convDim],
                DtBias = FillRandom(NVHead, rng, 0.1f),
                SsmNormWeight = FillNormVec(DState, rng),
                OutWeight = AllocFloatsUniform(gdnVDim * HiddenSize, rng, 0.05f),
                OutQuantType = QuantizationType.F32,
                OutInputDim = gdnVDim,
                OutOutputDim = HiddenSize,
            };
        }

        private Qwen3FullAttnWeights BuildFullAttn(Random rng)
        {
            int qOut = 2 * NumAttentionHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim;
            return new Qwen3FullAttnWeights
            {
                QWeight = AllocFloatsUniform(HiddenSize * qOut, rng, 0.05f),
                QQuantType = QuantizationType.F32,
                QInputDim = HiddenSize,
                QOutputDim = qOut,
                KWeight = AllocFloatsUniform(HiddenSize * kvOut, rng, 0.05f),
                KQuantType = QuantizationType.F32,
                KInputDim = HiddenSize,
                KOutputDim = kvOut,
                VWeight = AllocFloatsUniform(HiddenSize * kvOut, rng, 0.05f),
                VQuantType = QuantizationType.F32,
                VInputDim = HiddenSize,
                VOutputDim = kvOut,
                OWeight = AllocFloatsUniform(oIn * HiddenSize, rng, 0.05f),
                OQuantType = QuantizationType.F32,
                OInputDim = oIn,
                OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads,
                QNormWeight = FillNormVec(HeadDim, rng),
                KNormWeight = FillNormVec(HeadDim, rng),
            };
        }

        private MoeLayerWeights BuildMoe(Random rng, float sharedGateAmplitude)
        {
            float[] gate = FillRandom(NumExperts * HiddenSize, rng, 0.05f);
            var w1 = new nint[NumExperts];
            var w2 = new nint[NumExperts];
            var w3 = new nint[NumExperts];
            for (int e = 0; e < NumExperts; e++)
            {
                w1[e] = AllocFloatsUniform(MoeIntermediate * HiddenSize, rng, 0.05f);
                w2[e] = AllocFloatsUniform(HiddenSize * MoeIntermediate, rng, 0.05f);
                w3[e] = AllocFloatsUniform(MoeIntermediate * HiddenSize, rng, 0.05f);
            }
            nint[] sharedGate = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedUp = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedDown = [AllocFloatsUniform(HiddenSize * SharedIntermediate, rng, 0.05f)];
            float[] sharedExpertGate = FillRandom(HiddenSize, rng, sharedGateAmplitude);

            return new MoeLayerWeights(
                gate: gate,
                w1: w1, w2: w2, w3: w3,
                numExperts: NumExperts,
                numExpertsPerTok: NumExpertsPerTok,
                hiddenSize: HiddenSize,
                intermediateSize: MoeIntermediate,
                normTopKProb: true,
                sharedGateProj: sharedGate,
                sharedUpProj: sharedUp,
                sharedDownProj: sharedDown,
                sharedIntermediateSize: SharedIntermediate,
                sharedExpertGate: sharedExpertGate);
        }

        private nint AllocFloatsUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++)
                dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
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
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
                NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
