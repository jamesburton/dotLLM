using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU↔CUDA last-token-logits parity test for the gpt-oss-style alternating
/// sliding-window/dense attention pattern (<see cref="ModelConfig.SlidingWindowPattern"/>
/// = 2: even layers windowed, odd layers dense). Exercises the wiring added by
/// #366 — <see cref="CudaSlidingWindowResolver"/> plumbed through every
/// per-layer attention-dispatch call site in <see cref="CudaTransformerModel"/> —
/// against the CPU oracle's <c>TransformerModel.GetLayerSlidingWindow</c>
/// (TransformerModel.cs:706-715), which the resolver is documented to mirror.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why seqLen = 3× window.</b> The fixture uses 4 layers, window = 8,
/// seqLen = 24. With pattern = 2, layers 0/2 are windowed (attend only the
/// last 8 tokens) and layers 1/3 are dense (attend the full causal history,
/// which by the last token is 24 tokens — 3× the window). This is the
/// discriminating shape: if the resolver ever applied the window to the
/// wrong layer (or leaked it onto every layer), a dense layer's true
/// receptive field would collapse from 24 tokens to 8, producing a logit
/// divergence well above FP16 rounding noise. A shorter sequence (e.g.
/// seqLen ≤ window) could pass by accident because windowed and dense
/// attention are numerically identical whenever the true context never
/// exceeds the window.
/// </para>
/// <para>
/// <b>Tolerance.</b> Same FP16-internal CUDA forward as
/// <see cref="CudaTransformerDenseRopeParityTests"/> (dense weights upload as
/// F16, cuBLAS/GEMV FP16 matmul) vs the F32 CPU oracle, so the noise floor is
/// dominated by FP16 GEMM rounding rather than the sliding-window logic
/// itself. Empirically calibrated on this exact fixture (2026-08-15, RTX
/// 3060 — see <c>task-3-report.md</c> for the raw runs):
/// <list type="bullet">
/// <item><description>Correct wiring (this test, as committed): observed max
/// |diff| = <b>1.533E-004</b>.</description></item>
/// <item><description>Mutation check — <see cref="CudaSlidingWindowResolver.Resolve"/>
/// temporarily forced to return the window unconditionally (window applied to
/// ALL layers, including the dense odd layers): observed max |diff| =
/// <b>8.088E-003</b> — about 53× the pass-case observation. Note this is
/// smaller than the plan's O(0.1+) prediction: these last-token logits are
/// themselves only O(0.05–0.18) in magnitude, so an O(0.1+) *absolute*
/// divergence isn't reachable on this fixture, and RMSNorm at every layer
/// junction damps the perturbation from the two affected (should-be-dense)
/// layers. In relative terms the divergence is 15–24% per logit — a strong,
/// unambiguous signal. <see cref="AbsTol"/> is pinned at ~10× the pass-case
/// observation (1.5e-3), which sits in the gap: comfortably above the
/// 1.533e-4 noise floor and ~5× below the 8.088e-3 mutation-fail signal
/// (the plan's suggested "~100x margin" would have landed the tolerance
/// above the mutation-fail value, defeating discrimination on this fixture —
/// same class of task-spec-vs-measured deviation as
/// <see cref="CudaTransformerDenseRopeParityTests"/> documents at its own
/// tolerance comment). Mutation reverted via <c>git restore</c>; verified via
/// <c>git status</c> that no production file carries the mutation in the
/// committed state, and the pass-case run repeated identically
/// (1.533E-004) after the revert.</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Skip behaviour.</b> Skips cleanly when no CUDA driver is present or the
/// PTX directory cannot be located — same convention as the rest of this
/// directory's GPU-gated parity tests.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class CudaAlternatingSwaParityTests
{
    private readonly ITestOutputHelper _out;
    public CudaAlternatingSwaParityTests(ITestOutputHelper output) => _out = output;

    // ── Fixture shape (per brief: 4 layers, window=8, pattern=2, seqLen=24=3×window;
    //    dims kept pairwise-distinct so a stride/dim mix-up would show up as a shape
    //    mismatch or a gross numeric divergence rather than silently matching). ──
    private const int VocabSize = 10;
    private const int HiddenSize = 12;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 6;
    private const int RopeDim = 6;
    private const int IntermediateSize = 20;
    private const int NumLayers = 4;
    private const int SeqLen = 24;
    private const int MaxSeqLen = SeqLen;
    private const int SlidingWindowSize = 8;
    private const int SlidingWindowPattern = 2;

    // Tolerance band: F32 CPU oracle vs CUDA FP16-internal forward, same regime as
    // CudaTransformerDenseRopeParityTests (dense CudaTransformerModel path is
    // FP16-internal by default). Empirically calibrated on this fixture (2026-08-15,
    // RTX 3060): as-committed (correct wiring), observed max |diff| = 1.533E-004.
    // The required mutation check (CudaSlidingWindowResolver.Resolve forced to
    // return the window unconditionally, applying it to the dense odd layers too)
    // produced observed max |diff| = 8.088E-003 on this same fixture -- ~53x the
    // pass-case observation. AbsTol is pinned at ~10x the pass-case margin (not the
    // plan's suggested ~100x: 100x would land above the 8.088E-003 mutation-fail
    // value and make the mutation check pass, defeating discrimination on this
    // fixture -- these last-token logits are only O(0.05-0.18), so an O(0.1+)
    // absolute mutation-fail divergence as the plan predicted isn't reachable here).
    // See the class remarks above for the full writeup.
    private const float AbsTol = 1.5e-3f;
    private const float RelTol = 2.0e-3f;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    /// <summary>
    /// Locate the PTX directory next to the test assembly (csproj copies
    /// <c>native/ptx/*.ptx</c> into the test output) or fall back to the
    /// repo's canonical <c>native/ptx/</c>.
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

    /// <summary>
    /// Prefill parity on a 4-layer alternating-SWA model (pattern=2, window=8,
    /// seqLen=24). Layers 0/2 are windowed, layers 1/3 are dense with a true
    /// receptive field 3x the window — the shape that catches a resolver bug
    /// that applies the window to the wrong layer set.
    /// </summary>
    [SkippableFact]
    public void CudaForward_AlternatingSwa_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        var rng = new Random(7);
        int[] tokenIds = new int[SeqLen];
        int[] positions = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++)
        {
            tokenIds[i] = rng.Next(VocabSize);
            positions[i] = i;
        }

        using var fixture = AlternatingSwaFixture.Build(seed: 23);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions);
        float[] cudaLast = RunCudaPrefillLastRow(fixture, tokenIds, positions, ptxDir!);

        AssertLogitsMatch(cpuLast, cudaLast);
    }

    private static float[] RunCpuPrefillLastRow(
        AlternatingSwaFixture fixture, int[] tokenIds, int[] positions)
    {
        using var model = TransformerModel.BuildFromPrebuiltWeights(fixture.Weights, fixture.Config);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        // CPU Forward returns [seqLen, vocab]; slice the last row.
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * VocabSize);
        return span.Slice((tokenIds.Length - 1) * VocabSize, VocabSize).ToArray();
    }

    private static float[] RunCudaPrefillLastRow(
        AlternatingSwaFixture fixture, int[] tokenIds, int[] positions, string ptxDir)
    {
        using var model = CudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, deviceId: 0, ptxDir: ptxDir);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        // CUDA Forward returns [1, vocab] — only the last token's logits, by
        // design (saves an LM-head GEMM on prefill). The CPU run above is
        // sliced to match.
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize);
        return span.ToArray();
    }

    private void AssertLogitsMatch(float[] cpu, float[] cuda)
    {
        Assert.Equal(cpu.Length, cuda.Length);

        // Emit the full row pair for diagnostic context — these tests are
        // diagnostic-first and the deltas matter when triaging a divergence.
        float maxDiff = 0f;
        _out.WriteLine("col | cpu        | cuda       | |diff|");
        _out.WriteLine("----+------------+------------+----------");
        for (int c = 0; c < cpu.Length; c++)
        {
            float d = MathF.Abs(cpu[c] - cuda[c]);
            maxDiff = MathF.Max(maxDiff, d);
            _out.WriteLine($"{c,3} | {cpu[c],10:F6} | {cuda[c],10:F6} | {d:E3}");
        }
        _out.WriteLine($"max |diff| = {maxDiff:E3}");

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

    // ──────────────────────────────────────────────────────────────────────
    //  Fixture
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Synthetic 4-layer alternating-SWA transformer weight fixture in
    /// unmanaged memory. Mirrors <c>CudaTransformerDenseRopeParityTests.DenseFixture</c>
    /// (tests/DotLLM.Tests.Unit/Cuda/CudaTransformerDenseRopeParityTests.cs:236-353) —
    /// same construction mechanism, extended with
    /// <see cref="ModelConfig.SlidingWindowSize"/>/<see cref="ModelConfig.SlidingWindowPattern"/>
    /// and a 4-layer / 24-token shape. Owns every F32 aligned allocation and the
    /// wrapping <see cref="TransformerWeights"/>. Both CPU and CUDA
    /// <c>BuildFromPrebuiltWeights</c> entry points state the caller retains
    /// ownership of the input pointers, so both models can be constructed and
    /// disposed inside this fixture's lifetime.
    /// </summary>
    private sealed unsafe class AlternatingSwaFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights { get; private set; } = null!;

        public static AlternatingSwaFixture Build(int seed)
        {
            var b = new AlternatingSwaFixture();
            b.BuildInternal(seed);
            return b;
        }

        private void BuildInternal(int seed)
        {
            var rng = new Random(seed);

            Config = new ModelConfig
            {
                Architecture = Architecture.Llama,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = IntermediateSize,
                NumLayers = NumLayers,
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
                ChatTemplate = null,
                // gpt-oss-style alternating sliding-window/dense pattern: with
                // pattern=2, layer%2 < 1 (even layers 0,2) are windowed to 8
                // tokens; odd layers (1,3) are dense (full causal history).
                SlidingWindowSize = SlidingWindowSize,
                SlidingWindowPattern = SlidingWindowPattern,
            };

            nint tokenEmbed = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);
            float[] outputNorm = FillNormVec(HiddenSize, rng);
            nint output = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            int qOut = NumAttentionHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim;

            var layers = new TransformerLayerWeights[NumLayers];
            for (int i = 0; i < NumLayers; i++)
            {
                float[] attnNorm = FillNormVec(HiddenSize, rng);
                float[] ffnNorm = FillNormVec(HiddenSize, rng);

                nint qW = AllocFloatsUniform(qOut * HiddenSize, rng, 0.05f);
                nint kW = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f);
                nint vW = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f);
                nint oW = AllocFloatsUniform(HiddenSize * oIn, rng, 0.05f);

                nint gateW = AllocFloatsUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint upW = AllocFloatsUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint downW = AllocFloatsUniform(HiddenSize * IntermediateSize, rng, 0.05f);

                layers[i] = new TransformerLayerWeights(
                    attnNormWeight: attnNorm,
                    qWeight: qW, qQuantType: QuantizationType.F32, qOutputDim: qOut, qInputDim: HiddenSize,
                    kWeight: kW, kQuantType: QuantizationType.F32, kOutputDim: kvOut, kInputDim: HiddenSize,
                    vWeight: vW, vQuantType: QuantizationType.F32, vOutputDim: kvOut, vInputDim: HiddenSize,
                    oWeight: oW, oQuantType: QuantizationType.F32, oOutputDim: HiddenSize, oInputDim: oIn,
                    ffnNormWeight: ffnNorm,
                    gateWeight: gateW, gateQuantType: QuantizationType.F32, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                    upWeight: upW, upQuantType: QuantizationType.F32, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                    downWeight: downW, downQuantType: QuantizationType.F32, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
            }

            // CreateFromSafetensors is the public factory; despite the name it
            // is the canonical entry point for handing pre-built F32 host
            // pointers to TransformerWeights (used by every non-GGUF code path).
            // The ownedAllocations list is empty because this fixture owns the
            // raw allocations directly and frees them in Dispose.
            Weights?.Dispose();
            Weights = TransformerWeights.CreateFromSafetensors(
                tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
                vocabSize: VocabSize, hiddenSize: HiddenSize,
                layers: layers,
                outputNormWeight: outputNorm,
                outputWeight: output, outputQt: QuantizationType.F32,
                outputM: VocabSize, outputK: HiddenSize,
                ownedAllocations: new List<nint>());
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

        private static float[] FillNormVec(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        public void Dispose()
        {
            Weights?.Dispose();
            foreach (var p in _allocs)
                NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
