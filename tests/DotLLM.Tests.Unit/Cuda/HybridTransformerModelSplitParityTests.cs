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
/// CPU↔Hybrid (GPU/CPU split) last-token-logits parity tests for
/// <see cref="HybridTransformerModel"/> — the <c>--gpu-layers N</c> partial-offload
/// path. Pinned to the pure-CPU oracle (<see cref="TransformerModel"/>) on a
/// synthetic 4-layer Llama-style fixture (hidden=32, vocab=8) with the hybrid
/// model running the first 2 of 4 layers on GPU and the remaining 2 on CPU.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> The RoPE NeoX dispatch fix (#36) touched
/// <see cref="HybridTransformerModel"/>'s RoPE-type translation site, but the
/// only synthetic parity coverage added there was
/// <see cref="CudaTransformerDenseRopeParityTests"/> — the pure-CUDA dense path.
/// The hybrid split (GPU layers → D2H FP16→F32 boundary → CPU layers + LM head)
/// has its own additional moving parts that the dense test cannot reach:
/// </para>
/// <list type="bullet">
///   <item>The GPU-side RoPE rotation must match the CPU oracle on the GPU
///         layers (same fix as #36, different layer count).</item>
///   <item>The boundary D2H transfer must read GPU FP16 and write CPU F32 in
///         the right direction — see <c>HybridTransformerModel.ConvertFp16ToFp32</c>.</item>
///   <item>The CPU layers downstream must consume the FP16-rounded hidden state
///         and integrate cleanly with the residual stream + final RMS norm + LM
///         head, all of which run F32 on the CPU oracle too.</item>
/// </list>
/// <para>
/// This is what most users with limited VRAM actually run, so a CI-level
/// parity pin matters even for synthetic shapes.
/// </para>
/// <para>
/// <b>Tolerance band.</b> Carried over from the dense parity test (1.5e-3 abs /
/// 5e-3 rel) — empirically the hybrid path on this 4-layer F32 fixture lands
/// inside the band by a clear margin even though it stacks two error sources
/// (FP16 GEMM rounding on the GPU half + the F16↔F32 boundary conversion's
/// ULP-scale noise + 2 more CPU F32 layers cumulating that noise through the
/// residual stream + a final LM-head GEMV on perturbed inputs). The
/// boundary-bug trap-and-revert below confirms a real RoPE/transfer regression
/// would breach this band.
/// </para>
/// <para>
/// <b>Trap-the-bug evidence.</b> Replacing <c>ConvertFp16ToFp32</c>'s
/// <c>TensorPrimitives.ConvertToSingle</c> with a shift-by-one offset
/// (<c>dst[i] = (float)((Half*)srcFp16)[i + 1]</c> — reading neighbouring
/// hidden elements at the boundary) pushed max|diff| from ~9.7e-4 / ~5.2e-4
/// (NeoX / Norm baseline) to ~5.87e-1 — roughly a 600× gap, both variants
/// failing immediately on column 0. The bug is intentionally NOT committed;
/// only this summary and the inline tolerance-constant comment remain.
/// </para>
/// <para>
/// <b>Skip behaviour.</b> Each test skips cleanly when no CUDA driver is
/// present or the PTX directory cannot be located.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class HybridTransformerModelSplitParityTests
{
    private readonly ITestOutputHelper _out;
    public HybridTransformerModelSplitParityTests(ITestOutputHelper output) => _out = output;

    // ── Fixture shape ───────────────────────────────────────────────────────
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int NumGpuLayers = 2; // First 2 of 4 layers on GPU
    private const int MaxSeqLen = 8;

    // Tolerance band: F32 CPU oracle vs Hybrid GPU/CPU split. Same shape as
    // CudaTransformerDenseRopeParityTests — the hybrid path stacks the dense
    // CUDA forward's FP16 GEMM noise (≈8e-4 there) on top of an F16↔F32
    // boundary conversion (≈5e-4 ULP-scale on the hidden-size=32 transfer)
    // and 2 more CPU F32 layers downstream. Empirically max|diff| ≈ 9.7e-4
    // (NeoX) / 5.2e-4 (Norm) on this fixture — inside the 1.5e-3 band by a
    // clear margin. A synthetic shift-by-one regression at
    // HybridTransformerModel.ConvertFp16ToFp32 pushes max|diff| to ~5.87e-1
    // (a clean ~600× gap), so a real boundary or RoPE bug fails the band
    // cleanly without the band needing to be tightened. The task brief
    // allowed surfacing a >2× tolerance overshoot as a finding; this fixture
    // stayed comfortably inside it, no widening required.
    private const float AbsTol = 1.5e-3f;
    private const float RelTol = 5e-3f;

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
    /// <c>native/ptx/*.ptx</c> into the test output) or fall back to the
    /// repo's canonical <c>native/ptx/</c>. Mirrors the helper in
    /// <see cref="CudaTransformerDenseRopeParityTests"/>.
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
    /// Prefill parity on a 4-layer dense NeoX-RoPE model with the first 2
    /// layers offloaded to GPU. Exercises the full hybrid split:
    /// embedding + 2 GPU layers (FP16) → F16→F32 boundary → 2 CPU layers (F32)
    /// → final RMSNorm + LM head on CPU.
    /// </summary>
    [SkippableFact]
    public void HybridForward_DenseNeoxRope_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 17, ropeType: RoPEType.NeoX);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions);
        float[] hybridLast = RunHybridPrefillLastRow(fixture, tokenIds, positions, ptxDir!);

        AssertLogitsMatch(cpuLast, hybridLast, "NeoX");
    }

    /// <summary>
    /// Companion test on the same 4-layer fixture shape with
    /// <see cref="RoPEType.Norm"/>. Mirrors the dense Norm-RoPE companion:
    /// shares the dispatch path with NeoX but uses the value <c>0</c> which
    /// is encoded identically on both sides — should keep matching whether
    /// the RoPE translator is correct or not. Useful as a control for future
    /// refactors that might break Norm while fixing NeoX (or vice versa).
    /// </summary>
    [SkippableFact]
    public void HybridForward_DenseNormRope_PrefillVsCpu_LastTokenLogitsMatch()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 17, ropeType: RoPEType.Norm);

        float[] cpuLast = RunCpuPrefillLastRow(fixture, tokenIds, positions);
        float[] hybridLast = RunHybridPrefillLastRow(fixture, tokenIds, positions, ptxDir!);

        AssertLogitsMatch(cpuLast, hybridLast, "Norm");
    }

    private static float[] RunCpuPrefillLastRow(
        DenseFixture fixture, int[] tokenIds, int[] positions)
    {
        using var model = TransformerModel.BuildFromPrebuiltWeights(fixture.Weights, fixture.Config);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        // CPU Forward returns [seqLen, vocab]; slice the last row.
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * VocabSize);
        return span.Slice((tokenIds.Length - 1) * VocabSize, VocabSize).ToArray();
    }

    private static float[] RunHybridPrefillLastRow(
        DenseFixture fixture, int[] tokenIds, int[] positions, string ptxDir)
    {
        using var model = HybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numGpuLayers: NumGpuLayers,
            deviceId: 0, threading: ThreadingConfig.SingleThreaded, ptxDir: ptxDir);
        // The hybrid model's CPU half produces the LM head, so it returns
        // [1, vocabSize] — only the last token's logits, matching the CUDA
        // path. The CPU oracle above is sliced to match.
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize);
        return span.ToArray();
    }

    private void AssertLogitsMatch(float[] cpu, float[] hybrid, string variant)
    {
        Assert.Equal(cpu.Length, hybrid.Length);

        // Emit the full row pair for diagnostic context. The dense parity
        // tests follow the same diagnostic-first convention; deltas matter
        // when triaging a regression in either the GPU half, the boundary
        // transfer, or the CPU continuation.
        _out.WriteLine($"[{variant}] col | cpu        | hybrid     | |diff|");
        _out.WriteLine($"[{variant}] ----+------------+------------+----------");
        float maxAbs = 0f;
        double sumSq = 0.0;
        for (int c = 0; c < cpu.Length; c++)
        {
            float d = MathF.Abs(cpu[c] - hybrid[c]);
            if (d > maxAbs) maxAbs = d;
            sumSq += (double)d * d;
            _out.WriteLine($"[{variant}] {c,3} | {cpu[c],10:F6} | {hybrid[c],10:F6} | {d:E3}");
        }
        double rms = Math.Sqrt(sumSq / cpu.Length);
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3}  rms={rms:E3}  AbsTol={AbsTol:E3}");

        for (int c = 0; c < cpu.Length; c++)
        {
            float pref = cpu[c];
            float hyb = hybrid[c];
            Assert.True(float.IsFinite(pref), $"{variant} col={c}: cpu logit non-finite: {pref}");
            Assert.True(float.IsFinite(hyb), $"{variant} col={c}: hybrid logit non-finite: {hyb}");
            float diff = MathF.Abs(pref - hyb);
            float bar = AbsTol + RelTol * MathF.Abs(pref);
            Assert.True(diff <= bar,
                $"{variant} col={c}: cpu={pref:F6} vs hybrid={hyb:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  Fixture
    // ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Synthetic 4-layer dense transformer weight fixture in unmanaged memory.
    /// Mirrors <see cref="CudaTransformerDenseRopeParityTests"/>'s fixture but
    /// with <c>NumLayers = 4</c> so the hybrid split has at least one layer
    /// on each side. Owns every F32 aligned allocation and the wrapping
    /// <see cref="TransformerWeights"/>.
    /// </summary>
    /// <remarks>
    /// Both <see cref="TransformerModel.BuildFromPrebuiltWeights"/> and
    /// <see cref="HybridTransformerModel.BuildFromPrebuiltWeights"/> document
    /// that the caller retains ownership of the input pointers. The downstream
    /// models still call <c>Dispose</c> on the shared <see cref="TransformerWeights"/>
    /// — this is benign because the fixture passed an empty
    /// <c>ownedAllocations</c> list to <see cref="TransformerWeights.CreateFromSafetensors"/>,
    /// so double-Dispose only nulls the already-nulled <c>RepackedLayers</c>.
    /// This matches the contract used by
    /// <see cref="CudaTransformerDenseRopeParityTests"/>'s dense fixture.
    /// </remarks>
    private sealed unsafe class DenseFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights = null!;

        public static DenseFixture Build(int seed, RoPEType ropeType)
        {
            var b = new DenseFixture();
            b.BuildInternal(seed, ropeType);
            return b;
        }

        private void BuildInternal(int seed, RoPEType ropeType)
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
                // Identical theta on both NeoX and Norm tests so the parity
                // check exercises the pair-pattern axis exclusively, matching
                // the dense parity tests' configuration.
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: ropeType),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                ChatTemplate = null,
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
            // raw allocations directly and frees them in Dispose — see remarks
            // on the fixture class for the double-Dispose contract.
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
