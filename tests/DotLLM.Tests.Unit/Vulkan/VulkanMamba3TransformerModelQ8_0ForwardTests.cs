using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity tests for the Vulkan Mamba-3 forward path with Q8_0 projection-weight
/// upload. Mirrors the F32 Mamba-3 forward tests but injects Q8_0 raw-byte overlays onto
/// <see cref="Mamba3Weights"/> after the SafeTensors load — the production CPU loader
/// (<see cref="Mamba3WeightLoader"/>) is hard-locked to F32 source dtypes, and SafeTensors
/// itself has no native Q8_0 dtype, so the Q8_0 GEMV/GEMM dispatch can only be exercised
/// via direct overlay injection.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> The three matmul-target projections (<c>in_proj</c>, <c>out_proj</c>,
/// <c>lm_head</c>) run through the Q8_0 path. The token embedding always lands as F32
/// (the embedding gather is a byte-offset <c>vkCmdCopyBuffer</c> that needs a contiguous
/// F32 layout, same convention as <see cref="VulkanWeights"/> /
/// <see cref="VulkanNemotronHWeights"/>). Norms, biases, <c>D</c>, <c>dt_bias</c>, and the
/// MIMO per-rank weights stay F32 — production GGUF Mamba-3 builds keep these unquantised.
/// </para>
/// <para>
/// <b>Method.</b> The fixture is generated F32 and loaded via <see cref="Mamba3WeightLoader"/>
/// into a <see cref="Mamba3Weights"/>. For each matmul-target projection the F32 row data
/// is quantised with <see cref="MatMul.QuantizeF32ToQ8_0"/>, the resulting Q8_0 raw bytes
/// are stored in unmanaged memory and pinned to the overlay slots
/// (<see cref="Mamba3Weights.LmHeadQ8Ptr"/>,
/// <see cref="Mamba3LayerQuantOverlay.InProjQ8Ptr"/>,
/// <see cref="Mamba3LayerQuantOverlay.OutProjQ8Ptr"/>). The corresponding F32 pointer slots
/// in <see cref="Mamba3Weights"/> / <see cref="Mamba3LayerWeights"/> are replaced with
/// freshly-allocated buffers that hold the dequantised-from-Q8_0 values, so the CPU oracle
/// reads exactly the (lossy) values the Vulkan path consumes via the Q8_0 kernels. The
/// safetensors mmap region is read-only and stays untouched. Same approach as the Q8_0
/// MoE / NemotronH parity tests.
/// </para>
/// <para>
/// <b>Dimensions.</b> All Q8_0 contraction axes (<c>hiddenSize</c> for <c>in_proj</c> and
/// <c>lm_head</c>; <c>dInner = nHead * headDim</c> for <c>out_proj</c>) are bumped to
/// multiples of 32 — a hard requirement of the Vulkan Q8_0 matmul kernels. Tolerance
/// abs 5e-3 / rel 1e-3, same envelope as the existing F32 Mamba-3 forward parity tests;
/// Q8_0 is essentially lossless against this bar.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelQ8_0ForwardTests : IDisposable
{
    // Q8_0-friendly dimensions: HiddenSize must be a multiple of 32 (in_proj / lm_head
    // contract along hidden); DInner = NumHeads * HeadDim must also be a multiple of 32
    // (out_proj contracts along d_inner). The F32 fixture (HiddenSize=8, DInner=16) is
    // too small — bump to 32 / 32 here. Other dims are unconstrained.
    private const int HiddenSize = 32;
    private const int VocabSize = 16;
    private const int NumHeads = 4;
    private const int HeadDim = 8;             // d_inner = 32
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int NumBcHeads = 1;          // SISO; G = 1
    private const int BcDim = StateSize * NumBcHeads;
    // num_rope_angles = (state_size * rope_fraction) / 2 with rope_fraction=0.5 → 2.
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // MIMO fixture dimensions — same model dims as SISO but with rank > 1 B/C expansion.
    private const int MimoRank = 2;
    private const int MimoBcDim = StateSize * NumBcHeads * MimoRank;
    private const int MimoDInProj = 2 * DInner + 2 * MimoBcDim + 3 * NumHeads + NumRopeAngles;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;
    // Native allocations made by the Q8_0 overlay (Q8_0 raw-byte blocks + dequantised-
    // back-to-F32 mirror buffers). Freed in Dispose so memory survives the test methods.
    private readonly List<nint> _q8Allocs = new();

    public VulkanMamba3TransformerModelQ8_0ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-q8-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (var p in _q8Allocs)
            NativeMemory.AlignedFree((void*)p);
        _q8Allocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_Q8_0_Prefill_SingleLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 1, seqLen: 4, seed: 7);

    [SkippableFact]
    public void Forward_Q8_0_Prefill_MultiLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 2, seqLen: 4, seed: 17);

    [SkippableFact]
    public void Forward_Q8_0_Prefill_SingleLayer_Mimo_MatchesCpuReference()
        => AssertVulkanMatchesCpuMimo(numLayers: 1, seqLen: 4, seed: 23);

    [SkippableFact]
    public void Forward_Q8_0_Decode_WithStateContinuation_Siso_MatchesCpuReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int prefillLen = 4;
        const int seed = 31;
        string path = Path.Combine(_scratch, $"m3-q8-decode-s{seed}.safetensors");
        WriteSisoFixture(path, numLayers: 2, seed);
        ModelConfig config = BuildSisoConfig(numLayers: 2);

        int[] prefillIds = new int[prefillLen];
        int[] prefillPositions = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prefillIds[i] = i % VocabSize; prefillPositions[i] = i; }
        int[] decodeIds = [prefillLen % VocabSize];
        int[] decodePositions = [prefillLen];

        // ── CPU oracle: prefill then decode, same persistent state instance ───
        float[] cpuDecodeLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using var state = new Mamba3State(config);
            using (ITensor _ = model.Forward(prefillIds, prefillPositions, deviceId: -1, state)) { }
            using ITensor decodeLogits = model.Forward(decodeIds, decodePositions, deviceId: -1, state);
            cpuDecodeLogits = CopyLogits(decodeLogits);
        }

        // ── Vulkan: same prefill-then-decode sequence on one model instance (state
        //    is owned internally and threads across calls).
        float[] vkDecodeLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var device = VulkanDevice.Create();
            using var model = VulkanMamba3TransformerModel.BuildOnDevice(device, config, weights, spvDir);
            using (ITensor _ = model.Forward(prefillIds, prefillPositions, deviceId: -1)) { }
            using ITensor decodeLogits = model.Forward(decodeIds, decodePositions, deviceId: -1);
            Assert.Equal(1, decodeLogits.Shape[0]);
            Assert.Equal(VocabSize, decodeLogits.Shape[1]);
            vkDecodeLogits = CopyLogits(decodeLogits);
            // Vulkan model takes ownership of `weights` via BuildOnDevice → Upload, but
            // doesn't dispose the Mamba3Weights wrapper. The mmap anchor (sf) holds the
            // F32 weights alive; the overlay-owned Q8_0 buffers are freed in Dispose.
        }

        // CPU returns [seqLen, vocab] for the decode call; Vulkan returns [1, vocab].
        // Compare element-wise on the single token's logits.
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuDecodeLogits[c];
            float vk = vkDecodeLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"Q8 decode-continuation col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private void AssertVulkanMatchesCpuSiso(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-q8-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteSisoFixture(path, numLayers, seed);
        ModelConfig config = BuildSisoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var device = VulkanDevice.Create();
            using var model = VulkanMamba3TransformerModel.BuildOnDevice(device, config, weights, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"Q8 SISO numLayers={numLayers}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private void AssertVulkanMatchesCpuMimo(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-q8-mimo-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteMimoFixture(path, numLayers, seed);
        ModelConfig config = BuildMimoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ8OverlayInPlace(weights);
            using var device = VulkanDevice.Create();
            using var model = VulkanMamba3TransformerModel.BuildOnDevice(device, config, weights, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"Q8 MIMO numLayers={numLayers}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// Quantises every matmul-target projection (<c>lm_head</c> and per-layer
    /// <c>in_proj</c>/<c>out_proj</c>) to Q8_0, attaches the raw bytes to the overlay slots
    /// on <paramref name="weights"/>, and replaces the F32 source pointers with freshly-
    /// allocated buffers carrying dequantised-from-Q8_0 values so the CPU oracle consumes
    /// the same effective values as the Vulkan path. The safetensors mmap region is
    /// <see cref="MemoryMappedFiles.MemoryMappedFileAccess.Read"/>-only and stays untouched
    /// — same caveat as the Q8_0 MoE precedent at commit 9d81f17.
    /// </summary>
    private unsafe void ApplyQ8OverlayInPlace(Mamba3Weights weights)
    {
        int numLayers = weights.Layers.Length;

        // Allocate one overlay slot per layer.
        weights.LayerOverlays = new Mamba3LayerQuantOverlay[numLayers];
        for (int i = 0; i < numLayers; i++)
            weights.LayerOverlays[i] = new Mamba3LayerQuantOverlay();

        // ── lm_head ([vocab, hidden]) ─────────────────────────────────────────
        // The loader's F32 LmHead handle points either at the safetensors mmap region
        // (read-only) or at a 64-byte-aligned scratch. We allocate a fresh F32 buffer,
        // populate it with dequantised-from-Q8 values, and rewire the LmHead handle to
        // point at it. The original mmap region is left alone.
        {
            int vocab = weights.LmHead.Shape[0];
            int hidden = weights.LmHead.Shape[1];
            long elems = (long)vocab * hidden;
            ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)weights.LmHead.Pointer, checked((int)elems));
            nint q8Ptr = QuantizeF32ToQ8(srcF32, vocab, hidden);
            nint freshF32 = AllocAlignedF32(elems);
            DequantizeQ8RowsToF32Span(q8Ptr, vocab, hidden,
                new Span<float>((void*)freshF32, checked((int)elems)));

            // Rewire LmHead → fresh F32 buffer (CPU oracle reads it). Mark OwnsMemory=false
            // because the Mamba3Weights.Dispose path does NOT free our overlay buffer
            // (we own it via _q8Allocs). The original handle's lifetime is anchored by the
            // safetensors file held by the test for the duration of the call.
            weights.LmHead = new Mamba3TensorHandle(
                Pointer: freshF32,
                Shape: weights.LmHead.Shape,
                SourceDType: weights.LmHead.SourceDType,
                OwnsMemory: false);
            weights.LmHeadQ8Ptr = q8Ptr;
            weights.LmHeadQuantTypeOverlay = QuantizationType.Q8_0;
        }

        // ── per-layer in_proj / out_proj ─────────────────────────────────────
        for (int i = 0; i < numLayers; i++)
        {
            var lw = weights.Layers[i];
            var ov = weights.LayerOverlays[i];

            // in_proj [d_in_proj, hidden]
            {
                int rows = lw.InProj.Shape[0];
                int cols = lw.InProj.Shape[1];
                long elems = (long)rows * cols;
                ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)lw.InProj.Pointer, checked((int)elems));
                nint q8Ptr = QuantizeF32ToQ8(srcF32, rows, cols);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeQ8RowsToF32Span(q8Ptr, rows, cols,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    InProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.InProj.Shape,
                        SourceDType: lw.InProj.SourceDType, OwnsMemory: false),
                };
                ov.InProjQ8Ptr = q8Ptr;
                ov.InProjQuantTypeOverlay = QuantizationType.Q8_0;
            }

            // out_proj [hidden, d_inner]
            {
                int rows = lw.OutProj.Shape[0];
                int cols = lw.OutProj.Shape[1];
                long elems = (long)rows * cols;
                ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)lw.OutProj.Pointer, checked((int)elems));
                nint q8Ptr = QuantizeF32ToQ8(srcF32, rows, cols);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeQ8RowsToF32Span(q8Ptr, rows, cols,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    OutProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.OutProj.Shape,
                        SourceDType: lw.OutProj.SourceDType, OwnsMemory: false),
                };
                ov.OutProjQ8Ptr = q8Ptr;
                ov.OutProjQuantTypeOverlay = QuantizationType.Q8_0;
            }

            weights.Layers[i] = lw;
        }
    }

    /// <summary>Allocates a 64-byte-aligned F32 buffer of <paramref name="elems"/> elements;
    /// freed in <see cref="Dispose"/>.</summary>
    private unsafe nint AllocAlignedF32(long elems)
    {
        long bytes = elems * sizeof(float);
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _q8Allocs.Add(ptr);
        return ptr;
    }

    /// <summary>
    /// Quantises a [<paramref name="rows"/>, <paramref name="cols"/>] row-major F32 matrix
    /// to Q8_0 row-stride bytes via <see cref="MatMul.QuantizeF32ToQ8_0"/>. Returns an
    /// aligned native allocation that the test owns (freed in <see cref="Dispose"/>).
    /// <paramref name="cols"/> must be a multiple of the Q8_0 group size (32).
    /// </summary>
    private unsafe nint QuantizeF32ToQ8(ReadOnlySpan<float> src, int rows, int cols)
    {
        if ((cols % 32) != 0)
            throw new InvalidOperationException(
                $"Q8_0 quantisation requires cols % 32 == 0 (got cols={cols}). Bump fixture dims.");
        long rowBytes = Dequantize.RowByteSize(cols, QuantizationType.Q8_0);
        long totalBytes = rowBytes * rows;
        nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        _q8Allocs.Add(dst);

        fixed (float* srcPtr = src)
        {
            for (int row = 0; row < rows; row++)
            {
                byte* rowDst = (byte*)dst + (long)row * rowBytes;
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * cols, rowDst, cols);
            }
        }
        return dst;
    }

    /// <summary>
    /// Dequantises Q8_0 row-stride bytes back into F32 row-major elements at
    /// <paramref name="dst"/>. Used so the CPU oracle consumes the (lossy) dequantised
    /// values matching what the Vulkan side computes via the Q8_0 kernels.
    /// </summary>
    private static unsafe void DequantizeQ8RowsToF32Span(nint q8Ptr, int rows, int cols, Span<float> dst)
    {
        long rowBytes = Dequantize.RowByteSize(cols, QuantizationType.Q8_0);
        for (int row = 0; row < rows; row++)
        {
            nint rowSrc = q8Ptr + (nint)((long)row * rowBytes);
            Dequantize.ToFloat32(rowSrc, cols, QuantizationType.Q8_0, dst.Slice(row * cols, cols));
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildSisoConfig(int numLayers)
    {
        var m3 = new Mamba3Config
        {
            StateSize = StateSize,
            NumHeads = NumHeads,
            HeadDim = HeadDim,
            Expand = Expand,
            NumGroups = NumBcHeads,
            ChunkSize = 4,
            IsMimo = false,
            MimoRank = 4,
            AFloor = 1e-4f,
            DtInitFloor = 1e-4f,
            DtMin = 1e-3f,
            DtMax = 0.1f,
            UseL2Warp = false,
            RopeFraction = 0.5f,
            IsOutProjNorm = false,
            RescalePrenormResidual = true,
            ResidualInFp32 = true,
        };
        return new ModelConfig
        {
            Architecture = Architecture.Mamba3,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = 0,
            NumLayers = numLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 32,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            SlidingWindowSize = null,
            MlaConfig = null,
            HybridLayout = null,
            SsmConfig = null,
            Mamba3Config = m3,
            ChatTemplate = null,
        };
    }

    private static ModelConfig BuildMimoConfig(int numLayers)
    {
        var m3 = new Mamba3Config
        {
            StateSize = StateSize,
            NumHeads = NumHeads,
            HeadDim = HeadDim,
            Expand = Expand,
            NumGroups = NumBcHeads,
            ChunkSize = 4,
            IsMimo = true,
            MimoRank = MimoRank,
            AFloor = 1e-4f,
            DtInitFloor = 1e-4f,
            DtMin = 1e-3f,
            DtMax = 0.1f,
            UseL2Warp = false,
            RopeFraction = 0.5f,
            IsOutProjNorm = false,
            RescalePrenormResidual = true,
            ResidualInFp32 = true,
        };
        return new ModelConfig
        {
            Architecture = Architecture.Mamba3,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = 0,
            NumLayers = numLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 32,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            SlidingWindowSize = null,
            MlaConfig = null,
            HybridLayout = null,
            SsmConfig = null,
            Mamba3Config = m3,
            ChatTemplate = null,
        };
    }

    private static void WriteSisoFixture(string path, int numLayers, int seed)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, Mamba3TensorMapping.FinalNorm, [HiddenSize], 0.5f, seed + 1, center: 1.0f, jitter: 0.1f);
        AddRand(b, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < numLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            AddRand(b, Mamba3TensorMapping.LayerNorm(i), [HiddenSize],
                    amplitude: 0.5f, seed: s + 0, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], 0.02f, s + 1);
            AddRand(b, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], 0.05f, s + 2);
            AddRand(b, Mamba3TensorMapping.BNorm(i), [StateSize],
                    amplitude: 0.5f, seed: s + 3, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.CNorm(i), [StateSize],
                    amplitude: 0.5f, seed: s + 4, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize], 0.02f, s + 5);
            AddRand(b, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize], 0.02f, s + 6);
            AddRand(b, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, s + 7);
            AddRand(b, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, s + 8);
        }

        b.WriteTo(path);
    }

    private static void WriteMimoFixture(string path, int numLayers, int seed)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, Mamba3TensorMapping.FinalNorm, [HiddenSize], 0.5f, seed + 1, center: 1.0f, jitter: 0.1f);
        AddRand(b, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < numLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            AddRand(b, Mamba3TensorMapping.LayerNorm(i), [HiddenSize],
                    amplitude: 0.5f, seed: s + 0, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.InProj(i), [MimoDInProj, HiddenSize], 0.02f, s + 1);
            AddRand(b, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], 0.05f, s + 2);
            AddRand(b, Mamba3TensorMapping.BNorm(i), [StateSize],
                    amplitude: 0.5f, seed: s + 3, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.CNorm(i), [StateSize],
                    amplitude: 0.5f, seed: s + 4, center: 1.0f, jitter: 0.1f);
            AddRand(b, Mamba3TensorMapping.BBias(i), [NumHeads, MimoRank, StateSize], 0.02f, s + 5);
            AddRand(b, Mamba3TensorMapping.CBias(i), [NumHeads, MimoRank, StateSize], 0.02f, s + 6);
            AddRand(b, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, s + 7);
            AddRand(b, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, s + 8);
            AddRand(b, Mamba3TensorMapping.MimoX(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 9, center: 1.0f / MimoRank, jitter: 0.05f);
            AddRand(b, Mamba3TensorMapping.MimoZ(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 10, center: 1.0f, jitter: 0.05f);
            AddRand(b, Mamba3TensorMapping.MimoO(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 11, center: 1.0f / MimoRank, jitter: 0.05f);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            if (jitter > 0f)
                values[i] = center + jitter * cos;
            else
                values[i] = amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
