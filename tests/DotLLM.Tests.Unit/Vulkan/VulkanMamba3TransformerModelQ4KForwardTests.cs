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
/// End-to-end parity tests for the Vulkan Mamba-3 forward path with Q4_K_M
/// projection-weight upload — Phase 1 of the K-quant work. Mirrors the Q8_0 sibling
/// class step-for-step, swapping the overlay quant type and bumping the contraction
/// axes to multiples of 256 (the Q4_K super-block size). This proves the full
/// upload-path + RecordMatmul dispatch pipeline for Q4_K is wired through, on top of
/// the kernel-level parity already covered by
/// <see cref="VulkanMatMulQ4KGemvF32KernelTests"/> /
/// <see cref="VulkanMatMulQ4KGemmF32KernelTests"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> The three matmul-target projections (<c>in_proj</c>, <c>out_proj</c>,
/// <c>lm_head</c>) run through the Q4_K path. Token embedding always lands as F32
/// (gather is byte-offset <c>vkCmdCopyBuffer</c>); norms, biases, D, dt_bias stay F32
/// (production GGUFs do not quantise these).
/// </para>
/// <para>
/// <b>Method.</b> The fixture is generated F32 and loaded via
/// <see cref="Mamba3WeightLoader"/>. For each matmul-target projection the F32 row
/// data is quantised to Q4_K via <see cref="Q4KFixture.QuantizeRows"/> and the raw
/// 144-byte super-block bytes are pinned to the overlay slots
/// (<see cref="Mamba3Weights.LmHeadQ8Ptr"/>,
/// <see cref="Mamba3LayerQuantOverlay.InProjQ8Ptr"/>,
/// <see cref="Mamba3LayerQuantOverlay.OutProjQ8Ptr"/>) — the slot names are historical
/// (Q8_0 was first); they actually carry raw bytes for whichever format the companion
/// <c>*QuantTypeOverlay</c> field declares. The corresponding F32 pointer slots are
/// replaced with freshly-allocated buffers holding dequantised-from-Q4_K values, so
/// the CPU oracle reads the same lossy values the Vulkan path consumes via the Q4_K
/// kernels.
/// </para>
/// <para>
/// <b>Dimensions.</b> All Q4_K contraction axes (<c>HiddenSize</c> for <c>in_proj</c>
/// and <c>lm_head</c>; <c>DInner = NumHeads * HeadDim</c> for <c>out_proj</c>) are
/// bumped to multiples of 256 — a hard requirement of the Vulkan Q4_K matmul kernels.
/// Tolerance abs 5e-3 / rel 1e-3, matching the Q8_0 sibling. Q4_K's 4.5 bits/element
/// makes per-element drift larger than Q8_0, but with a small synthetic vocab and
/// only 1-2 layers the accumulated drift stays comfortably under the bar.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelQ4KForwardTests : IDisposable
{
    // Q4_K-friendly dimensions: HiddenSize must be a multiple of 256 (in_proj / lm_head
    // contract along hidden); DInner = NumHeads * HeadDim must also be a multiple of
    // 256 (out_proj contracts along d_inner). Other dims unconstrained.
    private const int HiddenSize = 256;
    private const int VocabSize = 16;
    private const int NumHeads = 8;
    private const int HeadDim = 32;            // d_inner = 256
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int NumBcHeads = 1;          // SISO; G = 1
    private const int BcDim = StateSize * NumBcHeads;
    // num_rope_angles = (state_size * rope_fraction) / 2 with rope_fraction=0.5 → 2.
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;
    // Native allocations made by the Q4_K overlay (raw super-block bytes + dequantised-
    // back-to-F32 mirror buffers). Freed in Dispose so memory survives the test methods.
    private readonly List<nint> _q4kAllocs = new();

    public VulkanMamba3TransformerModelQ4KForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-q4k-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (var p in _q4kAllocs)
            NativeMemory.AlignedFree((void*)p);
        _q4kAllocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_Q4K_Prefill_SingleLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 1, seqLen: 4, seed: 7);

    [SkippableFact]
    public void Forward_Q4K_Prefill_MultiLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 2, seqLen: 4, seed: 17);

    [SkippableFact]
    public void Forward_Q4K_Decode_SingleToken_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 1, seqLen: 1, seed: 31);

    private void AssertVulkanMatchesCpuSiso(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-q4k-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteSisoFixture(path, numLayers, seed);
        ModelConfig config = BuildSisoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ4KOverlayInPlace(weights);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyQ4KOverlayInPlace(weights);
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
                $"Q4_K SISO numLayers={numLayers}, seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// Quantises every matmul-target projection (<c>lm_head</c> and per-layer
    /// <c>in_proj</c>/<c>out_proj</c>) to Q4_K via the test fixture quantiser, attaches
    /// the raw bytes to the overlay slots on <paramref name="weights"/>, and replaces
    /// the F32 source pointers with freshly-allocated buffers carrying dequantised-from-
    /// Q4_K values so the CPU oracle consumes the same effective values as the Vulkan
    /// path. Same approach as the Q8_0 sibling
    /// (<see cref="VulkanMamba3TransformerModelQ8_0ForwardTests"/>).
    /// </summary>
    private unsafe void ApplyQ4KOverlayInPlace(Mamba3Weights weights)
    {
        int numLayers = weights.Layers.Length;

        // Allocate one overlay slot per layer.
        weights.LayerOverlays = new Mamba3LayerQuantOverlay[numLayers];
        for (int i = 0; i < numLayers; i++)
            weights.LayerOverlays[i] = new Mamba3LayerQuantOverlay();

        // ── lm_head ([vocab, hidden]) ─────────────────────────────────────────
        {
            int vocab = weights.LmHead.Shape[0];
            int hidden = weights.LmHead.Shape[1];
            long elems = (long)vocab * hidden;
            ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)weights.LmHead.Pointer, checked((int)elems));
            nint q4kPtr = QuantizeF32ToQ4K(srcF32, vocab, hidden);
            nint freshF32 = AllocAlignedF32(elems);
            DequantizeQ4KRowsToF32Span(q4kPtr, vocab, hidden,
                new Span<float>((void*)freshF32, checked((int)elems)));

            weights.LmHead = new Mamba3TensorHandle(
                Pointer: freshF32,
                Shape: weights.LmHead.Shape,
                SourceDType: weights.LmHead.SourceDType,
                OwnsMemory: false);
            weights.LmHeadQ8Ptr = q4kPtr;
            weights.LmHeadQuantTypeOverlay = QuantizationType.Q4_K;
        }

        // ── per-layer in_proj / out_proj ─────────────────────────────────────
        for (int i = 0; i < numLayers; i++)
        {
            var lw = weights.Layers[i];
            var ov = weights.LayerOverlays[i];

            // in_proj [d_in_proj, hidden]  — contraction axis = hidden
            {
                int rows = lw.InProj.Shape[0];
                int cols = lw.InProj.Shape[1];
                long elems = (long)rows * cols;
                ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)lw.InProj.Pointer, checked((int)elems));
                nint q4kPtr = QuantizeF32ToQ4K(srcF32, rows, cols);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeQ4KRowsToF32Span(q4kPtr, rows, cols,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    InProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.InProj.Shape,
                        SourceDType: lw.InProj.SourceDType, OwnsMemory: false),
                };
                ov.InProjQ8Ptr = q4kPtr;
                ov.InProjQuantTypeOverlay = QuantizationType.Q4_K;
            }

            // out_proj [hidden, d_inner]  — contraction axis = d_inner
            {
                int rows = lw.OutProj.Shape[0];
                int cols = lw.OutProj.Shape[1];
                long elems = (long)rows * cols;
                ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)lw.OutProj.Pointer, checked((int)elems));
                nint q4kPtr = QuantizeF32ToQ4K(srcF32, rows, cols);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeQ4KRowsToF32Span(q4kPtr, rows, cols,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    OutProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.OutProj.Shape,
                        SourceDType: lw.OutProj.SourceDType, OwnsMemory: false),
                };
                ov.OutProjQ8Ptr = q4kPtr;
                ov.OutProjQuantTypeOverlay = QuantizationType.Q4_K;
            }

            weights.Layers[i] = lw;
        }
    }

    private unsafe nint AllocAlignedF32(long elems)
    {
        long bytes = elems * sizeof(float);
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _q4kAllocs.Add(ptr);
        return ptr;
    }

    /// <summary>
    /// Quantises a [<paramref name="rows"/>, <paramref name="cols"/>] row-major F32
    /// matrix to Q4_K row-stride bytes via <see cref="Q4KFixture.QuantizeRows"/>.
    /// Returns an aligned native allocation that the test owns (freed in Dispose).
    /// <paramref name="cols"/> must be a multiple of the Q4_K super-block size (256).
    /// </summary>
    private unsafe nint QuantizeF32ToQ4K(ReadOnlySpan<float> src, int rows, int cols)
    {
        if ((cols % 256) != 0)
            throw new InvalidOperationException(
                $"Q4_K quantisation requires cols % 256 == 0 (got cols={cols}). Bump fixture dims.");
        // Materialise the source as a managed array — the fixture quantiser takes one.
        float[] arr = new float[(long)rows * cols];
        src.CopyTo(arr);
        byte[] q4kBytes = Q4KFixture.QuantizeRows(arr, rows, cols);

        long totalBytes = q4kBytes.Length;
        nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        _q4kAllocs.Add(dst);
        new ReadOnlySpan<byte>(q4kBytes).CopyTo(new Span<byte>((void*)dst, checked((int)totalBytes)));
        return dst;
    }

    /// <summary>
    /// Dequantises Q4_K row-stride bytes back into F32 row-major elements at
    /// <paramref name="dst"/>. Used so the CPU oracle consumes the (lossy) dequantised
    /// values matching what the Vulkan side computes via the Q4_K kernels.
    /// </summary>
    private static unsafe void DequantizeQ4KRowsToF32Span(nint q4kPtr, int rows, int cols, Span<float> dst)
    {
        long rowBytes = Dequantize.RowByteSize(cols, QuantizationType.Q4_K);
        for (int row = 0; row < rows; row++)
        {
            nint rowSrc = q4kPtr + (nint)((long)row * rowBytes);
            Dequantize.ToFloat32(rowSrc, cols, QuantizationType.Q4_K, dst.Slice(row * cols, cols));
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
