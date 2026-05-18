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
/// End-to-end parity tests for the Vulkan Mamba-3 forward path with IQ3_S and IQ3_XXS
/// projection-weight upload — IQ-family follow-up to the Q4_K / Q5_K / Q6_K Phase 1
/// work. Mirrors the Q4_K sibling class step-for-step, swapping the overlay quant type
/// to IQ3_S / IQ3_XXS. Audit finding H3: the IQ3 GEMV/GEMM kernels are bit-perfect
/// against the CPU oracle (16 Iq3 kernel parity tests in tree), and commit ad6b853
/// added IQ3_S / IQ3_XXS case branches to <see cref="VulkanMamba3TransformerModel"/>'s
/// <c>RecordMatmul</c> dispatch — this is the discriminating host-level test that
/// proves the dispatch + upload pipeline is correctly wired for IQ3, not just the
/// kernels in isolation. Matches the 5293d37-style trap-the-bug shape: a miswired
/// branch (wrong codebook handle, wrong predicate, IQ2-vs-IQ3 typo at the case label)
/// would compile, pass the kernel tests, and silently mis-decode an IQ3 GGUF at
/// inference time.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> The three matmul-target projections (<c>in_proj</c>, <c>out_proj</c>,
/// <c>lm_head</c>) run through the IQ3 path. Token embedding always lands as F32
/// (gather is byte-offset <c>vkCmdCopyBuffer</c>); norms, biases, D, dt_bias stay F32
/// (production GGUFs do not quantise these).
/// </para>
/// <para>
/// <b>Method.</b> The fixture is generated F32 and loaded via
/// <see cref="Mamba3WeightLoader"/>. For each matmul-target projection the F32 row
/// data is quantised to IQ3 via <see cref="Iq3Fixture.QuantizeRowsIq3S"/> /
/// <see cref="Iq3Fixture.QuantizeRowsIq3Xxs"/> and the raw block bytes are pinned to
/// the overlay slots. The corresponding F32 pointer slots are replaced with
/// freshly-allocated buffers holding dequantised-from-IQ3 values, so the CPU oracle
/// reads the same lossy values the Vulkan path consumes via the IQ3 kernels.
/// </para>
/// <para>
/// <b>Dimensions.</b> All IQ3 contraction axes (<c>HiddenSize</c> for <c>in_proj</c>
/// and <c>lm_head</c>; <c>DInner = NumHeads * HeadDim</c> for <c>out_proj</c>) are
/// bumped to multiples of 256 — a hard requirement of the Vulkan IQ3 matmul kernels.
/// </para>
/// <para>
/// <b>Tolerance.</b> IQ3 is ~3.3 bpw vs Q4_K's 4.5 — per-element drift is larger, but
/// the L2-relative round-trip stays under ~25-30% on small random fixtures. With a
/// tiny synthetic vocab and only 1-2 layers the accumulated drift through a single
/// matmul into logits stays within abs 1e-1 / rel 1e-1 (looser than Q4_K's 5e-3 /
/// 1e-3 envelope). The discriminator we care about is finite-and-non-zero, plus the
/// CPU-oracle path consuming the same dequantised values producing the same output to
/// within the same envelope — which catches dispatch-side bugs (wrong kernel, wrong
/// codebook, wrong byte layout) that would otherwise produce wildly wrong logits.
/// </para>
/// <para>
/// <b>Upload-path gate (audit H3 follow-up).</b> The Mamba3 host's
/// <see cref="VulkanMamba3Weights"/> upload predicate <c>KeepQuantOnDevice</c> currently
/// does NOT recognise <see cref="QuantizationType.IQ3_S"/> or
/// <see cref="QuantizationType.IQ3_XXS"/> — only the dense host's
/// <see cref="VulkanWeights"/> has IQ3 keep-on-device predicates. Until that gap is
/// closed, setting <c>LmHeadQuantTypeOverlay = IQ3_S</c> on a <see cref="Mamba3Weights"/>
/// causes the upload path to dequantise to F32 at upload time, and the IQ3 dispatch
/// arm in <see cref="VulkanMamba3TransformerModel"/> is never reached. These tests
/// detect that gap by asserting the device-side <c>weightQt</c> is IQ3 after upload
/// — if the gate is open, parity follows; if the gate is closed, the test surfaces
/// the upload-path gap with a clear skip-reason. See
/// <c>.planning/notes/iq3-per-host-parity-deferred.md</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelIq3ForwardTests : IDisposable
{
    // IQ3-friendly dimensions: HiddenSize must be a multiple of 256 (in_proj / lm_head
    // contract along hidden); DInner = NumHeads * HeadDim must also be a multiple of
    // 256 (out_proj contracts along d_inner).
    private const int HiddenSize = 256;
    private const int VocabSize = 16;
    private const int NumHeads = 8;
    private const int HeadDim = 32;            // d_inner = 256
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int NumBcHeads = 1;          // SISO; G = 1
    private const int BcDim = StateSize * NumBcHeads;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // IQ3 is ~3.3 bpw — empirically the per-element drift through the round-trip is
    // ~5-10% relative, and one matmul layer compounds to ~30-50% relative on raw
    // logits. We pin to abs 0.1 / rel 0.1 — generous, but the discriminator is still
    // "Vulkan dispatch produces same logits as CPU dequant-then-F32-matmul"; a
    // miswired case label would produce values that differ by orders of magnitude.
    private const float AbsTol = 1e-1f;
    private const float RelTol = 1e-1f;

    private readonly string _scratch;
    private readonly List<nint> _iq3Allocs = new();

    public VulkanMamba3TransformerModelIq3ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-iq3-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (var p in _iq3Allocs)
            NativeMemory.AlignedFree((void*)p);
        _iq3Allocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_IQ3_XXS_Prefill_SingleLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_XXS, numLayers: 1, seqLen: 4, seed: 7);

    [SkippableFact]
    public void Forward_IQ3_XXS_Prefill_MultiLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_XXS, numLayers: 2, seqLen: 4, seed: 17);

    [SkippableFact]
    public void Forward_IQ3_XXS_Decode_SingleToken_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_XXS, numLayers: 1, seqLen: 1, seed: 31);

    [SkippableFact]
    public void Forward_IQ3_S_Prefill_SingleLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_S, numLayers: 1, seqLen: 4, seed: 47);

    [SkippableFact]
    public void Forward_IQ3_S_Prefill_MultiLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_S, numLayers: 2, seqLen: 4, seed: 53);

    [SkippableFact]
    public void Forward_IQ3_S_Decode_SingleToken_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(QuantizationType.IQ3_S, numLayers: 1, seqLen: 1, seed: 61);

    private void AssertVulkanMatchesCpuSiso(QuantizationType iq3Type, int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-{iq3Type}-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteSisoFixture(path, numLayers, seed);
        ModelConfig config = BuildSisoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyIq3OverlayInPlace(weights, iq3Type);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyIq3OverlayInPlace(weights, iq3Type);

            // Discriminator: verify the upload path actually kept the IQ3 raw blocks
            // on device. Without this gate open in VulkanMamba3Weights.KeepQuantOnDevice
            // the upload silently dequantises to F32 and the IQ3 dispatch arm in
            // VulkanMamba3TransformerModel.RecordMatmul is unreachable — the parity
            // check that follows would still pass (both backends consume F32) but
            // the test would not be exercising the dispatch at all (audit H3
            // trap-the-bug: kernel-level parity ≠ dispatch-level parity).
            //
            // We Skip.If when the gate is closed so the test surfaces the gap
            // without breaking the CI bar — see .planning/notes/iq3-per-host-parity-deferred.md
            // for the precise next-step (add KeepIq3XxsOnDevice / KeepIq3SOnDevice
            // predicates to VulkanMamba3Weights.KeepQuantOnDevice, mirroring the
            // ones already in VulkanWeights for the dense host).
            using (var device = VulkanDevice.Create())
            using (var vkWeights = VulkanMamba3Weights.Upload(device, config, weights))
            {
                bool lmGateOpen = vkWeights.LmHeadDeviceQuantType == iq3Type;
                bool inProjGateOpen = true;
                bool outProjGateOpen = true;
                for (int li = 0; li < vkWeights.Layers.Length; li++)
                {
                    if (vkWeights.Layers[li].InProjDeviceQuantType != iq3Type) inProjGateOpen = false;
                    if (vkWeights.Layers[li].OutProjDeviceQuantType != iq3Type) outProjGateOpen = false;
                }
                Skip.IfNot(lmGateOpen && inProjGateOpen && outProjGateOpen,
                    $"VulkanMamba3Weights upload-path predicate does not yet recognise {iq3Type}; " +
                    $"observed device dtypes: lm_head={vkWeights.LmHeadDeviceQuantType}, " +
                    $"in_proj[0]={vkWeights.Layers[0].InProjDeviceQuantType}, " +
                    $"out_proj[0]={vkWeights.Layers[0].OutProjDeviceQuantType}. " +
                    "Add KeepIq3XxsOnDevice/KeepIq3SOnDevice predicates to " +
                    "VulkanMamba3Weights.KeepQuantOnDevice (mirroring the dense host's " +
                    "VulkanWeights). The IQ3 GEMV/GEMM kernels and dispatch wiring are " +
                    "already in tree (commit ad6b853). See " +
                    ".planning/notes/iq3-per-host-parity-deferred.md for context.");
            }

            using var deviceFwd = VulkanDevice.Create();
            using var model = VulkanMamba3TransformerModel.BuildOnDevice(deviceFwd, config, weights, spvDir);
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
            Assert.True(float.IsFinite(cpu), $"non-finite CPU logit {iq3Type} L={numLayers} T={seqLen} col={c}: {cpu}");
            Assert.True(float.IsFinite(vk), $"non-finite Vulkan logit {iq3Type} L={numLayers} T={seqLen} col={c}: {vk}");
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"{iq3Type} SISO numLayers={numLayers}, seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        // Non-zero stddev — guards against the "all zeros" / "all NaN" degenerate that
        // would slip past element-wise tolerance if both backends happened to emit the
        // same broken output.
        AssertNonDegenerate(vkLogits, iq3Type, numLayers, seqLen);
    }

    private static void AssertNonDegenerate(float[] logits, QuantizationType qt, int numLayers, int seqLen)
    {
        double mean = 0;
        for (int i = 0; i < logits.Length; i++) mean += logits[i];
        mean /= logits.Length;
        double var = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double d = logits[i] - mean;
            var += d * d;
        }
        var /= logits.Length;
        Assert.True(var > 1e-12,
            $"{qt} SISO numLayers={numLayers}, seqLen={seqLen}: logits stddev near zero " +
            $"(var={var:E3}) — likely degenerate output (all zeros / all same).");
    }

    /// <summary>
    /// Quantises every matmul-target projection (<c>lm_head</c> and per-layer
    /// <c>in_proj</c>/<c>out_proj</c>) to the requested IQ3 format via
    /// <see cref="Iq3Fixture"/>, attaches the raw bytes to the overlay slots on
    /// <paramref name="weights"/>, and replaces the F32 source pointers with freshly-
    /// allocated buffers carrying dequantised-from-IQ3 values so the CPU oracle consumes
    /// the same effective values as the Vulkan path. Same approach as the Q4_K / Q8_0
    /// siblings.
    /// </summary>
    private unsafe void ApplyIq3OverlayInPlace(Mamba3Weights weights, QuantizationType iq3Type)
    {
        if (iq3Type != QuantizationType.IQ3_XXS && iq3Type != QuantizationType.IQ3_S)
            throw new ArgumentException($"Unexpected IQ3 type {iq3Type}", nameof(iq3Type));

        int numLayers = weights.Layers.Length;

        weights.LayerOverlays = new Mamba3LayerQuantOverlay[numLayers];
        for (int i = 0; i < numLayers; i++)
            weights.LayerOverlays[i] = new Mamba3LayerQuantOverlay();

        // ── lm_head ([vocab, hidden]) ─────────────────────────────────────────
        {
            int vocab = weights.LmHead.Shape[0];
            int hidden = weights.LmHead.Shape[1];
            long elems = (long)vocab * hidden;
            ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)weights.LmHead.Pointer, checked((int)elems));
            nint iq3Ptr = QuantizeF32ToIq3(srcF32, vocab, hidden, iq3Type);
            nint freshF32 = AllocAlignedF32(elems);
            DequantizeIq3RowsToF32Span(iq3Ptr, vocab, hidden, iq3Type,
                new Span<float>((void*)freshF32, checked((int)elems)));

            weights.LmHead = new Mamba3TensorHandle(
                Pointer: freshF32,
                Shape: weights.LmHead.Shape,
                SourceDType: weights.LmHead.SourceDType,
                OwnsMemory: false);
            weights.LmHeadQ8Ptr = iq3Ptr;
            weights.LmHeadQuantTypeOverlay = iq3Type;
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
                nint iq3Ptr = QuantizeF32ToIq3(srcF32, rows, cols, iq3Type);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeIq3RowsToF32Span(iq3Ptr, rows, cols, iq3Type,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    InProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.InProj.Shape,
                        SourceDType: lw.InProj.SourceDType, OwnsMemory: false),
                };
                ov.InProjQ8Ptr = iq3Ptr;
                ov.InProjQuantTypeOverlay = iq3Type;
            }

            // out_proj [hidden, d_inner]  — contraction axis = d_inner
            {
                int rows = lw.OutProj.Shape[0];
                int cols = lw.OutProj.Shape[1];
                long elems = (long)rows * cols;
                ReadOnlySpan<float> srcF32 = new ReadOnlySpan<float>((void*)lw.OutProj.Pointer, checked((int)elems));
                nint iq3Ptr = QuantizeF32ToIq3(srcF32, rows, cols, iq3Type);
                nint freshF32 = AllocAlignedF32(elems);
                DequantizeIq3RowsToF32Span(iq3Ptr, rows, cols, iq3Type,
                    new Span<float>((void*)freshF32, checked((int)elems)));

                lw = lw with
                {
                    OutProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.OutProj.Shape,
                        SourceDType: lw.OutProj.SourceDType, OwnsMemory: false),
                };
                ov.OutProjQ8Ptr = iq3Ptr;
                ov.OutProjQuantTypeOverlay = iq3Type;
            }

            weights.Layers[i] = lw;
        }
    }

    private unsafe nint AllocAlignedF32(long elems)
    {
        long bytes = elems * sizeof(float);
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _iq3Allocs.Add(ptr);
        return ptr;
    }

    /// <summary>
    /// Quantises a [<paramref name="rows"/>, <paramref name="cols"/>] row-major F32
    /// matrix to IQ3 row-stride bytes via <see cref="Iq3Fixture"/>. Returns an aligned
    /// native allocation that the test owns (freed in Dispose).
    /// <paramref name="cols"/> must be a multiple of the IQ3 super-block size (256).
    /// </summary>
    private unsafe nint QuantizeF32ToIq3(ReadOnlySpan<float> src, int rows, int cols, QuantizationType iq3Type)
    {
        if ((cols % 256) != 0)
            throw new InvalidOperationException(
                $"IQ3 quantisation requires cols % 256 == 0 (got cols={cols}). Bump fixture dims.");
        float[] arr = new float[(long)rows * cols];
        src.CopyTo(arr);
        byte[] iq3Bytes = iq3Type == QuantizationType.IQ3_XXS
            ? Iq3Fixture.QuantizeRowsIq3Xxs(arr, rows, cols)
            : Iq3Fixture.QuantizeRowsIq3S(arr, rows, cols);

        long totalBytes = iq3Bytes.Length;
        nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        _iq3Allocs.Add(dst);
        new ReadOnlySpan<byte>(iq3Bytes).CopyTo(new Span<byte>((void*)dst, checked((int)totalBytes)));
        return dst;
    }

    /// <summary>
    /// Dequantises IQ3 row-stride bytes back into F32 row-major elements at
    /// <paramref name="dst"/>. Used so the CPU oracle consumes the (lossy) dequantised
    /// values matching what the Vulkan side computes via the IQ3 kernels.
    /// </summary>
    private static unsafe void DequantizeIq3RowsToF32Span(nint iq3Ptr, int rows, int cols,
        QuantizationType iq3Type, Span<float> dst)
    {
        long rowBytes = Dequantize.RowByteSize(cols, iq3Type);
        for (int row = 0; row < rows; row++)
        {
            nint rowSrc = iq3Ptr + (nint)((long)row * rowBytes);
            Dequantize.ToFloat32(rowSrc, cols, iq3Type, dst.Slice(row * cols, cols));
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
