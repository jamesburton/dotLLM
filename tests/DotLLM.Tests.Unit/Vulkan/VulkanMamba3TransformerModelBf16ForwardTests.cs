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
/// End-to-end parity tests for the Vulkan Mamba-3 forward path with BF16
/// projection-weight upload — Phase 8 sibling of the F16 forward tests.
/// </summary>
/// <remarks>
/// <para>
/// Tolerance: abs 1e-2 / rel 5e-3 — looser than F16 because BF16's ~7-bit
/// mantissa accumulates more drift through the per-layer norm + scan + MoE
/// matmuls than F16's ~10-bit. The bar matches the BF16 kernel-level parity
/// envelope; on this small synthetic model with 1-2 layers the per-layer
/// noise stays comfortably within it.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelBf16ForwardTests : IDisposable
{
    private const int HiddenSize = 64;
    private const int VocabSize = 16;
    private const int NumHeads = 8;
    private const int HeadDim = 8;
    private const int Expand = 1;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int NumBcHeads = 1;
    private const int BcDim = StateSize * NumBcHeads;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    private const float AbsTol = 1e-2f;
    private const float RelTol = 5e-3f;

    private readonly string _scratch;
    private readonly List<nint> _allocs = new();

    public VulkanMamba3TransformerModelBf16ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-bf16-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
        _allocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_Bf16_Prefill_SingleLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 1, seqLen: 4, seed: 7);

    [SkippableFact]
    public void Forward_Bf16_Prefill_MultiLayer_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 2, seqLen: 4, seed: 17);

    [SkippableFact]
    public void Forward_Bf16_Decode_SingleToken_Siso_MatchesCpuReference()
        => AssertVulkanMatchesCpuSiso(numLayers: 1, seqLen: 1, seed: 31);

    private void AssertVulkanMatchesCpuSiso(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-bf16-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteSisoFixture(path, numLayers, seed);
        ModelConfig config = BuildSisoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyBf16OverlayInPlace(weights);
            using var model = Mamba3TransformerModel.FromLoadedWeights(config, weights, lifetimeAnchor: sf);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            Mamba3Weights weights = Mamba3WeightLoader.Load(config, sf);
            ApplyBf16OverlayInPlace(weights);
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
                $"BF16 SISO numLayers={numLayers}, seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private unsafe void ApplyBf16OverlayInPlace(Mamba3Weights weights)
    {
        int numLayers = weights.Layers.Length;
        weights.LayerOverlays = new Mamba3LayerQuantOverlay[numLayers];
        for (int i = 0; i < numLayers; i++)
            weights.LayerOverlays[i] = new Mamba3LayerQuantOverlay();

        // ── lm_head ─────────────────────────────────────────────────
        {
            int rows = weights.LmHead.Shape[0];
            int cols = weights.LmHead.Shape[1];
            (nint bf16Ptr, nint freshF32) = QuantizeRoundTripBf16(weights.LmHead.Pointer, rows, cols);
            weights.LmHead = new Mamba3TensorHandle(
                Pointer: freshF32, Shape: weights.LmHead.Shape,
                SourceDType: weights.LmHead.SourceDType, OwnsMemory: false);
            weights.LmHeadQ8Ptr = bf16Ptr;
            weights.LmHeadQuantTypeOverlay = QuantizationType.BF16;
        }

        // ── per-layer in_proj / out_proj ────────────────────────────
        for (int i = 0; i < numLayers; i++)
        {
            var lw = weights.Layers[i];
            var ov = weights.LayerOverlays[i];

            {
                int rows = lw.InProj.Shape[0], cols = lw.InProj.Shape[1];
                (nint bf16Ptr, nint freshF32) = QuantizeRoundTripBf16(lw.InProj.Pointer, rows, cols);
                lw = lw with
                {
                    InProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.InProj.Shape,
                        SourceDType: lw.InProj.SourceDType, OwnsMemory: false),
                };
                ov.InProjQ8Ptr = bf16Ptr;
                ov.InProjQuantTypeOverlay = QuantizationType.BF16;
            }
            {
                int rows = lw.OutProj.Shape[0], cols = lw.OutProj.Shape[1];
                (nint bf16Ptr, nint freshF32) = QuantizeRoundTripBf16(lw.OutProj.Pointer, rows, cols);
                lw = lw with
                {
                    OutProj = new Mamba3TensorHandle(
                        Pointer: freshF32, Shape: lw.OutProj.Shape,
                        SourceDType: lw.OutProj.SourceDType, OwnsMemory: false),
                };
                ov.OutProjQ8Ptr = bf16Ptr;
                ov.OutProjQuantTypeOverlay = QuantizationType.BF16;
            }

            weights.Layers[i] = lw;
        }
    }

    private unsafe (nint bf16Ptr, nint roundTrippedF32Ptr) QuantizeRoundTripBf16(
        nint f32SrcPtr, int rows, int cols)
    {
        long elems = (long)rows * cols;
        float[] arr = new float[elems];
        new ReadOnlySpan<float>((void*)f32SrcPtr, checked((int)elems)).CopyTo(arr);

        byte[] bf16Bytes = F16Bf16Fixture.QuantizeRowsBf16(arr, rows, cols);
        nint bf16Ptr = (nint)NativeMemory.AlignedAlloc((nuint)bf16Bytes.Length, 64);
        _allocs.Add(bf16Ptr);
        new ReadOnlySpan<byte>(bf16Bytes).CopyTo(new Span<byte>((void*)bf16Ptr, bf16Bytes.Length));

        float[] decoded = F16Bf16Fixture.DecodeBf16(bf16Bytes, rows, cols);
        long bytes = elems * sizeof(float);
        nint freshF32 = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _allocs.Add(freshF32);
        new ReadOnlySpan<float>(decoded).CopyTo(
            new Span<float>((void*)freshF32, checked((int)elems)));

        return (bf16Ptr, freshF32);
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
            StateSize = StateSize, NumHeads = NumHeads, HeadDim = HeadDim,
            Expand = Expand, NumGroups = NumBcHeads, ChunkSize = 4,
            IsMimo = false, MimoRank = 4,
            AFloor = 1e-4f, DtInitFloor = 1e-4f, DtMin = 1e-3f, DtMax = 0.1f,
            UseL2Warp = false, RopeFraction = 0.5f, IsOutProjNorm = false,
            RescalePrenormResidual = true, ResidualInFp32 = true,
        };
        return new ModelConfig
        {
            Architecture = Architecture.Mamba3, VocabSize = VocabSize, HiddenSize = HiddenSize,
            IntermediateSize = 0, NumLayers = numLayers,
            NumAttentionHeads = NumHeads, NumKvHeads = NumHeads, HeadDim = HeadDim,
            MaxSequenceLength = 32, AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None, RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU, NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f, TiedEmbeddings = false, SlidingWindowSize = null,
            MlaConfig = null, HybridLayout = null, SsmConfig = null,
            Mamba3Config = m3, ChatTemplate = null,
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
        long n = 1; for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
