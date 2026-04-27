using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
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
/// End-to-end parity tests for the Vulkan MoE forward path with Q8_0 router-gate and
/// shared-expert weight upload. Mirrors the F32 MoE forward tests but injects Q8_0
/// raw-byte overlays onto <see cref="MoeLayerWeights"/> after the safetensors load —
/// production loaders today upcast every MoE projection to F32, so the Q8_0 GEMV/GEMM
/// dispatch can only be exercised via direct overlay injection.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> The router gate (<c>mlp.gate.weight</c>) and the per-shared-expert
/// gate/up/down projections + the Qwen1.5-MoE shared-expert sigmoid gate run through
/// the Q8_0 path. The per-routed-expert <c>W1</c>/<c>W2</c>/<c>W3</c> banks stay F32 —
/// the Vulkan <c>moe_indexed_matmul_f32</c> kernel is F32-only in tree, no Q8_0 indexed
/// variant exists yet (deferred future work). This still buys the meaningful Q8_0 win
/// for any MoE layer that has a non-trivial router or shared-expert branch.
/// </para>
/// <para>
/// <b>Method.</b> The fixture is generated F32, loaded into a <see cref="TransformerWeights"/>
/// via the standard safetensors loader, then for the Q8_0-able projections the F32 row
/// data is quantised with <see cref="MatMul.QuantizeF32ToQ8_0"/>, the resulting Q8_0
/// raw bytes are stored in unmanaged memory and pinned to the
/// <see cref="MoeLayerWeights"/> overlay fields. The original F32 row data is
/// dequantised back from those Q8_0 bytes so the CPU oracle and the Vulkan path consume
/// values that match exactly (modulo F32 reduction order). Same approach as the Q8_0
/// NemotronH parity tests.
/// </para>
/// <para>
/// <b>Dimensions.</b> All Q8_0 contraction axes (<c>hiddenSize</c> for gate +
/// shared-W1/W3 + sigmoid gate; <c>sharedIntermediate</c> for shared-W2) are bumped
/// to multiples of 32 — a hard requirement of the Vulkan Q8_0 matmul kernels.
/// Tolerance abs 5e-3 / rel 1e-3, same envelope as the existing F32 MoE forward
/// parity tests; Q8_0 is essentially lossless against this bar.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeQ8_0ForwardTests : IDisposable
{
    // Q8_0-friendly dimensions: hiddenSize, sharedIntermediateSize must be multiples of
    // 32 for the matmul_q8_0 kernels (k % 32 == 0 group-size requirement). The other
    // dims (intermediateSize, vocab, head dims) are unconstrained — kept small to keep
    // the synthetic forward fast.
    private const int HiddenSize = 32;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 8;          // NumHeads * HeadDim = 32 = HiddenSize
    private const int VocabSize = 8;
    private const int IntermediateSize = 32;
    private const int SharedIntermediateSize = 32;
    private const int NumExperts = 4;
    private const int TopK = 2;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;
    // Native allocations made by the Q8_0 overlay; freed in Dispose so memory survives
    // the duration of any single test method.
    private readonly List<nint> _q8Allocs = new();

    public VulkanTransformerModelMoeQ8_0ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-q8-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (var p in _q8Allocs)
            NativeMemory.AlignedFree((void*)p);
        _q8Allocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ── Mixtral-convention (no shared expert): router gate is the only Q8_0 path ─────
    [SkippableFact]
    public void Forward_Q8_0_RouterGate_Mixtral_SingleToken_MatchesCpuReference()
        => AssertMixtralVulkanMatchesCpu(seqLen: 1, seed: 42);

    [SkippableFact]
    public void Forward_Q8_0_RouterGate_Mixtral_ThreeTokenPrefill_MatchesCpuReference()
        => AssertMixtralVulkanMatchesCpu(seqLen: 3, seed: 7);

    // ── DeepSeek-V2-style ungated shared experts: Q8_0 router + Q8_0 shared W1/W2/W3 ─
    [SkippableFact]
    public void Forward_Q8_0_RouterAndUngatedShared_SingleToken_MatchesCpuReference()
        => AssertSharedVulkanMatchesCpu(seqLen: 1, numSharedExperts: 1, seed: 101, gated: false);

    [SkippableFact]
    public void Forward_Q8_0_RouterAndUngatedShared_ThreeTokenPrefill_MatchesCpuReference()
        => AssertSharedVulkanMatchesCpu(seqLen: 3, numSharedExperts: 2, seed: 17, gated: false);

    // ── Qwen1.5-MoE sigmoid-gated single shared expert: full Q8_0 stack ──────────────
    [SkippableFact]
    public void Forward_Q8_0_RouterAndGatedShared_Qwen15_SingleToken_MatchesCpuReference()
        => AssertSharedVulkanMatchesCpu(seqLen: 1, numSharedExperts: 1, seed: 71, gated: true);

    private void AssertMixtralVulkanMatchesCpu(int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"moe-q8-mixtral-s{seqLen}.safetensors");
        WriteMixtralFixture(path, seed);

        var config = BuildMixtralConfig();
        RunParityCheck(path, config, seqLen, expectShared: false, gated: false, spvDir);
    }

    private void AssertSharedVulkanMatchesCpu(int seqLen, int numSharedExperts, int seed, bool gated)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"moe-q8-shared-s{seqLen}-n{numSharedExperts}-g{(gated ? 1 : 0)}.safetensors");
        WriteSharedFixture(path, seed, numSharedExperts, gated);

        var config = BuildSharedConfig(numSharedExperts, gated);
        RunParityCheck(path, config, seqLen, expectShared: true, gated: gated, spvDir);
    }

    private void RunParityCheck(string path, ModelConfig config, int seqLen, bool expectShared, bool gated, string spvDir)
    {
        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        // Load CPU weights once. We mutate the MoE layer to attach Q8_0 overlays AND
        // dequantise back to F32 in-place so the CPU oracle and the Vulkan path consume
        // the same effective weight values. Reusing one TransformerWeights between both
        // backends would race on the underlying mmap so we load twice (matches the
        // existing F32 MoE forward tests).
        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            var cpuWeights = TransformerWeightsSafetensorsLoader.Load(sf, config);
            ApplyQ8OverlayInPlace(cpuWeights, expectShared, gated);
            using var model = TransformerModel.BuildFromPrebuiltWeights(cpuWeights, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            var cpuWeights = TransformerWeightsSafetensorsLoader.Load(sf, config);
            ApplyQ8OverlayInPlace(cpuWeights, expectShared, gated);
            using var device = VulkanDevice.Create();
            using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, config, cpuWeights, spvDir);
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
                $"seqLen={seqLen}, shared={expectShared}, gated={gated}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// For every MoE layer in <paramref name="weights"/>, quantise the router gate (and
    /// shared-expert gate/up/down + sigmoid gate when present) to Q8_0, attach the raw
    /// bytes as an overlay on <see cref="MoeLayerWeights"/>, and replace the F32
    /// projections with freshly-allocated buffers carrying dequantised-from-Q8_0 values
    /// so the CPU oracle consumes the same effective values as the Vulkan path. The
    /// per-routed-expert <c>W1</c>/<c>W2</c>/<c>W3</c> banks stay F32 in both backends
    /// — no Q8_0 indexed-matmul kernel exists yet.
    /// </summary>
    /// <remarks>
    /// The safetensors loader hands back mmap-backed pointers for F32 source tensors
    /// (read-only mapping), so we cannot mutate <see cref="MoeLayerWeights.SharedGateProj"/>
    /// in place; instead we allocate fresh F32 buffers and reassign the per-shared-expert
    /// pointer slots. The original mmap region stays untouched. Same approach applied to
    /// the router gate (a managed <c>float[]</c> — already mutable, but we keep the same
    /// logical "fresh F32" pattern for clarity).
    /// </remarks>
    private unsafe void ApplyQ8OverlayInPlace(TransformerWeights weights, bool expectShared, bool gated)
    {
        for (int i = 0; i < weights.Layers.Length; i++)
        {
            var moe = weights.Layers[i].Moe;
            if (moe is null) continue;

            int hidden = moe.HiddenSize;
            int sharedI = moe.SharedIntermediateSize;
            int numE = moe.NumExperts;
            int numShared = moe.NumSharedExperts;

            // ── Router gate ([numExperts, hiddenSize]) ───────────────────────────
            // moe.Gate is a managed float[] — safe to mutate in place.
            moe.GateQ8Ptr = QuantizeF32MatrixToQ8(moe.Gate, numE, hidden);
            moe.GateQuantTypeOverlay = QuantizationType.Q8_0;
            DequantizeQ8RowsToF32Span(moe.GateQ8Ptr, numE, hidden, moe.Gate);

            if (!expectShared) continue;
            Assert.True(moe.HasSharedExpert, $"Layer {i}: expected shared experts but loader produced none.");

            // ── Per-shared-expert gate/up/down ──────────────────────────────────
            // For each shared expert: gate ([sharedI, hidden]), up ([sharedI, hidden]),
            // down ([hidden, sharedI]). All three contract along axes that are
            // multiples of 32 in this fixture, so all are Q8_0-able. The original
            // SharedGateProj[s] points into the safetensors mmap region (read-only);
            // we allocate fresh F32 buffers, populate them with the dequantised-from-Q8
            // values, and reassign the slots so the CPU oracle reads the lossy values.
            moe.SharedGateProjQ8Ptrs = new nint[numShared];
            moe.SharedUpProjQ8Ptrs = new nint[numShared];
            moe.SharedDownProjQ8Ptrs = new nint[numShared];
            int gateCols = hidden;
            int gateRows = sharedI;
            int downCols = sharedI;
            int downRows = hidden;
            long gateElems = (long)gateRows * gateCols;
            long downElems = (long)downRows * downCols;

            for (int s = 0; s < numShared; s++)
            {
                // gate_proj
                moe.SharedGateProjQ8Ptrs[s] = QuantizeF32SpanToQ8(
                    new ReadOnlySpan<float>((void*)moe.SharedGateProj[s], (int)gateElems), gateRows, gateCols);
                nint freshGate = AllocAlignedF32(gateElems);
                DequantizeQ8RowsToF32Span(moe.SharedGateProjQ8Ptrs[s], gateRows, gateCols,
                    new Span<float>((void*)freshGate, (int)gateElems));
                moe.SharedGateProj[s] = freshGate;

                // up_proj — same shape as gate_proj
                moe.SharedUpProjQ8Ptrs[s] = QuantizeF32SpanToQ8(
                    new ReadOnlySpan<float>((void*)moe.SharedUpProj[s], (int)gateElems), gateRows, gateCols);
                nint freshUp = AllocAlignedF32(gateElems);
                DequantizeQ8RowsToF32Span(moe.SharedUpProjQ8Ptrs[s], gateRows, gateCols,
                    new Span<float>((void*)freshUp, (int)gateElems));
                moe.SharedUpProj[s] = freshUp;

                // down_proj [hidden, sharedI]
                moe.SharedDownProjQ8Ptrs[s] = QuantizeF32SpanToQ8(
                    new ReadOnlySpan<float>((void*)moe.SharedDownProj[s], (int)downElems), downRows, downCols);
                nint freshDown = AllocAlignedF32(downElems);
                DequantizeQ8RowsToF32Span(moe.SharedDownProjQ8Ptrs[s], downRows, downCols,
                    new Span<float>((void*)freshDown, (int)downElems));
                moe.SharedDownProj[s] = freshDown;
            }
            moe.SharedExpertProjQuantTypeOverlay = QuantizationType.Q8_0;

            // ── Shared-expert sigmoid gate (Qwen1.5-MoE) ─────────────────────────
            if (gated)
            {
                Assert.NotNull(moe.SharedExpertGate);
                // SharedExpertGate is a managed float[] (loaded via ResolveNorm which
                // copies the mmap data into a managed buffer) — safe to mutate in place.
                moe.SharedExpertGateQ8Ptr = QuantizeF32MatrixToQ8(moe.SharedExpertGate!, 1, hidden);
                moe.SharedExpertGateQuantTypeOverlay = QuantizationType.Q8_0;
                DequantizeQ8RowsToF32Span(moe.SharedExpertGateQ8Ptr, 1, hidden, moe.SharedExpertGate!);
            }
        }
    }

    /// <summary>Allocates a 64-byte-aligned F32 buffer of <paramref name="elems"/>
    /// elements; freed in <see cref="Dispose"/>.</summary>
    private unsafe nint AllocAlignedF32(long elems)
    {
        long bytes = elems * sizeof(float);
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        _q8Allocs.Add(ptr);
        return ptr;
    }

    /// <summary>
    /// Quantises a [<paramref name="rows"/>, <paramref name="cols"/>] row-major F32 matrix
    /// (held in <paramref name="src"/>) to Q8_0 row-stride bytes via
    /// <see cref="MatMul.QuantizeF32ToQ8_0"/>. Returns an aligned native allocation that
    /// the test owns (freed in <see cref="Dispose"/>). <paramref name="cols"/> must be a
    /// multiple of the Q8_0 group size (32).
    /// </summary>
    private unsafe nint QuantizeF32MatrixToQ8(float[] src, int rows, int cols)
        => QuantizeF32SpanToQ8(src.AsSpan(), rows, cols);

    private unsafe nint QuantizeF32SpanToQ8(ReadOnlySpan<float> src, int rows, int cols)
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
    /// <paramref name="dst"/>. Used so the CPU oracle consumes the (lossy)
    /// dequantised values matching what the Vulkan side computes via the Q8_0 kernels.
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

    private static ModelConfig BuildMixtralConfig()
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = Architecture.Mixtral,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = IntermediateSize,
            },
            ChatTemplate = null,
        };
    }

    private static ModelConfig BuildSharedConfig(int numSharedExperts, bool gated)
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = Architecture.QwenMoe,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = IntermediateSize,
                NormTopKProb = !gated,                  // false for Qwen1.5-MoE, true for DeepSeek-style
                SharedExpertIntermediateSize = SharedIntermediateSize,
                NumSharedExperts = numSharedExperts,
                HasSharedExpertGate = gated,
                DecoderSparseStep = 1,
            },
            ChatTemplate = null,
        };
    }

    private static void WriteMixtralFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 100 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight",
                [HiddenSize, NumHeads * HeadDim], 0.05f, s + 5);

            // Mixtral MoE FFN: router gate + (w1, w2, w3) per expert.
            AddRand(b, $"{prefix}.block_sparse_moe.gate.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w1.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 10 + e * 3);
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w2.weight",
                    [HiddenSize, IntermediateSize], 0.1f, s + 11 + e * 3);
                AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w3.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 12 + e * 3);
            }
        }

        b.WriteTo(path);
    }

    private static void WriteSharedFixture(string path, int seed, int numSharedExperts, bool gated)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 100 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight",
                [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight",
                [HiddenSize, NumHeads * HeadDim], 0.05f, s + 5);

            // Qwen-MoE FFN: router gate + (gate_proj, up_proj, down_proj) per expert.
            AddRand(b, $"{prefix}.mlp.gate.weight",
                [NumExperts, HiddenSize], 0.1f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 10 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight",
                    [IntermediateSize, HiddenSize], 0.1f, s + 11 + e * 3);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight",
                    [HiddenSize, IntermediateSize], 0.1f, s + 12 + e * 3);
            }

            if (gated)
            {
                // Qwen1.5-MoE: singular shared expert + sigmoid gate.
                AddRand(b, $"{prefix}.mlp.shared_expert.gate_proj.weight",
                    [SharedIntermediateSize, HiddenSize], 0.1f, s + 50);
                AddRand(b, $"{prefix}.mlp.shared_expert.up_proj.weight",
                    [SharedIntermediateSize, HiddenSize], 0.1f, s + 51);
                AddRand(b, $"{prefix}.mlp.shared_expert.down_proj.weight",
                    [HiddenSize, SharedIntermediateSize], 0.1f, s + 52);
                AddRand(b, $"{prefix}.mlp.shared_expert_gate.weight",
                    [HiddenSize], 0.1f, s + 60);
            }
            else
            {
                // DeepSeek-V2-style indexed-plural shared experts (no sigmoid gate).
                for (int k = 0; k < numSharedExperts; k++)
                {
                    AddRand(b, $"{prefix}.mlp.shared_experts.{k}.gate_proj.weight",
                        [SharedIntermediateSize, HiddenSize], 0.1f, s + 50 + k * 3);
                    AddRand(b, $"{prefix}.mlp.shared_experts.{k}.up_proj.weight",
                        [SharedIntermediateSize, HiddenSize], 0.1f, s + 51 + k * 3);
                    AddRand(b, $"{prefix}.mlp.shared_experts.{k}.down_proj.weight",
                        [HiddenSize, SharedIntermediateSize], 0.1f, s + 52 + k * 3);
                }
            }
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
