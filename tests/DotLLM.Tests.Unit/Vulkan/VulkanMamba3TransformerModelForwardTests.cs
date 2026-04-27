using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Vulkan Mamba-3 SISO forward path against the CPU
/// reference. Builds a synthetic tiny SISO checkpoint matching the HF Mamba-3 tensor
/// naming, loads it on both backends, runs <c>Forward</c> on the same tokens / positions,
/// and asserts the last-token logits match within an F32 tolerance.
/// </summary>
/// <remarks>
/// <para>
/// All Mamba-3 projections in the fixture are F32, so both backends consume identical
/// weight bytes. The numerical drift sources are: reduction order in the in_proj /
/// out_proj / lm_head matmuls; the SSD scan's per-(t, h) sequential update (deterministic
/// — same on both); the data-RoPE's cum_angle accumulation (deterministic); the per-token
/// softplus / sigmoid / RMSNorm host code (bit-identical to the CPU oracle). The kernel-
/// level parity tests already pin the kernels at abs 1e-4 / rel 1e-3 each; the end-to-end
/// logit drift compounds slightly through the layers + LM head, hence the looser
/// 5e-3 absolute / 1e-3 relative bar — same bar as the MLA / MoE / NemotronH end-to-end
/// parity tests in this branch.
/// </para>
/// <para>
/// Mirrors the fixture from <c>VulkanTransformerModelMlaForwardTests.WriteFixture</c>
/// — same shapes-pinning style, same deterministic golden-ratio cosine value generator.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMamba3TransformerModelForwardTests : IDisposable
{
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;              // d_inner = 16
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int NumBcHeads = 1;            // SISO; G = 1
    private const int BcDim = StateSize * NumBcHeads;
    // num_rope_angles = (state_size * rope_fraction) / 2 with rope_fraction=0.5 → 2.
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles; // 62

    // MIMO fixture dimensions — same model dims as SISO but with a rank > 1 B/C
    // expansion. Shares DInner / HeadDim / NumHeads / StateSize with SISO so the
    // recurrent state and out-projection paths exercise identical code with only
    // the rank axis changed.
    private const int MimoRank = 2;
    private const int MimoBcDim = StateSize * NumBcHeads * MimoRank;
    private const int MimoDInProj = 2 * DInner + 2 * MimoBcDim + 3 * NumHeads + NumRopeAngles;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanMamba3TransformerModelForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_Prefill_SingleLayer_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(numLayers: 1, seqLen: 4, seed: 7);
    }

    [SkippableFact]
    public void Forward_Prefill_MultiLayer_MatchesCpuReference()
    {
        AssertVulkanMatchesCpu(numLayers: 2, seqLen: 4, seed: 17);
    }

    [SkippableFact]
    public void Forward_Mimo_Prefill_SingleLayer_MatchesCpuReference()
    {
        AssertVulkanMatchesCpuMimo(numLayers: 1, seqLen: 4, seed: 23);
    }

    [SkippableFact]
    public void Forward_Mimo_Prefill_MultiLayer_MatchesCpuReference()
    {
        AssertVulkanMatchesCpuMimo(numLayers: 2, seqLen: 4, seed: 29);
    }

    [SkippableFact]
    public void Forward_Streaming_SplitChunk_MatchesMonolithic_Siso()
    {
        // Streaming-chunk parity: split a T=8 forward into 2×T=4 forwards on the
        // same persistent state and assert the last-token logits match a single
        // monolithic T=8 forward within abs 5e-3 / rel 1e-3. Exercises the
        // chunk-boundary k_state / v_state buffers added by this commit; without
        // them the second chunk would miss the canonical shifted_γ[T_prev-1]
        // term that a one-shot forward folds in at the chunk edge.
        AssertSplitMatchesMonolithicSiso(numLayers: 1, totalLen: 8, splitAt: 4, seed: 41);
    }

    [SkippableFact]
    public void Forward_Streaming_SplitChunk_MultiLayer_MatchesMonolithic_Siso()
    {
        // Same parity bar as the single-layer split test, but with two layers so
        // the boundary buffers are exercised independently per layer (each layer
        // owns its own k_state / v_state).
        AssertSplitMatchesMonolithicSiso(numLayers: 2, totalLen: 8, splitAt: 4, seed: 43);
    }

    [SkippableFact]
    public void Forward_Streaming_SplitChunk_MatchesMonolithic_Mimo()
    {
        // MIMO streaming parity. mimo_rank=2 → k_state holds [R, H, N] and the
        // boundary kernel sums across the rank axis (matches CPU oracle's
        // ExecuteMimoStreaming rank-sum form). Single-layer to keep the F32 drift
        // budget lean.
        AssertSplitMatchesMonolithicMimo(numLayers: 1, totalLen: 8, splitAt: 4, seed: 47);
    }

    [SkippableFact]
    public void Forward_Streaming_SplitChunk_MultiLayer_MatchesMonolithic_Mimo()
    {
        // MIMO streaming parity, multi-layer. mimo_rank=2 across 2 layers.
        AssertSplitMatchesMonolithicMimo(numLayers: 2, totalLen: 8, splitAt: 4, seed: 53);
    }

    private void AssertSplitMatchesMonolithicSiso(int numLayers, int totalLen, int splitAt, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        if (splitAt <= 0 || splitAt >= totalLen)
            throw new ArgumentException("splitAt must lie strictly inside (0, totalLen).", nameof(splitAt));

        string path = Path.Combine(_scratch, $"m3-stream-L{numLayers}-T{totalLen}-s{seed}.safetensors");
        WriteFixture(path, numLayers, seed);
        ModelConfig config = BuildConfig(numLayers);

        int[] tokenIds = new int[totalLen];
        int[] positions = new int[totalLen];
        for (int i = 0; i < totalLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // ── Vulkan monolithic: single Forward over the full T=8 sequence ──
        float[] vkMonoLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            vkMonoLogits = CopyLogits(logits);
        }

        // ── Vulkan split: two Forwards (T=splitAt then T=totalLen-splitAt) on the
        //    same model instance (which owns the persistent state internally —
        //    k_state / v_state thread across calls).
        float[] vkSplitLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            int firstLen = splitAt;
            int secondLen = totalLen - splitAt;
            int[] firstIds = tokenIds.AsSpan(0, firstLen).ToArray();
            int[] firstPos = positions.AsSpan(0, firstLen).ToArray();
            int[] secondIds = tokenIds.AsSpan(firstLen, secondLen).ToArray();
            int[] secondPos = positions.AsSpan(firstLen, secondLen).ToArray();

            using (ITensor _ = model.Forward(firstIds, firstPos, deviceId: -1)) { }
            using ITensor secondLogits = model.Forward(secondIds, secondPos, deviceId: -1);
            vkSplitLogits = CopyLogits(secondLogits);
        }

        AssertLogitsClose(vkMonoLogits, vkSplitLogits,
            $"SISO streaming split parity (numLayers={numLayers}, totalLen={totalLen}, splitAt={splitAt})");
    }

    private void AssertSplitMatchesMonolithicMimo(int numLayers, int totalLen, int splitAt, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        if (splitAt <= 0 || splitAt >= totalLen)
            throw new ArgumentException("splitAt must lie strictly inside (0, totalLen).", nameof(splitAt));

        string path = Path.Combine(_scratch, $"m3-mimo-stream-L{numLayers}-T{totalLen}-s{seed}.safetensors");
        WriteMimoFixture(path, numLayers, seed);
        ModelConfig config = BuildMimoConfig(numLayers);

        int[] tokenIds = new int[totalLen];
        int[] positions = new int[totalLen];
        for (int i = 0; i < totalLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // Vulkan monolithic: single Forward over T=totalLen.
        float[] vkMonoLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            vkMonoLogits = CopyLogits(logits);
        }

        // Vulkan split.
        float[] vkSplitLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            int firstLen = splitAt;
            int secondLen = totalLen - splitAt;
            int[] firstIds = tokenIds.AsSpan(0, firstLen).ToArray();
            int[] firstPos = positions.AsSpan(0, firstLen).ToArray();
            int[] secondIds = tokenIds.AsSpan(firstLen, secondLen).ToArray();
            int[] secondPos = positions.AsSpan(firstLen, secondLen).ToArray();

            using (ITensor _ = model.Forward(firstIds, firstPos, deviceId: -1)) { }
            using ITensor secondLogits = model.Forward(secondIds, secondPos, deviceId: -1);
            vkSplitLogits = CopyLogits(secondLogits);
        }

        AssertLogitsClose(vkMonoLogits, vkSplitLogits,
            $"MIMO streaming split parity (numLayers={numLayers}, totalLen={totalLen}, splitAt={splitAt})");
    }

    private static void AssertLogitsClose(float[] reference, float[] actual, string label)
    {
        // reference is the monolithic T=totalLen forward result — Vulkan returns the
        // last token's logits as [1, vocab]. actual is the second chunk's last-token
        // logits, also [1, vocab]. Both arrays are length VocabSize.
        Assert.Equal(VocabSize, reference.Length);
        Assert.Equal(VocabSize, actual.Length);
        for (int c = 0; c < VocabSize; c++)
        {
            float r = reference[c];
            float a = actual[c];
            float diff = MathF.Abs(r - a);
            float bar = AbsTol + RelTol * MathF.Abs(r);
            Assert.True(diff <= bar,
                $"{label}: col={c}: monolithic={r:F6} vs split={a:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Forward_Decode_WithStateContinuation_MatchesCpuReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int prefillLen = 4;
        const int seed = 31;

        string path = Path.Combine(_scratch, $"m3-decode-s{seed}.safetensors");
        WriteFixture(path, NumLayers, seed);
        ModelConfig config = BuildConfig(NumLayers);

        int[] prefillIds = new int[prefillLen];
        int[] prefillPositions = new int[prefillLen];
        for (int i = 0; i < prefillLen; i++) { prefillIds[i] = i % VocabSize; prefillPositions[i] = i; }
        int[] decodeIds = [(prefillLen) % VocabSize];
        int[] decodePositions = [prefillLen];

        // ── CPU oracle: prefill then decode, same persistent state instance ───
        float[] cpuDecodeLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);
            using var state = new Mamba3State(config);
            using (ITensor _ = model.Forward(prefillIds, prefillPositions, deviceId: -1, state)) { }
            using ITensor decodeLogits = model.Forward(decodeIds, decodePositions, deviceId: -1, state);
            cpuDecodeLogits = CopyLogits(decodeLogits);
        }

        // ── Vulkan under test: same prefill-then-decode sequence on one model ───
        float[] vkDecodeLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
            using (ITensor _ = model.Forward(prefillIds, prefillPositions, deviceId: -1)) { }
            using ITensor decodeLogits = model.Forward(decodeIds, decodePositions, deviceId: -1);
            Assert.Equal(1, decodeLogits.Shape[0]);
            Assert.Equal(VocabSize, decodeLogits.Shape[1]);
            vkDecodeLogits = CopyLogits(decodeLogits);
        }

        // CPU returns [seqLen, vocab] for the decode call, Vulkan returns [1, vocab].
        // The decode call is single-token so cpuDecodeLogits[0..vocab) is exactly the
        // last token's logits.
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuDecodeLogits[c];
            float vk = vkDecodeLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"decode-continuation col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private void AssertVulkanMatchesCpu(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteFixture(path, numLayers, seed);
        ModelConfig config = BuildConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
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
                $"numLayers={numLayers}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private void AssertVulkanMatchesCpuMimo(int numLayers, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string path = Path.Combine(_scratch, $"m3-mimo-L{numLayers}-T{seqLen}-s{seed}.safetensors");
        WriteMimoFixture(path, numLayers, seed);
        ModelConfig config = BuildMimoConfig(numLayers);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = VulkanMamba3TransformerModel.LoadFromSafetensors(sf, config, spvDir);
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
                $"MIMO numLayers={numLayers}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig(int numLayers)
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

    private static void WriteMimoFixture(string path, int numLayers, int seed)
    {
        // MIMO fixture: same shape pattern as the SISO fixture but with B/C biases
        // reshaped to the canonical [H, R, N] layout and the three MIMO-only per-rank
        // weights (mimo_x, mimo_z, mimo_o) populated. Centred small-amplitude cosines
        // on top of the canonical mimo_x = mimo_o = 1/R, mimo_z = 1 inits keep the
        // forward pass numerically tractable on tiny dims.
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
            // Canonical MIMO bias layout is [H, R, N].
            AddRand(b, Mamba3TensorMapping.BBias(i), [NumHeads, MimoRank, StateSize], 0.02f, s + 5);
            AddRand(b, Mamba3TensorMapping.CBias(i), [NumHeads, MimoRank, StateSize], 0.02f, s + 6);
            AddRand(b, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, s + 7);
            AddRand(b, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, s + 8);
            // Per-rank MIMO weights — small-amplitude cosines centred on the canonical
            // identity inits (mimo_x = mimo_o = 1/R, mimo_z = 1).
            AddRand(b, Mamba3TensorMapping.MimoX(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 9, center: 1.0f / MimoRank, jitter: 0.05f);
            AddRand(b, Mamba3TensorMapping.MimoZ(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 10, center: 1.0f, jitter: 0.05f);
            AddRand(b, Mamba3TensorMapping.MimoO(i), [NumHeads, MimoRank, HeadDim],
                    amplitude: 0.05f, seed: s + 11, center: 1.0f / MimoRank, jitter: 0.05f);
        }

        b.WriteTo(path);
    }

    private static void WriteFixture(string path, int numLayers, int seed)
    {
        // Mean-zero small-amplitude weights — same shape as the CPU forward test fixture
        // (Mamba3TransformerModelTests.WriteSmallWeightFixture) so the Mamba-3 recurrence
        // does not blow up under our tiny dims.
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
