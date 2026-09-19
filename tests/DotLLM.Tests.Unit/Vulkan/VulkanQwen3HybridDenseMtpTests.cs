using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan MTP ("NextN") parity against the CPU oracle — issue #435.
/// </summary>
/// <remarks>
/// <para>
/// The CPU host (<see cref="Qwen3HybridDenseTransformerModel.ForwardMtp"/>) is the reference
/// implementation and has been exercised end-to-end since #253, so these tests assert Vulkan
/// against it rather than against hand-computed expectations. <b>Draft token IDs must match
/// exactly</b> — an argmax over the same weights is the quantity speculative decoding actually
/// consumes, and any disagreement there is a correctness bug, not numerical drift. The logits
/// themselves are compared with a tolerance: the two backends reduce the same GEMM in a different
/// order.
/// </para>
/// <para>
/// Both fixture arms are covered because they take structurally different paths: with head-local
/// <c>nextn.*</c> tensors the MTP head embeds from and projects through its own matrices, and
/// without them it falls back to the trunk's <c>token_embd</c> / <c>output</c> — the arm that, on
/// a Hadamard-folded checkpoint, has to apply the trunk's own rotations. The synthetic fixture
/// carries no fold, so this pins the fallback plumbing; the rotation arms themselves need the real
/// Bonsai 2 MTP checkpoint (see the issue report).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseMtpTests : IDisposable
{
    private const int VocabSize = SyntheticQwen35HybridDenseMtpGguf.VocabSize;

    // Loose enough for a different GEMM reduction order, tight enough that a wrong tensor,
    // a missing norm or a mis-indexed KV row fails by orders of magnitude.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanQwen3HybridDenseMtpTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-qwen35-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string WriteFixture(bool withMtp, bool mtpHasOwnHeadTensors = true, string name = "qwen35-mtp.gguf")
        => SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, name), withMtp: withMtp, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors);

    /// <summary>
    /// The capability surface <c>MtpSpeculativeDecoder</c> gates on. Before #435 the Vulkan model
    /// reported <c>SupportsMtp == false</c> for a checkpoint that plainly carries an MTP head, and
    /// <c>SupportsRecurrentStateCheckpoint == false</c> — which would have let a rejected draft
    /// silently corrupt the trunk's GDN recurrence.
    /// </summary>
    [SkippableFact]
    public void Model_WithMtpCheckpoint_ExposesMtpAndRecurrentCheckpoint()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true);

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);

        Assert.True(model.SupportsMtp);
        Assert.True(model.SupportsRecurrentStateCheckpoint);

        using IMtpState? state = model.CreateMtpState();
        var vkState = Assert.IsType<VulkanMtpState>(state);
        Assert.Equal(config.HiddenSize, vkState.HiddenSize);
        Assert.Equal(0, vkState.CurrentLength);

        object? checkpoint = model.CheckpointRecurrentState();
        Assert.IsType<VulkanGdnStateCache>(checkpoint);
        model.RestoreRecurrentState(checkpoint);
        ((IDisposable)checkpoint!).Dispose();
    }

    /// <summary>A checkpoint without an MTP head must be completely unaffected.</summary>
    [SkippableFact]
    public void Model_WithoutMtpCheckpoint_ReportsNoMtpSupport()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: false, name: "qwen35-nomtp.gguf");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);

        Assert.False(model.SupportsMtp);
        Assert.Null(model.CreateMtpState());
    }

    /// <summary>
    /// The verify phase indexes one logit row per drafted position, so a short batch must come
    /// back as <c>[seq, vocab]</c>. Vulkan previously returned <c>[1, vocab]</c> for every batch
    /// length, which would have made every verify comparison read row 0's logits.
    /// </summary>
    [SkippableFact]
    public void Forward_ShortBatch_ReturnsOneLogitRowPerPosition_MatchingCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true);

        int[] tokenIds = [1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3];

        float[] cpuLogits;
        int cpuRows;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            cpuRows = logits.Shape[0];
            cpuLogits = Copy(logits, cpuRows * VocabSize);
        }

        Assert.Equal(tokenIds.Length, cpuRows);

        float[] vkLogits;
        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = Copy(logits, tokenIds.Length * VocabSize);
        }

        AssertClose(cpuLogits, vkLogits, "all-row logits");
    }

    /// <summary>
    /// The acceptance criterion: the same draft tokens, at the same positions, from the same
    /// checkpoint. Runs the full round shape <c>MtpSpeculativeDecoder</c> uses — a trunk forward
    /// that captures the pre-final-norm hidden state, a seed from the captured row, then K
    /// autoregressive head steps chained through their own argmax.
    /// </summary>
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void ForwardMtp_DraftTokens_MatchCpuOracle(bool mtpHasOwnHeadTensors)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = WriteFixture(withMtp: true, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors,
            name: $"qwen35-mtp-own{mtpHasOwnHeadTensors}.gguf");

        int[] tokenIds = [1, 2, 3];
        int[] positions = [0, 1, 2];
        const int DraftSteps = 4;

        var (cpuTokens, cpuLogits) = RunCpuDraft(path, tokenIds, positions, DraftSteps);
        var (vkTokens, vkLogits) = RunVulkanDraft(path, spvDir, tokenIds, positions, DraftSteps);

        Assert.Equal(cpuTokens, vkTokens);
        for (int i = 0; i < DraftSteps; i++)
            AssertClose(cpuLogits[i], vkLogits[i], $"draft step {i}");
    }

    private static (int[] tokens, float[][] logits) RunCpuDraft(
        string path, int[] tokenIds, int[] positions, int draftSteps)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kv, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        var tokens = new int[draftSteps];
        var logits = new float[draftSteps][];
        int draftToken = tokenIds[^1];
        for (int i = 0; i < draftSteps; i++)
        {
            using ITensor step = model.ForwardMtp(mtpState, draftToken, positions[^1] + i);
            logits[i] = Copy(step, VocabSize);
            draftToken = ArgMax(logits[i]);
            tokens[i] = draftToken;
        }
        return (tokens, logits);
    }

    private static (int[] tokens, float[][] logits) RunVulkanDraft(
        string path, string spvDir, int[] tokenIds, int[] positions, int draftSteps)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);
        using var mtpState = model.CreateMtpState()!;

        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kv, adapter: null, mtpState)) { }
        Assert.Equal(tokenIds.Length, mtpState.CapturedRowCount);
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        var tokens = new int[draftSteps];
        var logits = new float[draftSteps][];
        int draftToken = tokenIds[^1];
        for (int i = 0; i < draftSteps; i++)
        {
            using ITensor step = model.ForwardMtp(mtpState, draftToken, positions[^1] + i);
            Assert.Equal(1, step.Shape[0]);
            Assert.Equal(VocabSize, step.Shape[1]);
            logits[i] = Copy(step, VocabSize);
            draftToken = ArgMax(logits[i]);
            tokens[i] = draftToken;
            Assert.Equal(i + 1, mtpState.CurrentLength);
        }
        return (tokens, logits);
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}: index {i} is not finite ({actual[i]}).");
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(diff <= bar,
                $"{what}: index {i}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static float[] Copy(ITensor t, int count)
    {
        var copy = new float[count];
        unsafe
        {
            new ReadOnlySpan<float>((void*)t.DataPointer, count).CopyTo(copy);
        }
        return copy;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
