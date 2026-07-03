using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Single-device parity proof for the Vulkan pipeline-parallel (layer-spanning) primitives (#366):
/// a model split into stage-0 (layers <c>[0..K)</c>) + stage-1 (layers <c>[K..L)</c> + final norm + LM
/// head) must produce logits identical to the un-split full model. Stage-1 holds a windowed device-weight
/// upload (<c>VulkanWeights.Upload firstLayer=K</c>) and resumes from the hidden state handed off by
/// stage-0 (<c>DownloadHiddenState</c> → <c>ForwardFromHidden</c>), with its CPU-side per-layer lookups
/// offset by <c>_firstLayer</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Single-device on purpose.</b> This isolates the hard correctness deltas — windowed weights, the
/// <c>_firstLayer</c> offset (cpu weights + per-layer sliding window), and resume-from-hidden — WITHOUT
/// the cross-device handoff, so it runs on any single Vulkan GPU (incl. the Strix Halo iGPU). The actual
/// two-device run (prefill on GPU0 → host → decode on GPU1) is <c>VulkanCrossDevicePipelineParityTests</c>
/// on the Framework box.
/// </para>
/// <para>
/// <b>Tight tolerance.</b> Both the full model and both stages run the identical Vulkan FP32 kernels; the
/// only difference is the hidden state round-tripping through host FP32 between layers K-1 and K, which is
/// lossless. So logits should match to near-bit-exact — a tight band (2e-3 abs / 1e-2 rel) discriminates a
/// real off-by-layer bug (which shifts logits by ~0.5+) from FP32 dispatch-order noise.
/// </para>
/// <para>
/// <b>Split parameterisation</b> {1, 2, 3} over a 4-layer fixture mirrors <c>HybridVulkanCudaParityTests</c>:
/// split=1 catches missing-offset bugs where local index 0 ≡ global 0 would mask a reindex error; split=3
/// exercises the single-stage-1-layer edge and the <c>firstLayer=3</c> device-upload window.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanPipelineParityTests
{
    private readonly ITestOutputHelper _out;
    public VulkanPipelineParityTests(ITestOutputHelper output) => _out = output;

    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 8;

    public static TheoryData<int> SplitPoints => new() { 1, 2, 3 };

    private const float AbsTol = 2e-3f;
    private const float RelTol = 1e-2f;

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineSplit_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        _out.WriteLine($"device: {device.DeviceName}; split at layer {splitAt} of {NumLayers}");

        float[] full = RunFull(device, fixture, tokenIds, positions, spvDir!);
        float[] split = RunSplitPrefill(device, device, fixture, tokenIds, positions, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"prefill/split={splitAt}");
    }

    /// <summary>
    /// Cross-device spanning: stage-0 on physical device 0, stage-1 on physical device 1, hidden state
    /// handed off device0 → host → device1. Logits must match the single-device full model. Gated on ≥2
    /// Vulkan devices (skips on single-GPU boxes); runs on the Framework RTX 3060 + Intel Arc pair.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineSplit_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"stage0 device 0: {dev0.DeviceName}; stage1 device 1: {dev1.DeviceName}; split at {splitAt}/{NumLayers}");

        float[] full = RunFull(dev0, fixture, tokenIds, positions, spvDir!);
        float[] split = RunSplitPrefill(dev0, dev1, fixture, tokenIds, positions, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"xdev-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineSplit_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        _out.WriteLine($"device: {device.DeviceName}; split at layer {splitAt} of {NumLayers} (decode)");

        float[] full = RunFullDecode(device, fixture, prefillIds, prefillPos, decodeIds, decodePos, spvDir!);
        float[] split = RunSplitDecode(device, device, fixture, prefillIds, prefillPos, decodeIds, decodePos, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"decode/split={splitAt}");
    }

    /// <summary>
    /// Cross-device decode spanning: prefill + one decode step with stage-0 on device 0 and stage-1 on
    /// device 1 (each with its own device-local KV cache). Gated on ≥2 Vulkan devices.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineSplit_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"stage0 device 0: {dev0.DeviceName}; stage1 device 1: {dev1.DeviceName}; split at {splitAt}/{NumLayers} (decode)");

        float[] full = RunFullDecode(dev0, fixture, prefillIds, prefillPos, decodeIds, decodePos, spvDir!);
        float[] split = RunSplitDecode(dev0, dev1, fixture, prefillIds, prefillPos, decodeIds, decodePos, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"xdev-decode/split={splitAt}");
    }

    // ── IModel wrapper (VulkanPipelineTransformerModel) — exercises Forward + composite KV routing ──

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineModel_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] ids = [3, 1, 4, 2];
        int[] pos = [0, 1, 2, 3];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();

        float[] full = RunFull(device, fixture, ids, pos, spvDir!);
        float[] pipe = RunPipelineModelPrefill(device, device, fixture, ids, pos, splitAt, spvDir!);
        AssertLogitsMatch(full, pipe, $"model-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineModel_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] preIds = [3, 1, 4, 2]; int[] prePos = [0, 1, 2, 3];
        int[] decIds = [5]; int[] decPos = [4];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();

        float[] full = RunFullDecode(device, fixture, preIds, prePos, decIds, decPos, spvDir!);
        float[] pipe = RunPipelineModelDecode(device, device, fixture, preIds, prePos, decIds, decPos, splitAt, spvDir!);
        AssertLogitsMatch(full, pipe, $"model-decode/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineModel_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] ids = [3, 1, 4, 2];
        int[] pos = [0, 1, 2, 3];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"pipeline model: stage0 {dev0.DeviceName} / stage1 {dev1.DeviceName}; split {splitAt}/{NumLayers}");

        float[] full = RunFull(dev0, fixture, ids, pos, spvDir!);
        float[] pipe = RunPipelineModelPrefill(dev0, dev1, fixture, ids, pos, splitAt, spvDir!);
        AssertLogitsMatch(full, pipe, $"xdev-model-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineModel_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] preIds = [3, 1, 4, 2]; int[] prePos = [0, 1, 2, 3];
        int[] decIds = [5]; int[] decPos = [4];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"pipeline model decode: stage0 {dev0.DeviceName} / stage1 {dev1.DeviceName}; split {splitAt}/{NumLayers}");

        float[] full = RunFullDecode(dev0, fixture, preIds, prePos, decIds, decPos, spvDir!);
        float[] pipe = RunPipelineModelDecode(dev0, dev1, fixture, preIds, prePos, decIds, decPos, splitAt, spvDir!);
        AssertLogitsMatch(full, pipe, $"xdev-model-decode/split={splitAt}");
    }

    // ── VRAM trim (#368): stage roles must actually shed the embed table / LM head ──

    /// <summary>
    /// Discriminating check for the pipeline VRAM trim: a headless stage (no final norm + LM head)
    /// and an embed-less stage (no token-embedding table) must allocate strictly less device weight
    /// memory than the same layer window built untrimmed. Guards against a silent regression that
    /// re-uploads the stubbed tensors (parity tests can't see that — the trim is invisible to logits).
    /// </summary>
    [SkippableFact]
    public void TrimmedPipelineStages_AllocateLessDeviceMemory()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        const int split = 2;
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        var cfg0 = fixture.Config with { NumLayers = split };
        var cfg1 = fixture.Config with { NumLayers = NumLayers - split };

        using var stage0Full = VulkanTransformerModel.BuildFromPrebuiltWeights(device, cfg0, fixture.Weights, spvDir!, firstLayer: 0);
        using var stage0Trim = VulkanTransformerModel.BuildFromPrebuiltWeights(device, cfg0, fixture.Weights, spvDir!, firstLayer: 0, headless: true);
        using var stage1Full = VulkanTransformerModel.BuildFromPrebuiltWeights(device, cfg1, fixture.Weights, spvDir!, firstLayer: split);
        using var stage1Trim = VulkanTransformerModel.BuildFromPrebuiltWeights(device, cfg1, fixture.Weights, spvDir!, firstLayer: split, skipTokenEmbed: true);

        _out.WriteLine($"stage0 full={stage0Full.ComputeMemoryBytes} trimmed={stage0Trim.ComputeMemoryBytes}");
        _out.WriteLine($"stage1 full={stage1Full.ComputeMemoryBytes} trimmed={stage1Trim.ComputeMemoryBytes}");
        Assert.True(stage0Trim.ComputeMemoryBytes < stage0Full.ComputeMemoryBytes,
            "headless stage0 must shed the final-norm + LM-head weight bytes");
        Assert.True(stage1Trim.ComputeMemoryBytes < stage1Full.ComputeMemoryBytes,
            "embed-less stage1 must shed the token-embedding table bytes");
    }

    /// <summary>An embed-less stage entered via Forward (no hidden seed) must fail loudly, not gather from the stub.</summary>
    [SkippableFact]
    public void EmbedlessStage_ForwardWithoutSeed_Throws()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        var cfg1 = fixture.Config with { NumLayers = 2 };
        using var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(
            device, cfg1, fixture.Weights, spvDir!, firstLayer: 2, skipTokenEmbed: true);

        Assert.Throws<InvalidOperationException>(() =>
        {
            using var _ = stage1.Forward([1, 2], [0, 1], deviceId: 0, kvCache: null);
        });
    }

    // ── Micro-batch overlap: pipelined ForwardBatch must equal per-sequence serial Forward ──

    [SkippableFact]
    public void PipelinedForwardBatch_MatchesPerSequenceForward()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        RunPipelinedBatchParity(device, device, fixture, spvDir!, "single-device");
    }

    [SkippableFact]
    public void CrossDevicePipelinedForwardBatch_MatchesPerSequenceForward()
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"pipelined batch: stage0 {dev0.DeviceName} / stage1 {dev1.DeviceName}");
        RunPipelinedBatchParity(dev0, dev1, fixture, spvDir!, "cross-device");
    }

    private void RunPipelinedBatchParity(
        VulkanDevice dev0, VulkanDevice dev1, DenseFixture fx, string spvDir, string label)
    {
        const int split = 2;
        // Three independent sequences of different lengths.
        int[][] ids = [[3, 1, 4, 2], [5, 2, 1], [7, 3, 6, 1, 2]];
        int[][] pos = [[0, 1, 2, 3], [0, 1, 2], [0, 1, 2, 3, 4]];

        using var model = VulkanPipelineTransformerModel.BuildFromPrebuiltWeights(dev0, dev1, fx.Config, fx.Weights, split, spvDir);

        // Reference: per-sequence serial Forward, each with its own fresh KV cache.
        var reference = new float[ids.Length][];
        for (int i = 0; i < ids.Length; i++)
        {
            using var kv = model.CreateKvCache(MaxSeqLen);
            using ITensor lg = model.Forward(ids[i], pos[i], deviceId: 0, kv);
            reference[i] = LastRow(lg);
        }

        // Pipelined batch over the same three sequences, each with its own composite KV cache.
        var kvs = new IKvCache[ids.Length];
        var requests = new List<SequenceForwardRequest>(ids.Length);
        for (int i = 0; i < ids.Length; i++)
        {
            kvs[i] = model.CreateKvCache(MaxSeqLen);
            requests.Add(new SequenceForwardRequest { TokenIds = ids[i], Positions = pos[i], KvCache = kvs[i] });
        }
        try
        {
            var batched = model.ForwardBatch(requests, deviceId: 0);
            Assert.Equal(ids.Length, batched.Count);
            for (int i = 0; i < ids.Length; i++)
            {
                float[] got = LastRow(batched[i]);
                float maxAbs = 0f;
                for (int c = 0; c < got.Length; c++) maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[i][c] - got[c]));
                _out.WriteLine($"[{label}] seq {i}: max|diff| vs serial = {maxAbs:E3}");
                for (int c = 0; c < got.Length; c++)
                {
                    float bar = AbsTol + RelTol * MathF.Abs(reference[i][c]);
                    Assert.True(MathF.Abs(reference[i][c] - got[c]) <= bar,
                        $"[{label}] seq {i} col {c}: serial={reference[i][c]:F6} vs batched={got[c]:F6}");
                }
                (batched[i] as IDisposable)?.Dispose();
            }
        }
        finally
        {
            foreach (var kv in kvs) kv.Dispose();
        }
    }

    private static float[] RunPipelineModelPrefill(
        VulkanDevice dev0, VulkanDevice dev1, DenseFixture fx, int[] ids, int[] pos, int splitAt, string spvDir)
    {
        using var model = VulkanPipelineTransformerModel.BuildFromPrebuiltWeights(dev0, dev1, fx.Config, fx.Weights, splitAt, spvDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: 0, kvCache: null);
        return LastRow(logits);
    }

    private static float[] RunPipelineModelDecode(
        VulkanDevice dev0, VulkanDevice dev1, DenseFixture fx,
        int[] preIds, int[] prePos, int[] decIds, int[] decPos, int splitAt, string spvDir)
    {
        using var model = VulkanPipelineTransformerModel.BuildFromPrebuiltWeights(dev0, dev1, fx.Config, fx.Weights, splitAt, spvDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        using (var _ = model.Forward(preIds, prePos, deviceId: 0, kv)) { }
        using ITensor logits = model.Forward(decIds, decPos, deviceId: 0, kv);
        return LastRow(logits);
    }

    // ── Runners ──────────────────────────────────────────────────────────────

    private static float[] RunFull(VulkanDevice device, DenseFixture fx, int[] ids, int[] pos, string spvDir)
    {
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: 0, kvCache: null);
        return LastRow(logits);
    }

    private static float[] RunSplitPrefill(
        VulkanDevice stage0Device, VulkanDevice stage1Device,
        DenseFixture fx, int[] ids, int[] pos, int splitAt, string spvDir)
    {
        int seqLen = ids.Length;

        // Stage 0: layers [0..splitAt) on stage0Device. Run the full forward, discard logits, read hidden.
        var cfg0 = fx.Config with { NumLayers = splitAt };
        using var stage0 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage0Device, cfg0, fx.Weights, spvDir, firstLayer: 0);
        using (var _ = stage0.Forward(ids, pos, deviceId: 0, kvCache: null)) { }
        using ITensor hidden = stage0.DownloadHiddenState(seqLen); // device0 → host

        // Stage 1: layers [splitAt..L) + final norm + LM head on stage1Device, resumed from the host hidden.
        var cfg1 = fx.Config with { NumLayers = NumLayers - splitAt };
        using var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage1Device, cfg1, fx.Weights, spvDir, firstLayer: splitAt);
        var hiddenSpan = new ReadOnlySpan<float>((void*)hidden.DataPointer, seqLen * HiddenSize);
        using ITensor logits = stage1.ForwardFromHidden(hiddenSpan, pos, kvCache: null); // host → device1
        return LastRow(logits);
    }

    private static float[] RunFullDecode(
        VulkanDevice device, DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos, string spvDir)
    {
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        using (var _ = model.Forward(preIds, prePos, deviceId: 0, kv)) { }
        using ITensor logits = model.Forward(decIds, decPos, deviceId: 0, kv);
        return LastRow(logits);
    }

    private static float[] RunSplitDecode(
        VulkanDevice stage0Device, VulkanDevice stage1Device,
        DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos,
        int splitAt, string spvDir)
    {
        var cfg0 = fx.Config with { NumLayers = splitAt };
        var cfg1 = fx.Config with { NumLayers = NumLayers - splitAt };
        using var stage0 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage0Device, cfg0, fx.Weights, spvDir, firstLayer: 0);
        using var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage1Device, cfg1, fx.Weights, spvDir, firstLayer: splitAt);
        using var kv0 = stage0.CreateKvCache(MaxSeqLen); // K layers, local-indexed
        using var kv1 = stage1.CreateKvCache(MaxSeqLen); // L-K layers, local-indexed

        // Prefill through both stages.
        using (var _ = stage0.Forward(preIds, prePos, deviceId: 0, kv0)) { }
        using (ITensor h0 = stage0.DownloadHiddenState(preIds.Length))
        {
            var span0 = new ReadOnlySpan<float>((void*)h0.DataPointer, preIds.Length * HiddenSize);
            using var __ = stage1.ForwardFromHidden(span0, prePos, kv1);
        }

        // Decode one token through both stages.
        using (var _ = stage0.Forward(decIds, decPos, deviceId: 0, kv0)) { }
        using ITensor h1 = stage0.DownloadHiddenState(decIds.Length);
        var span1 = new ReadOnlySpan<float>((void*)h1.DataPointer, decIds.Length * HiddenSize);
        using ITensor logits = stage1.ForwardFromHidden(span1, decPos, kv1);
        return LastRow(logits);
    }

    private static float[] LastRow(ITensor logits)
    {
        // VulkanTransformerModel returns [rows, vocab]; take the last row.
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        Assert.Equal(VocabSize, vocab);
        int offset = (rows - 1) * vocab;
        return new ReadOnlySpan<float>((void*)(logits.DataPointer + (nint)offset * sizeof(float)), vocab).ToArray();
    }

    private void AssertLogitsMatch(float[] reference, float[] split, string variant)
    {
        Assert.Equal(reference.Length, split.Length);
        float maxAbs = 0f;
        for (int c = 0; c < reference.Length; c++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[c] - split[c]));
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3} (AbsTol={AbsTol:E3})");

        for (int c = 0; c < reference.Length; c++)
        {
            float refVal = reference[c], splitVal = split[c];
            Assert.True(float.IsFinite(refVal) && float.IsFinite(splitVal),
                $"{variant} col={c}: non-finite logit (ref={refVal}, split={splitVal}).");
            float diff = MathF.Abs(refVal - splitVal);
            float bar = AbsTol + RelTol * MathF.Abs(refVal);
            Assert.True(diff <= bar,
                $"{variant} col={c}: ref={refVal:F6} vs split={splitVal:F6} (|diff|={diff:E3} > {bar:E3}).");
        }
    }

    private static string? FindSpvDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    // ── Synthetic 4-layer dense fixture (mirrors HybridVulkanCudaParityTests.DenseFixture) ──
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
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
