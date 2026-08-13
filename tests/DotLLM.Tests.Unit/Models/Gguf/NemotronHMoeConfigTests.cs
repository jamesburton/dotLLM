using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #375 slice 1 — <c>nemotron_h_moe</c> (Nemotron 3.5 Lightning) recognition
/// and hybrid/MTP layout parsing. Metadata shapes mirror the real GGUF header
/// (read directly from two publishers' files in the 2026-08-13 research pass):
/// per-layer <c>head_count_kv</c>/<c>feed_forward_length</c> arrays over
/// block_count = trunk + <c>nextn_predict_layers</c>, where the trailing MTP
/// layer carries BOTH a KV-head count and a feed-forward length — previously a
/// hard <c>InvalidDataException</c> from the exclusive-kinds rule.
/// </summary>
public class NemotronHMoeConfigTests
{
    private static GgufMetadata BuildMetadata(Action<GgufTestData> configure)
    {
        var data = new GgufTestData(version: 3);
        configure(data);
        byte[] bytes = data.Build();

        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        var raw = GgufReader.ReadMetadata(reader, header);
        return new GgufMetadata(raw);
    }

    /// <summary>Scaled-down Nemotron-3.5-Lightning-shaped metadata: 6 trunk layers
    /// (2×SSM, 1×Attention, 3×MoE-FFN) + 1 trailing MTP layer (attention+MoE).</summary>
    private static GgufMetadata BuildNemotronHMoeMetadata(
        int[]? headCountKv = null, int[]? feedForwardLength = null, uint nextn = 1,
        uint? blockCount = null)
    {
        const string arch = "nemotron_h_moe";
        // layer:            0    1     2    3     4    5    | 6 (MTP)
        headCountKv     ??= [0,   2,    0,   0,    0,   0,     2];
        feedForwardLength ??= [0, 0,    64,  0,    64,  64,    64];
        return BuildMetadata(d =>
        {
            d.AddString("general.architecture", arch);
            d.AddUInt32($"{arch}.embedding_length", 32);
            d.AddUInt32($"{arch}.block_count", blockCount ?? (uint)headCountKv!.Length);
            d.AddUInt32($"{arch}.nextn_predict_layers", nextn);
            d.AddUInt32($"{arch}.attention.head_count", 4);
            d.AddInt32Array($"{arch}.attention.head_count_kv", headCountKv);
            d.AddInt32Array($"{arch}.feed_forward_length", feedForwardLength);
            d.AddUInt32($"{arch}.attention.key_length", 8);
            d.AddUInt32($"{arch}.context_length", 1024);
            d.AddUInt32($"{arch}.vocab_size", 64);
            d.AddFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f);
            // rope.* keys are converter artifacts (ignored since #372) but the real
            // file carries them — include them so the test proves they are harmless.
            d.AddFloat32($"{arch}.rope.freq_base", 10000f);
            d.AddUInt32($"{arch}.rope.dimension_count", 8);
            // Mamba2 SSM config (real file: conv 4, state 128, groups 8, heads 128).
            d.AddUInt32($"{arch}.ssm.conv_kernel", 4);
            d.AddUInt32($"{arch}.ssm.state_size", 16);
            d.AddUInt32($"{arch}.ssm.group_count", 2);
            d.AddUInt32($"{arch}.ssm.time_step_rank", 4);
            d.AddUInt32($"{arch}.ssm.inner_size", 64);
            // MoE keys present in the real header (full extraction is a later slice;
            // they must at minimum not break this one).
            d.AddUInt32($"{arch}.expert_count", 128);
            d.AddUInt32($"{arch}.expert_used_count", 6);
        });
    }

    [Fact]
    public void Extract_NemotronHMoe_ParsesArchTrunkAndMtpLayout()
    {
        var config = GgufModelConfigExtractor.Extract(BuildNemotronHMoeMetadata());

        Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
        // block_count 7 = 6 trunk + 1 MTP; NumLayers is the TRUNK count (the MTP
        // head layer is not executed in standard decode — llama.cpp skips it too).
        Assert.Equal(6, config.NumLayers);
        Assert.Equal(1, config.NextnPredictLayers);
        Assert.Equal(ActivationFunction.ReluSquared, config.ActivationFunction);
        Assert.NotNull(config.SsmConfig);

        Assert.NotNull(config.HybridLayout);
        Assert.Equal(6, config.HybridLayout!.LayerKind.Length);
        Assert.Equal(
            new[]
            {
                HybridLayerKind.Ssm, HybridLayerKind.Attention, HybridLayerKind.Ffn,
                HybridLayerKind.Ssm, HybridLayerKind.Ffn, HybridLayerKind.Ffn,
            },
            config.HybridLayout.LayerKind);
    }

    /// <summary>
    /// The exclusive-kinds rule still holds for TRUNK layers: a non-trailing layer
    /// with both keys set is corrupt metadata, not an MTP head. Discriminates
    /// against "just allow (true,true) anywhere", which would silently
    /// misclassify corrupt trunks.
    /// </summary>
    [Fact]
    public void Extract_BothKindsOnTrunkLayer_StillThrows()
    {
        var metadata = BuildNemotronHMoeMetadata(
            headCountKv: [0, 2, 0, 0, 0, 0, 2],
            feedForwardLength: [0, 64, 64, 0, 64, 64, 64]); // layer 1 = attn AND ffn

        var ex = Assert.Throws<InvalidDataException>(
            () => GgufModelConfigExtractor.Extract(metadata));
        Assert.Contains("Layer 1", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Extract_ArrayLengthNotCoveringMtpLayers_Throws()
    {
        // Arrays sized to the trunk only (6) while block_count still says 7 —
        // the parser must reject the mismatch, not silently mis-trim.
        var metadata = BuildNemotronHMoeMetadata(
            headCountKv: [0, 2, 0, 0, 0, 0],
            feedForwardLength: [0, 0, 64, 0, 64, 64],
            blockCount: 7); // arrays cover the trunk only — must throw, not mis-trim

        Assert.Throws<InvalidDataException>(() => GgufModelConfigExtractor.Extract(metadata));
    }

    /// <summary>
    /// #329's shared choke point must reject nemotron_h_moe on every backend's
    /// dense loader — otherwise the failure is a misleading
    /// "blk.0.attn_output.weight not present" KeyNotFoundException.
    /// </summary>
    [Fact]
    public void DenseLoaderGuard_RejectsNemotronHMoe_Accurately()
    {
        var config = GgufModelConfigExtractor.Extract(BuildNemotronHMoeMetadata());
        var ex = Assert.Throws<NotSupportedException>(
            () => TransformerWeights.ThrowIfArchitectureNeedsDedicatedLoader(config));
        Assert.Contains("NemotronHMoe", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ssm_in", ex.Message, StringComparison.Ordinal);
    }
}
