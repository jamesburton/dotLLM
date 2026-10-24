using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #815 (child of epic #814): <c>qwen4exp</c> architecture, config round-trip through a real GGUF, the tensor
/// contract against the synthetic fixture, the metadata sanity checks llama.cpp enforces, and the explicit refusals.
/// CPU-only: no device is created anywhere in this class.
/// </summary>
public sealed class Qwen4ExpConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-" + Guid.NewGuid().ToString("N"));

    public Qwen4ExpConfigTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private (GgufFile File, ModelConfig Config) Open(bool mtp = false)
    {
        string path = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, mtp ? "mtp.gguf" : "trunk.gguf"), includeMtp: mtp);
        var file = GgufFile.Open(path);
        return (file, GgufModelConfigExtractor.Extract(file.Metadata));
    }

    /// <summary>Extracts a config from the synthetic file's metadata after a caller-supplied mutation.</summary>
    private ModelConfig ExtractMutated(Action<Dictionary<string, GgufMetadataValue>> mutate, bool mtp = false)
    {
        var (file, _) = Open(mtp);
        using (file)
        {
            var dict = new Dictionary<string, GgufMetadataValue>();
            foreach (string key in file.Metadata.Keys)
            {
                file.Metadata.TryGetValue(key, out var v);
                dict[key] = v;
            }
            mutate(dict);
            return GgufModelConfigExtractor.Extract(new GgufMetadata(dict));
        }
    }

    private static GgufMetadataValue U32(uint v) => new(GgufValueType.UInt32, v);
    private static GgufMetadataValue I32Array(params int[] v) => new(GgufValueType.Array, v);
    private static GgufMetadataValue U64Array(params ulong[] v) => new(GgufValueType.Array, v);

    // ───────────────────────────── round trip ─────────────────────────────

    [Fact]
    public void Extract_SyntheticFile_RoundTripsEveryQwen4ExpField()
    {
        var (file, c) = Open();
        using (file)
        {
            Assert.Equal(Architecture.Qwen4Exp, c.Architecture);
            Assert.Equal(SyntheticQwen4ExpGguf.TrunkLayers, c.NumLayers);
            Assert.Equal(0, c.NextnPredictLayers);
            Assert.Equal(SyntheticQwen4ExpGguf.HiddenSize, c.HiddenSize);
            Assert.Equal(SyntheticQwen4ExpGguf.VocabSize, c.VocabSize);
            Assert.Equal(SyntheticQwen4ExpGguf.NumHeads, c.NumAttentionHeads);
            Assert.Equal(SyntheticQwen4ExpGguf.NumKvHeads, c.NumKvHeads);
            Assert.Equal(SyntheticQwen4ExpGguf.HeadDim, c.HeadDim);
            Assert.Null(c.SsmConfig);   // GDN reuses the ssm.* keys; the Mamba-2 extractor must not see them
            Assert.Null(c.Gemma3n);

            // Reused hybrid plumbing: GDN geometry, 3:1 layout, softmax MoE + sigmoid-gated shared expert.
            Assert.Equal(new GatedDeltaNetConfig(
                SyntheticQwen4ExpGguf.FullAttnInterval, SyntheticQwen4ExpGguf.NVHead, SyntheticQwen4ExpGguf.NKHead,
                SyntheticQwen4ExpGguf.DState, SyntheticQwen4ExpGguf.DState * SyntheticQwen4ExpGguf.NVHead, SyntheticQwen4ExpGguf.DConv), c.GdnConfig);
            Assert.Equal(
                [HybridLayerKind.GatedDeltaNet, HybridLayerKind.GatedDeltaNet, HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention],
                c.HybridLayout!.LayerKind);
            Assert.Equal(SyntheticQwen4ExpGguf.NumExperts, c.Moe!.NumExperts);
            Assert.Equal(SyntheticQwen4ExpGguf.NumExpertsUsed, c.Moe.NumExpertsPerTok);
            Assert.Equal(SyntheticQwen4ExpGguf.MoeIntermediate, c.Moe.MoeIntermediateSize);
            Assert.Equal(SyntheticQwen4ExpGguf.SharedIntermediate, c.Moe.SharedExpertIntermediateSize);
            Assert.Equal(1, c.Moe.NumSharedExperts);
            Assert.True(c.Moe.HasSharedExpertGate);
            Assert.True(c.Moe.NormTopKProb);   // softmax over all experts, then top-k, then renormalise
            Assert.Equal(10000000.0f, c.RoPEConfig!.Value.Theta);
            Assert.Equal(SyntheticQwen4ExpGguf.RopeDim, c.RoPEConfig.Value.DimensionCount);
            Assert.Equal(RoPEType.NeoX, c.RoPEConfig.Value.Type);

            var q = c.Qwen4Exp!;
            Assert.Equal(SyntheticQwen4ExpGguf.HcCount, q.HyperConnectionCount);
            Assert.Equal(SyntheticQwen4ExpGguf.HcLowRank, q.HyperConnectionLowRank);
            Assert.Equal(SyntheticQwen4ExpGguf.IndexerHeads, q.IndexerHeadCount);
            Assert.Equal(SyntheticQwen4ExpGguf.IndexerKeyLength, q.IndexerKeyLength);
            Assert.Equal(SyntheticQwen4ExpGguf.IndexerTopK, q.IndexerTopK);
            Assert.Equal([0, 0, 0, 4], q.CompressRatios);
            Assert.Equal(4, q.IndexerBlockSize);
            Assert.Equal([3, 3, 2, 0], q.RopeSections);

            var ple = q.Ple!;
            Assert.Equal([SyntheticQwen4ExpGguf.PleLayer], ple.Layers);
            Assert.Equal(3, ple.NgramSize);
            Assert.Equal(2, ple.HeadsPerNgram);
            Assert.Equal(4, ple.NumHeads);
            Assert.Equal(4, ple.ConvKernel);
            Assert.Equal(2, ple.EosTokenId);
            Assert.Equal(3, ple.ImageTokenId);
            Assert.Equal(8, ple.RowDim);
            Assert.Equal(60UL, ple.MinTableRows);
        }
    }

    [Fact]
    public void Extract_PleHashConstants_AreExactUInt64()
    {
        // 18446744073709551557 > 2^63 and > 2^53: a signed or floating-point round trip changes it.
        var (file, c) = Open();
        using (file)
        {
            var ple = c.Qwen4Exp!.Ple!;
            Assert.Equal(SyntheticQwen4ExpGguf.PleMultipliers, ple.LayerMultipliers);
            Assert.Equal(18446744073709551557UL, ple.LayerMultipliers[2]);
            Assert.Equal(23703573157769UL, ple.LayerMultipliers[0]);
            Assert.Equal(SyntheticQwen4ExpGguf.PleHeadOffsets, ple.HeadOffsets);
            Assert.Equal(SyntheticQwen4ExpGguf.PleHeadVocabSizes, ple.HeadVocabSizes);
            // And the raw GGUF array really is uint64 (the converter writes them that way).
            Assert.Equal(SyntheticQwen4ExpGguf.PleMultipliers, file.Metadata.GetUInt64Array("qwen4exp.ple.layer_multipliers"));
        }
    }

    [Fact]
    public void Extract_MtpFile_SeparatesTheTrailingBlockFromTheTrunk()
    {
        var (file, c) = Open(mtp: true);
        using (file)
        {
            Assert.Equal(SyntheticQwen4ExpGguf.TrunkLayers, c.NumLayers);   // block_count (5) minus the MTP block
            Assert.Equal(1, c.NextnPredictLayers);
            Assert.Equal(SyntheticQwen4ExpGguf.TrunkLayers, c.HybridLayout!.LayerKind.Length);
            Assert.Equal(5, c.Qwen4Exp!.CompressRatios.Count);              // the ratio table covers the MTP block too
            Assert.Equal(4, c.Qwen4Exp.CompressRatios[4]);
        }
    }

    // ───────────────────────────── tensor contract ─────────────────────────────

    [Fact]
    public void SyntheticFile_TensorTable_MatchesTheContractExactly()
    {
        var (file, c) = Open();
        using (file)
            Assert.Empty(Qwen4ExpTensors.FindProblems(file.TensorsByName, c));
    }

    [Fact]
    public void SyntheticMtpFile_TensorTable_MatchesTheContractExactly()
    {
        var (file, c) = Open(mtp: true);
        using (file)
        {
            Assert.Empty(Qwen4ExpTensors.FindProblems(file.TensorsByName, c, includeMtp: true));
            // Without asking for the MTP block it is (correctly) reported as unknown.
            Assert.Contains(Qwen4ExpTensors.FindProblems(file.TensorsByName, c), p => p.Contains("blk.4.nextn.eh_proj.weight"));
        }
    }

    [Fact]
    public void Contract_Discriminates_MissingMisshapenAndUnknownTensors()
    {
        var (file, c) = Open();
        using (file)
        {
            var table = new Dictionary<string, GgufTensorDescriptor>(file.TensorsByName);
            table.Remove("blk.3.indexer.k_proj.weight");
            var q = table["blk.3.attn_q.weight"];
            table["blk.3.attn_q.weight"] = q with { Shape = new DotLLM.Core.Tensors.TensorShape(q.Shape[0], q.Shape[1] / 2) }; // forgot the gate half
            table["blk.0.ffn_gate_up_exps.weight"] = q;                                                                        // fused experts: not this model

            var problems = Qwen4ExpTensors.FindProblems(table, c);

            Assert.Contains(problems, p => p.Contains("missing tensor 'blk.3.indexer.k_proj.weight'"));
            Assert.Contains(problems, p => p.Contains("'blk.3.attn_q.weight' has shape"));
            Assert.Contains(problems, p => p.Contains("unexpected tensor 'blk.0.ffn_gate_up_exps.weight'"));
            Assert.Equal(3, problems.Count);
        }
    }

    [Fact]
    public void Contract_OutputIsOptional_TiedToTokenEmbedding()
    {
        var (file, c) = Open();
        using (file)
        {
            var table = new Dictionary<string, GgufTensorDescriptor>(file.TensorsByName);
            table.Remove("output.weight");
            Assert.Empty(Qwen4ExpTensors.FindProblems(table, c));
        }
    }

    [Fact]
    public void Contract_PleTable_OnlyNeedsToCoverTheHeadRanges()
    {
        var (file, c) = Open();
        using (file)
        {
            var table = new Dictionary<string, GgufTensorDescriptor>(file.TensorsByName);
            var t = table["per_layer_token_embd.weight"];
            table["per_layer_token_embd.weight"] = t with { Shape = new DotLLM.Core.Tensors.TensorShape(8, 59) };   // one row short of 60
            Assert.Single(Qwen4ExpTensors.FindProblems(table, c));
            table["per_layer_token_embd.weight"] = t with { Shape = new DotLLM.Core.Tensors.TensorShape(8, 60) };
            Assert.Empty(Qwen4ExpTensors.FindProblems(table, c));
        }
    }

    [Fact]
    public void SyntheticFile_LoadsAsASplitSet_WithTheSameTableAndConfig()
    {
        // Ties the fixture to #756: the same model as shard 1 = metadata only + shards 2..3 = tensors.
        string first = SyntheticQwen4ExpGguf.WriteSplit(_dir, "q4e", shardCount: 3);

        using var file = GgufFile.Open(first);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);

        Assert.True(file.IsSplit);
        Assert.Equal(3, file.ShardCount);
        Assert.Equal(Architecture.Qwen4Exp, cfg.Architecture);
        Assert.Empty(Qwen4ExpTensors.FindProblems(file.TensorsByName, cfg));
        Assert.Equal(0UL, file.Header.TensorCount);   // the first shard carries no tensors, like the real files
        Assert.True(file.GetTensorShardIndex("token_embd.weight") >= 1);
    }

    // ───────────────────────────── metadata sanity (llama.cpp parity) ─────────────────────────────

    [Fact]
    public void Extract_HyperConnectionCountOfOne_IsRejected()
    {
        var ex = Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["qwen4exp.hyper_connection.count"] = U32(1)));
        Assert.Contains("hyper_connection.count", ex.Message);
    }

    [Theory]
    [InlineData("qwen4exp.hyper_connection.low_rank")]
    [InlineData("qwen4exp.attention.indexer.head_count")]
    [InlineData("qwen4exp.attention.indexer.key_length")]
    [InlineData("qwen4exp.attention.indexer.top_k")]
    [InlineData("qwen4exp.ple.ngram_size")]
    [InlineData("qwen4exp.ple.conv_kernel")]
    [InlineData("qwen4exp.ple.eos_token_id")]
    [InlineData("qwen4exp.embedding_length_per_layer_input")]
    [InlineData("qwen4exp.ple.layer_multipliers")]
    [InlineData("qwen4exp.ple.head_offsets")]
    [InlineData("qwen4exp.ple.head_vocab_sizes")]
    [InlineData("qwen4exp.rope.dimension_sections")]
    public void Extract_MissingRequiredKey_IsAnExplicitError(string key)
    {
        var ex = Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d.Remove(key)));
        Assert.Contains(key.Split('.')[^1], ex.Message);
    }

    [Fact]
    public void Extract_MissingImageTokenId_FallsBackToNull()
    {
        var c = ExtractMutated(d => d.Remove("qwen4exp.ple.image_token_id"));
        Assert.Null(c.Qwen4Exp!.Ple!.ImageTokenId);
    }

    [Fact]
    public void Extract_NoPleLayers_MeansNoPleModule()
    {
        var c = ExtractMutated(d => d["qwen4exp.ple.layers"] = I32Array());
        Assert.Null(c.Qwen4Exp!.Ple);
    }

    [Fact]
    public void Extract_ShortHashArrays_AreRejected_NotZeroPadded()
    {
        var ex = Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.ple.head_offsets"] = U64Array(0, 11, 24)));   // 3 of the 4 heads
        Assert.Contains("head_offsets", ex.Message);
        Assert.Contains("at least 4", ex.Message);
    }

    [Fact]
    public void Extract_NegativeSignedHashConstants_AreRejected()
    {
        var ex = Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.ple.layer_multipliers"] = new GgufMetadataValue(GgufValueType.Array, new long[] { 1, -2, 3 })));
        Assert.Contains("negative", ex.Message);
    }

    [Fact]
    public void Extract_PleOnAnAttentionLayer_IsRejected()
    {
        var ex = Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["qwen4exp.ple.layers"] = I32Array(3)));
        Assert.Contains("not a linear-attention", ex.Message);
    }

    [Fact]
    public void Extract_MultiplePleLayers_WithASingleSetOfConstants_AreRefusedWithAnExplanation()
    {
        // Two modules need one set of hash constants each (HF derives them per ple_layer_index); a llama.cpp-style single set cannot describe them.
        var ex = Assert.Throws<NotSupportedException>(() => ExtractMutated(d => d["qwen4exp.ple.layers"] = I32Array(0, 1)));
        Assert.Contains("one set per layer", ex.Message);
    }

    [Fact]
    public void Extract_PleLayers_MustBeStrictlyAscending()
    {
        // HF numbers a module by its position in the sorted, de-duplicated list; an unsorted / repeated list would silently re-number them.
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["qwen4exp.ple.layers"] = I32Array(1, 1)));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["qwen4exp.ple.layers"] = I32Array(2, 1)));
    }

    [Fact]
    public void Extract_ZeroHeadVocabSize_IsRejected()
    {
        Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.ple.head_vocab_sizes"] = U64Array(11, 0, 17, 19)));
    }

    [Fact]
    public void Extract_CompressRatiosDisagreeingWithTheLayerInterval_AreRejected()
    {
        // Ratio says layer 1 is QSA, the interval says it is a Gated-DeltaNet layer.
        var ex = Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.attention.compress_ratios"] = I32Array(0, 4, 0, 4)));
        Assert.Contains("Block 1", ex.Message);
    }

    [Fact]
    public void Extract_UnequalCompressRatiosAcrossQsaLayers_AreRejected()
    {
        // Trunk QSA layer 3 pools by 4, the MTP block (also QSA) by 8: llama.cpp has one block size for the whole model.
        var ex = Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.attention.compress_ratios"] = I32Array(0, 0, 0, 4, 8), mtp: true));
        Assert.Contains("share one compress ratio", ex.Message);
    }

    [Theory]
    [InlineData(1)]    // ratio 1 pools nothing
    [InlineData(3)]    // 8 % 3 != 0: blocks do not tile the token budget
    public void Extract_CompressRatioThatCannotTileTheBudget_IsRejected(int ratio)
    {
        var ex = Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.attention.compress_ratios"] = I32Array(0, 0, 0, ratio)));
        Assert.Contains("compress ratio", ex.Message);
    }

    [Fact]
    public void Extract_WrongLengthCompressRatios_AreRejected()
    {
        Assert.Throws<InvalidDataException>(() =>
            ExtractMutated(d => d["qwen4exp.attention.compress_ratios"] = I32Array(0, 0, 4)));
    }

    // ───────────────────────────── explicit refusals ─────────────────────────────

    [Fact]
    public void CpuLoader_LoadsQwen4Exp_WithTheOracleModel()
    {
        // #816: the explicit CPU refusal of #815 is replaced by the CPU reference forward.
        var (file, c) = Open();
        using (file)
        {
            using var model = ModelLoader.CreateCpuModelFromGguf(file, c);
            Assert.IsType<Qwen4ExpTransformerModel>(model);
        }
    }

    [Fact]
    public void CpuLoader_LoadFromGguf_RunsAForwardOnTheSyntheticFixture()
    {
        string path = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "whole.gguf"));
        var (model, gguf, _) = ModelLoader.LoadFromGguf(path);
        using (gguf) using (model)
        {
            using var logits = model.Forward([4, 7, 2, 9], [0, 1, 2, 3], -1);
            Assert.Equal(SyntheticQwen4ExpGguf.VocabSize, logits.Shape[1]);
        }
    }

    [Fact]
    public void CudaLoader_RefusesQwen4Exp_BeforeTouchingTheDevice()
    {
        var (file, c) = Open();
        using (file)
        {
            // The refusal precedes context creation, so this runs (and must throw) on a machine with no NVIDIA GPU.
            var ex = Assert.Throws<NotSupportedException>(() => CudaModelLoader.CreateFromGguf(file, c));
            Assert.Contains("CUDA", ex.Message);
            Assert.Contains("Qwen4Exp", ex.Message);
            Assert.Throws<NotSupportedException>(() => CudaTransformerModel.RejectUnsupportedArchitecture(c));
        }
    }

    [Fact]
    public void SharedDenseLoaderGuard_RefusesQwen4Exp_SoVulkanAndTheGenericPathsDoNotMisreport()
    {
        var (file, c) = Open();
        using (file)
        {
            // TransformerWeights.LoadFromGguf and VulkanTransformerModel.RejectUnsupportedArchitecture both call this first;
            // without the arm Vulkan would say "Hybrid SSM / Mamba architectures are not supported" and the CPU generic
            // path "blk.0.attn_output.weight not present".
            Assert.Throws<NotSupportedException>(() => TransformerWeights.ThrowIfArchitectureNeedsDedicatedLoader(c));
            var ex = Assert.Throws<NotSupportedException>(() => VulkanTransformerModel.RejectUnsupportedArchitecture(c));
            Assert.Contains("Qwen4Exp", ex.Message);
            Assert.Throws<NotSupportedException>(() => TransformerWeights.LoadFromGguf(file, c));
        }
    }

    [Fact]
    public void UnsupportedMessage_NamesTheBackendAndTheTrackingIssue()
    {
        string m = Qwen4ExpConfig.UnsupportedMessage("Vulkan");
        Assert.Contains("Vulkan", m);
        Assert.Contains("#814", m);
        Assert.False(GpuOffloadPlanner.SupportsPartialOffload(Architecture.Qwen4Exp));
    }
}
