using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #815: the extractor and the tensor contract checked against the REAL <c>unsloth/Qwen3.8-Flash-Next-GGUF</c> headers
/// (UD-Q4_K_XL, 4 shards = 1224 tensors; and the standalone Q4_K_M MTP file = 34 tensors), fetched as headers only. The fixture is
/// <c>TestData/qwen4exp-real-headers.json</c>: every <c>qwen4exp.*</c> / <c>split.*</c> key with exact values plus each shard's
/// (name, dims, ggml type) table. No weights are involved.
/// </summary>
public sealed class Qwen4ExpRealHeaderTests
{
    private static readonly JsonDocument s_doc = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Models", "Gguf", "TestData", "qwen4exp-real-headers.json")));

    private sealed record Real(ModelConfig Config, Dictionary<string, GgufTensorDescriptor> Tensors, int[][] ShardTensorCounts);

    private static Real Load(string which)
    {
        JsonElement root = s_doc.RootElement.GetProperty(which);

        var meta = new Dictionary<string, GgufMetadataValue>();
        foreach (JsonProperty kv in root.GetProperty("kv").EnumerateObject())
            meta[kv.Name] = ToMetadataValue(kv.Value.GetProperty("t").GetString()!, kv.Value.GetProperty("v"));
        // The tokenizer array is dropped from the fixture; the vocabulary size equals token_embd's second dim.
        meta["qwen4exp.vocab_size"] = new GgufMetadataValue(GgufValueType.UInt32, (uint)root.GetProperty("vocab").GetInt32());
        ModelConfig config = GgufModelConfigExtractor.Extract(new GgufMetadata(meta));

        var tensors = new Dictionary<string, GgufTensorDescriptor>();
        var counts = new List<int[]>();
        foreach (JsonElement shard in root.GetProperty("shards").EnumerateArray())
        {
            int n = 0;
            foreach (JsonElement t in shard.EnumerateArray())
            {
                string name = t[0].GetString()!;
                int[] dims = t[1].EnumerateArray().Select(d => d.GetInt32()).ToArray();
                uint rawType = t[2].GetUInt32();
                Assert.True(Enum.IsDefined(typeof(QuantizationType), (int)rawType), $"{name}: ggml type {rawType} is not a QuantizationType");
                tensors.Add(name, new GgufTensorDescriptor(name, new TensorShape(dims), (QuantizationType)rawType, 0));
                n++;
            }
            counts.Add([n]);
        }
        return new Real(config, tensors, counts.ToArray());
    }

    private static GgufMetadataValue ToMetadataValue(string type, JsonElement v) => type switch
    {
        "str" => new GgufMetadataValue(GgufValueType.String, v.GetString()!),
        "u16" => new GgufMetadataValue(GgufValueType.UInt16, v.GetUInt16()),
        "u32" => new GgufMetadataValue(GgufValueType.UInt32, v.GetUInt32()),
        "i32" => new GgufMetadataValue(GgufValueType.Int32, v.GetInt32()),
        "f32" => new GgufMetadataValue(GgufValueType.Float32, (float)v.GetDouble()),
        "arr<i32>" => new GgufMetadataValue(GgufValueType.Array, v.EnumerateArray().Select(e => e.GetInt32()).ToArray()),
        "arr<u64>" => new GgufMetadataValue(GgufValueType.Array, v.EnumerateArray().Select(e => e.GetUInt64()).ToArray()),
        _ => throw new NotSupportedException($"fixture value type {type}"),
    };

    private static Real Trunk => Load("trunk");
    private static Real Mtp => Load("mtp");

    // ───────────────────────────── config against real values ─────────────────────────────

    [Fact]
    public void TrunkConfig_MatchesTheRealHeader()
    {
        ModelConfig c = Trunk.Config;

        Assert.Equal(Architecture.Qwen4Exp, c.Architecture);
        Assert.Equal(48, c.NumLayers);
        Assert.Equal(0, c.NextnPredictLayers);      // the trunk file has NO nextn_predict_layers key
        Assert.Equal(2560, c.HiddenSize);
        Assert.Equal(248320, c.VocabSize);
        Assert.Equal(24, c.NumAttentionHeads);
        Assert.Equal(2, c.NumKvHeads);
        Assert.Equal(256, c.HeadDim);
        Assert.Equal(262144, c.MaxSequenceLength);
        Assert.Equal(1e-6f, c.NormEpsilon, 1e-9f);
        Assert.Equal(1e7f, c.RoPEConfig!.Value.Theta);
        Assert.Equal(64, c.RoPEConfig.Value.DimensionCount);

        Assert.Equal(new GatedDeltaNetConfig(4, 48, 16, 128, 6144, 4), c.GdnConfig);
        Assert.Equal(512, c.Moe!.NumExperts);
        Assert.Equal(10, c.Moe.NumExpertsPerTok);
        Assert.Equal(640, c.Moe.MoeIntermediateSize);
        Assert.Equal(640, c.Moe.SharedExpertIntermediateSize);
        Assert.True(c.Moe.HasSharedExpertGate);

        // (GDN, GDN, GDN, QSA) x 12.
        var kinds = c.HybridLayout!.LayerKind;
        Assert.Equal(48, kinds.Length);
        Assert.Equal(12, kinds.Count(k => k == HybridLayerKind.Attention));
        for (int i = 0; i < 48; i++)
            Assert.Equal((i + 1) % 4 == 0 ? HybridLayerKind.Attention : HybridLayerKind.GatedDeltaNet, kinds[i]);

        Qwen4ExpConfig q = c.Qwen4Exp!;
        Assert.Equal(4, q.HyperConnectionCount);
        Assert.Equal(320, q.HyperConnectionLowRank);
        Assert.Equal(4, q.IndexerHeadCount);
        Assert.Equal(128, q.IndexerKeyLength);
        Assert.Equal(2048, q.IndexerTopK);
        Assert.Equal(4, q.IndexerBlockSize);
        Assert.Equal(48, q.CompressRatios.Count);
        for (int i = 0; i < 48; i++)
            Assert.Equal((i + 1) % 4 == 0 ? 4 : 0, q.CompressRatios[i]);
        Assert.Equal([11, 11, 10, 0], q.RopeSections);
    }

    [Fact]
    public void TrunkConfig_PleHashConstants_AreTheExactRealUInt64s()
    {
        Qwen4ExpPleConfig ple = Trunk.Config.Qwen4Exp!.Ple!;

        Assert.Equal([1], ple.Layers);                       // zero-based in the GGUF (HF's 1-based ple_layer_ids is [2])
        Assert.Equal(3, ple.NgramSize);
        Assert.Equal(8, ple.HeadsPerNgram);
        Assert.Equal(16, ple.NumHeads);
        Assert.Equal(4, ple.ConvKernel);
        Assert.Equal(248044, ple.EosTokenId);
        Assert.Equal(248056, ple.ImageTokenId);
        Assert.Equal(160, ple.RowDim);
        Assert.Equal([23703573157769UL, 20109073645365UL, 8052911324071UL], ple.LayerMultipliers);
        Assert.Equal(16, ple.HeadOffsets.Count);
        Assert.Equal(0UL, ple.HeadOffsets[0]);
        Assert.Equal(20000003UL, ple.HeadOffsets[1]);
        Assert.Equal(300001275UL, ple.HeadOffsets[15]);
        Assert.Equal(20000003UL, ple.HeadVocabSizes[0]);
        Assert.Equal(20000171UL, ple.HeadVocabSizes[15]);
        // Each head's range starts where the previous one ended (distinct prime moduli, no gaps).
        for (int h = 1; h < 16; h++)
            Assert.Equal(ple.HeadOffsets[h - 1] + ple.HeadVocabSizes[h - 1], ple.HeadOffsets[h]);
        Assert.Equal(320001446UL, ple.MinTableRows);         // the table itself is padded to 320001536 rows
    }

    [Fact]
    public void MtpFileConfig_SeparatesTheNextnBlock()
    {
        ModelConfig c = Mtp.Config;

        Assert.Equal(48, c.NumLayers);                        // block_count 49 minus nextn_predict_layers 1
        Assert.Equal(1, c.NextnPredictLayers);
        Assert.Equal(49, c.Qwen4Exp!.CompressRatios.Count);
        Assert.Equal(4, c.Qwen4Exp.CompressRatios[48]);       // the MTP block is a QSA block
        Assert.Null(c.Qwen4Exp.Ple);                          // no PLE keys in the MTP file
    }

    // ───────────────────────────── tensor contract against the real tables ─────────────────────────────

    [Fact]
    public void TrunkTensorTable_IsExactlyTheContract()
    {
        Real r = Trunk;

        Assert.Equal(1224, r.Tensors.Count);                  // == split.tensors.count of the real set
        Assert.Equal([0, 297, 752, 175], r.ShardTensorCounts.Select(c => c[0]).ToArray());   // shard 1 carries no tensors
        Assert.Empty(Qwen4ExpTensors.FindProblems(r.Tensors, r.Config));
    }

    [Fact]
    public void MtpTensorTable_IsExactlyTheContract()
    {
        Real r = Mtp;

        Assert.Equal(34, r.Tensors.Count);
        Assert.Empty(Qwen4ExpTensors.FindProblems(r.Tensors, r.Config, includeTrunk: false, includeMtp: true));
        // Standalone MTP file: own token_embd + output, no trunk head mixer, no n-gram table.
        Assert.Contains("token_embd.weight", r.Tensors.Keys);
        Assert.Contains("output.weight", r.Tensors.Keys);
        Assert.DoesNotContain("output_hc_norm.weight", r.Tensors.Keys);
        Assert.DoesNotContain("per_layer_token_embd.weight", r.Tensors.Keys);
    }

    [Fact]
    public void RealNames_WhereTheDesignNotesDiffer_AreWhatTheContractUses()
    {
        Dictionary<string, GgufTensorDescriptor> t = Trunk.Tensors;

        // Indexer tensors are DOTTED, not `indexer_q_proj`.
        Assert.Equal([2560, 512], t["blk.3.indexer.q_proj.weight"].Shape.Dimensions);
        Assert.Equal([2560, 128], t["blk.3.indexer.k_proj.weight"].Shape.Dimensions);
        Assert.DoesNotContain("blk.3.indexer_q_proj.weight", t.Keys);
        // The final head mixer is `output_hc_*`; `hc_head_*` exists only inside the nextn block.
        Assert.Equal([10240], t["output_hc_norm.weight"].Shape.Dimensions);
        Assert.Equal([10240, 320], t["output_hc_down.weight"].Shape.Dimensions);
        Assert.Equal([320, 10240], t["output_hc_up.weight"].Shape.Dimensions);
        Assert.DoesNotContain(t.Keys, k => k.StartsWith("hc_head_", StringComparison.Ordinal));
        // Routed experts ship SPLIT (gate + up), not a fused gate_up bank.
        Assert.Equal([2560, 640, 512], t["blk.0.ffn_gate_exps.weight"].Shape.Dimensions);
        Assert.Equal([2560, 640, 512], t["blk.0.ffn_up_exps.weight"].Shape.Dimensions);
        Assert.Equal([640, 2560, 512], t["blk.0.ffn_down_exps.weight"].Shape.Dimensions);
        Assert.DoesNotContain(t.Keys, k => k.Contains("gate_up", StringComparison.Ordinal));
        // `ssm_a` has no .weight suffix; `ssm_dt` is a .bias.
        Assert.Contains("blk.0.ssm_a", t.Keys);
        Assert.Contains("blk.0.ssm_dt.bias", t.Keys);
        Assert.DoesNotContain("blk.0.ssm_a.weight", t.Keys);
        // The router (512 x 2560) and the sigmoid shared-expert gate (a single vector).
        Assert.Equal([2560, 512], t["blk.0.ffn_gate_inp.weight"].Shape.Dimensions);
        Assert.Equal([2560], t["blk.0.ffn_gate_inp_shexp.weight"].Shape.Dimensions);
        // q_proj carries [q | gate] per head: 24 heads x 256 x 2.
        Assert.Equal([2560, 12288], t["blk.3.attn_q.weight"].Shape.Dimensions);
        // PLE sub-tensors sit on block 1 only.
        Assert.Equal([2560, 10240], t["blk.1.ple_key.weight"].Shape.Dimensions);
        Assert.Equal([4, 10240], t["blk.1.ple_conv1d.weight"].Shape.Dimensions);
        Assert.DoesNotContain("blk.0.ple_key.weight", t.Keys);
    }

    [Fact]
    public void NgramTable_IsOneHugeTensor_ThatFitsTheByteCountArithmetic()
    {
        GgufTensorDescriptor table = Trunk.Tensors["per_layer_token_embd.weight"];

        Assert.Equal([160, 320001536], table.Shape.Dimensions);
        Assert.Equal(QuantizationType.IQ4_NL, table.QuantizationType);
        Assert.Equal(51_200_245_760L, table.Shape.ElementCount);                    // > Int32.MaxValue elements
        long bytes = table.QuantizationType.ComputeByteCount(table.Shape.ElementCount);
        Assert.Equal(51_200_245_760L / 32 * 18, bytes);                              // ~28.8 GB, exact (no int overflow)
    }

    [Fact]
    public void TensorTypes_IncludeEveryIdTheRealFilesUse()
    {
        // Q5_1 (7) experts, BF16 (30) indexer projections, IQ4_NL (20) table, Q4_K/Q5_K/Q6_K/Q8_0/F32/F16: all must be known.
        var used = Trunk.Tensors.Values.Select(d => d.QuantizationType).Distinct().ToHashSet();
        Assert.Contains(QuantizationType.Q5_1, used);
        Assert.Contains(QuantizationType.BF16, used);
        Assert.Contains(QuantizationType.IQ4_NL, used);
        Assert.Contains(QuantizationType.Q4_K, used);
    }
}
