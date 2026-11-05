using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #1643: the extractor, the census and the tokenizer wiring checked against the REAL Ripple-65B-Preview GGUF headers
/// (AtomicChat Q4_K_M = the llama.cpp PR #31692 contract, prometheusAIR Q6_K), fetched as header-only range reads (first 10 MiB,
/// no weights). Fixture <c>TestData/ricursive-real-headers.json</c>: every KV with its exact GGUF type (tokenizer arrays reduced to
/// their lengths plus the special-token strings), the chat template's SHA-256, and each file's full 1203-entry (name, dims, ggml type)
/// tensor table. These are the strings the synthetic fixture is modelled on; a test passing here passes on the real contract.
/// </summary>
public sealed class RicursiveRealHeaderTests
{
    private static readonly JsonDocument s_doc = JsonDocument.Parse(
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Models", "Gguf", "TestData", "ricursive-real-headers.json")));

    internal sealed record Real(ModelConfig Config, GgufMetadata Metadata, List<(string Name, int[] Dims)> Tensors, JsonElement Root);

    internal static Real Load(string which)
    {
        JsonElement root = s_doc.RootElement.GetProperty(which);
        var meta = new Dictionary<string, GgufMetadataValue>(StringComparer.Ordinal);
        foreach (JsonProperty kv in root.GetProperty("kv").EnumerateObject())
            meta[kv.Name] = ToMetadataValue(kv.Value.GetProperty("t").GetString()!, kv.Value.GetProperty("v"));

        // Rebuild a vocabulary of the real length with the real special-token strings/types at their real ids (everything else is a
        // placeholder), so the tokenizer factory sees the real id layout (bos/eos/eot/pad + text lookup of <|im_end|> / <|endoftext|>).
        int n = root.GetProperty("tokenizer_arrays").GetProperty("tokenizer.ggml.tokens").GetInt32();
        var tokens = new string[n];
        var types = new int[n];
        for (int i = 0; i < n; i++) { tokens[i] = "t" + i.ToString(System.Globalization.CultureInfo.InvariantCulture); types[i] = 1; }
        foreach (JsonProperty s in root.GetProperty("special_tokens").EnumerateObject())
        {
            int id = int.Parse(s.Name, System.Globalization.CultureInfo.InvariantCulture);
            tokens[id] = s.Value.GetProperty("text").GetString()!;
            types[id] = s.Value.GetProperty("type").GetInt32();
        }
        meta["tokenizer.ggml.tokens"] = new GgufMetadataValue(GgufValueType.Array, tokens);
        meta["tokenizer.ggml.token_type"] = new GgufMetadataValue(GgufValueType.Array, types);
        meta["tokenizer.ggml.merges"] = new GgufMetadataValue(GgufValueType.Array, Array.Empty<string>());

        var md = new GgufMetadata(meta);
        var tensors = new List<(string, int[])>();
        foreach (JsonElement t in root.GetProperty("tensors").EnumerateArray())
            tensors.Add((t[0].GetString()!, t[1].EnumerateArray().Select(d => d.GetInt32()).ToArray()));
        return new Real(GgufModelConfigExtractor.Extract(md), md, tensors, root);
    }

    private static GgufMetadataValue ToMetadataValue(string type, JsonElement v) => type switch
    {
        "str" => new GgufMetadataValue(GgufValueType.String, v.GetString()!),
        "u32" => new GgufMetadataValue(GgufValueType.UInt32, v.GetUInt32()),
        "f32" => new GgufMetadataValue(GgufValueType.Float32, (float)v.GetDouble()),
        "bool" => new GgufMetadataValue(GgufValueType.Bool, v.GetBoolean()),
        "arr<bool>" => new GgufMetadataValue(GgufValueType.Array, v.EnumerateArray().Select(e => e.GetBoolean()).ToArray()),
        _ => throw new NotSupportedException($"fixture value type {type}"),
    };

    public static IEnumerable<object[]> Files() => [["atomicchat"], ["prometheus"]];

    [Theory]
    [MemberData(nameof(Files))]
    public void Config_MatchesTheRealHeader(string which)
    {
        ModelConfig c = Load(which).Config;
        Assert.Equal(Architecture.Ricursive, c.Architecture);
        Assert.Equal(80, c.NumLayers);
        Assert.Equal(5120, c.HiddenSize);
        Assert.Equal(40960, c.IntermediateSize);
        Assert.Equal(80, c.NumAttentionHeads);
        Assert.Equal(8, c.NumKvHeads);
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(100352, c.VocabSize);        // prometheusAIR has no ricursive.vocab_size: falls back to the token array length
        Assert.Equal(2097152, c.MaxSequenceLength);
        Assert.Equal(1e-6f, c.NormEpsilon, 1e-12f);
        Assert.Equal(1024, c.SlidingWindowSize);
        Assert.Equal(71.5541763305664f, c.EmbeddingScale);   // F32 of sqrt(5120)
        Assert.Equal(new RoPEConfig(Theta: 10000f, DimensionCount: 128, Type: RoPEType.NeoX), c.RoPEConfig);
        Assert.True(c.UseQkNorm);

        // SSSSF x 16: the full (NoPE) layers are exactly 4, 9, ..., 79.
        Assert.Equal(Enumerable.Range(0, 16).Select(i => 5 * i + 4), c.NoRopeLayers);
        for (int l = 0; l < 80; l++)
            Assert.Equal(l % 5 == 4 ? null : 1024, c.SlidingWindowOf(l));

        var r = c.Ricursive!;
        Assert.True(r.FfnClampDeclared);
        Assert.All(r.FfnClamp, v => Assert.Equal(20f, v));
        Assert.False(r.EmbeddingScaleFolded);
        Assert.True(r.GateFullPrecision);
    }

    [Theory]
    [InlineData("atomicchat", "attn_vres_lambda")]
    [InlineData("prometheus", "attn_value_res")]
    public void Census_IsClean_WithEachFilesValueResidualName(string which, string expectedName)
    {
        var real = Load(which);
        var r = RicursiveTensorCensus.Check(real.Config, real.Tensors);
        Assert.True(r.IsClean, r + "\n" + string.Join("\n", r.Missing.Concat(r.Unexpected).Concat(r.ShapeMismatches)));
        Assert.Equal(1203, r.TensorCount);
        Assert.Equal(expectedName, r.ValueResidualName);
        Assert.True(r.HasGate);
        Assert.True(r.HasOutputHead);
    }

    [Fact]
    public void Census_RenamingOneLayersLogit_ToTheOtherFilesSpelling_IsRejected()
    {
        // The two real tables differ ONLY in the value-residual stem; swapping a single layer's name produces a mixed file.
        var real = Load("atomicchat");
        var mixed = real.Tensors.Select(t => t.Name == "blk.42.attn_vres_lambda.weight" ? ("blk.42.attn_value_res.weight", t.Dims) : t).ToList();
        var r = RicursiveTensorCensus.Check(real.Config, mixed);
        Assert.Equal(["blk.42.attn_vres_lambda.weight"], r.Missing);
        Assert.Equal(["blk.42.attn_value_res.weight"], r.Unexpected);

        var a = real.Tensors.Select(t => t.Name.Replace("attn_vres_lambda", "attn_value_res", StringComparison.Ordinal)).Order(StringComparer.Ordinal);
        var p = Load("prometheus").Tensors.Select(t => t.Name).Order(StringComparer.Ordinal);
        Assert.Equal(p, a);
    }

    [Fact]
    public void RealTensorShapes_AreTheDocumentedOnes()
    {
        var t = Load("atomicchat").Tensors.ToDictionary(x => x.Name, x => x.Dims);
        Assert.Equal([5120, 10240], t["blk.0.attn_q.weight"]);
        Assert.Equal([5120, 10240], t["blk.0.attn_gate.weight"]);   // separate full-width gate, not packed into Q
        Assert.Equal([10240, 5120], t["blk.0.attn_output.weight"]);
        Assert.Equal([5120, 1024], t["blk.0.attn_v.weight"]);
        Assert.Equal([128], t["blk.79.attn_k_norm.weight"]);
        Assert.Equal([1], t["blk.79.attn_vres_lambda.weight"]);
        Assert.Equal([5120, 100352], t["output.weight"]);
    }

    [Theory]
    [MemberData(nameof(Files))]
    public void Tokenizer_RealIds_NoBos_EosImEnd_EogAddsEndOfText(string which)
    {
        var real = Load(which);
        Assert.Equal("gpt2", real.Metadata.GetString("tokenizer.ggml.model"));
        Assert.Equal("dbrx", real.Metadata.GetString("tokenizer.ggml.pre"));
        Assert.Equal(which == "prometheus", real.Metadata.ContainsKey("tokenizer.ggml.add_bos_token"));   // AtomicChat has NO key
        Assert.False(GgufAddBosResolver.Resolve(real.Metadata));

        var tok = GgufBpeTokenizerFactory.Load(real.Metadata);
        Assert.False(tok.AddBosToken);
        Assert.Equal(100257, tok.BosTokenId);   // <|endoftext|>
        Assert.Equal(100265, tok.EosTokenId);   // <|im_end|>
        Assert.Equal([100257], tok.ExtraEndOfGenerationTokenIds);   // HF eos_token_id = [100265, 100257]
    }

    [Fact]
    public void ChatTemplate_EmbeddedInBothFiles_IsTheVendorJinjaByteForByte()
    {
        string template = File.ReadAllText(
            Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "ripple-65b-preview-chat-template.jinja"));
        string sha = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(template)));
        foreach (string which in new[] { "atomicchat", "prometheus" })
            Assert.Equal(sha, s_doc.RootElement.GetProperty(which).GetProperty("tokenizer_arrays").GetProperty("tokenizer.chat_template.sha256").GetString());
    }

    [Fact]
    public void QuantTypes_AreAllKnownToDotLlm()
    {
        foreach (string which in new[] { "atomicchat", "prometheus" })
            foreach (JsonElement t in s_doc.RootElement.GetProperty(which).GetProperty("tensors").EnumerateArray())
                Assert.True(Enum.IsDefined(typeof(QuantizationType), (int)t[2].GetUInt32()), t[0].GetString());
    }
}
