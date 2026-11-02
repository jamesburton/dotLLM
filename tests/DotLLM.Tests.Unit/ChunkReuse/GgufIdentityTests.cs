using DotLLM.ChunkReuse;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Content-based model and tokenizer identities for chunk-store keys (<see cref="GgufIdentity"/>): they follow the GGUF's
/// content, not its file name, so a re-quantized or re-downloaded file with the same name is a store miss, not a wrong hit.
/// </summary>
public sealed class GgufIdentityTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-ggufid-" + Guid.NewGuid().ToString("N"));

    public GgufIdentityTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private sealed record Spec
    {
        public string[] Tokens { get; init; } = ["a", "b", "c", "d"];
        public string[] Merges { get; init; } = ["a b"];
        public string ChatTemplate { get; init; } = "{{ messages }}";
        public string Name { get; init; } = "tiny";
        public int WeightIndexToFlip { get; init; } = -1; // flips a float in the second tensor
        public int SecondTensorFloats { get; init; } = 4096; // 16 KiB: floats 1024..3071 (bytes 4 KiB..12 KiB) are not sampled
        public bool Split { get; init; } // two shards: a.weight in shard 1, b.weight in shard 2
    }

    private (string ModelId, string TokenizerId) Ids(Spec spec, string fileName)
    {
        var (sampled, _, tok) = AllIds(spec, fileName);
        return (sampled, tok);
    }

    // Writes the file (or, for a split spec, the set "<stem>-00001-of-00002.gguf" + "-00002-") and returns the sampled model id,
    // the full-hash model id and the tokenizer id.
    private (string Sampled, string Full, string Tokenizer) AllIds(Spec spec, string fileName)
    {
        var w = new GgufWriter();
        if (spec.Split)
        {
            w.AddUInt16("split.no", 0);
            w.AddUInt16("split.count", 2);
            w.AddInt32("split.tensors.count", 2);
        }
        w.AddString("general.architecture", "llama");
        w.AddString("general.name", spec.Name);
        w.AddUInt32("general.alignment", 32);
        w.AddString("tokenizer.ggml.model", "gpt2");
        w.AddStringArray("tokenizer.ggml.tokens", spec.Tokens);
        w.AddStringArray("tokenizer.ggml.merges", spec.Merges);
        w.AddString("tokenizer.chat_template", spec.ChatTemplate);
        var first = new float[64];
        for (int i = 0; i < first.Length; i++) first[i] = i * 0.25f;
        var second = new float[spec.SecondTensorFloats];
        for (int i = 0; i < second.Length; i++) second[i] = 1f + i;
        if (spec.WeightIndexToFlip >= 0) second[spec.WeightIndexToFlip] = -7f;
        w.AddTensor("a.weight", [64], (uint)QuantizationType.F32, Bytes(first));
        string path;
        if (spec.Split)
        {
            string stem = Path.GetFileNameWithoutExtension(fileName);
            path = Path.Combine(_dir, $"{stem}-00001-of-00002.gguf");
            File.WriteAllBytes(path, w.Build());
            var w2 = new GgufWriter();
            w2.AddUInt16("split.no", 1);
            w2.AddUInt16("split.count", 2);
            w2.AddUInt32("general.alignment", 32);
            w2.AddTensor("b.weight", [second.Length], (uint)QuantizationType.F32, Bytes(second));
            File.WriteAllBytes(Path.Combine(_dir, $"{stem}-00002-of-00002.gguf"), w2.Build());
        }
        else
        {
            w.AddTensor("b.weight", [second.Length], (uint)QuantizationType.F32, Bytes(second));
            path = Path.Combine(_dir, fileName);
            File.WriteAllBytes(path, w.Build());
        }
        using var gguf = GgufFile.Open(path);
        Assert.Equal(spec.Split ? 2 : 1, gguf.ShardPaths.Count);
        return (GgufIdentity.ModelId(gguf), GgufIdentity.ModelId(gguf, fullHash: true), GgufIdentity.TokenizerId(gguf.Metadata));
    }

    private static byte[] Bytes(float[] f) => System.Runtime.InteropServices.MemoryMarshal.AsBytes(f.AsSpan()).ToArray();

    [Fact]
    public void SameContent_DifferentFileName_SameIds()
    {
        var a = Ids(new Spec(), "model-Q4_K_M.gguf");
        var b = Ids(new Spec(), "renamed-Q8_0.gguf");
        Assert.Equal(a, b);
        Assert.StartsWith("gguf-sha256:", a.ModelId, StringComparison.Ordinal);
        Assert.StartsWith("gguf-tokenizer-sha256:", a.TokenizerId, StringComparison.Ordinal);
    }

    [Fact]
    public void WeightChangeInSampledPrefix_ChangesModelId_NotTokenizerId()
    {
        var a = Ids(new Spec(), "m.gguf");
        var b = Ids(new Spec { WeightIndexToFlip = 10 }, "m.gguf");
        Assert.NotEqual(a.ModelId, b.ModelId);
        Assert.Equal(a.TokenizerId, b.TokenizerId);
    }

    [Fact]
    public void WeightChangeInSampledTail_ChangesModelId()
    {
        // The last 4 KiB of every tensor are sampled too: catches end-of-vocab row edits (e.g. trained reserved-token rows).
        Assert.NotEqual(Ids(new Spec(), "m.gguf").ModelId, Ids(new Spec { WeightIndexToFlip = 4090 }, "m.gguf").ModelId);
    }

    [Fact]
    public void WeightChangeInTheMiddle_MissedBySampledId_CaughtByFullHash_DocumentedLimit()
    {
        // Bytes 8 KiB into a 16 KiB tensor lie between the sampled head and tail: the sampled id misses the edit, the full hash
        // does not.
        var a = AllIds(new Spec(), "m.gguf");
        var b = AllIds(new Spec { WeightIndexToFlip = 2048 }, "m.gguf");
        Assert.Equal(a.Sampled, b.Sampled);
        Assert.NotEqual(a.Full, b.Full);
        Assert.StartsWith("gguf-full-sha256:", a.Full, StringComparison.Ordinal);
        Assert.NotEqual(a.Sampled, a.Full);
    }

    [Fact]
    public void ModelId_WorksWhileThisProcessHoldsAWritableHandle()
    {
        // e.g. a Vulkan zero-copy import alias opened read-write on the same file (#1190).
        string expected = Ids(new Spec(), "w.gguf").ModelId;
        string path = Path.Combine(_dir, "w.gguf");
        using var gguf = GgufFile.Open(path);
        using var writer = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        Assert.Equal(expected, GgufIdentity.ModelId(gguf));
    }

    [Fact]
    public void FullHash_SameContentDifferentName_SameId()
    {
        Assert.Equal(AllIds(new Spec(), "x.gguf").Full, AllIds(new Spec(), "y.gguf").Full);
    }

    [Fact]
    public void SplitSet_IsSupported_AndEveryShardIsHashed()
    {
        var a = AllIds(new Spec { Split = true }, "set.gguf");
        // Renaming the whole set keeps the ids.
        Assert.Equal(a, AllIds(new Spec { Split = true }, "renamed.gguf"));
        // A split set is not the same file as the single-file layout of the same tensors.
        Assert.NotEqual(a.Sampled, Ids(new Spec(), "single.gguf").ModelId);
        // Changes in the second shard's tensor: head and tail caught by the sampled id, the middle only by the full hash.
        Assert.NotEqual(a.Sampled, AllIds(new Spec { Split = true, WeightIndexToFlip = 10 }, "s2.gguf").Sampled);
        Assert.NotEqual(a.Sampled, AllIds(new Spec { Split = true, WeightIndexToFlip = 4090 }, "s3.gguf").Sampled);
        var mid = AllIds(new Spec { Split = true, WeightIndexToFlip = 2048 }, "s4.gguf");
        Assert.Equal(a.Sampled, mid.Sampled);
        Assert.NotEqual(a.Full, mid.Full);
    }

    [Fact]
    public void MetadataChange_ChangesModelId()
    {
        Assert.NotEqual(Ids(new Spec(), "m.gguf").ModelId, Ids(new Spec { Name = "other" }, "m.gguf").ModelId);
    }

    [Theory]
    [InlineData("tokens")]
    [InlineData("merges")]
    public void TokenizerContentChange_ChangesTokenizerId(string what)
    {
        var a = Ids(new Spec(), "m.gguf");
        var b = Ids(what == "tokens" ? new Spec { Tokens = ["a", "b", "c", "e"] } : new Spec { Merges = ["a c"] }, "m.gguf");
        Assert.NotEqual(a.TokenizerId, b.TokenizerId);
    }

    [Fact]
    public void ChatTemplateChange_KeepsTokenizerId()
    {
        // Only tokenizer.ggml.* (vocabulary, merges, types, special ids) maps text to token ids; the chat template does not.
        Assert.Equal(Ids(new Spec(), "m.gguf").TokenizerId, Ids(new Spec { ChatTemplate = "other" }, "m.gguf").TokenizerId);
    }

    [Fact]
    public void TokenizerId_BoundaryBetweenArrayItemsCannotBeForged()
    {
        Assert.NotEqual(Ids(new Spec { Tokens = ["ab", "c", "d", "e"] }, "m.gguf").TokenizerId,
                        Ids(new Spec { Tokens = ["a", "bc", "d", "e"] }, "m.gguf").TokenizerId);
    }
}
