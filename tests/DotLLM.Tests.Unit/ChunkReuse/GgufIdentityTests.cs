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
        public int SecondTensorFloats { get; init; } = 4096; // 16 KiB: an index past 4 KiB is outside the sampled prefix
    }

    private (string ModelId, string TokenizerId) Ids(Spec spec, string fileName)
    {
        var w = new GgufWriter();
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
        w.AddTensor("b.weight", [second.Length], (uint)QuantizationType.F32, Bytes(second));
        string path = Path.Combine(_dir, fileName);
        File.WriteAllBytes(path, w.Build());
        using var gguf = GgufFile.Open(path);
        return (GgufIdentity.ModelId(gguf, path), GgufIdentity.TokenizerId(gguf.Metadata));
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
    public void WeightChangeOutsideSampledPrefix_IsNotDetected_DocumentedLimit()
    {
        // The model id hashes the header, the tensor table and the first 4 KiB of every tensor; not a full-weight hash.
        var a = Ids(new Spec(), "m.gguf");
        var b = Ids(new Spec { WeightIndexToFlip = 2048 }, "m.gguf");
        Assert.Equal(a.ModelId, b.ModelId);
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
