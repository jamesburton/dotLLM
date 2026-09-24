using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #516: whether a model prepends BOS is a property of the vocab, and llama.cpp derives it
/// from the vocab type / pre-tokenizer with the <c>tokenizer.ggml.add_bos_token</c> KV as an
/// <i>override</i>, not as the source.
/// </summary>
/// <remarks>
/// The discriminating pair is <see cref="LlamaBpe_NoKey_AddsBos"/> and
/// <see cref="SmolLm_KeyFalse_DoesNotAddBos"/>, taken from two real fixtures in
/// <c>~/.dotllm/quant-ladder/</c>. Both are <c>tokenizer.ggml.model = "gpt2"</c>, so nothing but
/// the pre-type and the KV separates them:
/// <list type="bullet">
/// <item>Llama-3.2-1B — <c>pre = "llama-bpe"</c>, <b>no</b> <c>add_bos_token</c> key, and
/// llama.cpp prepends BOS anyway.</item>
/// <item>SmolLM-135M — <c>pre = "smollm"</c>, <c>add_bos_token = false</c>.</item>
/// </list>
/// Reading the KV with a <c>false</c> default gets Llama wrong; prepending unconditionally gets
/// SmolLM wrong. That is exactly the bug in #515: dotLLM scored Llama-3.2 with no BOS while
/// llama.cpp prepended one, so every chunk boundary was displaced by a token and the two engines
/// compared different text.
/// </remarks>
public class GgufAddBosResolverTests
{
    private static GgufMetadata Build(Action<GgufTestData> configure)
    {
        var data = new GgufTestData(version: 3);
        configure(data);
        using var stream = new MemoryStream(data.Build());
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        return new GgufMetadata(GgufReader.ReadMetadata(reader, header));
    }

    private static GgufMetadata Vocab(string model, string? pre = null, bool? addBos = null) =>
        Build(d =>
        {
            d.AddString("tokenizer.ggml.model", model);
            if (pre is not null) d.AddString("tokenizer.ggml.pre", pre);
            if (addBos is not null) d.AddBool("tokenizer.ggml.add_bos_token", addBos.Value);
        });

    [Fact]
    public void LlamaBpe_NoKey_AddsBos() =>
        Assert.True(GgufAddBosResolver.Resolve(Vocab("gpt2", "llama-bpe")));

    [Fact]
    public void SmolLm_KeyFalse_DoesNotAddBos() =>
        Assert.False(GgufAddBosResolver.Resolve(Vocab("gpt2", "smollm", addBos: false)));

    [Fact]
    public void ExplicitKey_OverridesPreTypeDefault_False() =>
        Assert.False(GgufAddBosResolver.Resolve(Vocab("gpt2", "llama-bpe", addBos: false)));

    [Fact]
    public void ExplicitKey_OverridesPreTypeDefault_True() =>
        Assert.True(GgufAddBosResolver.Resolve(Vocab("gpt2", "smollm", addBos: true)));

    [Fact]
    public void SentencePiece_AddsBos_WithoutPreType() =>
        Assert.True(GgufAddBosResolver.Resolve(Vocab("llama")));

    [Fact]
    public void UnknownBpePreType_DoesNotAddBos() =>
        Assert.False(GgufAddBosResolver.Resolve(Vocab("gpt2", "starcoder")));

    // llama-vocab.cpp routes this whole list through LLAMA_VOCAB_PRE_TYPE_LLAMA3, which sets
    // add_bos = true. Listed explicitly rather than looped over a shared constant so that a
    // future edit to the table has to touch a test that names the members.
    [Theory]
    [InlineData("llama3")]
    [InlineData("llama-v3")]
    [InlineData("llama-bpe")]
    [InlineData("falcon3")]
    [InlineData("falcon-h1")]
    [InlineData("pixtral")]
    [InlineData("midm-2.0")]
    [InlineData("lfm2")]
    [InlineData("jina-v5-nano")]
    [InlineData("tekken")]
    [InlineData("chameleon")]
    public void Llama3Family_AddsBos(string pre) =>
        Assert.True(GgufAddBosResolver.Resolve(Vocab("gpt2", pre)));

    [Fact]
    public void Gemma4_ForcedTrue_EvenWhenKeySaysFalse() =>
        Assert.True(GgufAddBosResolver.Resolve(Vocab("gemma4", "gemma4", addBos: false)));
}
