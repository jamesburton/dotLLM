using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// Issue #784 — plain <c>Encode()</c> on a gemma4 GGUF must prepend BOS (run/bench/server completions); the factory
/// previously enabled it for gemma/gemma2 only, so Gemma-4 prompts reached the model BOS-less and degenerated.
/// </summary>
public sealed class Gemma4AddBosTests
{
    [Fact]
    public void Gemma4Gguf_Tokenizer_PrependsBos()
    {
        string path = Path.Combine(Path.GetTempPath(), $"g4bos-{Guid.NewGuid():N}.gguf");
        try
        {
            SyntheticGemma4Gguf.WriteGemma4(path);
            using var gguf = GgufFile.Open(path);
            var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
            Assert.True(tok.AddBosToken);
            int[] ids = tok.Encode("a");
            Assert.Equal(tok.BosTokenId, ids[0]);
        }
        finally { try { File.Delete(path); } catch { } }
    }
}
