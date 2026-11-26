using System.Text.Json;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Tokenizers;

/// <summary>
/// The Inkling 2 tokenizer (<c>tokenizer.ggml.pre = inkling2</c>: GPT-4o pre-tokenizer, <c>ignore_merges</c>, no BOS) built from the real GGUF vocabulary
/// against llama.cpp's own <c>llama-tokenize</c> (b12780) on <c>Fixtures/inkling2-tokenizer-cases.json</c> (#2026; regenerate with
/// <c>Fixtures/generate_inkling2_tokenizer_cases.py</c>). The strings cover multi-digit numbers, contractions, CJK/kana runs, whitespace runs, newlines, code, emoji and mixed scripts.
/// Skipped when the GGUF is not in the HF cache.
/// </summary>
public sealed class Inkling2TokenizerParityTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2TokenizerParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void GgufTokenizer_MatchesLlamaTokenize()
    {
        string? gguf = Models.Architectures.Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var tok = GgufBpeTokenizerFactory.Load(file.Metadata);
        string cases = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "Fixtures", "inkling2-tokenizer-cases.json"));
        using var doc = JsonDocument.Parse(cases);
        var failures = new List<string>();
        foreach (var c in doc.RootElement.EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!, text = c.GetProperty("text").GetString()!;
            int[] want = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] got = tok.EncodeRaw(text);
            if (!got.AsSpan().SequenceEqual(want))
            {
                int i = 0; while (i < Math.Min(got.Length, want.Length) && got[i] == want[i]) i++;
                failures.Add($"{name}: first difference at token {i} (got {got.Length} tokens, llama.cpp {want.Length}): got {(i < got.Length ? got[i] : -1)} want {(i < want.Length ? want[i] : -1)}");
            }
            // and back: decoding must reproduce the text
            Assert.Equal(text, tok.Decode(want));
        }
        foreach (var f in failures) _output.WriteLine(f);
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [SkippableFact]
    public void GgufTokenizer_HasNoBos_AndEosIsTheTurnEnd()
    {
        string? gguf = Models.Architectures.Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        using var file = GgufFile.Open(gguf!);
        var tok = GgufBpeTokenizerFactory.Load(file.Metadata);
        // add_bos_token = false: a plain Encode must not prepend anything
        Assert.Equal(tok.EncodeRaw("Hello"), tok.Encode("Hello"));
        Assert.Equal(200006, tok.EosTokenId);
        Assert.Equal(200058, tok.VocabSize);
    }
}
