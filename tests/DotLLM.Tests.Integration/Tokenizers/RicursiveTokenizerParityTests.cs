using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Tokenizers;

/// <summary>
/// Ripple-65B-Preview tokenizer id parity (issue #1643). Expected ids come from HuggingFace <c>tokenizers</c> on the real
/// <c>ricursiveAI/Ripple-65B-Preview</c> <c>tokenizer.json</c> (fixture <c>Fixtures/Ricursive/ricursive-tokenizer-hf-expected.json</c>:
/// the vendor <c>TOKENIZER_PROBES</c> plus extras, a wikitext-2 sample, and the HF-rendered chat-template goldens). The same ids must come
/// out of dotLLM's GGUF path (<c>gpt2</c> model, <c>dbrx</c> pre-tokenizer) built from the REAL GGUF vocabulary/merges, read from a
/// header-only range read of the AtomicChat Q4_K_M file (the whole tokenizer lives in the first ~3.6 MB; no weights).
/// </summary>
/// <remarks>
/// The header is looked up at <c>$DOTLLM_RICURSIVE_GGUF_HEADER</c>, then <c>~/.dotllm/test-cache/ricursive-headers/atomicchat-q4km.head.gguf</c>
/// (a generated, non-model artefact). Produce it with an HTTP range request for bytes 0-4194303 of
/// <c>AtomicChat/Ripple-65B-Preview-GGUF/Ripple-65B-Preview-Q4_K_M.gguf</c>. Skips when absent.
/// </remarks>
public sealed class RicursiveTokenizerParityTests(ITestOutputHelper output)
{
    private static string? HeaderPath()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_RICURSIVE_GGUF_HEADER");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        string p = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache",
                                "ricursive-headers", "atomicchat-q4km.head.gguf");
        return File.Exists(p) ? p : null;
    }

    private static JsonDocument Expected() => JsonDocument.Parse(File.ReadAllText(
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Ricursive", "ricursive-tokenizer-hf-expected.json")));

    private static ITokenizer LoadTokenizer(out GgufMetadata metadata)
    {
        string? path = HeaderPath();
        Skip.If(path is null, "Ripple GGUF header not cached (set DOTLLM_RICURSIVE_GGUF_HEADER or see the class remarks).");
        (metadata, _) = GgufHeaderOnly.Read(path!);
        return GgufTokenizerFactory.Load(metadata);
    }

    private int Compare(ITokenizer tok, string label, string text, int[] expected)
    {
        int[] got = tok.Encode(text);
        if (expected.SequenceEqual(got)) return 0;
        int i = 0;
        while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
        output.WriteLine($"DIFF {label} at {i}: hf [{string.Join(",", expected.Skip(Math.Max(0, i - 2)).Take(8))}] "
                         + $"got [{string.Join(",", got.Skip(Math.Max(0, i - 2)).Take(8))}] (len {expected.Length} vs {got.Length})");
        return 1;
    }

    [SkippableFact]
    public void Probes_And_ChatRenders_MatchHuggingFace_IdForId()
    {
        var tok = LoadTokenizer(out var md);
        Assert.Equal("dbrx", md.GetString("tokenizer.ggml.pre"));
        using var doc = Expected();
        int bad = 0;
        foreach (var c in doc.RootElement.GetProperty("probes").EnumerateArray())
            bad += Compare(tok, "probe", c.GetProperty("text").GetString()!, c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray());
        foreach (var c in doc.RootElement.GetProperty("chat_renders").EnumerateArray())
            bad += Compare(tok, c.GetProperty("name").GetString()!, c.GetProperty("text").GetString()!,
                           c.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray());
        Assert.Equal(0, bad);
    }

    [SkippableFact]
    public void Encode_NeverPrependsBos_OnTheRealVocabulary()
    {
        var tok = LoadTokenizer(out var md);
        Assert.False(md.ContainsKey("tokenizer.ggml.add_bos_token"));   // the AtomicChat file: the trap the default must survive
        Assert.NotEqual(100257, tok.Encode("Hello")[0]);
        Assert.Equal(100265, tok.EosTokenId);
        Assert.Equal([100257], tok.ExtraEndOfGenerationTokenIds);
    }

    [SkippableFact]
    public void WikitextSample_MatchesHuggingFace_IdForId()
    {
        var tok = LoadTokenizer(out _);
        using var doc = Expected();
        var w = doc.RootElement.GetProperty("wikitext");
        string corpus = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora",
                                     "wikitext-2-raw", w.GetProperty("file").GetString()!);
        Skip.If(!File.Exists(corpus), $"wikitext-2 corpus not cached at {corpus}.");
        string text = File.ReadAllText(corpus)[..w.GetProperty("chars").GetInt32()];
        Skip.If(Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text))) != w.GetProperty("sha256").GetString(),
                "cached wikitext sample differs from the one the expected ids were made from.");
        Assert.Equal(0, Compare(tok, "wikitext", text, w.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray()));
    }
}
