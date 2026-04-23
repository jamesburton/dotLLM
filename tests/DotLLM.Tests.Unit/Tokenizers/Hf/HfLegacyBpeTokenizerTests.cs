using System.IO;
using System.Text;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Hf;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.Hf;

/// <summary>
/// Covers the legacy-trio loading path in <see cref="HfBpeTokenizerFactory"/>
/// — checkpoints that ship <c>vocab.json</c> + <c>merges.txt</c> +
/// <c>tokenizer_config.json</c> without the consolidated
/// <c>tokenizer.json</c>. The IBM Granite-3.0 MoE repo is the canonical
/// example; GPT-2 proper and older Llama-family repos follow the same layout.
/// </summary>
/// <remarks>
/// Tests are skipped when <c>C:/temp/dotllm-granite3-moe/</c> is absent so
/// the suite remains green on CI without the weights.
/// </remarks>
public class HfLegacyBpeTokenizerTests
{
    private const string GraniteMoeDir = "C:/temp/dotllm-granite3-moe";

    // -------------------------------------------------------------------------
    // Synthetic-fixture tests (no external files needed).
    // -------------------------------------------------------------------------

    [Fact]
    public void TryLoad_MissingTrio_ReturnsNull()
    {
        string tempDir = CreateEmptyTempDirectory();
        try
        {
            Assert.Null(HfLegacyBpeLoader.TryLoad(tempDir));
            Assert.Null(HfBpeTokenizerFactory.TryLoadFromDirectory(tempDir));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TryLoad_VocabWithoutMerges_ReturnsNull()
    {
        string tempDir = CreateEmptyTempDirectory();
        try
        {
            File.WriteAllText(Path.Combine(tempDir, "vocab.json"), "{\"a\":0}");
            // No merges.txt.
            Assert.Null(HfLegacyBpeLoader.TryLoad(tempDir));
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TryLoad_MinimalTrio_BuildsByteLevelSpec()
    {
        string tempDir = CreateEmptyTempDirectory();
        try
        {
            // Two-byte BPE: "ab" merges to a single token.
            File.WriteAllText(
                Path.Combine(tempDir, "vocab.json"),
                "{\"a\":0,\"b\":1,\"ab\":2,\"<unk>\":3}",
                Encoding.UTF8);
            File.WriteAllText(
                Path.Combine(tempDir, "merges.txt"),
                "#version: 0.2\na b\n",
                Encoding.UTF8);
            File.WriteAllText(
                Path.Combine(tempDir, "tokenizer_config.json"),
                """
                {
                  "tokenizer_class": "GPT2Tokenizer",
                  "unk_token": "<unk>",
                  "added_tokens_decoder": {
                    "3": {
                      "content": "<unk>",
                      "special": true
                    }
                  }
                }
                """,
                Encoding.UTF8);

            HfTokenizerSpec? spec = HfLegacyBpeLoader.TryLoad(tempDir);
            Assert.NotNull(spec);

            Assert.Equal(4, spec!.Vocab.Count);
            Assert.Equal(2, spec.Vocab["ab"]);
            Assert.Single(spec.Merges);
            Assert.Equal(("a", "b"), spec.Merges[0]);
            Assert.Equal(HfPreTokenizerKind.ByteLevel, spec.PreTokenizerKind);
            Assert.True(spec.ByteLevelUseRegex);
            Assert.Null(spec.ByteLevelSplitRegex);
            Assert.Equal(HfNormalizerKind.None, spec.NormalizerKind);
            Assert.Contains(HfDecoderStage.ByteLevel, spec.DecoderStages);

            // <unk> should be flagged special via added_tokens_decoder.
            Assert.Contains(spec.AddedTokens, t => t.Id == 3 && t.Content == "<unk>" && t.Special);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TryLoad_PromotesScalarBosEosToSpecials()
    {
        string tempDir = CreateEmptyTempDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(tempDir, "vocab.json"),
                "{\"<|endoftext|>\":0,\"a\":1,\"b\":2,\"ab\":3}",
                Encoding.UTF8);
            File.WriteAllText(
                Path.Combine(tempDir, "merges.txt"),
                "a b\n",
                Encoding.UTF8);
            File.WriteAllText(
                Path.Combine(tempDir, "tokenizer_config.json"),
                """
                {
                  "tokenizer_class": "GPT2Tokenizer",
                  "bos_token": "<|endoftext|>",
                  "eos_token": "<|endoftext|>"
                }
                """,
                Encoding.UTF8);

            HfTokenizerSpec? spec = HfLegacyBpeLoader.TryLoad(tempDir);
            Assert.NotNull(spec);

            // Even though added_tokens_decoder was empty, bos_token must surface
            // as a special (id 0) so the factory can pre-split it out.
            Assert.Contains(spec!.AddedTokens, t => t.Id == 0 && t.Content == "<|endoftext|>" && t.Special);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void TryLoad_SkipsCommentAndBlankLinesInMerges()
    {
        string tempDir = CreateEmptyTempDirectory();
        try
        {
            File.WriteAllText(
                Path.Combine(tempDir, "vocab.json"),
                "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3,\"bc\":4}",
                Encoding.UTF8);
            // Header, blank line, two merges — expect exactly 2 pairs.
            File.WriteAllText(
                Path.Combine(tempDir, "merges.txt"),
                "#version: 0.2\n\na b\nb c\n",
                Encoding.UTF8);

            HfTokenizerSpec? spec = HfLegacyBpeLoader.TryLoad(tempDir);
            Assert.NotNull(spec);
            Assert.Equal(2, spec!.Merges.Count);
            Assert.Equal(("a", "b"), spec.Merges[0]);
            Assert.Equal(("b", "c"), spec.Merges[1]);
        }
        finally
        {
            Directory.Delete(tempDir, recursive: true);
        }
    }

    // -------------------------------------------------------------------------
    // Real-checkpoint end-to-end tests (Granite-3.0 MoE).
    // -------------------------------------------------------------------------

    [SkippableFact]
    public void TryLoadFromDirectory_GraniteMoe_ReturnsNonNullTokenizer()
    {
        SkipIfLegacyMissing(GraniteMoeDir);

        ITokenizer? tok = HfBpeTokenizerFactory.TryLoadFromDirectory(GraniteMoeDir);
        Assert.NotNull(tok);
        Assert.True(tok!.VocabSize > 40_000, $"expected large vocab, got {tok.VocabSize}");
    }

    [SkippableFact]
    public void Encode_GraniteMoe_HelloWorld_RoundTrips()
    {
        SkipIfLegacyMissing(GraniteMoeDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(GraniteMoeDir)!;

        const string input = "Hello, world!";
        int[] ids = tok.Encode(input);

        Assert.NotEmpty(ids);
        foreach (int id in ids)
            Assert.InRange(id, 0, tok.VocabSize - 1);

        string decoded = tok.Decode(ids);
        // ByteLevel decode should be an exact round-trip for ASCII text.
        Assert.Equal(input, decoded);
    }

    [SkippableFact]
    public void Encode_GraniteMoe_NonTrivialPromptProducesMultipleTokens()
    {
        SkipIfLegacyMissing(GraniteMoeDir);
        ITokenizer tok = HfBpeTokenizerFactory.TryLoadFromDirectory(GraniteMoeDir)!;

        const string prompt = "The capital of France is";
        int[] ids = tok.Encode(prompt);

        Assert.True(ids.Length > 1, $"expected >1 tokens, got {ids.Length}");

        // Sanity: at least one token must decode to a non-empty non-special slice.
        bool anyNormal = false;
        foreach (int id in ids)
        {
            string piece = tok.DecodeToken(id);
            if (!string.IsNullOrEmpty(piece))
            {
                anyNormal = true;
                break;
            }
        }
        Assert.True(anyNormal, "expected at least one non-empty decoded piece");

        // Full-string decode round-trip (ByteLevel is lossless for ASCII).
        Assert.Equal(prompt, tok.Decode(ids));
    }

    // -------------------------------------------------------------------------
    // Helpers.
    // -------------------------------------------------------------------------

    private static string CreateEmptyTempDirectory()
    {
        string dir = Path.Combine(
            Path.GetTempPath(),
            "dotllm-legacy-bpe-" + Path.GetRandomFileName());
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void SkipIfLegacyMissing(string dir)
    {
        bool hasTrio =
            File.Exists(Path.Combine(dir, "vocab.json")) &&
            File.Exists(Path.Combine(dir, "merges.txt"));
        Skip.IfNot(
            hasTrio,
            $"Legacy trio (vocab.json + merges.txt) not present at {dir} — skip.");
    }
}
