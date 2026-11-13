using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Issue #1702: a live <see cref="PrefixCache"/> entry stores prompt + every generated token, but a request that stops on its
/// length limit never forwards its last sampled token, so the entry's KV holds one row fewer than its token sequence. A next
/// prompt that continues the reply must not reuse that unwritten row: it holds a stale row of whatever request used the cache
/// before. Runs the deterministic random-weight qwen3 of <see cref="PrefixCacheFailedRequestTests.TinyQwen"/> on the CPU.
/// </summary>
public sealed class PrefixResumeUnwrittenRowTests : IClassFixture<PrefixCacheFailedRequestTests.TinyQwen>
{
    private const int MaxTokens = 6;
    private const int Shared = 4;
    private readonly PrefixCacheFailedRequestTests.TinyQwen _tiny;
    private readonly ITestOutputHelper _out;

    public PrefixResumeUnwrittenRowTests(PrefixCacheFailedRequestTests.TinyQwen tiny, ITestOutputHelper output)
    {
        _tiny = tiny;
        _out = output;
    }

    private static int[] Ids(int count, int seed)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = 2 + (seed * 7 + i * 13) % (PrefixCacheFailedRequestTests.TinyQwen.Vocab - 2);
        return ids;
    }

    private static string Text(int[] ids) => string.Join(",", ids.Select(i => i.ToString(CultureInfo.InvariantCulture)));

    private static InferenceOptions Opts() => new() { Temperature = 0f, MaxTokens = MaxTokens, Logprobs = true, TopLogprobs = 1 };

    [Fact]
    public void ResumeAfterLengthStoppedReply_SkipsTheUnwrittenRow_AndMatchesAColdRun()
    {
        // X writes rows 0..X+MaxTokens-2 of a cache; A shares X's first tokens, so it takes X's cache and leaves X's K/V in
        // the row after its own last written row. C continues A's reply, so it matches A's whole stored sequence.
        int[] promptA = Ids(20, seed: 3);
        int[] promptX = [.. promptA[..Shared], .. Ids(26, seed: 11)];
        var tokenizer = new IdsTokenizer(PrefixCacheFailedRequestTests.TinyQwen.Vocab);
        using var prefix = new PrefixCache(maxEntries: 1);
        var gen = new TextGenerator(_tiny.Model, tokenizer, prefixCache: prefix, prefixCacheHeadroomTokens: 16);

        gen.Generate(Text(promptX), Opts());
        var a = gen.Generate(Text(promptA), Opts());
        Assert.Equal(Shared, a.Timings.CachedTokenCount);
        Assert.Equal(FinishReason.Length, a.FinishReason);
        int[] promptC = [.. promptA, .. a.GeneratedTokenIds, .. Ids(4, seed: 5)];
        var c = gen.Generate(Text(promptC), Opts());

        var cold = new TextGenerator(_tiny.Model, tokenizer).Generate(Text(promptC), Opts());
        _out.WriteLine($"C: cached {c.Timings.CachedTokenCount}, first logprob resumed {c.Logprobs![0].Logprob:R} vs cold {cold.Logprobs![0].Logprob:R}; " +
                       $"ids resumed [{Text(c.GeneratedTokenIds)}] cold [{Text(cold.GeneratedTokenIds)}]");

        // A's last sampled token was never forwarded: only promptA + MaxTokens - 1 rows hold K/V.
        Assert.Equal(promptA.Length + MaxTokens - 1, c.Timings.CachedTokenCount);
        Assert.Equal(cold.GeneratedTokenIds, c.GeneratedTokenIds);
        Assert.Equal(cold.Logprobs![0].Logprob, c.Logprobs![0].Logprob, 1e-4f);
    }

    /// <summary>Prompts are comma-separated token ids; no EOS, so every request generates <see cref="MaxTokens"/> tokens.</summary>
    private sealed class IdsTokenizer(int vocabSize) : ITokenizer
    {
        public int VocabSize => vocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => -1;
        public int[] Encode(string text) => text.Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray();
        public int CountTokens(string text) => Encode(text).Length;
        public string Decode(ReadOnlySpan<int> ids) => string.Join(",", ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool stripBosSpace) => Decode(ids);
        public string DecodeToken(int id) => id.ToString(CultureInfo.InvariantCulture);
    }
}
