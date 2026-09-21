using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #459. The scheduler accepted stop strings and never honoured them.
/// </summary>
/// <remarks>
/// <para>
/// <c>CheckStopAfterAppend</c> passed <c>ReadOnlySpan&lt;char&gt;.Empty</c> as the decoded tail —
/// self-labelled MVP — so every <c>StopStringCondition</c> was structurally unable to match, while
/// EOS and max-tokens (which read only the token id and the count) kept working. That is one root
/// cause for <b>both</b> reported symptoms: a user-supplied <c>stop</c> returned byte-identical
/// text, and <c>&lt;|eom_id|&gt;</c> ran on to <c>max_tokens</c> despite being registered in
/// <c>CommonStopSequences</c>. The built-in sequence is not special — it arrives as a
/// <c>StopStringCondition</c> like any other and died on the same empty tail.
/// </para>
/// <para>
/// <b>Scope, stated.</b> A stop string is detected when it is a <i>suffix</i> of the decoded text
/// at a token boundary. One that lands strictly inside a token (token text <c>"c!d"</c>, stop
/// <c>"!"</c>) is not detected; <c>TextGenerator</c> has the same boundary, so the two generation
/// paths agree. The last test pins that explicitly rather than leaving it to be discovered.
/// </para>
/// <para>
/// Declared <c>partial</c> over <c>ContinuousBatchSchedulerTests</c> to reuse its mock model,
/// paged-KV fixture and drive loop instead of standing up a second, subtly different harness.
/// </para>
/// </remarks>
public sealed partial class ContinuousBatchSchedulerTests
{
    /// <summary>
    /// A user-supplied stop string terminates generation mid-response. Against the pre-fix
    /// scheduler this runs to <c>maxTokens</c> and returns the untruncated text — the
    /// byte-identical output reported on the issue.
    /// </summary>
    [Fact]
    public async Task UserStopString_TerminatesGeneration_AndIsTrimmedFromTheText()
    {
        // Token 9 -> "ab", token 7 -> "c!". The stop string is "!".
        var tokenizer = new TextMapTokenizer(new Dictionary<int, string> { [9] = "ab", [7] = "c!" });

        // Two "ab" tokens, then the one carrying the stop string, then more that must never run.
        using var fix = new TestFixture(
            tokenScript: TokenScript.Sequence([9, 9, 7, 9, 9, 9, 9, 9]),
            tokenizer: tokenizer);

        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 8, stopSequences: ["!"]));
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;

        Assert.Equal(FinishReason.Stop, response.FinishReason);
        // The stop string is trimmed, but the rest of its token survives: "c" is real output.
        // Dropping the whole triggering token — the pre-fix Stop semantics — would give "abab".
        Assert.Equal("ababc", response.Text);
        // Stopped at the third generated token rather than running to maxTokens: 8.
        Assert.Equal(3, response.GeneratedTokenCount);
    }

    /// <summary>
    /// The Llama-3.2 symptom at scheduler level: <c>&lt;|eom_id|&gt;</c> registered exactly as the
    /// server registers it (a string in <c>CommonStopSequences</c>) must terminate generation.
    /// </summary>
    [Fact]
    public async Task BuiltInEomStopSequence_TerminatesGeneration()
    {
        var tokenizer = new TextMapTokenizer(
            new Dictionary<int, string> { [9] = "hello", [12] = "<|eom_id|>" });

        using var fix = new TestFixture(
            tokenScript: TokenScript.Sequence([9, 12, 9, 9, 9, 9]),
            tokenizer: tokenizer);

        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 6,
            stopSequences: ["<|eom_id|>", "<|eot_id|>"]));
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;

        Assert.Equal(FinishReason.Stop, response.FinishReason);
        Assert.Equal("hello", response.Text);
    }

    /// <summary>
    /// Negative control. A concurrent request that registers no stop string must be unaffected —
    /// it runs to max-tokens and keeps the text the other request's stop string cut. This is what
    /// shows the detokenizer is per-sequence and the fix cannot leak across a batch.
    /// </summary>
    [Fact]
    public async Task ConcurrentRequests_OnlyTheOneRegisteringTheStopStringStops()
    {
        var tokenizer = new TextMapTokenizer(new Dictionary<int, string> { [9] = "ab", [7] = "c!" });

        using var fix = new TestFixture(
            tokenScript: TokenScript.Sequence([9, 7, 9, 9]),
            tokenizer: tokenizer);

        var stopping = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4, stopSequences: ["!"]));
        var running = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 4));
        DriveUntilIdle(fix.Scheduler);

        var stopped = await stopping.Completion;
        var ran = await running.Completion;

        Assert.Equal(FinishReason.Stop, stopped.FinishReason);
        Assert.Equal("abc", stopped.Text);

        Assert.Equal(FinishReason.Length, ran.FinishReason);
        Assert.Contains("!", ran.Text, StringComparison.Ordinal);
    }

    /// <summary>
    /// The boundary this fix does not cross, pinned so it is a decision rather than a surprise: a
    /// stop string strictly interior to a token's text is not detected, because matching is
    /// <c>EndsWith</c> over the decoded tail at each token boundary. <c>TextGenerator</c> behaves
    /// the same way, so the two generation paths stay in agreement.
    /// </summary>
    [Fact]
    public async Task StopStringInteriorToAToken_IsNotDetected_MatchingTextGenerator()
    {
        var tokenizer = new TextMapTokenizer(new Dictionary<int, string> { [7] = "c!d" });

        using var fix = new TestFixture(
            tokenScript: TokenScript.Sequence([7, 7]),
            tokenizer: tokenizer);

        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 2, stopSequences: ["!"]));
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;

        Assert.Equal(FinishReason.Length, response.FinishReason);
        Assert.Equal("c!dc!d", response.Text);
    }

    /// <summary>
    /// Tokenizer whose decode is a literal per-token text map, so a test can place a stop string
    /// exactly where it needs it — at a token boundary, as a token suffix, or interior to a token.
    /// </summary>
    private sealed class TextMapTokenizer : ITokenizer
    {
        private readonly Dictionary<int, string> _texts;

        public TextMapTokenizer(Dictionary<int, string> texts) => _texts = texts;

        public int VocabSize => ContinuousBatchSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => ContinuousBatchSchedulerTests.EosTokenId;

        public int[] Encode(string text) => Array.Empty<int>();

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds)
                sb.Append(DecodeToken(id));
            return sb.ToString();
        }

        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);

        public string DecodeToken(int tokenId) =>
            _texts.TryGetValue(tokenId, out var t) ? t : tokenId.ToString(CultureInfo.InvariantCulture);

        public int CountTokens(string text) => 0;
    }
}
