using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Scheduler;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #1266: per-request token streaming from the continuous-batch scheduler. The streamed text must
/// concatenate to exactly the <see cref="InferenceResponse.Text"/> the same request returns without
/// streaming, whatever ended it, and a stream must survive cancellation and preemption cleanly.
/// </summary>
public sealed partial class ContinuousBatchSchedulerTests
{
    // Byte-level vocabulary: "€" is E2 82 AC, split across tokens 10 and 11, so the text decodes to
    // U+FFFD between them. 0 (EOS) has text of its own, which a token-level stop must not stream.
    private static readonly Dictionary<int, byte[]> StreamVocab = new()
    {
        [EosTokenId] = "</s>"u8.ToArray(),
        [9] = "a"u8.ToArray(),
        [10] = [0xE2],
        [11] = [0x82, 0xAC],
        [12] = "X"u8.ToArray(),
        [13] = "Y"u8.ToArray(),
        [14] = "z"u8.ToArray(),
    };

    public static TheoryData<int[], int, string[], FinishReason> StreamEndings => new()
    {
        // Stop string "XY" spanning tokens, after an "X" that turned out not to start it.
        { [9, 10, 11, 12, 9, 12, 13, 14, 14], 9, ["XY"], FinishReason.Stop },
        // EOS: the dropped token's text ("</s>") is not part of the response.
        { [9, 10, 11, 12, EosTokenId, 14], 9, [], FinishReason.Stop },
        // Length, ending on the first half of a split character: flushed, not withheld forever.
        { [9, 10, 11, 9, 10, 11], 5, ["XY"], FinishReason.Length },
    };

    [Theory]
    [MemberData(nameof(StreamEndings))]
    public async Task Stream_ConcatenatesToCompletionText(int[] script, int maxTokens, string[] stops, FinishReason expected)
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Sequence(script), tokenizer: new ByteTokenizer(StreamVocab));
        var channel = NewStream();
        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens, stopSequences: stops), default, channel.Writer);
        DriveUntilIdle(fix.Scheduler);

        var response = await handle.Completion;
        var tokens = Drain(channel);

        Assert.Equal(expected, response.FinishReason);
        Assert.Equal(response.Text, string.Concat(tokens.Select(t => t.Text)));
        // Mid-stream, an incomplete character is held back; only the final flush may carry one.
        Assert.All(tokens[..^1], t => Assert.DoesNotContain('�', t.Text));
        Assert.True(channel.Reader.Completion.IsCompleted);

        // Exactly one final token, last, carrying the outcome.
        Assert.Single(tokens, t => t.FinishReason is not null);
        var last = tokens[^1];
        Assert.Equal(expected, last.FinishReason);
        Assert.Equal(response.GeneratedTokenCount, last.GeneratedTokenCount);
        Assert.Equal(response.Timings, last.Timings);
    }

    [Fact]
    public async Task Stream_SplitCharacter_IsEmittedOnlyOnceComplete()
    {
        using var fix = new TestFixture(
            tokenScript: TokenScript.Sequence([9, 10, 11, 9, EosTokenId]), tokenizer: new ByteTokenizer(StreamVocab));
        var channel = NewStream();
        var handle = fix.Scheduler.Submit(MakeRequest(promptLen: 2, maxTokens: 8), default, channel.Writer);
        DriveUntilIdle(fix.Scheduler);
        await handle.Completion;

        // One token per generated id; the first byte of "€" carries no text, the second carries all of it.
        Assert.Equal(["a", "", "€", "a", ""], Drain(channel).Select(t => t.Text));
    }

    [Fact]
    public async Task Stream_CancelledMidStream_EndsTheStream_AndReturnsEveryBlock()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(tokenId: 5));
        using var cts = new CancellationTokenSource();
        var channel = NewStream();
        var handle = fix.Scheduler.Submit(MakeRequest(2, maxTokens: 40), cts.Token, channel.Writer);

        fix.Scheduler.Step();
        fix.Scheduler.Step();
        Assert.NotEmpty(Drain(channel));

        cts.Cancel();
        // The cancellation callback ends the stream at once, before the loop sweeps the sequence.
        Assert.True(channel.Reader.Completion.IsCompleted);
        fix.Scheduler.Step();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await handle.Completion);
        Assert.True(fix.Scheduler.IsIdle);
        Assert.Equal(fix.PagedPool.TotalBlocks, fix.PagedPool.FreeBlocks);
    }

    [Fact]
    public async Task Stream_PreemptedAndResumed_EmitsNoDuplicates()
    {
        // Same pressure setup as Preemption_ResumedSequence_MatchesUnpreemptedOutput.
        using var fix = new TestFixture(
            options: new ContinuousBatchSchedulerOptions
            {
                MaxActiveSequences = 4,
                ReserveBlocksPerSequence = 3,
                EnablePreemption = true,
            },
            totalBlocks: 3,
            inputEmitter: Ramp);

        var channel = NewStream();
        var low = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.Low), default, channel.Writer);
        fix.Scheduler.Step();
        Assert.Equal(SequenceState.Decoding, low.State);

        var high = fix.Scheduler.Submit(MakeRequest(promptLen: 4, maxTokens: 16, priority: RequestPriority.High));
        fix.Scheduler.Step();
        Assert.Equal(1L, fix.Scheduler.GetMetrics().PreemptionCount);

        DriveUntilCompleted(fix.Scheduler, high);
        DriveUntilCompleted(fix.Scheduler, low);

        var response = await low.Completion;
        var tokens = Drain(channel);

        // Ramp 5..9 then EOS: each generated id streamed once, then the final (EOS) token.
        Assert.Equal([5, 6, 7, 8, 9], response.GeneratedTokenIds);
        Assert.Equal([5, 6, 7, 8, 9, EosTokenId], tokens.Select(t => t.TokenId));
        Assert.Equal(response.Text, string.Concat(tokens.Select(t => t.Text)));
        Assert.Equal(fix.PagedPool.TotalBlocks, fix.PagedPool.FreeBlocks);
    }

    [Fact]
    public async Task TwoStreams_Interleave()
    {
        using var fix = new TestFixture(tokenScript: TokenScript.Constant(EosTokenId, afterNTokens: 6));
        var a = NewStream();
        var b = NewStream();
        var ha = fix.Scheduler.Submit(MakeRequest(promptLen: 3, maxTokens: 16), default, a.Writer);
        var hb = fix.Scheduler.Submit(MakeRequest(promptLen: 5, maxTokens: 16), default, b.Writer);

        fix.Scheduler.Step();
        fix.Scheduler.Step();

        // Both have streamed tokens while neither has finished: the streams share the batch.
        Assert.False(ha.Completion.IsCompleted);
        Assert.False(hb.Completion.IsCompleted);
        Assert.True(Drain(a).Count >= 2);
        Assert.True(Drain(b).Count >= 2);

        DriveUntilIdle(fix.Scheduler);
        Assert.Equal(6, (await ha.Completion).GeneratedTokenCount);
        Assert.Equal(6, (await hb.Completion).GeneratedTokenCount);
    }

    private static Channel<GenerationToken> NewStream() =>
        Channel.CreateUnbounded<GenerationToken>(new UnboundedChannelOptions { SingleReader = true });

    private static List<GenerationToken> Drain(Channel<GenerationToken> channel)
    {
        var tokens = new List<GenerationToken>();
        while (channel.Reader.TryRead(out var t)) tokens.Add(t);
        return tokens;
    }

    /// <summary>Byte-level tokenizer: each id maps to raw bytes, and a sequence decodes as UTF-8,
    /// so a character split across tokens decodes to U+FFFD until its last byte arrives.</summary>
    private sealed class ByteTokenizer(Dictionary<int, byte[]> vocab) : ITokenizer
    {
        public int VocabSize => ContinuousBatchSchedulerTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => ContinuousBatchSchedulerTests.EosTokenId;
        public int[] Encode(string text) => [];

        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var bytes = new List<byte>();
            foreach (int id in tokenIds) bytes.AddRange(vocab.TryGetValue(id, out var b) ? b : Encoding.UTF8.GetBytes($"<{id}>"));
            return Encoding.UTF8.GetString(bytes.ToArray());
        }

        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => Decode([tokenId]);
        public int CountTokens(string text) => 0;
    }
}
