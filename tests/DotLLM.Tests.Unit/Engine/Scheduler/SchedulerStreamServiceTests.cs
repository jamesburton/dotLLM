using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Xunit;

namespace DotLLM.Tests.Unit.Engine.Scheduler;

/// <summary>
/// #1266 at the service and server seams: a streaming request runs on the scheduler's run loop, so
/// it holds neither the request gate nor the model lease, and a slow client stalls nobody.
/// </summary>
public sealed partial class ContinuousBatchSchedulerServiceTests
{
    private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(10);

    [Fact]
    public async Task StreamAsync_StalledConsumer_DoesNotBlockTheLoopOrHoldTheLease()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 30);
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        await using var stream = fix.Service.StreamAsync(MakeRequest(promptLen: 3, maxTokens: 40)).GetAsyncEnumerator();
        Assert.True(await stream.MoveNextAsync());

        // The stream is mid-generation and its consumer is not reading. A batched request still
        // completes, and the model lease is free between steps.
        var other = await fix.Service.EnqueueAsync(MakeRequest(promptLen: 4, maxTokens: 40)).WaitAsync(Deadline);
        Assert.Equal(30, other.GeneratedTokenCount);
        using (var cts = new CancellationTokenSource(Deadline))
            (await fix.Service.AcquireModelAsync(cts.Token)).Dispose();

        int count = 1;
        GenerationToken last = stream.Current;
        while (await stream.MoveNextAsync()) { count++; last = stream.Current; }

        // 30 tokens, then the final (EOS) token carrying the authoritative count.
        Assert.Equal(31, count);
        Assert.Equal(FinishReason.Stop, last.FinishReason);
        Assert.Equal(30, last.GeneratedTokenCount);

        loopCts.Cancel();
        await loopTask;
    }

    [Fact]
    public async Task StreamAsync_AbandonedEnumeration_CancelsTheRequest_AndFreesItsBlocks()
    {
        using var fix = new ServiceFixture(emitToken: 5, afterNTokens: 1000);   // never stops on its own before max_tokens
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        await foreach (var _ in fix.Service.StreamAsync(MakeRequest(promptLen: 3, maxTokens: 50)))
            break;

        using (var cts = new CancellationTokenSource(Deadline))
        {
            while (!fix.Service.Inner.IsIdle)
                await Task.Delay(5, cts.Token);
        }
        Assert.Equal(fix.PagedFactory.Pool.TotalBlocks, fix.PagedFactory.Pool.FreeBlocks);

        loopCts.Cancel();
        await loopTask;
    }

    /// <summary>
    /// The server's scheduler token source (what the streaming endpoints use when a scheduler runs)
    /// with a client that stops reading mid-response: a concurrent non-streaming request still
    /// completes, and the usage chunk reports the scheduler's count.
    /// </summary>
    [Fact]
    public async Task SchedulerStream_CompletionsEndpoint_StalledClient_DoesNotBlockBatchedRequests()
    {
        using var fix = new ServiceFixture(emitToken: EosTokenId, afterNTokens: 12);
        using var loopCts = new CancellationTokenSource();
        var loopTask = Task.Run(() => fix.Service.RunLoopAsync(loopCts.Token));

        var body = new StallingStream();
        var ctx = new DefaultHttpContext();
        ctx.Features.Set<IHttpResponseBodyFeature>(new StreamResponseBodyFeature(body));
        var request = MakeRequest(promptLen: 3, maxTokens: 40);

        Task streaming = CompletionEndpoint.WriteCompletionStreamAsync(
            ctx,
            ServerState.SchedulerStream(fix.Service, ctx, request.TokenIds, request.Options),
            ServerState.WithoutGate,
            "cmpl-1", "m", includeUsageChunk: true, CancellationToken.None);

        await body.Stalled.WaitAsync(Deadline);
        var other = await fix.Service.EnqueueAsync(MakeRequest(promptLen: 4, maxTokens: 40)).WaitAsync(Deadline);
        Assert.Equal(12, other.GeneratedTokenCount);
        Assert.False(streaming.IsCompleted);

        body.Release();
        await streaming.WaitAsync(Deadline);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("\"completion_tokens\":12", raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);

        loopCts.Cancel();
        await loopTask;
    }

    /// <summary>A response body that blocks every write after the first until released.</summary>
    private sealed class StallingStream : MemoryStream
    {
        private readonly TaskCompletionSource _stalled = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource _release = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task Stalled => _stalled.Task;

        public void Release() => _release.TrySetResult();

        public override async ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await GateAsync();
            await base.WriteAsync(buffer, cancellationToken);
        }

        public override async Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
        {
            await GateAsync();
            await base.WriteAsync(buffer, offset, count, cancellationToken);
        }

        private Task GateAsync()
        {
            if (Length == 0) return Task.CompletedTask;
            _stalled.TrySetResult();
            return _release.Task;
        }
    }
}
