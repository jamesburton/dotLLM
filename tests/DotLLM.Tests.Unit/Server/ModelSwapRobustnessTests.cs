using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1023, the parts that need no model: the SSE keep-alive while a streaming request waits for the gate, the role chunk
/// sent only once the gate is held, the status object, and the free-memory device fit.
/// </summary>
[Collection("SequentialFileIO")]   // reads the process-global model-store env vars that SequentialFileIO tests mutate (#1395)
public sealed class ModelSwapRobustnessTests
{
    private static async IAsyncEnumerable<GenerationToken> Script(params string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(0, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    /// <summary>A response body that can be read while the keep-alive loop writes to it.</summary>
    private sealed class RecordingStream : MemoryStream
    {
        private readonly object _gate = new();

        public override void Write(byte[] buffer, int offset, int count) { lock (_gate) base.Write(buffer, offset, count); }

        public override void Write(ReadOnlySpan<byte> buffer) { lock (_gate) base.Write(buffer); }

        public override Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken ct) { Write(buffer, offset, count); return Task.CompletedTask; }

        public override ValueTask WriteAsync(ReadOnlyMemory<byte> buffer, CancellationToken ct = default) { Write(buffer.Span); return ValueTask.CompletedTask; }

        public string Text { get { lock (_gate) return Encoding.UTF8.GetString(ToArray()); } }
    }

    private static Task<string> StreamAsync(Func<Func<Task>, CancellationToken, Task> gate, TimeSpan? interval) =>
        StreamAsync(gate, interval, new RecordingStream());

    private static Task<string> StreamAsync(Func<Func<Task>, CancellationToken, Task> gate, TimeSpan? interval, RecordingStream body) =>
        StreamAsync(gate, interval, body, _ => Script("hi"));

    private static async Task<string> StreamAsync(Func<Func<Task>, CancellationToken, Task> gate, TimeSpan? interval, RecordingStream body,
        Func<CancellationToken, IAsyncEnumerable<GenerationToken>> tokens)
    {
        var ctx = new DefaultHttpContext();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, tokens, gate, "chatcmpl-1", "m",
            tools: null, toolParser: null, ReasoningPlan.Disabled, CancellationToken.None,
            waitReason: () => "loading owner/repo/x.gguf", keepAliveInterval: interval);
        return Encoding.UTF8.GetString(body.ToArray());
    }

    [Fact]
    public async Task QueuedStream_GetsKeepAliveComments_AndTheRoleChunkOnlyOnceTheGateIsHeld()
    {
        bool gateHeld = false;
        var body = new RecordingStream();

        // A gate that stays "busy" (another request generating, or a load) until two keep-alives have gone out - waiting on the
        // observable outcome rather than a fixed delay, so a starved thread pool under the full suite cannot make it flaky.
        async Task SlowGate(Func<Task> work, CancellationToken ct)
        {
            var deadline = DateTime.UtcNow.AddSeconds(30);
            while (body.Text.Split(": waiting for").Length - 1 < 2 && DateTime.UtcNow < deadline)
                await Task.Delay(10, ct);
            gateHeld = true;
            await work();
        }

        string raw = await StreamAsync(SlowGate, TimeSpan.FromMilliseconds(50), body);

        Assert.True(gateHeld);
        Assert.StartsWith(": waiting for loading owner/repo/x.gguf\n\n", raw);
        int lastComment = raw.LastIndexOf(": waiting for", StringComparison.Ordinal);
        int role = raw.IndexOf("\"role\":\"assistant\"", StringComparison.Ordinal);
        Assert.True(role > lastComment, "the role chunk (and its response id) must follow every keep-alive: it is sent only under the gate");
        Assert.True(raw.Split(": waiting for").Length - 1 >= 2, raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public async Task LongPrefill_GetsKeepAliveComments_BetweenTheRoleChunkAndTheFirstToken()
    {
        // Issue #1025: the role chunk goes out when the gate is taken, then the prefill runs before the first token. Here the
        // source blocks its caller SYNCHRONOUSLY, like the real prefill inside the first MoveNextAsync, until two comments
        // have been written after the role chunk. It waits on that observable outcome, not a fixed delay.
        var body = new RecordingStream();
        string raw = await StreamAsync((work, _) => work(), TimeSpan.FromMilliseconds(50), body,
            _ => SlowPrefill(body, "\"role\":\"assistant\""));

        int role = raw.IndexOf("\"role\":\"assistant\"", StringComparison.Ordinal);
        int firstComment = raw.IndexOf(": waiting for prompt processing", StringComparison.Ordinal);
        int lastComment = raw.LastIndexOf(": waiting for prompt processing", StringComparison.Ordinal);
        int content = raw.IndexOf("\"content\":\"hi\"", StringComparison.Ordinal);
        Assert.True(role >= 0 && firstComment > role, raw);
        Assert.True(content > lastComment, "no keep-alive may follow the first token: " + raw);
        Assert.True(raw.Split(": waiting for prompt processing").Length - 1 >= 2, raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    /// <summary>
    /// A token source whose first item blocks its caller SYNCHRONOUSLY, like the real prefill inside the first MoveNextAsync,
    /// until two prefill keep-alives have been written after <paramref name="marker"/>. It waits on that observable
    /// outcome, not a fixed delay, so a starved thread pool cannot make it flaky.
    /// </summary>
    private static async IAsyncEnumerable<GenerationToken> SlowPrefill(RecordingStream body, string marker, bool fail = false)
    {
        int CommentsAfterMarker()
        {
            string t = body.Text;
            int at = t.IndexOf(marker, StringComparison.Ordinal);
            return at < 0 ? 0 : t[at..].Split(": waiting for prompt processing").Length - 1;
        }

        var deadline = DateTime.UtcNow.AddSeconds(30);
        while (CommentsAfterMarker() < 2 && DateTime.UtcNow < deadline)
            Thread.Sleep(10);
        if (fail)
            throw new InvalidOperationException("prefill exploded");
        yield return new GenerationToken(0, "hi", FinishReason.Stop);
        await Task.CompletedTask;
    }

    [Fact]
    public async Task MessagesStream_LongPrefill_GetsKeepAliveComments_AfterMessageStart()
    {
        // PR #1026 review: the same prefill keep-alive on /v1/messages, at the interval it is given.
        var body = new RecordingStream();
        var ctx = new DefaultHttpContext { Response = { Body = body } };

        await MessagesEndpoint.WriteMessageStreamAsync(ctx, _ => SlowPrefill(body, "message_start"), (work, _) => work(),
            toolCallParser: null, requestStopSequences: null, messageId: "msg_1", modelId: "m", promptTokenCount: 3,
            CancellationToken.None, keepAliveInterval: TimeSpan.FromMilliseconds(50));

        string raw = body.Text;
        int start = raw.IndexOf("message_start", StringComparison.Ordinal);
        int firstComment = raw.IndexOf(": waiting for prompt processing", StringComparison.Ordinal);
        int lastComment = raw.LastIndexOf(": waiting for prompt processing", StringComparison.Ordinal);
        int delta = raw.IndexOf("content_block_delta", StringComparison.Ordinal);
        Assert.True(start >= 0 && firstComment > start, raw);
        Assert.True(delta > lastComment, "no keep-alive may follow the first token: " + raw);
        Assert.Contains("message_stop", raw);
    }

    [Fact]
    public async Task CompletionsStream_LongPrefill_GetsKeepAliveComments_BeforeTheFirstChunk()
    {
        var body = new RecordingStream();
        var ctx = new DefaultHttpContext { Response = { Body = body } };

        // No marker precedes the first keep-alive here (completions has no role chunk), so count from the start.
        await CompletionEndpoint.WriteCompletionStreamAsync(ctx, _ => SlowPrefill(body, ""), (work, _) => work(),
            "cmpl-1", "m", includeUsageChunk: false, CancellationToken.None, TimeSpan.FromMilliseconds(50));

        string raw = body.Text;
        Assert.StartsWith(": waiting for prompt processing", raw);
        Assert.True(raw.IndexOf("\"text\":\"hi\"", StringComparison.Ordinal) > raw.LastIndexOf(": waiting", StringComparison.Ordinal), raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public async Task CompletionsStream_FailureAfterAKeepAlive_IsReportedInStream_NotTruncated()
    {
        // PR #1026 review: a keep-alive commits the 200 + text/event-stream, so a later failure can no longer set a status.
        // It used to end the stream bare, which a client reads as a dropped connection.
        var body = new RecordingStream();
        var ctx = new DefaultHttpContext();
        // DefaultHttpContext never reports a started response; a real server does after the first byte.
        ctx.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpResponseFeature>(new StartsOnWriteFeature(body));
        ctx.Response.Body = body;

        await CompletionEndpoint.WriteCompletionStreamAsync(ctx, _ => SlowPrefill(body, "", fail: true), (work, _) => work(),
            "cmpl-1", "m", includeUsageChunk: false, CancellationToken.None, TimeSpan.FromMilliseconds(50));

        string raw = body.Text;
        Assert.StartsWith(": waiting for prompt processing", raw);
        Assert.True(raw.Contains("\"error\":{\"message\":\"prefill exploded\"", StringComparison.Ordinal), raw);
        Assert.Contains("prefill exploded", raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    private sealed class StartsOnWriteFeature(RecordingStream body) : Microsoft.AspNetCore.Http.Features.HttpResponseFeature
    {
        public override bool HasStarted => body.Length > 0;
    }

    [Fact]
    public async Task UncontendedStream_IsByteForByteFreeOfKeepAlives()
    {
        string raw = await StreamAsync((work, _) => work(), TimeSpan.FromSeconds(1));

        Assert.DoesNotContain(": waiting", raw);
        Assert.StartsWith("data: ", raw);
        Assert.Contains("\"role\":\"assistant\"", raw[..raw.IndexOf("\n\n", StringComparison.Ordinal)]);
    }

    [Fact]
    public async Task RoleChunk_IsNotSent_WhenTheGateNeverRunsTheWork()
    {
        // The gate gives up (e.g. the admitted model could not be re-activated): no role chunk / response id may have gone out.
        static Task FailingGate(Func<Task> work, CancellationToken ct) =>
            throw new ModelUnavailableException("m", "swapped out and reload failed");

        var ctx = new DefaultHttpContext { Response = { Body = new MemoryStream() } };
        await Assert.ThrowsAsync<ModelUnavailableException>(() => ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, _ => Script("hi"), FailingGate, "chatcmpl-1", "m",
            tools: null, toolParser: null, ReasoningPlan.Disabled, CancellationToken.None));

        Assert.Equal(0, ctx.Response.Body.Length);
    }

    [Fact]
    public async Task Status_ReportsNoModel_ThenBusyWithTheHolder_AndTheQueueDepth()
    {
        using var state = ServerStartup.CreateBareState(new ServerOptions { Model = "none", ModelId = "none" });
        Assert.Equal("no_model", state.GetStatus().State);

        var holding = new TaskCompletionSource();
        var release = new TaskCompletionSource();
        var first = state.ExecuteAsync(async () => { holding.SetResult(); await release.Task; }, CancellationToken.None);
        await holding.Task;

        var second = state.ExecuteAsync(() => Task.CompletedTask, CancellationToken.None);
        await WaitUntil(() => state.GetStatus().WaitingRequests == 1);

        ServerStatusResponse busy = state.GetStatus();
        Assert.Equal("busy", busy.State);
        Assert.Equal("generate", busy.BusyWith);
        Assert.NotNull(busy.BusySince);
        Assert.Equal(1, busy.WaitingRequests);
        Assert.Contains("busy: generate", state.DescribeWait());
        Assert.Contains("1 queued", state.DescribeWait());

        release.SetResult();
        await Task.WhenAll(first, second);
        var after = state.GetStatus();
        Assert.Equal("no_model", after.State);
        Assert.Null(after.BusyWith);
        Assert.Equal(0, after.WaitingRequests);
    }

    [Fact]
    public async Task AdmittedModelThatCannotBeReactivated_ThrowsModelUnavailable_InsteadOfRunningOnWhateverIsLoaded()
    {
        using var state = ServerStartup.CreateBareState(new ServerOptions { Model = "none", ModelId = "none" });
        bool ran = false;

        await Assert.ThrowsAsync<ModelUnavailableException>(() =>
            state.ExecuteAsync("owner/repo/gone-1023.gguf", _ => { ran = true; return Task.CompletedTask; }, CancellationToken.None));

        Assert.False(ran);
        Assert.Equal("no_model", state.GetStatus().State);   // the gate was released
    }

    private static async Task WaitUntil(Func<bool> condition)
    {
        for (int i = 0; i < 3000 && !condition(); i++) await Task.Delay(10);   // up to 30 s: the full suite can starve the pool
        Assert.True(condition());
    }

    // ── device fit on FREE memory ──

    private const long GiB = 1L << 30;

    private static DeviceInfoDto Gpu(long total, long? free) =>
        new() { Index = 0, Name = "RTX", DeviceString = "gpu:0", TotalMemoryBytes = total, FreeMemoryBytes = free };

    private static DeviceListResponse Devices(DeviceInfoDto gpu) => new()
    {
        Backends =
        [
            new BackendInfoDto { Name = "cpu", Available = true, Servable = true, DeviceCount = 1, Devices = [new DeviceInfoDto { Index = 0, Name = "cpu", DeviceString = "cpu" }] },
            new BackendInfoDto { Name = "cuda", Available = true, Servable = true, DeviceCount = 1, Devices = [gpu] },
        ],
    };

    [Fact]
    public void AutoFit_UsesFreeMemory_NotTotal()
    {
        // 12 GiB card, 8 GiB model: fits an empty card, but not one where another model still holds 9 GiB.
        Assert.True(DeviceSelector.FitsCuda(8 * GiB, Gpu(12 * GiB, free: 12 * GiB)));
        Assert.False(DeviceSelector.FitsCuda(8 * GiB, Gpu(12 * GiB, free: 3 * GiB)));
        Assert.Equal("cpu", DeviceSelector.Candidates(8 * GiB, Devices(Gpu(12 * GiB, free: 3 * GiB)))[0]);
        Assert.Equal("gpu:0", DeviceSelector.Candidates(8 * GiB, Devices(Gpu(12 * GiB, free: 12 * GiB)))[0]);
        // No free figure (NVML missing): the old total-based rule.
        Assert.True(DeviceSelector.FitsCuda(8 * GiB, Gpu(12 * GiB, free: null)));
    }

    [Fact]
    public void ExplicitGpu_WeightsLargerThanFreeMemory_IsAClearErrorUpFront_UnlessPartialOffload()
    {
        var spec = DeviceSpec.Parse("gpu");
        var full = DeviceSelector.Plan(spec, 20 * GiB, Devices(Gpu(12 * GiB, free: 11 * GiB)));
        Assert.Empty(full.Candidates);
        Assert.Contains("20.0 GiB of weights exceed the 11.0 GiB free on gpu:0", full.Error);

        var partial = DeviceSelector.Plan(spec, 20 * GiB, Devices(Gpu(12 * GiB, free: 11 * GiB)), fullOffload: false);
        Assert.Equal(["gpu:0"], partial.Candidates);

        // A tight fit that loads today is never refused: only the weights are compared, without headroom.
        Assert.Equal(["gpu:0"], DeviceSelector.Plan(spec, 10 * GiB, Devices(Gpu(12 * GiB, free: 11 * GiB))).Candidates);
    }
}
