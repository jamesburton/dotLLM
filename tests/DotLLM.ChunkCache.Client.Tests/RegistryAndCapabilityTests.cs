using System.Net;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="ChunkSpanRegistry"/> and <see cref="ChunkCapabilityCache"/>.</summary>
public sealed class RegistryAndCapabilityTests
{
    private static ChunkSpan[] Whole(string text) => [new ChunkSpan("id", ChunkHash.Of(text))];

    private static string Long(char c, int n = 64) => new(c, n);

    [Fact]
    public void Registry_ExactAndPrefixLookup_LongestPrefixWins()
    {
        var r = new ChunkSpanRegistry();
        string a = Long('a');
        string ab = a + Long('b');
        r.Register(a, Whole(a));
        r.Register(ab, Whole(ab));

        Assert.True(r.TryGet(a, out IReadOnlyList<ChunkSpan> s1));
        Assert.Equal(ChunkHash.Of(a), s1[0].Hash);
        Assert.False(r.TryGet(ab + "suffix", out _));

        Assert.True(r.TryFind(ab + "suffix", out string? t, out IReadOnlyList<ChunkSpan> s2));
        Assert.Equal(ab, t);
        Assert.Equal(ChunkHash.Of(ab), s2[0].Hash);
        Assert.True(r.TryFind(a + "zzz", out t, out _));
        Assert.Equal(a, t);
        Assert.False(r.TryFind("zzz" + a, out _, out _)); // a prefix, not a substring
    }

    [Fact]
    public void Registry_ShortTexts_ExactOnly()
    {
        var r = new ChunkSpanRegistry();
        r.Register("short", Whole("short"));
        Assert.True(r.TryFind("short", out _, out _));
        Assert.False(r.TryFind("short and more", out _, out _));
    }

    [Fact]
    public void Registry_LeastRecentlyUsedEvicted_AtCapacity()
    {
        var r = new ChunkSpanRegistry(capacity: 3);
        string a = Long('a'), b = Long('b'), c = Long('c'), d = Long('d');
        r.Register(a, Whole(a));
        r.Register(b, Whole(b));
        r.Register(c, Whole(c));
        Assert.True(r.TryGet(a, out _)); // touch a: b is now the least recently used
        r.Register(d, Whole(d));

        Assert.Equal(3, r.Count);
        Assert.False(r.TryGet(b, out _));
        Assert.False(r.TryFind(b + "x", out _, out _)); // gone from the prefix index too
        Assert.True(r.TryGet(a, out _));
        Assert.True(r.TryGet(d, out _));
    }

    /// <summary>Review S5: a total-chars budget, evicting least recently used first; an oversized text is not kept.</summary>
    [Fact]
    public void Registry_CharBudget_EvictsLeastRecentlyUsed()
    {
        var r = new ChunkSpanRegistry(capacity: 100, maxTotalChars: 200);
        string a = Long('a', 80), b = Long('b', 80), c = Long('c', 80);
        r.Register(a, Whole(a));
        r.Register(b, Whole(b));
        Assert.True(r.TryGet(a, out _)); // b is now the least recently used
        r.Register(c, Whole(c));          // 240 > 200: evict b

        Assert.Equal(160, r.TotalChars);
        Assert.False(r.TryGet(b, out _));
        Assert.True(r.TryGet(a, out _));
        Assert.True(r.TryGet(c, out _));

        string huge = Long('h', 201);
        r.Register(huge, Whole(huge));
        Assert.False(r.TryGet(huge, out _));
        Assert.Equal(160, r.TotalChars);  // nothing else was evicted for it

        r.Register(a, []);
        Assert.Equal(80, r.TotalChars);
        Assert.Equal(32L * 1024 * 1024, ChunkSpanRegistry.DefaultMaxTotalChars);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ChunkSpanRegistry(10, 0));
    }

    [Fact]
    public void Registry_DefaultCapacity_Is4096()
    {
        var r = new ChunkSpanRegistry();
        for (int i = 0; i < 4100; i++)
        {
            string t = "text " + i;
            r.Register(t, Whole(t));
        }

        Assert.Equal(ChunkSpanRegistry.DefaultCapacity, r.Count);
        Assert.Equal(4096, ChunkSpanRegistry.DefaultCapacity);
        Assert.False(r.TryGet("text 0", out _));
        Assert.True(r.TryGet("text 4099", out _));
    }

    [Fact]
    public void Registry_ReRegisterReplaces_EmptyRemoves_BadRangeThrows()
    {
        var r = new ChunkSpanRegistry();
        string a = Long('a');
        r.Register(a, Whole(a));
        r.Register(a, [new ChunkSpan("half", ChunkHash.Of(a[..32]), 0, 32)]);
        Assert.True(r.TryGet(a, out IReadOnlyList<ChunkSpan> s));
        Assert.Equal("half", Assert.Single(s).Id);

        r.Register(a, []);
        Assert.False(r.TryGet(a, out _));
        Assert.Equal(0, r.Count);

        Assert.Throws<ArgumentOutOfRangeException>(() => r.Register(a, [new ChunkSpan("x", "h", 10, 65)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => r.Register(a, [new ChunkSpan("x", "h", -1, 5)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => r.Register(a, [new ChunkSpan("x", "h", 20, 10)]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ChunkSpanRegistry(0));
    }

    [Fact]
    public async Task Registry_ConcurrentUse_Consistent()
    {
        var r = new ChunkSpanRegistry(capacity: 64);
        await Task.WhenAll(Enumerable.Range(0, 8).Select(w => Task.Run(() =>
        {
            for (int i = 0; i < 2000; i++)
            {
                string t = $"worker {w} item {i % 100} " + Long('x', 40);
                r.Register(t, Whole(t));
                if (r.TryFind(t + "!", out string? found, out IReadOnlyList<ChunkSpan> spans))
                    Assert.Equal(ChunkHash.Of(found), spans[0].Hash);
            }
        })));
        Assert.Equal(64, r.Count);
    }

    // ---- capability cache ------------------------------------------------------------------------------------------

    [Fact]
    public async Task Cache_OneProbe_ForManyCalls_UntilTtl()
    {
        var time = new ManualTime();
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe, successTtl: TimeSpan.FromMinutes(5), timeProvider: time);

        for (int i = 0; i < 10; i++)
            Assert.NotNull(await cache.GetAsync("m", CancellationToken.None));
        Assert.Equal(1, probe.Calls);

        time.Advance(TimeSpan.FromMinutes(6));
        Assert.NotNull(await cache.GetAsync("m", CancellationToken.None));
        Assert.Equal(2, probe.Calls);
    }

    [Fact]
    public async Task Cache_ModelChange_Reprobes_SameModelDoesNot()
    {
        var probe = new ScriptedProbe((_, _) => Task.FromResult<ChunkCacheCapabilities?>(Pipeline.Caps()));
        var cache = new ChunkCapabilityCache(probe);

        await cache.GetAsync("a", CancellationToken.None);
        await cache.GetAsync("a", CancellationToken.None);
        await cache.GetAsync("b", CancellationToken.None);
        await cache.GetAsync(null, CancellationToken.None); // no model in the body: keep the last answer
        Assert.Equal(2, probe.Calls);

        cache.Invalidate();
        await cache.GetAsync("b", CancellationToken.None);
        Assert.Equal(3, probe.Calls);
    }

    /// <summary>Ruling 16: 401, 403, 5xx, transport errors and 404 all mean "not capable"; failures are cached for the failure TTL.</summary>
    [Theory]
    [InlineData(401)]
    [InlineData(403)]
    [InlineData(500)]
    [InlineData(0)]
    [InlineData(404)]
    public async Task Cache_ProbeFailure_IsNull_CachedForFailureTtl(int status)
    {
        var time = new ManualTime();
        var probe = new ScriptedProbe((_, _) => status switch
        {
            404 => Task.FromResult<ChunkCacheCapabilities?>(null),
            0 => throw new HttpRequestException("refused"),
            _ => throw new ChunkCacheRequestException((HttpStatusCode)status, null, null, "no"),
        });
        var cache = new ChunkCapabilityCache(probe, failureTtl: TimeSpan.FromSeconds(30), timeProvider: time);

        Assert.Null(await cache.GetAsync("m", CancellationToken.None));
        Assert.Null(await cache.GetAsync("m", CancellationToken.None));
        Assert.Equal(1, probe.Calls);
        Assert.False(cache.LastResult!.IsCapable);

        time.Advance(TimeSpan.FromSeconds(31));
        Assert.Null(await cache.GetAsync("m", CancellationToken.None));
        Assert.Equal(2, probe.Calls);
    }

    [Fact]
    public async Task Cache_ProbeTimeout_IsNull_WithinTimeout()
    {
        var probe = new ScriptedProbe(async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct).ConfigureAwait(false);
            return Pipeline.Caps();
        });
        var cache = new ChunkCapabilityCache(probe, probeTimeout: TimeSpan.FromMilliseconds(150));

        var sw = System.Diagnostics.Stopwatch.StartNew();
        Assert.Null(await cache.GetAsync("m", CancellationToken.None).AsTask().WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), sw.Elapsed.ToString());
        Assert.IsAssignableFrom<OperationCanceledException>(cache.LastResult!.Failure);
    }

    [Fact]
    public async Task Cache_ConcurrentCallers_ShareOneProbe()
    {
        var gate = new TaskCompletionSource();
        var probe = new ScriptedProbe(async (_, _) =>
        {
            await gate.Task.ConfigureAwait(false);
            return Pipeline.Caps();
        });
        var cache = new ChunkCapabilityCache(probe);

        Task<ChunkCacheCapabilities?>[] calls = Enumerable.Range(0, 8).Select(_ => cache.GetAsync("m", CancellationToken.None).AsTask()).ToArray();
        gate.SetResult();
        ChunkCacheCapabilities?[] results = await Task.WhenAll(calls);

        Assert.All(results, Assert.NotNull);
        Assert.Equal(1, probe.Calls);
    }

    [Fact]
    public async Task Cache_CallerCancellation_Propagates()
    {
        var probe = new ScriptedProbe(async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct).ConfigureAwait(false);
            return Pipeline.Caps();
        });
        var cache = new ChunkCapabilityCache(probe, probeTimeout: TimeSpan.FromSeconds(30));
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cache.GetAsync("m", cts.Token).AsTask());
    }

    [Fact]
    public async Task Fixed_ReturnsItsValue()
    {
        ChunkCacheCapabilities caps = Pipeline.Caps();
        Assert.Same(caps, await new FixedChunkCapabilities(caps).GetAsync("m", CancellationToken.None));
        Assert.Null(await new FixedChunkCapabilities(null).GetAsync("m", CancellationToken.None));
    }
}
