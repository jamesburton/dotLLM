using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The read-ahead window arithmetic behind the #874 load-time fix, as a pure CPU test. The properties that matter: the cursor
/// never requests past the tensor, never more than <c>window</c> bytes ahead of the consumer, covers every byte exactly once,
/// and never re-requests bytes the consumer has already read.
/// </summary>
public class VulkanReadAheadCursorTests
{
    private const long MiB = 1L << 20;

    private static List<(long Off, long Len)> Drain(ref VulkanReadAheadCursor c, long consumed)
    {
        var l = new List<(long, long)>();
        while (c.TryNext(consumed, out long o, out long n)) l.Add((o, n));
        return l;
    }

    [Fact]
    public void FirstCall_FillsExactlyTheWindow_InPieces()
    {
        var c = new VulkanReadAheadCursor(total: 10_000 * MiB, window: 100 * MiB, piece: 16 * MiB);
        var r = Drain(ref c, consumed: 0);
        Assert.Equal(100 * MiB, r.Sum(x => x.Len));
        Assert.Equal(7, r.Count);                       // 6 x 16 MiB + 4 MiB
        Assert.All(r.Take(6), x => Assert.Equal(16 * MiB, x.Len));
        Assert.Equal(4 * MiB, r[^1].Len);
        Assert.Equal(0, r[0].Off);
    }

    [Fact]
    public void WindowSlidesWithTheConsumer_AndCoversEveryByteExactlyOnce()
    {
        const long total = 1000 * MiB + 12345;          // deliberately not piece-aligned
        var c = new VulkanReadAheadCursor(total, window: 64 * MiB, piece: 16 * MiB);
        var all = new List<(long Off, long Len)>();
        for (long consumed = 0; consumed < total; consumed += 32 * MiB)
        {
            var step = Drain(ref c, consumed);
            Assert.All(step, x => Assert.True(x.Off + x.Len <= Math.Min(total, consumed + 64 * MiB), "requested beyond the window"));
            all.AddRange(step);
        }
        all.Sort((a, b) => a.Off.CompareTo(b.Off));
        long cursor = 0;
        foreach (var (off, len) in all) { Assert.Equal(cursor, off); cursor += len; }   // contiguous, no gaps, no overlap
        Assert.Equal(total, cursor);
    }

    [Fact]
    public void NeverRequestsPastTheEnd_OrBehindTheConsumer()
    {
        var c = new VulkanReadAheadCursor(total: 50 * MiB, window: 1024 * MiB, piece: 16 * MiB);
        Assert.Equal(50 * MiB, Drain(ref c, 0).Sum(x => x.Len));
        Assert.Empty(Drain(ref c, 10 * MiB));            // already covered
        // A consumer that raced ahead of what was requested (queue was full / dropped) must not trigger stale requests.
        var d = new VulkanReadAheadCursor(total: 1000 * MiB, window: 32 * MiB, piece: 16 * MiB);
        var r = Drain(ref d, consumed: 500 * MiB);
        Assert.All(r, x => Assert.True(x.Off >= 500 * MiB));
        Assert.Equal(32 * MiB, r.Sum(x => x.Len));
    }

    [Fact]
    public void ZeroWindow_RequestsNothing()
    {
        var c = new VulkanReadAheadCursor(100 * MiB, window: 0, piece: 16 * MiB);
        Assert.Empty(Drain(ref c, 0));
    }

    [Fact]
    public void Applies_OnlyToLargeTensors()
    {
        Assert.False(VulkanWeightReadAhead.Applies(VulkanWeightReadAhead.MinTensorBytes - 1));
        Assert.Equal(VulkanWeightReadAhead.WindowBytes > 0, VulkanWeightReadAhead.Applies(VulkanWeightReadAhead.MinTensorBytes));
    }

    [Fact]
    public void Pump_OnAnonymousMemory_IsHarmlessAndCountsRequestedBytes()
    {
        // Read-ahead is advisory: on ordinary committed memory the workers just touch pages; no byte changes.
        unsafe
        {
            const long len = 24 * MiB;
            byte* buf = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)len, 4096);
            try
            {
                new Span<byte>(buf, (int)len).Fill(0x5A);
                long before = VulkanWeightReadAhead.RequestedBytes;
                var cur = VulkanWeightReadAhead.Begin(len);
                VulkanWeightReadAhead.Pump(ref cur, (nint)buf, 0);
                Assert.True(VulkanWeightReadAhead.RequestedBytes - before <= len);
                Thread.Sleep(100);
                for (long i = 0; i < len; i += 4096) Assert.Equal(0x5A, buf[i]);
            }
            finally { System.Runtime.InteropServices.NativeMemory.AlignedFree(buf); }
        }
    }
}

/// <summary>#874: the guard that keeps ahead-of-order bank allocation from changing which heap anything lands on.</summary>
public class VulkanBankPreallocPlacementGuardTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void AllowsAheadAllocation_WhenEverythingStillFitsWithMargin()
        => Assert.True(VulkanBankPrealloc.MayAllocateAhead(heapBytes: 68 * GiB, liveBytes: 20 * GiB, pendingBytes: 3 * GiB, need: 1 * GiB, margin: 3 * GiB));

    [Fact]
    public void Refuses_WhenLivePlusPendingPlusNeedEntersTheMargin()
    {
        // 68 - 3 margin = 65 usable: 60 live + 3 pending + 2 need = 65 still fits; 66 does not.
        Assert.True(VulkanBankPrealloc.MayAllocateAhead(68 * GiB, 60 * GiB, 3 * GiB, 2 * GiB, 3 * GiB));
        Assert.False(VulkanBankPrealloc.MayAllocateAhead(68 * GiB, 60 * GiB, 3 * GiB, 2 * GiB + 1, 3 * GiB));
    }

    [Fact]
    public void Refuses_OnceAFallbackHasInflatedLiveBytes()
        => Assert.False(VulkanBankPrealloc.MayAllocateAhead(68 * GiB, 80 * GiB, 0, 1 * GiB, 3 * GiB));
}
