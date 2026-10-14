using System.Runtime.InteropServices;
using DotLLM.Vulkan.Tiering;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan.Tiering;

/// <summary>
/// The tiering miss-path reader (#939): the bytes it delivers must be exactly the file's, at arbitrary (unaligned) expert offsets, with the page-cache-bypassing
/// Windows path and the portable fallback alike.
/// </summary>
public sealed unsafe class UnbufferedFileReaderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-tier-reader-" + Guid.NewGuid().ToString("N"));
    private readonly string _a, _b;
    private readonly byte[] _dataA, _dataB;

    public UnbufferedFileReaderTests()
    {
        Directory.CreateDirectory(_dir);
        var rng = new Random(77);
        _dataA = new byte[3 << 20]; _dataB = new byte[1 << 20];       // not a multiple of 64 KiB: odd tail
        rng.NextBytes(_dataA); rng.NextBytes(_dataB);
        _dataB = _dataB[..((1 << 20) - 1234)];
        _a = Path.Combine(_dir, "a.bin"); _b = Path.Combine(_dir, "b.bin");
        File.WriteAllBytes(_a, _dataA); File.WriteAllBytes(_b, _dataB);
    }

    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ReadsUnalignedPieces_FromTwoFiles_ExactlyAsTheFileHolds(bool unbuffered)
    {
        using var reader = new UnbufferedFileReader([_a, _b], unbuffered, queueDepth: 4);
        // (file, offset, bytes): odd offsets and lengths like a real expert slice, including the very end of file B.
        var pieces = new (int File, long Offset, int Bytes)[]
        {
            (0, 12345, 90001), (0, 1_000_003, 300_000), (1, 7, 1000), (0, 2_900_000, 150_000), (1, _dataB.Length - 5000, 5000),
            (0, 0, 4096), (1, 500_000, 123_457), (0, 2_097_152 + 17, 77_777),
        };
        long ringStride = 1 << 20;
        byte* raw = (byte*)NativeMemory.AlignedAlloc((nuint)(ringStride * pieces.Length + 4096), 4096);
        try
        {
            var reqs = new UnbufferedFileReader.Request[pieces.Length];
            var delta = new int[pieces.Length];
            for (int i = 0; i < pieces.Length; i++)
            {
                var (start, length, d) = VulkanExpertTierPieces.Span(pieces[i].Offset, pieces[i].Bytes);
                reqs[i] = new UnbufferedFileReader.Request(pieces[i].File, start, length, raw + i * ringStride, d + pieces[i].Bytes);
                delta[i] = d;
            }
            reader.Read(reqs);
            for (int i = 0; i < pieces.Length; i++)
            {
                var src = pieces[i].File == 0 ? _dataA : _dataB;
                var got = new ReadOnlySpan<byte>(raw + i * ringStride + delta[i], pieces[i].Bytes);
                Assert.True(got.SequenceEqual(src.AsSpan((int)pieces[i].Offset, pieces[i].Bytes)), $"piece {i} (file {pieces[i].File} @ {pieces[i].Offset}) differs");
            }
            Assert.Equal(pieces.Length, reader.RequestsCompleted);
            Assert.Equal(unbuffered && OperatingSystem.IsWindows(), reader.Unbuffered);
        }
        finally { NativeMemory.AlignedFree(raw); }
    }

    [Fact]
    public void MoreRequestsThanQueueDepth_AllComplete()
    {
        using var reader = new UnbufferedFileReader([_a], unbuffered: true, queueDepth: 2);
        const int n = 40, len = 8192;
        byte* raw = (byte*)NativeMemory.AlignedAlloc((nuint)(n * len), 4096);
        try
        {
            var reqs = new UnbufferedFileReader.Request[n];
            for (int i = 0; i < n; i++) reqs[i] = new UnbufferedFileReader.Request(0, (long)i * 65536, len, raw + i * len, len);
            reader.Read(reqs);
            for (int i = 0; i < n; i++)
                Assert.True(new ReadOnlySpan<byte>(raw + i * len, len).SequenceEqual(_dataA.AsSpan(i * 65536, len)), $"request {i} differs");
        }
        finally { NativeMemory.AlignedFree(raw); }
    }

    [Fact]
    public void ReadShortOfWhatIsNeeded_Throws()
    {
        using var reader = new UnbufferedFileReader([_b], unbuffered: true);
        byte* raw = (byte*)NativeMemory.AlignedAlloc(1 << 20, 4096);
        try
        {
            long start = (_dataB.Length / 4096) * 4096;                   // the last (partial) sector: only a few hundred bytes exist
            var req = new UnbufferedFileReader.Request(0, start, 8192, raw, 8192);   // demands more than the file has
            Assert.Throws<IOException>(() => reader.Read([req]));
        }
        finally { NativeMemory.AlignedFree(raw); }
    }

    [Theory]
    [InlineData(0L, 100L, 0L, 4096, 0)]
    [InlineData(4096L, 4096L, 4096L, 4096, 0)]
    [InlineData(4097L, 4096L, 4096L, 8192, 1)]
    [InlineData(12345L, 90001L, 12288L, 90112, 57)]
    public void PieceSpan_CoversTheWantedBytes_WithSectorAlignedStartAndLength(long offset, long bytes, long start, int length, int delta)
    {
        var s = VulkanExpertTierPieces.Span(offset, bytes);
        Assert.Equal((start, length, delta), s);
        Assert.Equal(0, s.Start % 4096); Assert.Equal(0, s.Length % 4096);
        Assert.True(s.Start + s.Delta == offset && s.Delta + bytes <= s.Length);
    }
}

/// <summary>Test-side alias for the internal piece-span helper.</summary>
internal static class VulkanExpertTierPieces
{
    public static (long Start, int Length, int Delta) Span(long offset, long bytes) => DotLLM.Vulkan.Tiering.VulkanExpertTier.PieceSpan(offset, bytes);
}
