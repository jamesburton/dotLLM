using System.Buffers.Binary;
using System.IO.Compression;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using DotLLM.Server;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// The image decoders and request bounds against hostile input (PR #2037 review): the server decodes attacker-supplied <c>data:</c> URIs with
/// hand-written PNG/PPM decoders, so malformed or oversized input must be refused cheaply and only with the decoder's documented exceptions.
/// </summary>
public sealed class Inkling2VisionUntrustedInputTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VisionUntrustedInputTests(ITestOutputHelper output) => _output = output;

    private static byte[] Png(int w, int h, int depth, int ctype, byte[] raw, byte[]? plte = null, int ihdrLength = 13)
    {
        var png = new List<byte> { 0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A };
        void Chunk(string type, byte[] body)
        {
            var len = new byte[4]; BinaryPrimitives.WriteInt32BigEndian(len, body.Length);
            png.AddRange(len); png.AddRange(System.Text.Encoding.ASCII.GetBytes(type)); png.AddRange(body); png.AddRange(new byte[4]);
        }
        var ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr, w); BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), h);
        ihdr[8] = (byte)depth; ihdr[9] = (byte)ctype;
        Chunk("IHDR", ihdr[..ihdrLength]);
        if (plte is not null) Chunk("PLTE", plte);
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, CompressionLevel.SmallestSize, leaveOpen: true)) z.Write(raw);
        Chunk("IDAT", ms.ToArray());
        Chunk("IEND", []);
        return png.ToArray();
    }

    private static byte[] Rows(int h, int stride, Func<int, int, byte> px, int filter = 0)
    {
        var raw = new byte[h * (stride + 1)];
        for (int y = 0; y < h; y++) { raw[y * (stride + 1)] = (byte)filter; for (int i = 0; i < stride; i++) raw[y * (stride + 1) + 1 + i] = px(y, i); }
        return raw;
    }

    /// <summary>Seed corpus: every colour type the decoder supports (including palette and sub-byte depths), plus a PPM.</summary>
    private static List<byte[]> Corpus()
    {
        var r = new Random(1);
        byte B(int y, int i) => (byte)r.Next(256);
        return
        [
            Png(5, 4, 8, 2, Rows(4, 15, B, 1)),
            Png(5, 4, 8, 6, Rows(4, 20, B, 4)),
            Png(5, 4, 16, 0, Rows(4, 10, B, 3)),
            Png(9, 3, 1, 0, Rows(3, 2, B, 2)),
            Png(6, 3, 4, 3, Rows(3, 3, (y, i) => (byte)((i % 3) * 0x11)), plte: [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            Png(4, 2, 8, 4, Rows(2, 8, B)),
            [.. "P6\n# c\n3 2\n255\n"u8.ToArray(), .. Enumerable.Range(0, 18).Select(i => (byte)i)],
        ];
    }

    private static bool Allowed(Exception ex) => ex is InvalidDataException or NotSupportedException or EndOfStreamException;

    [Fact]
    public void Corpus_DecodesCleanly()
    {
        foreach (var f in Corpus()) Assert.NotNull(ImageDecoder.Decode(f));
        // the palette image: indices 0,1,2 repeating -> palette rows
        var pal = ImageDecoder.Decode(Corpus()[4]);
        Assert.Equal(new byte[] { 1, 2, 3, 1, 2, 3, 4, 5, 6 }, pal.Rgb[..9]);
    }

    /// <summary>
    /// Fuzz: thousands of seeded mutations (byte flips, random overwrites of the length / IHDR fields, truncation, splices) of a valid corpus. The
    /// decoder may accept a mutant or refuse it with its documented exceptions, but must never throw anything else (an index, overflow or
    /// argument exception there is a 500 on the server and a sign of an unchecked read).
    /// </summary>
    [Fact]
    public void Fuzz_MutatedImages_ThrowOnlyDocumentedExceptions()
    {
        var corpus = Corpus();
        var rng = new Random(2037);
        int accepted = 0, refused = 0;
        for (int iter = 0; iter < 10000; iter++)
        {
            var src = corpus[iter % corpus.Count];
            var m = (byte[])src.Clone();
            switch (rng.Next(5))
            {
                case 0: for (int k = rng.Next(1, 4); k > 0; k--) m[rng.Next(8, m.Length)] ^= (byte)(1 << rng.Next(8)); break;
                case 1: m[rng.Next(8, m.Length)] = (byte)rng.Next(256); break;
                case 2: if (m.Length > 33) { int at = rng.Next(8, 33); m[at] = (byte)(rng.Next(2) == 0 ? 0xFF : rng.Next(256)); } break;   // chunk length / IHDR fields
                case 3: m = m[..rng.Next(1, m.Length)]; break;
                default: { int at = rng.Next(8, m.Length); var tail = new byte[rng.Next(1, 16)]; rng.NextBytes(tail); m = [.. m[..at], .. tail, .. m[at..]]; break; }
            }
            try { ImageDecoder.Decode(m); accepted++; }
            catch (Exception ex) when (Allowed(ex)) { refused++; }
            catch (Exception ex) { Assert.Fail($"iteration {iter}: {ex.GetType().Name}: {ex.Message} on {Convert.ToHexString(m)}"); }
        }
        _output.WriteLine($"accepted {accepted}, refused {refused}");
        Assert.True(refused > 1000 && accepted > 100, "the mutations should exercise both outcomes");
    }

    [Theory]
    [InlineData(65535, 65535)]   // a 4-gigapixel declaration from a few hundred bytes
    [InlineData(20000, 20000)]
    [InlineData(5000, 5000)]     // under the pixel cap: refused by the deflate-ratio bound
    [InlineData(int.MaxValue, 1)]
    public void DecompressionBomb_IsRefusedBeforeAllocating(int w, int h)
    {
        var png = Png(w, h, 8, 2, new byte[4096]);   // a tiny stream: nowhere near the declared size
        long before = GC.GetAllocatedBytesForCurrentThread();
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(png));
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        _output.WriteLine($"{w}x{h}: {allocated} bytes allocated");
        Assert.True(allocated < 1 << 20, $"{allocated} bytes allocated for a refused image");
    }

    [Fact]
    public void MalformedHeaders_AreInvalidData()
    {
        var rows = Rows(2, 6, (y, i) => 1);
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(2, 2, 8, 2, rows, ihdrLength: 8)));   // short IHDR
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(2, 2, 0, 2, rows)));                  // depth 0
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(2, 2, 4, 2, rows)));                  // depth 4 is not allowed for RGB
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(2, 2, 32, 0, rows)));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(2, 2, 8, 3, rows)));                  // palette image without PLTE
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode(Png(-5, 2, 8, 2, rows)));                 // width with the high bit set
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode("P6\n99999999999 2\n255\n"u8.ToArray()));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode("P6\n40000 40000\n255\n"u8.ToArray()));
        Assert.Throws<InvalidDataException>(() => ImageDecoder.Decode("P6\n2 2\n255"u8.ToArray()));
    }

    [Fact]
    public void ExtremeAspectRatio_IsRefusedInsteadOfPlanningUnboundedTokens()
    {
        Assert.Throws<NotSupportedException>(() => Inkling2ImagePreprocessor.CountTokens(1_000_000, 1));
        Assert.Throws<NotSupportedException>(() => Inkling2ImagePreprocessor.CountTokens(40, 4_000_000));
        Assert.InRange(Inkling2ImagePreprocessor.CountTokens(65536, 400), 1, Inkling2ImagePreprocessor.MaxTokens);
    }

    /// <summary>
    /// The over-limit grid rounding is <c>std::round</c> (half away from zero) in llama.cpp's <c>calc_size_preserved_ratio</c>. 1100x5860 is
    /// 27.5 x 146.5 cells: half-away gives 28 x 147 (over the limit, so the beta rescale yields 27 x 147 = 3969 tokens); half-to-even gives
    /// 28 x 146 = 4088 tokens and a different image size.
    /// </summary>
    [Fact]
    public void OverLimitRounding_IsHalfAwayFromZero_LikeTheReference()
    {
        Assert.Equal(27 * 147, Inkling2ImagePreprocessor.CountTokens(1100, 5860));
        var (w, h, _, _) = Inkling2ImagePreprocessor.PlanSize(1100, 5860);
        Assert.Equal((1080, 5880), (w, h));
    }

    [Fact]
    public void ServerDecode_BoundsImageCountAndPixels_AndMapsDecoderFaultsTo400()
    {
        string Uri(byte[] b) => "data:image/png;base64," + Convert.ToBase64String(b);
        var small = Uri(Corpus()[0]);
        Assert.Equal(2, ImageInput.DecodeAll([small, small]).Count);
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll(Enumerable.Repeat(small, ImageInput.MaxImagesPerRequest + 1).ToList()));
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll(["https://example.com/a.png"]));
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll(["data:image/png;base64,!!!notbase64"]));
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll([Uri(Png(65535, 65535, 8, 2, new byte[16]))]));
        // three 4096x4096 frames (16.7 MP each) exceed the 2^26-pixel request budget on the fourth; build them as PPM (cheap to make)
        byte[] big = [.. "P6\n4096 4096\n255\n"u8.ToArray(), .. new byte[4096 * 4096 * 3]];
        string bigUri = "data:image/x-portable-pixmap;base64," + Convert.ToBase64String(big);
        Assert.Equal(4, ImageInput.DecodeAll([bigUri, bigUri, bigUri, bigUri]).Count);   // exactly 2^26
        Assert.Throws<ImageInputException>(() => ImageInput.DecodeAll([bigUri, bigUri, bigUri, bigUri, small]));
    }

    private sealed class FixedEncoder(int tokens) : IImageEncoder
    {
        public int EmbeddingDimension => 8;
        public int CountTokens(int width, int height) => tokens;
        public ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default) => throw new InvalidOperationException("must not encode");
        public void Dispose() { }
    }

    [Fact]
    public void TokenBudget_IsCheckedFromCountTokens_BeforeAnyEncode()
    {
        var imgs = new List<DecodedImage> { new(1, 1, [0, 0, 0]), new(1, 1, [0, 0, 0]) };
        Assert.Equal(2 * 99, ImageInput.CheckTokenBudget(new FixedEncoder(100), imgs, 198));
        var ex = Assert.Throws<ImageInputException>(() => ImageInput.CheckTokenBudget(new FixedEncoder(100), imgs, 197));
        Assert.Equal("context_length_exceeded", ex.Code);
    }
}
