using System.Buffers.Binary;
using System.IO.Compression;
using DotLLM.Core.Vision;

namespace DotLLM.Models.Vision;

/// <summary>
/// Dependency-free still-image decoding to 8-bit RGB: binary PPM (<c>P6</c>) and non-interlaced PNG (all colour types, bit depths 1-16; alpha is
/// dropped like <c>stbi_load(..., 3)</c> in llama.cpp mtmd). JPEG and other formats are not supported yet (no license-compatible decoder is
/// vendored): callers get a <see cref="NotSupportedException"/> naming the format.
/// </summary>
/// <remarks>
/// The input is untrusted (server <c>data:</c> URIs). Every size is validated before it is used: chunk lengths against the buffer, IHDR fields
/// against the PNG specification, and the declared dimensions against <see cref="ImageLimits"/> (<see cref="MaxPixels"/> / <see cref="MaxSide"/> by default) <b>before</b> anything is
/// allocated from them, so a few-KB file cannot declare a multi-GB image (a decompression bomb). Malformed input throws
/// <see cref="InvalidDataException"/> (or <see cref="NotSupportedException"/> for valid but unsupported variants), never an index or overflow
/// exception. Chunk CRCs are not verified (as <c>stb_image</c>); the zlib stream's own Adler-32 is checked by <see cref="ZLibStream"/>.
/// </remarks>
public static class ImageDecoder
{
    /// <summary>
    /// Default for the largest decoded image accepted, in pixels (2^25, about 33.5 MP: 8K UHD fits); configurable through
    /// <see cref="ImageLimits.MaxPixelsPerImage"/>. It bounds the decode buffers (at most 8 bytes per pixel of filtered PNG rows plus 3 of RGB output).
    /// </summary>
    public const long MaxPixels = 1L << 25;

    /// <summary>Default for the largest accepted width or height in pixels (configurable through <see cref="ImageLimits.MaxSidePerImage"/>).</summary>
    public const int MaxSide = 1 << 16;

    /// <summary>Decodes <paramref name="data"/> (format sniffed from the magic bytes) within <paramref name="limits"/> (default: <see cref="ImageLimits.Default"/>).</summary>
    public static DecodedImage Decode(ReadOnlySpan<byte> data, ImageLimits? limits = null)
    {
        limits ??= ImageLimits.Default;
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') return DecodePng(data.ToArray(), limits);
        if (data.Length >= 2 && data[0] == 'P' && data[1] == '6') return DecodePpm(data.ToArray(), limits);
        if (data.Length >= 3 && data[0] == 0xFF && data[1] == 0xD8) throw new NotSupportedException("JPEG images are not supported yet; convert to PNG or PPM.");
        throw new NotSupportedException("unsupported image format (supported: PNG, binary PPM P6).");
    }

    /// <summary>Reads and decodes an image file.</summary>
    public static DecodedImage Load(string path, ImageLimits? limits = null) => Decode(File.ReadAllBytes(path), limits);

    /// <summary>Decodes a <c>data:</c> URI (<c>data:image/png;base64,...</c>).</summary>
    public static DecodedImage DecodeDataUri(string uri, ImageLimits? limits = null)
    {
        if (!uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("not a data URI", nameof(uri));
        int comma = uri.IndexOf(',');
        if (comma < 0 || !uri.AsSpan(0, comma).Contains(";base64", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException("only base64 data URIs are supported for images");
        return Decode(Convert.FromBase64String(uri[(comma + 1)..]), limits);
    }

    /// <summary>Throws unless <paramref name="w"/> x <paramref name="h"/> is a positive size within the side and pixel limits (default: <see cref="MaxSide"/> and <see cref="MaxPixels"/>).</summary>
    public static void CheckDimensions(long w, long h, ImageLimits? limits = null)
    {
        limits ??= ImageLimits.Default;
        if (w <= 0 || h <= 0) throw new InvalidDataException($"invalid image size {w}x{h}");
        if (w > limits.MaxSidePerImage || h > limits.MaxSidePerImage || w * h > limits.MaxPixelsPerImage)
            throw new InvalidDataException($"image too large: {w}x{h} (limit {limits.MaxSidePerImage} per side and {limits.MaxPixelsPerImage} pixels)");
    }

    private static DecodedImage DecodePpm(byte[] d, ImageLimits limits)
    {
        int pos = 2;
        int ReadInt()
        {
            while (pos < d.Length && (d[pos] == ' ' || d[pos] == '\n' || d[pos] == '\r' || d[pos] == '\t' || d[pos] == '#'))
            {
                if (d[pos] == '#') while (pos < d.Length && d[pos] != '\n') pos++;
                else pos++;
            }
            long v = 0; int n = 0;
            while (pos < d.Length && d[pos] >= '0' && d[pos] <= '9')
            {
                v = v * 10 + (d[pos] - '0'); pos++; n++;
                if (v > int.MaxValue) throw new InvalidDataException("PPM header value out of range");
            }
            if (n == 0) throw new InvalidDataException("malformed PPM header");
            return (int)v;
        }
        int w = ReadInt(), h = ReadInt(), max = ReadInt();
        if (pos >= d.Length || !(d[pos] is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t')) throw new InvalidDataException("malformed PPM header");
        pos++;   // single whitespace after the header
        if (max != 255) throw new NotSupportedException("only 8-bit PPM is supported");
        CheckDimensions(w, h, limits);
        if ((long)w * h * 3 > Array.MaxLength) throw new InvalidDataException($"image too large for the decoder: {w}x{h}");
        long need = (long)w * h * 3;
        if (d.Length - pos < need) throw new InvalidDataException("truncated PPM");
        return new DecodedImage(w, h, d.AsSpan(pos, (int)need).ToArray());
    }

    private static DecodedImage DecodePng(byte[] d, ImageLimits limits)
    {
        if (d.Length < 8 || d[4] != 0x0D || d[5] != 0x0A || d[6] != 0x1A || d[7] != 0x0A) throw new InvalidDataException("bad PNG signature");
        int pos = 8;
        int w = 0, h = 0, depth = 0, ctype = 0;
        bool haveHeader = false, sawEnd = false;
        byte[]? palette = null;
        using var idat = new MemoryStream();
        while (d.Length - pos >= 12)
        {
            uint ulen = BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(pos, 4));
            if (ulen > int.MaxValue || ulen > (uint)(d.Length - pos - 12)) throw new InvalidDataException("PNG chunk length exceeds the data");
            int len = (int)ulen;
            var type = d.AsSpan(pos + 4, 4);
            var body = d.AsSpan(pos + 8, len);
            pos += 12 + len;
            if (type.SequenceEqual("IHDR"u8))
            {
                if (haveHeader) throw new InvalidDataException("duplicate PNG IHDR");
                if (len != 13) throw new InvalidDataException("bad PNG IHDR length");
                uint uw = BinaryPrimitives.ReadUInt32BigEndian(body), uh = BinaryPrimitives.ReadUInt32BigEndian(body[4..]);
                CheckDimensions(uw, uh, limits);
                w = (int)uw; h = (int)uh;
                depth = body[8]; ctype = body[9];
                bool depthOk = ctype switch
                {
                    0 => depth is 1 or 2 or 4 or 8 or 16,
                    3 => depth is 1 or 2 or 4 or 8,
                    2 or 4 or 6 => depth is 8 or 16,
                    _ => throw new InvalidDataException("bad PNG colour type"),
                };
                if (!depthOk) throw new InvalidDataException($"bad PNG bit depth {depth} for colour type {ctype}");
                if (body[10] != 0 || body[11] != 0) throw new InvalidDataException("bad PNG compression / filter method");
                if (body[12] > 1) throw new InvalidDataException("bad PNG interlace method");
                if (body[12] != 0) throw new NotSupportedException("interlaced PNG is not supported; re-save without interlacing.");
                haveHeader = true;
            }
            else if (!haveHeader) throw new InvalidDataException("PNG does not start with IHDR");
            else if (type.SequenceEqual("PLTE"u8))
            {
                if (len == 0 || len % 3 != 0 || len > 768) throw new InvalidDataException("bad PNG palette length");
                palette = body.ToArray();
            }
            else if (type.SequenceEqual("IDAT"u8)) idat.Write(body);
            else if (type.SequenceEqual("IEND"u8)) { sawEnd = true; break; }
        }
        if (!haveHeader) throw new InvalidDataException("PNG without IHDR");
        if (!sawEnd && idat.Length == 0) throw new InvalidDataException("PNG without image data");
        if (ctype == 3 && palette is null) throw new InvalidDataException("palette PNG without PLTE");
        int channels = ctype switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, _ => 4 };
        int bitsPerPixel = channels * depth;
        int bpp = Math.Max(1, bitsPerPixel / 8);
        long stride = ((long)w * bitsPerPixel + 7) / 8;   // <= 8 * the side limit: fits an int for any side below 2^28
        // Bounded by CheckDimensions: (stride + 1) * h <= 8 * the pixel limit + the side limit, and rgb = 3 * the pixel limit bytes at most.
        long rawBytes = (stride + 1) * h;
        // A raised pixel limit must not reach managed array limits: refuse instead of overflowing an int index later.
        if (rawBytes > Array.MaxLength || (long)w * h * 3 > Array.MaxLength) throw new InvalidDataException($"image too large for the decoder: {w}x{h}");
        // Deflate cannot expand by more than ~1032:1, so a stream far shorter than that cannot hold the declared image: refuse it before the
        // buffer is allocated (a header-only "bomb" would otherwise cost the full declared size per request).
        if (rawBytes > idat.Length * 1032L + 1024) throw new InvalidDataException("truncated PNG image data");
        var raw = new byte[rawBytes];
        idat.Position = 0;
        using (var z = new ZLibStream(idat, CompressionMode.Decompress)) z.ReadExactly(raw);   // decompresses no more than the declared size
        var cur = new byte[stride];
        var prev = new byte[stride];
        var rgb = new byte[(long)w * h * 3];
        for (int y = 0; y < h; y++)
        {
            long rowStart = y * (stride + 1);
            int filter = raw[rowStart];
            if (filter > 4) throw new InvalidDataException("bad PNG filter");
            var line = new ReadOnlySpan<byte>(raw, (int)(rowStart + 1), (int)stride);
            for (int i = 0; i < stride; i++)
            {
                int a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
                int x = line[i];
                cur[i] = (byte)(filter switch
                {
                    0 => x,
                    1 => x + a,
                    2 => x + b,
                    3 => x + ((a + b) >> 1),
                    _ => x + Paeth(a, b, c),
                });
            }
            for (int x = 0; x < w; x++)
            {
                long o = ((long)y * w + x) * 3;
                int Sample(int ch)
                {
                    if (depth == 8) return cur[x * channels + ch];
                    if (depth == 16) return cur[(x * channels + ch) * 2];   // high byte
                    int bit = (x * channels + ch) * depth;
                    int v = (cur[bit >> 3] >> (8 - depth - (bit & 7))) & ((1 << depth) - 1);
                    return ctype == 3 ? v : v * 255 / ((1 << depth) - 1);
                }
                switch (ctype)
                {
                    case 0: case 4: { int g = Sample(0); rgb[o] = rgb[o + 1] = rgb[o + 2] = (byte)g; break; }
                    case 2: case 6: rgb[o] = (byte)Sample(0); rgb[o + 1] = (byte)Sample(1); rgb[o + 2] = (byte)Sample(2); break;
                    default:
                    {
                        int idx = Sample(0);
                        if (idx * 3 + 2 >= palette!.Length) throw new InvalidDataException("bad PNG palette index");
                        rgb[o] = palette[idx * 3]; rgb[o + 1] = palette[idx * 3 + 1]; rgb[o + 2] = palette[idx * 3 + 2];
                        break;
                    }
                }
            }
            (cur, prev) = (prev, cur);
        }
        return new DecodedImage(w, h, rgb);
    }

    private static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}
