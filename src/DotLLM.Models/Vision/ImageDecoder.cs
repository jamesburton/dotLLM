using System.IO.Compression;
using DotLLM.Core.Vision;

namespace DotLLM.Models.Vision;

/// <summary>
/// Dependency-free still-image decoding to 8-bit RGB: binary PPM (<c>P6</c>) and non-interlaced PNG (all colour types, bit depths 1-16; alpha is
/// dropped like <c>stbi_load(..., 3)</c> in llama.cpp mtmd). JPEG and other formats are not supported yet (no license-compatible decoder is
/// vendored): callers get a <see cref="NotSupportedException"/> naming the format.
/// </summary>
public static class ImageDecoder
{
    /// <summary>Decodes <paramref name="data"/> (format sniffed from the magic bytes).</summary>
    public static DecodedImage Decode(ReadOnlySpan<byte> data)
    {
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') return DecodePng(data.ToArray());
        if (data.Length >= 2 && data[0] == 'P' && data[1] == '6') return DecodePpm(data.ToArray());
        if (data.Length >= 3 && data[0] == 0xFF && data[1] == 0xD8) throw new NotSupportedException("JPEG images are not supported yet; convert to PNG or PPM.");
        throw new NotSupportedException("unsupported image format (supported: PNG, binary PPM P6).");
    }

    /// <summary>Reads and decodes an image file.</summary>
    public static DecodedImage Load(string path) => Decode(File.ReadAllBytes(path));

    /// <summary>Decodes a <c>data:</c> URI (<c>data:image/png;base64,...</c>).</summary>
    public static DecodedImage DecodeDataUri(string uri)
    {
        if (!uri.StartsWith("data:", StringComparison.OrdinalIgnoreCase)) throw new ArgumentException("not a data URI", nameof(uri));
        int comma = uri.IndexOf(',');
        if (comma < 0 || !uri.AsSpan(0, comma).Contains(";base64", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException("only base64 data URIs are supported for images");
        return Decode(Convert.FromBase64String(uri[(comma + 1)..]));
    }

    private static DecodedImage DecodePpm(byte[] d)
    {
        int pos = 2;
        int ReadInt()
        {
            while (pos < d.Length && (d[pos] == ' ' || d[pos] == '\n' || d[pos] == '\r' || d[pos] == '\t' || d[pos] == '#'))
            {
                if (d[pos] == '#') while (pos < d.Length && d[pos] != '\n') pos++;
                else pos++;
            }
            int v = 0, n = 0;
            while (pos < d.Length && d[pos] >= '0' && d[pos] <= '9') { v = checked(v * 10 + (d[pos] - '0')); pos++; n++; }
            if (n == 0) throw new InvalidDataException("malformed PPM header");
            return v;
        }
        int w = ReadInt(), h = ReadInt(), max = ReadInt();
        pos++;   // single whitespace after the header
        if (max != 255) throw new NotSupportedException("only 8-bit PPM is supported");
        long need = (long)w * h * 3;
        if (d.Length - pos < need) throw new InvalidDataException("truncated PPM");
        return new DecodedImage(w, h, d.AsSpan(pos, (int)need).ToArray());
    }

    private static DecodedImage DecodePng(byte[] d)
    {
        int pos = 8;
        int w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
        byte[]? palette = null;
        using var idat = new MemoryStream();
        while (pos + 8 <= d.Length)
        {
            int len = (d[pos] << 24) | (d[pos + 1] << 16) | (d[pos + 2] << 8) | d[pos + 3];
            string type = System.Text.Encoding.ASCII.GetString(d, pos + 4, 4);
            var body = d.AsSpan(pos + 8, len);
            pos += 12 + len;
            if (type == "IHDR")
            {
                w = (body[0] << 24) | (body[1] << 16) | (body[2] << 8) | body[3];
                h = (body[4] << 24) | (body[5] << 16) | (body[6] << 8) | body[7];
                depth = body[8]; ctype = body[9]; interlace = body[12];
            }
            else if (type == "PLTE") palette = body.ToArray();
            else if (type == "IDAT") idat.Write(body);
            else if (type == "IEND") break;
        }
        if (w <= 0 || h <= 0) throw new InvalidDataException("PNG without IHDR");
        if (interlace != 0) throw new NotSupportedException("interlaced PNG is not supported; re-save without interlacing.");
        int channels = ctype switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => throw new InvalidDataException("bad PNG colour type") };
        int bitsPerPixel = channels * depth;
        int bpp = Math.Max(1, bitsPerPixel / 8);
        long stride = ((long)w * bitsPerPixel + 7) / 8;
        var raw = new byte[(stride + 1) * h];
        idat.Position = 0;
        using (var z = new ZLibStream(idat, CompressionMode.Decompress)) z.ReadExactly(raw);
        var cur = new byte[stride];
        var prev = new byte[stride];
        var rgb = new byte[(long)w * h * 3];
        for (int y = 0; y < h; y++)
        {
            int filter = raw[y * (stride + 1)];
            var line = new ReadOnlySpan<byte>(raw, (int)(y * (stride + 1) + 1), (int)stride);
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
                    4 => x + Paeth(a, b, c),
                    _ => throw new InvalidDataException("bad PNG filter"),
                });
            }
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 3;
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
                        if (palette is null || idx * 3 + 2 >= palette.Length) throw new InvalidDataException("bad PNG palette index");
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
