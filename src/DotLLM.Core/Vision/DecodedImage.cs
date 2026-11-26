using System.Security.Cryptography;

namespace DotLLM.Core.Vision;

/// <summary>
/// A decoded still image: 8-bit RGB, interleaved, row-major (<c>R G B R G B ...</c>), no alpha. Source pixels, not tensor data, so a
/// managed array is fine; the encoder copies what it needs into native scratch.
/// </summary>
public sealed class DecodedImage
{
    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Interleaved RGB bytes, <c>Width * Height * 3</c> long.</summary>
    public byte[] Rgb { get; }

    /// <summary>Creates an image over <paramref name="rgb"/> (not copied).</summary>
    public DecodedImage(int width, int height, byte[] rgb)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width), "image dimensions must be positive");
        if (rgb.LongLength != (long)width * height * 3) throw new ArgumentException("rgb must hold width * height * 3 bytes", nameof(rgb));
        Width = width; Height = height; Rgb = rgb;
    }

    private string? _hash;

    /// <summary>
    /// SHA-256 of the dimensions and pixels (hex). Identifies the image content: it is what the prefix cache must key on, because two different
    /// images sit behind the same placeholder token ids.
    /// </summary>
    public string ContentHash => _hash ??= ComputeHash();

    private string ComputeHash()
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> dims = stackalloc byte[8];
        BitConverter.TryWriteBytes(dims, Width);
        BitConverter.TryWriteBytes(dims[4..], Height);
        sha.AppendData(dims);
        sha.AppendData(Rgb);
        return Convert.ToHexString(sha.GetHashAndReset());
    }
}
