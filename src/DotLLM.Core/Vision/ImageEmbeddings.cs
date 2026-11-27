using System.Runtime.InteropServices;

namespace DotLLM.Core.Vision;

/// <summary>
/// The output of an <see cref="IImageEncoder"/>: <see cref="Tokens"/> rows of <see cref="Dimension"/> values in the text model's embedding
/// space, held in native 64-byte-aligned memory as <b>F16</b> (CLAUDE.md: no FP32 stored; the F32 projector output is a per-call transient).
/// Row <c>r</c> replaces the token embedding at prompt position <c>start + r</c> (it bypasses the text embedding norm).
/// </summary>
public sealed unsafe class ImageEmbeddings : IDisposable
{
    private void* _data;

    /// <summary>Number of embedding rows (= image tokens in the prompt).</summary>
    public int Tokens { get; }

    /// <summary>Values per row (the text model's hidden size).</summary>
    public int Dimension { get; }

    /// <summary>Merged patch grid width (<c>Tokens == GridWidth * GridHeight</c>).</summary>
    public int GridWidth { get; }

    /// <summary>Merged patch grid height.</summary>
    public int GridHeight { get; }

    /// <summary>Content hash of the source image (<see cref="DecodedImage.ContentHash"/>).</summary>
    public string ContentHash { get; }

    /// <summary>Allocates uninitialised storage; the encoder fills it through <see cref="Rows"/>.</summary>
    public ImageEmbeddings(int gridWidth, int gridHeight, int dimension, string contentHash)
    {
        GridWidth = gridWidth; GridHeight = gridHeight; Tokens = gridWidth * gridHeight; Dimension = dimension; ContentHash = contentHash;
        long bytes = (long)Tokens * dimension * sizeof(ushort);
        _data = NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 64), 64);
    }

    /// <summary>The F16 payload, <c>Tokens * Dimension</c> halves.</summary>
    public Span<Half> Rows => _data == null ? throw new ObjectDisposedException(nameof(ImageEmbeddings)) : new Span<Half>(_data, Tokens * Dimension);

    /// <summary>Widens row <paramref name="row"/> into <paramref name="dest"/> (<see cref="Dimension"/> floats).</summary>
    public void ReadRow(int row, Span<float> dest)
    {
        if ((uint)row >= (uint)Tokens) throw new ArgumentOutOfRangeException(nameof(row));
        var src = Rows.Slice(row * Dimension, Dimension);
        for (int i = 0; i < src.Length; i++) dest[i] = (float)src[i];
    }

    /// <summary>Native bytes held.</summary>
    public long Bytes => (long)Tokens * Dimension * sizeof(ushort);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_data != null) { NativeMemory.AlignedFree(_data); _data = null; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Safety net only; callers dispose deterministically.</summary>
    ~ImageEmbeddings() { if (_data != null) NativeMemory.AlignedFree(_data); }
}
