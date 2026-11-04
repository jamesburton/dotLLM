namespace DotLLM.Models.Adapters.Ternary;

/// <summary>Storage type of the per-row ternary scales on a GPU (compute is always F32).</summary>
public enum TernaryScaleStorage
{
    /// <summary>F32 scales, matching the package format (default).</summary>
    F32 = 0,
    /// <summary>F16 scales (two per 32-bit word); converted exactly to F32 on the device.</summary>
    F16 = 1,
}

/// <summary>
/// One layer's ternary circuits packed for the GPU backends (#1618), shared by the GPU backends so they read the same bytes.
/// Circuits are concatenated in registration order (the rank-concatenation of the CPU composer).
/// </summary>
/// <remarks>
/// <b>Masks</b>: <c>2 * TotalRank</c> rows of <see cref="WordsPerRow"/> little-endian 32-bit words (= <c>ceil(d/64)</c> 64-bit
/// <c>[plus, minus]</c> pairs, exactly the CPU hook's layout): first <c>TotalRank</c> Down rows, then <c>TotalRank</c> UpT rows
/// (row j = column j of W_up). Bit <c>c &amp; 63</c> of block <c>c / 64</c> marks column c; plus = +1, minus = -1; 2 bits per weight, never expanded.
/// <b>Scales</b>: <c>downScale[TotalRank]</c> then, per circuit, <c>upScale[UpScaleStride]</c> starting at <see cref="UpScaleBase"/>
/// (F32 or F16 per <see cref="Storage"/>). <b>Meta</b>: 4 words per circuit: rank, rankOffset, alpha bits, threshold bits.
/// </remarks>
public sealed class TernaryDevicePack
{
    /// <summary>Little-endian mask bytes.</summary>
    public required byte[] Masks { get; init; }
    /// <summary>Scale bytes (F32 or F16 per <see cref="Storage"/>), padded to a multiple of 4 bytes.</summary>
    public required byte[] Scales { get; init; }
    /// <summary>Per-circuit metadata (16 bytes each).</summary>
    public required byte[] Meta { get; init; }
    /// <summary>Number of circuits.</summary>
    public int Circuits { get; init; }
    /// <summary>Summed rank.</summary>
    public int TotalRank { get; init; }
    /// <summary>Residual width.</summary>
    public int D { get; init; }
    /// <summary>Scale-element index of circuit 0's up scales (even).</summary>
    public int UpScaleBase { get; init; }
    /// <summary>Scale elements between circuits' up-scale arrays (even).</summary>
    public int UpScaleStride { get; init; }
    /// <summary>Scale storage.</summary>
    public TernaryScaleStorage Storage { get; init; }

    /// <summary>32-bit words per packed mask row for width <paramref name="d"/>: <c>ceil(d/64) * 4</c>.</summary>
    public static int WordsPerRow(int d) => (d + 63) / 64 * 4;

    /// <summary>Packs <paramref name="list"/> (all at one layer, all of width <paramref name="d"/>).</summary>
    public static TernaryDevicePack Build(IReadOnlyList<NsAdapterPackage> list, int d, TernaryScaleStorage storage)
    {
        int total = list.Sum(p => p.Manifest.Rank);
        int wprU = WordsPerRow(d);
        int upBase = (total + 1) & ~1;
        int upStride = (d + 1) & ~1;
        long scaleElems = upBase + (long)list.Count * upStride;
        long scaleBytes = (storage == TernaryScaleStorage.F16 ? scaleElems * 2 : scaleElems * 4 + 3) & ~3L;
        var masks = new uint[2L * total * wprU];
        var scales = new byte[scaleBytes];
        var meta = new uint[list.Count * 4];

        void PutScale(long idx, float v)
        {
            if (storage == TernaryScaleStorage.F16) BitConverter.TryWriteBytes(scales.AsSpan((int)idx * 2, 2), (Half)v);
            else BitConverter.TryWriteBytes(scales.AsSpan((int)idx * 4, 4), v);
        }
        void SetBit(long rowStart, int col, bool minus)
            => masks[rowStart + (col >> 6) * 4 + ((col >> 5) & 1) + (minus ? 2 : 0)] |= 1u << (col & 31);

        long upRows = (long)total * wprU;
        int off = 0;
        for (int c = 0; c < list.Count; c++)
        {
            var pkg = list[c];
            int rank = pkg.Manifest.Rank;
            meta[c * 4 + 0] = (uint)rank;
            meta[c * 4 + 1] = (uint)off;
            meta[c * 4 + 2] = BitConverter.SingleToUInt32Bits(pkg.Manifest.Alpha);
            meta[c * 4 + 3] = BitConverter.SingleToUInt32Bits(pkg.Manifest.ActivationThreshold);
            for (int j = 0; j < rank; j++)
            {
                PutScale(off + j, pkg.Down.Scales[j]);
                for (int col = 0; col < d; col++)
                {
                    sbyte w = pkg.Down[j, col];
                    if (w != 0) SetBit((long)(off + j) * wprU, col, w < 0);
                }
            }
            for (int r = 0; r < d; r++)
            {
                PutScale(upBase + (long)c * upStride + r, pkg.Up.Scales[r]);
                for (int j = 0; j < rank; j++)
                {
                    sbyte w = pkg.Up[r, j];
                    if (w != 0) SetBit(upRows + (long)(off + j) * wprU, r, w < 0);
                }
            }
            off += rank;
        }

        var maskBytes = new byte[masks.Length * 4L];
        Buffer.BlockCopy(masks, 0, maskBytes, 0, maskBytes.Length);
        var metaBytes = new byte[meta.Length * 4L];
        Buffer.BlockCopy(meta, 0, metaBytes, 0, metaBytes.Length);
        return new TernaryDevicePack
        {
            Masks = maskBytes, Scales = scales, Meta = metaBytes, Circuits = list.Count, TotalRank = total, D = d,
            UpScaleBase = upBase, UpScaleStride = upStride, Storage = storage,
        };
    }
}
