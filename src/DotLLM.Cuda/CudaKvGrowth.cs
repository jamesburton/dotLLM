using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Growth admission and byte arithmetic shared by the dense CUDA KV caches (<see cref="CudaKvCache"/>,
/// <see cref="CudaQuantizedKvCache"/>) when a retained prefix-cache entry must hold a longer next turn (issue #1594, the CUDA
/// side of #1781). A growth allocates larger per-layer buffers layer by layer, copies the live rows device to device in the
/// cache's own element format and frees the old ones, so the peak extra memory is the whole growth plus one old layer's
/// buffers in transit. It is admitted against free VRAM before anything is allocated: under WDDM an allocation past dedicated
/// VRAM does not fail, it spills into shared system memory (issue #1068).
/// </summary>
internal static class CudaKvGrowth
{
    /// <summary>
    /// Test hook: replaces the free-VRAM admission. Called with the peak extra bytes a growth needs; returns whether it fits.
    /// </summary>
    internal static Func<long, bool>? AdmissionOverride { get; set; }

    /// <summary>Growths carried out by any dense CUDA KV cache, process-wide (test observable).</summary>
    internal static long GrowthCount => Interlocked.Read(ref s_growths);
    private static long s_growths;

    /// <summary>Growths refused by admission, process-wide (test observable).</summary>
    internal static long RefusalCount => Interlocked.Read(ref s_refusals);
    private static long s_refusals;

    internal static void RecordGrowth() => Interlocked.Increment(ref s_growths);

    /// <summary>Free VRAM a growth must leave untouched (<see cref="CudaAdmission.ReserveBytes"/>).</summary>
    internal static long ReserveBytes => CudaAdmission.ReserveBytes;

    /// <summary>
    /// Peak extra device bytes of a layer-by-layer copying growth from <paramref name="oldLength"/> to
    /// <paramref name="newLength"/> positions: the added rows of every buffer plus the largest single layer's old buffers, which
    /// are still held while that layer's new ones are filled. <paramref name="rowBytesPerLayer"/> is the summed row bytes of the
    /// buffers each layer grows (K and V, in their stored format). <paramref name="fixedExtraBytes"/> is any other buffer allocated
    /// at its new size before the old one is freed (none today: the quantized cache's shared dequant scratch went in #1850).
    /// </summary>
    internal static long PeakGrowthBytes(ReadOnlySpan<long> rowBytesPerLayer, int oldLength, int newLength, long fixedExtraBytes = 0)
    {
        if (newLength <= oldLength) return 0;
        long growth = 0, transit = 0;
        foreach (long rowBytes in rowBytesPerLayer)
        {
            growth += (long)(newLength - oldLength) * rowBytes;
            transit = Math.Max(transit, (long)oldLength * rowBytes);
        }
        return growth + transit + fixedExtraBytes;
    }

    /// <summary>
    /// True when <paramref name="bytes"/> more can be allocated on the current context's device leaving <see cref="ReserveBytes"/>
    /// free, through <see cref="CudaAdmission.TryFit"/>. A failed reading refuses: no reading, no growth, and the caller falls back
    /// to a miss.
    /// </summary>
    internal static bool Admit(long bytes)
    {
        // #1848: one admission path for every CUDA allocation (CudaAdmission); the bool hook stays for the #1847 tests.
        bool ok = AdmissionOverride is { } admit ? admit(bytes) : CudaAdmission.TryFit(bytes, out _);
        if (!ok) Interlocked.Increment(ref s_refusals);
        return ok;
    }
}
