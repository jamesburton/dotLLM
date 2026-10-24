using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;

namespace DotLLM.Engine.KvCache;

/// <summary>
/// Row bookkeeping for compact sliding-window KV slots (issue #1169). A slot with window <c>W</c> keeps a buffer of
/// <c>2 * (W - 1 + Margin)</c> rows; row 0 holds position <see cref="BaseOf"/>. When an append would run past the
/// buffer, the last <c>W - 1 + Margin</c> committed rows move to the front and the base advances. The margin lets
/// speculative rollback and prefix truncation step back a little without losing rows a later query needs. A single
/// append larger than the buffer grows it; the next compaction that fits shrinks it back. Full slots (window 0) are
/// untouched and address rows by absolute position.
/// </summary>
internal sealed unsafe class SlidingWindowRows
{
    /// <summary>Rows kept beyond the window on compaction, so rollback / truncation by up to this many tokens still works.</summary>
    internal const int Margin = 64;

    private readonly int[] _window;
    private readonly int[] _base;
    private readonly int[] _capacity;
    private readonly int[] _defaultCapacity;

    /// <summary>Builds the bookkeeping for <paramref name="geometry"/>'s windowed slots.</summary>
    /// <param name="geometry">Geometry carrying per-slot windows.</param>
    /// <param name="maxSeqLen">Logical position capacity of the cache.</param>
    internal SlidingWindowRows(KvGeometry geometry, int maxSeqLen)
    {
        int n = geometry.LayerCount;
        _window = new int[n];
        _base = new int[n];
        _capacity = new int[n];
        _defaultCapacity = new int[n];
        for (int i = 0; i < n; i++)
        {
            int w = geometry.SlidingWindowOf(i);
            _window[i] = w;
            _defaultCapacity[i] = w > 0 ? 2 * (w - 1 + Margin) : 0;
            // A window buffer never needs more rows than the cache has positions.
            _capacity[i] = w > 0 ? Math.Min(_defaultCapacity[i], maxSeqLen) : maxSeqLen;
        }
    }

    /// <summary>True when <paramref name="layer"/> keeps only its recent rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal bool IsWindowed(int layer) => _window[layer] > 0;

    /// <summary>The position held in row 0 of <paramref name="layer"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal int BaseOf(int layer) => _base[layer];

    /// <summary>Rows allocated for <paramref name="layer"/>.</summary>
    internal int CapacityOf(int layer) => _capacity[layer];

    /// <summary>Records a new logical capacity for full slots after the cache grows.</summary>
    internal void SetFullCapacity(int maxSeqLen)
    {
        for (int i = 0; i < _window.Length; i++)
            if (_window[i] == 0) _capacity[i] = maxSeqLen;
    }

    /// <summary>
    /// Makes positions <c>[minPos, maxPos]</c> of a windowed slot addressable, compacting or resizing its K and V buffers
    /// when they would overflow. No-op for a full slot.
    /// </summary>
    /// <param name="layer">Slot index.</param>
    /// <param name="minPos">Lowest position about to be written.</param>
    /// <param name="maxPos">Highest position about to be written.</param>
    /// <param name="currentLength">Committed length before the write.</param>
    /// <param name="keys">The slot's key buffer; replaced when resized.</param>
    /// <param name="values">The slot's value buffer; replaced when resized.</param>
    /// <param name="keyRowBytes">Bytes per key row.</param>
    /// <param name="valueRowBytes">Bytes per value row.</param>
    /// <exception cref="InvalidOperationException">A write below the slot's base: those rows were already dropped.</exception>
    internal void Prepare(int layer, int minPos, int maxPos, int currentLength,
        ref nint keys, ref nint values, int keyRowBytes, int valueRowBytes)
    {
        int w = _window[layer];
        if (w == 0) return;
        int b = _base[layer];
        if (minPos < b)
            throw new InvalidOperationException(
                $"Write at position {minPos} is below the sliding-window base {b} of layer {layer}; those rows were dropped.");
        if (maxPos - b < _capacity[layer]) return;

        // Keep the rows the first new query can still see, plus the margin; everything older is dead.
        int start = Math.Min(minPos, currentLength);
        int newBase = Math.Max(b, start - (w - 1 + Margin));
        int keep = start - newBase;
        int capacity = Math.Max(_defaultCapacity[layer], maxPos - newBase + 1);
        int shift = newBase - b;

        if (capacity == _capacity[layer])
        {
            // Overlapping move to the front; Buffer.MemoryCopy is overlap-safe.
            Move(keys, shift, keep, keyRowBytes);
            Move(values, shift, keep, valueRowBytes);
        }
        else
        {
            keys = Resize(keys, shift, keep, keyRowBytes, capacity);
            values = Resize(values, shift, keep, valueRowBytes, capacity);
            _capacity[layer] = capacity;
        }
        _base[layer] = newBase;

        static void Move(nint buffer, int shift, int keep, int rowBytes)
        {
            if (shift == 0 || keep == 0) return;
            long bytes = (long)keep * rowBytes;
            Buffer.MemoryCopy((byte*)buffer + (long)shift * rowBytes, (void*)buffer, bytes, bytes);
        }

        static nint Resize(nint old, int shift, int keep, int rowBytes, int capacity)
        {
            nuint newBytes = (nuint)((long)capacity * rowBytes);
            void* p = NativeMemory.AlignedAlloc(newBytes, 64);
            long bytes = (long)keep * rowBytes;
            Buffer.MemoryCopy((byte*)old + (long)shift * rowBytes, p, (long)newBytes, bytes);
            NativeMemory.AlignedFree((void*)old);
            return (nint)p;
        }
    }

    /// <summary>
    /// True when every windowed slot still holds the rows a query at position <paramref name="length"/> needs, i.e. the
    /// cache can be cut back to <paramref name="length"/>. Cutting to 0 always succeeds (call <see cref="Reset"/>).
    /// </summary>
    internal bool CanTruncateTo(int length)
    {
        if (length == 0) return true;
        for (int i = 0; i < _window.Length; i++)
        {
            int w = _window[i];
            if (w > 0 && _base[i] > Math.Max(0, length - w + 1)) return false;
        }
        return true;
    }

    /// <summary>Moves every base back to 0; valid only when the cache is empty.</summary>
    internal void Reset() => Array.Clear(_base);

    /// <summary>Bytes allocated across all slots for the given per-slot row bytes (K + V).</summary>
    internal long AllocatedBytes(Func<int, long> rowBytesKv)
    {
        long total = 0;
        for (int i = 0; i < _window.Length; i++)
            total += _capacity[i] * rowBytesKv(i);
        return total;
    }
}
