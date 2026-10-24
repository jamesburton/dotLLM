using System.Runtime.CompilerServices;

namespace DotLLM.Core.Attention;

/// <summary>
/// Row bookkeeping for compact sliding-window KV slots (issues #1169, #1176). A slot with window <c>W</c> keeps a buffer of
/// <c>2 * (W - 1 + Margin)</c> rows; row 0 holds position <see cref="BaseOf"/>. When an append would run past the
/// buffer, the last <c>W - 1 + Margin</c> committed rows move to the front and the base advances. The margin lets
/// speculative rollback and prefix truncation step back a little without losing rows a later query needs. A single
/// append larger than the buffer grows it. Full slots (window 0) are untouched and address rows by absolute position.
/// </summary>
/// <remarks>
/// This class only plans: <see cref="TryPlan"/> says which rows move where, the backend moves them (a host memcpy on the
/// CPU, a device copy on Vulkan), then <see cref="Commit"/> records the new base and capacity.
/// </remarks>
public sealed class SlidingWindowRows
{
    /// <summary>Rows kept beyond the window on compaction, so rollback / truncation by up to this many tokens still works.</summary>
    public const int Margin = 64;

    private readonly int[] _window;
    private readonly int[] _base;
    private readonly int[] _capacity;
    private readonly int[] _defaultCapacity;
    private readonly bool _growOnly;
    private readonly int _baseAlignment;

    /// <summary>Builds the bookkeeping for <paramref name="geometry"/>'s windowed slots.</summary>
    /// <param name="geometry">Geometry carrying per-slot windows.</param>
    /// <param name="maxSeqLen">Logical position capacity of the cache.</param>
    /// <param name="growOnly">
    /// Never shrink a grown buffer back to the default ring. Vulkan sets this: a resize swaps the buffer handle, which
    /// invalidates cached descriptor sets, so it must stay rare and happen outside command recording.
    /// </param>
    /// <param name="baseAlignment">
    /// Keep every row base a multiple of this many positions, and size the default ring for it (up to
    /// <c>baseAlignment - 1</c> more rows kept and allocated). A kernel that tiles the KV range from row 0 then sees the same
    /// tiles as on a full cache, so its output stays bit-identical (Vulkan: the sink prefill kernel's 256-row tile).
    /// </param>
    public SlidingWindowRows(KvGeometry geometry, int maxSeqLen, bool growOnly = false, int baseAlignment = 1)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(baseAlignment, 1);
        int n = geometry.LayerCount;
        _window = new int[n];
        _base = new int[n];
        _capacity = new int[n];
        _defaultCapacity = new int[n];
        _growOnly = growOnly;
        _baseAlignment = baseAlignment;
        for (int i = 0; i < n; i++)
        {
            int w = geometry.SlidingWindowOf(i);
            _window[i] = w;
            _defaultCapacity[i] = w > 0 ? 2 * (w - 1 + Margin) + baseAlignment - 1 : 0;

            // A window buffer never needs more rows than the cache has positions.
            _capacity[i] = w > 0 ? Math.Min(_defaultCapacity[i], maxSeqLen) : maxSeqLen;
        }
    }

    /// <summary>True when <paramref name="layer"/> keeps only its recent rows.</summary>
    /// <param name="layer">Slot index.</param>
    /// <returns>Whether the slot is windowed.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsWindowed(int layer) => _window[layer] > 0;

    /// <summary>The position held in row 0 of <paramref name="layer"/>.</summary>
    /// <param name="layer">Slot index.</param>
    /// <returns>The slot's row base.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int BaseOf(int layer) => _base[layer];

    /// <summary>Rows allocated for <paramref name="layer"/>.</summary>
    /// <param name="layer">Slot index.</param>
    /// <returns>The slot's row capacity.</returns>
    public int CapacityOf(int layer) => _capacity[layer];

    /// <summary>Records a new logical capacity for full slots after the cache grows.</summary>
    /// <param name="maxSeqLen">The new position capacity.</param>
    public void SetFullCapacity(int maxSeqLen)
    {
        for (int i = 0; i < _window.Length; i++)
            if (_window[i] == 0) _capacity[i] = maxSeqLen;
    }

    /// <summary>
    /// Plans how to make positions <c>[minPos, maxPos]</c> of a windowed slot addressable. Returns false when they already
    /// fit (or the slot is full). Otherwise the caller moves <see cref="WindowMove.Keep"/> rows from row
    /// <see cref="WindowMove.Shift"/> to row 0 (into a new buffer of <see cref="WindowMove.Capacity"/> rows when
    /// <see cref="WindowMove.Resize"/> is set) and then calls <see cref="Commit"/>.
    /// </summary>
    /// <param name="layer">Slot index.</param>
    /// <param name="minPos">Lowest position about to be written.</param>
    /// <param name="maxPos">Highest position about to be written.</param>
    /// <param name="currentLength">Committed length before the write.</param>
    /// <param name="move">The planned move.</param>
    /// <returns>True when rows must move.</returns>
    /// <exception cref="InvalidOperationException">A write below the slot's base: those rows were already dropped.</exception>
    public bool TryPlan(int layer, int minPos, int maxPos, int currentLength, out WindowMove move)
    {
        move = default;
        int w = _window[layer];
        if (w == 0) return false;
        int b = _base[layer];
        if (minPos < b)
            throw new InvalidOperationException(
                $"Write at position {minPos} is below the sliding-window base {b} of layer {layer}; those rows were dropped.");
        if (maxPos - b < _capacity[layer]) return false;

        // Keep the rows the first new query can still see, plus the margin; everything older is dead.
        int start = Math.Min(minPos, currentLength);
        int newBase = Math.Max(b, start - (w - 1 + Margin));
        newBase -= newBase % _baseAlignment; // stays >= b: every base is a multiple of the alignment
        int floor = _growOnly ? _capacity[layer] : _defaultCapacity[layer];
        int capacity = Math.Max(floor, maxPos - newBase + 1);
        move = new WindowMove(newBase - b, start - newBase, newBase, capacity, capacity != _capacity[layer]);
        return true;
    }

    /// <summary>Records a move planned by <see cref="TryPlan"/> once the backend has carried it out.</summary>
    /// <param name="layer">Slot index.</param>
    /// <param name="move">The move that was carried out.</param>
    public void Commit(int layer, in WindowMove move)
    {
        _base[layer] = move.NewBase;
        _capacity[layer] = move.Capacity;
    }

    /// <summary>
    /// True when every windowed slot still holds the rows a query at position <paramref name="length"/> needs, i.e. the
    /// cache can be cut back to <paramref name="length"/>. Cutting to 0 always succeeds (call <see cref="Reset"/>).
    /// </summary>
    /// <param name="length">The length to cut back to.</param>
    /// <returns>Whether the cut is possible.</returns>
    public bool CanTruncateTo(int length)
    {
        if (length == 0) return true;
        for (int i = 0; i < _window.Length; i++)
        {
            int w = _window[i];
            if (w > 0 && _base[i] > Math.Max(0, length - w + 1)) return false;
        }
        return true;
    }

    /// <summary>Most rows a compaction keeps for <paramref name="layer"/> (0 for a full slot): the window, the margin and the alignment slack.</summary>
    /// <param name="layer">Slot index.</param>
    /// <returns>The row bound.</returns>
    public int MaxKeptRows(int layer) => _window[layer] > 0 ? _window[layer] - 1 + Margin + _baseAlignment - 1 : 0;

    /// <summary>Moves every base back to 0; valid only when the cache is empty.</summary>
    public void Reset() => Array.Clear(_base);

    /// <summary>Bytes allocated across all slots for the given per-slot row bytes (K + V).</summary>
    /// <param name="rowBytesKv">K + V bytes of one row of a slot.</param>
    /// <returns>Total bytes.</returns>
    public long AllocatedBytes(Func<int, long> rowBytesKv)
    {
        long total = 0;
        for (int i = 0; i < _window.Length; i++)
            total += _capacity[i] * rowBytesKv(i);
        return total;
    }
}

/// <summary>A compaction planned by <see cref="SlidingWindowRows.TryPlan"/>: move <paramref name="Keep"/> rows from row <paramref name="Shift"/> to row 0.</summary>
/// <param name="Shift">First row kept (it becomes row 0).</param>
/// <param name="Keep">Rows kept.</param>
/// <param name="NewBase">Position held in row 0 afterwards.</param>
/// <param name="Capacity">Rows the buffer holds afterwards.</param>
/// <param name="Resize">True when <paramref name="Capacity"/> differs from the current buffer: allocate a new one.</param>
public readonly record struct WindowMove(int Shift, int Keep, int NewBase, int Capacity, bool Resize);
