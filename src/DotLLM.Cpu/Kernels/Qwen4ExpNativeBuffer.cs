using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Growable, 64-byte-aligned native float buffer (<see cref="NativeMemory.AlignedAlloc"/>) backing the Qwen4-Exp per-sequence
/// state (QSA K/V rows, pooled indexer keys, row-snapshot scratch). Lazily allocated: an unused buffer costs nothing.
/// Contents are NOT zeroed on growth (position-indexed data is always written before it is read) unless asked.
/// </summary>
public sealed unsafe class Qwen4ExpNativeBuffer : IDisposable
{
    private float* _ptr;
    private long _capacity;

    /// <summary>Capacity in floats.</summary>
    public long Capacity => _capacity;

    /// <summary>Resident bytes.</summary>
    public long Bytes => _capacity * sizeof(float);

    /// <summary>Base pointer (null until the first <see cref="EnsureCapacity"/>).</summary>
    public float* Pointer => _ptr;

    /// <summary>Span over the first <paramref name="count"/> floats.</summary>
    public Span<float> Slice(long start, int count)
    {
        if (start < 0 || count < 0 || start + count > _capacity) throw new ArgumentOutOfRangeException(nameof(count));
        return new Span<float>(_ptr + start, count);
    }

    /// <summary>Grows (geometrically, preserving contents) so that at least <paramref name="floats"/> floats fit.</summary>
    public void EnsureCapacity(long floats)
    {
        if (floats <= _capacity) return;
        long cap = Math.Max(_capacity, 256);
        while (cap < floats) cap *= 2;
        _ptr = (float*)(_ptr == null
            ? NativeMemory.AlignedAlloc((nuint)(cap * sizeof(float)), 64)
            : NativeMemory.AlignedRealloc(_ptr, (nuint)(cap * sizeof(float)), 64));
        _capacity = cap;
    }

    /// <summary>Allocates exactly <paramref name="floats"/> floats if nothing is allocated yet or too little is.</summary>
    public void EnsureExact(long floats)
    {
        if (floats <= _capacity) return;
        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
        _ptr = (float*)NativeMemory.AlignedAlloc((nuint)(floats * sizeof(float)), 64);
        _capacity = floats;
    }

    /// <summary>Zeroes the allocated region.</summary>
    public void Clear()
    {
        if (_ptr != null) NativeMemory.Clear(_ptr, (nuint)(_capacity * sizeof(float)));
    }

    /// <summary>Frees the allocation (the buffer can be re-grown afterwards).</summary>
    public void Dispose()
    {
        if (_ptr != null) { NativeMemory.AlignedFree(_ptr); _ptr = null; _capacity = 0; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Finalizer: frees the allocation if the owner forgot to dispose it.</summary>
    ~Qwen4ExpNativeBuffer()
    {
        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
    }
}
