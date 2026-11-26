using System.Runtime.InteropServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Bump allocator over 64-byte-aligned native memory: the F32 transient activations of one Inkling 2 forward call (CLAUDE.md: no managed arrays
/// for tensor data). <see cref="Reset"/> rewinds it; if a call needed more than one chunk the chunks are consolidated into one so the steady state
/// is a single chunk and zero allocation. Memory is NOT zeroed: callers must write what they read.
/// </summary>
internal sealed unsafe class Inkling2Arena : IDisposable
{
    private const long MinChunk = 1 << 20;
    private readonly List<nint> _chunks = new();
    private readonly List<long> _sizes = new();
    private long _total, _off;

    /// <summary>Total native bytes held.</summary>
    public long Capacity => _total;

    /// <summary>Allocates <paramref name="count"/> elements, valid until the next <see cref="Reset"/>.</summary>
    public Span<T> Alloc<T>(int count) where T : unmanaged
    {
        long bytes = (((long)count * sizeof(T)) + 63) & ~63L;
        if (bytes == 0) bytes = 64;
        int last = _chunks.Count - 1;
        if (last < 0 || _off + bytes > _sizes[last])
        {
            AddChunk(Math.Max(bytes, Math.Max(_total, MinChunk)));
            last = _chunks.Count - 1; _off = 0;
        }
        var p = _chunks[last] + (nint)_off;
        _off += bytes;
        return new Span<T>((void*)p, count);
    }

    private void AddChunk(long bytes)
    {
        _chunks.Add((nint)NativeMemory.AlignedAlloc((nuint)bytes, 64));
        _sizes.Add(bytes); _total += bytes;
    }

    /// <summary>Rewinds the arena (all earlier allocations become invalid).</summary>
    public void Reset()
    {
        if (_chunks.Count > 1)
        {
            long total = _total;
            FreeAll();
            AddChunk(total);
        }
        _off = 0;
    }

    /// <summary>Frees everything if more than <paramref name="keepBytes"/> is held (the arena re-grows lazily).</summary>
    public void Trim(long keepBytes) { if (_total > keepBytes) { FreeAll(); _off = 0; } }

    private void FreeAll()
    {
        foreach (var c in _chunks) NativeMemory.AlignedFree((void*)c);
        _chunks.Clear(); _sizes.Clear(); _total = 0;
    }

    public void Dispose() => FreeAll();
}

/// <summary>Native scratch of one in-flight forward call: four arenas with different lifetimes plus the reusable block lists.</summary>
internal sealed class Inkling2Scratch : IDisposable
{
    /// <summary>Whole call: embeddings, the pushed blocks, the prelude output and the previous pass' output.</summary>
    public readonly Inkling2Arena Call = new();
    /// <summary>One layer visit's temporaries (rewound at the start of every visit).</summary>
    public readonly Inkling2Arena Layer = new();
    /// <summary>Loop messages of one core visit and the final output mix (rewound by the caller).</summary>
    public readonly Inkling2Arena Msg = new();
    /// <summary>Layer outputs, alternating: visit n writes <c>Out[n &amp; 1]</c> while reading the other.</summary>
    public readonly Inkling2Arena[] Out = [new Inkling2Arena(), new Inkling2Arena()];
    /// <summary>RMS-normalised block inputs (pointers into <see cref="Call"/>), immutable once pushed.</summary>
    public readonly List<nint> Blocks = new();
    /// <summary>The previous pass' bank blocks.</summary>
    public readonly List<nint> Bank = new();
    public int Visit;

    private const long KeepBytes = 512L << 20;

    public void Begin()
    {
        Call.Reset(); Layer.Reset(); Msg.Reset(); Out[0].Reset(); Out[1].Reset();
        Blocks.Clear(); Bank.Clear(); Visit = 0;
    }

    /// <summary>Called when the scratch goes back to the pool: bounds what a one-off huge prefill leaves resident.</summary>
    public void End()
    {
        Call.Trim(KeepBytes); Layer.Trim(KeepBytes); Msg.Trim(KeepBytes); Out[0].Trim(KeepBytes); Out[1].Trim(KeepBytes);
    }

    public void Dispose() { Call.Dispose(); Layer.Dispose(); Msg.Dispose(); Out[0].Dispose(); Out[1].Dispose(); }
}
