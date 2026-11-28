using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures.DSpark;

/// <summary>Receives the target's per-visit layer inputs during a forward (the DSpark drafter's context features).</summary>
internal interface IInkling2Tap
{
    /// <summary>Called with the input rows <c>[T][hidden]</c> of layer visit <paramref name="visit"/> (the last visit is the final normed output).</summary>
    void Capture(int visit, ReadOnlySpan<float> rows, int rowCount);
}

/// <summary>
/// Per-sequence DSpark state (issue #2039), driven through <see cref="IMtpState"/> so the MTP speculative decoder, its adaptive gate and streaming are reused:
/// a target forward carrying this state captures the tap rows (F16) and injects them into the drafter's per-layer context K/V; the first
/// <c>ForwardMtpArgMax</c> of a round drafts the whole block, later calls serve it; <see cref="Rollback"/> crops the context.
/// </summary>
/// <remarks>
/// Context K/V is F16 and windowed (issue #2060): the local (sliding-window) drafter layers keep a RING of <c>window + InjectChunk</c> positions, flat in the sequence
/// length (a block sees <c>pos0 - 1024 .. pos0 - 1</c>; the slack covers the rows a rejected verify wrote ahead and an injection chunk); only the global layer keeps the
/// whole context, in a buffer that grows on demand and is ADMITTED against the memory still available to the process (when it is not, the state turns
/// <see cref="Unusable"/>, logs once and drafts nothing useful: it never spills silently). Steady-state bytes per position: <c>kvDim x 2 x 2</c> (4 KiB for Inkling-2-Mini, was 12 KiB)
/// plus a fixed ring. The <c>f_rnn</c> rows are F16 in a small ring.
/// </remarks>
internal sealed unsafe class Inkling2DSparkState : IMtpState, IInkling2Tap
{
    private readonly Inkling2DSparkDrafter _drafter;
    private readonly int _kvDim, _h;
    private readonly int[] _slotOfVisit;
    private readonly NativeBuffer<ushort>[] _k, _v;   // per layer, [capacity][kvDim] Half bits
    private int _capacity;   // positions of the global layers' linear buffers
    private readonly int _ringPositions;
    private readonly bool[] _local;
    private readonly NativeBuffer<ushort> _frnn;
    private readonly int[] _frnnPos;
    private Half* _taps;
    private int _tapRowCapacity;
    private int _length;
    private readonly int[] _draft;
    private int _draftStart = -1, _draftCount;
    private bool _disposed;

    internal Inkling2DSparkState(Inkling2DSparkDrafter drafter, int visits, int ringPositions = 0)
    {
        _drafter = drafter;
        _kvDim = drafter.NumKvHeads * drafter.HeadDim;
        _h = drafter.HiddenSize;
        _slotOfVisit = drafter.SlotForVisit(visits);
        _k = new NativeBuffer<ushort>[drafter.NumLayers];
        _v = new NativeBuffer<ushort>[drafter.NumLayers];
        _ringPositions = ringPositions > 0 ? ringPositions : drafter.Window + Inkling2DSparkDrafter.InjectChunkRows;
        _local = new bool[drafter.NumLayers];
        for (int i = 0; i < _k.Length; i++)
        {
            _local[i] = drafter.IsLocalLayer(i);
            _k[i] = new NativeBuffer<ushort>(_local[i] ? (long)_ringPositions * _kvDim : 0);
            _v[i] = new NativeBuffer<ushort>(_local[i] ? (long)_ringPositions * _kvDim : 0);
        }
        _frnn = new NativeBuffer<ushort>((long)drafter.FrnnRows * _h);
        _frnnPos = new int[drafter.FrnnRows];
        Array.Fill(_frnnPos, -1);
        _draft = new int[drafter.BlockSize - 1];
    }

    /// <summary>The drafter this state belongs to.</summary>
    public Inkling2DSparkDrafter Drafter => _drafter;

    /// <summary>Positions the ring layers keep.</summary>
    internal int RingPositions => _ringPositions;

    /// <summary>Ring size of <paramref name="layer"/> (0 = linear: the global layer).</summary>
    internal int RingFor(int layer) => _local[layer] ? _ringPositions : 0;

    /// <summary>True once the global layer's context could not be admitted; the state then drafts nothing useful (see the type remarks).</summary>
    internal bool Unusable { get; private set; }

    /// <summary>Test hook: replaces the memory-admission check (bytes still wanted -> admitted).</summary>
    internal static Func<long, bool>? AdmissionOverride { get; set; }

    /// <summary>Resident bytes of the context K/V.</summary>
    public long KvBytes { get { long b = 0; foreach (var k in _k) b += k.Bytes * 2; return b; } }

    /// <inheritdoc/>
    public int CurrentLength => _length;
    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => ReadOnlySpan<float>.Empty;
    /// <inheritdoc/>
    public int CapturedRowCount => 0;
    /// <inheritdoc/>
    public int HiddenSize => _h;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if (length < 0 || length > _length) throw new ArgumentOutOfRangeException(nameof(length));
        _length = length;
        _draftStart = -1;
    }

    /// <inheritdoc/>
    /// <remarks>DSpark conditions on the injected context, not on a carried hidden row: nothing to seed.</remarks>
    public void SeedFromCapturedRow(int rowIndex) { }

    /// <summary>First context position the drafter may read (0: positions are indexed directly and everything below the length is present).</summary>
    internal int FirstContextPosition => 0;

    internal Half* Keys(int layer) => (Half*)_k[layer].Ptr;
    internal Half* Values(int layer) => (Half*)_v[layer].Ptr;

    private bool EnsureCapacity(int positions)
    {
        if (positions <= _capacity) return true;
        if (Unusable) return false;
        int next = Math.Max(positions, Math.Max(256, _capacity * 2));
        int globals = 0;
        for (int i = 0; i < _k.Length; i++) if (!_local[i]) globals++;
        long extra = (long)(next - _capacity) * _kvDim * 2 * 2 * globals;   // K and V, F16
        if (!Admit(extra))
        {
            Unusable = true;
            Console.Error.WriteLine($"[dspark] drafter context disabled: the global layer's K/V for {next} positions needs {extra / 1048576} MiB more and the host does not have it free; decoding continues without useful drafts.");
            return false;
        }
        for (int i = 0; i < _k.Length; i++) if (!_local[i]) { _k[i].Grow((long)next * _kvDim); _v[i].Grow((long)next * _kvDim); }
        _capacity = next;
        return true;
    }

    private static bool Admit(long extraBytes)
    {
        if (AdmissionOverride is { } o) return o(extraBytes);
        long total = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        long used = Environment.WorkingSet;
        return extraBytes <= total - used - (512L << 20);
    }

    /// <summary>Writes F32 K/V rows <c>[n][kvDim]</c> (post-norm, post-RoPE K) of <paramref name="positions"/> into layer <paramref name="layer"/> as F16.</summary>
    internal void WriteKv(int layer, ReadOnlySpan<int> positions, float* k, float* v)
    {
        if (!_local[layer])
        {
            int max = 0;
            foreach (int p in positions) max = Math.Max(max, p);
            if (!EnsureCapacity(max + 1)) return;
        }
        int ring = RingFor(layer);
        for (int t = 0; t < positions.Length; t++)
        {
            long off = (long)(ring == 0 ? positions[t] : positions[t] % ring) * _kvDim;
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(k + (long)t * _kvDim, _kvDim), new Span<Half>((Half*)_k[layer].Ptr + off, _kvDim));
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>(v + (long)t * _kvDim, _kvDim), new Span<Half>((Half*)_v[layer].Ptr + off, _kvDim));
        }
    }

    internal void StoreFrnn(int pos, ReadOnlySpan<float> row)
    {
        int slot = pos % _frnnPos.Length;
        TensorPrimitives.ConvertToHalf(row, new Span<Half>((Half*)_frnn.Ptr + (long)slot * _h, _h));
        _frnnPos[slot] = pos;
    }

    /// <summary>The <c>f_rnn</c> row of <paramref name="pos"/> as F32 (a copy in a reused buffer); false when that position was never injected.</summary>
    internal bool TryGetFrnn(int pos, out ReadOnlySpan<float> row)
    {
        row = default;
        if (pos < 0) return false;
        int slot = pos % _frnnPos.Length;
        if (_frnnPos[slot] != pos) return false;
        _frnnF32 ??= new NativeBuffer<float>(_h);
        TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>((Half*)_frnn.Ptr + (long)slot * _h, _h), _frnnF32.Span);
        row = _frnnF32.Span;
        return true;
    }

    private NativeBuffer<float>? _frnnF32;

    internal void NoteInjected(int pos) { if (pos + 1 > _length) _length = pos + 1; }

    /// <summary>Tap rows <c>[rows, TapRowWidth]</c> for the coming forward, grown to <paramref name="rows"/>.</summary>
    internal Half* EnsureTapRows(int rows)
    {
        if (rows > _tapRowCapacity)
        {
            if (_taps != null) NativeMemory.AlignedFree(_taps);
            _taps = (Half*)NativeMemory.AlignedAlloc((nuint)((long)rows * _drafter.TapRowWidth * sizeof(ushort)), 64);
            _tapRowCapacity = rows;
        }
        // A tap slot that a forward never fills must read as zeros, not as whatever the allocation held.
        new Span<Half>(_taps, checked(rows * _drafter.TapRowWidth)).Clear();
        return _taps;
    }

    /// <summary>Releases a tap buffer a long prefill grew beyond a verify's needs.</summary>
    internal void TrimTapRows()
    {
        if (_tapRowCapacity <= 4 * _drafter.BlockSize) return;
        NativeMemory.AlignedFree(_taps);
        _taps = null;
        _tapRowCapacity = 0;
    }

    /// <inheritdoc/>
    public void Capture(int visit, ReadOnlySpan<float> rows, int rowCount)
    {
        if ((uint)visit >= (uint)_slotOfVisit.Length) return;
        int slot = _slotOfVisit[visit];
        if (slot < 0) return;
        for (int r = 0; r < rowCount; r++)
            TensorPrimitives.ConvertToHalf(rows.Slice(r * _h, _h), new Span<Half>(_taps + (long)r * _drafter.TapRowWidth + (long)slot * _h, _h));
    }

    /// <summary>Injects the first <paramref name="positions"/>.Length captured rows.</summary>
    internal void InjectCaptured(ReadOnlySpan<int> positions)
    {
        _draftStart = -1;
        _drafter.Inject(this, _taps, positions);
    }

    /// <summary>The DSpark side of <c>ForwardMtpArgMax</c>: at the context end drafts the block anchored at <paramref name="tokenId"/>, else serves the cached block.</summary>
    internal int NextDraft(IInkling2DSparkTarget target, int tokenId, int position, int draftTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (Unusable) return tokenId;   // no admitted context: the anchor repeated for every position (the verify rejects it; correctness is unaffected)
        if (position == _length)
        {
            _draftCount = Math.Clamp(draftTokens, 1, _draft.Length);
            _drafter.Draft(this, target, tokenId, position, _draft.AsSpan(0, _draftCount));
            _draftStart = position;
            return _draft[0];
        }
        int i = position - _draftStart;
        if (_draftStart < 0 || i < 0 || i >= _draftCount)
            throw new InvalidOperationException($"DSpark draft step at position {position} has no block (context length {_length}, block at {_draftStart}, {_draftCount} drafts).");
        return _draft[i];
    }

    /// <summary>The (softcapped) logits row of the draft at <paramref name="position"/> + 1, after <see cref="NextDraft"/>.</summary>
    internal ReadOnlySpan<float> DraftLogits(int position) => _drafter.LastDraftLogits(position - _draftStart);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _k) b.Dispose();
        foreach (var b in _v) b.Dispose();
        _frnn.Dispose();
        _frnnF32?.Dispose();
        if (_taps != null) { NativeMemory.AlignedFree(_taps); _taps = null; }
    }
}
