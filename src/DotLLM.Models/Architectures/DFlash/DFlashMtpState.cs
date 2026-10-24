using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures.DFlash;

/// <summary>
/// Per-sequence DFlash state (issue #1175), driven through the <see cref="IMtpState"/> protocol so the MTP speculative
/// decoder, its gate, streaming, KV rollback and recurrent per-row snapshots are reused unchanged:
/// a target <c>Forward</c> carrying this state captures the tap rows and injects them into the drafter's context K/V
/// (the MTP "absorb"); the first <c>ForwardMtpArgMax</c> of a round drafts the whole block, later calls serve it;
/// <see cref="Rollback"/> crops the context exactly like the MTP head's KV.
/// </summary>
/// <remarks>
/// The context K/V is an F16 ring of <see cref="RingCapacity"/> rows per drafter layer: the drafter attends over a sliding
/// window, and the slack beyond the window covers a verify's worth of rows that a rollback can take back. Bytes:
/// <c>layers x capacity x kvDim x 2 (K, V) x 2</c> (Qwen3.8-27B DFlash2: 5 x 2064 x 1024 x 4 = 42 MB).
/// </remarks>
internal sealed unsafe class DFlashMtpState : IMtpState
{
    private readonly DFlashDrafter _drafter;
    private readonly int _kvDim;
    private nint _ring;            // [layer][K|V][capacity][kvDim] Half
    private Half* _taps;           // [tapRowCapacity][TapRowWidth] Half, grown on demand
    private int _tapRowCapacity;
    private int _length;
    private int _maxInjected = -1; // highest position ever written: ring slots below maxInjected - capacity are stale
    private readonly int[] _draft;
    private nint _anchors;           // LiLiCorr (#1237): [AnchorRows][hidden] F32 fused context features, backend memory
    private readonly int[] _anchorPos = [];
    private int _draftStart = -1;  // position of the anchor of the cached block, -1 = none
    private int _draftCount;
    private bool _disposed;

    // DOTLLM_DFLASH_TRACE=1: per-request draft / inject wall time on stderr at dispose.
    private static readonly bool Trace = Environment.GetEnvironmentVariable("DOTLLM_DFLASH_TRACE") == "1";
    private long _draftTicks, _injectTicks;
    private int _drafts, _injects, _injectRows;

    /// <summary>Creates a state for <paramref name="drafter"/>.</summary>
    /// <param name="drafter">The drafter whose context this state holds.</param>
    public DFlashMtpState(DFlashDrafter drafter)
    {
        _drafter = drafter;
        _kvDim = drafter.NumKvHeads * drafter.HeadDim;
        RingCapacity = (int)Math.Min((long)drafter.Window + 2L * drafter.BlockSize, 1 << 20);
        long halves = (long)drafter.NumLayers * 2 * RingCapacity * _kvDim;
        _ring = drafter.Backend.AllocHalf(halves);   // host or device, the drafter's backend decides
        RingBytes = halves * sizeof(ushort);
        _draft = new int[drafter.BlockSize - 1];
        if (drafter.IsLiLiCorr)
        {
            _anchors = drafter.Backend.Alloc((long)drafter.AnchorRows * drafter.HiddenSize);
            _anchorPos = new int[drafter.AnchorRows];
            Array.Fill(_anchorPos, -1);
        }
    }

    /// <summary>Bytes of the LiLiCorr anchor-feature ring (0 for other drafters).</summary>
    public long AnchorBytes => _anchors == 0 ? 0 : (long)_anchorPos.Length * _drafter.HiddenSize * sizeof(float);

    /// <summary>Keeps the fused context feature row at <paramref name="feature"/> (backend memory) as the anchor of <paramref name="pos"/>.</summary>
    /// <param name="feature">The row.</param>
    /// <param name="pos">Its position.</param>
    internal void StoreAnchor(nint feature, int pos)
    {
        int slot = pos % _anchorPos.Length;
        _drafter.Backend.Copy(feature, _anchors + (nint)((long)slot * _drafter.HiddenSize * sizeof(float)), _drafter.HiddenSize);
        _anchorPos[slot] = pos;
    }

    /// <summary>The anchor feature of <paramref name="pos"/> into <paramref name="dst"/>; zeros when that position was never injected.</summary>
    /// <param name="pos">Position (the one before a block's anchor token).</param>
    /// <param name="dst">Host destination, hidden wide.</param>
    internal void LoadAnchor(int pos, float[] dst)
    {
        int slot = pos < 0 ? -1 : pos % _anchorPos.Length;
        if (slot < 0 || _anchorPos[slot] != pos)
        {
            Array.Clear(dst);
            return;
        }
        fixed (float* p = dst)
            _drafter.Backend.Download(_anchors + (nint)((long)slot * _drafter.HiddenSize * sizeof(float)), p, _drafter.HiddenSize);
    }

    /// <summary>Rows per layer of the context ring.</summary>
    public int RingCapacity { get; }

    /// <summary>Bytes of the context ring.</summary>
    public long RingBytes { get; }

    /// <summary>Bytes of the tap capture buffer currently held.</summary>
    public long TapBytes => (long)_tapRowCapacity * _drafter.TapRowWidth * sizeof(ushort);

    /// <summary>The drafter this state belongs to.</summary>
    public DFlashDrafter Drafter => _drafter;

    /// <inheritdoc/>
    public int CurrentLength => _length;

    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => ReadOnlySpan<float>.Empty;

    /// <inheritdoc/>
    public int CapturedRowCount => 0;

    /// <inheritdoc/>
    public int HiddenSize => _drafter.HiddenSize;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if (length < 0 || length > _length) throw new ArgumentOutOfRangeException(nameof(length));
        _length = length;
    }

    /// <inheritdoc/>
    /// <remarks>DFlash conditions on the injected context, not on a carried hidden row: nothing to seed.</remarks>
    public void SeedFromCapturedRow(int rowIndex) { }

    /// <summary>Tap rows <c>[rows, TapRowWidth]</c> for the coming forward, grown to <paramref name="rows"/>.</summary>
    /// <param name="rows">Rows the forward will capture.</param>
    /// <returns>The buffer.</returns>
    public Half* EnsureTapRows(int rows)
    {
        if (rows > _tapRowCapacity)
        {
            if (_taps != null) NativeMemory.AlignedFree(_taps);
            _taps = (Half*)NativeMemory.AlignedAlloc((nuint)((long)rows * _drafter.TapRowWidth * sizeof(ushort)), 64);
            _tapRowCapacity = rows;
        }
        return _taps;
    }

    /// <summary>Releases a tap buffer a long prefill grew beyond a verify's needs.</summary>
    public void TrimTapRows()
    {
        if (_tapRowCapacity <= 4 * _drafter.BlockSize) return;
        NativeMemory.AlignedFree(_taps);
        _taps = null;
        _tapRowCapacity = 0;
    }

    /// <summary>Copies one target layer input into tap slot <paramref name="slot"/> of <paramref name="row"/>.</summary>
    /// <param name="row">Captured row.</param>
    /// <param name="slot">Index into <see cref="DFlashDrafter.TargetLayers"/>.</param>
    /// <param name="hidden">The target hidden row.</param>
    public void CaptureTap(int row, int slot, ReadOnlySpan<float> hidden)
        => TensorPrimitives.ConvertToHalf(hidden,
            new Span<Half>(_taps + (long)row * _drafter.TapRowWidth + (long)slot * _drafter.HiddenSize, _drafter.HiddenSize));

    /// <summary>Injects the first <paramref name="positions"/>.Length captured rows (llama.cpp's KV-injection pass).</summary>
    /// <param name="positions">Position of each captured row.</param>
    public void InjectCaptured(ReadOnlySpan<int> positions)
    {
        _draftStart = -1;   // the context changed: any cached block is stale
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        _drafter.Inject(this, _taps, positions);
        _injectTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
        _injects++;
        _injectRows += positions.Length;
    }

    internal nint RingKeys(int layer) => _ring + (nint)((long)layer * 2 * RingCapacity * _kvDim * sizeof(ushort));

    internal nint RingValues(int layer) => RingKeys(layer) + (nint)((long)RingCapacity * _kvDim * sizeof(ushort));

    internal void NoteInjected(int pos)
    {
        if (pos > _maxInjected) _maxInjected = pos;
        if (pos + 1 > _length) _length = pos + 1;
    }

    /// <summary>First context position a block anchored at <paramref name="start"/> may read: inside the window and not yet overwritten.</summary>
    internal int FirstVisible(int start, int window)
        => Math.Max(Math.Max(0, start - window + 1), _maxInjected - RingCapacity + 1);

    /// <summary>
    /// The DFlash side of <c>ForwardMtpArgMax</c>: at the context end (a new round) drafts the block anchored at
    /// <paramref name="tokenId"/>, otherwise serves the cached block.
    /// </summary>
    /// <param name="target">Target ops.</param>
    /// <param name="tokenId">The token at <paramref name="position"/>.</param>
    /// <param name="position">Its position.</param>
    /// <param name="draftTokens">Drafts to produce for a new block (1..BlockSize-1).</param>
    /// <returns>The draft for <paramref name="position"/> + 1.</returns>
    internal int NextDraft(IDFlashTargetOps target, int tokenId, int position, int draftTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (position == _length)
        {
            _draftCount = Math.Clamp(draftTokens, 1, _draft.Length);
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            _drafter.Draft(this, target, tokenId, position, _draft.AsSpan(0, _draftCount));
            _draftTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
            _drafts++;
            _draftStart = position;
            return _draft[0];
        }
        int i = position - _draftStart;
        if (_draftStart < 0 || i < 0 || i >= _draftCount)
            throw new InvalidOperationException(
                $"DFlash draft step at position {position} has no block (context length {_length}, block at {_draftStart}, {_draftCount} drafts).");
        return _draft[i];
    }

    /// <summary>The drafter's logits row for the draft at <paramref name="position"/> + 1, after <see cref="NextDraft"/>.</summary>
    internal ReadOnlySpan<float> DraftLogits(int position) => _drafter.LastDraftLogits(position - _draftStart);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (Trace && (_drafts > 0 || _injects > 0))
        {
            double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            var p = _drafter.TakeProfile();
            Console.Error.WriteLine($"[dflash] drafts={_drafts} draft_ms={_draftTicks * f / Math.Max(1, _drafts):F1}/draft " +
                $"injects={_injects} rows={_injectRows} inject_ms={_injectTicks * f:F0} total ({_injectTicks * f / Math.Max(1, _injects):F1}/inject) " +
                $"split_ms gemm={p.Gemm:F0} attn={p.Attn:F0} head={p.Head:F0} select={p.Select:F0}");
        }
        if (_ring != 0) { _drafter.Backend.Free(_ring); _ring = 0; }
        if (_anchors != 0) { _drafter.Backend.Free(_anchors); _anchors = 0; }
        if (_taps != null) { NativeMemory.AlignedFree(_taps); _taps = null; }
    }
}
