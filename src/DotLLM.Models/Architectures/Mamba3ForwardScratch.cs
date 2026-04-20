using System.Numerics;
using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Pre-allocated, 64-byte-aligned scratch buffers for the per-call temporaries
/// used by <see cref="Mamba3Block"/>.Forward and
/// <see cref="Mamba3Block.ForwardMimo"/>. Mirrors the
/// <see cref="NemotronHForwardState"/> pattern: one owned instance per model,
/// grown in power-of-two steps by <see cref="EnsureCapacity"/>, freed on
/// <see cref="Dispose"/>.
/// </summary>
/// <remarks>
/// <para>
/// Scratch versus state. <see cref="Mamba3State"/> owns cross-call recurrence
/// state (<c>ssm_state</c>, <c>cum_angle</c>) that persists between Forwards
/// on a single sequence. This class owns within-a-single-call temporaries
/// — <c>in_proj</c> output, split slices, DT / ADT / trap / gamma / scale,
/// the biased + broadcast B / C buffers, pre-RoPE qk dot, and the SSD scan
/// output — all of which were previously <c>new float[]</c>-allocated on
/// every <see cref="Mamba3Block"/>.Forward call.
/// </para>
/// <para>
/// Sizing is data-dependent on <c>seqLen</c>. The forward-scratch instance
/// is allocated with <see cref="EnsureCapacity(int)"/> growth: a fresh
/// instance is empty, the first Forward over T tokens rounds up to the next
/// power-of-two and allocates once; subsequent Forwards at the same-or-smaller
/// T reuse the buffers without further allocation. A Forward whose T exceeds
/// the current capacity frees and reallocates (still one allocation per
/// capacity jump — O(log T) lifetime allocations in the worst case).
/// </para>
/// <para>
/// Alias safety. Each named slot (<see cref="Proj"/>, <see cref="X"/>,
/// <see cref="Z"/>, ...) is a distinct 64-byte-aligned allocation. None of
/// them overlap. Callers receive <see cref="Span{Single}"/> views sized to
/// the caller's current T (not the capacity cap), so a Forward over T tokens
/// sees exactly T·stride elements even when the underlying backing is larger.
/// </para>
/// </remarks>
public sealed unsafe class Mamba3ForwardScratch : IDisposable
{
    // Dimensions — immutable after construction. Scratch capacity grows in T,
    // but per-token widths are fixed by the model config.
    private readonly int _dInProj;
    private readonly int _dInner;
    private readonly int _nHead;
    private readonly int _numRopeAngles;
    // B/C backing has width (R · H · N) where R ∈ {1, mimoRank}. We size for
    // the larger of the two so a single scratch instance handles both SISO
    // and MIMO paths of a given model config.
    private readonly int _bcWidth;

    private int _capacity;   // current max seqLen the buffers can hold.
    private bool _disposed;

    // Named buffer handles. All 64-byte aligned via NativeMemory.AlignedAlloc.
    private nint _proj;        // [T, d_in_proj]
    private nint _xBuf;        // [T, d_inner]
    private nint _zBuf;        // [T, d_inner]
    private nint _dt;          // [T, n_head]
    private nint _adt;         // [T, n_head]
    private nint _trap;        // [T, n_head]
    private nint _gamma;       // [T, n_head]
    private nint _scale;       // [T, n_head]
    private nint _anglesRaw;   // [T, num_rope_angles]
    private nint _bBuf;        // [T, R, H, d_state]  (SISO collapses R=1 to [T, H, N])
    private nint _cBuf;        // [T, R, H, d_state]
    private nint _qkPreDot;    // [T, n_head]
    private nint _yScan;       // [T, d_inner]

    /// <summary>Current scratch capacity in tokens (rounded to the next power of two of the largest Forward seen).</summary>
    public int Capacity => _capacity;

    /// <summary>Dimensions of a single token in the in-projection output.</summary>
    public int InProjWidth => _dInProj;

    /// <summary>Total bytes held by scratch at the current capacity.</summary>
    public long AllocatedBytes
    {
        get
        {
            long t = _capacity;
            if (t == 0) return 0;
            long floats = 0;
            floats += t * _dInProj;            // proj
            floats += t * _dInner * 3;         // xBuf, zBuf, yScan
            floats += t * _nHead * 6;          // dt, adt, trap, gamma, scale, qkPreDot
            floats += t * _numRopeAngles;      // anglesRaw
            floats += t * _bcWidth * 2;        // bBuf, cBuf
            return floats * sizeof(float);
        }
    }

    /// <summary>
    /// Constructs a zero-capacity scratch sized for the given Mamba-3 config.
    /// Buffers are allocated lazily on the first <see cref="EnsureCapacity(int)"/>
    /// call (or transitively, on the first <see cref="Mamba3Block"/>.Forward
    /// invocation that threads this scratch).
    /// </summary>
    /// <param name="config">Model config carrying the Mamba-3 hyperparameters.</param>
    /// <param name="initialCapacity">
    /// Optional initial capacity hint in tokens. If &gt; 0, the constructor
    /// immediately allocates to that capacity (rounded to the next power of
    /// two). Pass 0 (default) for fully lazy allocation — the first Forward
    /// will size on demand.
    /// </param>
    public Mamba3ForwardScratch(ModelConfig config, int initialCapacity = 0)
        : this(RequireM3(config), initialCapacity)
    {
    }

    private static Mamba3Config RequireM3(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config.Mamba3Config
            ?? throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated to allocate a Mamba3ForwardScratch.",
                nameof(config));
    }

    /// <summary>
    /// Constructs a zero-capacity scratch from an explicit <see cref="Mamba3Config"/>.
    /// Useful for tests that build a scratch without a full <see cref="ModelConfig"/>.
    /// </summary>
    public Mamba3ForwardScratch(Mamba3Config m3, int initialCapacity = 0)
    {
        ArgumentNullException.ThrowIfNull(m3);

        _dInner = m3.DInner;
        _nHead = m3.NumHeads;
        _numRopeAngles = m3.NumRopeAngles;

        // Scratch is sized for the wider of SISO (R=1) and MIMO (R=mimoRank)
        // so a single instance serves both paths of a given model.
        int effectiveRank = m3.IsMimo ? m3.MimoRank : 1;
        int bcPerToken = m3.StateSize * m3.NumGroups * effectiveRank;
        _dInProj = 2 * _dInner + 2 * bcPerToken + 3 * _nHead + _numRopeAngles;
        _bcWidth = effectiveRank * _nHead * m3.StateSize;

        if (_dInner <= 0 || _nHead <= 0 || _numRopeAngles <= 0 || _bcWidth <= 0)
            throw new ArgumentException(
                "Mamba3ForwardScratch requires positive dimensions — check Mamba3Config.",
                nameof(m3));

        _capacity = 0;
        if (initialCapacity > 0)
            EnsureCapacity(initialCapacity);
    }

    /// <summary>
    /// Constructs a zero-capacity scratch from raw Mamba-3 block dimensions.
    /// Useful for low-level kernel tests that don't carry a full
    /// <see cref="Mamba3Config"/>. The scratch is sized for the widest B/C
    /// rank the caller will use (pass <paramref name="mimoRank"/> = 1 for
    /// SISO-only).
    /// </summary>
    /// <param name="dInner">Inner dimension (<c>n_head · head_dim</c>).</param>
    /// <param name="nHead">Number of SSM heads.</param>
    /// <param name="dState">State width.</param>
    /// <param name="numBcHeads">B/C group count G (typically 1).</param>
    /// <param name="numRopeAngles">Number of rotated pairs S.</param>
    /// <param name="mimoRank">MIMO rank R; pass 1 for SISO.</param>
    /// <param name="initialCapacity">Optional initial capacity in tokens. 0 = lazy.</param>
    public static Mamba3ForwardScratch FromDimensions(
        int dInner, int nHead, int dState,
        int numBcHeads, int numRopeAngles, int mimoRank,
        int initialCapacity = 0)
    {
        if (dInner <= 0) throw new ArgumentOutOfRangeException(nameof(dInner));
        if (nHead <= 0) throw new ArgumentOutOfRangeException(nameof(nHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (numBcHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numBcHeads));
        if (numRopeAngles <= 0) throw new ArgumentOutOfRangeException(nameof(numRopeAngles));
        if (mimoRank <= 0) throw new ArgumentOutOfRangeException(nameof(mimoRank));

        return new Mamba3ForwardScratch(dInner, nHead, dState, numBcHeads, numRopeAngles, mimoRank, initialCapacity);
    }

    private Mamba3ForwardScratch(
        int dInner, int nHead, int dState,
        int numBcHeads, int numRopeAngles, int mimoRank,
        int initialCapacity)
    {
        _dInner = dInner;
        _nHead = nHead;
        _numRopeAngles = numRopeAngles;

        int bcPerToken = dState * numBcHeads * mimoRank;
        _dInProj = 2 * _dInner + 2 * bcPerToken + 3 * _nHead + _numRopeAngles;
        _bcWidth = mimoRank * _nHead * dState;

        _capacity = 0;
        if (initialCapacity > 0)
            EnsureCapacity(initialCapacity);
    }

    /// <summary>
    /// Grows every owned buffer so at least <paramref name="seqLen"/> tokens
    /// can be served without further allocation. Growth is power-of-two, so
    /// repeated Forwards at similar lengths amortise to zero allocation.
    /// </summary>
    /// <param name="seqLen">Required capacity in tokens.</param>
    public void EnsureCapacity(int seqLen)
    {
        ThrowIfDisposed();
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (seqLen <= _capacity) return;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeBuffers();

        _proj = AllocFloats((long)cap * _dInProj);
        _xBuf = AllocFloats((long)cap * _dInner);
        _zBuf = AllocFloats((long)cap * _dInner);
        _dt = AllocFloats((long)cap * _nHead);
        _adt = AllocFloats((long)cap * _nHead);
        _trap = AllocFloats((long)cap * _nHead);
        _gamma = AllocFloats((long)cap * _nHead);
        _scale = AllocFloats((long)cap * _nHead);
        _anglesRaw = AllocFloats((long)cap * _numRopeAngles);
        _bBuf = AllocFloats((long)cap * _bcWidth);
        _cBuf = AllocFloats((long)cap * _bcWidth);
        _qkPreDot = AllocFloats((long)cap * _nHead);
        _yScan = AllocFloats((long)cap * _dInner);

        _capacity = cap;
    }

    // ────────────────────────────────────────────────────────────────────────
    // Span accessors — sized to the caller's T, not the backing capacity.
    // Each asserts seqLen ≤ Capacity; callers invoke EnsureCapacity first.
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>In-projection output, shape <c>[T, d_in_proj]</c>.</summary>
    public Span<float> Proj(int seqLen) => Slice(_proj, seqLen, _dInProj);

    /// <summary>Per-token <c>x</c> split slice, shape <c>[T, d_inner]</c>.</summary>
    public Span<float> X(int seqLen) => Slice(_xBuf, seqLen, _dInner);

    /// <summary>Per-token <c>z</c> split slice (gate input), shape <c>[T, d_inner]</c>.</summary>
    public Span<float> Z(int seqLen) => Slice(_zBuf, seqLen, _dInner);

    /// <summary>Per-token per-head DT = softplus(dd_dt + dt_bias), shape <c>[T, n_head]</c>.</summary>
    public Span<float> Dt(int seqLen) => Slice(_dt, seqLen, _nHead);

    /// <summary>Per-token per-head <c>_A · DT</c>, shape <c>[T, n_head]</c>.</summary>
    public Span<float> Adt(int seqLen) => Slice(_adt, seqLen, _nHead);

    /// <summary>Per-token per-head <c>sigmoid(trap_raw)</c>, shape <c>[T, n_head]</c>.</summary>
    public Span<float> Trap(int seqLen) => Slice(_trap, seqLen, _nHead);

    /// <summary>Per-token per-head <c>γ = DT · trap</c>, shape <c>[T, n_head]</c>.</summary>
    public Span<float> Gamma(int seqLen) => Slice(_gamma, seqLen, _nHead);

    /// <summary>Per-token per-head <c>scale = γ + shifted_γ</c>, shape <c>[T, n_head]</c>.</summary>
    public Span<float> Scale(int seqLen) => Slice(_scale, seqLen, _nHead);

    /// <summary>Per-token raw angles (shared across heads), shape <c>[T, num_rope_angles]</c>.</summary>
    public Span<float> AnglesRaw(int seqLen) => Slice(_anglesRaw, seqLen, _numRopeAngles);

    /// <summary>Biased + broadcast B buffer. SISO: <c>[T, n_head, d_state]</c>. MIMO: <c>[T, R, n_head, d_state]</c>.</summary>
    public Span<float> B(int seqLen) => Slice(_bBuf, seqLen, _bcWidth);

    /// <summary>Biased + broadcast C buffer, same shape rules as <see cref="B(int)"/>.</summary>
    public Span<float> C(int seqLen) => Slice(_cBuf, seqLen, _bcWidth);

    /// <summary>Pre-RoPE <c>qk_pre_dot</c>, shape <c>[T, n_head]</c>. SISO stores the per-rank dot; MIMO stores Σ_r.</summary>
    public Span<float> QkPreDot(int seqLen) => Slice(_qkPreDot, seqLen, _nHead);

    /// <summary>SSD scan output, shape <c>[T, d_inner]</c>. Consumed by <c>out_proj</c>.</summary>
    public Span<float> YScan(int seqLen) => Slice(_yScan, seqLen, _dInner);

    // ────────────────────────────────────────────────────────────────────────
    // Internals
    // ────────────────────────────────────────────────────────────────────────

    private Span<float> Slice(nint ptr, int seqLen, int stride)
    {
        ThrowIfDisposed();
        if (seqLen <= 0)
            throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (seqLen > _capacity)
            throw new InvalidOperationException(
                $"Mamba3ForwardScratch requested {seqLen} tokens but capacity is {_capacity}. "
                + "Call EnsureCapacity(seqLen) before taking buffer slices.");
        return new Span<float>((float*)ptr, seqLen * stride);
    }

    private static nint AllocFloats(long count)
        => (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);

    private void FreeBuffers()
    {
        FreeIfNonZero(ref _proj);
        FreeIfNonZero(ref _xBuf);
        FreeIfNonZero(ref _zBuf);
        FreeIfNonZero(ref _dt);
        FreeIfNonZero(ref _adt);
        FreeIfNonZero(ref _trap);
        FreeIfNonZero(ref _gamma);
        FreeIfNonZero(ref _scale);
        FreeIfNonZero(ref _anglesRaw);
        FreeIfNonZero(ref _bBuf);
        FreeIfNonZero(ref _cBuf);
        FreeIfNonZero(ref _qkPreDot);
        FreeIfNonZero(ref _yScan);
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0)
        {
            NativeMemory.AlignedFree((void*)ptr);
            ptr = 0;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Mamba3ForwardScratch));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        FreeBuffers();
        _capacity = 0;
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>Finalizer — last-ditch free if the scratch was not disposed.</summary>
    ~Mamba3ForwardScratch()
    {
        if (_disposed) return;
        FreeBuffers();
    }
}
