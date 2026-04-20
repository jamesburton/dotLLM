using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence Mamba-3 recurrent state — the four buffers per layer that must
/// persist across calls to <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
/// to resume mid-sequence decode with bit-equivalent output to a one-shot
/// forward.
/// </summary>
/// <remarks>
/// <para>
/// For each Mamba-3 layer the state holds:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <c>ssm_state</c> — canonical SSM hidden state, shape
///       <c>[n_head, head_dim, d_state]</c> row-major. Matches the layout
///       produced + consumed by the SISO / MIMO canonical scan kernels
///       (<c>Mamba3CanonicalSsd.ExecuteSiso</c> / <c>ExecuteMimo</c>).
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>cum_angle</c> — running cumulative RoPE angle (canonical
///       <c>angle_dt_state</c>), shape <c>[n_head, num_rope_angles]</c>
///       row-major. The data-dependent RoPE accumulates
///       <c>tanh(angles_raw[t, s]) · π · DT[t, h]</c> into this buffer per
///       token; wraps modulo 2π.
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>k_state</c> — the previous chunk's last-token post-RoPE,
///       pre-scale K tensor. For SISO the layout is
///       <c>[n_head, d_state]</c> row-major; for MIMO (rank <c>R &gt; 1</c>)
///       it expands to <c>[mimo_rank, n_head, d_state]</c> to mirror the
///       canonical <c>(batch, R, nheads, d_state)</c> <c>k_state</c> produced
///       by the MIMO combined kernel (see
///       <c>state-spaces/mamba</c> <c>mamba3.py:434-445</c> and
///       <c>mamba3_mimo_fwd.py:275-279</c>). Consumed at the NEXT chunk's
///       first token to reconstruct the
///       <c>shifted_γ[T_prev-1] = DT[0_new]·(1-trap[0_new])</c> term that a
///       one-shot forward would have folded in at the chunk edge. Matches
///       canonical <c>final_k_state</c> / <c>input_k_state</c> from
///       <c>mamba3_siso_fwd.py:318-322,341-343</c> (SISO) and the MIMO
///       rank-sum boundary analog (<c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>).
///     </description>
///   </item>
///   <item>
///     <description>
///       <c>v_state</c> — the previous chunk's last-token V (= <c>x</c>)
///       tensor, shape <c>[n_head, head_dim]</c> row-major. Paired with
///       <c>k_state</c> in the chunk-boundary adjustment:
///       <c>ssm += v_state · k_state · DT[0_new] · (1-trap[0_new])</c>.
///       Matches canonical <c>final_v_state</c> / <c>input_v_state</c>.
///     </description>
///   </item>
/// </list>
/// <para>
/// <b>Allocation.</b> All buffers live in unmanaged memory
/// (<see cref="NativeMemory.AlignedAlloc"/>, 64-byte alignment) and are
/// zero-initialised on construction — a fresh state semantically represents
/// "start of sequence". A 48-layer 370M checkpoint with
/// <c>n_head=32, head_dim=64, d_state=128, num_rope_angles=32</c> pre-allocates
/// <c>48 · (32·64·128 + 32·32 + 32·128 + 32·64) · 4 B ≈ 51 MB</c> per state.
/// </para>
/// <para>
/// <b>Ownership.</b> Single-sequence, non-paged. Caller owns disposal; the
/// model does not retain a reference after a <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
/// returns.
/// </para>
/// </remarks>
public sealed unsafe class Mamba3State : IDisposable
{
    private readonly int _numLayers;
    private readonly int _ssmStateElementsPerLayer;   // n_head * head_dim * d_state
    private readonly int _cumAngleElementsPerLayer;   // n_head * num_rope_angles
    private readonly int _kStateElementsPerLayer;     // n_head * d_state
    private readonly int _vStateElementsPerLayer;     // n_head * head_dim

    private nint _ssmState;
    private nint _cumAngle;
    private nint _kState;
    private nint _vState;
    private bool _disposed;

    /// <summary>Number of Mamba-3 layers covered by this state.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>n_head · head_dim · d_state</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>n_head · num_rope_angles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>
    /// Elements per layer in the K state. For SISO this is <c>n_head · d_state</c>
    /// (<c>[H, N]</c> layout); for MIMO this is <c>mimo_rank · n_head · d_state</c>
    /// (<c>[R, H, N]</c> layout) — the previous chunk's last-token post-RoPE K
    /// per rank, mirroring canonical <c>k_state (B, R, H, N)</c>.
    /// </summary>
    public int KStateElementsPerLayer => _kStateElementsPerLayer;

    /// <summary>Elements per layer in the V state (<c>n_head · head_dim</c>) — the previous chunk's last-token V.</summary>
    public int VStateElementsPerLayer => _vStateElementsPerLayer;

    /// <summary>Total bytes allocated across all state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer
                            + _kStateElementsPerLayer + _vStateElementsPerLayer) * sizeof(float);

    /// <summary>
    /// Allocates a zero-initialised persistent decode state for a Mamba-3 model
    /// with the given <paramref name="config"/>. <paramref name="config"/> must
    /// have <see cref="ModelConfig.Mamba3Config"/> populated.
    /// </summary>
    /// <param name="config">Model config carrying both the global layer count and the Mamba-3 hyperparameters.</param>
    public Mamba3State(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        Mamba3Config m3 = config.Mamba3Config
            ?? throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated to allocate a Mamba3State.",
                nameof(config));

        _numLayers = config.NumLayers;
        // MIMO k_state carries a rank axis (canonical mamba3.py:434-445):
        //   is_mimo=False → [H, N]   (R == 1 implicit)
        //   is_mimo=True  → [R, H, N]
        // Everything else (SSM state, cum_angle, v_state) is rank-free in the
        // canonical decode cache.
        int kRank = (m3.IsMimo ? m3.MimoRank : 1);
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;
        _kStateElementsPerLayer = kRank * m3.NumHeads * m3.StateSize;
        _vStateElementsPerLayer = m3.NumHeads * m3.HeadDim;

        if (_numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(config),
                $"ModelConfig.NumLayers must be > 0, got {_numLayers}.");
        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0
            || _kStateElementsPerLayer <= 0 || _vStateElementsPerLayer <= 0)
            throw new ArgumentException(
                "Mamba3State requires positive ssm/cum_angle/k_state/v_state element counts; check Mamba3Config dims.",
                nameof(config));

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_numLayers * _kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_numLayers * _vStateElementsPerLayer * sizeof(float);

        _ssmState = (nint)NativeMemory.AlignedAlloc((nuint)ssmBytes, 64);
        _cumAngle = (nint)NativeMemory.AlignedAlloc((nuint)cumBytes, 64);
        _kState = (nint)NativeMemory.AlignedAlloc((nuint)kBytes, 64);
        _vState = (nint)NativeMemory.AlignedAlloc((nuint)vBytes, 64);

        NativeMemory.Clear((void*)_ssmState, (nuint)ssmBytes);
        NativeMemory.Clear((void*)_cumAngle, (nuint)cumBytes);
        NativeMemory.Clear((void*)_kState, (nuint)kBytes);
        NativeMemory.Clear((void*)_vState, (nuint)vBytes);
    }

    /// <summary>
    /// Returns the SSM hidden-state slice for layer <paramref name="layerIndex"/>.
    /// Layout: <c>[n_head, head_dim, d_state]</c> row-major. Mutate in place —
    /// the block forward reads it at entry and writes it back at exit.
    /// </summary>
    public Span<float> SsmState(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return new Span<float>(
            (float*)_ssmState + (long)layerIndex * _ssmStateElementsPerLayer,
            _ssmStateElementsPerLayer);
    }

    /// <summary>
    /// Returns the cumulative-RoPE-angle slice for layer <paramref name="layerIndex"/>.
    /// Layout: <c>[n_head, num_rope_angles]</c> row-major. Mutate in place.
    /// </summary>
    public Span<float> CumAngle(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return new Span<float>(
            (float*)_cumAngle + (long)layerIndex * _cumAngleElementsPerLayer,
            _cumAngleElementsPerLayer);
    }

    /// <summary>
    /// Returns the K state slice for layer <paramref name="layerIndex"/>. This
    /// holds the previous chunk's last-token post-RoPE (pre-scale) K, to be
    /// consumed at the NEXT chunk's first token for the boundary adjustment
    /// <c>ssm += v_state · k_state · DT[0] · (1-trap[0])</c>.
    /// Layout: <c>[n_head, d_state]</c> row-major. Mutate in place.
    /// </summary>
    public Span<float> KState(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return new Span<float>(
            (float*)_kState + (long)layerIndex * _kStateElementsPerLayer,
            _kStateElementsPerLayer);
    }

    /// <summary>
    /// Returns the V state slice for layer <paramref name="layerIndex"/>. This
    /// holds the previous chunk's last-token V (= <c>x</c>). Paired with
    /// <see cref="KState(int)"/> in the boundary adjustment.
    /// Layout: <c>[n_head, head_dim]</c> row-major. Mutate in place.
    /// </summary>
    public Span<float> VState(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return new Span<float>(
            (float*)_vState + (long)layerIndex * _vStateElementsPerLayer,
            _vStateElementsPerLayer);
    }

    /// <summary>
    /// Zeroes every layer's SSM state, cumulative RoPE angle, K state, and V
    /// state. Call at the start of a fresh sequence when reusing an
    /// already-allocated <see cref="Mamba3State"/> instance.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numLayers == 0) return;
        NativeMemory.Clear(
            (void*)_ssmState,
            (nuint)((long)_numLayers * _ssmStateElementsPerLayer * sizeof(float)));
        NativeMemory.Clear(
            (void*)_cumAngle,
            (nuint)((long)_numLayers * _cumAngleElementsPerLayer * sizeof(float)));
        NativeMemory.Clear(
            (void*)_kState,
            (nuint)((long)_numLayers * _kStateElementsPerLayer * sizeof(float)));
        NativeMemory.Clear(
            (void*)_vState,
            (nuint)((long)_numLayers * _vStateElementsPerLayer * sizeof(float)));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_ssmState != 0) { NativeMemory.AlignedFree((void*)_ssmState); _ssmState = 0; }
        if (_cumAngle != 0) { NativeMemory.AlignedFree((void*)_cumAngle); _cumAngle = 0; }
        if (_kState != 0) { NativeMemory.AlignedFree((void*)_kState); _kState = 0; }
        if (_vState != 0) { NativeMemory.AlignedFree((void*)_vState); _vState = 0; }
        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(Mamba3State));
    }

    /// <summary>Finalizer — last-ditch free if the state was not disposed.</summary>
    ~Mamba3State()
    {
        if (_disposed) return;
        if (_ssmState != 0) NativeMemory.AlignedFree((void*)_ssmState);
        if (_cumAngle != 0) NativeMemory.AlignedFree((void*)_cumAngle);
        if (_kState != 0) NativeMemory.AlignedFree((void*)_kState);
        if (_vState != 0) NativeMemory.AlignedFree((void*)_vState);
    }
}
