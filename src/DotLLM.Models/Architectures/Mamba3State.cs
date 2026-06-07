using System.Runtime.InteropServices;
using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Per-sequence Mamba-3 recurrent state — the two buffers per layer that must
/// persist across calls to <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
/// to resume mid-sequence decode.
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
/// </list>
/// <para>
/// <b>Allocation.</b> Both buffers live in unmanaged memory
/// (<see cref="NativeMemory.AlignedAlloc"/>, 64-byte alignment) and are
/// zero-initialised on construction — a fresh state semantically represents
/// "start of sequence". A 48-layer 370M checkpoint with
/// <c>n_head=32, head_dim=64, d_state=128, num_rope_angles=32</c> pre-allocates
/// <c>48 · (32·64·128 + 32·32) · 4 B ≈ 50 MB</c> per state.
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

    private nint _ssmState;
    private nint _cumAngle;
    private bool _disposed;

    /// <summary>Number of Mamba-3 layers covered by this state.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>n_head · head_dim · d_state</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>n_head · num_rope_angles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>Total bytes allocated across both state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer) * sizeof(float);

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
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;

        if (_numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(config),
                $"ModelConfig.NumLayers must be > 0, got {_numLayers}.");
        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0)
            throw new ArgumentException(
                "Mamba3State requires positive ssm/cum_angle element counts; check Mamba3Config dims.",
                nameof(config));

        long ssmBytes = (long)_numLayers * _ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_numLayers * _cumAngleElementsPerLayer * sizeof(float);

        _ssmState = (nint)NativeMemory.AlignedAlloc((nuint)ssmBytes, 64);
        _cumAngle = (nint)NativeMemory.AlignedAlloc((nuint)cumBytes, 64);

        NativeMemory.Clear((void*)_ssmState, (nuint)ssmBytes);
        NativeMemory.Clear((void*)_cumAngle, (nuint)cumBytes);
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
    /// Zeroes every layer's SSM state and cumulative RoPE angle. Call at the
    /// start of a fresh sequence when reusing an already-allocated
    /// <see cref="Mamba3State"/> instance.
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
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        if (_ssmState != 0) { NativeMemory.AlignedFree((void*)_ssmState); _ssmState = 0; }
        if (_cumAngle != 0) { NativeMemory.AlignedFree((void*)_cumAngle); _cumAngle = 0; }
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
    }
}
