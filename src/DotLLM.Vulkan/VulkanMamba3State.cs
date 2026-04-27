using DotLLM.Core.Models;
using DotLLM.Models.Architectures;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-local mirror of <see cref="DotLLM.Models.Architectures.Mamba3State"/>. Holds the
/// per-sequence Mamba-3 recurrent state (SSM hidden state + cumulative RoPE angle +
/// streaming-chunk K / V boundary buffers) on the Vulkan device for every layer of a
/// single sequence in zeroed device-local buffers.
/// </summary>
/// <remarks>
/// <para>
/// Per layer:
/// </para>
/// <list type="bullet">
///   <item><description>
///     <c>ssm_state</c> — SSM hidden state, <c>n_head · head_dim · d_state</c> F32
///     elements. Read/written by the SISO / MIMO scan kernels.
///   </description></item>
///   <item><description>
///     <c>cum_angle</c> — cumulative RoPE angle, <c>n_head · num_rope_angles</c> F32
///     elements. Read/written by the data-RoPE kernel via the <c>hasCumPrev</c> /
///     <c>writeCumOut</c> flags.
///   </description></item>
///   <item><description>
///     <c>k_state</c> — previous chunk's last-token post-RoPE (pre-scale) K. SISO layout
///     is <c>[n_head, d_state]</c>; MIMO expands to <c>[mimo_rank, n_head, d_state]</c>
///     (mirrors canonical <c>k_state (B, R, H, N)</c>). Read at chunk entry by the
///     boundary-adjustment kernel; written at chunk exit from
///     <c>VulkanMamba3ForwardScratch.B</c>'s last-token slice (post-RoPE B == K).
///   </description></item>
///   <item><description>
///     <c>v_state</c> — previous chunk's last-token V (= <c>x</c>),
///     <c>n_head · head_dim</c> F32 elements. Same lifecycle as <c>k_state</c>.
///   </description></item>
/// </list>
/// <para>
/// All four buffers are device-local and zero-initialised at construction. A fresh
/// state semantically represents "start of sequence"; the boundary adjustment is a
/// no-op when <c>k_state</c> / <c>v_state</c> are still zero (the dispatch happens but
/// adds zero — the orchestrator additionally short-circuits via
/// <see cref="HasBoundary"/> to skip the dispatch entirely on the first chunk).
/// </para>
/// <para>
/// <b>Ownership.</b> Single-sequence, non-paged. Caller owns disposal.
/// </para>
/// </remarks>
public sealed class VulkanMamba3State : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _numLayers;
    private readonly int _ssmStateElementsPerLayer;
    private readonly int _cumAngleElementsPerLayer;
    private readonly int _kStateElementsPerLayer;
    private readonly int _vStateElementsPerLayer;
    private readonly int _kRank;

    private readonly VulkanDevice.Buffer[] _ssmStateBuffers;
    private readonly VulkanDevice.Buffer[] _cumAngleBuffers;
    private readonly VulkanDevice.Buffer[] _kStateBuffers;
    private readonly VulkanDevice.Buffer[] _vStateBuffers;

    // Tracks whether the boundary buffers have been primed by a previous Forward.
    // Stays false for the first chunk of a sequence, then flips to true at the end
    // of the first chunk and remains true until Reset() clears the state. The
    // orchestrator uses this flag to decide whether to dispatch the boundary
    // adjustment kernel — skipping it on the first chunk preserves bit-equal
    // numerics with the CPU oracle (which ApplyChunkBoundaryAdjustment is also
    // a no-op for since kState/vState are all zero).
    private bool _hasBoundary;

    private bool _disposed;

    /// <summary>Number of Mamba-3 layers covered by this state.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>n_head · head_dim · d_state</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>n_head · num_rope_angles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>
    /// Elements per layer in the K state. SISO: <c>n_head · d_state</c> ([H, N]).
    /// MIMO: <c>mimo_rank · n_head · d_state</c> ([R, H, N]).
    /// </summary>
    public int KStateElementsPerLayer => _kStateElementsPerLayer;

    /// <summary>Elements per layer in the V state (<c>n_head · head_dim</c>).</summary>
    public int VStateElementsPerLayer => _vStateElementsPerLayer;

    /// <summary>
    /// K-state rank dimension: 1 for SISO, <c>mimo_rank</c> for MIMO. Matches the
    /// canonical <c>k_state (B, R, H, N)</c> layout's R axis.
    /// </summary>
    public int KStateRank => _kRank;

    /// <summary>
    /// True when the boundary buffers (<c>k_state</c>, <c>v_state</c>) have been
    /// written by a previous Forward call. The orchestrator checks this before
    /// dispatching the boundary-adjustment kernel — first chunk of a sequence
    /// (or post-<see cref="Reset"/>) skips the dispatch entirely.
    /// </summary>
    public bool HasBoundary => _hasBoundary;

    /// <summary>Total bytes allocated across all SSM state, cum-angle, K-state, and V-state buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer
                            + _kStateElementsPerLayer + _vStateElementsPerLayer) * sizeof(float);

    /// <summary>
    /// Allocates a zero-initialised persistent decode state for a Mamba-3 model with the
    /// given <paramref name="config"/>. <paramref name="config"/> must have
    /// <see cref="ModelConfig.Mamba3Config"/> populated.
    /// </summary>
    public VulkanMamba3State(VulkanDevice device, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        Mamba3Config m3 = config.Mamba3Config
            ?? throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated to allocate a VulkanMamba3State.",
                nameof(config));

        _device = device;
        _numLayers = config.NumLayers;
        _kRank = m3.IsMimo ? m3.MimoRank : 1;
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;
        _kStateElementsPerLayer = _kRank * m3.NumHeads * m3.StateSize;
        _vStateElementsPerLayer = m3.NumHeads * m3.HeadDim;

        if (_numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(config),
                $"ModelConfig.NumLayers must be > 0, got {_numLayers}.");
        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0
            || _kStateElementsPerLayer <= 0 || _vStateElementsPerLayer <= 0)
            throw new ArgumentException(
                "VulkanMamba3State requires positive ssm/cum_angle/k_state/v_state element counts; check Mamba3Config dims.",
                nameof(config));

        _ssmStateBuffers = new VulkanDevice.Buffer[_numLayers];
        _cumAngleBuffers = new VulkanDevice.Buffer[_numLayers];
        _kStateBuffers = new VulkanDevice.Buffer[_numLayers];
        _vStateBuffers = new VulkanDevice.Buffer[_numLayers];

        long ssmBytes = (long)_ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_vStateElementsPerLayer * sizeof(float);

        // Stage one zeroed buffer sized for the largest of the four — reused across every
        // device-local zero-init below. .NET zero-inits managed arrays for free, so the
        // host-side bytes are zero on first map.
        long maxBytes = Math.Max(Math.Max(ssmBytes, cumBytes), Math.Max(kBytes, vBytes));
        byte[] zeros = new byte[maxBytes];
        using var staging = device.Allocate(maxBytes);
        device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numLayers; i++)
        {
            _ssmStateBuffers[i] = device.AllocateDeviceLocal(ssmBytes);
            _cumAngleBuffers[i] = device.AllocateDeviceLocal(cumBytes);
            _kStateBuffers[i] = device.AllocateDeviceLocal(kBytes);
            _vStateBuffers[i] = device.AllocateDeviceLocal(vBytes);

            device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
            device.CopyBufferSynchronous(staging, _cumAngleBuffers[i], (ulong)cumBytes);
            device.CopyBufferSynchronous(staging, _kStateBuffers[i], (ulong)kBytes);
            device.CopyBufferSynchronous(staging, _vStateBuffers[i], (ulong)vBytes);
        }

        _hasBoundary = false;
    }

    /// <summary>
    /// Returns the SSM hidden-state device buffer for layer <paramref name="layerIndex"/>.
    /// Layout: <c>[n_head, head_dim, d_state]</c> row-major. Read/written by the SISO
    /// scan kernel.
    /// </summary>
    public VulkanDevice.Buffer GetSsmStateBuffer(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _ssmStateBuffers[layerIndex];
    }

    /// <summary>
    /// Returns the cumulative-RoPE-angle device buffer for layer <paramref name="layerIndex"/>.
    /// Layout: <c>[n_head, num_rope_angles]</c> row-major. Read/written by the data-RoPE
    /// kernel via the <c>hasCumPrev</c> / <c>writeCumOut</c> flag pair (we always pass both
    /// flags as true so the buffer cleanly threads across calls).
    /// </summary>
    public VulkanDevice.Buffer GetCumAngleBuffer(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _cumAngleBuffers[layerIndex];
    }

    /// <summary>
    /// Returns the K-state device buffer for layer <paramref name="layerIndex"/>. Holds
    /// the previous chunk's last-token post-RoPE (pre-scale) K. SISO layout
    /// <c>[n_head, d_state]</c>; MIMO <c>[mimo_rank, n_head, d_state]</c>.
    /// </summary>
    public VulkanDevice.Buffer GetKStateBuffer(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _kStateBuffers[layerIndex];
    }

    /// <summary>
    /// Returns the V-state device buffer for layer <paramref name="layerIndex"/>. Holds
    /// the previous chunk's last-token V (= <c>x</c>). Layout <c>[n_head, head_dim]</c>.
    /// </summary>
    public VulkanDevice.Buffer GetVStateBuffer(int layerIndex)
    {
        ThrowIfDisposed();
        if ((uint)layerIndex >= (uint)_numLayers)
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        return _vStateBuffers[layerIndex];
    }

    /// <summary>
    /// Marks the boundary buffers as primed. Called by the orchestrator at the END of
    /// each Forward call after writing this chunk's last-token K / V into every layer's
    /// <c>k_state</c> / <c>v_state</c> buffer.
    /// </summary>
    public void MarkBoundaryPrimed()
    {
        ThrowIfDisposed();
        _hasBoundary = true;
    }

    /// <summary>Re-zeroes every layer's state. Useful at the start of a fresh sequence.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numLayers == 0) return;

        long ssmBytes = (long)_ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_cumAngleElementsPerLayer * sizeof(float);
        long kBytes = (long)_kStateElementsPerLayer * sizeof(float);
        long vBytes = (long)_vStateElementsPerLayer * sizeof(float);
        long maxBytes = Math.Max(Math.Max(ssmBytes, cumBytes), Math.Max(kBytes, vBytes));

        byte[] zeros = new byte[maxBytes];
        using var staging = _device.Allocate(maxBytes);
        _device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numLayers; i++)
        {
            _device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
            _device.CopyBufferSynchronous(staging, _cumAngleBuffers[i], (ulong)cumBytes);
            _device.CopyBufferSynchronous(staging, _kStateBuffers[i], (ulong)kBytes);
            _device.CopyBufferSynchronous(staging, _vStateBuffers[i], (ulong)vBytes);
        }

        _hasBoundary = false;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VulkanMamba3State));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _numLayers; i++)
        {
            _ssmStateBuffers[i]?.Dispose();
            _cumAngleBuffers[i]?.Dispose();
            _kStateBuffers[i]?.Dispose();
            _vStateBuffers[i]?.Dispose();
        }
    }
}
