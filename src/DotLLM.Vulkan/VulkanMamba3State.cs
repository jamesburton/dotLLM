using DotLLM.Core.Models;
using DotLLM.Models.Architectures;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-local mirror of <see cref="DotLLM.Models.Architectures.Mamba3State"/>. Holds the
/// per-sequence Mamba-3 recurrent state (SSM hidden state + cumulative RoPE angle) on the
/// Vulkan device for every layer of a single sequence in zeroed device-local buffers.
/// </summary>
/// <remarks>
/// <para>
/// One <c>ssm_state</c> buffer of <c>n_head · head_dim · d_state</c> F32 elements per layer
/// and one <c>cum_angle</c> buffer of <c>n_head · num_rope_angles</c> F32 elements per layer.
/// Both are allocated device-local and zero-initialised at construction (a fresh state
/// represents start-of-sequence). Mirrors <see cref="VulkanSsmStateCache"/> for layout/
/// lifetime patterns; the Mamba-3 SISO path keeps state continuity through
/// <see cref="DotLLM.Vulkan.Kernels.Mamba3CanonicalSsdSisoF32Kernel"/> (ssm_state) and
/// <see cref="DotLLM.Vulkan.Kernels.Mamba3DataRopeF32Kernel"/> (cum_angle).
/// </para>
/// <para>
/// SISO only — <c>k_state</c> / <c>v_state</c> chunk-boundary buffers from the CPU
/// reference are NOT mirrored here. The first-cut Vulkan path uses each Forward as one
/// chunk (no split-chunk schedule) so the canonical <c>shifted_γ[T-1] = 0</c> boundary is
/// already implicit in the SSD scan; cross-chunk boundary correction is a follow-up if
/// streaming-decode parity becomes a regression bar.
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

    private readonly VulkanDevice.Buffer[] _ssmStateBuffers;
    private readonly VulkanDevice.Buffer[] _cumAngleBuffers;

    private bool _disposed;

    /// <summary>Number of Mamba-3 layers covered by this state.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Elements per layer in the SSM hidden state (<c>n_head · head_dim · d_state</c>).</summary>
    public int SsmStateElementsPerLayer => _ssmStateElementsPerLayer;

    /// <summary>Elements per layer in the cumulative RoPE angle buffer (<c>n_head · num_rope_angles</c>).</summary>
    public int CumAngleElementsPerLayer => _cumAngleElementsPerLayer;

    /// <summary>Total bytes allocated across all SSM state and cum-angle buffers.</summary>
    public long AllocatedBytes =>
        (long)_numLayers * (_ssmStateElementsPerLayer + _cumAngleElementsPerLayer) * sizeof(float);

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
        _ssmStateElementsPerLayer = m3.NumHeads * m3.HeadDim * m3.StateSize;
        _cumAngleElementsPerLayer = m3.NumHeads * m3.NumRopeAngles;

        if (_numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(config),
                $"ModelConfig.NumLayers must be > 0, got {_numLayers}.");
        if (_ssmStateElementsPerLayer <= 0 || _cumAngleElementsPerLayer <= 0)
            throw new ArgumentException(
                "VulkanMamba3State requires positive ssm/cum_angle element counts; check Mamba3Config dims.",
                nameof(config));

        _ssmStateBuffers = new VulkanDevice.Buffer[_numLayers];
        _cumAngleBuffers = new VulkanDevice.Buffer[_numLayers];

        long ssmBytes = (long)_ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_cumAngleElementsPerLayer * sizeof(float);

        // Stage one zeroed buffer sized for the larger of the two — reused across every
        // device-local zero-init below. .NET zero-inits managed arrays for free, so the
        // host-side bytes are zero on first map.
        long maxBytes = Math.Max(ssmBytes, cumBytes);
        byte[] zeros = new byte[maxBytes];
        using var staging = device.Allocate(maxBytes);
        device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numLayers; i++)
        {
            _ssmStateBuffers[i] = device.AllocateDeviceLocal(ssmBytes);
            _cumAngleBuffers[i] = device.AllocateDeviceLocal(cumBytes);

            device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
            device.CopyBufferSynchronous(staging, _cumAngleBuffers[i], (ulong)cumBytes);
        }
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

    /// <summary>Re-zeroes every layer's state. Useful at the start of a fresh sequence.</summary>
    public void Reset()
    {
        ThrowIfDisposed();
        if (_numLayers == 0) return;

        long ssmBytes = (long)_ssmStateElementsPerLayer * sizeof(float);
        long cumBytes = (long)_cumAngleElementsPerLayer * sizeof(float);
        long maxBytes = Math.Max(ssmBytes, cumBytes);

        byte[] zeros = new byte[maxBytes];
        using var staging = _device.Allocate(maxBytes);
        _device.Upload(zeros.AsSpan(0, (int)maxBytes), staging);

        for (int i = 0; i < _numLayers; i++)
        {
            _device.CopyBufferSynchronous(staging, _ssmStateBuffers[i], (ulong)ssmBytes);
            _device.CopyBufferSynchronous(staging, _cumAngleBuffers[i], (ulong)cumBytes);
        }
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
        }
    }
}
