using System.Collections.Frozen;
using DotLLM.Core.Models;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-side resources and dispatch helper for the PrismML Hadamard activation transform
/// (<c>prism.hadamard.*</c>) on a Vulkan model.
/// </summary>
/// <remarks>
/// <para>
/// Owns the FWHT kernel plus one resident F32 sign buffer per distinct activation width (Bonsai 2
/// has three: 5120, 6144 and 17408, uploaded once at load). Keeping the signs on the device is what
/// lets a folded projection rotate without a host round-trip — the alternative would be a
/// download/upload per folded matmul per token, hundreds of stalls per decode step on a 64-layer
/// model.
/// </para>
/// </remarks>
public sealed class VulkanHadamardRotation : IDisposable
{
    private readonly HadamardFoldConfig _fold;
    private readonly HadamardFwhtF32Kernel _kernel;
    private readonly FrozenDictionary<int, VulkanDevice.Buffer> _signBuffers;
    private readonly VulkanDevice.Buffer _emptySigns;
    private readonly HadamardFwhtF32Kernel.GdnPermute? _gdnPermute;
    private bool _disposed;

    private VulkanHadamardRotation(
        HadamardFoldConfig fold,
        HadamardFwhtF32Kernel kernel,
        FrozenDictionary<int, VulkanDevice.Buffer> signBuffers,
        VulkanDevice.Buffer emptySigns,
        HadamardFwhtF32Kernel.GdnPermute? gdnPermute)
    {
        _fold = fold;
        _kernel = kernel;
        _signBuffers = signBuffers;
        _emptySigns = emptySigns;
        _gdnPermute = gdnPermute;
    }

    /// <summary>
    /// Creates the device-side rotation resources, uploading one sign buffer per declared width.
    /// </summary>
    /// <param name="device">Target device.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <param name="fold">The parsed <c>prism.hadamard.*</c> declaration.</param>
    /// <param name="gdn">GDN geometry, for the <c>ssm_out</c> value-head permutation.</param>
    /// <returns>The created rotation helper.</returns>
    public static VulkanHadamardRotation Create(
        VulkanDevice device, string spvDir, HadamardFoldConfig fold, GatedDeltaNetConfig gdn)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(fold);

        if (fold.BlockSize > HadamardFwhtF32Kernel.MaxBlockSize)
            throw new NotSupportedException(
                $"prism.hadamard.block_size {fold.BlockSize} exceeds the Vulkan shader's " +
                $"shared-memory staging limit of {HadamardFwhtF32Kernel.MaxBlockSize}.");

        var kernel = HadamardFwhtF32Kernel.Create(device, spvDir);
        var buffers = new Dictionary<int, VulkanDevice.Buffer>();
        VulkanDevice.Buffer? empty = null;

        try
        {
            foreach (var (width, signs) in fold.SignsByWidth)
            {
                var data = new float[width];
                for (int i = 0; i < width; i++)
                    data[i] = signs[i];

                var buf = device.AllocateDeviceLocal((long)width * sizeof(float));
                device.Upload(data, buf);
                buffers[width] = buf;
            }

            // The descriptor layout always binds a signs buffer, even for the identity sign mode.
            empty = device.AllocateDeviceLocal(sizeof(float));

            HadamardFwhtF32Kernel.GdnPermute? permute = null;
            if (fold.GdnVGrouped)
            {
                if (gdn.NKHead <= 0 || gdn.NVHead <= 0 || gdn.NVHead % gdn.NKHead != 0)
                    throw new InvalidOperationException(
                        $"prism.hadamard.gdn_v_grouped: bad head geometry NVHead={gdn.NVHead}, NKHead={gdn.NKHead}.");
                permute = new HadamardFwhtF32Kernel.GdnPermute(gdn.DState, gdn.NKHead, gdn.VHeadsPerKHead);
            }

            return new VulkanHadamardRotation(fold, kernel, buffers.ToFrozenDictionary(), empty, permute);
        }
        catch
        {
            foreach (var b in buffers.Values) b.Dispose();
            empty?.Dispose();
            kernel.Dispose();
            throw;
        }
    }

    /// <summary>Drops cached descriptor sets; call when the scratch buffer has been reallocated.</summary>
    /// <remarks>
    /// The descriptor cache is keyed on buffer handles, and a freed handle can be recycled into a new
    /// allocation — which would silently bind the wrong buffer. Resetting on every scratch
    /// reallocation is what keeps that from happening.
    /// </remarks>
    public void InvalidateDescriptorCache() => _kernel.InvalidateDescriptorCache();

    /// <summary>
    /// Records the forward transform (signs then rotation) of a folded weight's input activation.
    /// </summary>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="src">Unrotated activation, <c>[rows, width]</c> F32.</param>
    /// <param name="dst">Rotated destination; must not be <paramref name="src"/>.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="width">Activation width.</param>
    /// <param name="permuteGdnValueHeads">True only for <c>*.ssm_out.weight</c>.</param>
    public void RecordForward(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        int rows, int width, bool permuteGdnValueHeads = false)
    {
        var (signs, apply) = SignsFor(width);
        _kernel.Record(cmdBuf, src, dst, signs, rows, width, _fold.BlockSize,
            applySigns: apply, inverse: false,
            permute: permuteGdnValueHeads ? _gdnPermute : null);
    }

    /// <summary>
    /// Records the inverse transform (rotation then signs) of rows fetched from a rotated lookup
    /// table, in place.
    /// </summary>
    /// <param name="cmdBuf">Command buffer to record into.</param>
    /// <param name="rows_">Buffer holding the fetched rows; transformed in place.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="width">Row width.</param>
    public void RecordInverseInPlace(nint cmdBuf, VulkanDevice.Buffer rows_, int rows, int width)
    {
        var (signs, apply) = SignsFor(width);
        // Safe in place: without the permute each workgroup reads only the block it writes.
        _kernel.Record(cmdBuf, rows_, rows_, signs, rows, width, _fold.BlockSize,
            applySigns: apply, inverse: true, permute: null);
    }

    private (VulkanDevice.Buffer Signs, bool Apply) SignsFor(int width)
    {
        if (_signBuffers.Count == 0)
            return (_emptySigns, false);

        if (!_signBuffers.TryGetValue(width, out var buf))
            throw new InvalidOperationException(
                $"prism.hadamard declares explicit signs but has no vector for activation width {width}.");

        return (buf, true);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in _signBuffers.Values) b.Dispose();
        _emptySigns.Dispose();
        _kernel.Dispose();
    }
}
