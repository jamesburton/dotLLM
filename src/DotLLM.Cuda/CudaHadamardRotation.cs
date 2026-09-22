using System.Collections.Frozen;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;

namespace DotLLM.Cuda;

/// <summary>
/// Device-side resources and dispatch helper for the PrismML Hadamard activation transform
/// (<c>prism.hadamard.*</c>, see <see cref="HadamardFoldConfig"/>) on a CUDA model — issue #479.
/// The CUDA twin of <c>VulkanHadamardRotation</c>.
/// </summary>
/// <remarks>
/// <para>
/// Owns one resident F32 ±1 sign buffer per distinct activation width (Bonsai 2 has three: 5120,
/// 6144 and 17408), uploaded once at load, so a folded projection rotates without any host
/// round-trip. The forward transform runs on the device via
/// <see cref="CudaKernels.LaunchHadamardFwhtF32"/>.
/// </para>
/// <para>
/// The <b>inverse</b> transform (the <c>token_embd</c> row lookup) runs on the host through the CPU
/// <see cref="HadamardActivationRotator"/>, because every embedding gather in the CUDA hybrid-dense
/// model is already a host-side row dequant followed by one H2D copy. Rotating those few rows
/// before the copy costs no extra launch and is the CPU reference's exact code path.
/// </para>
/// </remarks>
public sealed class CudaHadamardRotation : IDisposable
{
    private readonly HadamardFoldConfig _fold;
    private readonly CudaKernels _kernels;
    private readonly HadamardActivationRotator _host;
    private readonly FrozenDictionary<int, nint> _signBuffers;
    private readonly float _scale;
    private readonly int _permDState;
    private readonly int _permNKHead;
    private readonly int _permRep;
    private bool _disposed;

    private CudaHadamardRotation(
        HadamardFoldConfig fold, CudaKernels kernels, HadamardActivationRotator host,
        FrozenDictionary<int, nint> signBuffers, int permDState, int permNKHead, int permRep)
    {
        _fold = fold;
        _kernels = kernels;
        _host = host;
        _signBuffers = signBuffers;
        _scale = fold.Scale;
        _permDState = permDState;
        _permNKHead = permNKHead;
        _permRep = permRep;
    }

    /// <summary>The fold declaration this helper applies.</summary>
    public HadamardFoldConfig Fold => _fold;

    /// <summary>
    /// Validates the fold declaration against the <c>qwen35</c> fixed rotation sites and uploads
    /// the sign vectors. The CUDA context must be current.
    /// </summary>
    /// <param name="kernels">Loaded kernels; must report <see cref="CudaKernels.HasHadamardFwht"/>.</param>
    /// <param name="fold">The parsed <c>prism.hadamard.*</c> declaration.</param>
    /// <param name="gdn">GDN geometry (fold-set validation and the <c>ssm_out</c> permute).</param>
    /// <param name="fullLayerCount">
    /// The FULL trunk's layer count. A partial-offload head instance still validates against the
    /// whole model, because the checkpoint's declaration covers every layer.
    /// </param>
    /// <exception cref="NotSupportedException">
    /// The kernel is unavailable (stale PTX), the block size exceeds the kernel's staging limit, or
    /// the declared fold set differs from what the forward pass rotates.
    /// </exception>
    public static CudaHadamardRotation Create(
        CudaKernels kernels, HadamardFoldConfig fold, GatedDeltaNetConfig gdn, int fullLayerCount)
    {
        ArgumentNullException.ThrowIfNull(kernels);
        ArgumentNullException.ThrowIfNull(fold);

        if (!kernels.HasHadamardFwht)
            throw new NotSupportedException(
                "This checkpoint declares a PrismML Hadamard weight fold (prism.hadamard.*), but " +
                "hadamard_fwht.ptx is missing or stale, so the CUDA backend cannot apply it. " +
                "Regenerate the PTX (native/build.ps1) or run on the CPU/Vulkan backend.");

        if (fold.BlockSize <= 0 || (fold.BlockSize & (fold.BlockSize - 1)) != 0
            || fold.BlockSize > CudaKernels.HadamardFwhtMaxBlockSize)
            throw new NotSupportedException(
                $"prism.hadamard.block_size {fold.BlockSize} is not a power of two within the CUDA " +
                $"kernel's shared-memory staging limit of {CudaKernels.HadamardFwhtMaxBlockSize}.");

        // Same load-time guarantee the CPU host gives: the fixed rotation sites must cover exactly
        // the declared fold set (this also refuses a checkpoint that folds the MTP block's own
        // blk.{NumLayers}.* projections, which no backend rotates), and token_embd must be the only
        // inverse table.
        var host = new HadamardActivationRotator(fold, gdn);
        host.ValidateQwen35FoldSet(fullLayerCount, gdn.FullAttnInterval);

        int permDState = 0, permNKHead = 0, permRep = 0;
        if (fold.GdnVGrouped)
        {
            // HadamardActivationRotator's constructor has already rejected bad head geometry.
            permDState = gdn.DState;
            permNKHead = gdn.NKHead;
            permRep = gdn.VHeadsPerKHead;
        }

        var buffers = new Dictionary<int, nint>();
        try
        {
            foreach (var (width, signs) in fold.SignsByWidth)
            {
                var data = new float[width];
                for (int i = 0; i < width; i++)
                    data[i] = signs[i];

                long bytes = (long)width * sizeof(float);
                CudaDriverApi.cuMemAlloc_v2(out nint dPtr, (nuint)bytes).ThrowOnError();
                buffers[width] = dPtr;
                unsafe
                {
                    fixed (float* hPtr = data)
                        CudaDriverApi.cuMemcpyHtoD_v2(dPtr, (nint)hPtr, (nuint)bytes).ThrowOnError();
                }
            }

            return new CudaHadamardRotation(fold, kernels, host, buffers.ToFrozenDictionary(),
                permDState, permNKHead, permRep);
        }
        catch
        {
            foreach (nint b in buffers.Values)
                CudaDriverApi.cuMemFree_v2(b);
            throw;
        }
    }

    /// <summary>
    /// Enqueues the forward transform (optional GDN permute → signs → rotation) of a folded
    /// weight's input activation.
    /// </summary>
    /// <param name="src">Unrotated activation, device F32 <c>[rows, width]</c>.</param>
    /// <param name="dst">Rotated destination; must not alias <paramref name="src"/>.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="width">Activation width (the folded weight's input dimension).</param>
    /// <param name="permuteGdnValueHeads">
    /// True only for <c>*.ssm_out.weight</c>. Ignored unless the checkpoint sets
    /// <c>prism.hadamard.gdn_v_grouped</c>.
    /// </param>
    /// <param name="stream">CUDA stream.</param>
    public void RotateForward(nint src, nint dst, int rows, int width, bool permuteGdnValueHeads, nint stream)
    {
        var (signs, apply) = SignsFor(width);
        bool permute = permuteGdnValueHeads && _fold.GdnVGrouped;
        _kernels.LaunchHadamardFwhtF32(src, dst, signs, rows, width, _fold.BlockSize,
            applySigns: apply, inverse: false, permute: permute,
            _permDState, _permNKHead, _permRep, _scale, stream);
    }

    /// <summary>
    /// Applies the inverse transform (rotation, then signs) in place to host rows fetched from the
    /// rotated <c>token_embd.weight</c> table, before they are copied to the device.
    /// </summary>
    /// <param name="rows">Host rows, <paramref name="seqLen"/> · <paramref name="width"/> floats.</param>
    /// <param name="seqLen">Number of rows.</param>
    /// <param name="width">Row width (hidden size).</param>
    public unsafe void RotateInverseInPlaceHost(float* rows, int seqLen, int width)
        => _host.RotateInverseInPlace(rows, seqLen, width);

    private (nint Signs, bool Apply) SignsFor(int width)
    {
        if (_signBuffers.Count == 0)
            return (0, false);

        if (!_signBuffers.TryGetValue(width, out nint buf))
            throw new InvalidOperationException(
                $"prism.hadamard declares explicit signs but has no vector for activation width {width}.");

        return (buf, true);
    }

    /// <summary>Frees the sign buffers. The owning CUDA context must be current.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (nint b in _signBuffers.Values)
            CudaDriverApi.cuMemFree_v2(b);
    }
}
