using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Stages a CPU-side <see cref="ILoraAdapter"/>'s A/B factor pairs onto the
/// CUDA device as FP16 buffers, ready for inference-time delta application.
/// </summary>
/// <remarks>
/// <para>
/// Call <see cref="Stage"/> once per adapter; it iterates all
/// <c>(layer, proj)</c> sites, uploads the host F32 buffers to temporary
/// device scratch, converts them to FP16 via the dedicated PTX kernel, then
/// releases the F32 scratch.  The resulting FP16 device pointers are keyed in
/// a dictionary and retrieved via <see cref="TryGet"/>.
/// </para>
/// <para>
/// Only one synchronization point occurs — after all F32→F16 conversions are
/// queued — so staging is a single stream-ordered GPU burst followed by one
/// host wait.
/// </para>
/// </remarks>
public sealed unsafe class CudaLoraWeights : IDisposable
{
    private static readonly string[] ProjNames =
    [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ];

    private readonly Dictionary<(int Layer, string Proj), (nint AF16, nint BF16, int InputDim, int OutputDim)> _buffers;
    private bool _disposed;

    /// <summary>The adapter whose weights were staged onto the device.</summary>
    public ILoraAdapter Source { get; }

    /// <summary>LoRA rank — inner dimension of the A/B factorisation.</summary>
    public int Rank { get; }

    /// <summary>
    /// Pre-computed LoRA scaling factor: <c>Alpha / Rank</c>.
    /// Applied when accumulating the delta during the forward pass.
    /// </summary>
    public float Scale { get; }

    private CudaLoraWeights(
        ILoraAdapter source,
        int rank,
        float scale,
        Dictionary<(int, string), (nint, nint, int, int)> buffers)
    {
        Source = source;
        Rank = rank;
        Scale = scale;
        _buffers = buffers;
    }

    /// <summary>
    /// Retrieves the device-side FP16 A/B buffers for the given layer and projection.
    /// </summary>
    /// <param name="layer">Zero-based transformer layer index.</param>
    /// <param name="proj">
    /// Canonical projection name (e.g. <c>q_proj</c>, <c>gate_proj</c>).
    /// </param>
    /// <param name="aF16">
    /// Device pointer to the A matrix (FP16, shape <c>[OutputDim, Rank]</c>).
    /// </param>
    /// <param name="bF16">
    /// Device pointer to the B matrix (FP16, shape <c>[Rank, InputDim]</c>).
    /// </param>
    /// <param name="inputDim">Input dimension of the adapted projection.</param>
    /// <param name="outputDim">Output dimension of the adapted projection.</param>
    /// <returns>
    /// <c>true</c> when this adapter targets the specified site; <c>false</c> otherwise.
    /// </returns>
    public bool TryGet(int layer, string proj,
                       out nint aF16, out nint bF16,
                       out int inputDim, out int outputDim)
    {
        if (_buffers.TryGetValue((layer, proj), out var entry))
        {
            (aF16, bF16, inputDim, outputDim) = entry;
            return true;
        }

        aF16 = bF16 = 0;
        inputDim = outputDim = 0;
        return false;
    }

    /// <summary>
    /// Stages all adapter weights from <paramref name="adapter"/> onto the CUDA
    /// device as FP16 buffers. Safe to call from a host thread that owns
    /// <paramref name="stream"/>.
    /// </summary>
    /// <param name="adapter">Source adapter whose host F32 buffers to upload.</param>
    /// <param name="cfg">Model configuration; used to iterate layer indices.</param>
    /// <param name="kernels">CUDA kernel launcher (provides F32→F16 conversion).</param>
    /// <param name="stream">CUDA stream on which all uploads and conversions are queued.</param>
    /// <returns>A new <see cref="CudaLoraWeights"/> owning all device allocations.</returns>
    public static CudaLoraWeights Stage(
        ILoraAdapter adapter,
        ModelConfig cfg,
        CudaKernels kernels,
        nint stream)
    {
        int rank = adapter.Rank;
        float scale = adapter.Alpha / rank;

        var buffers = new Dictionary<(int, string), (nint, nint, int, int)>();
        // Temporary F32 device scratch pointers; freed after synchronization.
        var f32Scratch = new List<nint>();
        // Tracks every successfully-allocated persistent F16 pointer for error-path cleanup.
        var f16Allocs = new List<nint>();

        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                foreach (string proj in ProjNames)
                {
                    LoraLayerWeights? maybeW = adapter.GetLayerWeights(layer, proj);
                    if (maybeW is not { } w)
                        continue;

                    int aElems = w.OutputDim * rank;
                    int bElems = rank * w.InputDim;
                    long aF16Bytes = (long)aElems * sizeof(ushort);
                    long bF16Bytes = (long)bElems * sizeof(ushort);

                    // Alloc persistent F16 device buffers (the inference target dtype).
                    // Track in f16Allocs BEFORE any subsequent throw so error cleanup frees them.
                    CudaDriverApi.cuMemAlloc_v2(out nint aF16Dev, (nuint)aF16Bytes).ThrowOnError();
                    f16Allocs.Add(aF16Dev);
                    CudaDriverApi.cuMemAlloc_v2(out nint bF16Dev, (nuint)bF16Bytes).ThrowOnError();
                    f16Allocs.Add(bF16Dev);

                    // Stage A (up-projection) per its resolved dtype.
                    StageFactor(w.ResolvedAWeightDType, w.AHandle, aF16Dev, aElems,
                                kernels, stream, f32Scratch, layer, proj);
                    // Stage B (down-projection) per its dtype.
                    StageFactor(w.WeightDType, w.BHandle, bF16Dev, bElems,
                                kernels, stream, f32Scratch, layer, proj);

                    buffers[(layer, proj)] = (aF16Dev, bF16Dev, w.InputDim, w.OutputDim);
                }
            }

            // Single synchronization point: wait for all conversions before freeing F32 scratch.
            CudaDriverApi.cuStreamSynchronize(stream);

            // Free F32 scratch
            foreach (nint ptr in f32Scratch)
                CudaDriverApi.cuMemFree_v2(ptr);
            f32Scratch.Clear();
        }
        catch
        {
            // Free F32 scratch on error
            foreach (nint ptr in f32Scratch)
                CudaDriverApi.cuMemFree_v2(ptr);

            // Free any F16 buffers successfully allocated (via f16Allocs, not buffers.Values,
            // to avoid a double-free when the second cuMemAlloc throws before buffers[...] is set).
            foreach (nint ptr in f16Allocs)
                if (ptr != 0) CudaDriverApi.cuMemFree_v2(ptr);

            throw;
        }

        return new CudaLoraWeights(adapter, rank, scale, buffers);
    }

    /// <summary>
    /// Stages one A/B factor buffer from host into the pre-allocated F16 device
    /// target, branching on the factor's source dtype. Host reads are sized by
    /// the SOURCE dtype to avoid the F32-only over-read that triggered #89.
    /// </summary>
    private static void StageFactor(
        LoraWeightDType dtype,
        nint hostHandle,
        nint f16Dev,
        int elements,
        CudaKernels kernels,
        nint stream,
        List<nint> f32Scratch,
        int layer,
        string proj)
    {
        switch (dtype)
        {
            case LoraWeightDType.F32:
            {
                // Legacy path: upload F32 host bytes to scratch, convert F32→F16.
                // Byte-equivalent to the pre-#89 behaviour for F32 adapters.
                long f32Bytes = (long)elements * sizeof(float);
                CudaDriverApi.cuMemAlloc_v2(out nint f32Dev, (nuint)f32Bytes).ThrowOnError();
                f32Scratch.Add(f32Dev);
                CudaDriverApi.cuMemcpyHtoD_v2(f32Dev, hostHandle, (nuint)f32Bytes).ThrowOnError();
                kernels.LaunchConvertF32ToF16(f32Dev, f16Dev, elements, stream);
                break;
            }

            case LoraWeightDType.F16:
            {
                // Device target is already F16: upload host F16 bytes directly,
                // no F32 scratch and no conversion. This is the #89 fix.
                long f16Bytes = (long)elements * sizeof(ushort);
                CudaDriverApi.cuMemcpyHtoD_v2(f16Dev, hostHandle, (nuint)f16Bytes).ThrowOnError();
                break;
            }

            default:
                // BF16 (no BF16→F16 kernel exists) and Q8_0 (GPU dequant out of
                // scope) fail safely instead of silently over-reading the host
                // buffer. The F16 device buffers are already tracked in f16Allocs,
                // so the caller's catch frees them.
                throw new NotSupportedException(
                    $"GPU LoRA staging does not yet support {dtype} adapter weights " +
                    $"(proj '{proj}', layer {layer}); use --device cpu, or an F32/F16 adapter. " +
                    "Tracked in #89.");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        foreach (var (aF16, bF16, _, _) in _buffers.Values)
        {
            if (aF16 != 0) CudaDriverApi.cuMemFree_v2(aF16);
            if (bF16 != 0) CudaDriverApi.cuMemFree_v2(bF16);
        }

        _buffers.Clear();
    }
}
