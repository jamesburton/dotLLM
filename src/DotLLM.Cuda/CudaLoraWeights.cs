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

        try
        {
            for (int layer = 0; layer < cfg.NumLayers; layer++)
            {
                foreach (string proj in ProjNames)
                {
                    LoraLayerWeights? maybeW = adapter.GetLayerWeights(layer, proj);
                    if (maybeW is not { } w)
                        continue;

                    long aF32Bytes = (long)w.OutputDim * rank * sizeof(float);
                    long bF32Bytes = (long)rank * w.InputDim * sizeof(float);
                    long aF16Bytes = (long)w.OutputDim * rank * sizeof(ushort);
                    long bF16Bytes = (long)rank * w.InputDim * sizeof(ushort);

                    // Alloc F32 scratch, upload host data
                    CudaDriverApi.cuMemAlloc_v2(out nint aF32Dev, (nuint)aF32Bytes).ThrowOnError();
                    f32Scratch.Add(aF32Dev);
                    CudaDriverApi.cuMemcpyHtoD_v2(aF32Dev, w.AHandle, (nuint)aF32Bytes).ThrowOnError();

                    CudaDriverApi.cuMemAlloc_v2(out nint bF32Dev, (nuint)bF32Bytes).ThrowOnError();
                    f32Scratch.Add(bF32Dev);
                    CudaDriverApi.cuMemcpyHtoD_v2(bF32Dev, w.BHandle, (nuint)bF32Bytes).ThrowOnError();

                    // Alloc persistent F16 buffers
                    CudaDriverApi.cuMemAlloc_v2(out nint aF16Dev, (nuint)aF16Bytes).ThrowOnError();
                    CudaDriverApi.cuMemAlloc_v2(out nint bF16Dev, (nuint)bF16Bytes).ThrowOnError();

                    // Queue F32→F16 conversions
                    kernels.LaunchConvertF32ToF16(aF32Dev, aF16Dev, w.OutputDim * rank, stream);
                    kernels.LaunchConvertF32ToF16(bF32Dev, bF16Dev, rank * w.InputDim, stream);

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

            // Free any F16 buffers already stored
            foreach (var (af16, bf16, _, _) in buffers.Values)
            {
                if (af16 != 0) CudaDriverApi.cuMemFree_v2(af16);
                if (bf16 != 0) CudaDriverApi.cuMemFree_v2(bf16);
            }

            throw;
        }

        return new CudaLoraWeights(adapter, rank, scale, buffers);
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
