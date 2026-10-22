using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Issue #1109 precision-study seams. Every one is off by default and costs a static read when off:
/// <list type="bullet">
/// <item>F16-accumulate prefill GEMM with F32 promotion every <see cref="CudaGemm.Promote16FChunk"/> elements of K
/// (only while <see cref="CudaGemm.Use16FCompute"/> is on);</item>
/// <item><see cref="GemmPrecisionProbe"/>: runs each prefill GEMM's real operands through every accumulation /
/// activation format and records the error per projection shape;</item>
/// <item>the F16-accumulating flash prefill kernels (<see cref="CudaFlashAccumulatorExperiment"/>);</item>
/// <item><see cref="ReconfigureKvStorageForExperiment"/>: switch the KV storage format between runs in one process.</item>
/// </list>
/// </summary>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel
{
    private nint _gemmPromotePartials;
    private long _gemmPromotePartialsBytes;
    private nint _gemmPromoteOnes;
    private int _gemmPromoteOnesCount;

    /// <summary>Test-only: flash prefill launches served by an F16-accumulate experiment variant.</summary>
    internal int DebugFlashXaccCallsForTest { get; private set; }

    /// <summary>
    /// The prefill GEMM over dequantized F16 weights: <see cref="CudaGemm.LinearF16"/>, or its promoted-chunk form
    /// when that experiment is on, followed by the precision probe when it is enabled.
    /// </summary>
    private void PrefillLinearF16(nint xF16, nint wF16, nint yF16, int seqLen, int k, int m, nint streamH)
    {
        int chunk = CudaGemm.Promote16FChunk;
        if (chunk > 0 && CudaGemm.Use16FCompute && k > chunk && k % chunk == 0)
            LinearF16Promoted(xF16, wF16, yF16, seqLen, k, m, chunk, streamH);
        else
            CudaGemm.LinearF16(_cublas.Handle, xF16, wF16, yF16, seqLen, k, m, streamH);

        if (GemmPrecisionProbe.Enabled)
            ProbeGemm(xF16, wF16, seqLen, k, m, streamH);
    }

    // The promoted-chunk partials are bounded by running the GEMM in row slabs (the LM head's 248,320 outputs would
    // otherwise need (k / chunk) x rows x 248,320 halves).
    private const long GemmPromotePartialsMaxBytes = 256L << 20;

    private void LinearF16Promoted(nint xF16, nint wF16, nint yF16, int seqLen, int k, int m, int chunk, nint streamH)
    {
        int batches = k / chunk;
        int slab = (int)Math.Clamp(GemmPromotePartialsMaxBytes / ((long)batches * m * sizeof(ushort)), 1, seqLen);
        long bytes = (long)batches * slab * m * sizeof(ushort);
        if (bytes > _gemmPromotePartialsBytes)
        {
            _stream.Synchronize();
            FreeIfNonZero(ref _gemmPromotePartials);
            _gemmPromotePartials = AllocDevice(bytes);
            _gemmPromotePartialsBytes = bytes;
        }
        if (batches > _gemmPromoteOnesCount)
        {
            _stream.Synchronize();
            FreeIfNonZero(ref _gemmPromoteOnes);
            var ones = new Half[batches];
            Array.Fill(ones, Half.One);
            _gemmPromoteOnes = AllocDevice((long)batches * sizeof(ushort));
            fixed (Half* p = ones)
                CudaDriverApi.cuMemcpyHtoD_v2(_gemmPromoteOnes, (nint)p, (nuint)(batches * sizeof(ushort))).ThrowOnError();
            _gemmPromoteOnesCount = batches;
        }
        for (int r0 = 0; r0 < seqLen; r0 += slab)
        {
            int rows = Math.Min(slab, seqLen - r0);
            CudaGemm.LinearF16PromotedChunks(_cublas.Handle,
                xF16 + (nint)((long)r0 * k * sizeof(ushort)), wF16, yF16 + (nint)((long)r0 * m * sizeof(ushort)),
                rows, k, m, chunk, _gemmPromotePartials, _gemmPromoteOnes, streamH);
        }
    }

    /// <summary>
    /// Runs one prefill GEMM's operands through the F32-accumulate reference and every probe arm and reports the
    /// host copies to <see cref="GemmPrecisionProbe.Current"/>. Diagnostic only: synchronous, allocates per call.
    /// </summary>
    private void ProbeGemm(nint xF16, nint wF16, int seqLen, int k, int m, nint streamH)
    {
        _stream.Synchronize();
        long xElems = (long)seqLen * k, yElems = (long)seqLen * m;
        var x = new Half[xElems];
        var xq = new Half[xElems];
        var yRef = new Half[yElems];
        var y = new Half[yElems];
        nint dXq = AllocDevice(xElems * sizeof(ushort));
        nint dY = AllocDevice(yElems * sizeof(ushort));
        nint dRef = AllocDevice(yElems * sizeof(ushort));
        try
        {
            CopyToHost(x, xF16);
            GemmPrecisionProbe.QuantizeDequantizeQ8_1(x, xq);
            fixed (Half* p = xq)
                CudaDriverApi.cuMemcpyHtoD_v2(dXq, (nint)p, (nuint)(xElems * sizeof(ushort))).ThrowOnError();
            var probe = GemmPrecisionProbe.Current;
            probe.RecordActivations(k, m, seqLen, x, xq);

            CudaGemm.LinearF16WithCompute(_cublas.Handle, xF16, wF16, dRef, seqLen, k, m, compute16F: false, streamH);
            _stream.Synchronize();
            CopyToHost(yRef, dRef);

            CudaGemm.LinearF16WithCompute(_cublas.Handle, xF16, wF16, dY, seqLen, k, m, compute16F: true, streamH);
            _stream.Synchronize();
            CopyToHost(y, dY);
            probe.RecordOutput(k, m, seqLen, GemmPrecisionProbe.Arm.F16Accumulate, yRef, y);

            int chunk = GemmPrecisionProbe.ProbeChunk;
            if (k > chunk && k % chunk == 0)
            {
                LinearF16Promoted(xF16, wF16, dY, seqLen, k, m, chunk, streamH);
                _stream.Synchronize();
                CopyToHost(y, dY);
                probe.RecordOutput(k, m, seqLen, GemmPrecisionProbe.Arm.F16Promoted, yRef, y);
            }

            CudaGemm.LinearF16WithCompute(_cublas.Handle, dXq, wF16, dY, seqLen, k, m, compute16F: false, streamH);
            _stream.Synchronize();
            CopyToHost(y, dY);
            probe.RecordOutput(k, m, seqLen, GemmPrecisionProbe.Arm.Int8Activations, yRef, y);
        }
        finally
        {
            FreeIfNonZero(ref dXq);
            FreeIfNonZero(ref dY);
            FreeIfNonZero(ref dRef);
        }

        static void CopyToHost(Half[] dst, nint src)
        {
            fixed (Half* p = dst)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)((long)dst.Length * sizeof(ushort))).ThrowOnError();
        }
    }

    /// <summary>
    /// Launches the F16-accumulate flash prefill variant selected by <see cref="CudaFlashAccumulatorExperiment.Mode"/>
    /// in place of the shipped kernel. Returns <see langword="false"/> (launching nothing) in the default F32 mode.
    /// </summary>
    /// <exception cref="NotSupportedException">A variant is selected for a quantized KV pair other than Q4_0/Q4_0.</exception>
    private bool TryLaunchFlashPrefillXacc(bool kvq, nint q, nint k, nint v, nint output,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int positionOffset, nint streamH)
    {
        FlashAccumulatorMode mode = CudaFlashAccumulatorExperiment.Mode;
        if (mode == FlashAccumulatorMode.F32) return false;
        if (kvq && !CudaKernels.FlashXaccSupportsKvq(_kvKeyDType, _kvValueDType))
            throw new NotSupportedException(
                $"{CudaFlashAccumulatorExperiment.EnvVar}: F16-accumulate flash variants cover F16 and Q4_0/Q4_0 KV only, "
                + $"not {_kvKeyDType}/{_kvValueDType}.");
        _kernels.LaunchAttentionFlashPrefillHd256Xacc(mode, kvq, q, k, v, output, seqQ, seqKv, numHeads, numKvHeads,
            positionOffset, streamH);
        DebugFlashXaccCallsForTest++;
        return true;
    }

    /// <summary>
    /// Experiment seam (issue #1109): frees the KV storage and the sequence state and selects a new KV format, so one
    /// loaded model can measure several formats. The next attention forward allocates the storage afresh.
    /// </summary>
    /// <param name="key">K storage format (<see cref="KvCacheDType.F32"/> = the F16 cache).</param>
    /// <param name="value">V storage format.</param>
    /// <exception cref="NotSupportedException">The pair is not supported for this model's KV row width.</exception>
    internal void ReconfigureKvStorageForExperiment(KvCacheDType key, KvCacheDType value)
    {
        string? error = HybridKvStorage.Validate(key, value, Config.NumKvHeads * Config.HeadDim);
        if (error is not null) throw new NotSupportedException(error);
        _context.MakeCurrent();
        _stream.Synchronize();
        ResetSequenceState();
        FreeKvStorage();
        _f16CacheMaxSeqLen = 0;
        _f16CacheCurrentLength = 0;
        _lastForwardKvCache = null;
        FreeIfNonZero(ref _kvF16ScratchK);
        FreeIfNonZero(ref _kvF16ScratchV);
        _kvF16ScratchElems = 0;
        FreeIfNonZero(ref _f32KvStageK);
        FreeIfNonZero(ref _f32KvStageV);
        _f32KvStageElems = 0;
        _kvKeyDType = key;
        _kvValueDType = value;
    }

    private void FreePrecisionExperimentScratch()
    {
        FreeIfNonZero(ref _gemmPromotePartials);
        _gemmPromotePartialsBytes = 0;
        FreeIfNonZero(ref _gemmPromoteOnes);
        _gemmPromoteOnesCount = 0;
    }
}
