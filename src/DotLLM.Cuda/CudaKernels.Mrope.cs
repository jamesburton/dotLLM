using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

public sealed unsafe partial class CudaKernels
{
    private CudaModule? _mropeF32Module;
    private nint _mropeF32Func;

    /// <summary>
    /// True when the interleaved M-RoPE kernel (<c>rope_mrope_f32.ptx</c>, issue #2073) is loaded. Optional: a build without
    /// the PTX (it must be generated with nvcc on a CUDA box) still loads; image input then throws a descriptive error.
    /// </summary>
    public bool HasMropeF32 => _mropeF32Func != 0;

    /// <summary>
    /// Interleaved M-RoPE on FP32 Q/K with NeoX pairing (issue #2073). The rotary pair <c>j</c> of row <c>t</c> is rotated by
    /// <c>pos3[axis[j] * seqLen + t] * invFreq[j]</c> (axis 3 = position 0).
    /// </summary>
    /// <param name="q">Device pointer, <c>[seqLen, numHeads, headDim]</c> F32, rotated in place.</param>
    /// <param name="k">Device pointer, <c>[seqLen, numKvHeads, headDim]</c> F32, rotated in place.</param>
    /// <param name="pos3">Device pointer to <c>[3][seqLen]</c> int32 (t, y, x logical positions).</param>
    /// <param name="axisPerPair">Device pointer to <c>ropeDim / 2</c> bytes (<c>MropePositions.AxisPerPair</c>).</param>
    /// <param name="seqLen">Rows.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">Key heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="ropeDim">Rotated width (even).</param>
    /// <param name="theta">RoPE base.</param>
    /// <param name="stream">Stream.</param>
    /// <param name="ropeInvFreq">Device pointer to YaRN inverse frequencies, or 0.</param>
    /// <param name="ropeMscale">YaRN cos/sin multiplier (1 when off).</param>
    public void LaunchMropeF32(nint q, nint k, nint pos3, nint axisPerPair, int seqLen, int numHeads, int numKvHeads, int headDim,
                               int ropeDim, float theta, nint stream, nint ropeInvFreq = 0, float ropeMscale = 1.0f)
    {
        if (_mropeF32Func == 0)
            throw new NotSupportedException("rope_mrope_f32.ptx is not built: image input needs the M-RoPE kernel. Regenerate the PTX on a CUDA box " +
                "(native/build_ptx.bat, or build DotLLM.Cuda with nvcc available) and rebuild.");
        nint qArg = q, kArg = k, posArg = pos3, axArg = axisPerPair;
        int slArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim, rdArg = ropeDim;
        float thetaArg = theta, mscaleArg = ropeMscale;
        nint invFreqArg = ropeInvFreq;
        void** args = stackalloc void*[] { &qArg, &kArg, &posArg, &axArg, &slArg, &nhArg, &nkvArg, &hdArg, &rdArg, &thetaArg, &invFreqArg, &mscaleArg };
        int totalPairs = seqLen * Math.Max(numHeads, numKvHeads) * (ropeDim / 2);
        uint gridDim = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_mropeF32Func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
}
