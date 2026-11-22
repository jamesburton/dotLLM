using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Gate, dispatch and scratch for the dense split-KV tensor-core decode kernel (issue #1922, E1 / E3) in
/// <see cref="CudaTransformerModel"/>: <c>attention_mma_decode_v2_dense</c> replaces the scalar one-block-per-query-head
/// <c>attention_f16</c> / <c>attention_kvq</c> / <c>attention_kvtq</c> for an eager decode step at depth.
/// </summary>
/// <remarks>
/// <para>
/// <b>Scope.</b> Decode only (one query row that is the newest cache row), head dim 64 / 128 (TurboQuant also 256), GQA group
/// &lt;= 8 dividing evenly, no sliding window, no attention sinks, no FP16 window ring, and <c>seqKv &gt;= MinSeqKv</c>.
/// Everything else falls through to the existing kernel. The kernel takes a scalar seqKv / kvSplit, so it may be captured only by a
/// graph that is re-captured every step (the step graph, `CudaTransformerModel.StepGraph.cs`), never by a persistent, replayed one (the
/// shallow f16 / windowed-q8 device-scalar graphs and the MoE graph take their `_dyn` kernels instead).
/// </para>
/// <para>
/// <b>Scratch.</b> One partial buffer shared by every layer, allocated once here at model construction (never inside a decode
/// step), sized for the largest kvSplit any dense entry can launch with on this device: the normalised partial O in F16,
/// <c>numHeads x maxSplit x headDim x 2</c>, plus F32 m and l, <c>2 x numHeads x maxSplit x 4</c> bytes (Qwen3-4B on an
/// RTX 3060: maxSplit 3, 24 KiB + 768 B), admitted through `CudaAdmission` against free VRAM. If admission refuses or the
/// allocation fails, the path is simply off.
/// </para>
/// <para>
/// <b>Switches.</b> <c>DOTLLM_CUDA_DENSE_DECODE_V2=0</c> turns it off; <c>DOTLLM_CUDA_DENSE_DECODE_V2_MIN_KV</c> (default 512)
/// sets the engagement depth. Default on: James 2026-11-21, bench 2026-11-22 (#1922).
/// </para>
/// </remarks>
internal sealed class CudaDenseDecodeV2 : IDisposable
{
    /// <summary>Kill switch (default on). Mutable so tests can A/B in one process.</summary>
    internal static bool Enabled = Environment.GetEnvironmentVariable("DOTLLM_CUDA_DENSE_DECODE_V2") != "0";

    /// <summary>Smallest cache length at which a decode step takes the dense v2 kernel. Mutable for tests.</summary>
    internal static int MinSeqKv =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_DENSE_DECODE_V2_MIN_KV"), out int v) && v >= 1 ? v : 512;

    /// <summary>Count of dense v2 dispatches since process start (tests prove the path ran).</summary>
    internal static long DispatchCount;

    private readonly CudaKernels _kernels;
    private readonly int _numHeads;
    private readonly int _headDim;
    private nint _partialMax, _partialSum, _partialOut;

    /// <summary>Largest kvSplit the scratch holds; 0 = the path is unavailable on this model / device.</summary>
    internal int MaxSplit { get; }

    /// <summary>Scratch bytes held (reported for the precision / memory record).</summary>
    internal long ScratchBytes { get; }

    internal CudaDenseDecodeV2(CudaKernels kernels, int numHeads, int numKvHeads, int headDim)
    {
        _kernels = kernels;
        _numHeads = numHeads;
        _headDim = headDim;
        if (numKvHeads <= 0 || !CudaKernels.IsGqaGroupShapeSupported(numHeads, numKvHeads)) return;

        int maxSplit = 0;
        if (headDim is 64 or 128)
            foreach (var (k, vv) in new[] { (KvCacheDType.F32, KvCacheDType.F32), (KvCacheDType.Q8_0, KvCacheDType.Q8_0),
                         (KvCacheDType.Q4_0, KvCacheDType.Q4_0), (KvCacheDType.Q8_0, KvCacheDType.Q4_0), (KvCacheDType.Q4_0, KvCacheDType.Q8_0) })
                maxSplit = Math.Max(maxSplit, kernels.MaxSafeAttentionMmaDecodeDense(numKvHeads, headDim, k, vv));
        maxSplit = Math.Max(maxSplit, kernels.MaxSafeAttentionMmaDecodeDenseTq(numKvHeads, headDim));
        if (maxSplit <= 0) return;

        long statBytes = (long)numHeads * maxSplit * sizeof(float);
        long outBytes = (long)numHeads * maxSplit * headDim * CudaKernels.AttentionMmaDecodeDensePartialBytes;
        // The one CUDA admission path (#1848, no spill): this is an optional accelerator, so a refusal turns the path off (the
        // scalar kernels need no scratch) instead of failing the model. The allocations are itemised by CudaMemTrace through the
        // cuMemAlloc_v2 wrapper (site CudaDenseDecodeV2..ctor). Device memory ledger: #1933.
        if (!CudaAdmission.TryFit(2 * statBytes + outBytes, out _))
        {
            Console.Error.WriteLine($"[dotLLM.Cuda] Dense v2 decode off: its {2 * statBytes + outBytes} B split scratch was not admitted.");
            return;
        }
        if (CudaDriverApi.cuMemAlloc_v2(out _partialMax, (nuint)statBytes) != 0
            || CudaDriverApi.cuMemAlloc_v2(out _partialSum, (nuint)statBytes) != 0
            || CudaDriverApi.cuMemAlloc_v2(out _partialOut, (nuint)outBytes) != 0)
        {
            Dispose();
            return;
        }
        MaxSplit = maxSplit;
        ScratchBytes = 2 * statBytes + outBytes;
    }

    /// <summary>The shape / state gate shared by every format (pure, so it is testable without a GPU).</summary>
    internal static bool Eligible(bool enabled, int seqLen, int seqKv, int positionOffset, int slidingWindow, bool hasSinks,
        int window, int minSeqKv)
        => enabled && seqLen == 1 && seqKv >= minSeqKv && positionOffset == seqKv - 1
           && slidingWindow <= 0 && !hasSinks && window <= 0;

    private int Split(int seqKv, int maxSafe)
        => Math.Min(CudaKernels.ComputeAttentionMmaV2KvSplit(seqKv, maxSafe), MaxSplit);

    /// <summary>
    /// Runs the dense v2 decode kernel over F16 (<see cref="KvCacheDType.F32"/>) or Q8_0 / Q4_0 KV when the gate allows and returns
    /// true; returns false (launching nothing) otherwise, and the caller runs its existing kernel. <paramref name="output"/> is F16.
    /// </summary>
    internal bool TryRun(nint q, nint k, nint v, KvCacheDType kDtype, KvCacheDType vDtype, nint output,
        int seqLen, int seqKv, int positionOffset, int slidingWindow, nint sinks, int window,
        int numHeads, int numKvHeads, int headDim, nint stream)
    {
        if (MaxSplit <= 0 || headDim != _headDim || numHeads != _numHeads
            || !Eligible(Enabled, seqLen, seqKv, positionOffset, slidingWindow, sinks != 0, window, MinSeqKv)
            || !_kernels.HasAttentionMmaDecodeDense(headDim, kDtype, vDtype))
            return false;
        int maxSafe = _kernels.MaxSafeAttentionMmaDecodeDense(numKvHeads, headDim, kDtype, vDtype);
        if (maxSafe <= 0) return false;
        DispatchCount++;
        _kernels.LaunchAttentionMmaDecodeDense(q, k, v, kDtype, vDtype, output, seqKv, numHeads, numKvHeads, headDim,
            Split(seqKv, maxSafe), _partialMax, _partialSum, _partialOut, stream);
        return true;
    }

    /// <summary>TurboQuant in-place twin of <see cref="TryRun"/> (#1922 E3): code-space Q in, code-space F16 output.</summary>
    internal bool TryRunTq(nint q, nint kCodes, nint vCodes, nint kNorms, nint vNorms, nint centroids, nint output,
        int seqLen, int seqKv, int positionOffset, int slidingWindow, nint sinks,
        int numHeads, int numKvHeads, int headDim, int mseBits, int codeUintsPerVec, TqNormFormat normFormat, nint stream)
    {
        if (MaxSplit <= 0 || headDim != _headDim || numHeads != _numHeads
            || !Eligible(Enabled, seqLen, seqKv, positionOffset, slidingWindow, sinks != 0, 0, MinSeqKv)
            || !_kernels.HasAttentionMmaDecodeDenseTq(headDim))
            return false;
        int maxSafe = _kernels.MaxSafeAttentionMmaDecodeDenseTq(numKvHeads, headDim);
        if (maxSafe <= 0) return false;
        DispatchCount++;
        _kernels.LaunchAttentionMmaDecodeDenseTq(q, kCodes, vCodes, kNorms, vNorms, centroids, output, seqKv, numHeads, numKvHeads,
            headDim, Split(seqKv, maxSafe), _partialMax, _partialSum, _partialOut, mseBits, codeUintsPerVec, normFormat, stream);
        return true;
    }

    public void Dispose()
    {
        if (_partialMax != 0) { CudaDriverApi.cuMemFree_v2(_partialMax); _partialMax = 0; }
        if (_partialSum != 0) { CudaDriverApi.cuMemFree_v2(_partialSum); _partialSum = 0; }
        if (_partialOut != 0) { CudaDriverApi.cuMemFree_v2(_partialOut); _partialOut = 0; }
    }
}
