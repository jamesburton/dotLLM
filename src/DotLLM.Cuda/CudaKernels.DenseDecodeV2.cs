using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Split-KV tensor-core decode attention for the dense <see cref="CudaTransformerModel"/> (issue #1922, E1 / E3):
/// <c>attention_mma_decode_v2_dense.ptx</c>, the #988 v2 body generalised to head dim 64 / 128 (and 256 for TurboQuant),
/// F16 output and normalised F16 split partials (F32 m / l). Grid <c>(numKvHeads, kvSplit)</c>, 256 threads, cooperative launch.
/// </summary>
public sealed unsafe partial class CudaKernels
{
    /// <summary>Largest kvSplit the dense v2 kernels accept (shared combine table); MUST match <c>DV2_MAX_SPLIT</c>.</summary>
    public const int AttentionMmaDecodeDenseMaxSplit = 64;

    private CudaModule? _denseDecodeV2Module;
    // [hdIdx (0 = hd64, 1 = hd128)][fmt (0 f16, 1 q8q8, 2 q4q4, 3 q8q4, 4 q4q8)]
    private readonly nint[] _denseDecodeV2Funcs = new nint[2 * 5];
    private readonly int[] _denseDecodeV2MaxGrid = Enumerable.Repeat(-1, 2 * 5).ToArray();
    // [hdIdx (0 = hd64, 1 = hd128, 2 = hd256)]
    private readonly nint[] _denseDecodeV2TqFuncs = new nint[3];
    private readonly int[] _denseDecodeV2TqMaxGrid = Enumerable.Repeat(-1, 3).ToArray();

    /// <summary>Launch counter for the dense v2 decode kernels (all formats), so tests can prove the path ran.</summary>
    internal static long AttentionMmaDecodeDenseLaunches;

    private void LoadDenseDecodeV2(string ptxDir)
    {
        string path = Path.Combine(ptxDir, "attention_mma_decode_v2_dense.ptx");
        if (!File.Exists(path)) return;
        try
        {
            var m = CudaModule.LoadFromFile(path);
            string[] fmts = { "f16", "q8q8", "q4q4", "q8q4", "q4q8" };
            int[] hds = { 64, 128 };
            for (int h = 0; h < 2; h++)
                for (int f = 0; f < 5; f++)
                    _denseDecodeV2Funcs[h * 5 + f] = m.TryGetFunction($"attention_mma_decode_v2_hd{hds[h]}_{fmts[f]}");
            int[] tqHds = { 64, 128, 256 };
            for (int h = 0; h < 3; h++)
                _denseDecodeV2TqFuncs[h] = m.TryGetFunction($"attention_mma_decode_v2_hd{tqHds[h]}_tq");
            _denseDecodeV2Module = m;
        }
        catch (CudaException)
        {
            _denseDecodeV2Module = null;
            Array.Clear(_denseDecodeV2Funcs);
            Array.Clear(_denseDecodeV2TqFuncs);
        }
    }

    /// <summary>Format index: K/V both F16 (<see cref="KvCacheDType.F32"/> is FP16 on the GPU) = 0, else 1 + KvqIndex; -1 unsupported.</summary>
    private static int DenseFmtIndex(KvCacheDType k, KvCacheDType v)
    {
        if (k == KvCacheDType.F32 && v == KvCacheDType.F32) return 0;
        int i = KvqIndex(k, v);
        return i < 0 ? -1 : 1 + i;
    }

    private static int DenseSlot(int headDim, KvCacheDType k, KvCacheDType v)
    {
        int h = headDim == 64 ? 0 : headDim == 128 ? 1 : -1;
        int f = DenseFmtIndex(k, v);
        return h < 0 || f < 0 ? -1 : h * 5 + f;
    }

    /// <summary>
    /// True when the dense v2 decode kernel for this head dim (64 / 128) and K/V format pair is loaded. F16 KV is
    /// <see cref="KvCacheDType.F32"/>.
    /// </summary>
    public bool HasAttentionMmaDecodeDense(int headDim, KvCacheDType kDtype, KvCacheDType vDtype)
    {
        int i = DenseSlot(headDim, kDtype, vDtype);
        return i >= 0 && _denseDecodeV2Funcs[i] != 0;
    }

    /// <summary>
    /// Largest safe kvSplit (cooperative co-residency ceiling / numKvHeads, capped at <see cref="AttentionMmaDecodeDenseMaxSplit"/>)
    /// for the dense v2 kernel; 0 when it is not loaded or cannot launch at all.
    /// </summary>
    public int MaxSafeAttentionMmaDecodeDense(int numKvHeads, int headDim, KvCacheDType kDtype, KvCacheDType vDtype)
    {
        int i = DenseSlot(headDim, kDtype, vDtype);
        if (i < 0 || _denseDecodeV2Funcs[i] == 0 || numKvHeads <= 0) return 0;
        if (_denseDecodeV2MaxGrid[i] < 0) _denseDecodeV2MaxGrid[i] = QueryCoResidentGrid(_denseDecodeV2Funcs[i]);
        int g = _denseDecodeV2MaxGrid[i];
        return g <= 0 ? 0 : Math.Min(g / numKvHeads, AttentionMmaDecodeDenseMaxSplit);
    }

    /// <summary>Bytes per element of the split partial O: F16, stored normalised (James 2026-11-21, bench 2026-11-22).</summary>
    public const int AttentionMmaDecodeDensePartialBytes = 2;

    /// <summary>
    /// Dense v2 split-KV decode (seqQ == 1). <paramref name="q"/> F16 <c>[numHeads, headDim]</c>; <paramref name="k"/> /
    /// <paramref name="v"/> raw cache storage over the logical <c>[seqKv, numKvHeads*headDim]</c> grid; <paramref name="output"/>
    /// F16 <c>[numHeads, headDim]</c>. Scratch: <paramref name="partialMax"/> / <paramref name="partialSum"/> F32
    /// <c>[numHeads, kvSplit]</c>, <paramref name="partialOut"/> F16 <c>[numHeads, kvSplit, headDim]</c>. <paramref name="kvSplit"/>
    /// must not exceed <see cref="MaxSafeAttentionMmaDecodeDense"/> (checked: exceeding the co-residency ceiling is a hard CUDA error).
    /// </summary>
    public void LaunchAttentionMmaDecodeDense(nint q, nint k, nint v, KvCacheDType kDtype, KvCacheDType vDtype, nint output,
        int seqKv, int numHeads, int numKvHeads, int headDim, int kvSplit,
        nint partialMax, nint partialSum, nint partialOut, nint stream)
    {
        int i = DenseSlot(headDim, kDtype, vDtype);
        if (i < 0 || _denseDecodeV2Funcs[i] == 0)
            throw new InvalidOperationException($"Dense v2 decode kernel for hd{headDim} {kDtype}/{vDtype} is not loaded.");
        int maxSafe = MaxSafeAttentionMmaDecodeDense(numKvHeads, headDim, kDtype, vDtype);
        if (kvSplit < 1 || kvSplit > maxSafe)
            throw new ArgumentOutOfRangeException(nameof(kvSplit), $"kvSplit {kvSplit} outside [1, {maxSafe}] for the dense v2 decode kernel.");
        if (!IsGqaGroupShapeSupported(numHeads, numKvHeads))
            throw new ArgumentException($"Unsupported GQA shape {numHeads}/{numKvHeads} for the dense v2 decode kernel.");

        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, ksArg = kvSplit;
        float scArg = 1.0f / MathF.Sqrt(headDim);
        nint pmArg = partialMax, psArg = partialSum, poArg = partialOut;
        void** args = stackalloc void*[] { &qArg, &kArg, &vArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &scArg, &ksArg, &pmArg, &psArg, &poArg };
        Interlocked.Increment(ref AttentionMmaDecodeDenseLaunches);
        CudaDriverApi.cuLaunchCooperativeKernel(_denseDecodeV2Funcs[i],
                (uint)numKvHeads, (uint)kvSplit, 1, AttentionMmaDecodeGqaSplitBlockSize, 1, 1,
                0, stream, (nint)args).ThrowOnError();
    }

    private static int DenseTqSlot(int headDim) => headDim == 64 ? 0 : headDim == 128 ? 1 : headDim == 256 ? 2 : -1;

    /// <summary>True when the dense v2 TurboQuant decode kernel (#1922 E3) for this head dim is loaded.</summary>
    public bool HasAttentionMmaDecodeDenseTq(int headDim)
    {
        int i = DenseTqSlot(headDim);
        return i >= 0 && _denseDecodeV2TqFuncs[i] != 0;
    }

    /// <summary>Co-residency ceiling (max safe kvSplit) for the dense v2 TurboQuant decode kernel; 0 = unavailable.</summary>
    public int MaxSafeAttentionMmaDecodeDenseTq(int numKvHeads, int headDim)
    {
        int i = DenseTqSlot(headDim);
        if (i < 0 || _denseDecodeV2TqFuncs[i] == 0 || numKvHeads <= 0) return 0;
        if (_denseDecodeV2TqMaxGrid[i] < 0) _denseDecodeV2TqMaxGrid[i] = QueryCoResidentGrid(_denseDecodeV2TqFuncs[i]);
        int g = _denseDecodeV2TqMaxGrid[i];
        return g <= 0 ? 0 : Math.Min(g / numKvHeads, AttentionMmaDecodeDenseMaxSplit);
    }

    /// <summary>
    /// Dense v2 split-KV decode over the in-place TurboQuant cache (#1922 E3), in code space: <paramref name="q"/> must already be
    /// rotated into K code space and the F16 <paramref name="output"/> is in V code space (the caller rotates it back), exactly as
    /// for <see cref="LaunchAttentionKvTq"/>. Same scratch contract as <see cref="LaunchAttentionMmaDecodeDense"/>.
    /// </summary>
    public void LaunchAttentionMmaDecodeDenseTq(nint q, nint kCodes, nint vCodes, nint kNorms, nint vNorms, nint centroids,
        nint output, int seqKv, int numHeads, int numKvHeads, int headDim, int kvSplit,
        nint partialMax, nint partialSum, nint partialOut,
        int mseBits, int codeUintsPerVec, TqNormFormat normFormat, nint stream)
    {
        int i = DenseTqSlot(headDim);
        if (i < 0 || _denseDecodeV2TqFuncs[i] == 0)
            throw new InvalidOperationException($"Dense v2 TurboQuant decode kernel for hd{headDim} is not loaded.");
        CheckKvTq(headDim, mseBits, codeUintsPerVec, normFormat);
        int maxSafe = MaxSafeAttentionMmaDecodeDenseTq(numKvHeads, headDim);
        if (kvSplit < 1 || kvSplit > maxSafe)
            throw new ArgumentOutOfRangeException(nameof(kvSplit), $"kvSplit {kvSplit} outside [1, {maxSafe}] for the dense v2 TQ decode kernel.");
        if (!IsGqaGroupShapeSupported(numHeads, numKvHeads))
            throw new ArgumentException($"Unsupported GQA shape {numHeads}/{numKvHeads} for the dense v2 TQ decode kernel.");

        nint qArg = q, kcArg = kCodes, vcArg = vCodes, knArg = kNorms, vnArg = vNorms, ceArg = centroids, outArg = output;
        int skvArg = seqKv, nhArg = numHeads, nkvArg = numKvHeads, ksArg = kvSplit;
        float scArg = 1.0f / MathF.Sqrt(headDim);
        nint pmArg = partialMax, psArg = partialSum, poArg = partialOut;
        int mbArg = mseBits, cuArg = codeUintsPerVec, nfArg = (int)normFormat;
        void** args = stackalloc void*[] { &qArg, &kcArg, &vcArg, &knArg, &vnArg, &ceArg, &outArg,
                        &skvArg, &nhArg, &nkvArg, &scArg, &ksArg, &pmArg, &psArg, &poArg, &mbArg, &cuArg, &nfArg };
        Interlocked.Increment(ref AttentionMmaDecodeDenseLaunches);
        CudaDriverApi.cuLaunchCooperativeKernel(_denseDecodeV2TqFuncs[i],
                (uint)numKvHeads, (uint)kvSplit, 1, AttentionMmaDecodeGqaSplitBlockSize, 1, 1,
                0, stream, (nint)args).ThrowOnError();
    }
}
