using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

public static unsafe partial class MoeSwiGluMlp
{
    /// <summary>
    /// Decode (one token) routed-expert fast path. The grouped path runs the k selected experts as k independent single-threaded
    /// tasks (<c>Parallel.For</c>, inner GEMMs without a pool): only k cores work, the work per expert is serial gate -> up -> down,
    /// and each task pays ArrayPool rentals + .NET thread-pool wake-ups. Here the whole layer is two <see cref="ComputeThreadPool"/>
    /// dispatches that statically partition the flattened (expert x projection x row) space across every active worker:
    /// dispatch A = gate+up rows of all experts, dispatch B = down rows of all experts. Each row goes through the very same
    /// <c>ComputeRows</c> kernel the n=1 GEMM column uses, so a token's result is bit-identical to the grouped path.
    /// Disabled with <c>DOTLLM_CPU_MOE_DECODE_N1=0</c> (A/B baseline).
    /// </summary>
    public static bool DecodeN1Enabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_CPU_MOE_DECODE_N1") != "0";

    private static long _decodeN1Calls;

    /// <summary>Number of layers served by the n=1 decode fast path (diagnostic / test oracle that the path really ran).</summary>
    public static long DecodeN1Calls => Interlocked.Read(ref _decodeN1Calls);

    private struct DecodeN1Ctx
    {
        public int Segments;          // number of (expert, projection) segments
        public int RowsPerSegment;    // M of every segment in this dispatch
        public int RowAlignment;      // legacy four-row kernels must retain their grouped-path row boundaries
        public int BlockCount;
        public long WeightRowBytes0, WeightRowBytes1; // per projection kind (gate/up share a quant family but may differ in qt)
        public delegate*<byte*, byte*, float*, int, int, void> Rows0;
        public delegate*<byte*, byte*, float*, int, int, void> Rows1;
        public byte** SegWeights;     // [Segments]
        public float** SegResults;    // [Segments]
        public byte** SegInput;       // [Segments] pre-quantised input per segment
        public byte* SegKind;         // [Segments] 0 -> Rows0, 1 -> Rows1
    }

    private static void DecodeN1Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var c = ref Unsafe.AsRef<DecodeN1Ctx>((void*)ctxPtr);
        int groupsPerSegment = (c.RowsPerSegment + c.RowAlignment - 1) / c.RowAlignment;
        long total = (long)c.Segments * groupsPerSegment;
        long start = total * threadIdx / threadCount;
        long end = total * (threadIdx + 1) / threadCount;
        while (start < end)
        {
            int seg = (int)(start / groupsPerSegment);
            int group = (int)(start - (long)seg * groupsPerSegment);
            int groups = (int)Math.Min(groupsPerSegment - group, end - start);
            int row = group * c.RowAlignment;
            int n = Math.Min(c.RowsPerSegment - row, groups * c.RowAlignment);
            bool k1 = c.SegKind[seg] != 0;
            long rb = k1 ? c.WeightRowBytes1 : c.WeightRowBytes0;
            var fn = k1 ? c.Rows1 : c.Rows0;
            fn(c.SegWeights[seg] + row * rb, c.SegInput[seg], c.SegResults[seg] + row, n, c.BlockCount);
            start += groups;
        }
    }

    /// <summary>
    /// Tries the decode fast path for a single-token MoE layer. Returns false (nothing written) when the shape/quant combination is not
    /// covered, so the caller falls back to the grouped path.
    /// </summary>
    [SkipLocalsInit]
    private static bool TryExecuteRoutedDecodeN1(
        float* hiddenPtr, float* scatterDownPtr,
        nint gateBase, QuantizationType gateQt, long gateExpBytes,
        nint upBase, QuantizationType upQt, long upExpBytes,
        nint downBase, QuantizationType downQt, long downExpBytes,
        ReadOnlySpan<int> bucketCursors, ReadOnlySpan<int> bucketSlots,
        ReadOnlySpan<int> uniqueExperts, int uniqueCount,
        int numExpertsPerTok, int hiddenSize, int intermediateSize, bool useGeGLU, ComputeThreadPool pool, bool packedLegacy)
    {
        if (!DecodeN1Enabled || DiagF32Activations || uniqueCount < 1) return false;
        // Legacy rows quantize activations. Preserve the explicitly selected F32-activation fallback.
        if (!packedLegacy && (IsPackedLegacyQuant(gateQt) || IsPackedLegacyQuant(upQt) || IsPackedLegacyQuant(downQt))) return false;
        if (gateQt == QuantizationType.IQ1_M && !Iq1MPackedEnabled) return false;
        if (upQt == QuantizationType.IQ1_M && !Iq1MPackedEnabled) return false;
        if (!MatMul.TryGetDecodeRowKernel(gateQt, hiddenSize, out var gateRows, out long gateRowBytes, out int hBlocks, out int gateFam)) return false;
        if (!MatMul.TryGetDecodeRowKernel(upQt, hiddenSize, out var upRows, out long upRowBytes, out int hBlocks2, out int upFam)) return false;
        if (!MatMul.TryGetDecodeRowKernel(downQt, intermediateSize, out var downRows, out long downRowBytes, out int iBlocks, out _)) return false;
        if (gateFam != upFam || hBlocks != hBlocks2) return false;
        int gatePre = ComputePrequantBytes(gateQt, hiddenSize, 1);
        int downPre = ComputePrequantBytes(downQt, intermediateSize, 1);
        if (gatePre <= 0 || downPre <= 0) return false;
        if (upQt != gateQt && ComputePrequantBytes(upQt, hiddenSize, 1) != gatePre) return false;

        int u = uniqueCount;
        // Per-expert slot (token 0 -> top-k slot) and a batch==1 guard.
        Span<int> slots = stackalloc int[u];
        for (int i = 0; i < u; i++)
        {
            int e = uniqueExperts[i];
            if (bucketCursors[e + 1] - bucketCursors[e] != 1) return false;
            slots[i] = bucketSlots[bucketCursors[e]];
        }

        long t0 = CpuStageProf.T();
        int inter = intermediateSize;
        float[] actBuf = ArrayPool<float>.Shared.Rent(u * inter * 3);
        byte[] qBuf = ArrayPool<byte>.Shared.Rent(gatePre + u * downPre + 64);
        nint* ptrs = (nint*)NativeMemory.Alloc((nuint)(sizeof(nint) * (u * 6 + 8)));
        try
        {
            fixed (float* act = actBuf)
            fixed (byte* q = qBuf)
            {
                float* gateOut = act;               // [u][inter]
                float* upOut = act + u * inter;     // [u][inter]
                float* siluOut = act + 2 * u * inter;
                byte* xQ = q;
                byte* siluQ = q + gatePre;

                QuantizeBatchForWeightQt(gateQt, hiddenPtr, xQ, hiddenSize, 1);
                CpuStageProf.E(9, t0); t0 = CpuStageProf.T();

                // ── Dispatch A: gate + up rows of every expert ──
                int segs = 2 * u;
                byte** segW = (byte**)ptrs; float** segR = (float**)(ptrs + segs); byte** segI = (byte**)(ptrs + 2 * segs);
                byte* kind = (byte*)(ptrs + 3 * segs);
                for (int i = 0; i < u; i++)
                {
                    int e = uniqueExperts[i];
                    segW[2 * i] = (byte*)(gateBase + (nint)(e * gateExpBytes)); segR[2 * i] = gateOut + i * inter; segI[2 * i] = xQ; kind[2 * i] = 0;
                    segW[2 * i + 1] = (byte*)(upBase + (nint)(e * upExpBytes)); segR[2 * i + 1] = upOut + i * inter; segI[2 * i + 1] = xQ; kind[2 * i + 1] = 1;
                }
                var ctx = new DecodeN1Ctx
                {
                    Segments = segs, RowsPerSegment = inter, BlockCount = hBlocks,
                    RowAlignment = IsPackedLegacyQuant(gateQt) || IsPackedLegacyQuant(upQt) ? 4 : 1,
                    WeightRowBytes0 = gateRowBytes, WeightRowBytes1 = upRowBytes, Rows0 = gateRows, Rows1 = upRows,
                    SegWeights = segW, SegResults = segR, SegInput = segI, SegKind = kind,
                };
                pool.Dispatch((nint)(&ctx), &DecodeN1Worker);
                CpuStageProf.E(10, t0); t0 = CpuStageProf.T();

                // ── activation + down-input quantisation (tiny, serial) ──
                for (int i = 0; i < u; i++)
                {
                    var g = new ReadOnlySpan<float>(gateOut + i * inter, inter);
                    var up = new ReadOnlySpan<float>(upOut + i * inter, inter);
                    var o = new Span<float>(siluOut + i * inter, inter);
                    if (useGeGLU) FusedOps.GeGLUTanh(g, up, o); else FusedOps.SwiGLU(g, up, o);
                    QuantizeBatchForWeightQt(downQt, siluOut + i * inter, siluQ + (long)i * downPre, inter, 1);
                }
                CpuStageProf.E(11, t0); t0 = CpuStageProf.T();

                // ── Dispatch B: down rows of every expert, written straight into the scatter table ──
                for (int i = 0; i < u; i++)
                {
                    int e = uniqueExperts[i];
                    segW[i] = (byte*)(downBase + (nint)(e * downExpBytes)); segR[i] = scatterDownPtr + (long)slots[i] * hiddenSize;
                    segI[i] = siluQ + (long)i * downPre; kind[i] = 0;
                }
                var ctxB = new DecodeN1Ctx
                {
                    Segments = u, RowsPerSegment = hiddenSize, BlockCount = iBlocks,
                    RowAlignment = IsPackedLegacyQuant(downQt) ? 4 : 1,
                    WeightRowBytes0 = downRowBytes, WeightRowBytes1 = downRowBytes, Rows0 = downRows, Rows1 = downRows,
                    SegWeights = segW, SegResults = segR, SegInput = segI, SegKind = kind,
                };
                pool.Dispatch((nint)(&ctxB), &DecodeN1Worker);
                CpuStageProf.E(12, t0);
            }
            Interlocked.Increment(ref _decodeN1Calls);
            return true;
        }
        finally
        {
            NativeMemory.Free(ptrs);
            ArrayPool<float>.Shared.Return(actBuf);
            ArrayPool<byte>.Shared.Return(qBuf);
        }
    }
}
