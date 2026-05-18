using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using DotLLM.Cpu.Kernels;

namespace LoraQ8Stage1Probe;

/// <summary>
/// Microbench for LoRA stage-1 Q8_0 candidate kernels.
/// Geometry: M=rank=16, K=2048, N=seqLen=512 — the canonical Llama-3.2-1B
/// hidden-size, one prefill chunk. All kernels read pre-quantised activation
/// (xQ8 already produced by the base projection's QuantizeF32ToQ8_0) so
/// activation-quant cost is excluded from measurement (production has it
/// amortised across base GEMM + LoRA stage 1).
/// </summary>
[MemoryDiagnoser]
public unsafe class LoraStage1Bench
{
    [Params(2048)] public int K { get; set; }
    [Params(512)] public int N { get; set; }
    /// <summary>
    /// Stage-2 output dimension. Sweep covers k/v_proj (≈512), q/o_proj (≈2048),
    /// and FFN gate/up_proj (≈5632) on Llama-3.2-1B.
    /// </summary>
    [Params(512, 2048, 5632)] public int OutputDim { get; set; } = 2048;
    public const int Rank = 16;

    private byte* _xQ8;          // [N, K_blocks * 34] Q8_0 (already quantised)
    private byte* _bRowMajor;    // [Rank, K_blocks * 34] Q8_0
    private byte* _bR16;         // [K_blocks, Rank * 34] R16 interleaved Q8_0
    private byte* _bR4;          // [fullGroups, blockCount, 4 * 34] R4 layout matching the base-model GEMM
    private float* _bF32;        // [Rank, K] F32 dequant scratch (legacy path)
    private float* _xF32;        // [N, K] F32 (only for GemmF32 baseline)
    private float* _tmp;         // [N, Rank] output (stage-1 result)
    private float* _aF32;        // [outputDim, Rank] F32 for stage-2 (row-major, production layout)
    private float* _aT;          // [Rank, outputDim] F32 — A transposed (Path E1 layout)
    private float* _yOut;        // [N, outputDim] F32 stage-2 output

    private int _blockCount;
    private int _rowBytes;

    [GlobalSetup]
    public void Setup()
    {
        _blockCount = K / Kernels.Q8_0Group;
        _rowBytes = _blockCount * Kernels.Q8_0Block;

        // Allocate native-aligned buffers (64-byte alignment for AVX-512 friendliness).
        _xF32 = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * K * sizeof(float)), 64);
        _bF32 = (float*)NativeMemory.AlignedAlloc((nuint)((long)Rank * K * sizeof(float)), 64);
        _xQ8 = (byte*)NativeMemory.AlignedAlloc((nuint)((long)N * _rowBytes), 64);
        _bRowMajor = (byte*)NativeMemory.AlignedAlloc((nuint)((long)Rank * _rowBytes), 64);
        _tmp = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * Rank * sizeof(float)), 64);

        // Fill x and B with reproducible random data.
        var rng = new Random(7);
        for (long i = 0; i < (long)N * K; i++) _xF32[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.4f;
        for (long i = 0; i < (long)Rank * K; i++) _bF32[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.4f;

        // Pre-quantise x (matches production: base proj already did this).
        for (int t = 0; t < N; t++)
            MatMul.QuantizeF32ToQ8_0(_xF32 + (long)t * K, _xQ8 + (long)t * _rowBytes, K);

        // Quantise B rows.
        for (int r = 0; r < Rank; r++)
            MatMul.QuantizeF32ToQ8_0(_bF32 + (long)r * K, _bRowMajor + (long)r * _rowBytes, K);

        // Build R16-interleaved B layout once (amortised over many LoRA forwards).
        _bR16 = Kernels.RepackR16(_bRowMajor, Rank, _blockCount, _rowBytes);

        // Build R4-grouped B layout once. Lets us route LoRA stage-1 through the
        // existing MatMul.OuterProductGemmQ8_0 kernel (4 rows × 6 tokens kept
        // live in 24 ZMM accumulators) with zero new SIMD. Investigation report
        // .planning/notes/lora-q8-stage1-investigation.md §3 + §7 for rationale.
        _bR4 = Kernels.RepackR4(_bRowMajor, Rank, _blockCount, _rowBytes);

        // Stage-2 fixtures: A [outputDim, rank] F32 and y [N, outputDim] output buffer.
        _aF32 = (float*)NativeMemory.AlignedAlloc((nuint)((long)OutputDim * Rank * sizeof(float)), 64);
        _yOut = (float*)NativeMemory.AlignedAlloc((nuint)((long)N * OutputDim * sizeof(float)), 64);
        for (long i = 0; i < (long)OutputDim * Rank; i++) _aF32[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.4f;
        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;

        // Build the transposed A layout for the outer-product fused stage-2.
        _aT = Kernels.TransposeA_OutputDimByRank(_aF32, OutputDim, Rank);

        // Pre-populate _tmp with a stage-1 result so stage-2 has real data to chew on.
        Kernels.Stage1_PathBC_R16Interleaved_Avx512(_xQ8, _bR16, _tmp, N, _blockCount);

        // ─── Correctness gate — cross-check kernels against scalar F32 ref ──────
        // We compare bit-equivalence-up-to-Q8_0 for the explicit-locals kernels.
        // GemmF32 baseline reads dequanted B (matches the production dequant-once
        // path). Outputs may differ by a few percent due to Q8_0 stage-1
        // rounding — that's the tolerance LoraDeltaQuantizedQ8_0Tests already use.
        var refOut = new float[N * Rank];
        var pathCOut = new float[N * Rank];
        var pathC2Out = new float[N * Rank];
        var pathBCOut = new float[N * Rank];

        // F32 ref: GemmF32 reads (dequanted) B and F32 x — but here we want a
        // Q8_0×Q8_0 reference so the tolerance bound is small. Use the existing
        // GemmQ8_0(preQuantizedInput) path as the bit-exact reference.
        fixed (float* refPtr = refOut)
            MatMul.GemmQ8_0(_bRowMajor, b: null, c: refPtr, m: Rank, k: K, n: N, preQuantizedInput: _xQ8);

        fixed (float* outC = pathCOut)
            Kernels.Stage1_PathC_R16_Avx512(_xQ8, _bRowMajor, outC, N, _blockCount, _rowBytes);
        fixed (float* outC2 = pathC2Out)
            Kernels.Stage1_PathC2_R16_Avx512Dual(_xQ8, _bRowMajor, outC2, N, _blockCount, _rowBytes);
        fixed (float* outBC = pathBCOut)
            Kernels.Stage1_PathBC_R16Interleaved_Avx512(_xQ8, _bR16, outBC, N, _blockCount);

        AssertCloseAbs(refOut, pathCOut, "PathC", absTol: 1e-2f);
        AssertCloseAbs(refOut, pathC2Out, "PathC2", absTol: 1e-2f);
        AssertCloseAbs(refOut, pathBCOut, "PathBC", absTol: 1e-2f);

        // ─── Stage-2 correctness gate ─────────────────────────────────────────
        // Compare S2_OuterProduct_R16 output against the production GemvPerToken
        // path. Both consume the same _tmp + the same A (one row-major, one
        // transposed). y starts at 0, so a single Apply of either kernel writes
        // the full delta. Reset y between runs.
        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
        S2_GemvPerToken();
        var s2Ref = new float[N * OutputDim];
        new Span<float>(_yOut, N * OutputDim).CopyTo(s2Ref);

        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
        S2_OuterProduct_R16();
        var s2Cand = new float[N * OutputDim];
        new Span<float>(_yOut, N * OutputDim).CopyTo(s2Cand);

        AssertCloseAbs(s2Ref, s2Cand, "S2_OuterProduct_R16", absTol: 1e-3f);

        // ─── E2E correctness gate ─────────────────────────────────────────────
        // The fully-fused kernel must produce the same result as production
        // (stage 1 + stage 2). y is read+modified, so each E2E call must start
        // from the same y0 state. We compare two E2E paths against each other,
        // both starting from y=0.

        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
        E2E_Production_F32_DequantOnce();
        var e2eRef = new float[N * OutputDim];
        new Span<float>(_yOut, N * OutputDim).CopyTo(e2eRef);

        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
        E2E_New_F32_OuterProduct();
        var e2eOP = new float[N * OutputDim];
        new Span<float>(_yOut, N * OutputDim).CopyTo(e2eOP);

        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
        E2E_New_F32_FullyFused();
        var e2eFused = new float[N * OutputDim];
        new Span<float>(_yOut, N * OutputDim).CopyTo(e2eFused);

        AssertCloseAbs(e2eRef, e2eOP, "E2E_OuterProduct", absTol: 1e-3f);
        AssertCloseAbs(e2eRef, e2eFused, "E2E_FullyFused", absTol: 1e-3f);

        // Reset y so benchmark iterations start from a known state.
        for (long i = 0; i < (long)N * OutputDim; i++) _yOut[i] = 0;
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (_xF32 != null) NativeMemory.AlignedFree(_xF32);
        if (_bF32 != null) NativeMemory.AlignedFree(_bF32);
        if (_xQ8 != null) NativeMemory.AlignedFree(_xQ8);
        if (_bRowMajor != null) NativeMemory.AlignedFree(_bRowMajor);
        if (_bR16 != null) NativeMemory.AlignedFree(_bR16);
        if (_bR4 != null) NativeMemory.AlignedFree(_bR4);
        if (_tmp != null) NativeMemory.AlignedFree(_tmp);
        if (_aF32 != null) NativeMemory.AlignedFree(_aF32);
        if (_aT != null) NativeMemory.AlignedFree(_aT);
        if (_yOut != null) NativeMemory.AlignedFree(_yOut);
    }

    // ─────────────────────── Baselines & candidates ─────────────────────────────
    // Each benchmark performs the stage-1 work once. Kernel-level metric is mean
    // wall time per call.

    /// <summary>
    /// Production baseline (Phase 4d.4): dequant B once into F32 scratch, then
    /// run GemmF32 stage 1. This is the path that's currently shipped — the bar
    /// to beat.
    /// </summary>
    [Benchmark(Baseline = true, Description = "F32_DequantOnce")]
    public void F32_DequantOnce()
    {
        // Dequant cost is amortised once per LoRA call (F32 scratch fits in L2).
        // We include it in the baseline because that's exactly what production
        // pays per ApplyLoraDelta. Use ArrayPool to match production's exact
        // allocation behaviour (rather than a per-call new[] which would skew
        // the comparison against itself).
        int bElems = Rank * K;
        float[] bBuf = System.Buffers.ArrayPool<float>.Shared.Rent(bElems);
        try
        {
            fixed (float* bF32Ptr = bBuf)
            {
                // Dequant Q8_0 B → F32 scratch using the production helper
                // path (DotLLM.Cpu.Kernels.Dequantize.ToFloat32 has SIMD).
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32(
                    (nint)_bRowMajor, (long)bElems,
                    DotLLM.Core.Configuration.QuantizationType.Q8_0,
                    new Span<float>(bF32Ptr, bElems));
                MatMul.GemmF32(bF32Ptr, _xF32, _tmp, Rank, K, N);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    /// <summary>
    /// Just the GEMM (no dequant) — measures the F32 stage-1 floor.
    /// </summary>
    [Benchmark(Description = "F32_OnlyGemm")]
    public void F32_OnlyGemm()
    {
        // bF32 is already populated from setup — measure only the GEMM step.
        MatMul.GemmF32(_bF32, _xF32, _tmp, Rank, K, N);
    }

    /// <summary>
    /// Just the dequant (no GEMM) — measures the dequant cost in isolation.
    /// </summary>
    [Benchmark(Description = "F32_OnlyDequant")]
    public void F32_OnlyDequant()
    {
        int bElems = Rank * K;
        float[] bBuf = System.Buffers.ArrayPool<float>.Shared.Rent(bElems);
        try
        {
            fixed (float* bF32Ptr = bBuf)
            {
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32(
                    (nint)_bRowMajor, (long)bElems,
                    DotLLM.Core.Configuration.QuantizationType.Q8_0,
                    new Span<float>(bF32Ptr, bElems));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    /// <summary>
    /// Phase 4d.5 (gated) baseline: GemmQ8_0(preQuantizedInput=xQ8). The current
    /// implementation behind DOTLLM_LORA_FORCE_Q8_PREQUANT=1.
    /// </summary>
    [Benchmark(Description = "Q8_0_GemmPreQuantX")]
    public void Q8_0_GemmPreQuantX()
        => MatMul.GemmQ8_0(_bRowMajor, b: null, c: _tmp, m: Rank, k: K, n: N, preQuantizedInput: _xQ8);

    /// <summary>Path C — explicit Vector256/512 locals, single-block inner.</summary>
    [Benchmark(Description = "PathC_R16_Single")]
    public void PathC_R16_Single()
        => Kernels.Stage1_PathC_R16_Avx512(_xQ8, _bRowMajor, _tmp, N, _blockCount, _rowBytes);

    /// <summary>Path C2 — explicit Vector512 locals, dual-block inner (closest analogue of _4Rows).</summary>
    [Benchmark(Description = "PathC2_R16_Dual")]
    public void PathC2_R16_Dual()
        => Kernels.Stage1_PathC2_R16_Avx512Dual(_xQ8, _bRowMajor, _tmp, N, _blockCount, _rowBytes);

    /// <summary>Path B + C — explicit locals + R16 interleaved layout.</summary>
    [Benchmark(Description = "PathBC_R16Interleaved")]
    public void PathBC_R16Interleaved()
        => Kernels.Stage1_PathBC_R16Interleaved_Avx512(_xQ8, _bR16, _tmp, N, _blockCount);

    /// <summary>
    /// Path B3 — re-use the existing <c>MatMul.OuterProductGemmQ8_0</c> kernel
    /// by repacking B at adapter-load into R4 layout. Zero new SIMD; the
    /// 4×6 AVX-512 microkernel (24 ZMM accumulators) is what wins on base-model
    /// prefill. The investigation in .planning/notes/lora-q8-stage1-investigation.md
    /// argues this should beat row-major B because the per-token B re-reads
    /// dissolve into a streamed access pattern that Zen 5's prefetcher tracks.
    /// </summary>
    [Benchmark(Description = "Q8_0_R4_Reuse_OuterProduct")]
    public void Q8_0_R4_Reuse_OuterProduct()
        => MatMul.OuterProductGemmQ8_0(
            _bR4, _xQ8, _tmp,
            fullGroups: Rank / 4, tailRows: Rank % 4,
            blockCount: _blockCount, m: Rank, n: N);

    // ─── Stage-2 baselines & candidates ────────────────────────────────────────
    // Stage 2: y[t, o] += scale * sum_r A[o, r] * tmp[t, r]. Geometry: A is
    // [outputDim=2048, rank=16] F32, tmp is [N=512, rank=16] F32, y is
    // [N, outputDim] F32. Production calls GemvF32(A, tmp+t*rank, delta, ...)
    // per token then TensorPrimitives.MultiplyAdd into y. We measure the
    // existing path and a fused outer-product variant that needs no scratch.

    [Benchmark(Description = "S2_GemvPerToken")]
    public void S2_GemvPerToken()
    {
        // Production path: per token, GemvF32(A, tmp_t, delta) then y += scale*delta.
        const float scale = 0.5f;
        float[] deltaBuf = System.Buffers.ArrayPool<float>.Shared.Rent(OutputDim);
        try
        {
            fixed (float* delta = deltaBuf)
            {
                for (int t = 0; t < N; t++)
                {
                    MatMul.GemvF32(_aF32, _tmp + (long)t * Rank, delta, OutputDim, Rank);
                    var deltaSpan = new ReadOnlySpan<float>(delta, OutputDim);
                    var ySpan = new Span<float>(_yOut + (long)t * OutputDim, OutputDim);
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(deltaSpan, scale, ySpan, ySpan);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(deltaBuf);
        }
    }

    /// <summary>
    /// Path E1 — outer-product stage-2 with transposed A. Uses the AT layout
    /// rebuilt at adapter load (one-shot cost). Per-token: 16 broadcasts +
    /// outputDim/16 tiles of 16 FMAs each + scaled accumulate into y.
    /// </summary>
    [Benchmark(Description = "S2_OuterProduct_R16")]
    public void S2_OuterProduct_R16()
        => Kernels.Stage2_OuterProduct_R16_Avx512(_aT, _tmp, _yOut, N, OutputDim, scale: 0.5f);

    // ─── End-to-end LoRA delta benchmarks (stage 1 + stage 2) ──────────────────
    // These reflect the FULL LoRA Apply call cost — what the macro-bench
    // actually pays per projection per layer per prefill.

    /// <summary>
    /// Production end-to-end: dequant-once F32 stage 1 + GemvPerToken stage 2.
    /// Mirrors the path TransformerModel.ApplyLoraDelta hits today on Q8_0-B
    /// adapters with the env var off.
    /// </summary>
    [Benchmark(Description = "E2E_Production_F32_DequantOnce")]
    public void E2E_Production_F32_DequantOnce()
    {
        F32_DequantOnce();           // stage 1
        S2_GemvPerToken();           // stage 2
    }

    /// <summary>
    /// New: dequant-once F32 stage 1 + outer-product stage 2.
    /// Only stage 2 changes vs production. Minimal disruption.
    /// </summary>
    [Benchmark(Description = "E2E_New_F32_OuterProduct")]
    public void E2E_New_F32_OuterProduct()
    {
        F32_DequantOnce();              // stage 1 unchanged
        S2_OuterProduct_R16();          // stage 2 outer-product
    }

    /// <summary>
    /// Path F: fully fused F32 stage-1 + outer-product stage-2 in one kernel.
    /// No materialised tmp buffer; per-token tmp lives in registers.
    /// </summary>
    [Benchmark(Description = "E2E_New_F32_FullyFused")]
    public void E2E_New_F32_FullyFused()
    {
        // Dequant B once into a pooled scratch (matches production memory pattern).
        int bElems = Rank * K;
        float[] bBuf = System.Buffers.ArrayPool<float>.Shared.Rent(bElems);
        try
        {
            fixed (float* bF32Ptr = bBuf)
            {
                DotLLM.Cpu.Kernels.Dequantize.ToFloat32(
                    (nint)_bRowMajor, (long)bElems,
                    DotLLM.Core.Configuration.QuantizationType.Q8_0,
                    new Span<float>(bF32Ptr, bElems));
                Kernels.Stage12_Fused_F32_R16_Avx512(_xF32, bF32Ptr, _aT, _yOut,
                    N, K, OutputDim, scale: 0.5f);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(bBuf);
        }
    }

    private static void AssertCloseAbs(float[] expected, float[] actual, string label, float absTol)
    {
        if (expected.Length != actual.Length)
            throw new InvalidOperationException($"{label}: length mismatch");
        float maxDiff = 0;
        int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]);
            if (d > maxDiff) { maxDiff = d; worst = i; }
        }
        Console.WriteLine($"[correctness] {label}: maxAbsDiff = {maxDiff:G4} at idx {worst} (tol {absTol})");
        if (maxDiff > absTol)
            throw new InvalidOperationException($"{label}: maxAbsDiff {maxDiff} > tol {absTol}");
    }
}

public static class Program
{
    public static void Main(string[] args)
    {
        // For interactive iteration: also allow `--quick` to run a single in-proc shot.
        if (args.Length > 0 && args[0] == "--quick")
        {
            QuickRun();
            return;
        }

        var config = DefaultConfig.Instance
            .AddDiagnoser(MemoryDiagnoser.Default)
            .AddJob(Job.Default
                .WithStrategy(RunStrategy.Throughput)
                .WithWarmupCount(3)
                .WithIterationCount(8));
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
    }

    /// <summary>
    /// Quick in-proc smoke run. Avoids BenchmarkDotNet overhead — just gives
    /// us per-call wall-time for fast iteration before we run the real bench.
    /// </summary>
    private static unsafe void QuickRun()
    {
        // OutputDim swept manually in quick mode.
        foreach (int outputDim in new[] { 512, 2048, 5632 })
        {
            Console.WriteLine();
            Console.WriteLine($"════ OutputDim = {outputDim} ════");
            var bench = new LoraStage1Bench { K = 2048, N = 512, OutputDim = outputDim };
            bench.Setup();
            RunQuickFor(bench);
            bench.Cleanup();
        }
    }

    private static void RunQuickFor(LoraStage1Bench bench)
    {

        // Simple per-kernel timing.
        Time("F32_DequantOnce", iters: 200, () => bench.F32_DequantOnce());
        Time("F32_OnlyGemm", iters: 200, () => bench.F32_OnlyGemm());
        Time("F32_OnlyDequant", iters: 200, () => bench.F32_OnlyDequant());
        Time("Q8_0_GemmPreQuantX", iters: 200, () => bench.Q8_0_GemmPreQuantX());
        Time("PathC_R16_Single", iters: 200, () => bench.PathC_R16_Single());
        Time("PathC2_R16_Dual", iters: 200, () => bench.PathC2_R16_Dual());
        Time("PathBC_R16Interleaved", iters: 200, () => bench.PathBC_R16Interleaved());
        Time("Q8_0_R4_Reuse_OuterProduct", iters: 200, () => bench.Q8_0_R4_Reuse_OuterProduct());
        Time("S2_GemvPerToken", iters: 200, () => bench.S2_GemvPerToken());
        Time("S2_OuterProduct_R16", iters: 200, () => bench.S2_OuterProduct_R16());
        Console.WriteLine();
        Console.WriteLine("--- End-to-end (stage 1 + stage 2) ---");
        Time("E2E_Production_F32_DequantOnce", iters: 100, () => bench.E2E_Production_F32_DequantOnce());
        Time("E2E_New_F32_OuterProduct", iters: 100, () => bench.E2E_New_F32_OuterProduct());
        Time("E2E_New_F32_FullyFused", iters: 100, () => bench.E2E_New_F32_FullyFused());
    }

    private static void Time(string label, int iters, Action body)
    {
        // Warm.
        for (int i = 0; i < 30; i++) body();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) body();
        sw.Stop();
        double usPerCall = sw.Elapsed.TotalMicroseconds / iters;
        Console.WriteLine($"[quick]  {label,-30} {usPerCall,8:F2} us/call ({iters} iters)");
    }
}
