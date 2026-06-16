using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Numeric parity + complete-path benchmark for the G3 tensor-core prefill-attention
/// prototype: QK^T (cuBLAS strided-batched GEMM) → causal softmax (custom kernel) →
/// P·V (cuBLAS strided-batched GEMM), all FP16 in/out with FP32 GEMM accumulation,
/// GQA expressed as a group-sized batch with KV stride 0. The reference is the
/// fused <c>attention_f16</c> CUDA-core kernel.
///
/// <para>
/// <b>Output layout differs between the two paths.</b> The P·V GEMM writes per query
/// head a <b>column-major</b> <c>[s × headDim]</c> block: element <c>(tq, d)</c> at
/// <c>hq*s*headDim + tq + d*s</c>. <c>attention_f16</c> writes <b>row-major</b>
/// <c>[seq, heads, headDim]</c>: element <c>(tq, hq, d)</c> at
/// <c>tq*numHeads*headDim + hq*headDim + d</c>. The comparison reindexes both rather
/// than memcpy-and-compare-linearly (which would fail a perfectly correct kernel).
/// </para>
///
/// <para>
/// <b>Tolerance.</b> Per-element pass if <c>abs ≤ 5e-3</c> OR <c>rel ≤ 5e-3</c>,
/// mirroring the repo's coopmat FP16 precedent. This path keeps the QK scores in FP16
/// before softmax (the FP32 reference keeps them in FP32), so the bar is not tightened
/// below 5e-3. It is still discriminating: disabling the causal zeroing or swapping
/// the softmax reduction axis to stride-1 both turn the test red (verified manually
/// during bring-up — see the kernel header).
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaTensorCoreAttentionParityTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    private readonly ITestOutputHelper _output;

    public CudaTensorCoreAttentionParityTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(128, 32, 8, 64)]
    [InlineData(256, 32, 8, 64)]
    [InlineData(512, 32, 8, 64)]
    [InlineData(1024, 32, 8, 64)]
    public void CublasSoftmaxPath_MatchesAttentionF16(int s, int numHeads, int numKvHeads, int headDim)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string ptxDir = ResolvePtxDir();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "attention_softmax_causal.ptx")),
            "attention_softmax_causal.ptx not built.");

        using var ctx = CudaContext.Create(0);
        ctx.MakeCurrent();
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        Skip.IfNot(kernels.HasAttentionSoftmaxCausal, "Causal-softmax kernel not loaded.");

        int group = numHeads / numKvHeads;
        float scale = 1.0f / MathF.Sqrt(headDim);

        int qElems = s * numHeads * headDim;
        int kvElems = s * numKvHeads * headDim;
        int scoreElems = numHeads * s * s;
        int outElems = s * numHeads * headDim;

        // Same random Q/K/V into both paths (matching the bench's fill scale).
        ushort[] qHost = RandomHalf(qElems, seed: 1);
        ushort[] kHost = RandomHalf(kvElems, seed: 2);
        ushort[] vHost = RandomHalf(kvElems, seed: 3);

        using var q = CudaTensor.Allocate(new TensorShape(qElems), DType.Float16, 0);
        using var k = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
        using var v = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
        using var scores = CudaTensor.Allocate(new TensorShape(scoreElems), DType.Float16, 0);
        using var outCublas = CudaTensor.Allocate(new TensorShape(outElems), DType.Float16, 0);
        using var outRef = CudaTensor.Allocate(new TensorShape(outElems), DType.Float16, 0);

        Upload(q.DataPointer, qHost);
        Upload(k.DataPointer, kHost);
        Upload(v.DataPointer, vHost);

        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;

        // ── Reference: fused attention_f16 (row-major output) ──
        kernels.LaunchAttention(q.DataPointer, k.DataPointer, v.DataPointer, outRef.DataPointer,
            seqQ: s, seqKv: s, numHeads, numKvHeads, headDim,
            positionOffset: 0, slidingWindow: 0, stream.Handle);

        ushort[] refHost = Download(outRef.DataPointer, outElems);

        // ── G3 path: QK GEMM → causal softmax → P·V GEMM (col-major output) ──
        // Verify both softmax variants (coalesced = shipping path, non-coalesced sibling).
        foreach (bool coalesced in new[] { true, false })
        {
            RunCublasSoftmaxPath(cublas, kernels, stream, q, k, v, scores, outCublas,
                s, numHeads, numKvHeads, headDim, group, qStride, kvStride, scale, coalesced);
            stream.Synchronize();

            ushort[] cublasHost = Download(outCublas.DataPointer, outElems);

            // Reindex per the two output layouts and compare.
            int mismatches = 0;
            float maxAbs = 0f, maxRel = 0f;
            int worstTq = -1, worstHq = -1, worstD = -1;
            for (int hq = 0; hq < numHeads; hq++)
            {
                for (int tq = 0; tq < s; tq++)
                {
                    for (int d = 0; d < headDim; d++)
                    {
                        int refIdx = tq * numHeads * headDim + hq * headDim + d;      // row-major
                        int cubIdx = hq * s * headDim + tq + d * s;                   // col-major per head
                        float a = (float)BitConverter.UInt16BitsToHalf(refHost[refIdx]);
                        float b = (float)BitConverter.UInt16BitsToHalf(cublasHost[cubIdx]);
                        float absDiff = MathF.Abs(a - b);
                        float relDiff = absDiff / (MathF.Abs(a) + 1e-6f);
                        bool pass = absDiff <= AbsTol || relDiff <= RelTol;
                        if (!pass)
                        {
                            mismatches++;
                            if (absDiff > maxAbs) { maxAbs = absDiff; maxRel = relDiff; worstTq = tq; worstHq = hq; worstD = d; }
                        }
                    }
                }
            }

            _output.WriteLine($"s={s} heads={numHeads} kv={numKvHeads} hd={headDim} coalesced={coalesced}: "
                + $"mismatches={mismatches}/{outElems} (tol abs {AbsTol} OR rel {RelTol})");
            if (mismatches > 0)
                _output.WriteLine($"worst @ (tq={worstTq},hq={worstHq},d={worstD}) absDiff={maxAbs} relDiff={maxRel}");

            Assert.True(mismatches == 0,
                $"cuBLAS+softmax (coalesced={coalesced}) vs attention_f16: {mismatches}/{outElems} elements "
              + $"outside tolerance (abs {AbsTol} OR rel {RelTol}); worst abs {maxAbs} rel {maxRel} at "
              + $"(tq={worstTq},hq={worstHq},d={worstD}).");
        }
    }

    /// <summary>
    /// Complete-path timing vs <c>attention_f16</c>: QK GEMM → causal softmax → P·V GEMM.
    /// CUDA events, interleaved, min over reps (the 3060 clocks drift ~2× across heavy
    /// runs — interleave + min, never divide separate fresh-process mins). Opt-in via
    /// <c>DOTLLM_CUDA_ATTN_BENCH=1</c>.
    /// </summary>
    [SkippableFact]
    public void CompletePathTimingVsAttentionF16()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_ATTN_BENCH") == "1", "DOTLLM_CUDA_ATTN_BENCH=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string ptxDir = ResolvePtxDir();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "attention_softmax_causal.ptx")), "attention_softmax_causal.ptx not built.");

        int numHeads = EnvInt("DOTLLM_CUDA_ATTN_HEADS", 32);
        int numKvHeads = EnvInt("DOTLLM_CUDA_ATTN_KVHEADS", 8);
        int headDim = EnvInt("DOTLLM_CUDA_ATTN_HEADDIM", 64);
        int group = numHeads / numKvHeads;
        int[] seqs = { 256, 512, 1024, 2048 };
        int reps = 30, warmup = 6;
        float scale = 1.0f / MathF.Sqrt(headDim);

        using var ctx = CudaContext.Create(0);
        ctx.MakeCurrent();
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        Skip.IfNot(kernels.HasAttentionSoftmaxCausal, "Causal-softmax kernel not loaded.");

        _output.WriteLine("G3 complete cuBLAS+softmax path vs attention_f16 (full QK + softmax + PV)");
        _output.WriteLine($"heads={numHeads} kvHeads={numKvHeads} headDim={headDim} group={group}  reps={reps} (min ms, interleaved)");
        _output.WriteLine($"{"seq",6} | {"attn_f16",10} | {"cb+sm(uncoal)",14} | {"spd",6} | {"cb+sm(coal)",13} | {"spd",6}");
        _output.WriteLine(new string('-', 70));

        foreach (int s in seqs)
        {
            int qElems = s * numHeads * headDim;
            int kvElems = s * numKvHeads * headDim;
            int scoreElems = numHeads * s * s;
            int outElems = s * numHeads * headDim;

            using var q = CudaTensor.Allocate(new TensorShape(qElems), DType.Float16, 0);
            using var k = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
            using var v = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
            using var scores = CudaTensor.Allocate(new TensorShape(scoreElems), DType.Float16, 0);
            using var outBuf = CudaTensor.Allocate(new TensorShape(outElems), DType.Float16, 0);

            Upload(q.DataPointer, RandomHalf(qElems, 1));
            Upload(k.DataPointer, RandomHalf(kvElems, 2));
            Upload(v.DataPointer, RandomHalf(kvElems, 3));

            int qStride = numHeads * headDim;
            int kvStride = numKvHeads * headDim;

            void RunAttention() => kernels.LaunchAttention(
                q.DataPointer, k.DataPointer, v.DataPointer, outBuf.DataPointer,
                seqQ: s, seqKv: s, numHeads, numKvHeads, headDim, 0, 0, stream.Handle);

            void RunUncoal() => RunCublasSoftmaxPath(cublas, kernels, stream, q, k, v, scores, outBuf,
                s, numHeads, numKvHeads, headDim, group, qStride, kvStride, scale, coalesced: false);
            void RunCoal() => RunCublasSoftmaxPath(cublas, kernels, stream, q, k, v, scores, outBuf,
                s, numHeads, numKvHeads, headDim, group, qStride, kvStride, scale, coalesced: true);

            for (int w = 0; w < warmup; w++) { RunAttention(); RunUncoal(); RunCoal(); }
            stream.Synchronize();

            double attnMin = double.MaxValue, uncoalMin = double.MaxValue, coalMin = double.MaxValue;
            for (int r = 0; r < reps; r++)
            {
                attnMin = Math.Min(attnMin, TimeGpu(stream, RunAttention));
                uncoalMin = Math.Min(uncoalMin, TimeGpu(stream, RunUncoal));
                coalMin = Math.Min(coalMin, TimeGpu(stream, RunCoal));
            }

            _output.WriteLine($"{s,6} | {attnMin,8:F3}ms | {uncoalMin,12:F3}ms | {attnMin / uncoalMin,5:F2}x | {coalMin,11:F3}ms | {attnMin / coalMin,5:F2}x");
        }
        _output.WriteLine(new string('-', 70));
        _output.WriteLine("spd = attn_f16 / (cublas+softmax). coal = one-thread-per-row (coalesced reads).");
    }

    private static void RunCublasSoftmaxPath(
        CudaCublasHandle cublas, CudaKernels kernels, CudaStream stream,
        CudaTensor q, CudaTensor k, CudaTensor v, CudaTensor scores, CudaTensor outBuf,
        int s, int numHeads, int numKvHeads, int headDim, int group,
        int qStride, int kvStride, float scale, bool coalesced)
    {
        float one = 1.0f, zero = 0.0f, sc = scale;
        for (int h = 0; h < numKvHeads; h++)
        {
            nint qBase = q.DataPointer + (nint)((long)h * group * headDim * 2);
            nint kBase = k.DataPointer + (nint)((long)h * headDim * 2);
            nint vBase = v.DataPointer + (nint)((long)h * headDim * 2);
            nint scBase = scores.DataPointer + (nint)((long)h * group * s * s * 2);
            nint oBase = outBuf.DataPointer + (nint)((long)h * group * s * headDim * 2);

            // QK^T: scores[tq,tk] = scale * Σ_d Q[tq,d]·K[tk,d]  (col-major [s × s], ldc=s)
            CublasApi.cublasGemmStridedBatchedEx(cublas.Handle,
                CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                s, s, headDim,
                (nint)(&sc),
                qBase, CublasApi.CUDA_R_16F, qStride, headDim,
                kBase, CublasApi.CUDA_R_16F, kvStride, 0,
                (nint)(&zero),
                scBase, CublasApi.CUDA_R_16F, s, (long)s * s,
                group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }

        // Causal softmax over the whole scores buffer (all numHeads planes), in place.
        kernels.LaunchAttentionSoftmaxCausal(scores.DataPointer, s, numHeads, stream.Handle, coalesced);

        for (int h = 0; h < numKvHeads; h++)
        {
            nint vBase = v.DataPointer + (nint)((long)h * headDim * 2);
            nint scBase = scores.DataPointer + (nint)((long)h * group * s * s * 2);
            nint oBase = outBuf.DataPointer + (nint)((long)h * group * s * headDim * 2);

            // P·V: out[tq,d] = Σ_tk scores[tq,tk]·V[tk,d]  (col-major [s × headDim], ldc=s)
            CublasApi.cublasGemmStridedBatchedEx(cublas.Handle,
                CublasApi.CUBLAS_OP_N, CublasApi.CUBLAS_OP_T,
                s, headDim, s,
                (nint)(&one),
                scBase, CublasApi.CUDA_R_16F, s, (long)s * s,
                vBase, CublasApi.CUDA_R_16F, kvStride, 0,
                (nint)(&zero),
                oBase, CublasApi.CUDA_R_16F, s, (long)s * headDim,
                group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
        }
    }

    private static double TimeGpu(CudaStream stream, Action work)
    {
        CudaDriverApi.cuEventCreate(out nint start, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint stop, CudaDriverApi.CU_EVENT_DEFAULT).ThrowOnError();
        try
        {
            CudaDriverApi.cuEventRecord(start, stream.Handle).ThrowOnError();
            work();
            CudaDriverApi.cuEventRecord(stop, stream.Handle).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(stop).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, start, stop).ThrowOnError();
            return ms;
        }
        finally
        {
            CudaDriverApi.cuEventDestroy_v2(start);
            CudaDriverApi.cuEventDestroy_v2(stop);
        }
    }

    private static ushort[] RandomHalf(long elems, int seed)
    {
        var rng = new Random(seed);
        var host = new ushort[elems];
        for (long i = 0; i < elems; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextSingle() - 0.5f) * 0.2f));
        return host;
    }

    private static void Upload(nint dst, ushort[] host)
    {
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(host.Length * 2)).ThrowOnError();
    }

    private static ushort[] Download(nint src, int elems)
    {
        var host = new ushort[elems];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(elems * 2)).ThrowOnError();
        return host;
    }

    private static int EnvInt(string name, int dflt)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : dflt;

    private static string ResolvePtxDir()
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            string cand = Path.Combine(dir, "native", "ptx");
            if (Directory.Exists(cand)) return cand;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return Path.Combine(AppContext.BaseDirectory, "native", "ptx");
    }
}
