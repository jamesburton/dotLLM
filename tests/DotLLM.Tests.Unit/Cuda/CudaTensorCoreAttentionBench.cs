using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// GEMM-only ceiling check for a tensor-core prefill-attention rewrite.
///
/// Prefill attention is the #1 GPU prize (profiled at ~43% of pp256 on the RTX 3060) and the
/// current <c>native/kernels/attention.cu</c> is a flash-style FP32 kernel running on CUDA cores —
/// the tensor cores sit idle. Before committing to a fused FP16 flash kernel (the correctness-heavy
/// path), this measures the <b>lower bound</b> of a cuBLAS tensor-core implementation: just the two
/// batched GEMMs (QK^T → scores, then scores·V → out), with <b>no softmax and no correctness</b>.
/// The two GEMMs dominate the FLOPs, so their combined time is the floor a real kernel can approach
/// but never beat (softmax + the score round-trip only add cost). If this floor does not beat the
/// existing <c>attention_f16</c> by a shippable margin, a cuBLAS-based rewrite is not worth it and a
/// hand-fused flash kernel (keeping scores in registers/shared, never materialising them) is the only
/// path — that is the go/no-go this test answers, plus the crossover seq length.
///
/// GQA is expressed as one strided-batched GEMM per KV head (batch = group size, K/V stride 0 so the
/// shared KV head broadcasts across its query-head group). Opt-in via <c>DOTLLM_CUDA_ATTN_BENCH=1</c>.
/// Shapes default to Llama-3.2-1B (numHeads=32, numKvHeads=8, headDim=64); override head dims via
/// <c>DOTLLM_CUDA_ATTN_HEADS</c> / <c>_KVHEADS</c> / <c>_HEADDIM</c>.
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaTensorCoreAttentionBench
{
    private readonly ITestOutputHelper _output;

    public CudaTensorCoreAttentionBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void GemmOnlyCeilingVsAttentionF16()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_ATTN_BENCH") == "1", "DOTLLM_CUDA_ATTN_BENCH=1 not set.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        int numHeads = EnvInt("DOTLLM_CUDA_ATTN_HEADS", 32);
        int numKvHeads = EnvInt("DOTLLM_CUDA_ATTN_KVHEADS", 8);
        int headDim = EnvInt("DOTLLM_CUDA_ATTN_HEADDIM", 64);
        int group = numHeads / numKvHeads;
        int[] seqs = { 256, 512, 1024, 2048 };
        int reps = 30, warmup = 6;
        string ptxDir = ResolvePtxDir();

        using var ctx = CudaContext.Create(0);
        ctx.MakeCurrent();
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        _output.WriteLine($"tensor-core prefill-attention GEMM-only ceiling vs attention_f16");
        _output.WriteLine($"heads={numHeads} kvHeads={numKvHeads} headDim={headDim} group={group}  reps={reps} (min ms, interleaved)");
        _output.WriteLine($"{"seq",6} | {"attn_f16",10} | {"gemm-only",10} | {"ceiling vs attn",16}");
        _output.WriteLine(new string('-', 54));

        float scale = 1.0f / MathF.Sqrt(headDim);

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

            FillRandomHalf(q, qElems, seed: 1);
            FillRandomHalf(k, kvElems, seed: 2);
            FillRandomHalf(v, kvElems, seed: 3);

            int qStride = numHeads * headDim;
            int kvStride = numKvHeads * headDim;

            void RunAttention()
            {
                kernels.LaunchAttention(q.DataPointer, k.DataPointer, v.DataPointer, outBuf.DataPointer,
                    seqQ: s, seqKv: s, numHeads, numKvHeads, headDim,
                    positionOffset: 0, slidingWindow: 0, stream.Handle);
            }

            void RunGemmOnly()
            {
                float one = 1.0f, zero = 0.0f, sc = scale;
                for (int h = 0; h < numKvHeads; h++)
                {
                    nint qBase = q.DataPointer + (nint)((long)h * group * headDim * 2);
                    nint kBase = k.DataPointer + (nint)((long)h * headDim * 2);
                    nint vBase = v.DataPointer + (nint)((long)h * headDim * 2);
                    nint scBase = scores.DataPointer + (nint)((long)h * group * s * s * 2);
                    nint oBase = outBuf.DataPointer + (nint)((long)h * group * s * headDim * 2);

                    // QK^T: scores[tq,tk] = scale * Σ_d Q[tq,d]·K[tk,d]  (col-major scores [s × s], ldc=s)
                    CublasApi.cublasGemmStridedBatchedEx(cublas.Handle,
                        CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
                        s, s, headDim,
                        (nint)(&sc),
                        qBase, CublasApi.CUDA_R_16F, qStride, headDim,
                        kBase, CublasApi.CUDA_R_16F, kvStride, 0,
                        (nint)(&zero),
                        scBase, CublasApi.CUDA_R_16F, s, (long)s * s,
                        group, CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();

                    // P·V: out[tq,d] = Σ_tk scores[tq,tk]·V[tk,d]  (col-major out [s × headDim], ldc=s)
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

            for (int w = 0; w < warmup; w++) { RunAttention(); RunGemmOnly(); }
            stream.Synchronize();

            double attnMin = double.MaxValue, gemmMin = double.MaxValue;
            for (int r = 0; r < reps; r++)
            {
                attnMin = Math.Min(attnMin, TimeGpu(stream, RunAttention));
                gemmMin = Math.Min(gemmMin, TimeGpu(stream, RunGemmOnly));
            }

            _output.WriteLine($"{s,6} | {attnMin,8:F3}ms | {gemmMin,8:F3}ms | {attnMin / gemmMin,13:F2}x");
        }

        _output.WriteLine(new string('-', 54));
        _output.WriteLine("ceiling = attn_f16 / gemm-only. >1 means a tensor-core GEMM attention *could* be faster;");
        _output.WriteLine("the real fused kernel lands between gemm-only (floor) and attn_f16. <~1.3x at a seq = not worth a cuBLAS rewrite there.");
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

    private static void FillRandomHalf(CudaTensor t, long elems, int seed)
    {
        var rng = new Random(seed);
        var host = new ushort[elems];
        for (long i = 0; i < elems; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextSingle() - 0.5f) * 0.2f));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(t.DataPointer, (nint)p, (nuint)(elems * 2)).ThrowOnError();
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
