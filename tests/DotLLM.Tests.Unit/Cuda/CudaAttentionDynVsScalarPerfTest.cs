using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Diagnostic-only (issue #213): isolates whether <c>attention_f16_dyn</c> (the CUDA-Graphs
/// decode entry point, which reads <c>seq_kv</c>/<c>position_offset</c> via a device-pointer
/// dereference) is inherently slower per-launch than <c>attention_f16</c> (scalar kernel args),
/// INDEPENDENT of CUDA Graph replay itself — no graph capture happens in this test at all, both
/// kernels are launched directly on a stream via <see cref="CudaKernels.LaunchAttention"/> /
/// <see cref="CudaKernels.LaunchAttentionDyn"/>. Both entry points share the exact same
/// <c>__forceinline__</c> body (attention.cu), so if there is a genuine per-launch cost
/// difference that GROWS with seq_kv, that is strong evidence of a compiled-code (SASS)
/// difference between the two entry points, matching issue #213's "not yet tried" item #2 —
/// checking without needing Nsight Compute. Timed via CUDA events around a batch of
/// back-to-back launches (no per-launch host sync) to capture pure GPU execution time.
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaAttentionDynVsScalarPerfTest
{
    private readonly ITestOutputHelper _out;
    public CudaAttentionDynVsScalarPerfTest(ITestOutputHelper output) => _out = output;

    // BitNet-2B-4T's real decode shape (from GgufModelConfigExtractor: HiddenSize=2560,
    // NumHeads=20, NumKvHeads=5, HeadDim=128), seqQ=1 (decode).
    [SkippableFact]
    public void AttentionDyn_VsScalar_ScalesWithSeqKv()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptxDir = ResolvePtxDir();
        Skip.IfNot(Directory.Exists(ptxDir) && File.Exists(Path.Combine(ptxDir, "attention.ptx")),
            "attention.ptx not built");

        using var ctx = CudaContext.Create(0);
        ctx.MakeCurrent();
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);

        const int numHeads = 20, numKvHeads = 5, headDim = 128, seqQ = 1;
        int[] depths = { 64, 256, 512, 768, 1024, 1536, 2048 };
        const int batch = 100; // launches per timed batch (back-to-back, no per-launch sync)
        const int warmupBatches = 2;

        int maxSeqKv = depths[^1];
        int qElems = seqQ * numHeads * headDim;
        int kvElems = maxSeqKv * numKvHeads * headDim;
        int outElems = seqQ * numHeads * headDim;

        using var q = CudaTensor.Allocate(new TensorShape(qElems), DType.Float16, 0);
        using var k = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
        using var v = CudaTensor.Allocate(new TensorShape(kvElems), DType.Float16, 0);
        using var outScalar = CudaTensor.Allocate(new TensorShape(outElems), DType.Float16, 0);
        using var outDyn = CudaTensor.Allocate(new TensorShape(outElems), DType.Float16, 0);

        Upload(q.DataPointer, RandomHalf(qElems, 1));
        Upload(k.DataPointer, RandomHalf(kvElems, 2));
        Upload(v.DataPointer, RandomHalf(kvElems, 3));

        // Device-resident seq_kv / position_offset scalars for the Dyn entry point.
        CudaDriverApi.cuMemAlloc_v2(out nint seqKvDev, (nuint)sizeof(int)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint posOffDev, (nuint)sizeof(int)).ThrowOnError();

        CudaDriverApi.cuEventCreate(out nint evStart, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint evEnd, 0).ThrowOnError();

        try
        {
            var stats = kernels.DebugGetAttentionFuncStats();
            _out.WriteLine($"attention_f16: regs={stats.regsScalar} localBytes={stats.localScalar}");
            _out.WriteLine($"attention_f16_dyn: regs={stats.regsDyn} localBytes={stats.localDyn}");

            _out.WriteLine("seq_kv,scalar_us_per_launch,dyn_us_per_launch,dyn_minus_scalar_us");
            foreach (int seqKv in depths)
            {
                unsafe
                {
                    int skv = seqKv, po = seqKv - 1;
                    CudaDriverApi.cuMemcpyHtoD_v2(seqKvDev, (nint)(&skv), sizeof(int)).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD_v2(posOffDev, (nint)(&po), sizeof(int)).ThrowOnError();
                }

                // --- warm-up (both variants) ---
                for (int w = 0; w < warmupBatches; w++)
                {
                    for (int i = 0; i < batch; i++)
                        kernels.LaunchAttention(q.DataPointer, k.DataPointer, v.DataPointer, outScalar.DataPointer,
                            seqQ, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, 0, stream.Handle);
                    for (int i = 0; i < batch; i++)
                        kernels.LaunchAttentionDyn(q.DataPointer, k.DataPointer, v.DataPointer, outDyn.DataPointer,
                            seqQ, seqKvDev, numHeads, numKvHeads, headDim, posOffDev, 0, stream.Handle);
                }
                stream.Synchronize();

                // --- timed: scalar (attention_f16) ---
                CudaDriverApi.cuEventRecord(evStart, stream.Handle).ThrowOnError();
                for (int i = 0; i < batch; i++)
                    kernels.LaunchAttention(q.DataPointer, k.DataPointer, v.DataPointer, outScalar.DataPointer,
                        seqQ, seqKv, numHeads, numKvHeads, headDim, seqKv - 1, 0, stream.Handle);
                CudaDriverApi.cuEventRecord(evEnd, stream.Handle).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(evEnd).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float scalarMs, evStart, evEnd).ThrowOnError();

                // --- timed: dyn (attention_f16_dyn) ---
                CudaDriverApi.cuEventRecord(evStart, stream.Handle).ThrowOnError();
                for (int i = 0; i < batch; i++)
                    kernels.LaunchAttentionDyn(q.DataPointer, k.DataPointer, v.DataPointer, outDyn.DataPointer,
                        seqQ, seqKvDev, numHeads, numKvHeads, headDim, posOffDev, 0, stream.Handle);
                CudaDriverApi.cuEventRecord(evEnd, stream.Handle).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(evEnd).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float dynMs, evStart, evEnd).ThrowOnError();

                double scalarUs = scalarMs * 1000.0 / batch;
                double dynUs = dynMs * 1000.0 / batch;
                _out.WriteLine($"{seqKv},{scalarUs:F3},{dynUs:F3},{(dynUs - scalarUs):F3}");
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy_v2(evStart);
            CudaDriverApi.cuEventDestroy_v2(evEnd);
            CudaDriverApi.cuMemFree_v2(seqKvDev);
            CudaDriverApi.cuMemFree_v2(posOffDev);
        }
    }

    private static ushort[] RandomHalf(long elems, int seed)
    {
        var rng = new Random(seed);
        var host = new ushort[elems];
        for (long i = 0; i < elems; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)((rng.NextSingle() - 0.5f) * 1.0f));
        return host;
    }

    private static void Upload(nint dst, ushort[] host)
    {
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(host.Length * 2)).ThrowOnError();
    }

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
