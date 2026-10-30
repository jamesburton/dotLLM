using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Diagnostic microbench (#1172), opt-in with <c>DOTLLM_Q4_1_BENCH=1</c>: the in-place Q4_1 kernels against the CUDA
/// kernels that already exist for neighbouring types at Flint's shapes. Decode rows report weight GB/s (weight bytes per
/// launch / time); prefill rows report effective TFLOP/s and the scratch each path needs. Timed with CUDA events over
/// back-to-back launches. Weight contents are random bytes: timing only, outputs are not checked here.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQ4_1KernelBench
{
    private readonly ITestOutputHelper _out;

    public CudaQ4_1KernelBench(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void Q4_1_Vs_Existing_Kernels()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_Q4_1_BENCH") == "1", "set DOTLLM_Q4_1_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        using var lfm = new DotLLM.Cuda.Architectures.CudaLfm2Kernels(ptx);
        using var cublas = CudaCublasHandle.Create();
        nint s = stream.Handle;
        var allocs = new List<nint>();
        nint Alloc(long bytes)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(d, 0x11, (nuint)bytes).ThrowOnError();   // finite halfs everywhere
            allocs.Add(d);
            return d;
        }

        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        double TimeUs(Action launch, int reps)
        {
            for (int i = 0; i < 3; i++) launch();
            CudaDriverApi.cuEventRecord(e0, s).ThrowOnError();
            for (int i = 0; i < reps; i++) launch();
            CudaDriverApi.cuEventRecord(e1, s).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
            return ms * 1000.0 / reps;
        }

        try
        {
            nint xh = Alloc(512L * 8192 * 2), xf = Alloc(512L * 8192 * 4), yh = Alloc(512L * 8192 * 2), yf = Alloc(512L * 8192 * 4);
            nint q8 = Alloc(512L * 8192 / 32 * 36);

            _out.WriteLine("decode GEMV (n x k)            | kernel                     |   us   | weight GB/s");
            foreach (var (n, k, label) in new[]
                     {
                         (2048, 2048, "attn q/o"), (512, 2048, "expert gate/up"), (2048, 512, "expert down"),
                         (6144, 2048, "dense gate/up"), (2048, 6144, "dense down"), (150144, 2048, "LM head"),
                     })
            {
                nint w41 = Alloc((long)n * k / 32 * 20), w40 = Alloc((long)n * k / 32 * 18);
                nint w80 = Alloc((long)n * k / 32 * 34), w4k = Alloc((long)n * k / 256 * 144);
                Row($"{n}x{k} {label}", "Q4_1 f16 (new)", (long)n * k / 32 * 20,
                    TimeUs(() => kernels.LaunchQuantizedGemv(w41, QuantizationType.Q4_1, xh, yh, n, k, s), 200));
                Row($"{n}x{k} {label}", "Q4_1 f32 (new, MoE)", (long)n * k / 32 * 20,
                    TimeUs(() => kernels.LaunchQuantizedGemmQ4_1F32(w41, xf, yf, n, k, 1, s), 200));
                Row($"{n}x{k} {label}", "Q4_0 lfm2_gemv_q4_0_f32", (long)n * k / 32 * 18,
                    TimeUs(() => lfm.GemvQ4(w40, xf, yf, n, k, s), 200));
                Row($"{n}x{k} {label}", "Q8_0 quantized_gemv_q8_0", (long)n * k / 32 * 34,
                    TimeUs(() => kernels.LaunchQuantizedGemv(w80, QuantizationType.Q8_0, xh, yh, n, k, s), 200));
                if (kernels.HasMmq(QuantizationType.Q4_K))
                    Row($"{n}x{k} {label}", "Q4_K mmq (production)", (long)n * k / 256 * 144,
                        TimeUs(() => kernels.LaunchQuantizedGemvMmq(w4k, QuantizationType.Q4_K, xh, yh, n, k, s), 200));
            }

            // MoE grouped decode: top-6 experts of 512 x 2048 (gate or up).
            {
                const int kAct = 6, m = 512, k = 2048;
                long ptrBytes = 2L * kAct * sizeof(long);
                foreach (var (qt, blockBytes, per) in new[] { (QuantizationType.Q4_1, 20, 32), (QuantizationType.Q4_K, 144, 256), (QuantizationType.Q8_0, 34, 32) })
                {
                    if (!kernels.HasMoeGroupedGemv(qt)) continue;
                    long wBytes = (long)m * k / per * blockBytes;
                    var ptrs = new long[2 * kAct];
                    for (int e = 0; e < kAct; e++) { ptrs[e] = Alloc(wBytes); ptrs[kAct + e] = Alloc(m * 2); }
                    nint dPtrs = Alloc(ptrBytes);
                    fixed (long* p = ptrs) CudaDriverApi.cuMemcpyHtoD_v2(dPtrs, (nint)p, (nuint)ptrBytes).ThrowOnError();
                    Row($"grouped 6x{m}x{k}", $"{qt} moe_grouped_gemv", kAct * wBytes,
                        TimeUs(() => kernels.LaunchMoeGroupedGemv(dPtrs, dPtrs + kAct * sizeof(long), xh, qt, m, k, kAct, s), 200));
                }
            }

            _out.WriteLine("");
            _out.WriteLine("prefill (n x k, m tokens)     | kernel                         |   us   | TFLOP/s | scratch bytes");
            foreach (var (n, k, m) in new[] { (2048, 2048, 512), (6144, 2048, 512), (512, 2048, 64) })
            {
                nint w41 = Alloc((long)n * k / 32 * 20), w40 = Alloc((long)n * k / 32 * 18), deq = Alloc((long)n * k * 2);
                double flop = 2.0 * n * k * m;
                void P(string kernel, double us, long scratch)
                    => _out.WriteLine($"{n}x{k} m={m,-4}                 | {kernel,-30} | {us,6:F1} | {flop / us / 1e6,7:F2} | {scratch}");
                P("Q4_1 gemm f16 (new, in place)", TimeUs(() => kernels.LaunchQuantizedGemmQ4_1F16(w41, xh, yh, n, k, m, s), 20), 0);
                P("Q4_1 gemm f32 (new, in place)", TimeUs(() => kernels.LaunchQuantizedGemmQ4_1F32(w41, xf, yf, n, k, m, s), 20), 0);
                P("Q4_1 dequant + cuBLAS HGEMM", TimeUs(() =>
                {
                    kernels.LaunchDequantToF16(w41, QuantizationType.Q4_1, deq, n * k, s);
                    CudaGemm.LinearF16(cublas.Handle, xh, deq, yh, m, k, n, s);
                }, 20), (long)n * k * 2);
                P("Q4_0 lfm2 q8_1 quant + gemm", TimeUs(() =>
                {
                    lfm.QuantizeQ8(xf, q8, m * k / 32, s);
                    lfm.GemmQ4Q8(w40, q8, yf, n, k, m, s);
                }, 20), (long)m * k / 32 * 36);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy_v2(e0);
            CudaDriverApi.cuEventDestroy_v2(e1);
            foreach (nint d in allocs) CudaDriverApi.cuMemFree_v2(d);
        }
    }

    private void Row(string shape, string kernel, long weightBytes, double us)
        => _out.WriteLine($"{shape,-30} | {kernel,-26} | {us,6:F1} | {weightBytes / us / 1e3,8:F1}");
}
