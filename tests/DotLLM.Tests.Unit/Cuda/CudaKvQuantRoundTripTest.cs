using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #956: KV-format round trip (<c>LaunchQuantKv</c> then <c>LaunchKvDequantToF16</c>) on RANDOM
/// data with a per-element error bound. The pre-existing KV tests feed smooth monotone ramps, which a
/// within-block element permutation barely disturbs -- exactly how the KV-Q4_0 (adjacent-pair)
/// versus weight-Q4_0 (low/high half) layout mismatch in the eager dequant went unnoticed. On
/// random data a permuted unpack errs by ~the data range, far above the quantization step.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaKvQuantRoundTripTest : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    public CudaKvQuantRoundTripTest()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null) _kernels = new CudaKernels(ptxDir);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, 127.0)]
    [InlineData(KvCacheDType.Q4_0, 7.0)]
    public void QuantThenKvDequant_RandomData_WithinHalfQuantStep(KvCacheDType dtype, double levels)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");

        const int rows = 6;
        const int kvStride = 1024;          // real Bonsai-2 shape: 4 KV heads x 256
        int n = rows * kvStride;
        var rng = new Random(956);
        var host = new float[n];
        var half = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            host[i] = (float)((rng.NextDouble() * 2 - 1) * 3.0);
            half[i] = BitConverter.HalfToUInt16Bits((Half)host[i]);
            host[i] = (float)BitConverter.UInt16BitsToHalf(half[i]);
        }

        long qBytes = dtype == KvCacheDType.Q8_0 ? n / 32 * 34L : n / 32 * 18L;
        CudaDriverApi.cuMemAlloc_v2(out nint srcDev, (nuint)(n * 2)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint qDev, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dstDev, (nuint)(n * 2)).ThrowOnError();
        try
        {
            fixed (ushort* p = half)
                CudaDriverApi.cuMemcpyHtoD_v2(srcDev, (nint)p, (nuint)(n * 2)).ThrowOnError();

            _kernels!.LaunchQuantKv(srcDev, qDev, n, dtype, _stream!.Handle);
            _kernels.LaunchKvDequantToF16(qDev, dtype, dstDev, n, _stream.Handle);
            _stream.Synchronize();

            var back = new ushort[n];
            fixed (ushort* p = back)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dstDev, (nuint)(n * 2)).ThrowOnError();

            for (int b = 0; b < n / 32; b++)
            {
                float maxAbs = 0;
                for (int j = 0; j < 32; j++) maxAbs = Math.Max(maxAbs, Math.Abs(host[b * 32 + j]));
                double step = maxAbs / levels;
                // half a step + fp16 storage of scale/value rounding slack
                double bound = step * 0.5 + maxAbs * 2e-3 + 1e-4;
                for (int j = 0; j < 32; j++)
                {
                    int i = b * 32 + j;
                    double got = (double)BitConverter.UInt16BitsToHalf(back[i]);
                    Assert.True(Math.Abs(got - host[i]) <= bound,
                        $"{dtype} block {b} elem {j}: expected {host[i]}, got {got}, bound {bound} " +
                        "(a within-block permutation shows up here as an error ~ data range).");
                }
            }

            if (dtype == KvCacheDType.Q4_0)
            {
                // Negative control: the GGUF weight dequant unpacks Q4_0 as low/high HALVES, so on KV
                // (adjacent-pair) data it must violate the same bound. If this ever passes, the two
                // layouts were unified and this test's premise (and the KV dequant kernel) needs revisiting.
                _kernels.LaunchDequantToF16(qDev, QuantizationType.Q4_0, dstDev, n, _stream.Handle);
                _stream.Synchronize();
                var wrong = new ushort[n];
                fixed (ushort* p = wrong)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dstDev, (nuint)(n * 2)).ThrowOnError();
                int violations = 0;
                for (int b = 0; b < n / 32; b++)
                {
                    float maxAbs = 0;
                    for (int j = 0; j < 32; j++) maxAbs = Math.Max(maxAbs, Math.Abs(host[b * 32 + j]));
                    double bound = maxAbs / levels * 0.5 + maxAbs * 2e-3 + 1e-4;
                    for (int j = 0; j < 32; j++)
                        if (Math.Abs((double)BitConverter.UInt16BitsToHalf(wrong[b * 32 + j]) - host[b * 32 + j]) > bound)
                            violations++;
                }
                Assert.True(violations > n / 4,
                    $"negative control: generic weight dequant should misread KV Q4_0 ({violations}/{n} elements off)");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(srcDev);
            CudaDriverApi.cuMemFree_v2(qDev);
            CudaDriverApi.cuMemFree_v2(dstDev);
        }
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
