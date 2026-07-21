using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the fused copy+RMSNorm kernel (<see cref="CudaKernels.LaunchCopyRmsNormF32"/>) —
/// replaces the decode-time host pattern of a separate <c>cuMemcpyDtoDAsync</c> (save the
/// pre-norm value) immediately followed by <see cref="CudaKernels.LaunchRmsNormF32"/> on the
/// same input. Asserts BOTH outputs (the residual copy and the normalized result) exactly/
/// numerically match what the two separate calls would have produced.
/// </summary>
[Trait("Category", "GPU")]
public class CudaCopyRmsNormF32Test
{
    private readonly ITestOutputHelper _out;
    public CudaCopyRmsNormF32Test(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
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

    [SkippableTheory]
    [InlineData(1, 5120)]    // decode shape: seqLen=1, real Bonsai hiddenSize
    [InlineData(5, 5120)]    // prefill shape
    [InlineData(1, 37)]      // odd n (non-multiple-of-2) — exercises the scalar tail path.
                              // rows MUST stay 1 here: row-stride float2 casts require each
                              // row's byte offset (row*n*4) to be a multiple of 8, which odd n
                              // breaks for row>0 (a pre-existing float2-cast limitation shared
                              // with rmsnorm_f32.cu, never hit in production since real hidden
                              // sizes are always even) — multi-row + odd n triggers a genuine
                              // CUDA misaligned-address fault, not a bug in this kernel's fusion.
    public void CopyRmsNormF32_MatchesSeparateCopyPlusRmsNorm(int rows, int hiddenSize)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Run(rows, hiddenSize);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(int rows, int hiddenSize)
    {
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasCopyRmsNormF32, "copy_rmsnorm_f32 not loaded (PTX may be stale)");

        const float eps = 1e-5f;
        var rng = new Random(0xC0FFEE ^ rows ^ (hiddenSize << 8));
        float[] input = new float[rows * hiddenSize];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        float[] weight = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) weight[i] = 0.5f + (float)rng.NextDouble();

        nint dIn = 0, dWeight = 0, dResSep = 0, dOutSep = 0, dResFused = 0, dOutFused = 0;
        try
        {
            long inBytes = (long)input.Length * sizeof(float);
            long wBytes = (long)hiddenSize * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dIn, (nuint)inBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dWeight, (nuint)wBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dResSep, (nuint)inBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutSep, (nuint)inBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dResFused, (nuint)inBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutFused, (nuint)inBytes).ThrowOnError();

            fixed (float* p = input) CudaDriverApi.cuMemcpyHtoD_v2(dIn, (nint)p, (nuint)inBytes).ThrowOnError();
            fixed (float* p = weight) CudaDriverApi.cuMemcpyHtoD_v2(dWeight, (nint)p, (nuint)wBytes).ThrowOnError();

            nint s = stream.Handle;

            // Reference: separate D2D copy + RmsNormF32 (the pattern this kernel replaces).
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dResSep, dIn, (nuint)inBytes, s).ThrowOnError();
            kernels.LaunchRmsNormF32(dIn, dWeight, dOutSep, hiddenSize, eps, rows, s);

            // Fused kernel under test.
            kernels.LaunchCopyRmsNormF32(dIn, dResFused, dWeight, dOutFused, hiddenSize, eps, rows, s);

            stream.Synchronize();

            float[] resSep = new float[input.Length], outSep = new float[input.Length];
            float[] resFused = new float[input.Length], outFused = new float[input.Length];
            fixed (float* p = resSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dResSep, (nuint)inBytes).ThrowOnError();
            fixed (float* p = outSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutSep, (nuint)inBytes).ThrowOnError();
            fixed (float* p = resFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dResFused, (nuint)inBytes).ThrowOnError();
            fixed (float* p = outFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutFused, (nuint)inBytes).ThrowOnError();

            Assert.Equal(resSep, resFused);   // copy is a pure bit-exact gather
            Assert.Equal(outSep, outFused);   // identical reduction math -> bit-exact
        }
        finally
        {
            if (dIn != 0) CudaDriverApi.cuMemFree_v2(dIn);
            if (dWeight != 0) CudaDriverApi.cuMemFree_v2(dWeight);
            if (dResSep != 0) CudaDriverApi.cuMemFree_v2(dResSep);
            if (dOutSep != 0) CudaDriverApi.cuMemFree_v2(dOutSep);
            if (dResFused != 0) CudaDriverApi.cuMemFree_v2(dResFused);
            if (dOutFused != 0) CudaDriverApi.cuMemFree_v2(dOutFused);
        }

        _out.WriteLine($"copy_rmsnorm_f32 rows={rows} hidden={hiddenSize}: exact match vs separate copy+rmsnorm_f32");
    }
}
