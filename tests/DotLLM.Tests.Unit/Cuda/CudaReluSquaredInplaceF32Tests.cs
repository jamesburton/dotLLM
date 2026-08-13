using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>Correctness anchor for <see cref="CudaKernels.LaunchReluSquaredInplaceF32"/>
/// against <see cref="ReluSquared.Execute"/>.</summary>
[Trait("Category", "GPU")]
public class CudaReluSquaredInplaceF32Tests
{
    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(37)]
    [InlineData(4096)]
    public void Launch_MatchesCpuReference(int n)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x517 ^ n);
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4.0 - 2.0); // includes negatives

        float[] cpuOut = (float[])x.Clone();
        ReluSquared.Execute(cpuOut, cpuOut);

        nint dX = 0;
        try
        {
            long bytes = (long)n * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)bytes).ThrowOnError();
            unsafe { fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)bytes).ThrowOnError(); }

            kernels.LaunchReluSquaredInplaceF32(dX, n, stream.Handle);
            stream.Synchronize();

            float[] gpuOut = new float[n];
            unsafe { fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dX, (nuint)bytes).ThrowOnError(); }

            for (int i = 0; i < n; i++)
                Assert.Equal(cpuOut[i], gpuOut[i], precision: 5);
        }
        finally
        {
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
        }
    }
}
