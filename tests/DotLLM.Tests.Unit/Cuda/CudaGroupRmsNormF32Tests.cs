// tests/DotLLM.Tests.Unit/Cuda/CudaGroupRmsNormF32Tests.cs
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchGroupRmsNormF32"/>
/// (native/kernels/group_rmsnorm.cu) against a per-group application of
/// <see cref="RmsNorm.Execute"/> (the CPU reference NemotronH's ForwardSsmBody step 9 uses,
/// once per group, with each group's own weight slice).
/// </summary>
[Trait("Category", "GPU")]
public class CudaGroupRmsNormF32Tests
{
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-4f;

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
    [InlineData(1, 2, 8)]     // smallest: 1 token, 2 groups
    [InlineData(4, 4, 32)]    // multi-token prefill
    [InlineData(1, 10, 8)]    // Nemotron-H-realistic: nGroup=10, groupDim=8 (dInner=80)
    public void Launch_MatchesCpuReference(int seqLen, int nGroup, int groupDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x9E3 ^ seqLen ^ (nGroup << 8) ^ (groupDim << 16));
        int dInner = nGroup * groupDim;
        const float eps = 1e-5f;

        float[] x = new float[seqLen * dInner];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        float[] weight = new float[dInner];
        for (int i = 0; i < weight.Length; i++) weight[i] = 1.0f + (float)(rng.NextDouble() * 0.1 - 0.05);

        // CPU reference: apply RmsNorm.Execute once per (t, g) over its groupDim-wide slice with
        // that group's own weight slice — exactly NemotronHTransformerModel.ForwardSsmBody step 9.
        float[] cpuOut = new float[x.Length];
        for (int t = 0; t < seqLen; t++)
        {
            for (int g = 0; g < nGroup; g++)
            {
                int off = t * dInner + g * groupDim;
                RmsNorm.Execute(
                    x.AsSpan(off, groupDim),
                    weight.AsSpan(g * groupDim, groupDim),
                    eps,
                    cpuOut.AsSpan(off, groupDim));
            }
        }

        nint dX = 0, dW = 0;
        try
        {
            long xBytes = (long)x.Length * sizeof(float);
            long wBytes = (long)weight.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dW, (nuint)wBytes).ThrowOnError();
            unsafe
            {
                fixed (float* px = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)px, (nuint)xBytes).ThrowOnError();
                fixed (float* pw = weight) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)pw, (nuint)wBytes).ThrowOnError();
            }

            kernels.LaunchGroupRmsNormF32(dX, dW, eps, seqLen, nGroup, groupDim, stream.Handle);
            stream.Synchronize();

            float[] gpuOut = new float[x.Length];
            unsafe { fixed (float* p = gpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dX, (nuint)xBytes).ThrowOnError(); }

            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                float bar = AbsTol + RelTol * MathF.Abs(cpuOut[i]);
                Assert.True(diff <= bar, $"[{i}]: cpu={cpuOut[i]:F6} vs cuda={gpuOut[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
        finally
        {
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dW != 0) CudaDriverApi.cuMemFree_v2(dW);
        }
    }
}
