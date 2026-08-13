using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3ChunkBoundaryF32"/> against
/// the documented closed-form formula (<c>Mamba3Block.ApplyChunkBoundaryAdjustment</c> /
/// <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>'s boundary block):
/// <c>state[h,p,n] += vState[h,p] * (sum_r kState[r,h,n]) * coef[h]</c>. Issue #346.
/// </summary>
/// <remarks>
/// Uses a small tolerance, NOT bit-exact <c>SequenceEqual</c>. The rank-sum reduction
/// (<c>kSum += kState[...]</c>) is bit-exact between CPU and GPU — same sequential
/// accumulation order on both sides, no transcendentals involved. The final combine,
/// however, is an inherited, documented O(ulp) difference from the Vulkan original: the
/// kernel (native/kernels/mamba3_chunk_boundary_f32.cu) computes <c>(v*kSum)*coef</c>,
/// while the true CPU references (<c>Mamba3Block.ApplyChunkBoundaryAdjustment</c>,
/// <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>) compute <c>(v*coef)*kSum</c> — see
/// src/DotLLM.Vulkan/Kernels/Mamba3ChunkBoundaryF32Kernel.cs:24-29 for the same disclosure
/// on the Vulkan kernel this was ported from. Floating-point multiplication is not
/// associative, so the two orders can differ by up to 1 ULP per element. 1e-5 is far
/// above the observed worst case and still far below any real bug's O(1) signature (wrong
/// stride/index/broadcast/rank-sum order).
/// </remarks>
[Trait("Category", "GPU")]
public class CudaMamba3ChunkBoundaryF32Tests
{
    private const float Tolerance = 1e-5f;

    private readonly ITestOutputHelper _out;
    public CudaMamba3ChunkBoundaryF32Tests(ITestOutputHelper output) => _out = output;

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
    [InlineData(1, 4, 4, 8)]   // SISO: nRank=1
    [InlineData(3, 4, 4, 8)]   // MIMO: nRank=3
    [InlineData(1, 32, 64, 128)] // ib-ssm/mamba3-370M-10BT shape
    public void Mamba3ChunkBoundaryF32_MatchesClosedFormFormula(int nRank, int nHead, int headDim, int dState)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3ChunkBoundary, "mamba3_chunk_boundary_f32 PTX symbol not found (stale build)");

        var rng = new Random(0xB0DA ^ nRank ^ nHead ^ (headDim << 8) ^ (dState << 16));
        int stateLen = nHead * headDim * dState;
        int vLen = nHead * headDim;
        int kLen = nRank * nHead * dState;

        float[] stateCpu = RandomArray(rng, stateLen);
        float[] stateGpu = (float[])stateCpu.Clone();
        float[] vState = RandomArray(rng, vLen);
        float[] kState = RandomArray(rng, kLen);
        float[] coef = new float[nHead];
        for (int h = 0; h < nHead; h++)
            coef[h] = h % 3 == 0 ? 0f : (float)(rng.NextDouble() * 0.5); // exercise the coef==0 early-out too

        // CPU oracle: closed-form formula, matching Mamba3Block.ApplyChunkBoundaryAdjustment /
        // Mamba3CanonicalSsd.ExecuteMimoStreaming's boundary block exactly: (v*coef)*kSum.
        for (int h = 0; h < nHead; h++)
        {
            float c = coef[h];
            if (c == 0f) continue;
            for (int p = 0; p < headDim; p++)
            {
                float vpC = vState[h * headDim + p] * c;
                for (int n = 0; n < dState; n++)
                {
                    float kSum = 0f;
                    for (int r = 0; r < nRank; r++)
                        kSum += kState[r * nHead * dState + h * dState + n];
                    stateCpu[h * headDim * dState + p * dState + n] += vpC * kSum;
                }
            }
        }

        nint dState_ = 0, dV = 0, dK = 0, dCoef = 0;
        try
        {
            long stateBytes = (long)stateLen * sizeof(float);
            long vBytes = (long)vLen * sizeof(float);
            long kBytes = (long)kLen * sizeof(float);
            long coefBytes = (long)nHead * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCoef, (nuint)coefBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = vState) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = kState) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kBytes).ThrowOnError();
                fixed (float* p = coef) CudaDriverApi.cuMemcpyHtoD_v2(dCoef, (nint)p, (nuint)coefBytes).ThrowOnError();
            }

            kernels.LaunchMamba3ChunkBoundaryF32(dState_, dV, dK, dCoef, nHead, headDim, dState, nRank, stream.Handle);
            stream.Synchronize();

            float[] stateGpuOut = new float[stateLen];
            unsafe
            {
                fixed (float* p = stateGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            float maxDiff = MaxAbsDiff(stateCpu, stateGpuOut);
            Assert.True(maxDiff <= Tolerance, $"chunk-boundary state mismatch: maxAbsDiff={maxDiff} > {Tolerance}.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} headDim={headDim} dState={dState}: maxAbsDiff={maxDiff} (tolerance {Tolerance}).");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dCoef != 0) CudaDriverApi.cuMemFree_v2(dCoef);
        }
    }

    private static float[] RandomArray(Random rng, int len)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]);
            if (d > max) max = d;
        }
        return max;
    }
}
