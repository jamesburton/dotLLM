using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Equivalence tests for the Phase B MoE grouped quantized GEMV kernel.
/// Compares <see cref="CudaKernels.LaunchMoeGroupedGemv"/> against per-expert
/// <see cref="CudaKernels.LaunchQuantizedGemv"/> calls as the oracle. The two
/// paths must produce bit-identical outputs (same superblock body, same
/// reduction order, FP32 accumulator, single FP16 store) — we admit a tiny
/// FP16 noise tolerance since they share the same kernel body and any
/// observable drift would be a bug.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMoeGroupedGemvTests
{
    private readonly ITestOutputHelper _out;
    public CudaMoeGroupedGemvTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    /// <summary>
    /// V2-Lite-Q4_K_M-class shapes: K=hidden=2048 (8 super-blocks/row),
    /// M=intermediate=1408 (rounded up to a multiple of 16 for the fixture).
    /// K_active=4 covers the most common decode-path bucket (top-k select with
    /// a couple of inactive experts).
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 256, 256)]   // tiny synthetic — exercises kernel correctness fast
    [InlineData(4, 1408, 2048)] // V2-Lite shape
    [InlineData(6, 1408, 2048)] // V2-Lite full top-k=6
    [InlineData(2, 256, 512)]
    public void GroupedQ4K_MatchesPerCallWithinFp16Tolerance(int kActive, int M, int K)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunGroupedEquivalence(QuantizationType.Q4_K, kActive, M, K, blockBytes: 144,
            (rng, span) => SynthesiseQ4KBlock(rng, span));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunGroupedEquivalence(QuantizationType qt,
        int kActive, int M, int K, int blockBytes,
        Action<Random, Span<byte>> synthesiseBlock)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMoeGroupedGemv(qt), $"Grouped MoE GEMV for {qt} not loaded (PTX may be stale)");

        var rng = new Random(31415 ^ (int)qt ^ kActive ^ M ^ K);
        int superblocksPerRow = K / 256;
        int rowBytes = superblocksPerRow * blockBytes;
        long perExpertWeightBytes = (long)M * rowBytes;

        // Build synthetic per-expert weights — different for each expert so we
        // detect any cross-expert pointer-shuffling bug in the kernel.
        byte[][] hostWeights = new byte[kActive][];
        for (int e = 0; e < kActive; e++)
        {
            hostWeights[e] = new byte[perExpertWeightBytes];
            var span = hostWeights[e].AsSpan();
            for (int row = 0; row < M; row++)
            for (int sb = 0; sb < superblocksPerRow; sb++)
                synthesiseBlock(rng, span.Slice(row * rowBytes + sb * blockBytes, blockBytes));
        }

        // Single shared input — half of [-2.5, 2.5] roughly normal-ish.
        Half[] hostX = new Half[K];
        for (int i = 0; i < K; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        long xBytes = (long)K * sizeof(ushort);
        long yPerExpertBytes = (long)M * sizeof(ushort);

        // Allocate device buffers: K_active per-expert weights + K_active
        // per-expert grouped outputs + K_active per-expert per-call outputs +
        // input + the two ptr-array uploads.
        nint devX = 0, devPtrArrays = 0;
        nint[] devWeights = new nint[kActive];
        nint[] devYGrouped = new nint[kActive];
        nint[] devYPerCall = new nint[kActive];
        Half[][] yGrouped = new Half[kActive][];
        Half[][] yPerCall = new Half[kActive][];
        for (int e = 0; e < kActive; e++)
        {
            yGrouped[e] = new Half[M];
            yPerCall[e] = new Half[M];
        }

        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            for (int e = 0; e < kActive; e++)
            {
                CudaDriverApi.cuMemAlloc_v2(out devWeights[e], (nuint)perExpertWeightBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out devYGrouped[e], (nuint)yPerExpertBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out devYPerCall[e], (nuint)yPerExpertBytes).ThrowOnError();
                fixed (byte* pW = hostWeights[e])
                    CudaDriverApi.cuMemcpyHtoD_v2(devWeights[e], (nint)pW, (nuint)perExpertWeightBytes).ThrowOnError();
            }

            // Upload per-expert weight + output ptr arrays for the grouped call.
            // Layout: [W0..Wk-1, Y0..Yk-1] (matches the kernel's pointer layout).
            long ptrArrayBytes = 2L * kActive * sizeof(long);
            CudaDriverApi.cuMemAlloc_v2(out devPtrArrays, (nuint)ptrArrayBytes).ThrowOnError();
            long* hostPtrs = stackalloc long[2 * kActive];
            for (int e = 0; e < kActive; e++)
            {
                hostPtrs[0 * kActive + e] = (long)devWeights[e];
                hostPtrs[1 * kActive + e] = (long)devYGrouped[e];
            }
            CudaDriverApi.cuMemcpyHtoD_v2(devPtrArrays, (nint)hostPtrs, (nuint)ptrArrayBytes).ThrowOnError();

            nint weightsPtrDev = devPtrArrays;
            nint outputsPtrDev = devPtrArrays + (nint)((long)kActive * sizeof(long));

            // Path A: grouped — single launch over (M × K_active) blocks.
            kernels.LaunchMoeGroupedGemv(
                weightsPtrDev, outputsPtrDev, devX, qt, M, K, kActive, stream.Handle);

            // Path B: per-call oracle — K_active separate LaunchQuantizedGemv launches.
            for (int e = 0; e < kActive; e++)
            {
                kernels.LaunchQuantizedGemv(devWeights[e], qt,
                    devX, devYPerCall[e], M, K, stream.Handle);
            }
            stream.Synchronize();

            // Pull both paths' outputs back to host.
            for (int e = 0; e < kActive; e++)
            {
                fixed (Half* pY = yGrouped[e])
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYGrouped[e], (nuint)yPerExpertBytes).ThrowOnError();
                fixed (Half* pY = yPerCall[e])
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYPerCall[e], (nuint)yPerExpertBytes).ThrowOnError();
            }
        }
        finally
        {
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devPtrArrays != 0) CudaDriverApi.cuMemFree_v2(devPtrArrays);
            for (int e = 0; e < kActive; e++)
            {
                if (devWeights[e] != 0) CudaDriverApi.cuMemFree_v2(devWeights[e]);
                if (devYGrouped[e] != 0) CudaDriverApi.cuMemFree_v2(devYGrouped[e]);
                if (devYPerCall[e] != 0) CudaDriverApi.cuMemFree_v2(devYPerCall[e]);
            }
        }

        // Both paths share kernel body, accumulator order, and FP16 cast — they
        // should be bit-identical. We allow a 1e-3 absolute tolerance to account
        // for any compiler reordering between PTX modules; in practice peak
        // observed drift on V2-Lite shapes is 0.0.
        float maxAbs = 0f, refMax = 0f;
        int totalElems = 0;
        for (int e = 0; e < kActive; e++)
        {
            for (int i = 0; i < M; i++)
            {
                float a = (float)yPerCall[e][i];
                float b = (float)yGrouped[e][i];
                float diff = MathF.Abs(a - b);
                if (diff > maxAbs) maxAbs = diff;
                if (MathF.Abs(a) > refMax) refMax = MathF.Abs(a);
                totalElems++;
            }
        }

        _out.WriteLine($"Grouped vs per-call {qt} K_active={kActive} M={M} K={K}: " +
                        $"ref|max|={refMax:F3}  max-abs-diff={maxAbs:F6}  ({totalElems} elems)");

        Assert.True(maxAbs < 1e-3f,
            $"Grouped GEMV diverges from per-call (max-abs-diff={maxAbs}, refMax={refMax}). " +
            "Both paths share the same Q4_K body — any divergence is a bug.");
    }

    private static unsafe void SynthesiseQ4KBlock(Random rng, Span<byte> block)
    {
        // Q4_K layout (144 bytes): half d, half dmin, uint8[12] scales, uint8[128] qs.
        Half d = (Half)(rng.NextDouble() * 0.05 + 0.01);
        Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.04);

        fixed (byte* pBlock = block)
        {
            *(Half*)pBlock = d;
            *(Half*)(pBlock + 2) = dmin;
        }
        for (int i = 4; i < 144; i++)
            block[i] = (byte)rng.Next(0, 256);
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
