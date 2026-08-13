using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3SsdScanMimoF32"/> against
/// its CPU oracle, <see cref="Mamba3CanonicalSsd.ExecuteMimo"/>. Issue #346.
/// </summary>
/// <remarks>
/// Uses a small tolerance, NOT bit-exact <c>SequenceEqual</c>, even though the kernel is
/// compiled -fmad=false (native/kernels/mamba3_ssd_scan_mimo_f32.cu's own header documents
/// this). <c>decay = expf(adt[...])</c> feeds the state recurrence
/// (<c>newState = decay * state + vp * (kSum * scl)</c>) at every token, and the optional
/// gate's <c>expf(-zGated)</c> is a second transcendental site — CUDA's precise libdevice
/// <c>expf</c> is not guaranteed bit-identical to .NET's <c>MathF.Exp</c> for the same input
/// (IEEE 754 does not mandate correctly-rounded transcendentals) — the same class of drift
/// documented on <c>CudaMamba3SsdScanSisoF32Tests</c> and <see cref="CudaKernels"/>'s GDN
/// kernel block. Because this kernel folds the ENTIRE sequence into one launch with a
/// sequential per-token state update, any sub-ULP drift introduced at t=0 compounds through
/// every later token's state and y. Measured on this box (RTX 3060): worst observed
/// maxAbsDiff across both shape/gate combinations was 2.38e-7 y / 1.19e-7 state (the
/// nRank=2/hasZ=false case) — over 400x smaller than the 1e-4 tolerance below, and still
/// several orders of magnitude tighter than a real transcription bug (wrong
/// stride/index/broadcast/accumulator-order/rank-sum-omission) would produce, which shows
/// up as O(0.1+). Per-case values are also reported live via <see cref="ITestOutputHelper"/>
/// in the test output.
/// </remarks>
[Trait("Category", "GPU")]
public class CudaMamba3SsdScanMimoF32Tests
{
    private const float Tolerance = 1e-4f;

    private readonly ITestOutputHelper _out;
    public CudaMamba3SsdScanMimoF32Tests(ITestOutputHelper output) => _out = output;

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
    [InlineData(3, 4, 4, 8, 5, true)]   // tiny MIMO shape, rank=3, with gate
    [InlineData(2, 4, 4, 8, 5, false)]  // tiny MIMO shape, rank=2, no gate
    public void Mamba3SsdScanMimoF32_MatchesCpuReference(
        int nRank, int nHead, int headDim, int dState, int seqLen, bool hasZ)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3SsdScanMimo, "mamba3_ssd_scan_mimo_f32 PTX symbol not found (stale build)");

        var rng = new Random(0x3310 ^ nRank ^ nHead ^ (headDim << 4) ^ (dState << 12) ^ (seqLen << 20));

        int stateLen = nHead * headDim * dState;
        int vLen = seqLen * nHead * headDim;
        int bcLen = seqLen * nRank * nHead * dState;
        int hdrLen = seqLen * nHead;
        int mimoLen = nHead * nRank * headDim;

        float[] stateCpu = RandomArray(rng, stateLen);
        float[] stateGpu = (float[])stateCpu.Clone();
        float[] v = RandomArray(rng, vLen);
        float[] qRoped = RandomArray(rng, bcLen);
        float[] kRoped = RandomArray(rng, bcLen);
        float[] qkPreDotSum = RandomArray(rng, hdrLen);
        float[] scale = RandomArray(rng, hdrLen);
        float[] gamma = RandomArray(rng, hdrLen);
        float[] adt = new float[hdrLen];
        for (int i = 0; i < hdrLen; i++) adt[i] = -(float)rng.NextDouble() * 2f; // A*DT <= 0 (decay)
        float[] d = RandomArray(rng, nHead);
        float[] z = hasZ ? RandomArray(rng, vLen) : Array.Empty<float>();
        float[] mimoZ = RandomArray(rng, mimoLen);
        float[] mimoO = RandomArray(rng, mimoLen);
        float[] yCpu = new float[vLen];

        Mamba3CanonicalSsd.ExecuteMimo(
            stateCpu, v, qRoped, kRoped, qkPreDotSum, scale, gamma, adt, d, z, mimoZ, mimoO,
            yCpu, yPerRank: Span<float>.Empty, seqLen, nRank, nHead, headDim, dState);

        nint dSt = 0, dV = 0, dQ = 0, dK = 0, dQkp = 0, dScl = 0, dGm = 0, dAdt = 0, dD = 0, dZ = 0, dMz = 0, dMo = 0, dY = 0;
        try
        {
            long stateBytes = (long)stateLen * sizeof(float);
            long vBytes = (long)vLen * sizeof(float);
            long bcBytes = (long)bcLen * sizeof(float);
            long hdrBytes = (long)hdrLen * sizeof(float);
            long dBytes = (long)nHead * sizeof(float);
            long mimoBytes = (long)mimoLen * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dSt, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQkp, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dScl, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGm, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAdt, (nuint)hdrBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)dBytes).ThrowOnError();
            if (hasZ) CudaDriverApi.cuMemAlloc_v2(out dZ, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dMz, (nuint)mimoBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dMo, (nuint)mimoBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)vBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyHtoD_v2(dSt, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = qRoped) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = kRoped) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = qkPreDotSum) CudaDriverApi.cuMemcpyHtoD_v2(dQkp, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = scale) CudaDriverApi.cuMemcpyHtoD_v2(dScl, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = gamma) CudaDriverApi.cuMemcpyHtoD_v2(dGm, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = adt) CudaDriverApi.cuMemcpyHtoD_v2(dAdt, (nint)p, (nuint)hdrBytes).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)dBytes).ThrowOnError();
                if (hasZ) fixed (float* p = z) CudaDriverApi.cuMemcpyHtoD_v2(dZ, (nint)p, (nuint)vBytes).ThrowOnError();
                fixed (float* p = mimoZ) CudaDriverApi.cuMemcpyHtoD_v2(dMz, (nint)p, (nuint)mimoBytes).ThrowOnError();
                fixed (float* p = mimoO) CudaDriverApi.cuMemcpyHtoD_v2(dMo, (nint)p, (nuint)mimoBytes).ThrowOnError();
            }

            kernels.LaunchMamba3SsdScanMimoF32(dSt, dV, dQ, dK, dQkp, dScl, dGm, dAdt, dD, dZ, dMz, dMo, dY,
                seqLen, nRank, nHead, headDim, dState, hasZ, stream.Handle);
            stream.Synchronize();

            float[] yGpu = new float[vLen];
            float[] stateGpuOut = new float[stateLen];
            unsafe
            {
                fixed (float* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)vBytes).ThrowOnError();
                fixed (float* p = stateGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dSt, (nuint)stateBytes).ThrowOnError();
            }

            float yMaxDiff = MaxAbsDiff(yCpu, yGpu);
            float stateMaxDiff = MaxAbsDiff(stateCpu, stateGpuOut);

            Assert.True(yMaxDiff <= Tolerance, $"y output mismatch: maxAbsDiff={yMaxDiff} > {Tolerance}.");
            Assert.True(stateMaxDiff <= Tolerance, $"final ssm_state mismatch: maxAbsDiff={stateMaxDiff} > {Tolerance}.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} headDim={headDim} dState={dState} seqLen={seqLen} hasZ={hasZ}: " +
                $"maxAbsDiff y={yMaxDiff} state={stateMaxDiff} (tolerance {Tolerance}).");
        }
        finally
        {
            if (dSt != 0) CudaDriverApi.cuMemFree_v2(dSt);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dQkp != 0) CudaDriverApi.cuMemFree_v2(dQkp);
            if (dScl != 0) CudaDriverApi.cuMemFree_v2(dScl);
            if (dGm != 0) CudaDriverApi.cuMemFree_v2(dGm);
            if (dAdt != 0) CudaDriverApi.cuMemFree_v2(dAdt);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dZ != 0) CudaDriverApi.cuMemFree_v2(dZ);
            if (dMz != 0) CudaDriverApi.cuMemFree_v2(dMz);
            if (dMo != 0) CudaDriverApi.cuMemFree_v2(dMo);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
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
            float diff = MathF.Abs(expected[i] - actual[i]);
            if (diff > max) max = diff;
        }
        return max;
    }
}
