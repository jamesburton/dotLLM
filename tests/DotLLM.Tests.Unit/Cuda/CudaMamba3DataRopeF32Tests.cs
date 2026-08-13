using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba3DataRopeF32"/> (the
/// <c>mamba3_data_rope_f32</c> CUDA kernel, native/kernels/mamba3_data_rope_f32.cu)
/// against its CPU oracle, <see cref="Mamba3DataRoPE.ExecuteCanonical"/>. Issue #346.
/// </summary>
/// <remarks>
/// Uses a small ULP-scale tolerance, NOT bit-exact <c>SequenceEqual</c>, despite the kernel
/// being compiled -fmad=false for CPU parity. Measured: the <c>cum</c> accumulator (plain
/// mul+add, no transcendentals) IS bit-exact between CPU and GPU, confirming -fmad=false and
/// the accumulation order match. But <c>cosf</c>/<c>sinf</c>/<c>tanhf</c> of that
/// bit-identical value differ from .NET's <c>MathF.Cos</c>/<c>Sin</c>/<c>Tanh</c> by up to 1
/// ULP (observed max abs diff 5.4e-7 on values ~0.1-0.9, i.e. ~1-2 ULP of float32) — IEEE 754
/// does not mandate correctly-rounded transcendentals, so CUDA's precise device library and
/// .NET's MathF are not guaranteed bit-identical even for the same input. This is the same
/// class of issue documented on <c>CudaKernels</c>'s GDN kernel block ("CUDA's precise expf is
/// not guaranteed bit-identical to MathF.Exp"). 1e-5 is ~20x the observed worst case and still
/// far below any real bug's signature (wrong stride/index/mode would show as O(1) errors, not
/// O(1e-7)). NOTE: with a pathological seed, a cum value landing within 1 ULP of a 2π boundary
/// could floor to opposite sides on CPU vs GPU, producing an ~6.28 absolute (but angularly
/// equivalent) cum divergence — not hit by this test's fixed seed, not engineered around here.
/// </remarks>
[Trait("Category", "GPU")]
public class CudaMamba3DataRopeF32Tests
{
    private const float Tolerance = 1e-5f;

    private readonly ITestOutputHelper _out;
    public CudaMamba3DataRopeF32Tests(ITestOutputHelper output) => _out = output;

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
    [InlineData(1, 4, 8, 2, 3)]     // SISO-shaped: nRank=1, mode=Pairwise
    [InlineData(1, 32, 128, 32, 5)] // ib-ssm/mamba3-370M-10BT shape (nHead=32, dState=128, numRopeAngles=32)
    public void Mamba3DataRopeF32_MatchesCpuReference_Pairwise(
        int nRank, int nHead, int dState, int numRopeAngles, int seqLen)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3DataRope, "mamba3_data_rope_f32 PTX symbol not found (stale build)");

        var rng = new Random(0xA3CE ^ nHead ^ (dState << 8) ^ (seqLen << 16));
        int bcLen = seqLen * nRank * nHead * dState;
        int dtLen = seqLen * nHead;
        int angLen = seqLen * numRopeAngles;
        int cumLen = nHead * numRopeAngles;

        float[] bCpu = RandomArray(rng, bcLen), cCpu = RandomArray(rng, bcLen);
        float[] bGpu = (float[])bCpu.Clone(), cGpu = (float[])cCpu.Clone();
        float[] anglesRaw = RandomArray(rng, angLen);
        float[] dt = new float[dtLen];
        for (int i = 0; i < dtLen; i++) dt[i] = (float)rng.NextDouble() * 0.1f; // dt > 0, post-softplus range

        float[] cumOutCpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            bCpu, cCpu, anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty, cumAngleOut: cumOutCpu,
            seqLen, nRank, nHead, dState, numRopeAngles, Mamba3RoPEMode.Pairwise);

        nint dB = 0, dC = 0, dAng = 0, dDt = 0, dCumPrev = 0, dCumOut = 0;
        try
        {
            long bcBytes = (long)bcLen * sizeof(float);
            long dtBytes = (long)dtLen * sizeof(float);
            long angBytes = (long)angLen * sizeof(float);
            long cumBytes = (long)cumLen * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAng, (nuint)angBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)dtBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCumPrev, (nuint)Math.Max(cumBytes, 4)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dCumOut, (nuint)cumBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = bGpu) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cGpu) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = anglesRaw) CudaDriverApi.cuMemcpyHtoD_v2(dAng, (nint)p, (nuint)angBytes).ThrowOnError();
                fixed (float* p = dt) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)dtBytes).ThrowOnError();
            }

            kernels.LaunchMamba3DataRopeF32(dB, dC, dAng, dDt, dCumPrev, dCumOut,
                seqLen, nRank, nHead, dState, numRopeAngles, mode: 0,
                hasCumPrev: false, writeCumOut: true, stream.Handle);
            stream.Synchronize();

            float[] bGpuOut = new float[bcLen], cGpuOut = new float[bcLen], cumOutGpu = new float[cumLen];
            unsafe
            {
                fixed (float* p = bGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cGpuOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dC, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cumOutGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dCumOut, (nuint)cumBytes).ThrowOnError();
            }

            float bMaxDiff = MaxAbsDiff(bCpu, bGpuOut);
            float cMaxDiff = MaxAbsDiff(cCpu, cGpuOut);
            float cumMaxDiff = MaxAbsDiff(cumOutCpu, cumOutGpu);

            Assert.True(bMaxDiff <= Tolerance, $"B rotation mismatch: maxAbsDiff={bMaxDiff} > {Tolerance}.");
            Assert.True(cMaxDiff <= Tolerance, $"C rotation mismatch: maxAbsDiff={cMaxDiff} > {Tolerance}.");
            Assert.True(cumMaxDiff <= Tolerance, $"cum_angle output mismatch: maxAbsDiff={cumMaxDiff} > {Tolerance}.");
            _out.WriteLine($"nRank={nRank} nHead={nHead} dState={dState} numRopeAngles={numRopeAngles} seqLen={seqLen}: " +
                $"maxAbsDiff B={bMaxDiff} C={cMaxDiff} cum={cumMaxDiff} (tolerance {Tolerance}).");
        }
        finally
        {
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dAng != 0) CudaDriverApi.cuMemFree_v2(dAng);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dCumPrev != 0) CudaDriverApi.cuMemFree_v2(dCumPrev);
            if (dCumOut != 0) CudaDriverApi.cuMemFree_v2(dCumOut);
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
