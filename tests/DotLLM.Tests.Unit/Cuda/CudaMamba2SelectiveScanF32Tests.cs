// tests/DotLLM.Tests.Unit/Cuda/CudaMamba2SelectiveScanF32Tests.cs
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchMamba2SelectiveScanF32"/>
/// (native/kernels/mamba2_selective_scan.cu) against its CPU oracle,
/// <see cref="Mamba2SelectiveScan.Execute"/>. Mirrors
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanMamba2SelectiveScanF32KernelTests"/>'s shapes and
/// tolerance (abs 1e-3 / rel 1e-3 — softplus + exp + the inner k-loop recurrence accumulate F32
/// rounding faster than pointwise kernels; the per-thread loop order matches the CPU's, but
/// exp/log emission can shift the last bits across iterations) and
/// <see cref="CudaGdnScanStepF32Tests"/>'s CUDA device-buffer idiom.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMamba2SelectiveScanF32Tests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _out;
    public CudaMamba2SelectiveScanF32Tests(ITestOutputHelper output) => _out = output;

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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(2, 4, 8, 1, 1)]        // smallest decode shape
    [InlineData(4, 8, 16, 2, 1)]       // groups, single token
    [InlineData(4, 8, 16, 2, 4)]       // groups, multi-token prefill
    [InlineData(10, 80, 128, 10, 1)]   // Nemotron-H-realistic decode
    [InlineData(10, 80, 128, 10, 8)]   // Nemotron-H-realistic prefill
    public void Launch_MatchesCpuReference(int nHead, int headDim, int dState, int nGroup, int seqLen)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0x4A31 ^ (nHead * 131) ^ (headDim * 71) ^ (dState * 53) ^ (nGroup * 23) ^ seqLen);
        int dInner = nHead * headDim;

        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dtRaw = SmallRandom(rng, seqLen * nHead);
        float[] dtBias = SmallRandom(rng, nHead);
        float[] a = NegativeRandom(rng, nHead);
        float[] d = SmallRandom(rng, nHead);
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        // CPU reference: pre-add dtBias (the CUDA kernel does this internally), guarded softplus
        // is inside Mamba2SelectiveScan itself; add D-skip manually after, matching
        // NemotronHTransformerModel.ForwardSsmBody steps 6-7 exactly.
        float[] dtBiased = new float[seqLen * nHead];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < nHead; h++)
                dtBiased[t * nHead + h] = dtRaw[t * nHead + h] + dtBias[h];

        float[] stateCpu = (float[])state0.Clone();
        float[] yCpu = new float[seqLen * dInner];
        Mamba2SelectiveScan.Execute(stateCpu, x, dtBiased, a, b, c, yCpu, nHead, headDim, dState, nGroup, seqLen);
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < nHead; h++)
                for (int i = 0; i < headDim; i++)
                    yCpu[t * dInner + h * headDim + i] += x[t * dInner + h * headDim + i] * d[h];

        nint dState_ = 0, dX = 0, dDt = 0, dDtb = 0, dA = 0, dD = 0, dB = 0, dC = 0, dY = 0;
        try
        {
            long stateBytes = (long)state0.Length * sizeof(float);
            long xBytes = (long)x.Length * sizeof(float);
            long dtBytes = (long)dtRaw.Length * sizeof(float);
            long headBytes = (long)nHead * sizeof(float);
            long bcBytes = (long)b.Length * sizeof(float);
            long yBytes = (long)yCpu.Length * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)dtBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDtb, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)headBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)bcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)yBytes).ThrowOnError();

            unsafe
            {
                float[] state0Copy = (float[])state0.Clone();
                fixed (float* p = state0Copy) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)xBytes).ThrowOnError();
                fixed (float* p = dtRaw) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)dtBytes).ThrowOnError();
                fixed (float* p = dtBias) CudaDriverApi.cuMemcpyHtoD_v2(dDtb, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = a) CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)headBytes).ThrowOnError();
                fixed (float* p = b) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = c) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)bcBytes).ThrowOnError();
            }

            kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                nHead, headDim, dState, nGroup, seqLen, stream.Handle);
            stream.Synchronize();

            float[] yGpu = new float[yCpu.Length];
            float[] stateGpu = new float[stateCpu.Length];
            unsafe
            {
                fixed (float* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)yBytes).ThrowOnError();
                fixed (float* p = stateGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            for (int i = 0; i < yCpu.Length; i++)
            {
                float diff = MathF.Abs(yCpu[i] - yGpu[i]);
                float bar = AbsTol + RelTol * MathF.Abs(yCpu[i]);
                Assert.True(diff <= bar, $"y[{i}]: cpu={yCpu[i]:F6} vs cuda={yGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
            for (int i = 0; i < stateCpu.Length; i++)
            {
                float diff = MathF.Abs(stateCpu[i] - stateGpu[i]);
                float bar = AbsTol + RelTol * MathF.Abs(stateCpu[i]);
                Assert.True(diff <= bar, $"state[{i}]: cpu={stateCpu[i]:F6} vs cuda={stateGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
            _out.WriteLine($"nHead={nHead} headDim={headDim} dState={dState} nGroup={nGroup} seqLen={seqLen}: within tolerance.");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dDtb != 0) CudaDriverApi.cuMemFree_v2(dDtb);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    /// <summary>Splitting a seqLen=8 scan into two seqLen=4 calls on the same state buffer must
    /// match a single seqLen=8 call — the property the decode loop relies on. Ports
    /// <c>VulkanMamba2SelectiveScanF32KernelTests.Launch_StatePersistsAcrossCalls</c>.</summary>
    [SkippableFact]
    public void Launch_StatePersistsAcrossCalls()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        const int nHead = 4, headDim = 8, dState = 16, nGroup = 2, seqLen = 8;
        int dInner = nHead * headDim;

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(unchecked((int)0xBEEFCAFE));
        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dtRaw = SmallRandom(rng, seqLen * nHead);
        float[] dtBias = SmallRandom(rng, nHead);
        float[] a = NegativeRandom(rng, nHead);
        float[] d = SmallRandom(rng, nHead);
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        long stateBytes = (long)state0.Length * sizeof(float);
        nint dState_ = 0, dX = 0, dDt = 0, dDtb = 0, dA = 0, dD = 0, dB = 0, dC = 0, dY = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dX, (nuint)((long)x.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)((long)dtRaw.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDtb, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dD, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)((long)b.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dC, (nuint)((long)c.Length * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dY, (nuint)((long)seqLen * dInner * sizeof(float))).ThrowOnError();

            unsafe
            {
                fixed (float* p = dtBias) CudaDriverApi.cuMemcpyHtoD_v2(dDtb, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
                fixed (float* p = a) CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
                fixed (float* p = d) CudaDriverApi.cuMemcpyHtoD_v2(dD, (nint)p, (nuint)((long)nHead * sizeof(float))).ThrowOnError();
            }

            // 1. One-shot seqLen=8.
            unsafe
            {
                float[] s0 = (float[])state0.Clone();
                fixed (float* p = s0) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
                fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)p, (nuint)((long)x.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = dtRaw) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)((long)dtRaw.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = b) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)((long)b.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = c) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)((long)c.Length * sizeof(float))).ThrowOnError();
            }
            kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                nHead, headDim, dState, nGroup, seqLen, stream.Handle);
            stream.Synchronize();
            float[] yOneShot = new float[seqLen * dInner];
            float[] stateOneShot = new float[state0.Length];
            unsafe
            {
                fixed (float* p = yOneShot) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)((long)yOneShot.Length * sizeof(float))).ThrowOnError();
                fixed (float* p = stateOneShot) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            // 2. Two seqLen=4 calls on the same state buffer.
            unsafe
            {
                float[] s0 = (float[])state0.Clone();
                fixed (float* p = s0) CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
            }
            RunHalf(kernels, stream, dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                x, dtRaw, b, c, 0, 4, nHead, headDim, dState, nGroup, dInner, out float[] yFirstHalf);
            RunHalf(kernels, stream, dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
                x, dtRaw, b, c, 4, 4, nHead, headDim, dState, nGroup, dInner, out float[] ySecondHalf);
            float[] stateSplit = new float[state0.Length];
            unsafe
            {
                fixed (float* p = stateSplit) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
            }

            for (int i = 0; i < 4 * dInner; i++) Assert.Equal(yOneShot[i], yFirstHalf[i]);
            for (int i = 0; i < 4 * dInner; i++) Assert.Equal(yOneShot[4 * dInner + i], ySecondHalf[i]);
            for (int i = 0; i < state0.Length; i++) Assert.Equal(stateOneShot[i], stateSplit[i]);
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dX != 0) CudaDriverApi.cuMemFree_v2(dX);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dDtb != 0) CudaDriverApi.cuMemFree_v2(dDtb);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dD != 0) CudaDriverApi.cuMemFree_v2(dD);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
            if (dC != 0) CudaDriverApi.cuMemFree_v2(dC);
            if (dY != 0) CudaDriverApi.cuMemFree_v2(dY);
        }
    }

    private static unsafe void RunHalf(CudaKernels kernels, CudaStream stream,
        nint dState_, nint dX, nint dDt, nint dDtb, nint dA, nint dD, nint dB, nint dC, nint dY,
        float[] x, float[] dtRaw, float[] b, float[] c,
        int tokenOffset, int halfLen, int nHead, int headDim, int dState, int nGroup, int dInner,
        out float[] yHalf)
    {
        float[] xSlice = x.AsSpan(tokenOffset * dInner, halfLen * dInner).ToArray();
        float[] dtSlice = dtRaw.AsSpan(tokenOffset * nHead, halfLen * nHead).ToArray();
        float[] bSlice = b.AsSpan(tokenOffset * nGroup * dState, halfLen * nGroup * dState).ToArray();
        float[] cSlice = c.AsSpan(tokenOffset * nGroup * dState, halfLen * nGroup * dState).ToArray();

        fixed (float* px = xSlice) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)px, (nuint)((long)xSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pdt = dtSlice) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)pdt, (nuint)((long)dtSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pb = bSlice) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)pb, (nuint)((long)bSlice.Length * sizeof(float))).ThrowOnError();
        fixed (float* pc = cSlice) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)pc, (nuint)((long)cSlice.Length * sizeof(float))).ThrowOnError();

        kernels.LaunchMamba2SelectiveScanF32(dState_, dX, dDt, dDtb, dA, dD, dB, dC, dY,
            nHead, headDim, dState, nGroup, halfLen, stream.Handle);
        stream.Synchronize();

        yHalf = new float[halfLen * dInner];
        fixed (float* py = yHalf) CudaDriverApi.cuMemcpyDtoH_v2((nint)py, dY, (nuint)((long)yHalf.Length * sizeof(float))).ThrowOnError();
    }

    private static float[] SmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return arr;
    }

    private static float[] NegativeRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(-(rng.NextDouble() * 0.45 + 0.05));
        return arr;
    }
}
