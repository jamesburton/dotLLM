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
/// being compiled -fmad=false for CPU parity. Measured: for a tiny shape the <c>cum</c>
/// accumulator happened to come out bit-exact between CPU and GPU, which confirms
/// -fmad=false and the accumulation order match (the recurrence itself is
/// <c>cum += dt * tanh(raw)*π</c>, so it DOES involve a transcendental — <c>tanh</c> — not
/// "plain mul+add"). At larger shapes/seqLen, cum itself also drifts by up to ~1 ULP
/// (observed max abs diff 4.8e-7) because <c>tanhf</c>'s CUDA-vs-.NET implementations
/// aren't guaranteed to round identically even for identical inputs, and the sequential
/// cumsum lets that drift compound slightly across tokens. On top of that, <c>cosf</c>/
/// <c>sinf</c> of the (possibly already 1-ULP-off) cum value add their own up-to-1-ULP
/// rounding difference for the b/c rotation outputs. IEEE 754 does not mandate
/// correctly-rounded transcendentals, so CUDA's precise device library and .NET's MathF are
/// not guaranteed bit-identical even for the same input. This is the same class of issue
/// documented on <c>CudaKernels</c>'s GDN kernel block ("CUDA's precise expf is not
/// guaranteed bit-identical to MathF.Exp"). 1e-5 is ~20x the worst observed diff and still
/// far below any real bug's signature (wrong stride/index/mode/broadcast would show as O(1)
/// errors, not O(1e-7)). NOTE: with a pathological seed, a cum value landing within 1 ULP of
/// a 2π boundary could floor to opposite sides on CPU vs GPU, producing an ~6.28 absolute
/// (but angularly equivalent) cum divergence — not hit by this test's fixed seeds, not
/// engineered around here.
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
    [InlineData(1, 4, 8, 2, 3, false)]     // SISO-shaped: nRank=1, mode=Pairwise
    [InlineData(1, 32, 128, 32, 5, false)] // ib-ssm/mamba3-370M-10BT shape (nHead=32, dState=128, numRopeAngles=32)
    [InlineData(2, 4, 8, 2, 3, false)]     // nRank=2 (MIMO rank slices), mode=Pairwise
    [InlineData(1, 4, 16, 4, 3, true)]     // mode=Halved, nRank=1
    [InlineData(2, 4, 16, 4, 3, true)]     // mode=Halved AND nRank=2 combined
    public void Mamba3DataRopeF32_MatchesCpuReference(
        int nRank, int nHead, int dState, int numRopeAngles, int seqLen, bool halved)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3DataRope, "mamba3_data_rope_f32 PTX symbol not found (stale build)");

        var mode = halved ? Mamba3RoPEMode.Halved : Mamba3RoPEMode.Pairwise;
        int modeArg = halved ? 1 : 0;

        var rng = new Random(0xA3CE ^ nHead ^ (dState << 8) ^ (seqLen << 16) ^ (halved ? 0x5555 : 0) ^ (nRank << 24));
        int bcLen = seqLen * nRank * nHead * dState;
        int dtLen = seqLen * nHead;
        int angLen = seqLen * numRopeAngles;
        int cumLen = nHead * numRopeAngles;

        float[] bCpu = RandomArray(rng, bcLen), cCpu = RandomArray(rng, bcLen);
        float[] bIn = (float[])bCpu.Clone(), cIn = (float[])cCpu.Clone();
        float[] anglesRaw = RandomArray(rng, angLen);
        float[] dt = new float[dtLen];
        for (int i = 0; i < dtLen; i++) dt[i] = (float)rng.NextDouble() * 0.1f; // dt > 0, post-softplus range

        float[] cumOutCpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            bCpu, cCpu, anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty, cumAngleOut: cumOutCpu,
            seqLen, nRank, nHead, dState, numRopeAngles, mode);

        var (bGpuOut, cGpuOut, cumOutGpu) = RunGpu(
            kernels, stream, bIn, cIn, anglesRaw, dt, cumPrevIn: null,
            seqLen, nRank, nHead, dState, numRopeAngles, modeArg,
            hasCumPrev: false, writeCumOut: true, cumLen);

        float bMaxDiff = MaxAbsDiff(bCpu, bGpuOut);
        float cMaxDiff = MaxAbsDiff(cCpu, cGpuOut);
        float cumMaxDiff = MaxAbsDiff(cumOutCpu, cumOutGpu);

        Assert.True(bMaxDiff <= Tolerance, $"B rotation mismatch: maxAbsDiff={bMaxDiff} > {Tolerance}.");
        Assert.True(cMaxDiff <= Tolerance, $"C rotation mismatch: maxAbsDiff={cMaxDiff} > {Tolerance}.");
        Assert.True(cumMaxDiff <= Tolerance, $"cum_angle output mismatch: maxAbsDiff={cumMaxDiff} > {Tolerance}.");
        _out.WriteLine($"nRank={nRank} nHead={nHead} dState={dState} numRopeAngles={numRopeAngles} seqLen={seqLen} " +
            $"mode={mode}: maxAbsDiff B={bMaxDiff} C={cMaxDiff} cum={cumMaxDiff} (tolerance {Tolerance}).");
    }

    /// <summary>
    /// Exercises the seeded-continuity path (<c>hasCumPrev=true</c>) — the common decode
    /// case per <see cref="CudaKernels.LaunchMamba3DataRopeF32"/>'s own doc ("decode
    /// continuity needs it every call"). Runs two sequential chunks on BOTH the GPU and the
    /// CPU oracle — chunk 1 seeded from zero, chunk 2 seeded from chunk 1's real (non-zero)
    /// <c>cumOut</c> — and compares chunk 2's outputs, so the test actually reads a non-zero
    /// <c>cumPrev</c> buffer on the GPU side rather than only ever exercising the
    /// seed-from-zero path covered by <see cref="Mamba3DataRopeF32_MatchesCpuReference"/>.
    /// </summary>
    [SkippableFact]
    public void Mamba3DataRopeF32_SeededContinuity_MatchesTwoChunkCpuReference()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMamba3DataRope, "mamba3_data_rope_f32 PTX symbol not found (stale build)");

        const int nRank = 1, nHead = 4, dState = 16, numRopeAngles = 4;
        const int chunkLen = 4; // two chunks of 4 tokens each ("decode-style" chunking)
        const Mamba3RoPEMode mode = Mamba3RoPEMode.Pairwise;
        const int modeArg = 0;

        int bcChunkLen = chunkLen * nRank * nHead * dState;
        int dtChunkLen = chunkLen * nHead;
        int angChunkLen = chunkLen * numRopeAngles;
        int cumLen = nHead * numRopeAngles;

        var rng = new Random(0x0C0FFEE1);
        // Keep the pristine (pre-rotation) originals — ExecuteCanonical mutates its b/c spans
        // in place, so the GPU inputs must be cloned from these BEFORE the CPU oracle runs,
        // not from the (by-then-rotated) CPU arrays afterwards.
        float[] b1Orig = RandomArray(rng, bcChunkLen), c1Orig = RandomArray(rng, bcChunkLen);
        float[] b2Orig = RandomArray(rng, bcChunkLen), c2Orig = RandomArray(rng, bcChunkLen);
        float[] ang1 = RandomArray(rng, angChunkLen), ang2 = RandomArray(rng, angChunkLen);
        float[] dt1 = new float[dtChunkLen], dt2 = new float[dtChunkLen];
        for (int i = 0; i < dtChunkLen; i++) { dt1[i] = (float)rng.NextDouble() * 0.1f; dt2[i] = (float)rng.NextDouble() * 0.1f; }

        // --- CPU oracle: chunk 1 seeded from zero, chunk 2 seeded from chunk 1's cumOut. ---
        // Operates on clones so b1Orig/b2Orig/etc. stay pristine for the GPU run below.
        float[] b1Cpu = (float[])b1Orig.Clone(), c1Cpu = (float[])c1Orig.Clone();
        float[] cumAfterChunk1Cpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            b1Cpu, c1Cpu, ang1, dt1,
            cumAnglePrev: ReadOnlySpan<float>.Empty, cumAngleOut: cumAfterChunk1Cpu,
            chunkLen, nRank, nHead, dState, numRopeAngles, mode);
        float[] b2Cpu = (float[])b2Orig.Clone(), c2Cpu = (float[])c2Orig.Clone();
        float[] cumAfterChunk2Cpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            b2Cpu, c2Cpu, ang2, dt2,
            cumAnglePrev: cumAfterChunk1Cpu, cumAngleOut: cumAfterChunk2Cpu,
            chunkLen, nRank, nHead, dState, numRopeAngles, mode);

        // --- GPU: same two chunks (from the pristine originals), threading GPU chunk 1's
        // real cumOut into chunk 2's cumPrev. ---
        float[] b1In = (float[])b1Orig.Clone(), c1In = (float[])c1Orig.Clone();
        var (b1GpuOut, c1GpuOut, cumAfterChunk1Gpu) = RunGpu(
            kernels, stream, b1In, c1In, ang1, dt1, cumPrevIn: null,
            chunkLen, nRank, nHead, dState, numRopeAngles, modeArg,
            hasCumPrev: false, writeCumOut: true, cumLen);

        // Sanity: chunk 1 alone must already match the CPU chunk-1 reference (same tolerance
        // rationale as the class remarks) before trusting it as chunk 2's seed.
        float chunk1BDiff = MaxAbsDiff(b1Cpu, b1GpuOut);
        float chunk1CumDiff = MaxAbsDiff(cumAfterChunk1Cpu, cumAfterChunk1Gpu);
        Assert.True(chunk1BDiff <= Tolerance, $"Chunk1 B mismatch: maxAbsDiff={chunk1BDiff} > {Tolerance}.");
        Assert.True(chunk1CumDiff <= Tolerance, $"Chunk1 cum mismatch: maxAbsDiff={chunk1CumDiff} > {Tolerance}.");

        float[] b2In = (float[])b2Orig.Clone(), c2In = (float[])c2Orig.Clone();
        var (b2GpuOut, c2GpuOut, cumAfterChunk2Gpu) = RunGpu(
            kernels, stream, b2In, c2In, ang2, dt2, cumPrevIn: cumAfterChunk1Gpu,
            chunkLen, nRank, nHead, dState, numRopeAngles, modeArg,
            hasCumPrev: true, writeCumOut: true, cumLen);

        float bMaxDiff = MaxAbsDiff(b2Cpu, b2GpuOut);
        float cMaxDiff = MaxAbsDiff(c2Cpu, c2GpuOut);
        float cumMaxDiff = MaxAbsDiff(cumAfterChunk2Cpu, cumAfterChunk2Gpu);

        Assert.True(bMaxDiff <= Tolerance, $"Chunk2 B mismatch: maxAbsDiff={bMaxDiff} > {Tolerance}.");
        Assert.True(cMaxDiff <= Tolerance, $"Chunk2 C mismatch: maxAbsDiff={cMaxDiff} > {Tolerance}.");
        Assert.True(cumMaxDiff <= Tolerance, $"Chunk2 cum mismatch: maxAbsDiff={cumMaxDiff} > {Tolerance}.");
        _out.WriteLine($"Seeded continuity: chunk1 maxAbsDiff B={chunk1BDiff} cum={chunk1CumDiff}; " +
            $"chunk2 maxAbsDiff B={bMaxDiff} C={cMaxDiff} cum={cumMaxDiff} (tolerance {Tolerance}).");
    }

    /// <summary>
    /// Uploads inputs, launches <see cref="CudaKernels.LaunchMamba3DataRopeF32"/>, downloads
    /// results, and frees all device buffers. <paramref name="cumPrevIn"/> may be null (used
    /// only as a placeholder buffer when <paramref name="hasCumPrev"/> is false).
    /// </summary>
    private static (float[] b, float[] c, float[] cum) RunGpu(
        CudaKernels kernels, CudaStream stream,
        float[] bIn, float[] cIn, float[] anglesRaw, float[] dt, float[]? cumPrevIn,
        int seqLen, int nRank, int nHead, int dState, int numRopeAngles, int mode,
        bool hasCumPrev, bool writeCumOut, int cumLen)
    {
        int bcLen = seqLen * nRank * nHead * dState;
        int dtLen = seqLen * nHead;
        int angLen = seqLen * numRopeAngles;

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
                fixed (float* p = bIn) CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cIn) CudaDriverApi.cuMemcpyHtoD_v2(dC, (nint)p, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = anglesRaw) CudaDriverApi.cuMemcpyHtoD_v2(dAng, (nint)p, (nuint)angBytes).ThrowOnError();
                fixed (float* p = dt) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)dtBytes).ThrowOnError();
                if (hasCumPrev && cumPrevIn != null)
                {
                    fixed (float* p = cumPrevIn)
                        CudaDriverApi.cuMemcpyHtoD_v2(dCumPrev, (nint)p, (nuint)cumBytes).ThrowOnError();
                }
            }

            kernels.LaunchMamba3DataRopeF32(dB, dC, dAng, dDt, dCumPrev, dCumOut,
                seqLen, nRank, nHead, dState, numRopeAngles, mode,
                hasCumPrev, writeCumOut, stream.Handle);
            stream.Synchronize();

            float[] bOut = new float[bcLen], cOut = new float[bcLen], cumOut = new float[cumLen];
            unsafe
            {
                fixed (float* p = bOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dB, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dC, (nuint)bcBytes).ThrowOnError();
                fixed (float* p = cumOut) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dCumOut, (nuint)cumBytes).ThrowOnError();
            }
            return (bOut, cOut, cumOut);
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
