using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for the FP32 RoPE kernel. Exercises both the
/// interleaved Norm pair pattern (GPT-J) and the rotate-half NeoX pattern (Llama, Qwen,
/// Phi) via <see cref="CudaKernels.LaunchRoPEF32"/>, comparing against the CPU
/// <see cref="RoPE.Execute"/> reference.
/// </summary>
/// <remarks>
/// <para>
/// Trap-the-bug verification for the NeoX bug fixed in #36: feeding the raw C# enum
/// value <c>(int)RoPEType.NeoX = 2</c> to the kernel falls into the "anything but 1 →
/// interleaved" branch and the GPU output diverges from the CPU NeoX reference. Passing
/// the translated value via <see cref="CudaKernels.ToCudaRopeType"/> (which maps NeoX →
/// 1) makes both paths match. Empirically verified by temporarily replacing the
/// translator call with <c>(int)RoPEType.NeoX</c> in the test below — the parity
/// assertion fails with large element-wise differences. Restored before commit.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class RoPEF32ParityTests : IDisposable
{
    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    [SkippableFact]
    public unsafe void RoPEF32_Norm_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        // SmolLM-style geometry, small enough to run in <200 ms.
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 32; // must be even
        const int seqLen = 4;
        const int ropeDim = headDim;
        const int halfRope = ropeDim / 2;
        const float theta = 10000.0f;

        var rng = new Random(42);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * numHeads * headDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        // CPU reference — Norm/interleaved.
        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cosTable, sinTable);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, numHeads, numKvHeads, headDim, ropeDim,
                     cosTable, sinTable, RoPEType.Norm);

        // GPU — use the translator (Norm → 0) just as production CUDA forward paths do.
        int cudaRopeType = CudaKernels.ToCudaRopeType(RoPEType.Norm);
        Assert.Equal(0, cudaRopeType);
        float[] gpuQ = RunGpuRoPE(q, k, positions, seqLen, numHeads, numKvHeads, headDim,
                                   ropeDim, theta, cudaRopeType, out float[] gpuK);

        CudaKernelTestHarness.AssertClose("RoPEF32-Norm Q", cpuQ, gpuQ, absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
        CudaKernelTestHarness.AssertClose("RoPEF32-Norm K", cpuK, gpuK, absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
    }

    [SkippableFact]
    public unsafe void RoPEF32_NeoX_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        // Qwen2.5-ish geometry (theta=1M) — exercises the NeoX rotate-half path.
        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 32;
        const int seqLen = 4;
        const int ropeDim = headDim;
        const int halfRope = ropeDim / 2;
        const float theta = 1_000_000.0f;

        var rng = new Random(43);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * numHeads * headDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cosTable, sinTable);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, numHeads, numKvHeads, headDim, ropeDim,
                     cosTable, sinTable, RoPEType.NeoX);

        // The translator is the single chokepoint. Production code MUST call this; tests
        // that bypass it (e.g. passing (int)RoPEType.NeoX = 2) silently get Norm rotation.
        int cudaRopeType = CudaKernels.ToCudaRopeType(RoPEType.NeoX);
        Assert.Equal(1, cudaRopeType);
        float[] gpuQ = RunGpuRoPE(q, k, positions, seqLen, numHeads, numKvHeads, headDim,
                                   ropeDim, theta, cudaRopeType, out float[] gpuK);

        CudaKernelTestHarness.AssertClose("RoPEF32-NeoX Q", cpuQ, gpuQ, absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
        CudaKernelTestHarness.AssertClose("RoPEF32-NeoX K", cpuK, gpuK, absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
    }

    /// <summary>
    /// Trap-the-bug guard: explicitly assert that bypassing
    /// <see cref="CudaKernels.ToCudaRopeType"/> and casting the C# enum directly
    /// (NeoX = 2) does NOT produce a NeoX rotation. The kernel falls through to the
    /// interleaved branch (anything != 1), so the result diverges from the CPU NeoX
    /// reference by macroscopic amounts. If this test ever passes, the kernel encoding
    /// changed and the translator should be revisited.
    /// </summary>
    [SkippableFact]
    public unsafe void RoPEF32_NeoX_RawEnumCast_DivergesFromCpu()
    {
        _harness.SkipIfUnavailable();

        const int numHeads = 4;
        const int numKvHeads = 2;
        const int headDim = 32;
        const int seqLen = 4;
        const int ropeDim = headDim;
        const int halfRope = ropeDim / 2;
        const float theta = 1_000_000.0f;

        var rng = new Random(44);
        float[] q = CudaKernelTestHarness.RandomF32(rng, seqLen * numHeads * headDim);
        float[] k = CudaKernelTestHarness.RandomF32(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, theta, cosTable, sinTable);

        float[] cpuQ = (float[])q.Clone();
        float[] cpuK = (float[])k.Clone();
        RoPE.Execute(cpuQ, cpuK, positions, numHeads, numKvHeads, headDim, ropeDim,
                     cosTable, sinTable, RoPEType.NeoX);

        // BUG INJECTION: feed the raw enum value (=2) without translation.
        int buggyRopeType = (int)RoPEType.NeoX;
        Assert.Equal(2, buggyRopeType);
        float[] gpuQ = RunGpuRoPE(q, k, positions, seqLen, numHeads, numKvHeads, headDim,
                                   ropeDim, theta, buggyRopeType, out float[] gpuK);

        // We expect at least one element of Q to diverge by far more than the parity tolerance.
        // (Whole heads get the Norm rotation instead of NeoX, so max-abs-diff is O(1).)
        float maxAbsDiffQ = 0;
        for (int i = 0; i < cpuQ.Length; i++)
            maxAbsDiffQ = MathF.Max(maxAbsDiffQ, MathF.Abs(cpuQ[i] - gpuQ[i]));

        Assert.True(maxAbsDiffQ > 1e-3f,
            $"Raw (int)RoPEType.NeoX = 2 produced a Q result indistinguishable from the CPU NeoX " +
            $"reference (maxAbs={maxAbsDiffQ:E4}). The CUDA kernel's int→pair-pattern encoding " +
            $"may have changed; ToCudaRopeType requires review.");
    }

    private float[] RunGpuRoPE(float[] q, float[] k, int[] positions,
                                int seqLen, int numHeads, int numKvHeads, int headDim,
                                int ropeDim, float theta, int ropeType, out float[] gpuK)
    {
        nint devQ = _harness.Upload(q);
        nint devK = _harness.Upload(k);
        nint devPos = _harness.Upload(positions);

        _harness.Kernels.LaunchRoPEF32(devQ, devK, devPos, seqLen, numHeads, numKvHeads,
                                       headDim, ropeDim, theta, ropeType, _harness.StreamHandle);
        _harness.Synchronize();

        float[] gpuQ = _harness.DownloadFloats(devQ, q.Length);
        gpuK = _harness.DownloadFloats(devK, k.Length);
        return gpuQ;
    }
}
