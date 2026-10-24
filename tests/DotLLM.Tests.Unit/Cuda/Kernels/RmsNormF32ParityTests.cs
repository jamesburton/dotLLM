using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for FP32 RMS normalization
/// (<see cref="CudaKernels.LaunchRmsNormF32"/>) against the CPU
/// <see cref="RmsNorm.Execute"/> reference. RMS normalization sits in front of every
/// transformer block; a regression here breaks every layer of every model.
/// </summary>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class RmsNormF32ParityTests : IDisposable
{
    private const float Epsilon = 1e-5f;

    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    [SkippableFact]
    public unsafe void RmsNormF32_SingleRow_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        const int hidden = 64;
        const int rows = 1;
        var rng = new Random(42);

        float[] input = CudaKernelTestHarness.RandomF32(rng, hidden);
        float[] weight = CudaKernelTestHarness.RandomF32(rng, hidden);

        float[] cpuOut = new float[hidden];
        RmsNorm.Execute(input, weight, Epsilon, cpuOut);

        float[] gpuOut = RunGpu(input, weight, hidden, rows);

        // RmsNorm is one division + multiply-broadcast — FMA reorder is the only diff.
        CudaKernelTestHarness.AssertClose("RmsNormF32-1row", cpuOut, gpuOut,
                                          absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
    }

    [SkippableFact]
    public unsafe void RmsNormF32_MultiRow_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        const int hidden = 64;
        const int rows = 4;
        var rng = new Random(43);

        float[] input = CudaKernelTestHarness.RandomF32(rng, hidden * rows);
        float[] weight = CudaKernelTestHarness.RandomF32(rng, hidden);

        float[] cpuOut = new float[hidden * rows];
        for (int r = 0; r < rows; r++)
            RmsNorm.Execute(input.AsSpan(r * hidden, hidden), weight, Epsilon,
                            cpuOut.AsSpan(r * hidden, hidden));

        float[] gpuOut = RunGpu(input, weight, hidden, rows);

        CudaKernelTestHarness.AssertClose("RmsNormF32-4row", cpuOut, gpuOut,
                                          absoluteTolerance: 1e-5f, relativeTolerance: 1e-4f);
    }

    private float[] RunGpu(float[] input, float[] weight, int hidden, int rows)
    {
        nint devIn = _harness.Upload(input);
        nint devW = _harness.Upload(weight);
        nint devOut = _harness.Allocate((long)input.Length * sizeof(float));

        _harness.Kernels.LaunchRmsNormF32(devIn, devW, devOut, hidden, Epsilon, rows, _harness.StreamHandle);
        _harness.Synchronize();

        return _harness.DownloadFloats(devOut, input.Length);
    }
}
