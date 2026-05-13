using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for FP32 SwiGLU
/// (<see cref="CudaKernels.LaunchSwiGLUF32"/>) against the CPU
/// <see cref="FusedOps.SwiGLU"/> reference. Both compute
/// <c>out[i] = gate[i] * sigmoid(gate[i]) * up[i]</c> elementwise — the only divergence
/// path is sigmoid implementation. CPU uses <see cref="System.Numerics.Tensors.TensorPrimitives.Sigmoid"/>;
/// CUDA uses <c>__expf</c>-derived sigmoid. Tolerance below is sized for this.
/// </summary>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class SwiGLUF32ParityTests : IDisposable
{
    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    [SkippableFact]
    public unsafe void SwiGLUF32_MatchesCpuReference()
    {
        _harness.SkipIfUnavailable();

        const int n = 64;       // intermediate width per token
        const int seqLen = 4;
        var rng = new Random(42);

        float[] gate = CudaKernelTestHarness.RandomF32(rng, n * seqLen);
        float[] up = CudaKernelTestHarness.RandomF32(rng, n * seqLen);

        float[] cpuOut = new float[n * seqLen];
        FusedOps.SwiGLU(gate, up, cpuOut);

        float[] gpuOut = RunGpu(gate, up, n, seqLen);

        // Sigmoid polynomial / __expf differences leave room for ~1e-4; bit-equal would
        // require both sides to use identical exp approximations.
        CudaKernelTestHarness.AssertClose("SwiGLUF32", cpuOut, gpuOut,
                                          absoluteTolerance: 1e-4f, relativeTolerance: 1e-3f);
    }

    private float[] RunGpu(float[] gate, float[] up, int n, int seqLen)
    {
        nint devGate = _harness.Upload(gate);
        nint devUp = _harness.Upload(up);
        nint devOut = _harness.Allocate((long)gate.Length * sizeof(float));

        _harness.Kernels.LaunchSwiGLUF32(devGate, devUp, devOut, n, seqLen, _harness.StreamHandle);
        _harness.Synchronize();

        return _harness.DownloadFloats(devOut, gate.Length);
    }
}
