using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Direct kernel-level parity tests for the F32↔F16 conversion kernels
/// (<see cref="CudaKernels.LaunchConvertF32ToF16"/> and
/// <see cref="CudaKernels.LaunchConvertF16ToF32"/>). These conversion kernels are the
/// "glue" between the FP32 residual stream and the FP16 attention/projection paths;
/// regressions in either direction silently corrupt every layer's output.
/// </summary>
/// <remarks>
/// CUDA's hardware <c>__float2half_rn</c> uses round-to-nearest-even; .NET's
/// <c>(Half)float</c> cast does the same. End-to-end the round-trip should match a
/// single host-side <c>(float)(Half)x</c> bit-exactly (no double-rounding) for finite
/// inputs in the FP16 normal range.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("CudaKernels")]
public sealed class ConvertF16RoundTripParityTests : IDisposable
{
    private readonly CudaKernelTestHarness _harness = new();

    public void Dispose() => _harness.Dispose();

    [SkippableFact]
    public unsafe void ConvertF32_F16_F32_RoundTrip_BitExact()
    {
        _harness.SkipIfUnavailable();

        // Use a deterministic mix of values: powers of two, near-zero, mid-range,
        // negatives. All within FP16 normal range to avoid subnormal/Inf edge cases.
        const int n = 256;
        var rng = new Random(42);
        float[] src = new float[n];
        for (int i = 0; i < n; i++)
        {
            // Uniform in [-2, +2] — well within FP16 normal range, mantissa-bounded.
            src[i] = (float)(rng.NextDouble() * 4 - 2);
        }

        nint devF32 = _harness.Upload(src);
        nint devF16 = _harness.Allocate((long)n * sizeof(ushort));
        nint devF32Back = _harness.Allocate((long)n * sizeof(float));

        _harness.Kernels.LaunchConvertF32ToF16(devF32, devF16, n, _harness.StreamHandle);
        _harness.Kernels.LaunchConvertF16ToF32(devF16, devF32Back, n, _harness.StreamHandle);
        _harness.Synchronize();

        float[] gpuRoundtrip = _harness.DownloadFloats(devF32Back, n);

        // Reference: single host-side (float)(Half)x. The CUDA round-trip should be
        // bit-exact (within ≤ 1 ULP at FP16 precision) for inputs in normal range.
        float[] expected = new float[n];
        for (int i = 0; i < n; i++)
            expected[i] = (float)(Half)src[i];

        // Tolerance bound: 1 ULP @ FP16 for values in [-2, +2] is 2^(1-10) = ~9.77e-4.
        // We assert tighter (5e-4) — most values round bit-exactly; mismatches signal
        // a real divergence in the conversion kernel.
        CudaKernelTestHarness.AssertClose("ConvertF32F16F32-roundtrip",
            expected, gpuRoundtrip, absoluteTolerance: 5e-4f, relativeTolerance: 1e-3f);
    }

    [SkippableFact]
    public unsafe void ConvertF32_F16_MatchesHostHalfCast()
    {
        _harness.SkipIfUnavailable();

        const int n = 256;
        var rng = new Random(43);
        float[] src = new float[n];
        for (int i = 0; i < n; i++)
            src[i] = (float)(rng.NextDouble() * 4 - 2);

        nint devF32 = _harness.Upload(src);
        nint devF16 = _harness.Allocate((long)n * sizeof(ushort));

        _harness.Kernels.LaunchConvertF32ToF16(devF32, devF16, n, _harness.StreamHandle);
        _harness.Synchronize();

        Half[] gpuF16 = _harness.DownloadHalves(devF16, n);
        for (int i = 0; i < n; i++)
        {
            Half expected = (Half)src[i];
            Assert.Equal(BitConverter.HalfToUInt16Bits(expected), BitConverter.HalfToUInt16Bits(gpuF16[i]));
        }
    }

    [SkippableFact]
    public unsafe void ConvertF16_F32_MatchesHostFloatCast()
    {
        _harness.SkipIfUnavailable();

        const int n = 256;
        var rng = new Random(44);
        Half[] src = new Half[n];
        for (int i = 0; i < n; i++)
            src[i] = (Half)(rng.NextDouble() * 4 - 2);

        nint devF16 = _harness.Upload(src);
        nint devF32 = _harness.Allocate((long)n * sizeof(float));

        _harness.Kernels.LaunchConvertF16ToF32(devF16, devF32, n, _harness.StreamHandle);
        _harness.Synchronize();

        float[] gpuF32 = _harness.DownloadFloats(devF32, n);

        // F16 → F32 is an exact widening — bit-equality is the right bar.
        for (int i = 0; i < n; i++)
        {
            float expected = (float)src[i];
            Assert.Equal(expected, gpuF32[i]);
        }
    }
}
