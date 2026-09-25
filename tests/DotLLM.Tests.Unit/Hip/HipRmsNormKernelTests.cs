using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using DotLLM.Hip.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Scaffold test for the HIP RMS-norm kernel. Skips cleanly when ROCm / AMD
/// GPU is not present. Mirrors the CUDA RmsNormF32 comparison test.
/// </summary>
[Trait("Category", "GPU")]
public sealed class HipRmsNormKernelTests : IDisposable
{
    private const int HiddenSize = 1024;
    private const float RmsEps = 1e-5f;

    private readonly ITestOutputHelper _output;
    private readonly HipContext? _ctx;
    private readonly HipStream? _stream;
    private readonly RmsNormKernel? _rmsNorm;
    private readonly bool _available;

    private readonly string? _initError;

    public HipRmsNormKernelTests(ITestOutputHelper output)
    {
        _output = output;
        if (!HipDevice.IsAvailable()) return;

        try
        {
            _ctx = HipContext.Create(0);
            _stream = HipStream.Create();

            string? coPath = FindCoFile("rmsnorm.co");
            if (coPath != null)
                _rmsNorm = new RmsNormKernel(coPath);

            _available = _rmsNorm != null;
        }
        catch (HipException ex)
        {
            // On some Windows ROCm + iGPU combos (e.g. gfx1151 Strix Halo on a
            // fresh 7.x driver), hipModuleLoadData returns hipErrorSharedObjectInitFailed
            // even for a valid code object — runtime init for the compute queue fails.
            // Skip rather than fail: the scaffold is still correct; the runtime is
            // the blocker.
            _initError = $"HIP initialization failed ({ex.ErrorCode}): {ex.Message}";
            _output.WriteLine(_initError);
        }
    }

    [SkippableFact]
    public unsafe void RmsNormF32_MatchesCpuReference()
    {
        SkipIfUnavailable();

        int n = HiddenSize;
        uint rows = 1;
        var rng = new Random(42);

        float[] input = RandomF32(rng, n);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        // CPU reference
        float[] cpuResult = new float[n];
        RmsNorm.Execute(input, weight, RmsEps, cpuResult);

        // GPU
        nint s = _stream!.Handle;
        long inputBytes = (long)n * sizeof(float);

        nint devInput = HipDevice.Allocate(inputBytes);
        nint devWeight = HipDevice.Allocate(inputBytes);
        nint devOutput = HipDevice.Allocate(inputBytes);

        try
        {
            fixed (float* pIn = input)
                HipDevice.Upload(devInput, (nint)pIn, inputBytes);
            fixed (float* pW = weight)
                HipDevice.Upload(devWeight, (nint)pW, inputBytes);

            _rmsNorm!.Launch(devInput, devWeight, devOutput, n, RmsEps, rows, s);
            _stream!.Synchronize();

            float[] gpuResult = new float[n];
            fixed (float* pOut = gpuResult)
                HipDevice.Download((nint)pOut, devOutput, inputBytes);

            CompareResults("RmsNormF32", cpuResult, gpuResult, tolerance: 5e-4f);
        }
        finally
        {
            HipDevice.Free(devInput);
            HipDevice.Free(devWeight);
            HipDevice.Free(devOutput);
        }
    }

    private void SkipIfUnavailable()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No AMD GPU / ROCm runtime available");
        Skip.IfNot(_initError is null, _initError ?? "HIP init error");
        Skip.If(!_available, "HIP code object (rmsnorm.co) not found");
    }

    private static string? FindCoFile(string fileName)
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "co", fileName),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "hip", "co", fileName),
        };

        foreach (var c in candidates)
        {
            var full = Path.GetFullPath(c);
            if (File.Exists(full))
                return full;
        }
        return null;
    }

    private static float[] RandomF32(Random rng, int count, float scale = 1.0f)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return arr;
    }

    private void CompareResults(string name, float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        float maxDiff = 0, sumDiff = 0;
        int maxIdx = 0, mismatchCount = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            sumDiff += diff;
            if (diff > maxDiff) { maxDiff = diff; maxIdx = i; }
            if (diff > tolerance) mismatchCount++;
        }

        float meanDiff = sumDiff / expected.Length;

        _output.WriteLine($"[{name}] n={expected.Length}  maxDiff={maxDiff:E4} @idx={maxIdx}  " +
                          $"meanDiff={meanDiff:E4}  mismatches(>{tolerance})={mismatchCount}/{expected.Length}");

        Assert.True(mismatchCount == 0,
            $"[{name}] {mismatchCount}/{expected.Length} elements exceed tolerance {tolerance}. " +
            $"maxDiff={maxDiff:E4} at index {maxIdx}, meanDiff={meanDiff:E4}");
    }

    public void Dispose()
    {
        _rmsNorm?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
