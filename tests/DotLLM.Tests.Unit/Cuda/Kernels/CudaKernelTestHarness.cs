using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda.Kernels;

/// <summary>
/// Shared CUDA driver + kernel + stream setup for direct kernel parity tests.
/// Provides type-safe <c>Upload</c> / <c>Download</c> / device-allocation helpers so each
/// per-kernel parity test file stays focused on the algorithmic comparison logic.
/// </summary>
/// <remarks>
/// <para>
/// One instance per test class (xUnit creates a new test class instance per test, so context
/// and kernels are created and disposed per test, matching the pattern of existing
/// <c>CudaKernelComparisonTests</c>). Cheap relative to the per-test cuMemAlloc/cuMemcpy
/// traffic the kernels themselves drive.
/// </para>
/// <para>
/// All device pointers handed out by <see cref="Allocate"/> are tracked and freed on
/// <see cref="Dispose"/>, so individual tests don't have to interleave their assertions
/// with try/finally cleanup.
/// </para>
/// </remarks>
internal sealed unsafe class CudaKernelTestHarness : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly List<nint> _allocations = new();
    private bool _disposed;

    /// <summary>
    /// Constructs the harness. When CUDA is not available or PTX files cannot be located,
    /// <see cref="IsAvailable"/> reports <c>false</c> and tests should skip via
    /// <see cref="SkipIfUnavailable"/>.
    /// </summary>
    public CudaKernelTestHarness()
    {
        if (!CudaDevice.IsAvailable()) return;

        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();

        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

    /// <summary>True when CUDA driver, device, and PTX kernels are all available.</summary>
    public bool IsAvailable => _kernels != null && _stream != null && _ctx != null;

    /// <summary>Accessor for the loaded kernels module. Throws if unavailable — callers must skip first.</summary>
    public CudaKernels Kernels => _kernels ?? throw new InvalidOperationException("CUDA kernels unavailable; test should have been skipped.");

    /// <summary>Stream handle for kernel launches. Throws if unavailable.</summary>
    public nint StreamHandle => _stream?.Handle ?? throw new InvalidOperationException("CUDA stream unavailable.");

    /// <summary>Blocks until all queued kernels on the stream have completed.</summary>
    public void Synchronize() => _stream!.Synchronize();

    /// <summary>Skips the calling test when CUDA or PTX is missing. Standard preamble for every parity test.</summary>
    public void SkipIfUnavailable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
    }

    /// <summary>
    /// Searches the standard locations for the compiled PTX kernels — first next to the
    /// test assembly, then at the repo-relative <c>native/ptx</c> directory. Returns
    /// <c>null</c> when no directory containing <c>*.ptx</c> can be found.
    /// </summary>
    private static string? FindPtxDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        ];

        foreach (string dir in candidates)
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    // ──────────────────── Device allocation ────────────────────

    /// <summary>Allocates a tracked block of device memory. Freed automatically on <see cref="Dispose"/>.</summary>
    /// <param name="bytes">Size in bytes.</param>
    /// <returns>Device pointer.</returns>
    public nint Allocate(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)bytes).ThrowOnError();
        _allocations.Add(dev);
        return dev;
    }

    // ──────────────────── Upload helpers ────────────────────

    /// <summary>Allocates device memory for <paramref name="src"/>, copies it host→device, returns the device pointer.</summary>
    public nint Upload(ReadOnlySpan<float> src)
    {
        long bytes = (long)src.Length * sizeof(float);
        nint dev = Allocate(bytes);
        fixed (float* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    /// <summary>Allocates device memory for <paramref name="src"/>, copies it host→device, returns the device pointer.</summary>
    public nint Upload(ReadOnlySpan<int> src)
    {
        long bytes = (long)src.Length * sizeof(int);
        nint dev = Allocate(bytes);
        fixed (int* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    /// <summary>Allocates device memory for <paramref name="src"/>, copies it host→device, returns the device pointer.</summary>
    public nint Upload(ReadOnlySpan<byte> src)
    {
        long bytes = src.Length;
        nint dev = Allocate(bytes);
        fixed (byte* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    /// <summary>Allocates device memory for <paramref name="src"/>, copies it host→device, returns the device pointer.</summary>
    public nint Upload(ReadOnlySpan<Half> src)
    {
        long bytes = (long)src.Length * sizeof(ushort);
        nint dev = Allocate(bytes);
        fixed (Half* p = src)
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)p, (nuint)bytes).ThrowOnError();
        return dev;
    }

    // ──────────────────── Download helpers ────────────────────

    /// <summary>Downloads <paramref name="count"/> floats from <paramref name="devPtr"/>.</summary>
    public float[] DownloadFloats(nint devPtr, int count)
    {
        float[] dst = new float[count];
        long bytes = (long)count * sizeof(float);
        fixed (float* p = dst)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devPtr, (nuint)bytes).ThrowOnError();
        return dst;
    }

    /// <summary>Downloads <paramref name="count"/> halves from <paramref name="devPtr"/>.</summary>
    public Half[] DownloadHalves(nint devPtr, int count)
    {
        Half[] dst = new Half[count];
        long bytes = (long)count * sizeof(ushort);
        fixed (Half* p = dst)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devPtr, (nuint)bytes).ThrowOnError();
        return dst;
    }

    // ──────────────────── Random data generation ────────────────────

    /// <summary>Generates a fixed-seed uniform random <c>float[]</c> in <c>[-scale, +scale]</c>.</summary>
    public static float[] RandomF32(Random rng, int count, float scale = 1.0f)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return arr;
    }

    /// <summary>Generates a fixed-seed Gaussian (Box–Muller) <c>Half[]</c>.</summary>
    public static Half[] RandomF16(Random rng, int count, float scale = 1.0f)
    {
        Half[] arr = new Half[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            arr[i] = (Half)(g * scale);
        }
        return arr;
    }

    // ──────────────────── Tolerance comparison ────────────────────

    /// <summary>
    /// Compares CPU reference and GPU output arrays. Asserts every element is within
    /// <paramref name="absoluteTolerance"/> OR within <paramref name="relativeTolerance"/>
    /// of the expected value — whichever is larger. Mixed-tolerance form handles both
    /// near-zero values (absolute matters) and large values (relative matters), which
    /// pure-absolute tolerance fails for on accumulated dot products.
    /// </summary>
    /// <param name="name">Label for diagnostic messages.</param>
    /// <param name="expected">CPU reference.</param>
    /// <param name="actual">GPU output.</param>
    /// <param name="absoluteTolerance">Absolute tolerance (max acceptable abs(a-b)).</param>
    /// <param name="relativeTolerance">Relative tolerance (max acceptable abs(a-b)/abs(a)).</param>
    public static void AssertClose(string name, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
                                    float absoluteTolerance, float relativeTolerance)
    {
        Assert.Equal(expected.Length, actual.Length);

        float maxDiff = 0, maxRelDiff = 0;
        int maxIdx = 0;
        int mismatchCount = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float relDiff = diff / MathF.Max(MathF.Abs(expected[i]), 1e-30f);

            if (diff > maxDiff)
            {
                maxDiff = diff;
                maxIdx = i;
            }
            if (relDiff > maxRelDiff) maxRelDiff = relDiff;

            // Pass if either absolute OR relative tolerance is satisfied.
            if (diff > absoluteTolerance && relDiff > relativeTolerance)
                mismatchCount++;
        }

        Assert.True(mismatchCount == 0,
            $"[{name}] {mismatchCount}/{expected.Length} elements exceed tolerance " +
            $"(abs={absoluteTolerance}, rel={relativeTolerance}). " +
            $"maxAbs={maxDiff:E4} @idx={maxIdx} (expected={expected[maxIdx]:F6}, actual={actual[maxIdx]:F6}), " +
            $"maxRel={maxRelDiff:E4}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Free in reverse order so allocations that may reference one another (paranoia —
        // these are independent cuMemAlloc blocks) come down cleanly.
        for (int i = _allocations.Count - 1; i >= 0; i--)
        {
            try { CudaDriverApi.cuMemFree_v2(_allocations[i]); }
            catch { /* swallow during disposal */ }
        }
        _allocations.Clear();

        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
