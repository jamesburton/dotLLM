using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #486: the device argmax (<c>argmax_f32</c>) must return exactly what the host decoder's
/// <see cref="TensorPrimitives.IndexOfMax{T}(ReadOnlySpan{T})"/> returns on the same row — including
/// ties (lowest index wins), NaN (first NaN wins) and <c>+0</c> vs <c>-0</c>. The host contract itself
/// is pinned by <c>MtpSpeculativeDecoderTests.HostArgMax_Contract_IsPinned</c>.
/// </summary>
/// <remarks>
/// Ties are placed where a reduction bug would show: two indices owned by the same thread (1024
/// apart), by adjacent lanes, across a warp boundary (31/32), across the two reduction stages
/// (different warps), and at index 0 / the last index. Skips when the PTX has not been generated.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaArgMaxF32Tests
{
    private const int Vocab = 248320;   // Bonsai 2 / Qwen3.5 vocabulary

    public static TheoryData<string> Cases() => new()
    {
        "random", "tie-same-thread", "tie-adjacent-lanes", "tie-warp-boundary", "tie-across-warps",
        "tie-first-last", "max-at-last", "all-equal", "pos-neg-zero", "neg-pos-zero", "nan-late",
        "two-nans", "all-neg-inf", "short-1", "short-100",
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void DeviceArgMax_MatchesTensorPrimitivesIndexOfMax(string name)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, CudaArgMaxF32.PtxFileName)),
            $"{CudaArgMaxF32.PtxFileName} not generated (nvcc -ptx -arch=compute_75 on a CUDA box)");

        float[] x = Build(name);
        int expected = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)x);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var argmax = CudaArgMaxF32.TryLoad(ptxDir!);
        Assert.NotNull(argmax);   // present but unloadable is a failure, not a skip

        nint d = Upload(x);
        try
        {
            argmax!.Launch(d, x.Length, stream.Handle);
            stream.Synchronize();
            Assert.Equal(expected, argmax.Result);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(d);
        }
    }

    private static float[] Build(string name)
    {
        var rng = new Random(0x486 + name.Sum(c => (int)c) * 31 + name.Length);   // stable across runs
        int n = name switch { "short-1" => 1, "short-100" => 100, _ => Vocab };
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 20 - 10);
        const float top = 50f;
        switch (name)
        {
            case "tie-same-thread": x[70_005] = top; x[70_005 + 1024] = top; x[70_005 + 4096] = top; break;
            case "tie-adjacent-lanes": x[123_457] = top; x[123_456] = top; break;
            case "tie-warp-boundary": x[200_000 + 32] = top; x[200_000 + 31] = top; break;
            case "tie-across-warps": x[1024 * 100 + 900] = top; x[1024 * 150 + 17] = top; break;
            case "tie-first-last": x[n - 1] = top; x[0] = top; break;
            case "max-at-last": x[n - 1] = top; break;
            case "all-equal": Array.Fill(x, 1.5f); break;
            case "pos-neg-zero":
                Array.Fill(x, -1f); x[500] = -0f; x[90_000] = 0f; x[100] = -0f; break;
            case "neg-pos-zero":
                Array.Fill(x, -1f); x[500] = 0f; x[90_000] = -0f; x[100] = -0f; break;
            case "nan-late": x[240_000] = float.NaN; x[5] = top; break;
            case "two-nans": x[150_000] = float.NaN; x[40_000] = float.NaN; x[3] = top; break;
            case "all-neg-inf": Array.Fill(x, float.NegativeInfinity); break;
        }
        return x;
    }

    private static unsafe nint Upload(float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        fixed (float* p = data) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}
