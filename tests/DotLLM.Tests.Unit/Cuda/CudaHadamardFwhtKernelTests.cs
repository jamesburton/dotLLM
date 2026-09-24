using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU↔CUDA parity for the PrismML Hadamard activation transform (<c>hadamard_fwht_f32</c>,
/// issue #479). <see cref="Hadamard"/> is the oracle (itself validated against the dense rotation
/// matrix in <c>HadamardTests</c>).
/// </summary>
/// <remarks>
/// The kernel is compiled with <c>-fmad=false</c>, takes the <c>1/sqrt(n)</c> scale from the host,
/// and its butterflies are pure add/sub in the same Sylvester order as the CPU, so it is expected to
/// be bit-exact. The assertion still uses a tight relative bound (a wrong transform is off by order
/// 1, not 1e-6) so that a toolchain which contracts differently does not produce a false alarm; the
/// exact-match count is printed so drift is visible.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaHadamardFwhtKernelTests
{
    private const float RelTol = 2e-6f;

    private readonly ITestOutputHelper _out;

    public CudaHadamardFwhtKernelTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(1, 16, 16)]        // single block, tiny
    [InlineData(3, 64, 16)]        // several blocks per row, several rows
    [InlineData(2, 512, 256)]
    [InlineData(1, 1024, 1024)]    // single Bonsai-2-sized block
    [InlineData(1, 5120, 1024)]    // Bonsai 2 hidden — attn/ffn inputs, lm_head
    [InlineData(4, 5120, 1024)]    // multi-row prefill
    [InlineData(1, 17408, 1024)]   // Bonsai 2 ffn_down input
    [InlineData(2, 8, 2)]          // smallest non-trivial block
    public void Forward_WithoutSigns_MatchesCpuOracle(int rows, int width, int blockSize)
    {
        using var env = CudaEnv.Create();
        var src = RandomFloats(new Random(0x4AD0 + rows * 31 + width + blockSize), rows * width);

        var expected = new float[src.Length];
        for (int t = 0; t < rows; t++)
            Hadamard.ForwardRow(src.AsSpan(t * width, width), ReadOnlySpan<sbyte>.Empty,
                expected.AsSpan(t * width, width), blockSize);

        var actual = env.Run(src, signs: null, rows, width, blockSize, inverse: false, perm: null);
        AssertClose(expected, actual, rows, width);
    }

    [SkippableTheory]
    [InlineData(1, 16, 16)]
    [InlineData(3, 256, 256)]
    [InlineData(2, 6144, 1024)]    // Bonsai 2 ssm_out / attn_output input width
    [InlineData(1, 17408, 1024)]
    public void Forward_WithSigns_MatchesCpuOracle(int rows, int width, int blockSize)
    {
        using var env = CudaEnv.Create();
        var src = RandomFloats(new Random(0x51A + width + rows), rows * width);
        var signs = RandomSigns(new Random(0x51B + width), width);

        var expected = new float[src.Length];
        for (int t = 0; t < rows; t++)
            Hadamard.ForwardRow(src.AsSpan(t * width, width), signs,
                expected.AsSpan(t * width, width), blockSize);

        var actual = env.Run(src, signs, rows, width, blockSize, inverse: false, perm: null);
        AssertClose(expected, actual, rows, width);
    }

    /// <summary>
    /// The inverse applies signs AFTER the rotation. A kernel that reuses the forward order passes
    /// every sign-free test above, so this is the case that catches it. Run in place, which is how
    /// a row-lookup inverse would use it.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 16, 16)]
    [InlineData(4, 5120, 1024)]
    [InlineData(3, 1024, 256)]
    public void Inverse_AppliesSignsAfterRotation_InPlace_MatchesCpuOracle(int rows, int width, int blockSize)
    {
        using var env = CudaEnv.Create();
        var src = RandomFloats(new Random(0x11B + width), rows * width);
        var signs = RandomSigns(new Random(0x11C + width), width);

        var expected = new float[src.Length];
        for (int t = 0; t < rows; t++)
            Hadamard.InverseRow(src.AsSpan(t * width, width), signs,
                expected.AsSpan(t * width, width), blockSize);

        var actual = env.Run(src, signs, rows, width, blockSize, inverse: true, perm: null, inPlace: true);
        AssertClose(expected, actual, rows, width);

        // Discriminator: the forward order must give a different answer, or this proves nothing.
        var forward = new float[src.Length];
        for (int t = 0; t < rows; t++)
            Hadamard.ForwardRow(src.AsSpan(t * width, width), signs,
                forward.AsSpan(t * width, width), blockSize);
        Assert.True(MaxAbsDiff(expected, forward) > 1e-4f,
            "forward and inverse sign order must differ, or the test cannot detect a swapped kernel");
    }

    /// <summary>
    /// The <c>ssm_out</c> path: tiled→grouped value-head remap on load, signs indexed in the
    /// permuted order. Covers a small non-degenerate geometry (heads narrower than a block AND a
    /// block spanning several heads) and the real Bonsai 2 geometry (dState 128, nKHead 16, rep 3).
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 8, 2, 3, 16)]       // width 48, 3 blocks of 16, each block spans 2 heads
    [InlineData(3, 16, 2, 2, 8)]       // width 64, heads wider than a block
    [InlineData(2, 128, 16, 3, 1024)]  // Bonsai 2: width 6144
    public void Forward_WithGdnPermute_MatchesCpuOracle(int rows, int dState, int nKHead, int rep, int blockSize)
    {
        using var env = CudaEnv.Create();
        int width = dState * nKHead * rep;

        var src = RandomFloats(new Random(0x6144 + rows + width), rows * width);
        var signs = RandomSigns(new Random(0x6145 + width), width);

        var expected = new float[src.Length];
        var permuted = new float[width];
        for (int t = 0; t < rows; t++)
        {
            Hadamard.PermuteTiledToGrouped(src.AsSpan(t * width, width), permuted, dState, nKHead, rep);
            Hadamard.ForwardRow(permuted, signs, expected.AsSpan(t * width, width), blockSize);
        }

        var actual = env.Run(src, signs, rows, width, blockSize, inverse: false, perm: (dState, nKHead, rep));
        AssertClose(expected, actual, rows, width);

        // Discriminator: without the permute the answer must differ.
        var unpermuted = env.Run(src, signs, rows, width, blockSize, inverse: false, perm: null);
        Assert.True(MaxAbsDiff(expected, unpermuted) > 1e-4f,
            "permuted and unpermuted results must differ, or the test cannot detect an ignored permute");
    }

    [SkippableFact]
    public void Launch_RejectsBadArguments()
    {
        using var env = CudaEnv.Create();
        var k = env.Kernels;
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            k.LaunchHadamardFwhtF32(1, 2, 0, 1, 2048, 2048, false, false, false, 0, 0, 0, 1f, env.Stream.Handle));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            k.LaunchHadamardFwhtF32(1, 2, 0, 1, 24, 12, false, false, false, 0, 0, 0, 1f, env.Stream.Handle));
        Assert.Throws<ArgumentException>(() =>
            k.LaunchHadamardFwhtF32(1, 2, 0, 1, 24, 16, false, false, false, 0, 0, 0, 1f, env.Stream.Handle));
        Assert.Throws<ArgumentException>(() =>   // permute in place
            k.LaunchHadamardFwhtF32(1, 1, 0, 1, 48, 16, false, false, true, 8, 2, 3, 1f, env.Stream.Handle));
    }

    // ── harness ───────────────────────────────────────────────────────────────

    private sealed class CudaEnv : IDisposable
    {
        public required CudaContext Context { get; init; }
        public required CudaStream Stream { get; init; }
        public required CudaKernels Kernels { get; init; }

        public static CudaEnv Create()
        {
            Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir is null, "PTX files not found");
            Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "hadamard_fwht.ptx")),
                "hadamard_fwht.ptx not generated (run native/build.ps1 on a CUDA box)");

            var ctx = CudaContext.Create(0);
            var stream = CudaStream.Create();
            var kernels = new CudaKernels(ptxDir!);
            // A present-but-stale PTX (no hadamard_fwht_f32 entry point) is a real failure, not a skip.
            Assert.True(kernels.HasHadamardFwht, "hadamard_fwht.ptx is present but has no hadamard_fwht_f32 entry point");
            return new CudaEnv { Context = ctx, Stream = stream, Kernels = kernels };
        }

        public unsafe float[] Run(float[] src, sbyte[]? signs, int rows, int width, int blockSize,
            bool inverse, (int DState, int NKHead, int Rep)? perm, bool inPlace = false)
        {
            long bytes = (long)src.Length * sizeof(float);
            nint dSrc = 0, dDst = 0, dSigns = 0;
            try
            {
                CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)bytes).ThrowOnError();
                fixed (float* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)bytes).ThrowOnError();
                if (inPlace)
                {
                    dDst = dSrc;
                }
                else
                {
                    CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)bytes).ThrowOnError();
                }

                if (signs is not null)
                {
                    var f = new float[width];
                    for (int i = 0; i < width; i++) f[i] = signs[i];
                    long sb = (long)width * sizeof(float);
                    CudaDriverApi.cuMemAlloc_v2(out dSigns, (nuint)sb).ThrowOnError();
                    fixed (float* p = f) CudaDriverApi.cuMemcpyHtoD_v2(dSigns, (nint)p, (nuint)sb).ThrowOnError();
                }

                Kernels.LaunchHadamardFwhtF32(dSrc, dDst, dSigns, rows, width, blockSize,
                    applySigns: signs is not null, inverse: inverse, permute: perm is not null,
                    perm?.DState ?? 0, perm?.NKHead ?? 0, perm?.Rep ?? 0,
                    1f / MathF.Sqrt(blockSize), Stream.Handle);
                Stream.Synchronize();

                var result = new float[src.Length];
                fixed (float* p = result) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)bytes).ThrowOnError();
                return result;
            }
            finally
            {
                if (dSigns != 0) CudaDriverApi.cuMemFree_v2(dSigns);
                if (dDst != 0 && dDst != dSrc) CudaDriverApi.cuMemFree_v2(dDst);
                if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            }
        }

        public void Dispose()
        {
            Kernels.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }
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
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private void AssertClose(float[] expected, float[] actual, int rows, int width)
    {
        float scale = 1e-9f;
        foreach (float e in expected) scale = MathF.Max(scale, MathF.Abs(e));

        float worst = 0;
        int worstIdx = -1, exact = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] == actual[i]) exact++;
            float rel = MathF.Abs(expected[i] - actual[i]) / scale;
            if (rel > worst) { worst = rel; worstIdx = i; }
        }
        _out.WriteLine($"{rows}x{width}: {exact}/{expected.Length} bit-exact, max rel err {worst:E3}");

        if (worst > RelTol)
            Assert.Fail(
                $"max relative error {worst:E3} at index {worstIdx} (row {worstIdx / width}, col {worstIdx % width} " +
                $"of {rows}x{width}); expected {expected[worstIdx]}, actual {actual[worstIdx]}");
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static sbyte[] RandomSigns(Random rng, int count)
    {
        var arr = new sbyte[count];
        for (int i = 0; i < count; i++) arr[i] = rng.Next(2) == 0 ? (sbyte)-1 : (sbyte)1;
        return arr;
    }
}
