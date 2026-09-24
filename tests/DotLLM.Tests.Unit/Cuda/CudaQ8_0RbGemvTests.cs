using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #486: the register-blocked Q8_0 GEMV (<c>q8_0_gemv_f32in_rb</c> / <c>_rb_multi</c>) must be
/// <b>bit-identical</b> to the kernel the MTP path used before it
/// (<see cref="CudaKernels.LaunchQuantizedGemvF32In"/>) — single-column, and per column of the
/// multi-column form. Exact bit comparison, not a tolerance: that is what keeps the MTP draft and the
/// batched absorb bit-identical to the per-token path.
/// </summary>
/// <remarks>
/// <para>
/// Shapes cover every Bonsai 2 MTP-head projection and the kernel's geometry edges: a row count that
/// is not a multiple of the 4 rows per block, 1..8 active warps (bpr below 32, between, and above
/// 256 with a short final chunk), a row stride that is not 16-byte aligned (bpr % 8 != 0) and a weight
/// base offset by 2 bytes — both take the 2-byte staging path.
/// </para>
/// <para>
/// The multi-column test poisons every output slot with NaN, uses <c>ldx &gt; k</c> (the absorb's
/// concat rows) and <c>ldy &gt; n</c>, and asserts the padding between output columns is untouched —
/// so a kernel that ignores the column offset, the column count or the output stride fails.
/// Skips when the PTX has not been generated.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQ8_0RbGemvTests
{
    private readonly ITestOutputHelper _out;

    public CudaQ8_0RbGemvTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(5120, 10240, 0, true)]   // nextn.eh_proj
    [InlineData(12288, 5120, 0, true)]   // attn_q (gated: 2 * 24 * 256)
    [InlineData(1024, 5120, 0, true)]    // attn_k / attn_v
    [InlineData(5120, 6144, 0, true)]    // attn_output
    [InlineData(17408, 5120, 0, true)]   // ffn_gate / ffn_up
    [InlineData(5120, 17408, 0, true)]   // ffn_down (bpr 544: 256 + 256 + 32)
    [InlineData(3, 96, 0, false)]        // n < 4 rows; bpr 3 -> 1 warp, 2-byte staging
    [InlineData(7, 32 * 257, 0, false)]  // 1-block final chunk, odd row stride
    [InlineData(1, 32, 0, false)]        // single block
    [InlineData(65, 32 * 300, 0, false)] // bpr 300: 8 mod 16 row stride -> 2-byte staging, short chunk
    [InlineData(9, 32 * 72, 0, false)]   // bpr 72: 3 warps, last segment 8 blocks; 16-byte staging
    [InlineData(6, 32 * 160, 2, false)]  // Bonsai k=5120 geometry, weight base 2 bytes off -> 2-byte path
    public void Rb_IsBitIdentical_ToQuantizedGemvF32In(int n, int k, int weightOffset, bool time)
    {
        using var env = Env.Open(_out);
        if (env is null) return;

        var (w, x) = RandomProblem(n, k, 1);
        nint dWBase = 0, dX = 0, dY0 = 0, dY1 = 0;
        try
        {
            dWBase = Upload(w, weightOffset);
            nint dW = dWBase + weightOffset;
            dX = Upload(x, 0);
            dY0 = Alloc(n);
            dY1 = Alloc(n);
            Poison(dY1, n);

            env.Kernels.LaunchQuantizedGemvF32In(dW, dX, dY0, n, k, env.Stream.Handle);
            Assert.True(CudaQ8_0RbGemv.Accepts(dX, k, k));
            env.Rb.Launch(dW, dX, dY1, n, k, env.Stream.Handle);
            env.Stream.Synchronize();

            AssertBitEqual(Download(dY0, n), Download(dY1, n), $"n={n} k={k} off={weightOffset}");

            if (time)
            {
                double t0 = TimeMs(() => env.Kernels.LaunchQuantizedGemvF32In(dW, dX, dY0, n, k, env.Stream.Handle), env.Stream);
                double t2 = TimeMs(() => env.Rb.Launch(dW, dX, dY1, n, k, env.Stream.Handle), env.Stream);
                string staged = "";
                if (env.Staged is { } st)
                {
                    double t1 = TimeMs(() => st.Launch(dW, dX, dY1, n, k, env.Stream.Handle), env.Stream);
                    staged = $"  staged {t1:F3} ms ({w.Length / 1e6 / t1:F0} GB/s)";
                }
                _out.WriteLine($"  original {t0:F3} ms ({w.Length / 1e6 / t0:F0} GB/s){staged}  " +
                               $"rb {t2:F3} ms ({w.Length / 1e6 / t2:F0} GB/s)");
            }
        }
        finally
        {
            Free(dWBase); Free(dX); Free(dY0); Free(dY1);
        }
    }

    [SkippableTheory]
    [InlineData(5120, 10240, 5, true)]   // absorb eh_proj over S = K+1 = 5
    [InlineData(1024, 5120, 5, true)]    // absorb K / V
    [InlineData(7, 32 * 257, 1, false)]
    [InlineData(7, 32 * 257, 2, false)]
    [InlineData(13, 32 * 72, 8, false)]  // exactly one column group
    [InlineData(13, 32 * 300, 9, false)] // two column groups (8 + 1), 2-byte staging
    [InlineData(3, 96, 3, false)]
    public void RbMulti_EachColumn_IsBitIdentical_ToSingleColumnOriginal(int n, int k, int cols, bool time)
    {
        using var env = Env.Open(_out);
        if (env is null) return;
        Skip.IfNot(env.Rb.HasMulti, $"{CudaQ8_0RbGemv.MultiKernelName} missing from the PTX");

        int ldx = k + 64;                     // column stride > k, still a multiple of 4
        int ldy = n + 3;                      // output stride > n: gaps must stay poisoned
        var (w, xDense) = RandomProblem(n, k, cols);
        var x = new float[(long)cols * ldx];
        for (int c = 0; c < cols; c++)
            Array.Copy(xDense, (long)c * k, x, (long)c * ldx, k);

        nint dW = 0, dX = 0, dYRef = 0, dY = 0;
        try
        {
            dW = Upload(w, 0);
            dX = Upload(x, 0);
            dYRef = Alloc(n);
            dY = Alloc(cols * ldy);
            Poison(dY, cols * ldy);

            // Both dispatch shapes, against the same single-column oracle: the default (issue #492's
            // NCOLS-specialised entry point when the PTX carries one) and the generic 8-column
            // kernel the specialisation must match bit for bit.
            foreach (bool specialized in new[] { true, false })
            {
                if (specialized && !env.Rb.HasSpecialized(Math.Min(cols, CudaQ8_0RbGemv.MaxColumns)))
                    continue;                          // PTX predates #492: only the generic shape exists
                Poison(dY, cols * ldy);
                if (specialized)
                    env.Rb.LaunchMulti(dW, dX, ldx, dY, ldy, n, k, cols, env.Stream.Handle);
                else
                    env.Rb.LaunchMultiGeneric(dW, dX, ldx, dY, ldy, n, k, cols, env.Stream.Handle);
                env.Stream.Synchronize();
                float[] y = Download(dY, cols * ldy);

                for (int c = 0; c < cols; c++)
                {
                    nint xc = dX + (nint)((long)c * ldx * sizeof(float));
                    env.Kernels.LaunchQuantizedGemvF32In(dW, xc, dYRef, n, k, env.Stream.Handle);
                    env.Stream.Synchronize();
                    float[] yRef = Download(dYRef, n);
                    AssertBitEqual(yRef, y.AsSpan(c * ldy, n).ToArray(),
                        $"{(specialized ? "specialized" : "generic")} column {c} (n={n} k={k} cols={cols})");
                    for (int g = n; g < ldy; g++)
                        Assert.True(float.IsNaN(y[c * ldy + g]),
                            $"{(specialized ? "specialized" : "generic")} column {c}: padding slot {g} was written");
                }
            }

            if (time)
            {
                double tMulti = TimeMs(() => env.Rb.LaunchMulti(dW, dX, ldx, dY, ldy, n, k, cols, env.Stream.Handle), env.Stream);
                double tGeneric = TimeMs(() => env.Rb.LaunchMultiGeneric(dW, dX, ldx, dY, ldy, n, k, cols, env.Stream.Handle), env.Stream);
                double tSingle = TimeMs(() =>
                {
                    for (int c = 0; c < cols; c++)
                        env.Rb.Launch(dW, dX + (nint)((long)c * ldx * sizeof(float)),
                                      dY + (nint)((long)c * ldy * sizeof(float)), n, k, env.Stream.Handle);
                }, env.Stream);
                double gb = (double)w.Length / 1e6;    // MB of weights read once per multi launch
                _out.WriteLine($"  {cols} columns: multi {tMulti:F3} ms ({gb / tMulti:F0} GB/s) vs " +
                               $"generic {tGeneric:F3} ms ({gb / tGeneric:F0} GB/s) vs {cols} x single {tSingle:F3} ms " +
                               $"(specialized/generic {tGeneric / tMulti:F2}x, vs single {tSingle / tMulti:F2}x)");
                if (env.Rb.TryGetKernelInfo(cols, k, out int r1, out int s1, out int l1, out int b1))
                    _out.WriteLine($"    _c{cols}: {r1} regs, {s1} B smem, {l1} B local, {b1} blocks/SM");
                if (env.Rb.TryGetGenericKernelInfo(k, out int r0, out int s0, out int l0, out int b0))
                    _out.WriteLine($"    generic: {r0} regs, {s0} B smem, {l0} B local, {b0} blocks/SM");
            }
        }
        finally
        {
            Free(dW); Free(dX); Free(dYRef); Free(dY);
        }
    }

    /// <summary>
    /// Issue #492: the whole point of the specialised entry points is the launch shape, so assert it
    /// rather than only the bits. At eh_proj's geometry each <c>_c{cols}</c> kernel must stay inside
    /// the <c>__launch_bounds__(256, 2)</c> budget (registers capped, no local spill) and must be
    /// resident at least as many blocks per SM as the generic 8-column kernel it replaces — which was
    /// one, at 184 registers. A regression that pushes registers back up shows here as a lower
    /// blocks/SM, long before anyone re-runs a model-level profile.
    /// </summary>
    [SkippableTheory]
    [InlineData(5120, 10240)]   // nextn.eh_proj — blockDim 256, the shape that was 1 block/SM
    [InlineData(1024, 5120)]    // absorb K / V — blockDim 160
    public void RbMulti_SpecializedKernels_AreNotOccupancyStarved(int n, int k)
    {
        using var env = Env.Open(_out);
        if (env is null) return;
        Skip.IfNot(env.Rb.HasMulti, $"{CudaQ8_0RbGemv.MultiKernelName} missing from the PTX");
        Skip.IfNot(env.Rb.HasSpecialized(1),
            $"{CudaQ8_0RbGemv.SpecializedKernelName(1)} missing from the PTX (pre-#492 build, or " +
            $"{CudaQ8_0RbGemv.DisableSpecializedEnvVar}=0)");

        Assert.True(env.Rb.TryGetGenericKernelInfo(k, out int gRegs, out int gSmem, out int gLocal, out int gBlocks));
        _out.WriteLine($"n={n} k={k}: generic {gRegs} regs, {gSmem} B smem, {gLocal} B local, {gBlocks} blocks/SM");
        for (int cols = 1; cols <= CudaQ8_0RbGemv.MaxColumns; cols++)
        {
            Assert.True(env.Rb.HasSpecialized(cols), $"{CudaQ8_0RbGemv.SpecializedKernelName(cols)} missing");
            Assert.True(env.Rb.TryGetKernelInfo(cols, k, out int regs, out int smem, out int local, out int blocks));
            _out.WriteLine($"  _c{cols}: {regs} regs, {smem} B smem, {local} B local, {blocks} blocks/SM" +
                           (local > 0 ? "   ** SPILLED **" : ""));
            Assert.True(regs <= 128, $"_c{cols}: {regs} registers — __launch_bounds__(256, 2) should cap at 128");
            // Spill is reported, not asserted: it is a perf signal whose acceptable level depends on
            // the card, and the occupancy assertion below is the one that must hold.
            Assert.True(blocks >= gBlocks,
                $"_c{cols}: {blocks} blocks/SM vs the generic kernel's {gBlocks} — the specialisation lost occupancy");
        }
    }

    // ── fixture ─────────────────────────────────────────────────────────────

    private sealed class Env : IDisposable
    {
        public required CudaContext Context { get; init; }
        public required CudaStream Stream { get; init; }
        public required CudaKernels Kernels { get; init; }
        public required CudaQ8_0RbGemv Rb { get; init; }
        public CudaQ8_0StagedGemv? Staged { get; init; }

        public static Env? Open(ITestOutputHelper output)
        {
            Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir is null, "PTX files not found");
            Skip.IfNot(File.Exists(Path.Combine(ptxDir!, CudaQ8_0RbGemv.PtxFileName)),
                $"{CudaQ8_0RbGemv.PtxFileName} not generated (nvcc -ptx -arch=compute_75 on a CUDA box)");
            Skip.If(CudaQ8_0RbGemv.DisabledByEnv, $"{CudaQ8_0RbGemv.DisableEnvVar}=0");

            var ctx = CudaContext.Create(0);
            var stream = CudaStream.Create();
            var kernels = new CudaKernels(ptxDir!);
            var rb = CudaQ8_0RbGemv.TryLoad(ptxDir!);
            // Present but unloadable is a real failure, not a skip.
            Assert.NotNull(rb);
            return new Env
            {
                Context = ctx, Stream = stream, Kernels = kernels, Rb = rb!,
                Staged = CudaQ8_0StagedGemv.TryLoad(ptxDir!),
            };
        }

        public void Dispose()
        {
            Staged?.Dispose();
            Rb.Dispose();
            Kernels.Dispose();
            Stream.Dispose();
            Context.Dispose();
        }
    }

    private static (byte[] W, float[] X) RandomProblem(int n, int k, int cols)
    {
        var rng = new Random(0x486 + n * 7 + k + cols * 131);
        int bpr = k / 32;
        byte[] w = new byte[(long)n * bpr * 34];
        for (long b = 0; b < (long)n * bpr; b++)
        {
            long o = b * 34;
            ushort d = BitConverter.HalfToUInt16Bits((Half)(float)(rng.NextDouble() * 0.02 + 1e-4));
            w[o] = (byte)d;
            w[o + 1] = (byte)(d >> 8);
            for (int j = 0; j < 32; j++) w[o + 2 + j] = (byte)(sbyte)rng.Next(-128, 128);
        }
        float[] x = new float[(long)k * cols];
        for (long i = 0; i < x.LongLength; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return (w, x);
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{what}: row {i}: original {expected[i]:R} vs rb {actual[i]:R}");
        }
    }

    private static double TimeMs(Action launch, CudaStream stream)
    {
        for (int i = 0; i < 3; i++) launch();
        stream.Synchronize();
        const int iters = 20;
        long t = Stopwatch.GetTimestamp();
        for (int i = 0; i < iters; i++) launch();
        stream.Synchronize();
        return (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency / iters;
    }

    private static nint Alloc(int floats)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)((long)floats * sizeof(float))).ThrowOnError();
        return d;
    }

    private static void Poison(nint d, int floats) =>
        CudaDriverApi.cuMemsetD8_v2(d, 0xFF, (nuint)((long)floats * sizeof(float))).ThrowOnError();

    private static void Free(nint d)
    {
        if (d != 0) CudaDriverApi.cuMemFree_v2(d);
    }

    /// <summary>Uploads <paramref name="data"/> at byte <paramref name="offset"/> of a fresh allocation.</summary>
    private static unsafe nint Upload<T>(T[] data, int offset) where T : unmanaged
    {
        long bytes = (long)data.Length * sizeof(T);
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)(bytes + offset + 16)).ThrowOnError();
        fixed (T* p = data) CudaDriverApi.cuMemcpyHtoD_v2(d + offset, (nint)p, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static unsafe float[] Download(nint d, int n)
    {
        var r = new float[n];
        fixed (float* p = r) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)((long)n * sizeof(float))).ThrowOnError();
        return r;
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
