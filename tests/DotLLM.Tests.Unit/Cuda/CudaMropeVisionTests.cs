using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #2073, CUDA text half of Bonsai 2.5 image input. The bookkeeping tests (<see cref="VisionRopeState"/>) are GPU-free and pin the same
/// semantics as the CPU model; the kernel tests need a CUDA device and the generated <c>rope_mrope_f32.ptx</c> and skip otherwise.
/// </summary>
public sealed class CudaMropeVisionBookkeepingTests
{
    private static ImageSpan Image(int start, int gw, int gh)
        => new(start, new ImageEmbeddings(gw, gh, 8, "h"));

    [Fact]
    public void WithImage_UsesGridAxes_AndRecordsTheDelta()
    {
        var kv = new object();
        var st = new VisionRopeState();
        // 3 text rows (0..2), a 4x2 image at slots 3..10, 2 text rows (11, 12)
        var img = Image(3, gw: 4, gh: 2);
        int[] slots = Enumerable.Range(0, 13).ToArray();
        st.Prepare(kv, slots, [img]);
        Assert.Equal(VisionRopeMode.Mrope, st.Mode);
        Assert.Equal(8 - 4, st.LastDelta);                       // tokens - max(gw, gh)
        Assert.Equal(4, st.DeltaFor(kv));
        var a = st.Axes.ToArray();
        int n = slots.Length;
        int[] t = a[..n], y = a[n..(2 * n)], x = a[(2 * n)..];
        Assert.Equal([0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 7, 8], t);
        Assert.Equal([0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 7, 8], y);   // y = base + row
        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8], x);   // x = base + col, non-square so y != x
    }

    [Fact]
    public void DecodeAfterImage_IsShiftedBackByTheDelta_AndSlotZeroClearsIt()
    {
        var kv = new object();
        var st = new VisionRopeState();
        var img = Image(2, gw: 3, gh: 3);                          // 9 tokens, delta 6
        st.Prepare(kv, Enumerable.Range(0, 12).ToArray(), [img]);
        Assert.Equal(6, st.DeltaFor(kv));

        st.Prepare(kv, [12], null);                                // a decode step
        Assert.Equal(VisionRopeMode.Shifted, st.Mode);
        Assert.Equal(6, st.Shift);
        Assert.Equal([6], st.Axes.ToArray());                      // slot 12 rotates at 12 - 6

        st.Prepare(kv, [13, 14, 15], null);                        // a verify
        Assert.Equal([7, 8, 9], st.Axes.ToArray());

        st.Prepare(kv, [0, 1], null);                              // a new sequence on the same cache
        Assert.Equal(VisionRopeMode.Plain, st.Mode);
        Assert.Equal(0, st.DeltaFor(kv));
        Assert.Equal(0, st.LastDelta);
        st.Prepare(kv, [2], null);
        Assert.Equal(VisionRopeMode.Plain, st.Mode);
    }

    [Fact]
    public void DeltaIsKeyedPerSequence()
    {
        var kvA = new object(); var kvB = new object();
        var st = new VisionRopeState();
        st.Prepare(kvA, Enumerable.Range(0, 6).ToArray(), [Image(0, 2, 3)]);
        st.Prepare(kvB, [7], null);
        Assert.Equal(VisionRopeMode.Plain, st.Mode);               // B never saw an image
        st.Prepare(kvA, [6], null);
        Assert.Equal(VisionRopeMode.Shifted, st.Mode);
        Assert.Equal(3, st.Shift);
    }

    [Fact]
    public void ChunkedPrefill_MatchesTheSinglePass()
    {
        var kv = new object();
        var img = Image(2, 5, 3);                                  // 15 tokens at 2..16
        int[] all = Enumerable.Range(0, 24).ToArray();
        var whole = new VisionRopeState();
        whole.Prepare(kv, all, [img]);
        int n = all.Length;
        var w = whole.Axes.ToArray();
        var chunked = new VisionRopeState();
        for (int c = 0; c < n; c += 7)
        {
            int m = Math.Min(7, n - c);
            chunked.Prepare(kv, all.AsSpan(c, m), [img]);           // spans cover the whole request on every call
            var a = chunked.Axes.ToArray();
            for (int axis = 0; axis < 3; axis++)
                Assert.Equal(w.AsSpan(axis * n + c, m).ToArray(), a.AsSpan(axis * m, m).ToArray());
        }
        Assert.Equal(15 - 5, chunked.LastDelta);
    }
}

[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaMropeKernelTests
{
    private const int Heads = 6, KvHeads = 2, HeadDim = 256, RopeDim = 64;   // Bonsai 2.5 27B shapes (32 rotary pairs)

    [SkippableFact]
    public unsafe void MropeKernel_MatchesCpuOracle_OnANonSquareGrid()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMropeF32, "rope_mrope_f32.ptx not generated (nvcc -ptx on a CUDA box)");

        // a 5x3 grid (non-square) starting at slot 4, text before and after
        var img = new ImageSpan(4, new ImageEmbeddings(5, 3, 8, "h"));
        int n = 24;
        int[] slots = Enumerable.Range(0, n).ToArray();
        var t = new int[n]; var y = new int[n]; var x = new int[n];
        MropePositions.Fill(slots, [img], t, y, x);
        Assert.NotEqual(y, x);   // the fixture must distinguish the axes
        byte[] axis = MropePositions.AxisPerPair(MropePositions.DefaultSections, RopeDim / 2);

        var rng = new Random(2073);
        float[] q0 = Enumerable.Range(0, n * Heads * HeadDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] k0 = Enumerable.Range(0, n * KvHeads * HeadDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();

        const float theta = 1e7f;
        var rope = new RoPERotation(new RoPEConfig(Theta: theta, DimensionCount: RopeDim), Architecture.Qwen3HybridDense, RopeDim);
        var qRef = (float[])q0.Clone(); var kRef = (float[])k0.Clone();
        rope.ApplyMrope(qRef, kRef, t, y, x, axis, Heads, KvHeads, HeadDim);

        var (qGpu, kGpu) = RunGpu(kernels, stream, q0, k0, t.Concat(y).Concat(x).ToArray(), axis, n, theta);
        AssertClose(qRef, qGpu, "q"); AssertClose(kRef, kGpu, "k");

        // control: the axes must matter (a swapped y/x run differs from the oracle), so agreement above is not vacuous
        var (qSwap, _) = RunGpu(kernels, stream, q0, k0, t.Concat(x).Concat(y).ToArray(), axis, n, theta);
        Assert.True(MaxAbsDiff(qRef, qSwap) > 1e-2f, "swapping y and x must change the rotation");
    }

    [SkippableFact]
    public unsafe void MropeKernel_WithEqualAxes_IsBitIdenticalToTheNeoxKernel()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMropeF32, "rope_mrope_f32.ptx not generated (nvcc -ptx on a CUDA box)");

        int n = 9;
        int[] pos = [0, 3, 4, 17, 18, 100, 101, 5000, 5001];
        byte[] axis = MropePositions.AxisPerPair(MropePositions.DefaultSections, RopeDim / 2);
        var rng = new Random(11);
        float[] q0 = Enumerable.Range(0, n * Heads * HeadDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        float[] k0 = Enumerable.Range(0, n * KvHeads * HeadDim).Select(_ => (float)rng.NextDouble() - 0.5f).ToArray();
        const float theta = 1e7f;

        var (qM, kM) = RunGpu(kernels, stream, q0, k0, pos.Concat(pos).Concat(pos).ToArray(), axis, n, theta);

        nint dq = Upload(q0), dk = Upload(k0), dp = Upload(pos);
        try
        {
            kernels.LaunchRoPEF32(dq, dk, dp, n, Heads, KvHeads, HeadDim, RopeDim, theta, 1, stream.Handle);
            stream.Synchronize();
            var qN = Download(dq, q0.Length); var kN = Download(dk, k0.Length);
            Assert.Equal(qN, qM); Assert.Equal(kN, kM);
        }
        finally { CudaDriverApi.cuMemFree_v2(dq); CudaDriverApi.cuMemFree_v2(dk); CudaDriverApi.cuMemFree_v2(dp); }
    }

    private static unsafe (float[] q, float[] k) RunGpu(CudaKernels kernels, CudaStream stream, float[] q0, float[] k0, int[] pos3, byte[] axis, int n, float theta)
    {
        nint dq = Upload(q0), dk = Upload(k0), dp = Upload(pos3), da = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out da, (nuint)axis.Length).ThrowOnError();
            fixed (byte* p = axis) CudaDriverApi.cuMemcpyHtoD_v2(da, (nint)p, (nuint)axis.Length).ThrowOnError();
            kernels.LaunchMropeF32(dq, dk, dp, da, n, Heads, KvHeads, HeadDim, RopeDim, theta, stream.Handle);
            stream.Synchronize();
            return (Download(dq, q0.Length), Download(dk, k0.Length));
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dq); CudaDriverApi.cuMemFree_v2(dk); CudaDriverApi.cuMemFree_v2(dp);
            if (da != 0) CudaDriverApi.cuMemFree_v2(da);
        }
    }

    // sinf/cosf on the device and the CPU's PortableMath differ in the last ulps of the angle (positions reach ~30, theta 1e7).
    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        float d = MaxAbsDiff(expected, actual);
        Assert.True(d < 1e-4f, $"{what}: max |cpu - gpu| = {d}");
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static unsafe nint Upload<T>(T[] data) where T : unmanaged
    {
        long bytes = (long)data.Length * sizeof(T);
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        fixed (T* p = data) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static unsafe float[] Download(nint d, int count)
    {
        var r = new float[count];
        fixed (float* p = r) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)((long)count * sizeof(float))).ThrowOnError();
        return r;
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
