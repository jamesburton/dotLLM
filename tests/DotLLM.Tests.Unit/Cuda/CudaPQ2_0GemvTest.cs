using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the CUDA PQ2_0 (PrismML Bonsai ternary) GEMV against the CPU float reference
/// (<see cref="MatMul.GemvPQ2_0"/>). The F32-in kernel is an exact analog of the CPU path, so it
/// must match to fp32 reduction-order error. Uses synthetic packed tensors at real Bonsai-27B
/// dims (hidden=5120, ffn=17408 — see <c>qwen35.embedding_length</c>/<c>qwen35.feed_forward_length</c>
/// in the real GGUF) — no model file needed.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaPQ2_0GemvTest
{
    private readonly ITestOutputHelper _out;
    public CudaPQ2_0GemvTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableTheory]
    [InlineData(512, 5120)]   // attention-shaped, hidden=5120 (Bonsai qwen35.embedding_length)
    [InlineData(512, 17408)]  // FFN-shaped, k=17408 (Bonsai qwen35.feed_forward_length)
    public void PQ2_0GemvF32In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Run(n, k);
    }

    /// <summary>
    /// Validates the v2 F16 production decode kernel (shared-x staging + warp-per-row,
    /// PQ2_0_ROWS_PER_BLOCK=16) against the CPU float reference. Covers real Bonsai attention
    /// and FFN shapes plus n values that are NOT multiples of 16 (37, 3) to exercise the
    /// tail-row clamp path (<c>min(rowBase+rr, n-1)</c> when a block's last warp only partially
    /// overlaps valid rows, and the n=3 case where a single block covers far more rows than
    /// exist).
    ///
    /// k=5120 and k=17408 also exercise <see cref="CudaKernels.LaunchPQ2_0GemvF16In"/>'s
    /// dispatch-by-k routing to the <c>_small</c>/default kernel variants respectively (#157
    /// round 4 — see native/kernels/pq2_0_gemv.cu's "Small-K specialization" comment). k=5248
    /// (Pq2_0MaxKSmall+128, one group above the small threshold) specifically guards the
    /// dispatch BOUNDARY: if the routing were off-by-one and sent this shape into the
    /// <c>xs[PQ2_0_MAX_K_SMALL]</c>-sized kernel, the vectorized staging loop would write past
    /// the end of that shared-memory array (undefined behavior, most likely visible here as a
    /// wrong/corrupted result rather than a crash) — this test would catch that even though
    /// k=5120/17408 alone would not.
    /// </summary>
    [SkippableTheory]
    [InlineData(512, 5120)]
    [InlineData(512, 17408)]
    [InlineData(512, 5248)]
    [InlineData(37, 5120)]
    [InlineData(3, 5120)]
    public void PQ2_0GemvF16In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunF16(n, k);
    }

    /// <summary>
    /// Validates the F32-native production decode kernel (<see cref="CudaKernels.LaunchPQ2_0GemvF32Native"/>,
    /// issue #161 — converts F32&lt;-&gt;F16 inline in the kernel's own vectorized stage/store steps
    /// instead of via surrounding convert_f32_to_f16/convert_f16_to_f32 launches) against the CPU
    /// float reference. Same shapes/tail-clamp/dispatch-boundary coverage as
    /// <see cref="PQ2_0GemvF16In_MatchesCpuFloatReference"/> — see that test's doc comment for the
    /// rationale of each (n, k) pair, in particular k=5248 guarding the small/default dispatch
    /// boundary for the new <c>_f32io</c>/<c>_f32io_small</c> kernel pair.
    /// </summary>
    [SkippableTheory]
    [InlineData(512, 5120)]
    [InlineData(512, 17408)]
    [InlineData(512, 5248)]
    [InlineData(37, 5120)]
    [InlineData(3, 5120)]
    public void PQ2_0GemvF32Native_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunF32Native(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunF32Native(int n, int k)
    {
        var rng = new Random(5678);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference — full-precision float, same math the kernel approximates via half xs[].
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvPQ2_0(w, px, py, n, k, null);

        // GPU — F32-native in/out production path (#161): x/y stay float end-to-end, no Half
        // marshaling on the C# side and no convert_f32_to_f16/convert_f16_to_f32 kernel launches.
        float[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            long splitLen = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devWSplit, (nuint)splitLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (float* px = x)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(float))).ThrowOnError();

                // Repack interleaved -> split layout (mirrors the real production load path,
                // CudaQwen3HybridDenseTransformerModel.UploadRawTensor) — same as RunF16.
                kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);
                kernels.LaunchPQ2_0GemvF32Native(devWSplit, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new float[n];
                fixed (float* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devWSplit);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devY);
            }
        }

        float maxAbsDiff = 0, sumAbsDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(cpu[i] - gpu[i]);
            sumAbsDiff += d;
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
        float meanAbsDiff = sumAbsDiff / n;
        _out.WriteLine($"PQ2_0 GEMV F32Native {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        // Internal accumulation still goes through half xs[] (unchanged from the F16In kernel),
        // so the tolerance bar matches PQ2_0GemvF16In_MatchesCpuFloatReference's — the only
        // difference from that kernel is where the F32<->F16 conversion happens (fused into
        // stage/store vs a separate launch), not how much rounding occurs. The output side is, if
        // anything, slightly MORE precise here (no extra half-rounding on the way out through
        // rowOut[]/y[] — see native/kernels/pq2_0_gemv.cu's "F32-native activations" section), so
        // this bar should never bind tighter than the F16In test's.
        Assert.True(maxAbsDiff <= 5e-2f, $"max abs diff {maxAbsDiff} exceeds 5e-2 (F16-internal-precision vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-2f, $"mean abs diff {meanAbsDiff} exceeds 1e-2");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunF16(int n, int k)
    {
        var rng = new Random(5678);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        Half[] xh = new Half[k];
        for (int i = 0; i < k; i++) xh[i] = (Half)x[i];

        // CPU reference — full-precision float, same math the kernel approximates in F16.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvPQ2_0(w, px, py, n, k, null);

        // GPU — F16 in/out production path.
        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            long splitLen = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devWSplit, (nuint)splitLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (Half* px = xh)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(ushort))).ThrowOnError();

                // Repack interleaved -> split layout (mirrors the real production load path,
                // CudaQwen3HybridDenseTransformerModel.UploadRawTensor) — this also validates the
                // repack kernel itself, not just the GEMV kernel's split-layout reads in isolation.
                kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);
                kernels.LaunchPQ2_0GemvF16In(devWSplit, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new Half[n];
                fixed (Half* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devWSplit);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devY);
            }
        }

        float maxAbsDiff = 0, sumAbsDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(cpu[i] - (float)gpu[i]);
            sumAbsDiff += d;
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
        float meanAbsDiff = sumAbsDiff / n;
        _out.WriteLine($"PQ2_0 GEMV F16In {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        // F16 has ~3 decimal digits of precision; the CPU reference is full float32. Looser than
        // the F32In exact-match test, but still tight — max magnitude here is O(1-10) given the
        // synthetic scale/x ranges, so an absolute 5e-2 bar comfortably separates "F16 rounding"
        // from "wrong row/tail-clamp/shared-stage logic".
        Assert.True(maxAbsDiff <= 5e-2f, $"max abs diff {maxAbsDiff} exceeds 5e-2 (F16 vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-2f, $"mean abs diff {meanAbsDiff} exceeds 1e-2");
    }

    /// <summary>
    /// Validates the prefill-path dequant kernel (<see cref="CudaKernels.LaunchDequantPQ2_0ToF16"/>,
    /// v2: one warp per group, coalesced) against the CPU dequant reference
    /// (<see cref="Dequantize.ToFloat32"/>). Covers non-multiple-of-32-groups-per-warp-batch
    /// shapes (n=37 doesn't divide evenly into the warp-per-group grid-stride) to exercise the
    /// tail.
    /// </summary>
    [SkippableTheory]
    [InlineData(512, 5120)]
    [InlineData(37, 5120)]
    [InlineData(3, 17408)]
    public void DequantPQ2_0ToF16_MatchesCpuReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunDequant(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunDequant(int n, int k)
    {
        var rng = new Random(9012);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        // CPU reference — dequantizes the whole [n, k] matrix via the shared CPU codepath.
        float[] cpu = new float[(long)n * k];
        fixed (byte* w = packed)
            Dequantize.ToFloat32((nint)w, (long)n * k, DotLLM.Core.Configuration.QuantizationType.PQ2_0, cpu);

        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            long splitLen = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devWSplit, (nuint)splitLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devDst, (nuint)((long)n * k * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();

                // Repack interleaved -> split layout — see RunF16's identical comment above.
                kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);
                kernels.LaunchDequantPQ2_0ToF16(devWSplit, devDst, n, k, s);
                stream.Synchronize();

                gpu = new Half[(long)n * k];
                fixed (Half* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devDst, (nuint)((long)n * k * sizeof(ushort))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devWSplit);
                CudaDriverApi.cuMemFree_v2(devDst);
            }
        }

        float maxAbsDiff = 0, sumAbsDiff = 0;
        for (long i = 0; i < cpu.Length; i++)
        {
            float d = MathF.Abs(cpu[i] - (float)gpu[i]);
            sumAbsDiff += d;
            if (d > maxAbsDiff) maxAbsDiff = d;
        }
        float meanAbsDiff = sumAbsDiff / cpu.Length;
        _out.WriteLine($"PQ2_0 dequant {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        // F16 output vs full-float32 CPU reference — same rounding-only tolerance as the GEMV
        // F16In test (values here are single ternary*scale terms, no summation, so error is
        // pure F16 rounding — tighter bound than the GEMV test's accumulated-sum tolerance).
        Assert.True(maxAbsDiff <= 5e-4f, $"max abs diff {maxAbsDiff} exceeds 5e-4 (F16 rounding vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-4f, $"mean abs diff {meanAbsDiff} exceeds 1e-4");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(int n, int k)
    {
        var rng = new Random(1234);
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        long packedLen = (long)n * rowBytes;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++) groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        byte[] packed = Pack(ternary, groupScales, n, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvPQ2_0(w, px, py, n, k, null);

        // GPU.
        float[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(float))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (float* px = x)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(float))).ThrowOnError();

                kernels.LaunchPQ2_0GemvF32In(devW, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new float[n];
                fixed (float* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
                CudaDriverApi.cuMemFree_v2(devX);
                CudaDriverApi.cuMemFree_v2(devY);
            }
        }

        float maxDiff = 0, sumDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(cpu[i] - gpu[i]);
            sumDiff += d;
            if (d > maxDiff) maxDiff = d;
        }
        float meanDiff = sumDiff / n;
        _out.WriteLine($"PQ2_0 GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        Assert.True(maxDiff <= 1e-3f, $"max abs diff {maxDiff} exceeds 1e-3 (CPU vs GPU should match to fp32)");
        Assert.True(meanDiff <= 1e-4f, $"mean abs diff {meanDiff} exceeds 1e-4");
    }

    /// <summary>
    /// Validates the fused 2-way GEMV (<see cref="CudaKernels.LaunchPQ2_0Gemv2F16In"/>, used for
    /// decode-time dense-FFN gate+up and full-attention K+V) produces the same output as two
    /// separate <see cref="CudaKernels.LaunchPQ2_0GemvF16In"/> calls. Uses odd, unequal n0/n1
    /// (mirroring the I2_S fused-decode test) to exercise the row-selection/tail-clamp boundary
    /// between the two virtually-concatenated weight matrices.
    ///
    /// k=5120 (attention/GDN shape) and k=17408 (FFN shape) exercise
    /// <see cref="CudaKernels.LaunchPQ2_0Gemv2F16In"/>'s dispatch-by-k routing to the
    /// <c>pq2_0_gemv2_f16in_small</c>/<c>pq2_0_gemv2_f16in</c> variants respectively (#157 round
    /// 4). k=5248 is the same boundary-above-threshold guard as the single-GEMV test above,
    /// applied to the fused kernel's own <c>xs[]</c> staging buffer.
    /// </summary>
    [SkippableTheory]
    [InlineData(5120)]
    [InlineData(17408)]
    [InlineData(5248)]
    public void PQ2_0GemvFusedDecode_MatchesSeparateLaunches(int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunFusedDecode(k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunFusedDecode(int k)
    {
        const int n0 = 37;
        const int n1 = 53;

        var rng = new Random(2468);
        byte[] w0 = RandomPacked(rng, n0, k);
        byte[] w1 = RandomPacked(rng, n1, k);

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.5f);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        nint s = stream.Handle;

        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        static nuint PackedBytes(int n, int rowBytes) => (nuint)((long)n * rowBytes);

        long splitLen0 = CudaKernels.PQ2_0SplitLayoutBytes(n0, k);
        long splitLen1 = CudaKernels.PQ2_0SplitLayoutBytes(n1, k);

        CudaDriverApi.cuMemAlloc_v2(out nint devW0, PackedBytes(n0, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1, PackedBytes(n1, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW0Split, (nuint)splitLen0).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1Split, (nuint)splitLen1).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        try
        {
            fixed (byte* p = w0) CudaDriverApi.cuMemcpyHtoD_v2(devW0, (nint)p, PackedBytes(n0, rowBytes)).ThrowOnError();
            fixed (byte* p = w1) CudaDriverApi.cuMemcpyHtoD_v2(devW1, (nint)p, PackedBytes(n1, rowBytes)).ThrowOnError();
            fixed (Half* p = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)p, (nuint)(k * sizeof(ushort))).ThrowOnError();

            // Repack interleaved -> split layout — see RunF16's identical comment above. Each
            // virtually-concatenated array (w0/w1) is a physically separate tensor with its own
            // split-layout buffer, mirroring how UploadRawTensor repacks each tensor independently.
            kernels.LaunchPQ2_0RepackSplitF16(devW0, devW0Split, n0, k, s);
            kernels.LaunchPQ2_0RepackSplitF16(devW1, devW1Split, n1, k, s);

            kernels.LaunchPQ2_0GemvF16In(devW0Split, devX, sep0, n0, k, s);
            kernels.LaunchPQ2_0GemvF16In(devW1Split, devX, sep1, n1, k, s);
            kernels.LaunchPQ2_0Gemv2F16In(devW0Split, devW1Split, devX, fus0, fus1, n0, n1, k, s);
            stream.Synchronize();

            AssertHalfClose(sep0, fus0, n0);
            AssertHalfClose(sep1, fus1, n1);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW0); CudaDriverApi.cuMemFree_v2(devW1);
            CudaDriverApi.cuMemFree_v2(devW0Split); CudaDriverApi.cuMemFree_v2(devW1Split);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(sep0); CudaDriverApi.cuMemFree_v2(sep1);
            CudaDriverApi.cuMemFree_v2(fus0); CudaDriverApi.cuMemFree_v2(fus1);
        }
    }

    /// <summary>
    /// Validates the F32-native fused 2-way GEMV (<see cref="CudaKernels.LaunchPQ2_0Gemv2F32Native"/>,
    /// issue #161 — the F32-native analog of <see cref="CudaKernels.LaunchPQ2_0Gemv2F16In"/>)
    /// produces the same output as two separate <see cref="CudaKernels.LaunchPQ2_0GemvF32Native"/>
    /// calls. Same shape coverage as <see cref="PQ2_0GemvFusedDecode_MatchesSeparateLaunches"/>.
    /// </summary>
    [SkippableTheory]
    [InlineData(5120)]
    [InlineData(17408)]
    [InlineData(5248)]
    public void PQ2_0GemvFusedDecodeF32Native_MatchesSeparateLaunches(int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunFusedDecodeF32Native(k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunFusedDecodeF32Native(int k)
    {
        const int n0 = 37;
        const int n1 = 53;

        var rng = new Random(2468);
        byte[] w0 = RandomPacked(rng, n0, k);
        byte[] w1 = RandomPacked(rng, n1, k);

        float[] x = new float[k];
        for (int i = 0; i < k; i++)
            x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        nint s = stream.Handle;

        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        static nuint PackedBytes(int n, int rowBytes) => (nuint)((long)n * rowBytes);

        long splitLen0 = CudaKernels.PQ2_0SplitLayoutBytes(n0, k);
        long splitLen1 = CudaKernels.PQ2_0SplitLayoutBytes(n1, k);

        CudaDriverApi.cuMemAlloc_v2(out nint devW0, PackedBytes(n0, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1, PackedBytes(n1, rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW0Split, (nuint)splitLen0).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1Split, (nuint)splitLen1).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)(k * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep0, (nuint)(n0 * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep1, (nuint)(n1 * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus0, (nuint)(n0 * sizeof(float))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus1, (nuint)(n1 * sizeof(float))).ThrowOnError();
        try
        {
            fixed (byte* p = w0) CudaDriverApi.cuMemcpyHtoD_v2(devW0, (nint)p, PackedBytes(n0, rowBytes)).ThrowOnError();
            fixed (byte* p = w1) CudaDriverApi.cuMemcpyHtoD_v2(devW1, (nint)p, PackedBytes(n1, rowBytes)).ThrowOnError();
            fixed (float* p = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)p, (nuint)(k * sizeof(float))).ThrowOnError();

            // Repack interleaved -> split layout — see RunF16's identical comment above.
            kernels.LaunchPQ2_0RepackSplitF16(devW0, devW0Split, n0, k, s);
            kernels.LaunchPQ2_0RepackSplitF16(devW1, devW1Split, n1, k, s);

            kernels.LaunchPQ2_0GemvF32Native(devW0Split, devX, sep0, n0, k, s);
            kernels.LaunchPQ2_0GemvF32Native(devW1Split, devX, sep1, n1, k, s);
            kernels.LaunchPQ2_0Gemv2F32Native(devW0Split, devW1Split, devX, fus0, fus1, n0, n1, k, s);
            stream.Synchronize();

            AssertFloatClose(sep0, fus0, n0);
            AssertFloatClose(sep1, fus1, n1);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW0); CudaDriverApi.cuMemFree_v2(devW1);
            CudaDriverApi.cuMemFree_v2(devW0Split); CudaDriverApi.cuMemFree_v2(devW1Split);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(sep0); CudaDriverApi.cuMemFree_v2(sep1);
            CudaDriverApi.cuMemFree_v2(fus0); CudaDriverApi.cuMemFree_v2(fus1);
        }
    }

    private static unsafe void AssertFloatClose(nint expectedDevice, nint actualDevice, int n)
    {
        float[] expected = new float[n];
        float[] actual = new float[n];
        fixed (float* p = expected)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, expectedDevice, (nuint)(n * sizeof(float))).ThrowOnError();
        fixed (float* p = actual)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, actualDevice, (nuint)(n * sizeof(float))).ThrowOnError();

        for (int i = 0; i < n; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= 1e-3f, $"index {i}: expected {expected[i]}, actual {actual[i]}");
        }
    }

    private static unsafe void AssertHalfClose(nint expectedDevice, nint actualDevice, int n)
    {
        Half[] expected = new Half[n];
        Half[] actual = new Half[n];
        fixed (Half* p = expected)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, expectedDevice, (nuint)(n * sizeof(ushort))).ThrowOnError();
        fixed (Half* p = actual)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, actualDevice, (nuint)(n * sizeof(ushort))).ThrowOnError();

        for (int i = 0; i < n; i++)
        {
            float diff = MathF.Abs((float)expected[i] - (float)actual[i]);
            Assert.True(diff <= 1e-3f, $"index {i}: expected {(float)expected[i]}, actual {(float)actual[i]}");
        }
    }

    private static byte[] RandomPacked(Random rng, int n, int k)
    {
        int groupsPerRow = k / 128;
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++)
            ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] groupScales = new float[n * groupsPerRow];
        for (int i = 0; i < groupScales.Length; i++)
            groupScales[i] = 0.01f + rng.NextSingle() * 0.05f;
        return Pack(ternary, groupScales, n, k);
    }

    /// <summary>Packs ternary {-1,0,+1} into dotLLM PQ2_0 layout: per-128-group Half scale + codes.</summary>
    private static byte[] Pack(sbyte[] ternary, float[] groupScales, int n, int k)
    {
        int groupsPerRow = k / 128;
        int rowBytes = groupsPerRow * 34;
        byte[] buf = new byte[(long)n * rowBytes];

        for (int r = 0; r < n; r++)
        {
            long rowBase = (long)r * rowBytes;
            for (int g = 0; g < groupsPerRow; g++)
            {
                long groupBase = rowBase + (long)g * 34;
                Half scale = (Half)groupScales[r * groupsPerRow + g];
                BitConverter.GetBytes(BitConverter.HalfToUInt16Bits(scale)).CopyTo(buf, (int)groupBase);

                int baseIdx = r * k + g * 128;
                for (int gp = 0; gp < 32; gp++)
                {
                    int code0 = ternary[baseIdx + gp] + 1;
                    int code1 = ternary[baseIdx + gp + 32] + 1;
                    int code2 = ternary[baseIdx + gp + 64] + 1;
                    int code3 = ternary[baseIdx + gp + 96] + 1;
                    buf[groupBase + 2 + gp] = (byte)((code0 << 6) | (code1 << 4) | (code2 << 2) | code3);
                }
            }
        }
        return buf;
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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
