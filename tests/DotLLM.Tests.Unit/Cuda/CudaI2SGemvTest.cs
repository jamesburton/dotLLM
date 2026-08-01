using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates the CUDA I2_S (BitNet ternary) GEMV against the CPU float reference (MatMul.GemvI2_S).
/// Variant A (W2A16, here the FP32-in twin) is an exact analog of the CPU path, so it must match to
/// fp32 reduction-order error. Uses synthetic packed tensors at real BitNet 2B4T dims — no model file.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaI2SGemvTest
{
    private readonly ITestOutputHelper _out;
    public CudaI2SGemvTest(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableTheory]
    [InlineData(2560, 2560)]  // attention projection
    [InlineData(2560, 6912)]  // FFN down (k = 6912) — BitNet b1.58-2B-4T's own bound
    // Regression coverage for issue #207: the x-staging shared buffer used to be a fixed-size
    // STATIC `__shared__ float xs[6912]`, sized specifically for BitNet-2B-4T's own largest k
    // (its FFN-down projection). A non-BitNet Llama-body I2_S GGUF whose intermediate size
    // exceeds 6912 dispatches its FFN-down decode GEMV (Project() → LaunchI2_SGemvF16In, k =
    // DownInputDim = intermediateSize) through the exact same kernel and used to silently
    // overflow that array, corrupting shared memory and producing a CUDA "illegal memory
    // access" fault. These two cases are the REAL Down-projection shapes from the two models
    // that surfaced the bug (Falcon-E-3B-Instruct: hidden=2048/intermediate=13312; Falcon3-3B-
    // Base-1.58bit: hidden=3072/intermediate=9216) — both exceed the old 6912 static bound and
    // must now succeed (dynamic shared memory, sized per-call — see i2_s_gemv.cu).
    [InlineData(2048, 13312)] // Falcon-E-3B-Instruct FFN down (k = intermediateSize = 13312)
    [InlineData(3072, 9216)]  // Falcon3-3B-Base-1.58bit FFN down (k = intermediateSize = 9216)
    public void I2SGemvF32In_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Run(n, k);
    }

    /// <summary>
    /// Same regression as <see cref="I2SGemvF32In_MatchesCpuFloatReference"/> but through the
    /// FP16 GEMV (<see cref="CudaKernels.LaunchI2_SGemvF16In"/>) — the kernel actually dispatched
    /// in production for decode-time projections (<c>CudaTransformerModel.Project</c>), including
    /// the FFN-down projection that triggered issue #207 on Llama-arch I2_S GGUFs.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, 13312)] // Falcon-E-3B-Instruct FFN down
    [InlineData(3072, 9216)]  // Falcon3-3B-Base-1.58bit FFN down
    public void I2SGemvF16In_LargeNonBitNetIntermediateK_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunF16(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Run(int n, int k)
    {
        var rng = new Random(1234);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        // Random ternary weights + a small positive per-tensor scale at the tensor tail.
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference.
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvI2_S(w, px, py, n, k, null);

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

                kernels.LaunchI2_SGemvF32In(devW, devX, devY, n, k, s);
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
        _out.WriteLine($"I2_S GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        // Tolerance scales with sqrt(k), because BOTH sides accumulate in float32 in DIFFERENT
        // orders and the error that produces is proportional to the result magnitude, which itself
        // grows as sqrt(k) for a k-term dot product of zero-mean values. A fixed absolute bound
        // therefore cannot hold across k: it silently tightens as k grows.
        //
        // This is not GPU-vs-CPU imprecision in the usual sense. The reference is MatMul.GemvI2_S,
        // the production SIMD kernel, so its summation order depends on the host's vector width —
        // the same test can pass on one machine and fail on another purely from AVX2-vs-AVX-512
        // lane counts, against an unchanged GPU. The old fixed 1e-3 was set when k <= 6912 and was
        // not revisited when #207/#211 added the 9216 and 13312 shapes.
        //
        // Measured max|diff| / sqrt(k) across the four shapes is 7.6e-5 .. 1.5e-4, so 4e-4 leaves
        // ~2.7x headroom on the worst case. That still discriminates hard: a real defect (bad block
        // indexing, shared-memory overflow — the #207 bug) perturbs results by O(|y|), which is
        // thousands of times above this bound, not a few times.
        float maxTol = 4e-4f * MathF.Sqrt(k);
        float meanTol = 1e-4f * MathF.Sqrt(k);
        Assert.True(maxDiff <= maxTol, $"max abs diff {maxDiff} exceeds {maxTol} (4e-4·√k, k={k})");
        Assert.True(meanDiff <= meanTol, $"mean abs diff {meanDiff} exceeds {meanTol} (1e-4·√k, k={k})");
    }

    /// <summary>
    /// FP16-in/out twin of <see cref="Run"/>, exercising <see cref="CudaKernels.LaunchI2_SGemvF16In"/> —
    /// the kernel <c>CudaTransformerModel.Project</c> actually dispatches to for decode-time I2_S
    /// projections (issue #207: this is specifically the kernel the FFN-down projection hits, with
    /// k = intermediateSize). FP16 output rounding widens the tolerance vs the FP32 twin.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunF16(int n, int k)
    {
        var rng = new Random(5678);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        float[] xf = new float[k];
        for (int i = 0; i < k; i++) xf[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        Half[] x = new Half[k];
        for (int i = 0; i < k; i++) x[i] = (Half)xf[i];

        // CPU reference (full float precision — the FP16 GPU result is compared against this
        // with a widened tolerance for the F16 input/output rounding).
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = xf, py = cpu)
            MatMul.GemvI2_S(w, px, py, n, k, null);

        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (Half* px = x)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(ushort))).ThrowOnError();

                // Pre-fix, this call staged x into a fixed 6912-float STATIC shared array — for
                // k > 6912 (both cases here) that overflowed the buffer, corrupting shared memory
                // and typically faulting with CUDA error 700 (illegal memory access) either on this
                // launch or the Synchronize() below.
                kernels.LaunchI2_SGemvF16In(devW, devX, devY, n, k, s);
                stream.Synchronize();

                gpu = new Half[n];
                fixed (Half* py = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)py, devY, (nuint)((long)n * sizeof(ushort))).ThrowOnError();
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
            float d = MathF.Abs(cpu[i] - (float)gpu[i]);
            sumDiff += d;
            if (d > maxDiff) maxDiff = d;
            Assert.False(float.IsNaN((float)gpu[i]) || float.IsInfinity((float)gpu[i]),
                $"index {i}: non-finite GPU output {(float)gpu[i]} (would indicate leftover shared-memory corruption)");
        }
        float meanDiff = sumDiff / n;
        _out.WriteLine($"I2_S GEMV F16 {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        Assert.True(maxDiff <= 5e-2f, $"max abs diff {maxDiff} exceeds 5e-2 (CPU fp32 vs GPU fp16)");
        Assert.True(meanDiff <= 5e-3f, $"mean abs diff {meanDiff} exceeds 5e-3");
    }

    [SkippableTheory]
    [InlineData(2560, 2560)]  // attention projection
    [InlineData(2560, 6912)]  // FFN down (k = 6912)
    public void I2SGemvA8_MatchesInt8Reference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunA8(n, k);
    }

    [SkippableFact]
    public void I2SGemvFusedDecode_MatchesSeparateLaunches()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunFusedDecode();
    }

    /// <summary>
    /// Validates the prefill-path dequant kernel (<see cref="CudaKernels.LaunchDequantI2_SToF16"/>,
    /// v2: one warp per 128-element block, coalesced) against the CPU dequant reference
    /// (<see cref="Dequantize.ToFloat32"/>).
    /// </summary>
    [SkippableTheory]
    [InlineData(2560, 2560)]
    [InlineData(37, 2560)]   // non-round n — exercises the warp-grid-stride tail
    public void DequantI2_SToF16_MatchesCpuReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunDequant(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunDequant(int n, int k)
    {
        var rng = new Random(3456);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        // CPU reference — dequantizes the whole [n, k] matrix via the shared CPU codepath.
        float[] cpu = new float[(long)n * k];
        fixed (byte* w = packed)
            Dequantize.ToFloat32((nint)w, (long)n * k, DotLLM.Core.Configuration.QuantizationType.I2_S, cpu);

        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devDst, (nuint)((long)n * k * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();

                kernels.LaunchDequantI2_SToF16(devW, devDst, n, k, s);
                stream.Synchronize();

                gpu = new Half[(long)n * k];
                fixed (Half* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devDst, (nuint)((long)n * k * sizeof(ushort))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
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
        _out.WriteLine($"I2_S dequant {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        Assert.True(maxAbsDiff <= 5e-4f, $"max abs diff {maxAbsDiff} exceeds 5e-4 (F16 rounding vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-4f, $"mean abs diff {meanAbsDiff} exceeds 1e-4");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunFusedDecode()
    {
        const int k = 256;
        const int n0 = 37;
        const int n1 = 53;
        const int n2 = 71;

        var rng = new Random(2468);
        byte[] w0 = RandomPacked(rng, n0, k, 0.021f);
        byte[] w1 = RandomPacked(rng, n1, k, 0.034f);
        byte[] w2 = RandomPacked(rng, n2, k, 0.047f);

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.5f);

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        nint s = stream.Handle;

        static nuint PackedBytes(int n, int k) => (nuint)((long)n * (k / 4) + 4);
        CudaDriverApi.cuMemAlloc_v2(out nint devW0, PackedBytes(n0, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW1, PackedBytes(n1, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devW2, PackedBytes(n2, k)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint sep2, (nuint)(n2 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus0, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus1, (nuint)(n1 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint fus2, (nuint)(n2 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint normW, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint normX, (nuint)(k * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint normSep, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint normFus, (nuint)(n0 * sizeof(ushort))).ThrowOnError();
        try
        {
            Half[] norm = new Half[k];
            for (int i = 0; i < k; i++)
                norm[i] = (Half)(0.5f + (float)rng.NextDouble());

            fixed (byte* p = w0) CudaDriverApi.cuMemcpyHtoD_v2(devW0, (nint)p, PackedBytes(n0, k)).ThrowOnError();
            fixed (byte* p = w1) CudaDriverApi.cuMemcpyHtoD_v2(devW1, (nint)p, PackedBytes(n1, k)).ThrowOnError();
            fixed (byte* p = w2) CudaDriverApi.cuMemcpyHtoD_v2(devW2, (nint)p, PackedBytes(n2, k)).ThrowOnError();
            fixed (Half* p = x) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)p, (nuint)(k * sizeof(ushort))).ThrowOnError();
            fixed (Half* p = norm) CudaDriverApi.cuMemcpyHtoD_v2(normW, (nint)p, (nuint)(k * sizeof(ushort))).ThrowOnError();

            kernels.LaunchI2_SGemvF16In(devW0, devX, sep0, n0, k, s);
            kernels.LaunchI2_SGemvF16In(devW1, devX, sep1, n1, k, s);
            kernels.LaunchI2_SGemvF16In(devW2, devX, sep2, n2, k, s);
            kernels.LaunchI2_SGemv2F16In(devW0, devW1, devX, fus0, fus1, n0, n1, k, s);
            kernels.LaunchI2_SGemv3F16In(devW0, devW1, devW2, devX, fus0, fus1, fus2, n0, n1, n2, k, s);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(normX, devX, (nuint)(k * sizeof(ushort)), s).ThrowOnError();
            kernels.LaunchRmsNorm(normX, normW, normX, k, 1e-5f, 1, s);
            kernels.LaunchI2_SGemvF16In(devW0, normX, normSep, n0, k, s);
            kernels.LaunchI2_SGemvNormF16In(devW0, devX, normW, normFus, n0, k, 1e-5f, s);
            stream.Synchronize();

            AssertHalfClose(sep0, fus0, n0);
            AssertHalfClose(sep1, fus1, n1);
            AssertHalfClose(sep2, fus2, n2);
            AssertHalfClose(normSep, normFus, n0);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW0); CudaDriverApi.cuMemFree_v2(devW1); CudaDriverApi.cuMemFree_v2(devW2);
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(sep0); CudaDriverApi.cuMemFree_v2(sep1); CudaDriverApi.cuMemFree_v2(sep2);
            CudaDriverApi.cuMemFree_v2(fus0); CudaDriverApi.cuMemFree_v2(fus1); CudaDriverApi.cuMemFree_v2(fus2);
            CudaDriverApi.cuMemFree_v2(normW); CudaDriverApi.cuMemFree_v2(normX);
            CudaDriverApi.cuMemFree_v2(normSep); CudaDriverApi.cuMemFree_v2(normFus);
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

    private static byte[] RandomPacked(Random rng, int n, int k, float scale)
    {
        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++)
            ternary[i] = (sbyte)(rng.Next(3) - 1);
        return Pack(ternary, n, k, scale);
    }

    /// <summary>
    /// Validates Variant B (W2A8, __dp4a) against an INT8 reference computed on the CPU.
    ///
    /// The kernel quantizes nothing itself — activations are quantized on the host (per-token symmetric
    /// absmax, s_act = 127/absmax, xq = round(x·s_act)), and the kernel computes the exact int32 dot
    /// Σ xq_i·(code_i−1), scaled by weightScale·invActScale (invActScale = absmax/127 = 1/s_act). The CPU
    /// int8 reference does the SAME integer dot with the SAME int8 inputs, so the two must match to
    /// integer-math precision (the only float work is the final scale·invActScale multiply + the fp32
    /// reduction, hence ≤ 1e-4, not bit-exact).
    ///
    /// NOTE: the dp4a result vs the *float* CPU reference (MatMul.GemvI2_S over full-precision x) will
    /// differ — that gap is the expected per-token activation-quant error, NOT a kernel bug. We therefore
    /// assert against the int8 reference here, exactly as the spec (§2a, §7 row B0) prescribes.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunA8(int n, int k)
    {
        var rng = new Random(4321);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = Pack(ternary, n, k, scale);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // Host per-token int8 quant: symmetric absmax. s_act = 127/absmax, invActScale = absmax/127.
        float absmax = 0f;
        for (int i = 0; i < k; i++) absmax = MathF.Max(absmax, MathF.Abs(x[i]));
        if (absmax == 0f) absmax = 1f;            // degenerate guard (matches a real quantizer)
        float sAct = 127f / absmax;
        float invActScale = absmax / 127f;
        sbyte[] xq = new sbyte[k];
        for (int i = 0; i < k; i++)
        {
            int q = (int)MathF.Round(x[i] * sAct);
            if (q > 127) q = 127;
            if (q < -127) q = -127;               // symmetric range, mirror BitNet
            xq[i] = (sbyte)q;
        }

        // INT8 CPU reference: out = scale · invActScale · Σ xq_i·(code_i−1). Exact integer dot in long,
        // then the single float epilogue — mirrors the kernel's math.
        float[] cpu = new float[n];
        for (int r = 0; r < n; r++)
        {
            long iacc = 0;
            long rowOff = (long)r * k;
            for (int col = 0; col < k; col++)
                iacc += (long)xq[col] * ternary[rowOff + col]; // ternary already == code-1
            cpu[r] = (float)iacc * scale * invActScale;
        }

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
            CudaDriverApi.cuMemAlloc_v2(out nint devX, (nuint)((long)k * sizeof(sbyte))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devY, (nuint)((long)n * sizeof(float))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
                fixed (sbyte* px = xq)
                    CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)k * sizeof(sbyte))).ThrowOnError();

                kernels.LaunchI2_SGemvA8(devW, devX, devY, n, k, invActScale, s);
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
        _out.WriteLine($"I2_S A8 GEMV {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        // Integer math on both sides → only the float epilogue/reduction differs. Tight tolerance.
        Assert.True(maxDiff <= 1e-4f, $"max abs diff {maxDiff} exceeds 1e-4 (dp4a vs int8 reference)");
        Assert.True(meanDiff <= 1e-5f, $"mean abs diff {meanDiff} exceeds 1e-5");
    }

    // ──────────────────── Ragged K (k % 128 != 0) — issue #206 ────────────────────
    //
    // The aligned kernels above assume k % 128 == 0 (uint4/shared-block addressing) and crash
    // with "CUDA error 716: misaligned address" otherwise. i2_s_gemv_f32in_ragged /
    // dequant_i2_s_f16_ragged are the correctness-first fallbacks reached when k is not
    // 128-aligned (e.g. bitnet_b1_58-xl's ffn_down: hidden=2048, intermediate=5460).

    [SkippableTheory]
    [InlineData(16, 200)]   // small synthetic ragged k (200 % 128 == 72)
    [InlineData(32, 5460)]  // real bitnet_b1_58-xl ffn_down shape; m=32 covers all 32 distinct
                             // row-start bit-phases for k=5460 (gcd(5460,128)=4 -> period 32)
    public void I2SGemvF32InRagged_MatchesCpuFloatReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunRagged(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunRagged(int n, int k)
    {
        Assert.NotEqual(0, k % 128); // sanity: this test exists specifically for ragged k
        Assert.Equal(0, ((long)n * k) % 128); // PackRagged's exact-byte-sizing precondition

        var rng = new Random(1234);
        long total = (long)n * k;
        long packedLen = total / 4 + 4;

        sbyte[] ternary = new sbyte[total];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = PackRagged(ternary, n, k, scale);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;

        // CPU reference (now ragged-safe — MatMul.GemvI2_S dispatches to the ragged path itself).
        float[] cpu = new float[n];
        fixed (byte* w = packed)
        fixed (float* px = x, py = cpu)
            MatMul.GemvI2_S(w, px, py, n, k, null);

        // GPU (ragged kernel).
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

                kernels.LaunchI2_SGemvF32InRagged(devW, devX, devY, n, k, s);
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
        _out.WriteLine($"I2_S GEMV ragged {n}×{k}: max abs diff={maxDiff:E4}, mean={meanDiff:E4}");

        // Same sqrt(k) scaling as Run() — see the rationale there. The ragged shapes are small
        // enough that the old fixed bound still held, but the tolerance must track k for the same
        // reason, otherwise adding a larger ragged case later silently re-introduces the failure.
        float maxTol = 4e-4f * MathF.Sqrt(k);
        float meanTol = 1e-4f * MathF.Sqrt(k);
        Assert.True(maxDiff <= maxTol, $"max abs diff {maxDiff} exceeds {maxTol} (4e-4·√k, k={k})");
        Assert.True(meanDiff <= meanTol, $"mean abs diff {meanDiff} exceeds {meanTol} (1e-4·√k, k={k})");
    }

    [SkippableTheory]
    [InlineData(16, 200)]
    [InlineData(32, 5460)]
    public void DequantI2_SToF16Ragged_MatchesCpuReference(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunDequantRagged(n, k);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunDequantRagged(int n, int k)
    {
        Assert.NotEqual(0, k % 128);
        Assert.Equal(0, ((long)n * k) % 128);

        var rng = new Random(3456);
        long total = (long)n * k;
        long packedLen = total / 4 + 4;

        sbyte[] ternary = new sbyte[total];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float scale = 0.02f + (float)rng.NextDouble() * 0.03f;
        byte[] packed = PackRagged(ternary, n, k, scale);

        // CPU reference — DequantizeI2_S already treats elementCount as the flattened total, so
        // it's already ragged-correct as long as the TOTAL element count is 128-aligned (which we
        // assert above); no CPU-side change was needed for this path.
        float[] cpu = new float[total];
        fixed (byte* w = packed)
            Dequantize.ToFloat32((nint)w, total, DotLLM.Core.Configuration.QuantizationType.I2_S, cpu);

        Half[] gpu;
        {
            using var ctx = CudaContext.Create(0);
            using var stream = CudaStream.Create();
            string? ptxDir = FindPtxDir();
            Skip.If(ptxDir == null, "PTX files not found");
            using var kernels = new CudaKernels(ptxDir!);
            nint s = stream.Handle;

            CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint devDst, (nuint)(total * sizeof(ushort))).ThrowOnError();
            try
            {
                fixed (byte* w = packed)
                    CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();

                kernels.LaunchDequantI2_SToF16Ragged(devW, devDst, n, k, s);
                stream.Synchronize();

                gpu = new Half[total];
                fixed (Half* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devDst, (nuint)(total * sizeof(ushort))).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(devW);
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
        _out.WriteLine($"I2_S dequant ragged {n}×{k}: max abs diff={maxAbsDiff:E4}, mean={meanAbsDiff:E4}");

        Assert.True(maxAbsDiff <= 5e-4f, $"max abs diff {maxAbsDiff} exceeds 5e-4 (F16 rounding vs CPU float32)");
        Assert.True(meanAbsDiff <= 1e-4f, $"mean abs diff {meanAbsDiff} exceeds 1e-4");
    }

    /// <summary>
    /// Microbenchmark: Variant A (W2A16, FP16 activations) vs Variant B (W2A8, dp4a int8 activations)
    /// at the GEMV level on the real BitNet projection shapes. Informational — always passes; run with
    /// <c>--logger "console;verbosity=detailed"</c> to see the timings. Decides which variant the
    /// forward integration should dispatch.
    /// </summary>
    [SkippableFact]
    public void Benchmark_I2SGemv_AvsB()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Bench(kernels, stream, 2560, 2560);
        Bench(kernels, stream, 2560, 6912);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void Bench(CudaKernels kernels, CudaStream stream, int n, int k)
    {
        const int Iters = 500, Warmup = 30;
        var rng = new Random(7);
        int rowBytes = k / 4;
        long packedLen = (long)n * rowBytes + 4;

        sbyte[] ternary = new sbyte[(long)n * k];
        for (long i = 0; i < ternary.LongLength; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        byte[] packed = Pack(ternary, n, k, 0.03f);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        Half[] xh = new Half[k];
        for (int i = 0; i < k; i++) xh[i] = (Half)x[i];
        float absmax = 0f; for (int i = 0; i < k; i++) absmax = MathF.Max(absmax, MathF.Abs(x[i]));
        float invAct = absmax / 127f;
        sbyte[] xq = new sbyte[k];
        for (int i = 0; i < k; i++) xq[i] = (sbyte)Math.Clamp((int)MathF.Round(x[i] * 127f / absmax), -127, 127);

        nint s = stream.Handle;
        CudaDriverApi.cuMemAlloc_v2(out nint devW, (nuint)packedLen).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devXh, (nuint)((long)k * 2)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devXq, (nuint)k).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devYh, (nuint)((long)n * 2)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devYf, (nuint)((long)n * 4)).ThrowOnError();
        try
        {
            fixed (byte* w = packed) CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packedLen).ThrowOnError();
            fixed (Half* p = xh) CudaDriverApi.cuMemcpyHtoD_v2(devXh, (nint)p, (nuint)((long)k * 2)).ThrowOnError();
            fixed (sbyte* p = xq) CudaDriverApi.cuMemcpyHtoD_v2(devXq, (nint)p, (nuint)k).ThrowOnError();

            for (int i = 0; i < Warmup; i++) kernels.LaunchI2_SGemvF16In(devW, devXh, devYh, n, k, s);
            stream.Synchronize();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            for (int i = 0; i < Iters; i++) kernels.LaunchI2_SGemvF16In(devW, devXh, devYh, n, k, s);
            stream.Synchronize(); sw.Stop();
            double aUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;

            for (int i = 0; i < Warmup; i++) kernels.LaunchI2_SGemvA8(devW, devXq, devYf, n, k, invAct, s);
            stream.Synchronize();
            sw.Restart();
            for (int i = 0; i < Iters; i++) kernels.LaunchI2_SGemvA8(devW, devXq, devYf, n, k, invAct, s);
            stream.Synchronize(); sw.Stop();
            double bUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;

            double wBytes = (double)n * rowBytes; // packed weight bytes read per GEMV
            _out.WriteLine($"I2_S GEMV {n}x{k}:  A(W2A16)={aUs:F1}us {wBytes / aUs / 1e3:F1}GB/s   " +
                           $"B(dp4a)={bUs:F1}us {wBytes / bUs / 1e3:F1}GB/s   B/A={bUs / aUs:F2}x");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devW); CudaDriverApi.cuMemFree_v2(devXh); CudaDriverApi.cuMemFree_v2(devXq);
            CudaDriverApi.cuMemFree_v2(devYh); CudaDriverApi.cuMemFree_v2(devYf);
        }
    }

    /// <summary>Packs ternary {-1,0,+1} into dotLLM I2_S layout + trailing per-tensor float32 scale.</summary>
    private static byte[] Pack(sbyte[] ternary, int n, int k, float scale)
    {
        int rowBytes = k / 4;
        byte[] buf = new byte[(long)n * rowBytes + 4];
        for (int r = 0; r < n; r++)
        {
            long rowBase = (long)r * rowBytes;
            for (int col = 0; col < k; col++)
            {
                int blk = col / 128, j = col % 128, gi = j / 32, gp = j % 32;
                int code = ternary[(long)r * k + col] + 1; // {-1,0,+1} → {0,1,2}
                buf[rowBase + blk * 32 + gp] |= (byte)(code << (6 - 2 * gi));
            }
        }
        BitConverter.GetBytes(scale).CopyTo(buf, (int)((long)n * rowBytes));
        return buf;
    }

    /// <summary>
    /// Ragged-K packer (issue #206): unlike <see cref="Pack"/> above, block boundaries are NOT
    /// reset per row — the block interleave is computed over the FLATTENED n*k element stream
    /// (matching the real on-disk layout / upstream bitnet.cpp writer; see MatMul.I2S.cs's class
    /// remarks and I2STests.PackI2S, the CPU-side twin of this helper). For aligned k the two
    /// packers are bit-identical (every row boundary is also a block boundary); they only diverge
    /// when k % 128 != 0, which is exactly the case this helper exists to test.
    /// </summary>
    private static byte[] PackRagged(sbyte[] ternary, int n, int k, float scale)
    {
        long total = (long)n * k;
        byte[] buf = new byte[total / 4 + 4];
        for (long e = 0; e < total; e++)
        {
            long block = e / 128;
            int j = (int)(e % 128), gi = j / 32, gp = j % 32;
            int code = ternary[e] + 1;
            buf[block * 32 + gp] |= (byte)(code << (6 - 2 * gi));
        }
        BitConverter.GetBytes(scale).CopyTo(buf, (int)(total / 4));
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
