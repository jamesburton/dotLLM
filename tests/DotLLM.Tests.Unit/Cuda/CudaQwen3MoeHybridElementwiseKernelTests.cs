using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness tests for the four fused-op FP32 kernels backing the
/// Qwen3MoeHybrid host-fallback replacement:
///   • gdn_decay_f32     — fused softplus + exp for the per-token decay g.
///   • sigmoid_f32       — in-place elementwise sigmoid.
///   • silu_f32          — in-place elementwise SiLU.
///   • sigmoid_mul_f32   — out[i] *= sigmoid(b[i]).
/// Each test synthesises a random F32 input, computes the CPU reference using
/// the same scalar arithmetic (<see cref="MathF.Exp"/>, <see cref="MathF.Log"/>),
/// runs the CUDA kernel, and compares.
///
/// <para>
/// <b>Tolerance.</b> The kernels are compiled with <c>-fmad=false</c> to disable
/// FMA fusion, but each calls <c>expf</c> / <c>logf</c> — and CUDA's precise
/// expf/logf are not bit-equal to .NET's <see cref="MathF.Exp"/> / <see cref="MathF.Log"/>
/// in every input. The accepted tolerance is therefore ≤ 4 ULP (chosen
/// empirically — observed peak on Ampere is ≤ 2 ULP for the basic transcendentals
/// and ≤ 4 ULP after a multiply-then-exp composition).
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public class CudaQwen3MoeHybridElementwiseKernelTests
{
    private readonly ITestOutputHelper _out;
    public CudaQwen3MoeHybridElementwiseKernelTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    /// <summary>
    /// Search for the PTX directory next to the test assembly (preferred,
    /// because the csproj copies <c>native/ptx/*.ptx</c> into the test output)
    /// or walk back up to the repo root and use the canonical <c>native/ptx/</c>.
    /// </summary>
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

    /// <summary>
    /// Tolerance used by all four kernel tests. ≤ 4 ULP is the empirical bound
    /// observed across the input ranges these tests use (uniform [-4, 4]); the
    /// theoretical bound for CUDA precise expf vs MathF.Exp is 1 ULP, plus
    /// another ~2 ULP for the surrounding multiply/divide.
    /// </summary>
    private const int MaxUlpDiff = 4;

    /// <summary>
    /// ULP distance between two FP32 values. Standard reinterpret-as-int trick;
    /// negatives are sign-magnitude so we transform them into a continuous
    /// ordered space first. Returns long.MaxValue on NaN to keep the test
    /// asserting in the face of pathological inputs (which shouldn't appear in
    /// the synthetic data anyway).
    /// </summary>
    private static long UlpDistance(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b)) return long.MaxValue;
        if (a == b) return 0;
        int ai = BitConverter.SingleToInt32Bits(a);
        int bi = BitConverter.SingleToInt32Bits(b);
        // Map sign-magnitude to a continuous monotonic ordering: flip the sign
        // bit on positives, invert negatives so they sort below positives.
        if (ai < 0) ai = int.MinValue - ai;
        if (bi < 0) bi = int.MinValue - bi;
        return Math.Abs((long)ai - (long)bi);
    }

    // ── Test 1: gdn_decay_f32 ───────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(1, 16)]
    [InlineData(8, 64)]
    [InlineData(64, 32)]   // longer sequence x typical Qwen3MoeHybrid n_v_head
    public void GdnDecayF32_MatchesCpuReference(int seqLen, int nVHead)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasGdnDecayF32, "gdn_decay_f32 not loaded (PTX may be stale)");

        // Synthetic inputs. alpha in [-4, 4] keeps softplus moderate (max ~4.018);
        // dt_bias in [-1, 1]; A in [-0.5, -0.05] roughly matches the empirical
        // distribution of negative-slope decay coefficients in real models.
        var rng = new Random(unchecked((int)0xCAFEFACE) ^ seqLen ^ (nVHead << 8));
        float[] alphaIn = new float[seqLen * nVHead];
        float[] dtBias = new float[nVHead];
        float[] aCoef = new float[nVHead];
        for (int i = 0; i < alphaIn.Length; i++) alphaIn[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        for (int h = 0; h < nVHead; h++)
        {
            dtBias[h] = (float)(rng.NextDouble() * 2.0 - 1.0);
            aCoef[h] = (float)(-rng.NextDouble() * 0.45 - 0.05);
        }

        // CPU reference — same arithmetic as the host fallback at
        // CudaQwen3MoeHybridTransformerModel.LaunchGdnDecayHostFallback.
        float[] cpu = new float[alphaIn.Length];
        for (int t = 0; t < seqLen; t++)
        {
            int off = t * nVHead;
            for (int h = 0; h < nVHead; h++)
            {
                float alpha = alphaIn[off + h] + dtBias[h];
                float sp = MathF.Log(1f + MathF.Exp(alpha));
                cpu[off + h] = MathF.Exp(sp * aCoef[h]);
            }
        }

        // GPU
        nint dAlpha = 0, dDt = 0, dA = 0;
        try
        {
            long aBytes = (long)alphaIn.Length * sizeof(float);
            long hBytes = (long)nVHead * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dAlpha, (nuint)aBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)hBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)hBytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = alphaIn)
                    CudaDriverApi.cuMemcpyHtoD_v2(dAlpha, (nint)p, (nuint)aBytes).ThrowOnError();
                fixed (float* p = dtBias)
                    CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)hBytes).ThrowOnError();
                fixed (float* p = aCoef)
                    CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)hBytes).ThrowOnError();
            }

            kernels.LaunchGdnDecayF32(dAlpha, dDt, dA, seqLen, nVHead, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[alphaIn.Length];
            unsafe
            {
                fixed (float* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dAlpha, (nuint)aBytes).ThrowOnError();
            }

            AssertWithinUlp(cpu, gpu, "gdn_decay_f32");
        }
        finally
        {
            if (dAlpha != 0) CudaDriverApi.cuMemFree_v2(dAlpha);
            if (dDt != 0) CudaDriverApi.cuMemFree_v2(dDt);
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
        }
    }

    // ── Test 2: sigmoid_f32 ─────────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(7)]
    [InlineData(256)]
    [InlineData(4096)]
    public void SigmoidF32_MatchesCpuReference(int n)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasElementwiseF32, "elementwise_f32 PTX not loaded");

        var rng = new Random(0xBAD0BAD ^ n);
        float[] input = new float[n];
        for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 8.0 - 4.0);

        // CPU oracle — matches the host fallback line-for-line.
        float[] cpu = new float[n];
        for (int i = 0; i < n; i++)
            cpu[i] = 1f / (1f + MathF.Exp(-input[i]));

        nint dBuf = 0;
        try
        {
            long bytes = (long)n * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dBuf, (nuint)bytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = input)
                    CudaDriverApi.cuMemcpyHtoD_v2(dBuf, (nint)p, (nuint)bytes).ThrowOnError();
            }

            kernels.LaunchSigmoidF32(dBuf, n, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[n];
            unsafe
            {
                fixed (float* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dBuf, (nuint)bytes).ThrowOnError();
            }

            AssertWithinUlp(cpu, gpu, "sigmoid_f32");
        }
        finally
        {
            if (dBuf != 0) CudaDriverApi.cuMemFree_v2(dBuf);
        }
    }

    // ── Test 3: silu_f32 ────────────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(7)]
    [InlineData(512)]
    [InlineData(8192)]
    public void SiluF32_MatchesCpuReference(int n)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasElementwiseF32, "elementwise_f32 PTX not loaded");

        var rng = new Random(unchecked((int)0xF00DBA11) ^ n);
        float[] input = new float[n];
        for (int i = 0; i < n; i++) input[i] = (float)(rng.NextDouble() * 8.0 - 4.0);

        float[] cpu = new float[n];
        for (int i = 0; i < n; i++)
        {
            float x = input[i];
            cpu[i] = x * (1f / (1f + MathF.Exp(-x)));
        }

        nint dBuf = 0;
        try
        {
            long bytes = (long)n * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dBuf, (nuint)bytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = input)
                    CudaDriverApi.cuMemcpyHtoD_v2(dBuf, (nint)p, (nuint)bytes).ThrowOnError();
            }

            kernels.LaunchSiluF32(dBuf, n, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[n];
            unsafe
            {
                fixed (float* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dBuf, (nuint)bytes).ThrowOnError();
            }

            AssertWithinUlp(cpu, gpu, "silu_f32");
        }
        finally
        {
            if (dBuf != 0) CudaDriverApi.cuMemFree_v2(dBuf);
        }
    }

    // ── Test 4: sigmoid_mul_f32 ─────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(7)]
    [InlineData(256)]
    [InlineData(8192)]
    public void SigmoidMulF32_MatchesCpuReference(int n)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasElementwiseF32, "elementwise_f32 PTX not loaded");

        var rng = new Random(unchecked((int)0xABCDEF01) ^ n);
        float[] aIn = new float[n];
        float[] bIn = new float[n];
        for (int i = 0; i < n; i++)
        {
            aIn[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
            bIn[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        }

        // CPU oracle — matches the host fallback line-for-line.
        float[] cpu = new float[n];
        for (int i = 0; i < n; i++)
        {
            float bi = bIn[i];
            cpu[i] = aIn[i] * (1f / (1f + MathF.Exp(-bi)));
        }

        nint dA = 0, dB = 0;
        try
        {
            long bytes = (long)n * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dB, (nuint)bytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = aIn)
                    CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)bytes).ThrowOnError();
                fixed (float* p = bIn)
                    CudaDriverApi.cuMemcpyHtoD_v2(dB, (nint)p, (nuint)bytes).ThrowOnError();
            }

            kernels.LaunchSigmoidMulF32(dA, dB, n, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[n];
            unsafe
            {
                fixed (float* p = gpu)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dA, (nuint)bytes).ThrowOnError();
            }

            AssertWithinUlp(cpu, gpu, "sigmoid_mul_f32");
        }
        finally
        {
            if (dA != 0) CudaDriverApi.cuMemFree_v2(dA);
            if (dB != 0) CudaDriverApi.cuMemFree_v2(dB);
        }
    }

    /// <summary>
    /// Compare CPU oracle to GPU output element-wise and assert the worst-case
    /// distance is within <see cref="MaxUlpDiff"/>. Prints diagnostics on
    /// failure to aid maintenance: position, both values, ulp distance,
    /// absolute and relative deltas.
    /// </summary>
    private void AssertWithinUlp(float[] cpu, float[] gpu, string kernel)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        long maxUlp = 0;
        int maxIdx = -1;
        float maxAbs = 0f;
        for (int i = 0; i < cpu.Length; i++)
        {
            long ulp = UlpDistance(cpu[i], gpu[i]);
            if (ulp > maxUlp)
            {
                maxUlp = ulp;
                maxIdx = i;
            }
            float abs = MathF.Abs(cpu[i] - gpu[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        _out.WriteLine($"{kernel}: n={cpu.Length}, max-ulp={maxUlp} at {maxIdx} " +
                        $"(cpu={(maxIdx < 0 ? 0f : cpu[maxIdx])}, gpu={(maxIdx < 0 ? 0f : gpu[maxIdx])}), max-abs={maxAbs:E3}");

        Assert.True(maxUlp <= MaxUlpDiff,
            $"{kernel}: max ULP {maxUlp} exceeds tolerance {MaxUlpDiff}. " +
            $"At [{maxIdx}] cpu={(maxIdx < 0 ? 0f : cpu[maxIdx])} gpu={(maxIdx < 0 ? 0f : gpu[maxIdx])}.");
    }
}
