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

    // ── Test 1b: gdn_decay_sigmoid_f32 (fused) ───────────────────────────────

    [SkippableTheory]
    [InlineData(1, 48)]   // decode shape: seqLen=1, real Bonsai nVHead
    [InlineData(5, 48)]   // prefill shape
    public void GdnDecaySigmoidF32_MatchesSeparateDecayPlusSigmoid(int seqLen, int nVHead)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasGdnDecaySigmoidF32, "gdn_decay_sigmoid_f32 not loaded (PTX may be stale)");

        var rng = new Random(0x5EED ^ seqLen ^ (nVHead << 8));
        float[] alphaIn = new float[seqLen * nVHead];
        float[] betaIn = new float[seqLen * nVHead];
        float[] dtBias = new float[nVHead];
        float[] aCoef = new float[nVHead];
        for (int i = 0; i < alphaIn.Length; i++) alphaIn[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        for (int i = 0; i < betaIn.Length; i++) betaIn[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        for (int h = 0; h < nVHead; h++)
        {
            dtBias[h] = (float)(rng.NextDouble() * 2.0 - 1.0);
            aCoef[h] = (float)(-rng.NextDouble() * 0.45 - 0.05);
        }

        nint dAlphaSep = 0, dBetaSep = 0, dAlphaFused = 0, dBetaFused = 0, dDt = 0, dA = 0;
        try
        {
            long aBytes = (long)alphaIn.Length * sizeof(float);
            long hBytes = (long)nVHead * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dAlphaSep, (nuint)aBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dBetaSep, (nuint)aBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dAlphaFused, (nuint)aBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dBetaFused, (nuint)aBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDt, (nuint)hBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dA, (nuint)hBytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = alphaIn)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(dAlphaSep, (nint)p, (nuint)aBytes).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD_v2(dAlphaFused, (nint)p, (nuint)aBytes).ThrowOnError();
                }
                fixed (float* p = betaIn)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(dBetaSep, (nint)p, (nuint)aBytes).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD_v2(dBetaFused, (nint)p, (nuint)aBytes).ThrowOnError();
                }
                fixed (float* p = dtBias) CudaDriverApi.cuMemcpyHtoD_v2(dDt, (nint)p, (nuint)hBytes).ThrowOnError();
                fixed (float* p = aCoef) CudaDriverApi.cuMemcpyHtoD_v2(dA, (nint)p, (nuint)hBytes).ThrowOnError();
            }

            nint s = stream.Handle;
            kernels.LaunchGdnDecayF32(dAlphaSep, dDt, dA, seqLen, nVHead, s);
            kernels.LaunchSigmoidF32(dBetaSep, (long)seqLen * nVHead, s);
            kernels.LaunchGdnDecaySigmoidF32(dAlphaFused, dBetaFused, dDt, dA, seqLen, nVHead, s);
            stream.Synchronize();

            float[] alphaSep = new float[alphaIn.Length], betaSep = new float[alphaIn.Length];
            float[] alphaFused = new float[alphaIn.Length], betaFused = new float[alphaIn.Length];
            unsafe
            {
                fixed (float* p = alphaSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dAlphaSep, (nuint)aBytes).ThrowOnError();
                fixed (float* p = betaSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dBetaSep, (nuint)aBytes).ThrowOnError();
                fixed (float* p = alphaFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dAlphaFused, (nuint)aBytes).ThrowOnError();
                fixed (float* p = betaFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dBetaFused, (nuint)aBytes).ThrowOnError();
            }

            Assert.Equal(alphaSep, alphaFused);
            Assert.Equal(betaSep, betaFused);
            _out.WriteLine($"gdn_decay_sigmoid_f32 seqLen={seqLen} nVHead={nVHead}: exact match vs separate calls");
        }
        finally
        {
            if (dAlphaSep != 0) CudaDriverApi.cuMemFree_v2(dAlphaSep);
            if (dBetaSep != 0) CudaDriverApi.cuMemFree_v2(dBetaSep);
            if (dAlphaFused != 0) CudaDriverApi.cuMemFree_v2(dAlphaFused);
            if (dBetaFused != 0) CudaDriverApi.cuMemFree_v2(dBetaFused);
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

    // ── Test 5: deinterleave_qgate_f32 ─────────────────────────────────────
    // Pure gather (no floating-point math) — asserts EXACT equality, not ULP.

    [SkippableTheory]
    [InlineData(1, 4, 8)]     // decode shape: seqLen=1
    [InlineData(3, 40, 128)]  // multi-token, real Bonsai-ish head count/dim
    public void DeinterleaveQGateF32_MatchesCpuReference(int seqLen, int numHeads, int headDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasDeinterleaveF32, "deinterleave kernels not loaded (PTX may be stale)");

        var rng = new Random(0xDEEE ^ seqLen ^ (numHeads << 8) ^ (headDim << 16));
        int qElems = numHeads * headDim;
        float[] qg = new float[seqLen * 2 * qElems];
        for (int i = 0; i < qg.Length; i++) qg[i] = (float)rng.NextDouble();

        // CPU reference — same per-head [Q(headDim) | Gate(headDim)] interleave the
        // host-loop fallback (and the original per-model host loop) assumed.
        float[] cpuQ = new float[seqLen * qElems];
        float[] cpuGate = new float[seqLen * qElems];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < numHeads; h++)
                for (int d = 0; d < headDim; d++)
                {
                    int qgBase = t * 2 * qElems + h * 2 * headDim;
                    cpuQ[t * qElems + h * headDim + d] = qg[qgBase + d];
                    cpuGate[t * qElems + h * headDim + d] = qg[qgBase + headDim + d];
                }

        nint dQg = 0, dQ = 0, dGate = 0;
        try
        {
            long qgBytes = (long)qg.Length * sizeof(float);
            long qBytes = (long)cpuQ.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dQg, (nuint)qgBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dGate, (nuint)qBytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = qg)
                    CudaDriverApi.cuMemcpyHtoD_v2(dQg, (nint)p, (nuint)qgBytes).ThrowOnError();
            }

            kernels.LaunchDeinterleaveQGateF32(dQg, dQ, dGate, numHeads, headDim, seqLen, stream.Handle);
            stream.Synchronize();

            float[] gpuQ = new float[cpuQ.Length];
            float[] gpuGate = new float[cpuGate.Length];
            unsafe
            {
                fixed (float* p = gpuQ)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQ, (nuint)qBytes).ThrowOnError();
                fixed (float* p = gpuGate)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dGate, (nuint)qBytes).ThrowOnError();
            }

            Assert.Equal(cpuQ, gpuQ);
            Assert.Equal(cpuGate, gpuGate);
        }
        finally
        {
            if (dQg != 0) CudaDriverApi.cuMemFree_v2(dQg);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dGate != 0) CudaDriverApi.cuMemFree_v2(dGate);
        }
    }

    // ── Test 6: deinterleave_gdn_qkv_f32 ───────────────────────────────────

    [SkippableTheory]
    [InlineData(1, 2048, 6144)]  // decode shape: seqLen=1, Bonsai-ish kDim/vDim
    [InlineData(3, 16, 24)]      // multi-token, small dims
    public void DeinterleaveGdnQkvF32_MatchesCpuReference(int seqLen, int kDim, int vDim)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasDeinterleaveF32, "deinterleave kernels not loaded (PTX may be stale)");

        var rng = new Random(0xF00D ^ seqLen ^ (kDim << 8) ^ (vDim << 16));
        int convDim = 2 * kDim + vDim;
        float[] src = new float[seqLen * convDim];
        for (int i = 0; i < src.Length; i++) src[i] = (float)rng.NextDouble();

        // CPU reference — per-token [Q(kDim) | K(kDim) | V(vDim)] split.
        float[] cpuQ = new float[seqLen * kDim];
        float[] cpuK = new float[seqLen * kDim];
        float[] cpuV = new float[seqLen * vDim];
        for (int t = 0; t < seqLen; t++)
        {
            int rowBase = t * convDim;
            Array.Copy(src, rowBase, cpuQ, t * kDim, kDim);
            Array.Copy(src, rowBase + kDim, cpuK, t * kDim, kDim);
            Array.Copy(src, rowBase + 2 * kDim, cpuV, t * vDim, vDim);
        }

        nint dSrc = 0, dQ = 0, dK = 0, dV = 0;
        try
        {
            long srcBytes = (long)src.Length * sizeof(float);
            long qkBytes = (long)cpuQ.Length * sizeof(float);
            long vBytes = (long)cpuV.Length * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)srcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = src)
                    CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)srcBytes).ThrowOnError();
            }

            kernels.LaunchDeinterleaveGdnQkvF32(dSrc, dQ, dK, dV, kDim, vDim, seqLen, stream.Handle);
            stream.Synchronize();

            float[] gpuQ = new float[cpuQ.Length];
            float[] gpuK = new float[cpuK.Length];
            float[] gpuV = new float[cpuV.Length];
            unsafe
            {
                fixed (float* p = gpuQ)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQ, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = gpuK)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dK, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = gpuV)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dV, (nuint)vBytes).ThrowOnError();
            }

            Assert.Equal(cpuQ, gpuQ);
            Assert.Equal(cpuK, gpuK);
            Assert.Equal(cpuV, gpuV);
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
        }
    }

    // ── Test 7: gdn_deinterleave_l2norm_decode_f32 (fused, issue #170) ──────

    [SkippableTheory]
    [InlineData(16, 48, 128)]  // decode-realistic: real Bonsai-ish nVHead, d_state=128
    [InlineData(2, 3, 8)]      // small dims
    public void GdnDeinterleaveL2NormDecodeF32_MatchesSeparateDeinterleavePlusL2Norm(
        int nKHead, int nVHead, int dState)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasGdnDeinterleaveL2NormDecodeF32,
            "gdn_deinterleave_l2norm_decode_f32 not loaded (PTX may be stale)");
        Skip.IfNot(kernels.HasDeinterleaveF32, "deinterleave kernels not loaded (PTX may be stale)");

        int kDim = nKHead * dState;
        int vDim = nVHead * dState;
        int convDim = 2 * kDim + vDim;
        const float eps = 1e-6f;

        var rng = new Random(0xFEED ^ nKHead ^ (nVHead << 8) ^ (dState << 16));
        float[] src = new float[convDim];
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        nint dSrc = 0;
        nint dQSep = 0, dKSep = 0, dVSep = 0;
        nint dQFused = 0, dKFused = 0, dVFused = 0;
        try
        {
            long srcBytes = (long)src.Length * sizeof(float);
            long qkBytes = (long)kDim * sizeof(float);
            long vBytes = (long)vDim * sizeof(float);
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)srcBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQSep, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dKSep, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dVSep, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQFused, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dKFused, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dVFused, (nuint)vBytes).ThrowOnError();
            unsafe
            {
                fixed (float* p = src)
                    CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)srcBytes).ThrowOnError();
            }

            nint s = stream.Handle;

            // Separate path (the general seqLen>1-capable path, called with seqLen=1):
            // deinterleave, then L2-normalize Q and K independently.
            kernels.LaunchDeinterleaveGdnQkvF32(dSrc, dQSep, dKSep, dVSep, kDim, vDim, seqLen: 1, s);
            kernels.LaunchL2NormalizeHeadsF32(dQSep, nKHead, dState, eps, s);
            kernels.LaunchL2NormalizeHeadsF32(dKSep, nKHead, dState, eps, s);

            // Fused decode-only path under test.
            kernels.LaunchGdnDeinterleaveL2NormDecodeF32(
                dSrc, dQFused, dKFused, dVFused, nKHead, nVHead, dState, eps, s);

            stream.Synchronize();

            float[] qSep = new float[kDim], kSep = new float[kDim], vSep = new float[vDim];
            float[] qFused = new float[kDim], kFused = new float[kDim], vFused = new float[vDim];
            unsafe
            {
                fixed (float* p = qSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQSep, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = kSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dKSep, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = vSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dVSep, (nuint)vBytes).ThrowOnError();
                fixed (float* p = qFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQFused, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = kFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dKFused, (nuint)qkBytes).ThrowOnError();
                fixed (float* p = vFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dVFused, (nuint)vBytes).ThrowOnError();
            }

            Assert.Equal(qSep, qFused);
            Assert.Equal(kSep, kFused);
            Assert.Equal(vSep, vFused);
            _out.WriteLine(
                $"gdn_deinterleave_l2norm_decode_f32 nKHead={nKHead} nVHead={nVHead} dState={dState}: " +
                "exact match vs deinterleave+2xL2Normalize");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dQSep != 0) CudaDriverApi.cuMemFree_v2(dQSep);
            if (dKSep != 0) CudaDriverApi.cuMemFree_v2(dKSep);
            if (dVSep != 0) CudaDriverApi.cuMemFree_v2(dVSep);
            if (dQFused != 0) CudaDriverApi.cuMemFree_v2(dQFused);
            if (dKFused != 0) CudaDriverApi.cuMemFree_v2(dKFused);
            if (dVFused != 0) CudaDriverApi.cuMemFree_v2(dVFused);
        }
    }
}
