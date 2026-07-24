using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness anchor for <see cref="CudaKernels.LaunchGdnScanStepF32"/> (the <c>gdn_scan_step_f32</c>
/// CUDA kernel, native/kernels/gated_delta_net_scan.cu) against its CPU oracle,
/// <see cref="GatedDeltaNetScan.Execute"/>. No focused CUDA test existed for this kernel before
/// issue #173's kernel-internal rewrite (fused decay+retrieve / write+read passes, one fewer
/// __syncthreads()) — this test both validates that rewrite and gives the kernel permanent
/// regression coverage against the CPU reference it claims bit-perfect parity with.
/// </summary>
[Trait("Category", "GPU")]
public class CudaGdnScanStepF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaGdnScanStepF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

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
    /// Runs <paramref name="steps"/> consecutive decode steps (seqLen==1 per call, matching how
    /// <c>ForwardGdnBody</c> drives this kernel) through BOTH the GPU kernel and the CPU reference,
    /// feeding the SAME random per-step q/k/v/g/beta rows to both, and asserts bit-exact equality of
    /// the output AND the evolving recurrence state after every step — not just the first. Both
    /// implementations are documented as bit-perfect ports of the same reference algorithm (the
    /// kernel's own header comment; -fmad=false; identical row-ordered accumulation), so exact
    /// equality is the correct bar, matching e.g. <c>GdnDeinterleaveL2NormDecodeF32_MatchesSeparateDeinterleavePlusL2Norm</c>'s precedent.
    /// </summary>
    [SkippableTheory]
    [InlineData(2, 1, 4, 5)]      // tiny shape, VHeadsPerKHead=2, several steps
    [InlineData(32, 16, 128, 4)]  // real Bonsai-27B Qwen3HybridDense GDN shape
    public void GdnScanStepF32_MatchesCpuReference_AcrossMultipleSteps(int nVHead, int nKHead, int dState, int steps)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        var rng = new Random(0xBEEF ^ nVHead ^ nKHead ^ (dState << 8) ^ (steps << 20));

        int statePerHead = dState * dState;
        int qkPerToken = nKHead * dState;
        int vPerToken = nVHead * dState;

        float[] cpuState = new float[nVHead * statePerHead];
        for (int i = 0; i < cpuState.Length; i++) cpuState[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        float[] gpuStateInit = (float[])cpuState.Clone();

        nint dState_ = 0, dQ = 0, dK = 0, dV = 0, dG = 0, dBeta = 0, dOut = 0;
        try
        {
            long stateBytes = (long)cpuState.Length * sizeof(float);
            long qkBytes = (long)qkPerToken * sizeof(float);
            long vBytes = (long)vPerToken * sizeof(float);
            long gbBytes = (long)nVHead * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dState_, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)qkBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)vBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dG, (nuint)gbBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dBeta, (nuint)gbBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOut, (nuint)vBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = gpuStateInit)
                    CudaDriverApi.cuMemcpyHtoD_v2(dState_, (nint)p, (nuint)stateBytes).ThrowOnError();
            }

            nint s = stream.Handle;
            float[] cpuOutput = new float[vPerToken];

            for (int step = 0; step < steps; step++)
            {
                float[] q = new float[qkPerToken];
                float[] k = new float[qkPerToken];
                float[] v = new float[vPerToken];
                float[] g = new float[nVHead];
                float[] beta = new float[nVHead];
                for (int i = 0; i < qkPerToken; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                for (int i = 0; i < qkPerToken; i++) k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                for (int i = 0; i < vPerToken; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                for (int i = 0; i < nVHead; i++) g[i] = (float)rng.NextDouble();          // decay in (0,1]
                for (int i = 0; i < nVHead; i++) beta[i] = (float)rng.NextDouble();       // write-gate in [0,1]

                // CPU reference: L2-norm is the caller's job (already applied upstream in the real
                // pipeline); q/k here are treated as pre-normalized inputs to isolate the recurrence
                // kernel itself, matching how gdn_scan_step_f32's own contract documents q_t/k_t as
                // "already L2-normed by caller".
                GatedDeltaNetScan.Execute(cpuState, q, k, v, g, beta, cpuOutput,
                    nVHead, nKHead, dState, seqLen: 1);

                unsafe
                {
                    fixed (float* pq = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)pq, (nuint)qkBytes).ThrowOnError();
                    fixed (float* pk = k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)pk, (nuint)qkBytes).ThrowOnError();
                    fixed (float* pv = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)pv, (nuint)vBytes).ThrowOnError();
                    fixed (float* pg = g) CudaDriverApi.cuMemcpyHtoD_v2(dG, (nint)pg, (nuint)gbBytes).ThrowOnError();
                    fixed (float* pb = beta) CudaDriverApi.cuMemcpyHtoD_v2(dBeta, (nint)pb, (nuint)gbBytes).ThrowOnError();
                }

                kernels.LaunchGdnScanStepF32(dState_, dQ, dK, dV, dG, dBeta, dOut, nVHead, nKHead, dState, s);
                stream.Synchronize();

                float[] gpuOutput = new float[vPerToken];
                float[] gpuState = new float[cpuState.Length];
                unsafe
                {
                    fixed (float* p = gpuOutput) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)vBytes).ThrowOnError();
                    fixed (float* p = gpuState) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dState_, (nuint)stateBytes).ThrowOnError();
                }

                Assert.True(cpuOutput.AsSpan().SequenceEqual(gpuOutput),
                    $"step {step}: output mismatch (nVHead={nVHead}, nKHead={nKHead}, dState={dState}).");
                Assert.True(cpuState.AsSpan().SequenceEqual(gpuState),
                    $"step {step}: recurrence state mismatch after update (nVHead={nVHead}, nKHead={nKHead}, dState={dState}).");
            }

            _out.WriteLine($"nVHead={nVHead} nKHead={nKHead} dState={dState} steps={steps}: exact match every step, output and state.");
        }
        finally
        {
            if (dState_ != 0) CudaDriverApi.cuMemFree_v2(dState_);
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dK != 0) CudaDriverApi.cuMemFree_v2(dK);
            if (dV != 0) CudaDriverApi.cuMemFree_v2(dV);
            if (dG != 0) CudaDriverApi.cuMemFree_v2(dG);
            if (dBeta != 0) CudaDriverApi.cuMemFree_v2(dBeta);
            if (dOut != 0) CudaDriverApi.cuMemFree_v2(dOut);
        }
    }
}
