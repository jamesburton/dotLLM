using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness tests for the fused decode-time causal-conv1d kernel
/// (<see cref="CudaKernels.LaunchGdnConv1dCausalDecodeF32"/>, issue #168) against the general
/// path it replaces for <c>seqLen==1</c>: manual <c>[state; qkv]</c> concat memcpy ×2,
/// <see cref="CudaKernels.LaunchConv1dCausalF32"/>, <see cref="CudaKernels.LaunchSiluF32"/>,
/// trailing-state-extract memcpy.
/// </summary>
[Trait("Category", "GPU")]
public class CudaGdnConv1dCausalDecodeF32Tests
{
    private readonly ITestOutputHelper _out;
    public CudaGdnConv1dCausalDecodeF32Tests(ITestOutputHelper output) => _out = output;

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
    /// Runs <paramref name="steps"/> consecutive decode steps through BOTH the general
    /// (memcpy-concat + conv1d_causal_f32 + silu_f32 + memcpy-extract) path and the new fused
    /// <c>gdn_conv1d_causal_decode_f32</c> kernel, feeding the SAME random per-step raw qkv rows
    /// to both, and asserts bit-exact equality of both the conv+SiLU output AND the evolving
    /// rolling conv-state after every step — not just the first. Both paths compile from the
    /// same <c>-fmad=false</c> translation unit with identical accumulation/SiLU order, so exact
    /// (not ULP-tolerant) equality is the correct bar here, matching
    /// <c>GdnDecaySigmoidF32_MatchesSeparateDecayPlusSigmoid</c>'s precedent for GPU-vs-GPU
    /// same-math comparisons.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 16, 5)]       // small channel count, several steps
    [InlineData(4, 10240, 3)]    // real Bonsai convDim = (2*16+48)*128, a few steps
    public void FusedDecode_MatchesGeneralPath_AcrossMultipleSteps(int dConv, int channels, int steps)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasGdnConv1dCausalDecodeF32, "gdn_conv1d_causal_decode_f32 not loaded (PTX may be stale)");

        var rng = new Random(0xC0FFEE ^ dConv ^ channels ^ (steps << 16));

        int stateRows = dConv - 1;
        float[] initState = new float[stateRows * channels];
        for (int i = 0; i < initState.Length; i++) initState[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] weight = new float[channels * dConv]; // channel-major: w(k,c) at c*dConv+k
        for (int i = 0; i < weight.Length; i++) weight[i] = (float)(rng.NextDouble() * 1.0 - 0.5);

        float[] bias = new float[channels];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        nint dStateSep = 0, dStateFused = 0, dWeight = 0, dBias = 0;
        nint dConcatScratch = 0, dQkvSep = 0, dQkvFused = 0;
        try
        {
            long stateBytes = (long)initState.Length * sizeof(float);
            long weightBytes = (long)weight.Length * sizeof(float);
            long biasBytes = (long)bias.Length * sizeof(float);
            long qkvBytes = (long)channels * sizeof(float);
            long concatBytes = (long)(stateRows + 1) * channels * sizeof(float);

            CudaDriverApi.cuMemAlloc_v2(out dStateSep, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dStateFused, (nuint)stateBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dWeight, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dBias, (nuint)biasBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dConcatScratch, (nuint)concatBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQkvSep, (nuint)qkvBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dQkvFused, (nuint)qkvBytes).ThrowOnError();

            unsafe
            {
                fixed (float* p = initState)
                {
                    CudaDriverApi.cuMemcpyHtoD_v2(dStateSep, (nint)p, (nuint)stateBytes).ThrowOnError();
                    CudaDriverApi.cuMemcpyHtoD_v2(dStateFused, (nint)p, (nuint)stateBytes).ThrowOnError();
                }
                fixed (float* p = weight)
                    CudaDriverApi.cuMemcpyHtoD_v2(dWeight, (nint)p, (nuint)weightBytes).ThrowOnError();
                fixed (float* p = bias)
                    CudaDriverApi.cuMemcpyHtoD_v2(dBias, (nint)p, (nuint)biasBytes).ThrowOnError();
            }

            nint s = stream.Handle;
            long stateRowBytes = (long)channels * sizeof(float);

            for (int step = 0; step < steps; step++)
            {
                float[] qkvIn = new float[channels];
                for (int i = 0; i < channels; i++) qkvIn[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

                unsafe
                {
                    fixed (float* p = qkvIn)
                    {
                        CudaDriverApi.cuMemcpyHtoD_v2(dQkvSep, (nint)p, (nuint)qkvBytes).ThrowOnError();
                        CudaDriverApi.cuMemcpyHtoD_v2(dQkvFused, (nint)p, (nuint)qkvBytes).ThrowOnError();
                    }
                }

                // ── General path: memcpy-concat, conv1d_causal_f32, silu_f32, memcpy-extract ──
                CudaDriverApi.cuMemcpyDtoDAsync_v2(dConcatScratch, dStateSep, (nuint)stateBytes, s).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync_v2(dConcatScratch + (nint)stateBytes, dQkvSep, (nuint)qkvBytes, s).ThrowOnError();
                kernels.LaunchConv1dCausalF32(dConcatScratch, dWeight, dBias, dQkvSep, dConv, channels, seqLen: 1, s);
                kernels.LaunchSiluF32(dQkvSep, channels, s);
                nint trailSrc = dConcatScratch + (nint)stateRowBytes; // last (dConv-1) rows of the (dConv rows) concat buffer
                CudaDriverApi.cuMemcpyDtoDAsync_v2(dStateSep, trailSrc, (nuint)stateBytes, s).ThrowOnError();

                // ── Fused path: gdn_conv1d_causal_decode_f32, in-place on qkv (aliased) ──
                kernels.LaunchGdnConv1dCausalDecodeF32(dStateFused, dQkvFused, dWeight, dBias,
                    dQkvFused, dConv, channels, s);

                stream.Synchronize();

                float[] qkvOutSep = new float[channels];
                float[] qkvOutFused = new float[channels];
                float[] stateOutSep = new float[initState.Length];
                float[] stateOutFused = new float[initState.Length];
                unsafe
                {
                    fixed (float* p = qkvOutSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQkvSep, (nuint)qkvBytes).ThrowOnError();
                    fixed (float* p = qkvOutFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dQkvFused, (nuint)qkvBytes).ThrowOnError();
                    fixed (float* p = stateOutSep) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dStateSep, (nuint)stateBytes).ThrowOnError();
                    fixed (float* p = stateOutFused) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dStateFused, (nuint)stateBytes).ThrowOnError();
                }

                Assert.True(qkvOutSep.AsSpan().SequenceEqual(qkvOutFused),
                    $"step {step}: conv+SiLU output mismatch (dConv={dConv}, channels={channels}).");
                Assert.True(stateOutSep.AsSpan().SequenceEqual(stateOutFused),
                    $"step {step}: rolling conv-state mismatch after update (dConv={dConv}, channels={channels}).");
            }

            _out.WriteLine($"dConv={dConv} channels={channels} steps={steps}: exact match every step, output and state.");
        }
        finally
        {
            if (dStateSep != 0) CudaDriverApi.cuMemFree_v2(dStateSep);
            if (dStateFused != 0) CudaDriverApi.cuMemFree_v2(dStateFused);
            if (dWeight != 0) CudaDriverApi.cuMemFree_v2(dWeight);
            if (dBias != 0) CudaDriverApi.cuMemFree_v2(dBias);
            if (dConcatScratch != 0) CudaDriverApi.cuMemFree_v2(dConcatScratch);
            if (dQkvSep != 0) CudaDriverApi.cuMemFree_v2(dQkvSep);
            if (dQkvFused != 0) CudaDriverApi.cuMemFree_v2(dQkvFused);
        }
    }
}
