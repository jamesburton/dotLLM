using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Equivalence tests for the MMQ-style fused dequant+matmul GEMV kernels.
/// Compares the dp4a-based MMQ output against the legacy FP-fmuladd kernel for
/// Q5_K and Q6_K weight types on synthetic data. The MMQ path quantizes the
/// input activation to INT8 per 32-element chunk before accumulating, so a
/// small relative drift vs the legacy kernel is expected and tolerated.
/// </summary>
[Trait("Category", "GPU")]
public class CudaMmqKernelTests
{
    private readonly ITestOutputHelper _out;
    public CudaMmqKernelTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    [SkippableTheory]
    [InlineData(4, 256)]    // 1 superblock × few rows (smaller than MMQ_ROWS_PER_BLOCK)
    [InlineData(8, 512)]    // 2 superblocks × MMQ_ROWS_PER_BLOCK × 2 blocks
    [InlineData(64, 1024)]  // 4 superblocks × many rows
    [InlineData(2048, 2048)] // Multi-superblock × multi-row stress (8 sb/row × 512 rows)
    public void MmqQ2K_MatchesLegacyWithinTolerance(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q2_K, n, k, blockBytes: 84,
            (rng, span) => SynthesiseQ2KBlock(rng, span));
    }

    [SkippableTheory]
    [InlineData(4, 256)]    // 1 superblock × few rows (smaller than MMQ_ROWS_PER_BLOCK)
    [InlineData(8, 512)]    // 2 superblocks × MMQ_ROWS_PER_BLOCK × 2 blocks
    [InlineData(64, 1024)]  // 4 superblocks × many rows
    public void MmqQ5K_MatchesLegacyWithinTolerance(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q5_K, n, k, blockBytes: 176,
            (rng, span) => SynthesiseQ5KBlock(rng, span));
    }

    [SkippableTheory]
    [InlineData(4, 256)]
    [InlineData(8, 512)]
    [InlineData(64, 1024)]
    public void MmqQ6K_MatchesLegacyWithinTolerance(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q6_K, n, k, blockBytes: 210,
            (rng, span) => SynthesiseQ6KBlock(rng, span));
    }

    // ── MMVQ-large equivalence tests ─────────────────────────────────────────
    // Validate the 1-row-per-block × 128-thread MMVQ-large kernel against the
    // legacy FP-fmuladd path on Qwen3-8B-class shapes (k=4096):
    //   - n=4096 covers the QkvProj fused n=q+k+v output dim for 4096-d models.
    //   - n=11008 stresses an MlpDown intermediate shape.
    //   - n=24576 = 2 × 12288 covers Qwen3-8B's fused gate+up MlpUp.
    // The dispatcher routes these to mmvq_large because k=4096 >= the 1024
    // threshold; k<1024 cases above continue to validate the MMQ-4-rows path.

    [SkippableTheory]
    [InlineData(4096, 4096)]
    [InlineData(11008, 4096)]
    [InlineData(24576, 4096)]
    public void MmvqLargeQ4K_MatchesLegacy_Qwen3_8B_Shapes(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q4_K, n, k, blockBytes: 144,
            (rng, span) => SynthesiseQ4KBlock(rng, span), requireMmvqLarge: true);
    }

    /// <summary>
    /// Synthetic Llama-70B-class shape: intermediate=14336 → 448 input chunks. Validates
    /// the dynamic shared-memory sizing works past the legacy compile-time MMQ_MAX_CHUNKS=384
    /// budget. We don't have a real 70B model locally; this case exists purely to prove
    /// the kernel adapts at launch time.
    /// </summary>
    [SkippableTheory]
    [InlineData(4096, 14336)]
    public void MmvqLargeQ4K_MatchesLegacy_Llama70B_Synthetic(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q4_K, n, k, blockBytes: 144,
            (rng, span) => SynthesiseQ4KBlock(rng, span), requireMmvqLarge: true);
    }

    [SkippableTheory]
    [InlineData(4096, 4096)]
    [InlineData(11008, 4096)]
    [InlineData(24576, 4096)]
    public void MmvqLargeQ5K_MatchesLegacy_Qwen3_8B_Shapes(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q5_K, n, k, blockBytes: 176,
            (rng, span) => SynthesiseQ5KBlock(rng, span), requireMmvqLarge: true);
    }

    [SkippableTheory]
    [InlineData(4096, 4096)]
    [InlineData(11008, 4096)]
    [InlineData(24576, 4096)]
    public void MmvqLargeQ6K_MatchesLegacy_Qwen3_8B_Shapes(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q6_K, n, k, blockBytes: 210,
            (rng, span) => SynthesiseQ6KBlock(rng, span), requireMmvqLarge: true);
    }

    // ── Pre-Q8_1 equivalence tests ──────────────────────────────────────────
    // Verify that pre-quantizing the input via quantize_x_to_q8_1 + the `_preq`
    // GEMV variants matches the on-the-fly kernel within the same K-quant noise
    // budget. The pre-quant kernel produces bit-identical scratch to the in-kernel
    // Stage 1 (same rounding, clamp, and per-half-chunk sum reduction) so the
    // GEMV outputs should agree to within FP16 add-order noise.

    [SkippableTheory]
    [InlineData(64, 1024)]    // small-k path (MMQ-4-rows preq)
    [InlineData(4096, 4096)]  // large-k path (MMVQ-large preq)
    [InlineData(24576, 4096)] // Qwen3-8B fused gate+up shape
    public void PreqQ4K_MatchesOnTheFly(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunPreqEquivalence(QuantizationType.Q4_K, n, k, blockBytes: 144,
            (rng, span) => SynthesiseQ4KBlock(rng, span));
    }

    [SkippableTheory]
    [InlineData(64, 1024)]
    [InlineData(4096, 4096)]
    [InlineData(24576, 4096)]
    public void PreqQ5K_MatchesOnTheFly(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunPreqEquivalence(QuantizationType.Q5_K, n, k, blockBytes: 176,
            (rng, span) => SynthesiseQ5KBlock(rng, span));
    }

    [SkippableTheory]
    [InlineData(64, 1024)]
    [InlineData(4096, 4096)]
    [InlineData(24576, 4096)]
    public void PreqQ6K_MatchesOnTheFly(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunPreqEquivalence(QuantizationType.Q6_K, n, k, blockBytes: 210,
            (rng, span) => SynthesiseQ6KBlock(rng, span));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunPreqEquivalence(QuantizationType qt, int n, int k, int blockBytes,
                                            Action<Random, Span<byte>> synthesiseBlock)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMmq(qt), $"MMQ kernel for {qt} not loaded (PTX may be stale)");
        Skip.IfNot(kernels.HasPreQ8_1, "Pre-Q8_1 quantize_x kernel not loaded (PTX may be stale)");

        // Force MMVQ-large ON for the large-k cases so we exercise both `_preq` variants.
        bool prevDisableQ4K = CudaKernels.DisableMmvqLargeQ4K;
        bool prevDisableQ5K = CudaKernels.DisableMmvqLargeQ5K;
        bool prevDisableQ6K = CudaKernels.DisableMmvqLargeQ6K;
        bool prevDisablePreq = CudaKernels.DisablePreQ8_1;
        CudaKernels.DisableMmvqLargeQ4K = false;
        CudaKernels.DisableMmvqLargeQ5K = false;
        CudaKernels.DisableMmvqLargeQ6K = false;
        CudaKernels.DisablePreQ8_1 = false;

        var rng = new Random(5678 ^ (int)qt ^ n ^ k);
        int superblocksPerRow = k / 256;
        int rowBytes = superblocksPerRow * blockBytes;
        long weightBytes = (long)n * rowBytes;

        byte[] hostWeight = new byte[weightBytes];
        var weightSpan = hostWeight.AsSpan();
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < superblocksPerRow; sb++)
            synthesiseBlock(rng, weightSpan.Slice(row * rowBytes + sb * blockBytes, blockBytes));

        Half[] hostX = new Half[k];
        for (int i = 0; i < k; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        long xBytes = (long)k * sizeof(ushort);
        long yBytes = (long)n * sizeof(ushort);
        long scratchBytes = CudaForwardState.PreQ8_1ScratchBytes(k);

        nint devW = 0, devX = 0, devYOnTheFly = 0, devYPreq = 0, devScratch = 0;
        Half[] yOnTheFly = new Half[n];
        Half[] yPreq = new Half[n];

        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYOnTheFly, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYPreq, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devScratch, (nuint)scratchBytes).ThrowOnError();

            fixed (byte* pW = hostWeight)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            // Path A: on-the-fly Stage 1 (no scratch passed → original kernel variant).
            kernels.LaunchQuantizedGemvMmq(devW, qt, devX, devYOnTheFly, n, k, preqScratch: 0, stream.Handle);
            // Path B: pre-Q8_1 + `_preq` kernel variant.
            kernels.LaunchQuantizeXToQ8_1(devX, devScratch, k, stream.Handle);
            kernels.LaunchQuantizedGemvMmq(devW, qt, devX, devYPreq, n, k, devScratch, stream.Handle);
            stream.Synchronize();

            fixed (Half* pY = yOnTheFly)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYOnTheFly, (nuint)yBytes).ThrowOnError();
            fixed (Half* pY = yPreq)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYPreq, (nuint)yBytes).ThrowOnError();
        }
        finally
        {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devYOnTheFly != 0) CudaDriverApi.cuMemFree_v2(devYOnTheFly);
            if (devYPreq != 0) CudaDriverApi.cuMemFree_v2(devYPreq);
            if (devScratch != 0) CudaDriverApi.cuMemFree_v2(devScratch);
            CudaKernels.DisableMmvqLargeQ4K = prevDisableQ4K;
            CudaKernels.DisableMmvqLargeQ5K = prevDisableQ5K;
            CudaKernels.DisableMmvqLargeQ6K = prevDisableQ6K;
            CudaKernels.DisablePreQ8_1 = prevDisablePreq;
        }

        // Pre-quant produces bit-identical INT8/dx/sx2 to the in-kernel Stage 1
        // (same rounding, clamp, half-warp sum). The only divergence is the
        // chunk-sum recombination for Q4_K/Q5_K (sx2[lo]+sx2[hi] vs full-warp sum):
        // an FP16 add-order difference of < 1 ULP per chunk. End-to-end we expect
        // peak-relative drift well under 1%.
        float maxAbs = 0f, refMax = 0f;
        double sumAbs = 0.0;
        for (int i = 0; i < n; i++)
        {
            float a = (float)yOnTheFly[i];
            float b = (float)yPreq[i];
            float diff = MathF.Abs(a - b);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(a) > refMax) refMax = MathF.Abs(a);
        }
        float meanAbs = (float)(sumAbs / n);
        float peakRelMax = refMax > 0 ? maxAbs / refMax : 0f;
        float peakRelMean = refMax > 0 ? meanAbs / refMax : 0f;

        _out.WriteLine($"PREQ {qt} n={n} k={k}: ref|max|={refMax:F3}  |preq-onTheFly| max={maxAbs:F4} mean={meanAbs:F4} peak-rel max={peakRelMax:P3} mean={peakRelMean:P3}");

        // 1% peak-relative tolerance — much tighter than the 3% cross-kernel test
        // because Stage 1 itself is bit-identical; only post-Stage-1 FP16 add-order
        // and the lazy sx_lo+sx_hi recombination differ.
        Assert.True(peakRelMax < 0.01f, $"Peak-relative max diff {peakRelMax:P3} exceeds 1% (max={maxAbs}, refMax={refMax})");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunMmqEquivalence(QuantizationType qt, int n, int k, int blockBytes,
                                           Action<Random, Span<byte>> synthesiseBlock,
                                           bool requireMmvqLarge = false)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMmq(qt), $"MMQ kernel for {qt} not loaded (PTX may be stale)");

        // MMVQ-large is now ON by default (pre-Q8_1 lands the win that was missing before).
        // Per-quant-type Disable knobs let us force the MMQ-4-rows path for A/B comparison.
        // We snapshot + restore the disable knobs to avoid leaking test state across cases.
        bool prevDisableQ4K = CudaKernels.DisableMmvqLargeQ4K;
        bool prevDisableQ5K = CudaKernels.DisableMmvqLargeQ5K;
        bool prevDisableQ6K = CudaKernels.DisableMmvqLargeQ6K;
        if (requireMmvqLarge)
        {
            // Make sure no prior test left the kernel disabled.
            CudaKernels.DisableMmvqLargeQ4K = false;
            CudaKernels.DisableMmvqLargeQ5K = false;
            CudaKernels.DisableMmvqLargeQ6K = false;
            Skip.IfNot(kernels.HasMmvqLarge(qt), $"MMVQ-large kernel for {qt} not loaded (PTX may be stale)");
        }

        var rng = new Random(1234 ^ (int)qt ^ n ^ k);
        int superblocksPerRow = k / 256;
        int rowBytes = superblocksPerRow * blockBytes;
        long weightBytes = (long)n * rowBytes;

        // Synthetic weight: realistic per-block scales, random qs/qh body.
        byte[] hostWeight = new byte[weightBytes];
        var weightSpan = hostWeight.AsSpan();
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < superblocksPerRow; sb++)
            synthesiseBlock(rng, weightSpan.Slice(row * rowBytes + sb * blockBytes, blockBytes));

        // Realistic input magnitudes (normal-ish distribution, scaled).
        Half[] hostX = new Half[k];
        for (int i = 0; i < k; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        long xBytes = (long)k * sizeof(ushort);
        long yBytes = (long)n * sizeof(ushort);

        nint devW = 0, devX = 0, devYLegacy = 0, devYMmq = 0;
        Half[] yLegacy = new Half[n];
        Half[] yMmq = new Half[n];

        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYLegacy, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYMmq, (nuint)yBytes).ThrowOnError();

            fixed (byte* pW = hostWeight)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            kernels.LaunchQuantizedGemv(devW, qt, devX, devYLegacy, n, k, stream.Handle);
            kernels.LaunchQuantizedGemvMmq(devW, qt, devX, devYMmq, n, k, stream.Handle);
            stream.Synchronize();

            fixed (Half* pY = yLegacy)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYLegacy, (nuint)yBytes).ThrowOnError();
            fixed (Half* pY = yMmq)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYMmq, (nuint)yBytes).ThrowOnError();
        }
        finally
        {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devYLegacy != 0) CudaDriverApi.cuMemFree_v2(devYLegacy);
            if (devYMmq != 0) CudaDriverApi.cuMemFree_v2(devYMmq);
            // Restore prior MMVQ-large disable knobs (see top of method).
            CudaKernels.DisableMmvqLargeQ4K = prevDisableQ4K;
            CudaKernels.DisableMmvqLargeQ5K = prevDisableQ5K;
            CudaKernels.DisableMmvqLargeQ6K = prevDisableQ6K;
        }

        // Compare against the **peak magnitude** of the legacy output rather than
        // per-element. INT8 input requantization injects ~1/127 ≈ 0.8% per-element
        // noise that accumulates with cancellation across k elements, so individual
        // near-zero outputs can drift by a large per-element fraction while the
        // peak-relative drift stays small. Peak-relative drift is what matters
        // for downstream softmax / argmax stability — that is the invariant
        // verified end-to-end by CudaLogitsMatchPyTorchReferenceTests.
        float maxAbs = 0f, refMax = 0f;
        double sumAbs = 0.0;
        for (int i = 0; i < n; i++)
        {
            float a = (float)yLegacy[i];
            float b = (float)yMmq[i];
            float diff = MathF.Abs(a - b);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(a) > refMax) refMax = MathF.Abs(a);
        }
        float meanAbs = (float)(sumAbs / n);
        float peakRelMax = refMax > 0 ? maxAbs / refMax : 0f;
        float peakRelMean = refMax > 0 ? meanAbs / refMax : 0f;

        _out.WriteLine($"{qt} n={n} k={k}: ref|max|={refMax:F3}  |MMQ-legacy| max={maxAbs:F4} mean={meanAbs:F4} peak-rel max={peakRelMax:P2} mean={peakRelMean:P2}");

        // Peak-relative tolerance: 3% admits INT8 input quantization noise across
        // the wide range of synthetic magnitudes (random scales/qs/qh produce
        // outputs from ~10 to ~1000). Real-model decode produces much smaller
        // outputs and tighter ratios in practice (the end-to-end logits parity
        // test on SmolLM-135M Q4_K_M passes within Tight tolerance).
        Assert.True(peakRelMax < 0.03f, $"Peak-relative max diff {peakRelMax:P2} exceeds 3% (max={maxAbs}, refMax={refMax})");
    }

    private static unsafe void SynthesiseQ2KBlock(Random rng, Span<byte> block)
    {
        // Q2_K layout (84 bytes): uint8[16] scales, uint8[64] qs, half d (offset 80), half dmin (offset 82).
        // scales: low nibble = 4-bit sub-block scale, high nibble = 4-bit dmin coef.
        // Smaller d/dmin magnitude than Q4_K because q∈[0,3] × sc∈[0,15] means each
        // element's per-d contribution is up to 45 (vs Q4_K's 15×15=225).
        for (int i = 0; i < 80; i++)
            block[i] = (byte)rng.Next(0, 256);

        Half d = (Half)(rng.NextDouble() * 0.05 + 0.01);
        Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.02);
        fixed (byte* pBlock = block)
        {
            *(Half*)(pBlock + 80) = d;
            *(Half*)(pBlock + 82) = dmin;
        }
    }

    private static unsafe void SynthesiseQ4KBlock(Random rng, Span<byte> block)
    {
        // Q4_K layout (144 bytes): half d, half dmin, uint8[12] scales, uint8[128] qs.
        Half d = (Half)(rng.NextDouble() * 0.05 + 0.01);
        Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.04);

        fixed (byte* pBlock = block)
        {
            *(Half*)pBlock = d;
            *(Half*)(pBlock + 2) = dmin;
        }
        for (int i = 4; i < 144; i++)
            block[i] = (byte)rng.Next(0, 256);
    }

    private static unsafe void SynthesiseQ5KBlock(Random rng, Span<byte> block)
    {
        // Q5_K layout (176 bytes): half d, half dmin, uint8[12] scales, uint8[32] qh, uint8[128] qs.
        // d/dmin in realistic range; scales/qh/qs uniform random.
        Half d = (Half)(rng.NextDouble() * 0.05 + 0.01);
        Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.04);

        fixed (byte* pBlock = block)
        {
            *(Half*)pBlock = d;
            *(Half*)(pBlock + 2) = dmin;
        }
        for (int i = 4; i < 176; i++)
            block[i] = (byte)rng.Next(0, 256);
    }

    private static unsafe void SynthesiseQ6KBlock(Random rng, Span<byte> block)
    {
        // Q6_K layout (210 bytes): uint8[128] ql, uint8[64] qh, int8[16] scales, half d (at offset 208).
        for (int i = 0; i < 192; i++)
            block[i] = (byte)rng.Next(0, 256);

        // int8 scales: bias toward small magnitudes (typical Q6_K scale range).
        for (int i = 0; i < 16; i++)
            block[192 + i] = (byte)(sbyte)(rng.Next(-32, 33));

        Half d = (Half)(rng.NextDouble() * 0.005 + 0.001);
        fixed (byte* pBlock = block)
            *(Half*)(pBlock + 208) = d;
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
