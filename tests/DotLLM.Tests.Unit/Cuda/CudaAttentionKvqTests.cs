using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1850: the dense quantized KV cache's in-place attention kernels against the pre-#1850 path, which dequantized the history
/// into an FP16 scratch and ran the F16 kernels on it.
/// <list type="bullet">
/// <item><c>attention_kvq</c> (decode and the prefill fallback) must equal <c>LaunchKvDequantToF16</c> into a scratch plus
/// <c>attention_f16</c> BIT FOR BIT: every Q8_0 / Q4_0 pair, head_dim 64 and 128, decode and prefill at an offset, sliding window,
/// sinks, and the FP16 window ring (wrapping, and with the quantized length above seqKv - window as a rollback leaves it).</item>
/// <item><c>attention_flash_mma_prefill_kvq</c> over quantized blocks must equal its own F16 entry over the dequantized scratch bit
/// for bit (the loaders write the same halves), AND agree with the scalar kernel within F16 rounding. The second check is the one
/// that can see a bug in the shared body (masking, offset, GQA mapping), which the first cannot.</item>
/// </list>
/// Fixtures: two or more KV heads (no degenerate head broadcast), at least two 32-blocks per head-row, a different amplitude per
/// block (a wrong block scale shows), seqKv not a multiple of any tile, non-zero offsets. Ring slots and quantized rows that must
/// NOT be read hold different, finite data, so reading the wrong region changes the output.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe partial class CudaAttentionKvqTests : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly ITestOutputHelper _out;

    public CudaAttentionKvqTests(ITestOutputHelper output)
    {
        _out = output;
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null) _kernels = new CudaKernels(ptxDir);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    public static IEnumerable<object[]> ScalarCases() =>
    [
        // kFmt, vFmt, headDim, numHeads, numKvHeads, seqQ, offset, slidingWindow, sinks, window, quantLen (-1 = seqKv)
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 8, 2, 1, 300, 0, false, 0, -1],
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0, 64, 6, 2, 1, 300, 0, false, 0, -1],
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0, 64, 6, 2, 37, 20, 0, false, 0, -1],
        [KvCacheDType.Q4_0, KvCacheDType.Q8_0, 128, 8, 2, 19, 0, 0, false, 0, -1],
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 6, 2, 5, 296, 50, false, 0, -1],
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 6, 2, 3, 100, 0, true, 0, -1],
        // FP16 window ring: quantized [0, 269), ring rows 269..300 (wraps at slot 32 -> 0)
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 6, 2, 1, 300, 0, false, 32, 269],
        // quantized length above seqKv - window (what a rollback leaves): ring rows 275..300 only
        [KvCacheDType.Q4_0, KvCacheDType.Q8_0, 128, 8, 2, 1, 300, 0, false, 32, 275],
        // windowed prefill chunk
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 6, 2, 10, 291, 0, false, 32, 269],
        // head_dim % 32 != 0 (Phi-2 / StableLM hd80): KV head 1 starts mid-block (element 80), so the per-head block loop
        // must not assume 32-aligned heads. Both formats, decode and a windowed prefill.
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 80, 4, 2, 1, 300, 0, false, 0, -1],
        [KvCacheDType.Q4_0, KvCacheDType.Q8_0, 80, 4, 2, 19, 0, 0, false, 0, -1],
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0, 80, 6, 2, 10, 291, 0, false, 32, 269],
        // head_dim 96 (Phi-3) and 256 (Gemma / Qwen3.5 full attention): aligned, but outside the tensor-core prefill's hd64/128.
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 96, 32, 8, 20, 50, 0, false, 0, -1],
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0, 256, 8, 2, 17, 40, 0, false, 0, -1],
    ];

    [SkippableTheory]
    [MemberData(nameof(ScalarCases))]
    public void AttentionKvq_IsBitIdenticalToDequantScratchPlusAttentionF16(KvCacheDType kFmt, KvCacheDType vFmt, int headDim,
        int numHeads, int numKvHeads, int seqQ, int offset, int slidingWindow, bool sinks, int window, int quantLen)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionKvq, "attention_kvq PTX not available");
        int seqKv = offset + seqQ;
        if (quantLen < 0) quantLen = seqKv;
        using var f = new Fixture(this, kFmt, vFmt, headDim, numHeads, numKvHeads, seqQ, seqKv, quantLen, window, sinks);

        _kernels!.LaunchAttention(f.Q, f.KScratch, f.VScratch, f.OutRef, seqQ, seqKv, numHeads, numKvHeads, headDim,
            offset, slidingWindow, _stream!.Handle, f.Sinks);
        _kernels.LaunchAttentionKvq(f.Q, f.KQuant, f.VQuant, f.KRing, f.VRing, f.OutNew, seqQ, seqKv, numHeads, numKvHeads,
            headDim, offset, slidingWindow, _stream.Handle, f.Sinks, kFmt, vFmt, quantLen, window);
        _stream.Synchronize();

        ushort[] reference = f.Read(f.OutRef), actual = f.Read(f.OutNew);
        AssertFinite(reference);
        int diff = FirstDiff(reference, actual);
        Assert.True(diff < 0, $"attention_kvq differs from scratch + attention_f16 at element {diff}: " +
            (diff >= 0 ? $"{HalfToFloat(actual[diff])} vs {HalfToFloat(reference[diff])}" : ""));
    }

    public static IEnumerable<object[]> FlashCases() =>
    [
        // kFmt, vFmt, headDim, numHeads, numKvHeads, seqQ, offset
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 14, 2, 45, 0],      // Qwen2.5-0.5B shape, group 7, cold
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 64, 14, 2, 45, 70],     // ... suffix prefill at an offset (#1847)
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 16, 4, 40, 0],     // TwIL-8B-like hd128, group 4
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 16, 4, 33, 291],
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0, 128, 16, 2, 33, 50],    // group 8
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0, 128, 8, 2, 17, 0],
        [KvCacheDType.Q4_0, KvCacheDType.Q8_0, 64, 6, 2, 64, 30],
        // #1923 raw cp.async staging: odd KV head counts (head rows at 8- / 4-byte but not 16-byte offsets), seqKv = 16n + 1 (the
        // tail clamp reads the last row; the NaN guard rows past the cache catch an unclamped read), a 4K-key history, and
        // K Q8_0 with V Q4_0 at hd64.
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 12, 3, 33, 288],
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0, 128, 15, 5, 40, 4057],
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0, 64, 9, 3, 47, 2002],
        [KvCacheDType.Q4_0, KvCacheDType.Q8_0, 64, 6, 3, 17, 0],
    ];

    [SkippableTheory]
    [MemberData(nameof(FlashCases))]
    public void FlashPrefillKvq_MatchesItsF16EntryBitForBit_AndTheScalarKernel(KvCacheDType kFmt, KvCacheDType vFmt, int headDim,
        int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionKvq, "attention_kvq PTX not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillKvq(headDim, kFmt, vFmt) && _kernels.HasAttentionFlashPrefillKvq(headDim, null, null),
            "flash prefill kvq kernel not loaded (needs Ampere+)");
        int seqKv = offset + seqQ;
        using var f = new Fixture(this, kFmt, vFmt, headDim, numHeads, numKvHeads, seqQ, seqKv, seqKv, window: 0, sinks: false);

        long launches = CudaKernels.AttentionFlashPrefillKvqLaunches;
        _kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KScratch, f.VScratch, null, null, f.OutRef,
            seqQ, seqKv, numHeads, numKvHeads, headDim, offset, _stream!.Handle);
        _kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KQuant, f.VQuant, kFmt, vFmt, f.OutNew,
            seqQ, seqKv, numHeads, numKvHeads, headDim, offset, _stream.Handle);
        _kernels.LaunchAttentionKvq(f.Q, f.KQuant, f.VQuant, 0, 0, f.OutScalar, seqQ, seqKv, numHeads, numKvHeads,
            headDim, offset, 0, _stream.Handle, 0, kFmt, vFmt, seqKv, 0);
        _stream.Synchronize();
        Assert.Equal(launches + 1, CudaKernels.AttentionFlashPrefillKvqLaunches);

        ushort[] f16 = f.Read(f.OutRef), quant = f.Read(f.OutNew), scalar = f.Read(f.OutScalar);
        AssertFinite(quant);
        int diff = FirstDiff(f16, quant);
        Assert.True(diff < 0, $"quantized flash entry differs from the F16 entry over dequantized K/V at element {diff}");

        // vs the scalar kernel: P is rounded to F16 for the PV mma and the output is F16, so allow a few F16 ulps at the output's
        // magnitude. A masking / offset / head-mapping bug moves outputs by O(0.1).
        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < scalar.Length; i++)
        {
            double a = HalfToFloat(quant[i]), b = HalfToFloat(scalar[i]);
            double err = Math.Abs(a - b) / (3e-3 + 3e-3 * Math.Abs(b));
            if (err > worst) { worst = err; worstAt = i; }
        }
        _out.WriteLine($"{kFmt}/{vFmt} hd{headDim} {numHeads}/{numKvHeads} seqQ {seqQ} offset {offset}: worst normalised error " +
                       $"{worst:F3} at {worstAt} ({HalfToFloat(quant[worstAt])} vs scalar {HalfToFloat(scalar[worstAt])})");
        Assert.True(worst <= 1.0, $"flash prefill kvq disagrees with attention_kvq at {worstAt}: " +
            $"{HalfToFloat(quant[worstAt])} vs {HalfToFloat(scalar[worstAt])}");
    }

    /// <summary>
    /// #1923: the staged packed-half2 decoders against the F16 entry over the eager dequant, with a third of the blocks scaled so
    /// small that their F16 scale d is subnormal. Bit identity holds only if mul.rn.f16x2 rounds a subnormal product as
    /// __float2half(d * (float)q) does; the regular FlashCases never produce a subnormal d.
    /// </summary>
    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q8_0, 128, 16, 4, 33, 291)]
    [InlineData(KvCacheDType.Q4_0, KvCacheDType.Q4_0, 128, 12, 3, 40, 57)]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.Q4_0, 64, 6, 2, 17, 30)]
    public void FlashPrefillKvq_SubnormalBlockScales_MatchItsF16EntryBitForBit(KvCacheDType kFmt, KvCacheDType vFmt, int headDim,
        int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillKvq(headDim, kFmt, vFmt) && _kernels.HasAttentionFlashPrefillKvq(headDim, null, null),
            "flash prefill kvq kernel not loaded (needs Ampere+)");
        int seqKv = offset + seqQ;
        using var f = new Fixture(this, kFmt, vFmt, headDim, numHeads, numKvHeads, seqQ, seqKv, seqKv, window: 0, sinks: false,
            tinyBlocks: true);
        _kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KScratch, f.VScratch, null, null, f.OutRef,
            seqQ, seqKv, numHeads, numKvHeads, headDim, offset, _stream!.Handle);
        _kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KQuant, f.VQuant, kFmt, vFmt, f.OutNew,
            seqQ, seqKv, numHeads, numKvHeads, headDim, offset, _stream.Handle);
        _stream.Synchronize();
        ushort[] f16 = f.Read(f.OutRef), quant = f.Read(f.OutNew);
        AssertFinite(quant);
        int diff = FirstDiff(f16, quant);
        Assert.True(diff < 0, $"quantized flash entry differs from the F16 entry at element {diff} with subnormal block scales");
    }

    public static IEnumerable<object[]> F16FlashCases() =>
    [
        // headDim, numHeads, numKvHeads, seqQ, offset  (seqKv = offset + seqQ, never a multiple of 16)
        [128, 32, 8, 1000, 1024],   // Qwen3-4B shape, a later 1024-row chunk (partial last q-tile)
        [128, 32, 8, 33, 0],        // cold, two q-tiles + 1 row
        [128, 32, 8, 1000, 291],    // offset off the 16-row grid
        [128, 16, 2, 33, 1024],     // group 8
        [64, 14, 2, 1000, 291],     // Qwen2.5-0.5B shape, group 7
        [64, 14, 2, 33, 0],
    ];

    /// <summary>
    /// Issue #1923: the kvq kernel's F16 entry, which now serves the dense F16 cache's prefills, against the scalar
    /// <c>attention_f16</c> it replaces (and G3 replaced before it), on F16 K/V with block-varying amplitude. Non-degenerate shapes:
    /// several KV heads with group 2..8 (a head-mapping bug shows), seqQ and seqKv off every tile, offsets off the q-tile grid.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(F16FlashCases))]
    public void FlashPrefillF16Entry_AgreesWithAttentionF16(int headDim, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillKvq(headDim, null, null), "flash prefill F16 entry not loaded (needs Ampere+)");
        int seqKv = offset + seqQ;
        // The scratch of a Q8_0 fixture is plain F16 K/V (the dequantized rows); only it is read here.
        using var f = new Fixture(this, KvCacheDType.Q8_0, KvCacheDType.Q8_0, headDim, numHeads, numKvHeads, seqQ, seqKv, seqKv,
            window: 0, sinks: false);

        long launches = CudaKernels.AttentionFlashPrefillF16Launches;
        _kernels.LaunchAttention(f.Q, f.KScratch, f.VScratch, f.OutRef, seqQ, seqKv, numHeads, numKvHeads, headDim,
            offset, 0, _stream!.Handle);
        _kernels.LaunchAttentionFlashPrefillKvq(f.Q, f.KScratch, f.VScratch, null, null, f.OutNew,
            seqQ, seqKv, numHeads, numKvHeads, headDim, offset, _stream.Handle);
        _stream.Synchronize();
        Assert.Equal(launches + 1, CudaKernels.AttentionFlashPrefillF16Launches);

        ushort[] reference = f.Read(f.OutRef), flash = f.Read(f.OutNew);
        AssertFinite(flash);
        double worst = 0;
        int worstAt = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double a = HalfToFloat(flash[i]), b = HalfToFloat(reference[i]);
            double err = Math.Abs(a - b) / (3e-3 + 3e-3 * Math.Abs(b));
            if (err > worst) { worst = err; worstAt = i; }
        }
        _out.WriteLine($"F16 hd{headDim} {numHeads}/{numKvHeads} seqQ {seqQ} offset {offset}: worst normalised error {worst:F3} at " +
                       $"{worstAt} ({HalfToFloat(flash[worstAt])} vs attention_f16 {HalfToFloat(reference[worstAt])})");
        Assert.True(worst <= 1.0, $"F16 flash prefill disagrees with attention_f16 at {worstAt}: " +
            $"{HalfToFloat(flash[worstAt])} vs {HalfToFloat(reference[worstAt])}");
    }

    /// <summary>
    /// Opt-in microbench (#1923, <c>DOTLLM_CUDA_KVQ_BENCH=1</c>): wall time per launch of the flash prefill kvq entries on one
    /// 1024-row chunk at depth (Qwen3-4B shape, 32/8/128), F16 vs Q8_0 vs Q4_0 K/V. Same-process relative figures only.
    /// Optional <c>DOTLLM_CUDA_KVQ_BENCH_DEPTH</c> (default 15360 cached rows before the chunk).
    /// </summary>
    [SkippableFact]
    public void FlashPrefillKvq_Microbench()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_CUDA_KVQ_BENCH") == "1", "DOTLLM_CUDA_KVQ_BENCH=1 not set");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null || !_kernels.HasAttentionFlashPrefillKvq(128, null, null), "flash prefill kvq kernel not loaded");
        int offset = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_CUDA_KVQ_BENCH_DEPTH"), out int d) ? d : 15360;
        const int seqQ = 1024, heads = 32, kvHeads = 8, hd = 128, reps = 10;
        int seqKv = offset + seqQ;
        foreach (var fmt in new[] { KvCacheDType.Q8_0, KvCacheDType.Q4_0 })
        {
            using var f = new Fixture(this, fmt, fmt, hd, heads, kvHeads, seqQ, seqKv, seqKv, window: 0, sinks: false);
            double Time(nint k, nint v, KvCacheDType? kf, KvCacheDType? vf)
            {
                for (int i = 0; i < 2; i++)
                    _kernels!.LaunchAttentionFlashPrefillKvq(f.Q, k, v, kf, vf, f.OutNew, seqQ, seqKv, heads, kvHeads, hd, offset, _stream!.Handle);
                _stream!.Synchronize();
                double best = double.MaxValue;
                for (int r = 0; r < 3; r++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < reps; i++)
                        _kernels!.LaunchAttentionFlashPrefillKvq(f.Q, k, v, kf, vf, f.OutNew, seqQ, seqKv, heads, kvHeads, hd, offset, _stream.Handle);
                    _stream.Synchronize();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / reps);
                }
                return best;
            }
            double tF16 = Time(f.KScratch, f.VScratch, null, null), tQ = Time(f.KQuant, f.VQuant, fmt, fmt);
            double tF16b = Time(f.KScratch, f.VScratch, null, null);
            _out.WriteLine($"depth {offset} + {seqQ} rows: f16f16 {tF16:F3} / {tF16b:F3} ms, {fmt} {tQ:F3} ms, {fmt}/f16 {tQ / Math.Min(tF16, tF16b):F3}");
        }
    }

    /// <summary>
    /// The quantize step that feeds both paths: one launch over a run of rows writes the same bytes as one launch per row (#1845's
    /// batching relies on it).
    /// </summary>
    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void QuantKv_OneLaunchPerRun_EqualsOneLaunchPerRow(KvCacheDType fmt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        const int rows = 37, stride = 256;
        int rowBytes = stride / 32 * (fmt == KvCacheDType.Q8_0 ? 34 : 18);
        var host = new ushort[rows * stride];
        var rng = new Random(5);
        for (int i = 0; i < host.Length; i++) host[i] = FloatToHalf((float)(rng.NextDouble() * 2 - 1) * (1 + i / stride % 3));
        nint src = Upload(host);
        CudaDriverApi.cuMemAlloc_v2(out nint a, (nuint)(rows * rowBytes)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint b, (nuint)(rows * rowBytes)).ThrowOnError();
        try
        {
            _kernels!.LaunchQuantKv(src, a, rows * stride, fmt, _stream!.Handle);
            for (int r = 0; r < rows; r++)
                _kernels.LaunchQuantKv(src + r * stride * 2, b + r * rowBytes, stride, fmt, _stream.Handle);
            _stream.Synchronize();
            Assert.True(ReadBytes(a, rows * rowBytes).AsSpan().SequenceEqual(ReadBytes(b, rows * rowBytes)));
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(src);
            CudaDriverApi.cuMemFree_v2(a);
            CudaDriverApi.cuMemFree_v2(b);
        }
    }

    /// <summary>Device buffers for one attention call: Q, quantized K/V, the oracle's dequantized scratch, the ring and outputs.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly List<nint> _allocs = [];
        public nint Q, KQuant, VQuant, KScratch, VScratch, KRing, VRing, OutRef, OutNew, OutScalar, Sinks;
        private readonly int _outElems;

        public Fixture(CudaAttentionKvqTests t, KvCacheDType kFmt, KvCacheDType vFmt, int headDim, int numHeads, int numKvHeads,
            int seqQ, int seqKv, int quantLen, int window, bool sinks, bool tinyBlocks = false)
        {
            int stride = numKvHeads * headDim;
            var rng = new Random(1850 + headDim + seqKv + quantLen);
            // True K/V history, block-varying amplitude; and "wrong" rows for regions that must not be read.
            float[] k = Rows(rng, seqKv, stride, 1.0f, tinyBlocks), v = Rows(rng, seqKv, stride, 0.8f, tinyBlocks);
            float[] kWrong = Rows(rng, seqKv, stride, 2.5f), vWrong = Rows(rng, seqKv, stride, 2.0f);

            var q = new ushort[seqQ * numHeads * headDim];
            for (int i = 0; i < q.Length; i++) q[i] = FloatToHalf((float)(rng.NextDouble() * 2 - 1));
            Q = Track(t.Upload(q));

            // Quantized buffer: rows < quantLen are the true history, later rows are wrong data (they live in the ring).
            KQuant = Track(t.Quantize(Mix(k, kWrong, stride, quantLen), seqKv, stride, kFmt));
            VQuant = Track(t.Quantize(Mix(v, vWrong, stride, quantLen), seqKv, stride, vFmt));

            // Oracle scratch, as PrepareAttentionScratch built it: dequantized rows [0, quantLen), then the FP16 window rows.
            KScratch = Track(t.Alloc((long)seqKv * stride * 2));
            VScratch = Track(t.Alloc((long)seqKv * stride * 2));
            if (quantLen > 0)
            {
                t._kernels!.LaunchKvDequantToF16(KQuant, kFmt, KScratch, quantLen * stride, t._stream!.Handle);
                t._kernels.LaunchKvDequantToF16(VQuant, vFmt, VScratch, quantLen * stride, t._stream.Handle);
            }
            if (quantLen < seqKv)
            {
                t.CopyHalfRows(k, quantLen, seqKv, stride, KScratch);
                t.CopyHalfRows(v, quantLen, seqKv, stride, VScratch);
            }

            if (window > 0)
            {
                // Ring: slot s holds row r (r >= quantLen, r % window == s); unused slots hold wrong rows.
                var kr = new ushort[window * stride];
                var vr = new ushort[window * stride];
                for (int s = 0; s < window; s++)
                    for (int i = 0; i < stride; i++)
                    {
                        kr[s * stride + i] = FloatToHalf(kWrong[s * stride + i]);
                        vr[s * stride + i] = FloatToHalf(vWrong[s * stride + i]);
                    }
                for (int r = quantLen; r < seqKv; r++)
                    for (int i = 0; i < stride; i++)
                    {
                        kr[r % window * stride + i] = FloatToHalf(k[r * stride + i]);
                        vr[r % window * stride + i] = FloatToHalf(v[r * stride + i]);
                    }
                KRing = Track(t.Upload(kr));
                VRing = Track(t.Upload(vr));
            }

            if (sinks)
            {
                var hs = new float[numHeads];
                for (int h = 0; h < numHeads; h++) hs[h] = 0.5f + h * 0.3f;
                Sinks = Track(t.Upload(hs));
            }

            _outElems = seqQ * numHeads * headDim;
            OutRef = Track(t.Alloc(_outElems * 2));
            OutNew = Track(t.Alloc(_outElems * 2));
            OutScalar = Track(t.Alloc(_outElems * 2));
        }

        public ushort[] Read(nint dev)
        {
            var h = new ushort[_outElems];
            fixed (ushort* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(_outElems * 2)).ThrowOnError();
            return h;
        }

        private nint Track(nint p) { _allocs.Add(p); return p; }

        public void Dispose()
        {
            foreach (nint p in _allocs) CudaDriverApi.cuMemFree_v2(p);
        }

        // tinyBlocks: every third 32-block has amplitude 2e-4, so its Q8_0 / Q4_0 scale d is an F16 subnormal (< 6.1e-5): the regime
        // where the packed-half2 decoder's mul.rn.f16x2 must still equal __float2half(d * (float)q) (#1923).
        private static float[] Rows(Random rng, int rows, int stride, float scale, bool tinyBlocks = false)
        {
            var a = new float[rows * stride];
            for (int r = 0; r < rows; r++)
                for (int i = 0; i < stride; i++)
                {
                    float amp = tinyBlocks && (r + i / 32) % 3 == 0 ? 2e-4f : scale * (0.25f + 0.5f * ((r * 7 + i / 32) % 5));
                    a[r * stride + i] = (float)(rng.NextDouble() * 2 - 1) * amp;
                }
            return a;
        }

        private static float[] Mix(float[] right, float[] wrong, int stride, int rightRows)
        {
            var a = (float[])wrong.Clone();
            Array.Copy(right, a, (long)rightRows * stride);
            return a;
        }
    }

    private nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)Math.Max(bytes, 2)).ThrowOnError();
        return p;
    }

    private nint Upload(ushort[] h)
    {
        nint p = Alloc(h.Length * 2L);
        fixed (ushort* s = h) CudaDriverApi.cuMemcpyHtoD_v2(p, (nint)s, (nuint)(h.Length * 2)).ThrowOnError();
        return p;
    }

    private nint Upload(float[] h)
    {
        nint p = Alloc(h.Length * 4L);
        fixed (float* s = h) CudaDriverApi.cuMemcpyHtoD_v2(p, (nint)s, (nuint)(h.Length * 4)).ThrowOnError();
        return p;
    }

    private nint Quantize(float[] rows, int count, int stride, KvCacheDType fmt)
    {
        var h = new ushort[rows.Length];
        for (int i = 0; i < h.Length; i++) h[i] = FloatToHalf(rows[i]);
        nint src = Upload(h);
        long bytes = (long)count * (stride / 32) * (fmt == KvCacheDType.Q8_0 ? 34 : 18);
        // 64 guard rows of 0xFF past the cache (#1923; 64 = the largest E4 key tile, #1943): every block scale there is an F16 NaN, so a kernel that reads past seqKv
        // (an unclamped tail tile) produces NaN instead of silently reading zeros.
        long guard = 64L * (stride / 32) * (fmt == KvCacheDType.Q8_0 ? 34 : 18);
        nint dst = Alloc(bytes + guard);
        CudaDriverApi.cuMemsetD8_v2(dst + (nint)bytes, 0xFF, (nuint)guard).ThrowOnError();
        _kernels!.LaunchQuantKv(src, dst, count * stride, fmt, _stream!.Handle);
        _stream.Synchronize();
        CudaDriverApi.cuMemFree_v2(src);
        return dst;
    }

    private void CopyHalfRows(float[] rows, int from, int to, int stride, nint dst)
    {
        var h = new ushort[(to - from) * stride];
        for (int i = 0; i < h.Length; i++) h[i] = FloatToHalf(rows[from * stride + i]);
        fixed (ushort* s = h)
            CudaDriverApi.cuMemcpyHtoD_v2(dst + (nint)((long)from * stride * 2), (nint)s, (nuint)(h.Length * 2)).ThrowOnError();
    }

    private static byte[] ReadBytes(nint dev, int bytes)
    {
        var h = new byte[bytes];
        fixed (byte* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)bytes).ThrowOnError();
        return h;
    }

    private static int FirstDiff(ushort[] a, ushort[] b)
    {
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return i;
        return -1;
    }

    private static void AssertFinite(ushort[] h)
    {
        for (int i = 0; i < h.Length; i++)
            Assert.True((h[i] & 0x7C00) != 0x7C00, $"non-finite output at {i}");
    }

    private static float HalfToFloat(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);

    private static ushort FloatToHalf(float f) => BitConverter.HalfToUInt16Bits((Half)f);

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
