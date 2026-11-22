using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1892 Task 6: the dense CUDA in-place TurboQuant attention (<see cref="CudaTurboQuantInPlaceAttention"/>: encode, rotate Q with
/// the K signs, <c>attention_kvtq</c> or the tensor-core <c>_tqtq</c> prefill, rotate O with the V signs) against the scratch
/// oracle (<see cref="CudaTurboQuantKvCache"/>: the same codes and the same F16 norm format, dequantized into an FP16 history
/// scratch, then <c>attention_f16</c>). Tolerance <c>2e-3 * sqrt(seqKv * headDim)</c> on every output element.
/// <para>Fixtures are built so the tolerance can fail: K and V sign seeds differ (V = K ^ <see cref="TurboQuantCodec.VSeedMask"/>),
/// every K / V row has its own amplitude (a dropped norm shows), Q is large enough for a peaked softmax and V large enough that the
/// oracle output RMS is well above the tolerance (asserted per case), bits 3 (codes straddle words) and 8 at every head dim, hd16
/// bits 3 (the only shape where padded and tight packing differ), several KV heads, offsets crossing the 256-row tile.</para>
/// Runs on framework (RTX 3060).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaTurboQuantInPlaceAttentionTests : IDisposable
{
    private const ulong Seed = 0x7A3C_91E5_2B44UL;
    private static readonly ConcurrentDictionary<(int, int, ulong), TurboQuantCodec> s_codecs = new();

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly ITestOutputHelper _out;

    public CudaTurboQuantInPlaceAttentionTests(ITestOutputHelper output)
    {
        _out = output;
        if (!IsCudaDriverPresent()) return;
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

    private void RequireKernels()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.If(_kernels is null, "PTX files not found");
        // A loaded module set without these is a stale build: fail, do not skip (a skipped CUDA test is not a pass).
        Assert.True(_kernels!.HasAttentionKvTq, "attention_kvtq not loaded (stale attention_kvtq.ptx?)");
        Assert.True(_kernels.TqRotateOF16Available, "tq_rotate_o_f16 not loaded (stale turboquant_rotate.ptx?)");
        Assert.True(_kernels.TqRotateAvailable && _kernels.TurboQuantF16Available, "TurboQuant codec / rotation PTX not loaded");
    }

    private static TurboQuantCodec Codec(int hd, int bits, ulong seed) => s_codecs.GetOrAdd((hd, bits, seed), k => new TurboQuantCodec(k.Item1, k.Item2, k.Item3));

    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)Math.Max(bytes, 4)).ThrowOnError(); return p; }
    private static void Free(nint p) { if (p != 0) CudaDriverApi.cuMemFree_v2(p); }
    private static void Up<T>(nint dst, T[] src) where T : unmanaged
    { fixed (T* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(T))).ThrowOnError(); }
    private static void Down<T>(T[] dst, nint src) where T : unmanaged
    { fixed (T* p = dst) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(dst.Length * sizeof(T))).ThrowOnError(); }

    private static float Gauss(Random r)
    {
        double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    /// <summary>[rows, heads*hd] half data; each (row, head) vector gets its own amplitude in [lo, hi] times <paramref name="scale"/>.</summary>
    private static Half[] Rows(Random r, int rows, int heads, int hd, float scale, float lo, float hi)
    {
        var a = new Half[rows * heads * hd];
        for (int v = 0; v < rows * heads; v++)
        {
            float amp = scale * (lo + (hi - lo) * (float)r.NextDouble());
            for (int i = 0; i < hd; i++) a[v * hd + i] = (Half)(Gauss(r) * amp);
        }
        return a;
    }

    private static float[] ToFloat(Half[] h) { var f = new float[h.Length]; for (int i = 0; i < h.Length; i++) f[i] = (float)h[i]; return f; }

    private static double Tolerance(int seqKv, int hd) => 2e-3 * Math.Sqrt((double)seqKv * hd);

    private sealed class CaseResult
    {
        public required float[] InPlace;
        public required float[] Oracle;
        public required float[] Fresh;      // attention_f16 over the unquantized K/V (rows never encoded)
        public required int SeqKv;
        public required bool Flash;
        public required long ScalarLaunches;
        public required long FlashLaunches;
        public Half[] K = [], V = [], Q = [];   // the host inputs, for the F64 reference (#1892 precision gate)
    }

    /// <summary>
    /// Encodes <paramref name="offset"/> history rows into both caches, then runs one attention call for <paramref name="seqQ"/>
    /// current rows: in place through <see cref="CudaTurboQuantInPlaceAttention.Run"/> (which encodes them), and through the oracle.
    /// </summary>
    private CaseResult RunCase(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset, int sliding, bool sinks,
                               bool flashEnabled, int seed, float kScale = 1f, float vScale = 2f, float qScale = 2.5f,
                               float histKScale = 1f, TqNormFormat? normFormat = null, float kBias = 0f)
    {
        var codecK = Codec(hd, bits, Seed);
        var codecV = Codec(hd, bits, Seed ^ TurboQuantCodec.VSeedMask);
        Assert.False(codecK.RotationSigns.SequenceEqual(codecV.RotationSigns), "fixture: K and V signs must differ");

        int total = offset + seqQ;
        int kvStride = numKvHeads * hd, qStride = numHeads * hd;
        var rng = new Random(seed);
        Half[] k = Rows(rng, total, numKvHeads, hd, kScale, 0.5f, 2f);
        for (int i = 0; i < offset * kvStride; i++) k[i] = (Half)((float)k[i] * histKScale);
        if (kBias != 0f)
        {
            // Qwen-style K: a large constant bias on a few coordinates of every row (the same in every row and head).
            for (int row = 0; row < total * numKvHeads; row++)
                for (int i = 0; i < 4; i++) k[row * hd + 3 * i + 1] = (Half)((float)k[row * hd + 3 * i + 1] + ((i & 1) == 0 ? kBias : -kBias));
        }
        Half[] v = Rows(rng, total, numKvHeads, hd, vScale, 0.5f, 2f);
        Half[] q = Rows(rng, seqQ, numHeads, hd, qScale, 1f, 1f);
        float[] sinkVals = new float[numHeads];
        for (int i = 0; i < numHeads; i++) sinkVals[i] = Gauss(rng) * 2f;
        int[] positions = new int[total];
        for (int i = 0; i < total; i++) positions[i] = i;

        long rowBytes = (long)kvStride * sizeof(ushort);
        nint dK = 0, dV = 0, dQ = 0, dQo = 0, dQf = 0, dOut = 0, dOutO = 0, dOutF = 0, dSinks = 0;
        try
        {
            dK = Alloc(total * rowBytes); dV = Alloc(total * rowBytes);
            dQ = Alloc((long)seqQ * qStride * 2); dQo = Alloc((long)seqQ * qStride * 2); dQf = Alloc((long)seqQ * qStride * 2);
            dOut = Alloc((long)seqQ * qStride * 2); dOutO = Alloc((long)seqQ * qStride * 2); dOutF = Alloc((long)seqQ * qStride * 2);
            dSinks = Alloc(numHeads * sizeof(float));
            Up(dK, k); Up(dV, v); Up(dQ, q); Up(dQo, q); Up(dQf, q); Up(dSinks, sinkVals);
            nint sinksPtr = sinks ? dSinks : 0;
            nint s = _stream!.Handle;

            using var ip = new CudaTurboQuantInPlaceKvCache(1, numKvHeads, hd, total + 3, bits,
                codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD, normFormat);
            using var oracle = new CudaTurboQuantKvCache(1, numKvHeads, hd, total + 3, bits,
                codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD, normFormat);
            Assert.Equal(ip.NormFormat, oracle.NormFormat);
            if (normFormat is { } nf) Assert.Equal(nf, ip.NormFormat);

            if (offset > 0)
            {
                ip.UpdateDevice(dK, dV, positions.AsSpan(0, offset), offset, 0, s, _kernels!);
                oracle.UpdateDevice(dK, dV, positions.AsSpan(0, offset), offset, 0, s, _kernels!);
            }

            long scalar0 = CudaKernels.AttentionKvTqLaunches, flash0 = CudaKernels.AttentionFlashPrefillKvTqLaunches;
            bool flash = CudaTurboQuantInPlaceAttention.Run(_kernels!, ip, 0, dQ, dK + (nint)(offset * rowBytes), dV + (nint)(offset * rowBytes),
                positions.AsSpan(offset, seqQ), seqQ, numHeads, numKvHeads, hd, sliding, sinksPtr, dOut, s, flashEnabled);
            long scalarRuns = CudaKernels.AttentionKvTqLaunches - scalar0, flashRuns = CudaKernels.AttentionFlashPrefillKvTqLaunches - flash0;
            Assert.Equal(total, ip.CurrentLength);

            oracle.UpdateDevice(dK + (nint)(offset * rowBytes), dV + (nint)(offset * rowBytes), positions.AsSpan(offset, seqQ), seqQ, 0, s, _kernels!);
            var (kS, vS) = oracle.PrepareAttentionScratch(0, s, _kernels!);
            _kernels!.LaunchAttention(dQo, kS, vS, dOutO, seqQ, total, numHeads, numKvHeads, hd, offset, sliding, s, sinksPtr);
            _kernels.LaunchAttention(dQf, dK, dV, dOutF, seqQ, total, numHeads, numKvHeads, hd, offset, sliding, s, sinksPtr);
            _stream.Synchronize();

            var got = new Half[seqQ * qStride]; var exp = new Half[seqQ * qStride]; var fresh = new Half[seqQ * qStride];
            Down(got, dOut); Down(exp, dOutO); Down(fresh, dOutF);
            return new CaseResult
            {
                InPlace = ToFloat(got), Oracle = ToFloat(exp), Fresh = ToFloat(fresh), SeqKv = total, Flash = flash,
                ScalarLaunches = scalarRuns, FlashLaunches = flashRuns, K = k, V = v, Q = q,
            };
        }
        finally
        {
            Free(dK); Free(dV); Free(dQ); Free(dQo); Free(dQf); Free(dOut); Free(dOutO); Free(dOutF); Free(dSinks);
        }
    }

    private static (double maxAbs, int at, double rmsRef) Compare(float[] got, float[] reference)
    {
        double max = 0, ss = 0; int at = -1;
        for (int i = 0; i < got.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"non-finite output at {i}");
            double d = Math.Abs(got[i] - reference[i]);
            if (d > max) { max = d; at = i; }
            ss += (double)reference[i] * reference[i];
        }
        return (max, at, Math.Sqrt(ss / got.Length));
    }

    private void AssertMatchesOracle(CaseResult r, int hd, string label)
    {
        double tol = Tolerance(r.SeqKv, hd);
        var (max, at, rms) = Compare(r.InPlace, r.Oracle);
        _out.WriteLine($"{label}: seqKv={r.SeqKv} flash={r.Flash} maxAbs={max:E3} (at {at}) tol={tol:E3} oracleRms={rms:F3}");
        // Precondition: the oracle output must dwarf the tolerance, else a wrong kernel could hide inside it.
        Assert.True(rms > 2 * tol, $"{label}: fixture too weak, oracle RMS {rms:F4} <= 2 x tol {tol:F4}");
        Assert.True(max <= tol, $"{label}: in-place vs oracle maxAbs {max:E4} > tol {tol:E4} at element {at} " +
                                $"(in-place {r.InPlace[at]}, oracle {r.Oracle[at]})");
    }

    // ── F16 output rotation (D1) ─────────────────────────────────────────────────────────────────────────

    public static TheoryData<int, int, int> RotateGeometries => new() { { 16, 3, 2 }, { 64, 3, 2 }, { 128, 8, 4 }, { 256, 2, 3 } };

    [SkippableTheory]
    [MemberData(nameof(RotateGeometries))]
    public void RotateOutputF16_InvertsCodec(int headDim, int numHeads, int rows)
    {
        RequireKernels();
        var codecV = Codec(headDim, 4, Seed ^ TurboQuantCodec.VSeedMask);
        var rng = new Random(headDim * 7 + rows);
        int n = rows * numHeads * headDim;
        Half[] a = Rows(rng, rows, numHeads, headDim, 2f, 0.5f, 2f);
        float[] expected = ToFloat(a);
        for (int vec = 0; vec < rows * numHeads; vec++) codecV.InverseRotate(expected.AsSpan(vec * headDim, headDim));

        float[] sv = codecV.RotationSigns.ToArray();
        // Two sentinel rows after the live ones must stay untouched.
        int extra = 2 * numHeads * headDim;
        var init = new Half[n + extra];
        Array.Copy(a, init, n);
        for (int i = n; i < init.Length; i++) init[i] = (Half)123.5f;
        nint dO = Alloc(init.Length * 2), dS = Alloc(sv.Length * 4);
        try
        {
            Up(dO, init); Up(dS, sv);
            _kernels!.LaunchTqRotateOF16(dO, rows, numHeads, headDim, dS, _stream!.Handle);
            _stream.Synchronize();
            var got = new Half[init.Length];
            Down(got, dO);
            double max = 0;
            for (int i = 0; i < n; i++)
            {
                double d = Math.Abs(expected[i] - (float)got[i]);
                max = Math.Max(max, d);
                // one F16 rounding of the result (2^-11 relative) plus F32 butterfly drift
                Assert.True(d <= 1e-3 * Math.Abs(expected[i]) + 2e-3, $"i={i}: codec={expected[i]} gpu={(float)got[i]}");
            }
            for (int i = n; i < init.Length; i++) Assert.Equal(123.5f, (float)got[i]);
            Assert.True(TurboQuantRotationTestsMaxDiff(ToFloat(a), expected, n) > 0.1, "fixture: rotation must change the data");
            _out.WriteLine($"hd={headDim}: maxAbsDiff={max:E3}");
        }
        finally { Free(dO); Free(dS); }
    }

    private static double TurboQuantRotationTestsMaxDiff(float[] a, float[] b, int n)
    {
        double m = 0; for (int i = 0; i < n; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m;
    }

    // ── Scalar arm (attention_kvtq) ──────────────────────────────────────────────────────────────────────

    public static IEnumerable<object[]> ScalarCases() =>
    [
        // bits, hd, numHeads, numKvHeads, seqQ, offset, slidingWindow, sinks
        [3, 16, 4, 2, 1, 40, 0, false],     // hd16 bits 3: 48 code bits in 2 words -- the padded-vs-tight packing discriminator
        [3, 16, 4, 2, 7, 33, 0, true],
        [8, 16, 4, 2, 1, 20, 0, false],
        [2, 64, 6, 2, 1, 300, 0, false],    // crosses the 256-row tile
        [3, 64, 6, 2, 7, 120, 0, true],
        [4, 64, 6, 2, 16, 0, 0, false],     // a cold 16-row prefill on the scalar arm (flash disabled)
        [8, 64, 6, 2, 64, 37, 0, false],
        [3, 64, 8, 8, 1, 90, 25, false],    // MHA, sliding window
        [2, 128, 8, 2, 7, 0, 0, false],
        [3, 128, 8, 2, 1, 301, 0, false],
        [4, 128, 8, 2, 1, 100, 50, false],  // sliding window decode
        [8, 128, 8, 2, 16, 40, 0, true],
        [3, 128, 16, 1, 7, 50, 0, false],   // group 16 (> 8: never the tensor-core arm)
        [2, 256, 4, 2, 1, 90, 0, false],
        [3, 256, 4, 2, 7, 50, 0, false],
        [4, 256, 8, 2, 64, 0, 0, false],
        [8, 256, 4, 1, 16, 20, 0, true],
        [3, 256, 4, 2, 1, 60, 20, false],
        [5, 64, 6, 2, 7, 45, 0, false],     // bits 5/6/7: codes straddle a 32-bit word (64-bit buffer refill, V straddle)
        [6, 128, 8, 2, 1, 70, 0, true],
        [7, 128, 8, 2, 16, 33, 0, false],
    ];

    [SkippableTheory]
    [MemberData(nameof(ScalarCases))]
    public void Scalar_MatchesOracle(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset, int sliding, bool sinks)
    {
        RequireKernels();
        var r = RunCase(bits, hd, numHeads, numKvHeads, seqQ, offset, sliding, sinks, flashEnabled: false,
                        seed: bits * 1000 + hd * 10 + seqQ);
        Assert.False(r.Flash);
        Assert.Equal(1, r.ScalarLaunches);
        Assert.Equal(0, r.FlashLaunches);
        AssertMatchesOracle(r, hd, $"scalar b{bits} hd{hd} {numHeads}/{numKvHeads} q{seqQ}@{offset} sw{sliding} sinks={sinks}");
    }

    // ── Tensor-core prefill arm (attention_flash_mma_prefill_kvq_hd{64,128}_tqtq) ─────────────────────────

    public static IEnumerable<object[]> FlashCases() =>
    [
        // bits, hd, numHeads, numKvHeads, seqQ, offset
        [3, 64, 6, 2, 16, 0],
        [8, 64, 6, 2, 64, 45],
        [2, 64, 9, 3, 40, 5],       // SmolLM-like group 3, a partial q-tile
        [4, 128, 8, 2, 16, 77],
        [2, 128, 8, 2, 64, 0],
        [3, 128, 32, 4, 64, 10],    // group 8: 52 KB dynamic shared memory
        [8, 128, 8, 8, 16, 0],      // MHA
        [3, 128, 8, 2, 64, 260],    // history past one 256-row span
        [7, 128, 8, 2, 16, 20],     // b7: shift + 8b > 64, the third-word read
        [5, 64, 6, 2, 16, 9],
    ];

    [SkippableTheory]
    [MemberData(nameof(FlashCases))]
    public void FlashPrefill_MatchesOracle(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        RequireKernels();
        Assert.True(_kernels!.HasAttentionFlashPrefillKvTq(hd), $"tensor-core TurboQuant prefill hd{hd} not loaded (stale PTX / pre-Ampere?)");
        var r = RunCase(bits, hd, numHeads, numKvHeads, seqQ, offset, 0, false, flashEnabled: true,
                        seed: 77 + bits * 100 + hd + seqQ);
        Assert.True(r.Flash, "the tensor-core arm must be the one dispatched for this shape");
        Assert.Equal(1, r.FlashLaunches);
        Assert.Equal(0, r.ScalarLaunches);
        AssertMatchesOracle(r, hd, $"flash b{bits} hd{hd} {numHeads}/{numKvHeads} q{seqQ}@{offset}");
    }

    /// <summary>
    /// The non-default norm formats (F32Reference = 0, Bf16 = 2) through both arms: kvtq_load_norm / pk_tq_norm branch on the format,
    /// and the F16 default alone never reaches the other branches.
    /// </summary>
    [SkippableTheory]
    [InlineData(TqNormFormat.F32Reference, false)]
    [InlineData(TqNormFormat.Bf16, false)]
    [InlineData(TqNormFormat.F32Reference, true)]
    [InlineData(TqNormFormat.Bf16, true)]
    public void NormFormats_MatchOracle(TqNormFormat fmt, bool flash)
    {
        RequireKernels();
        var r = flash
            ? RunCase(4, 128, 8, 2, 16, 30, 0, false, flashEnabled: true, seed: 5150 + (int)fmt, normFormat: fmt)
            : RunCase(3, 64, 6, 2, 1, 80, 0, true, flashEnabled: false, seed: 5160 + (int)fmt, normFormat: fmt);
        Assert.Equal(flash, r.Flash);
        Assert.Equal(flash ? 1 : 0, r.FlashLaunches);
        Assert.Equal(flash ? 0 : 1, r.ScalarLaunches);
        AssertMatchesOracle(r, flash ? 128 : 64, $"norm {fmt} {(flash ? "flash" : "scalar")}");
    }

    /// <summary>D3: a dense hd256 prefill (no tensor-core TurboQuant twin) takes the scalar TQ body, not a dequant path, and matches.</summary>
    [SkippableTheory]
    [InlineData(3, 16, 0)]
    [InlineData(8, 64, 30)]
    public void Hd256Prefill_TakesScalarPath_MatchesOracle(int bits, int seqQ, int offset)
    {
        RequireKernels();
        var r = RunCase(bits, 256, 8, 2, seqQ, offset, 0, false, flashEnabled: true, seed: 900 + bits + seqQ);
        Assert.False(r.Flash);
        Assert.Equal(1, r.ScalarLaunches);
        Assert.Equal(0, r.FlashLaunches);
        AssertMatchesOracle(r, 256, $"hd256 prefill b{bits} q{seqQ}@{offset}");
    }

    /// <summary>
    /// The current chunk's rows are attended from the cache (codes), never from the fresh K/V. At 2 bits, with the current rows
    /// dominating the softmax (small history keys), attention over the fresh rows is far outside the tolerance of the oracle
    /// (asserted: the fixture can tell the two apart), while the in-place output matches the oracle. seqQ 16 is the tensor-core arm
    /// at offset 0 (a cold prefill, where every row is current); seqQ 7 is the scalar arm over a short history.
    /// </summary>
    [SkippableTheory]
    [InlineData(16, 0, true)]
    [InlineData(7, 3, false)]
    public void CurrentChunkRows_ComeFromTheCache(int seqQ, int offset, bool expectFlash)
    {
        RequireKernels();
        var r = RunCase(2, 64, 6, 2, seqQ, offset, 0, false, flashEnabled: true, seed: 4242 + seqQ, histKScale: 0.1f);
        Assert.Equal(expectFlash, r.Flash);
        double tol = Tolerance(r.SeqKv, 64);
        var (freshMax, _, _) = Compare(r.Fresh, r.Oracle);
        _out.WriteLine($"fresh-K/V attention vs oracle maxAbs={freshMax:E3} (tol {tol:E3})");
        Assert.True(freshMax > 4 * tol, $"fixture cannot tell fresh rows from cached codes: {freshMax:E3} <= 4 x {tol:E3}");
        AssertMatchesOracle(r, 64, $"current-chunk q{seqQ}@{offset}");
        var (ipVsFresh, _, _) = Compare(r.InPlace, r.Fresh);
        Assert.True(ipVsFresh > 2 * tol, $"in-place output tracks the fresh K/V ({ipVsFresh:E3}), not the cache");
    }

    // ── Model level: the CudaTransformerModel wiring, decode stays eager (D2) ─────────────────────────────

    /// <summary>
    /// SmolLM-135M (hd64, 9/3 heads) through <see cref="CudaTransformerModel"/> with the in-place cache from <c>CreateKvCache</c>:
    /// a 16+ token prefill runs the tensor-core TurboQuant arm on every layer, and each decode step runs eager (graph capture never
    /// engages for this cache type, D2) on the scalar arm; logits track the scratch oracle (same codec constants) at every step.
    /// Precondition: graph capture is on and a plain F16 cache's decode does get captured, so "not captured" is not vacuous.
    /// Bits 4 and 8 only: at tq3 SmolLM is so degraded (oracle-vs-F16 cos 0.79 by decode step 1) that rounding-point differences
    /// between two correct paths flip codes downstream and cos(in-place, oracle) drops to 0.98 with every argmax still equal
    /// (measured on framework, #1892 Task 6); bits 3 is gated strictly at the kernel level instead.
    /// </summary>
    [SkippableTheory]
    [InlineData(4)]
    [InlineData(8)]
    public void TqDecode_RunsEager_NotCaptured_MatchesOracle(int bits)
    {
        RequireKernels();
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode("The quick brown fox jumps over the lazy dog, and then the capital of France is Paris while the capital of Germany is");
        Assert.True(prompt.Length >= CudaTransformerModel.KvqFlashPrefillMinRows, $"prompt has only {prompt.Length} tokens");
        const int decodeSteps = 3;
        int maxSeq = prompt.Length + decodeSteps + 2;
        int layers = config.NumLayers, vocab = config.VocabSize;
        string? ptxDir = FindPtxDir();

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptxDir!);
        Skip.IfNot(model.UseGraphCapture, "graph capture disabled in this environment (DOTLLM_DISABLE_GRAPH_CAPTURE)");
        Assert.True(CudaKernels.IsAttentionFlashPrefillKvqShape(config.NumAttentionHeads, config.NumKvHeads, config.HeadDim));

        // Control: the F16 cache's decode IS captured.
        using (var f16 = model.CreateKvCache(maxSeq))
        {
            int cap0 = model.GraphCaptureCount;
            Forward(model, f16, prompt, 0).Dispose();
            int next = prompt.Length;
            for (int i = 0; i < 2; i++) Forward(model, f16, [1], next++).Dispose();
            Assert.True(model.GraphCaptureCount > cap0, "control: an F16 decode must engage graph capture");
        }

        var kvConfig = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, bits, Seed);
        var codecK = Codec(config.HeadDim, bits, Seed);
        var codecV = Codec(config.HeadDim, bits, Seed ^ TurboQuantCodec.VSeedMask);
        using var ipCache = model.CreateKvCache(maxSeq, kvConfig);
        Assert.IsType<CudaTurboQuantInPlaceKvCache>(ipCache);
        using var oracle = model.CreateTurboQuantKvCache(maxSeq, codecK.MseBits, codecK.Centroids, codecK.RotationSigns,
                                                          codecV.RotationSigns, codecK.InvSqrtD);

        long flash0 = CudaKernels.AttentionFlashPrefillKvTqLaunches, scalar0 = CudaKernels.AttentionKvTqLaunches;
        int capBefore = model.GraphCaptureCount, eagerBefore = model.EagerDecodeCount;

        var ipRows = new List<float[]>(); var orRows = new List<float[]>(); var fed = new List<int>();
        using (var l = Forward(model, ipCache, prompt, 0)) ipRows.Add(LastRow(l, vocab));
        using (var l = Forward(model, oracle, prompt, 0)) orRows.Add(LastRow(l, vocab));
        Assert.Equal(layers, CudaKernels.AttentionFlashPrefillKvTqLaunches - flash0);
        Assert.Equal(0, CudaKernels.AttentionKvTqLaunches - scalar0);

        int pos = prompt.Length;
        for (int step = 0; step < decodeSteps; step++, pos++)
        {
            int token = Argmax(ipRows[^1]);   // teacher-forced: both caches see the same token
            fed.Add(token);
            using (var l = Forward(model, ipCache, [token], pos)) ipRows.Add(LastRow(l, vocab));
            using (var l = Forward(model, oracle, [token], pos)) orRows.Add(LastRow(l, vocab));
        }

        Assert.Equal(capBefore, model.GraphCaptureCount);
        // Both the in-place and the oracle decodes are eager.
        Assert.Equal(eagerBefore + 2 * decodeSteps, model.EagerDecodeCount);
        Assert.Equal(decodeSteps * layers, CudaKernels.AttentionKvTqLaunches - scalar0);
        Assert.Equal(layers, CudaKernels.AttentionFlashPrefillKvTqLaunches - flash0);
        Assert.Equal(prompt.Length + decodeSteps, ipCache.CurrentLength);

        // F16-KV reference over the same tokens (run last: its decode is graph-captured).
        var f16Rows = new List<float[]>();
        using (var f16 = model.CreateKvCache(maxSeq))
        {
            using (var l = Forward(model, f16, prompt, 0)) f16Rows.Add(LastRow(l, vocab));
            for (int step = 0; step < decodeSteps; step++)
                using (var l = Forward(model, f16, [fed[step]], prompt.Length + step)) f16Rows.Add(LastRow(l, vocab));
        }
        for (int i = 0; i <= decodeSteps; i++)
            AssertLogitsTrack(ipRows[i], orRows[i], f16Rows[i], i == 0 ? "prefill" : $"decode {i - 1}", MinCos(bits));
    }

    /// <summary>
    /// Per-bit floor on cos(in-place, oracle), set about 3x the measured 1 - cos (framework RTX 3060, a965cd33: worst step tq4
    /// 0.999080, tq8 0.999940).
    /// </summary>
    private static double MinCos(int bits) => bits switch { 8 => 0.9998, 4 => 0.997, _ => throw new ArgumentOutOfRangeException(nameof(bits)) };

    /// <summary>
    /// Diagnostic, not a gate (#1892 Task 6 fix round): the drift floor between two correct paths. It does not by itself justify
    /// leaving tq3 out of the model-level gate: that rests on the model (SmolLM-135M is degraded at 3 bits, oracle vs F16 logit cosine
    /// about 0.79, so the codec dominates the comparison) and on the tq4 and tq8 evidence (the 1-ulp control at tq4 is below the
    /// in-place-vs-oracle distance, tq8 cosine 0.99994), with tq3 gated strictly at kernel level. The tq3 numbers here are a lower
    /// bound, not proof of parity. Every variant is teacher-forced on the scratch oracle's greedy tokens.
    /// <list type="bullet">
    /// <item>oracle vs oracle with invSqrtD one F32 ulp up (every encode input moves by about 1 ulp);</item>
    /// <item>in-place vs in-place with the last bit of one F16 key norm flipped (layer 0, position 0, head 0) after prefill;</item>
    /// <item>in-place vs oracle, and each one's distance to F16 KV.</item>
    /// </list>
    /// </summary>
    [SkippableTheory]
    [InlineData(3)]
    [InlineData(4)]
    public void Diagnostic_DriftFloor_OneUlpControl(int bits)
    {
        RequireKernels();
        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_SMOLLM_Q8_GGUF")
            ?? TestFixtureResolver.ResolveFile([], "QuantFactory", "SmolLM-135M-GGUF", ["SmolLM-135M.Q8_0.gguf"], null).Path ?? "";
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] prompt = tokenizer.Encode("The quick brown fox jumps over the lazy dog, and then the capital of France is Paris while the capital of Germany is");
        const int steps = 3;
        int maxSeq = prompt.Length + steps + 2, vocab = config.VocabSize;
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, FindPtxDir()!);

        var kvConfig = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, 0, bits, Seed);
        var codecK = Codec(config.HeadDim, bits, Seed);
        var codecV = Codec(config.HeadDim, bits, Seed ^ TurboQuantCodec.VSeedMask);
        float invUp = MathF.BitIncrement(codecK.InvSqrtD);
        Assert.NotEqual(codecK.InvSqrtD, invUp);

        var fed = new List<int>();
        List<float[]> oracle, oracleUlp, ip, ipFlip, f16;
        using (var c = model.CreateTurboQuantKvCache(maxSeq, codecK.MseBits, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD))
            oracle = RunSeq(model, c, prompt, fed, steps, vocab);
        using (var c = model.CreateTurboQuantKvCache(maxSeq, codecK.MseBits, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, invUp))
            oracleUlp = RunSeq(model, c, prompt, fed, steps, vocab);
        using (var c = model.CreateKvCache(maxSeq, kvConfig)) ip = RunSeq(model, c, prompt, fed, steps, vocab);
        using (var c = (CudaTurboQuantInPlaceKvCache)model.CreateKvCache(maxSeq, kvConfig))
        {
            Assert.Equal(TqNormFormat.F16, c.NormFormat);
            ipFlip = RunSeq(model, c, prompt, fed, steps, vocab, afterPrefill: () =>
            {
                var h = new ushort[1];
                Down(h, c.GetKeyNormsPtr(0));
                h[0] ^= 1;
                Up(c.GetKeyNormsPtr(0), h);
            });
        }
        using (var c = model.CreateKvCache(maxSeq)) f16 = RunSeq(model, c, prompt, fed, steps, vocab);

        for (int i = 0; i <= steps; i++)
        {
            string label = i == 0 ? "prefill" : $"decode {i - 1}";
            _out.WriteLine($"DRIFT b{bits} {label}: cos(oracle,oracle+ulp(invSqrtD))={Cos(oracle[i], oracleUlp[i]).cos:F6} " +
                           $"cos(ip,ip+normLsbFlip)={Cos(ip[i], ipFlip[i]).cos:F6} cos(ip,oracle)={Cos(ip[i], oracle[i]).cos:F6} " +
                           $"cos(ip,f16)={Cos(ip[i], f16[i]).cos:F6} cos(oracle,f16)={Cos(oracle[i], f16[i]).cos:F6} " +
                           $"argmax oracle={Argmax(oracle[i])} ulp={Argmax(oracleUlp[i])} ip={Argmax(ip[i])} flip={Argmax(ipFlip[i])} f16={Argmax(f16[i])}");
        }
        // Precondition only: the flip lands after prefill, so the two in-place prefill rows are the same computation.
        Assert.Equal(1.0, Cos(ip[0], ipFlip[0]).cos, 6);
    }

    private static List<float[]> RunSeq(CudaTransformerModel model, DotLLM.Core.Attention.IKvCache cache, int[] prompt, List<int> fed,
                                        int steps, int vocab, Action? afterPrefill = null)
    {
        var rows = new List<float[]>();
        using (var l = Forward(model, cache, prompt, 0)) rows.Add(LastRow(l, vocab));
        afterPrefill?.Invoke();
        for (int s = 0; s < steps; s++)
        {
            if (fed.Count <= s) fed.Add(Argmax(rows[^1]));
            using var l = Forward(model, cache, [fed[s]], prompt.Length + s);
            rows.Add(LastRow(l, vocab));
        }
        return rows;
    }

    private static DotLLM.Core.Tensors.ITensor Forward(CudaTransformerModel model, DotLLM.Core.Attention.IKvCache cache, int[] tokens, int start)
    {
        var positions = new int[tokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = start + i;
        return model.Forward(tokens, positions, 0, cache);
    }

    private static (double cos, double maxAbs) Cos(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0, max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(float.IsFinite(a[i]), $"non-finite logit {i}");
            dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }
        return (dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30), max);
    }

    /// <summary>
    /// In-place vs the scratch oracle across 30 layers: the two paths round at different points, and a 1-ulp change in a
    /// later layer's K/V input can flip a TurboQuant code (quantization digitizes ULP drift), so whole-model logits are not
    /// held to the kernel tolerance. Gates: cos(in-place, oracle) > <paramref name="minCos"/> (per bit, <see cref="MinCos"/>), the argmax agrees unless the oracle's top-2 gap is
    /// below 0.1, and the in-place path is no farther from the F16-KV logits than about the oracle is
    /// (1 - cos_ip_f16 &lt;= 2 (1 - cos_oracle_f16) + 1e-3). A wrong kernel or rotation destroys all three.
    /// </summary>
    private void AssertLogitsTrack(float[] got, float[] oracle, float[] f16, string label, double minCos)
    {
        var (cos, max) = Cos(got, oracle);
        var (cosIpF16, _) = Cos(got, f16);
        var (cosOrF16, _) = Cos(oracle, f16);
        int ga = Argmax(got), oa = Argmax(oracle);
        float[] sorted = (float[])oracle.Clone(); Array.Sort(sorted);
        double gap = sorted[^1] - sorted[^2];
        _out.WriteLine($"{label}: cos(ip,oracle)={cos:F6} maxAbsLogit={max:F4} cos(ip,f16)={cosIpF16:F6} cos(oracle,f16)={cosOrF16:F6} " +
                       $"argmax ip={ga} oracle={oa} f16={Argmax(f16)} oracleTop2Gap={gap:F4}");
        Assert.True(cos > minCos, $"{label}: logits diverged from the oracle, cos={cos:F6} <= {minCos}");
        Assert.True(1 - cosIpF16 <= 2 * (1 - cosOrF16) + 1e-3,
            $"{label}: in-place is farther from F16 KV (cos {cosIpF16:F6}) than the oracle is (cos {cosOrF16:F6})");
        if (gap > 0.1) Assert.Equal(oa, ga);
    }

    private static float[] LastRow(DotLLM.Core.Tensors.ITensor logits, int vocab)
    {
        Assert.Equal(1, logits.Shape[0]);
        var row = new float[vocab];
        float* p = (float*)logits.DataPointer;
        for (int i = 0; i < vocab; i++) row[i] = p[i];
        return row;
    }

    private static int Argmax(float[] row)
    {
        int best = 0; float bv = row[0];
        for (int i = 1; i < row.Length; i++) if (row[i] > bv) { bv = row[i]; best = i; }
        return best;
    }

    // ── Precision against an F64 reference (#1892: CUDA in-place vs oracle floor) ─────────────────────────────────────────
    //
    // The oracle comparison above has a tolerance of 2e-3 * sqrt(seqKv * hd) (about 0.2 absolute at these shapes), wide enough to
    // pass a query or output rotation stored at BF16 precision. These cases measure each path against an F64 attention over the
    // SAME decoded history (the CPU codec is byte-exact with the CUDA encode; the norm is rounded to F16 as the cache stores it), so
    // the codec error cancels and what is left is each path's own rounding. Pre-registered gate (set before the first measurement):
    // the in-place error must not exceed twice the oracle's (both round to F16 three times, at different points) and its mean error
    // must stay inside 4 F16 half-ulps of the mean output magnitude (4 * 2^-11 = 1.95e-3).

    /// <summary>F64 causal attention over the codec-decoded K / V (F16 norms), [seqQ, numHeads*hd].</summary>
    private static double[] ReferenceF64(CaseResult r, int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        var codecK = Codec(hd, bits, Seed);
        var codecV = Codec(hd, bits, Seed ^ TurboQuantCodec.VSeedMask);
        int total = offset + seqQ, nv = total * numKvHeads;
        var dk = new float[nv * hd]; var dv = new float[nv * hd];
        var codes = new byte[codecK.CodeBytesPerVector]; var vec = new float[hd];
        for (int x = 0; x < nv; x++)
        {
            for (int i = 0; i < hd; i++) vec[i] = (float)r.K[x * hd + i];
            float n = (float)(Half)codecK.Encode(vec, codes);
            codecK.Decode(codes, n, dk.AsSpan(x * hd, hd));
            for (int i = 0; i < hd; i++) vec[i] = (float)r.V[x * hd + i];
            n = (float)(Half)codecV.Encode(vec, codes);
            codecV.Decode(codes, n, dv.AsSpan(x * hd, hd));
        }
        int g = numHeads / numKvHeads; double scale = 1.0 / Math.Sqrt(hd);
        var o = new double[seqQ * numHeads * hd]; var sc = new double[total]; var acc = new double[hd];
        for (int t = 0; t < seqQ; t++)
            for (int h = 0; h < numHeads; h++)
            {
                int hk = h / g, pq = offset + t; double mx = double.NegativeInfinity;
                int qo = (t * numHeads + h) * hd;
                for (int j = 0; j <= pq; j++)
                {
                    double s = 0; int ko = (j * numKvHeads + hk) * hd;
                    for (int i = 0; i < hd; i++) s += (double)(float)r.Q[qo + i] * dk[ko + i];
                    sc[j] = s * scale; mx = Math.Max(mx, sc[j]);
                }
                double sum = 0; Array.Clear(acc);
                for (int j = 0; j <= pq; j++)
                {
                    double w = Math.Exp(sc[j] - mx); sum += w; int vo = (j * numKvHeads + hk) * hd;
                    for (int i = 0; i < hd; i++) acc[i] += w * dv[vo + i];
                }
                for (int i = 0; i < hd; i++) o[qo + i] = acc[i] / sum;
            }
        return o;
    }

    private static (double meanRel, double maxAbs) ErrVs(float[] got, double[] reference)
    {
        double se = 0, sr = 0, mx = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double e = Math.Abs(got[i] - reference[i]);
            se += e; sr += Math.Abs(reference[i]); mx = Math.Max(mx, e);
        }
        return (se / sr, mx);
    }

    public static TheoryData<int, int, int, bool, float> PrecisionCases => new()
    {
        // bits, seqQ, offset, (Qwen3-4B geometry: hd 128, 32/8 heads), flash, K bias
        { 8, 1, 120, false, 0f },
        { 8, 1, 120, false, 60f },
        { 8, 64, 0, true, 0f },
        { 8, 64, 0, true, 60f },
        { 8, 64, 0, false, 60f },
        { 4, 1, 120, false, 60f },
        { 4, 64, 0, true, 60f },
    };

    /// <summary>
    /// Each path (in place, oracle) against the F64 attention over the same decoded history. Guards a precision defect in the
    /// in-place Q / O rotations or the code-space kernels that the oracle tolerance cannot see.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(PrecisionCases))]
    public void Precision_InPlaceWithinF16BudgetOfF64Reference(int bits, int seqQ, int offset, bool flash, float kBias)
    {
        RequireKernels();
        const int hd = 128, nh = 32, nkv = 8;
        var r = RunCase(bits, hd, nh, nkv, seqQ, offset, 0, false, flashEnabled: flash, seed: 4242 + bits + seqQ,
                        kScale: 1f, vScale: 2f, qScale: 2.5f, kBias: kBias);
        Assert.Equal(flash, r.Flash);
        var reference = ReferenceF64(r, bits, hd, nh, nkv, seqQ, offset);
        var (ipRel, ipMax) = ErrVs(r.InPlace, reference);
        var (orRel, orMax) = ErrVs(r.Oracle, reference);
        var (frRel, _) = ErrVs(r.Fresh, reference);
        _out.WriteLine($"PRECISION b{bits} q{seqQ}@{offset} flash={r.Flash} bias={kBias}: in-place meanRel {ipRel:E3} maxAbs {ipMax:E3} | " +
                       $"oracle meanRel {orRel:E3} maxAbs {orMax:E3} | ratio {ipRel / orRel:F2} | fresh (codec effect) meanRel {frRel:E3}");
        Assert.True(ipRel <= 2 * orRel, $"in-place meanRel {ipRel:E3} > 2 x oracle {orRel:E3}");
        Assert.True(ipRel <= 1.95e-3, $"in-place meanRel {ipRel:E3} above the 4 F16 half-ulp budget 1.95e-3");
    }
}
