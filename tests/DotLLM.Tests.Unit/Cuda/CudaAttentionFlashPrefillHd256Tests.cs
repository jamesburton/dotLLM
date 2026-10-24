using System.Diagnostics;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #962: kernel-level correctness (and an opt-in microbenchmark) for the head_dim-256
/// tensor-core flash PREFILL attention kernel versus the shipping <c>attention_f32</c> kernel, at
/// Bonsai-2 27B's real shape (24 query heads / 4 KV heads / head_dim 256).
///
/// <para>
/// The reference consumes the F32 expansion of exactly the same F16-rounded Q/K/V the kernel
/// reads, so the remaining differences are the F16 rounding of the softmax weights fed to the PV
/// mma and the reassociated accumulation -- expected well under 1e-3 relative-RMS.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaAttentionFlashPrefillHd256Tests : IDisposable
{
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    // Measured: relRms 0.9e-4 .. 2.0e-4, maxAbs/maxRef <= 2.1e-4 across every case below. The
    // tolerances sit ~5x above that and >100x below the wrong-mask / wrong-head controls.
    private const double RelRmsTol = 1e-3;
    private const double MaxAbsOverMaxRefTol = 2e-3;

    private readonly ITestOutputHelper _out;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    public CudaAttentionFlashPrefillHd256Tests(ITestOutputHelper output)
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

    private static ushort[] RandomHalf(Random rng, long n, float magnitude, float[]? floatMirror = null)
    {
        var h = new ushort[n];
        for (long i = 0; i < n; i++)
        {
            float f = (float)((rng.NextDouble() * 2 - 1) * magnitude);
            var half = (Half)f;
            h[i] = BitConverter.HalfToUInt16Bits(half);
            if (floatMirror != null) floatMirror[i] = (float)half;
        }
        return h;
    }

    private static nint UploadF16(ushort[] host)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)((long)host.Length * 2)).ThrowOnError();
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)((long)host.Length * 2)).ThrowOnError();
        return d;
    }

    private static nint UploadF32(float[] host)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)((long)host.Length * 4)).ThrowOnError();
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)((long)host.Length * 4)).ThrowOnError();
        return d;
    }

    private static float[] Download(nint d, long n)
    {
        var h = new float[n];
        fixed (float* p = h)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * 4)).ThrowOnError();
        return h;
    }

    private readonly record struct Err(double RelRms, double MaxAbs, double MaxAbsOverMaxRef, double MaxRef);

    private static Err Compare(float[] got, float[] want, int rowLo = 0, int rowHi = int.MaxValue, int stride = 1)
    {
        double sd = 0, sr = 0, maxAbs = 0, maxRef = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double d = got[i] - want[i];
            sd += d * d; sr += (double)want[i] * want[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            maxRef = Math.Max(maxRef, Math.Abs(want[i]));
        }
        return new Err(Math.Sqrt(sd / Math.Max(sr, 1e-30)), maxAbs, maxAbs / Math.Max(maxRef, 1e-30), maxRef);
    }

    [SkippableTheory]
    [InlineData(1, 0)]
    [InlineData(1, 300)]
    [InlineData(7, 0)]
    [InlineData(7, 5)]
    [InlineData(16, 0)]
    [InlineData(16, 300)]
    [InlineData(100, 0)]
    [InlineData(100, 5)]
    [InlineData(100, 300)]
    [InlineData(100, 4500)]
    [InlineData(256, 0)]
    [InlineData(256, 300)]
    [InlineData(1000, 0)]
    [InlineData(1000, 5)]
    [InlineData(1000, 300)]
    [InlineData(1000, 3600)]
    public void PrefillMatchesF32Reference(int seqQ, int posOffset)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256, "flash prefill hd256 kernel not loaded (pre-Ampere or stale PTX)");

        int seqKv = posOffset + seqQ;
        int qStride = NumHeads * HeadDim, kvStride = NumKvHeads * HeadDim;
        var rng = new Random(962 + seqQ * 31 + posOffset);

        // N(0,1)-ish magnitude so q.k/sqrt(256) has O(1) spread: softmax is non-trivial and any
        // mask / GQA-mapping mistake changes the output visibly.
        var qF = new float[(long)seqQ * qStride];
        var kF = new float[(long)seqKv * kvStride];
        var vF = new float[(long)seqKv * kvStride];
        var qH = RandomHalf(rng, qF.Length, 1.7f, qF);
        var kH = RandomHalf(rng, kF.Length, 1.7f, kF);
        var vH = RandomHalf(rng, vF.Length, 1.0f, vF);

        nint dQh = UploadF16(qH), dKh = UploadF16(kH), dVh = UploadF16(vH);
        nint dQf = UploadF32(qF), dKf = UploadF32(kF), dVf = UploadF32(vF);
        long outElems = (long)seqQ * qStride;
        CudaDriverApi.cuMemAlloc_v2(out nint dOutFlash, (nuint)(outElems * 4)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOutRef, (nuint)(outElems * 4)).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOutSwa, (nuint)(outElems * 4)).ThrowOnError();
        try
        {
            // Poison the output so an unwritten row cannot masquerade as a match.
            var poison = new float[outElems];
            Array.Fill(poison, float.NaN);
            fixed (float* p = poison)
                CudaDriverApi.cuMemcpyHtoD_v2(dOutFlash, (nint)p, (nuint)(outElems * 4)).ThrowOnError();

            _kernels.LaunchAttentionFlashPrefillHd256(dQh, dKh, dVh, dOutFlash,
                seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream!.Handle);
            _kernels.LaunchAttentionF32(dQf, dKf, dVf, dOutRef,
                seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, _stream.Handle);
            // Sensitivity control: same reference with a 64-key sliding window (a different mask).
            _kernels.LaunchAttentionF32(dQf, dKf, dVf, dOutSwa,
                seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 64, _stream.Handle);
            _stream.Synchronize();

            var got = Download(dOutFlash, outElems);
            var want = Download(dOutRef, outElems);
            var swa = Download(dOutSwa, outElems);

            var e = Compare(got, want);
            _out.WriteLine($"seqQ={seqQ} off={posOffset} seqKv={seqKv}: relRms={e.RelRms:E2} maxAbs={e.MaxAbs:E2} " +
                           $"maxAbs/maxRef={e.MaxAbsOverMaxRef:E2} (maxRef={e.MaxRef:F3})");
            Assert.False(float.IsNaN((float)e.RelRms), "NaN in kernel output (unwritten row or softmax NaN)");
            Assert.True(e.RelRms < RelRmsTol, $"relRms {e.RelRms:E3} >= {RelRmsTol}");
            Assert.True(e.MaxAbsOverMaxRef < MaxAbsOverMaxRefTol, $"maxAbs/maxRef {e.MaxAbsOverMaxRef:E3}");

            // Controls (only meaningful once some row actually has > 64 visible keys).
            if (seqKv > 128)
            {
                var eSwa = Compare(swa, want);
                _out.WriteLine($"   control(sliding-window mask vs causal): relRms={eSwa.RelRms:E2}");
                Assert.True(eSwa.RelRms > 10 * RelRmsTol,
                    "metric is not sensitive to the causal mask (sliding-window control passes the tolerance)");
            }
            // Wrong GQA mapping: compare head h against the reference of the neighbouring KV group's head.
            var shifted = new float[got.Length];
            for (long r = 0; r < seqQ; r++)
                for (int h = 0; h < NumHeads; h++)
                {
                    int hs = (h + NumHeads / NumKvHeads) % NumHeads;   // next KV head's group
                    Array.Copy(want, r * qStride + hs * HeadDim, shifted, r * qStride + h * HeadDim, HeadDim);
                }
            if (seqKv > 2)
            {
                var eMap = Compare(got, shifted);
                Assert.True(eMap.RelRms > 10 * RelRmsTol, "metric is not sensitive to the GQA head mapping");
            }
        }
        finally
        {
            foreach (var p in new[] { dQh, dKh, dVh, dQf, dKf, dVf, dOutFlash, dOutRef, dOutSwa })
                CudaDriverApi.cuMemFree_v2(p);
        }
    }

    /// <summary>
    /// Microbenchmark, opt-in (<c>DOTLLM_FLASH_BENCH=1</c>): kernel ms vs attention_f32 at
    /// seqQ=256 over a deep KV history. Printed, not asserted.
    /// </summary>
    [SkippableTheory]
    [InlineData(1024)]
    [InlineData(4096)]
    [InlineData(16384)]
    [InlineData(65536)]
    public void Bench_Flash_vs_F32(int seqKv)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_FLASH_BENCH") == "1", "set DOTLLM_FLASH_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256, "kernel not loaded");

        const int seqQ = 256;
        int posOffset = seqKv - seqQ;
        int qStride = NumHeads * HeadDim, kvStride = NumKvHeads * HeadDim;
        var rng = new Random(1);
        // Build F16 once; F32 copies are expanded on the host in chunks to bound memory.
        var qH = RandomHalf(rng, (long)seqQ * qStride, 1.7f);
        var kH = RandomHalf(rng, (long)seqKv * kvStride, 1.7f);
        var vH = RandomHalf(rng, (long)seqKv * kvStride, 1.0f);
        nint dQh = UploadF16(qH), dKh = UploadF16(kH), dVh = UploadF16(vH);
        nint dQf = 0, dKf = 0, dVf = 0;
        CudaDriverApi.cuMemAlloc_v2(out nint dOut, (nuint)((long)seqQ * qStride * 4)).ThrowOnError();
        try
        {
            var qf = new float[qH.Length]; for (int i = 0; i < qf.Length; i++) qf[i] = (float)BitConverter.UInt16BitsToHalf(qH[i]);
            dQf = UploadF32(qf);
            CudaDriverApi.cuMemAlloc_v2(out dKf, (nuint)((long)kH.Length * 4)).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dVf, (nuint)((long)vH.Length * 4)).ThrowOnError();
            _kernels.LaunchConvertF16ToF32(dKh, dKf, kH.Length, _stream!.Handle);
            _kernels.LaunchConvertF16ToF32(dVh, dVf, vH.Length, _stream.Handle);
            _stream.Synchronize();

            double Time(Action launch, int iters)
            {
                launch(); _stream.Synchronize();
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) launch();
                _stream.Synchronize();
                return sw.Elapsed.TotalMilliseconds / iters;
            }

            double flash = Time(() => _kernels.LaunchAttentionFlashPrefillHd256(dQh, dKh, dVh, dOut,
                seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream.Handle), 5);
            double f32 = Time(() => _kernels.LaunchAttentionF32(dQf, dKf, dVf, dOut,
                seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, _stream.Handle), seqKv >= 16384 ? 1 : 3);
            // Full-attention FLOPs actually performed by the causal tiles (QK + PV, 2 flop/MAC).
            double visible = (double)seqQ * posOffset + (double)seqQ * (seqQ + 1) / 2.0;
            double tflops = 4.0 * visible * HeadDim * NumHeads / (flash * 1e-3) / 1e12;
            _out.WriteLine($"BENCH seqQ={seqQ} seqKv={seqKv}: flash={flash:F2} ms ({tflops:F1} TFLOP/s) " +
                           $"attention_f32={f32:F2} ms speedup={f32 / flash:F1}x");
        }
        finally
        {
            foreach (var p in new[] { dQh, dKh, dVh, dQf, dKf, dVf, dOut })
                if (p != 0) CudaDriverApi.cuMemFree_v2(p);
        }
    }
}
