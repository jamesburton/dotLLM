using System;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// The CUDA half of #532: does attention output depend on the <b>padded</b> KV length?
/// </summary>
/// <remarks>
/// <para>
/// Deliberately mirrors <c>Probe532AttentionKvLengthInvarianceTests</c> (Vulkan) — same
/// experiment, same reported statistic (count differing / worst abs), same controls — so the two
/// backends are directly comparable rather than each being characterised its own way.
/// </para>
/// <para>
/// <b>The experiment.</b> Hold the real prefix fixed (identical Q, identical K/V rows
/// <c>0..posQ</c>) and present two different <c>seqKv</c> values. Everything past <c>posQ</c> is
/// causally masked, so both calls must compute the same thing. Any difference is reduction-order
/// drift caused by the cache length — #525's defect, on the GPU.
/// </para>
/// <para>
/// <b>Prediction from reading the kernels</b>, recorded before running so the result can falsify
/// it: <c>attention_f32.cu</c> is <b>invariant</b> (tiles from 0 by fixed <c>TILE_KV 256</c>, so
/// real rows keep their slot), and the split-KV path is <b>exposed</b>
/// (<c>chunk = ceil(seq_kv/kv_split)</c>, <c>kv_lo = s*chunk</c> — the boundaries move with the
/// cache length). CUDA should be exposed <i>less</i> than Vulkan, because <c>kv_split</c> is
/// fixed at <see cref="CudaKernels.AttentionKvSplit"/> whereas Vulkan's <c>numSplits</c> also
/// varies with <c>seqKv</c>.
/// </para>
/// <para>
/// <b>Reports, does not gate.</b> Asserting today's split-KV behaviour would freeze the defect.
/// Only the single-pass invariance is asserted, plus the determinism control and a loose sanity
/// bound that separates ULP drift from a real bug.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class Probe532CudaAttentionKvLengthInvarianceTests
{
    private readonly ITestOutputHelper _out;
    public Probe532CudaAttentionKvLengthInvarianceTests(ITestOutputHelper output) => _out = output;

    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    /// <summary>Single-pass must be bit-identical across padded cache lengths — asserted.</summary>
    [SkippableTheory]
    [InlineData(300, 301)]    // self-control
    [InlineData(300, 320)]    // padding inside the same TILE_KV tile
    [InlineData(300, 600)]    // padding that opens a SECOND tile
    [InlineData(255, 1024)]   // prefix ends one row before a tile boundary
    public void SinglePass_IsBitIdentical_AcrossPaddedKvLengths(int posQ, int paddedKv)
    {
        var h = Harness.TryCreate(paddedKv);
        Skip.If(h is null, "No CUDA GPU / PTX available");
        using (h)
        {
            float[] tight = h!.RunSinglePass(posQ, posQ + 1);
            float[] padded = h.RunSinglePass(posQ, paddedKv);

            int diff = Compare(tight, padded, out float worst);
            _out.WriteLine($"#532 CUDA single-pass posQ={posQ} kv {posQ + 1} vs {paddedKv}: differing={diff}/{tight.Length} worst={worst:E3}");
            Assert.Equal(0, diff);
        }
    }

    /// <summary>Split-KV — measured and reported.</summary>
    [SkippableTheory]
    [InlineData(300, 4096)]
    [InlineData(300, 8192)]
    [InlineData(1000, 4096)]
    public void SplitKv_KvLengthDependence_IsMeasured(int posQ, int paddedKv)
    {
        var h = Harness.TryCreate(paddedKv, needSplitKv: true);
        Skip.If(h is null, "No CUDA GPU / PTX / split-KV unavailable or unsafe for this shape");
        using (h)
        {
            int tightKv = posQ + 1;
            int split = CudaKernels.AttentionKvSplit;

            // Discrimination guard: the chunk geometry must actually move.
            int chunkTight = (tightKv + split - 1) / split;
            int chunkPadded = (paddedKv + split - 1) / split;
            Skip.If(chunkTight == chunkPadded, $"chunk width identical ({chunkTight}) — cannot discriminate.");

            float[] tight = h!.RunSplitKv(posQ, tightKv);
            float[] padded = h.RunSplitKv(posQ, paddedKv);

            int diff = Compare(tight, padded, out float worst);
            _out.WriteLine(
                $"#532 CUDA split-KV posQ={posQ} kv {tightKv}(split={split}, chunk={chunkTight}) " +
                $"vs {paddedKv}(split={split}, chunk={chunkPadded}): differing={diff}/{tight.Length} worst_abs={worst:E3}");

            foreach (float f in padded) Assert.True(float.IsFinite(f), "split-KV padded run produced a non-finite value.");
            Assert.True(worst < 1e-3f, $"divergence {worst:E3} is far beyond reduction-order drift — a real bug, not ULP.");
        }
    }

    /// <summary>
    /// CONTROL — the arm that makes the split-KV number mean "KV-length dependence" rather than
    /// "this kernel is nondeterministic". Sensitive by construction: nondeterminism would fail
    /// here, at the same shape, through the same harness.
    /// </summary>
    [SkippableTheory]
    [InlineData(300, 4096)]
    [InlineData(300, 8192)]
    public void SplitKv_SameKvLength_IsBitDeterministic(int posQ, int seqKv)
    {
        var h = Harness.TryCreate(seqKv, needSplitKv: true);
        Skip.If(h is null, "No CUDA GPU / PTX / split-KV unavailable or unsafe for this shape");
        using (h)
        {
            float[] a = h!.RunSplitKv(posQ, seqKv);
            float[] b = h.RunSplitKv(posQ, seqKv);

            int diff = Compare(a, b, out float worst);
            _out.WriteLine($"#532 CUDA control split-KV posQ={posQ} kv={seqKv} twice: differing={diff}/{a.Length} worst={worst:E3}");
            Assert.Equal(0, diff);
        }
    }

    private static int Compare(float[] a, float[] b, out float worstAbs)
    {
        int n = 0; worstAbs = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i].Equals(b[i])) continue;
            n++;
            worstAbs = Math.Max(worstAbs, Math.Abs(a[i] - b[i]));
        }
        return n;
    }

    private sealed class Harness : IDisposable
    {
        private readonly CudaContext _ctx;
        private readonly CudaStream _stream;
        private readonly CudaKernels _kernels;
        private readonly float[] _q, _k, _v;

        private Harness(CudaContext ctx, CudaStream stream, CudaKernels kernels, float[] q, float[] k, float[] v)
        { _ctx = ctx; _stream = stream; _kernels = kernels; _q = q; _k = k; _v = v; }

        public static Harness? TryCreate(int maxKv, bool needSplitKv = false)
        {
            if (!IsCudaDriverPresent()) return null;
            string? ptxDir = FindPtxDir();
            if (ptxDir is null) return null;

            var ctx = CudaContext.Create(0);
            var stream = CudaStream.Create();
            var kernels = new CudaKernels(ptxDir);

            if (needSplitKv && (!kernels.HasAttentionF32SplitKv || !kernels.IsAttentionSplitKvSafe(NumHeads, HeadDim)))
            {
                kernels.Dispose(); stream.Dispose(); ctx.Dispose();
                return null;
            }

            var rng = new Random(0x532);
            int kvElems = NumKvHeads * HeadDim;
            return new Harness(ctx, stream, kernels,
                RandomVec(rng, NumHeads * HeadDim),
                RandomVec(rng, maxKv * kvElems),
                RandomVec(rng, maxKv * kvElems));
        }

        public float[] RunSinglePass(int posQ, int seqKv) => Run(posQ, seqKv, split: false);
        public float[] RunSplitKv(int posQ, int seqKv) => Run(posQ, seqKv, split: true);

        private float[] Run(int posQ, int seqKv, bool split)
        {
            int qElems = NumHeads * HeadDim;
            int kvElems = NumKvHeads * HeadDim;
            long qBytes = (long)qElems * sizeof(float);
            long kvBytes = (long)seqKv * kvElems * sizeof(float);

            nint dQ = 0, dK = 0, dV = 0, dOut = 0, dPMax = 0, dPSum = 0, dPOut = 0;
            try
            {
                CudaDriverApi.cuMemAlloc_v2(out dQ, (nuint)qBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dK, (nuint)kvBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dV, (nuint)kvBytes).ThrowOnError();
                CudaDriverApi.cuMemAlloc_v2(out dOut, (nuint)qBytes).ThrowOnError();

                unsafe
                {
                    fixed (float* p = _q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
                    fixed (float* p = _k) CudaDriverApi.cuMemcpyHtoD_v2(dK, (nint)p, (nuint)kvBytes).ThrowOnError();
                    fixed (float* p = _v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)kvBytes).ThrowOnError();
                }

                nint s = _stream.Handle;
                if (!split)
                {
                    _kernels.LaunchAttentionF32(dQ, dK, dV, dOut, seqQ: 1, seqKv, NumHeads, NumKvHeads, HeadDim,
                        posQ, slidingWindow: 0, s);
                }
                else
                {
                    long scalarBytes = (long)NumHeads * CudaKernels.AttentionKvSplit * sizeof(float);
                    CudaDriverApi.cuMemAlloc_v2(out dPMax, (nuint)scalarBytes).ThrowOnError();
                    CudaDriverApi.cuMemAlloc_v2(out dPSum, (nuint)scalarBytes).ThrowOnError();
                    CudaDriverApi.cuMemAlloc_v2(out dPOut, (nuint)(scalarBytes * HeadDim)).ThrowOnError();
                    _kernels.LaunchAttentionF32SplitKv(dQ, dK, dV, dOut, seqKv, NumHeads, NumKvHeads, HeadDim,
                        posQ, slidingWindow: 0, dPMax, dPSum, dPOut, s);
                }
                _stream.Synchronize();

                var host = new float[qElems];
                unsafe
                {
                    fixed (float* p = host) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)qBytes).ThrowOnError();
                }
                return host;
            }
            finally
            {
                foreach (nint p in new[] { dQ, dK, dV, dOut, dPMax, dPSum, dPOut })
                    if (p != 0) CudaDriverApi.cuMemFree_v2(p);
            }
        }

        public void Dispose() { _kernels.Dispose(); _stream.Dispose(); _ctx.Dispose(); }

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
            string[] candidates =
            [
                Path.Combine(AppContext.BaseDirectory, "ptx"),
                Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
            ];
            foreach (var dir in candidates)
            {
                var full = Path.GetFullPath(dir);
                if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                    return full;
            }
            return null;
        }

        private static float[] RandomVec(Random rng, int n)
        {
            var v = new float[n];
            for (int i = 0; i < n; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            return v;
        }
    }
}
