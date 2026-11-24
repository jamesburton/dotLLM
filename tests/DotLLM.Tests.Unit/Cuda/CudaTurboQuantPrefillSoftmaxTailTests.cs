using DotLLM.Core.Attention;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1953 on the TurboQuant tensor-core prefill: every entry (Legacy = pk_prefill_body over norm * centroid tiles; Specialized / Staged =
/// pk_tq_tile, whose P.V operand is P * |v| over bare-centroid tiles) against the scalar <c>attention_kvtq</c> oracle (F32 P).
/// <list type="bullet">
/// <item><b>Tail</b>: a sink key 15.6-18.4 nats above 32,770 others with V norm 0, so the output is the tail mass alone. The tail's V
/// norms grow with position from 1 to 500, so pk_tq_tile's adaptive scale exponent k (#1953) drops many times along the row: an
/// unshifted P flushes the tail to 0, and a k drop that is not rescaled mixes scales.</item>
/// <item><b>Overflow</b>: uniform scores with V norm 400 on every key: a fixed 2^15 scale in pk_tq_tile packs P * |v| = 1.3e7 into F16
/// (inf); the adaptive k must keep it finite and exact.</item>
/// </list>
/// All codes are 0 (centroid c_0 on every dimension) and Q is c_0-aligned, so the score of key j is exactly its K norm.
/// </summary>
public sealed unsafe partial class CudaTurboQuantPrefillTailTests
{
    public static TheoryData<int, int, bool> SoftmaxTailCases => new()
    {
        // bits, hd, overflow-case
        { 4, 128, false }, { 2, 128, false }, { 8, 128, false }, { 4, 64, false },
        { 4, 128, true }, { 2, 64, true },
    };

    [SkippableTheory]
    [MemberData(nameof(SoftmaxTailCases))]
    public void FlashPrefill_SinkDominatedLongRow_AndLargeVNorms_MatchScalarOracle(int bits, int hd, bool overflowCase)
    {
        Require(hd);
        const int numHeads = 8, numKvHeads = 2, seqQ = 16;
        int seqKv = overflowCase ? 77 : 32771;
        int cu = new TurboQuantLayout(hd, numKvHeads, bits, TqNormFormat.F16).CodeUintsPerVec;
        float[] cent = new TurboQuantCodec(hd, bits, Seed).Centroids.ToArray();
        float c0 = cent[0];
        var rng = new Random(1953 + bits * 7 + hd + seqKv);
        var kn = new ushort[seqKv * numKvHeads];
        var vn = new ushort[seqKv * numKvHeads];
        for (int j = 0; j < seqKv; j++)
            for (int h = 0; h < numKvHeads; h++)
            {
                float k, v;
                if (overflowCase) { k = 1f; v = 400f; }
                else if (j == 0) { k = 20f; v = 0f; }
                else { k = 1.6f + 2.8f * (float)rng.NextDouble(); v = 1f + 499f * j / seqKv; }
                kn[j * numKvHeads + h] = BitConverter.HalfToUInt16Bits((Half)k);
                vn[j * numKvHeads + h] = BitConverter.HalfToUInt16Bits((Half)v);
            }
        // q . c(code 0) * scale = K-norm multiplier 1: q_d = 1 / (sqrt(hd) c_0) on every dimension.
        var q = new ushort[seqQ * numHeads * hd];
        for (int i = 0; i < q.Length; i++) q[i] = BitConverter.HalfToUInt16Bits((Half)(1f / (MathF.Sqrt(hd) * c0)));
        long words = (long)seqKv * numKvHeads * cu + 4;
        nint dq = Alloc(q.Length * 2L), kc = Alloc(words * 4), vc = Alloc(words * 4), dkn = Alloc(kn.Length * 2L + 4), dvn = Alloc(vn.Length * 2L + 4),
             dc = Alloc(cent.Length * 4L), oRef = Alloc(q.Length * 2L), oNew = Alloc(q.Length * 2L);
        try
        {
            CudaDriverApi.cuMemsetD8_v2(kc, 0, (nuint)(words * 4)).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(vc, 0, (nuint)(words * 4)).ThrowOnError();
            Up(dq, q); Up(dkn, kn); Up(dvn, vn); Up(dc, cent);
            int offset = seqKv - seqQ;
            _kernels!.LaunchAttentionKvTq(dq, kc, vc, dkn, dvn, dc, oRef, seqQ, seqKv, numHeads, numKvHeads, hd, offset, 0,
                _stream!.Handle, 0, bits, cu, TqNormFormat.F16);
            _stream.Synchronize();
            var want = new Half[q.Length];
            Down(want, oRef);
            var failures = new List<string>();
            foreach (var entry in Entries(_kernels, hd, bits))
            {
                CudaKernels.TqPrefillVariantOverride = entry;
                try
                {
                    _kernels.LaunchAttentionFlashPrefillKvTq(dq, kc, vc, dkn, dvn, dc, oNew, seqQ, seqKv, numHeads, numKvHeads, hd, offset,
                        bits, cu, TqNormFormat.F16, _stream.Handle);
                }
                finally { CudaKernels.TqPrefillVariantOverride = null; }
                _stream.Synchronize();
                var got = new Half[q.Length];
                Down(got, oNew);
                double worst = 0;
                int at = 0;
                for (int i = 0; i < got.Length; i++)
                {
                    double a = (double)got[i], b = (double)want[i];
                    double err = Math.Abs(a - b) / Math.Max(Math.Abs(b), 1e-30);
                    if (!(err <= worst)) { worst = err; at = i; }   // NaN / inf count as the worst
                }
                string line = $"tq{bits} hd{hd} {(overflowCase ? "overflow" : "tail")} seqKv {seqKv} {entry}: worst relative error {worst:E3} " +
                              $"at {at} ({(double)got[at]:E4} vs attention_kvtq {(double)want[at]:E4})";
                _out.WriteLine(line);
                if (!(worst <= 0.01)) failures.Add(line);
            }
            Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        }
        finally
        {
            foreach (nint p in new[] { dq, kc, vc, dkn, dvn, dc, oRef, oNew }) CudaDriverApi.cuMemFree_v2(p);
        }
    }
}
