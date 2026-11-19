using System;
using System.Collections.Generic;
using DotLLM.Core.Attention;
using DotLLM.Engine.KvCache.Codecs;

namespace DotLLM.Tests.Unit.Engine.KvCache;

/// <summary>
/// Shared input corpus and CPU reference for the cross-backend encode byte-exactness gate (#1892).
/// The corpus mixes random vectors over a wide norm range (incl. F16-saturating and F16-subnormal norms,
/// zero / one-hot / constant vectors) with vectors built so that, <b>after normalise and rotate</b>, some
/// coordinates sit exactly equidistant (in F32) between two adjacent centroids. Those ties are what make the
/// argmin tie policy observable on a GPU backend; random data alone essentially never hits one.
/// </summary>
internal static class TurboQuantByteExactCorpus
{
    public static readonly int[] BitWidths = { 1, 2, 3, 4, 8 };
    public static readonly int[] HeadDims = { 64, 128, 256 };

    public const ulong Seed = 0x7B_E5_ACE1UL;
    public const ulong VSeedXor = 0xD1B54A32D192ED03UL; // TurboQuantKvCache's V-seed derivation
    public const int NumKvHeads = 4;
    public const int SeqLen = 10;                       // 40 vectors per side

    public sealed class SideCorpus
    {
        public required float[] Data { get; init; }          // [SeqLen * NumKvHeads * HeadDim]
        public required int TieCoords { get; init; }         // exact-tie coordinates after normalise+rotate
        public required int TieVectors { get; init; }
        public required TurboQuantCodec Codec { get; init; }
    }

    public sealed class Corpus
    {
        public required SideCorpus K { get; init; }
        public required SideCorpus V { get; init; }
    }

    /// <param name="halfInputs">Round every input through binary16 (the CUDA forward feeds <c>half</c> K/V), searching ties on the rounded values.</param>
    public static Corpus Build(int bits, int headDim, bool halfInputs)
    {
        var ck = new TurboQuantCodec(headDim, bits, Seed, useQjl: false);
        var cv = new TurboQuantCodec(headDim, bits, Seed ^ VSeedXor, useQjl: false);
        int tag = bits * 1000 + headDim + (halfInputs ? 7 : 0);
        return new Corpus
        {
            K = BuildSide(ck, new Random(0x4B_00 + tag), halfInputs),
            V = BuildSide(cv, new Random(0x56_00 + tag), halfInputs),
        };
    }

    private static SideCorpus BuildSide(TurboQuantCodec codec, Random rng, bool half)
    {
        int d = codec.HeadDim;
        int n = SeqLen * NumKvHeads;
        var data = new float[n * d];
        float Q(float v) => half ? (float)(Half)v : v;

        int vec = 0;
        void Put(float[] x) { for (int i = 0; i < d; i++) data[vec * d + i] = Q(x[i]); vec++; }

        // special vectors
        Put(new float[d]);                                                        // zero
        var onehot = new float[d]; onehot[d / 3] = 7.5f; Put(onehot);             // one-hot
        var ones = new float[d]; Array.Fill(ones, 0.5f); Put(ones);               // constant
        Put(Gaussian(rng, d, 1e4f));                                              // norm >> 65504: F16 saturates
        Put(Gaussian(rng, d, 3e-6f));                                             // F16-subnormal norm
        Put(Gaussian(rng, d, 5e-6f));
        // random vectors over a wide norm range
        int randomCount = n - vec - 14;
        for (int i = 0; i < randomCount; i++)
            Put(Gaussian(rng, d, MathF.Exp(Lerp(MathF.Log(1e-2f), MathF.Log(5e2f), (float)rng.NextDouble()))));

        // tie-injected vectors
        var ties = TieValues(codec);
        int tieVectors = 0, tieCoords = 0;
        if (ties.Count > 0)
        {
            var rot = new float[d];
            int attempts = 0;
            while (vec < n && attempts < 400_000)
            {
                attempts++;
                float[] x = TieCandidate(codec, rng, ties, Q);
                codec.NormalizeAndRotate(x, rot);
                int t = CountTieCoords(codec, rot);
                if (t == 0) continue;
                Put(x); tieVectors++; tieCoords += t;
            }
        }
        // top up with ordinary vectors if the codebook admits no exact F32 ties (or the search ran out of budget)
        while (vec < n) Put(Gaussian(rng, d, 1f + vec));

        return new SideCorpus { Data = data, TieCoords = tieCoords, TieVectors = tieVectors, Codec = codec };
    }

    private static float Lerp(float a, float b, float t) => a + (b - a) * t;

    private static float[] Gaussian(Random rng, int d, float scale)
    {
        var x = new float[d];
        for (int i = 0; i < d; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            x[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)) * scale;
        }
        return x;
    }

    /// <summary>
    /// F32 values <c>t</c> with <c>|t - c_k| == |c_(k+1) - t|</c> exactly (rounded F32 distances), for adjacent
    /// centroids. Where the float midpoint is not itself a tie, nearby floats are tried.
    /// </summary>
    internal static List<(float t, int k)> TieValues(TurboQuantCodec codec)
    {
        var c = codec.Centroids;
        var res = new List<(float, int)>();
        for (int k = 0; k + 1 < c.Length; k++)
        {
            float m = 0.5f * (c[k] + c[k + 1]);
            float lo = m, hi = m;
            for (int step = 0; step <= 6; step++)
            {
                foreach (float t in step == 0 ? new[] { m } : new[] { lo, hi })
                    if (MathF.Abs(t - c[k]) == MathF.Abs(c[k + 1] - t)) { res.Add((t, k)); goto next; }
                lo = MathF.BitDecrement(lo); hi = MathF.BitIncrement(hi);
            }
        next:;
        }
        return res;
    }

    // Builds x whose rotated unit direction should land on tie values at a subset of coordinates: pick the target
    // rotated vector s in double, invert the forward RHT (u = invSqrtD * D * H * s), scale, quantise. Float rounding
    // spreads the realised coordinates by a few ulps, so only some coordinates hit; the caller counts the real ties.
    private static float[] TieCandidate(TurboQuantCodec codec, Random rng, List<(float t, int k)> ties, Func<float, float> q)
    {
        int d = codec.HeadDim;
        var s = new double[d];
        double sd = codec.InvSqrtD;
        for (int i = 0; i < d; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            s[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * sd;
        }
        int targets = Math.Max(4, d / 2);
        for (int j = 0; j < targets; j++)
            s[rng.Next(d)] = ties[rng.Next(ties.Count)].t;

        var h = (double[])s.Clone();
        for (int len = 1; len < d; len <<= 1)
            for (int i = 0; i < d; i += len << 1)
                for (int j = i; j < i + len; j++) { double u = h[j], v = h[j + len]; h[j] = u + v; h[j + len] = u - v; }
        var signs = codec.RotationSigns;
        double scale = Math.Exp(rng.NextDouble() * 4.0 - 1.0);
        var x = new float[d];
        for (int i = 0; i < d; i++) x[i] = q((float)(h[i] * sd * signs[i] * scale));
        return x;
    }

    /// <summary>Coordinates whose two nearest centroids are exactly equidistant in F32 (the tie path).</summary>
    internal static int CountTieCoords(TurboQuantCodec codec, ReadOnlySpan<float> rotated)
    {
        var c = codec.Centroids;
        int count = 0;
        foreach (float v in rotated)
        {
            float best = float.MaxValue; int hits = 0;
            for (int j = 0; j < c.Length; j++)
            {
                float dist = MathF.Abs(v - c[j]);
                if (dist < best) { best = dist; hits = 1; }
                else if (dist == best) hits++;
            }
            if (hits >= 2) count++;
        }
        return count;
    }

    /// <summary>
    /// CPU reference for one side: per head-vector the packed codes (zero-padded to <c>codeUintsPerVec</c> uints) and
    /// the norm bytes in <paramref name="norm"/> (through <see cref="TurboQuantNormCodec"/>).
    /// </summary>
    public static (byte[] codes, byte[] norms) CpuEncode(SideCorpus side, TqNormFormat norm)
    {
        var codec = side.Codec;
        int d = codec.HeadDim;
        int n = SeqLen * NumKvHeads;
        int codeBytes = codec.CodeBytesPerVector;
        int cu = (codeBytes + 3) / 4;
        var codes = new byte[n * cu * 4];
        var norms = new float[n];
        var tmp = new byte[codeBytes];
        for (int v = 0; v < n; v++)
        {
            norms[v] = codec.Encode(side.Data.AsSpan(v * d, d), tmp);
            tmp.AsSpan(0, codeBytes).CopyTo(codes.AsSpan(v * cu * 4));
        }
        var nb = new byte[n * TurboQuantNormCodec.BytesPerNorm(norm)];
        TurboQuantNormCodec.Encode(norm, norms, nb);
        return (codes, nb);
    }

    /// <summary>Per-vector byte comparison of a backend's stored codes and norms against the CPU reference.</summary>
    public static (int codeVectors, int normVectors, int total) Compare(
        byte[] cpuCodes, byte[] cpuNorms, ReadOnlySpan<byte> gpuCodes, ReadOnlySpan<byte> gpuNorms, int codeStrideBytes, int normBytes)
    {
        int total = cpuNorms.Length / normBytes;
        int codeBad = 0, normBad = 0;
        for (int v = 0; v < total; v++)
        {
            if (!cpuCodes.AsSpan(v * codeStrideBytes, codeStrideBytes).SequenceEqual(gpuCodes.Slice(v * codeStrideBytes, codeStrideBytes))) codeBad++;
            if (!cpuNorms.AsSpan(v * normBytes, normBytes).SequenceEqual(gpuNorms.Slice(v * normBytes, normBytes))) normBad++;
        }
        return (codeBad, normBad, total);
    }

    public static System.Collections.Generic.IEnumerable<object[]> BitsAndHeadDims()
    {
        foreach (int b in BitWidths)
            foreach (int d in HeadDims)
                yield return new object[] { b, d };
    }
}
