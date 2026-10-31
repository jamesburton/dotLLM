using System.Globalization;
using DotLLM.ChunkReuse;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace ChunkKvLab;

/// <summary>
/// RoPE-shift precision at large offsets. Layer-0 K depends only on token and position (it is written to the cache before
/// layer-0 attention), so a forward of a chunk at positions Δ..Δ+n-1 into an otherwise empty cache gives the model's own
/// layer-0 keys there; they are compared with <see cref="RopeShift"/> of the chunk's isolated (0..n-1) keys.
/// </summary>
internal static class RopePrecision
{
    public static int Run(string modelPath, string corpusDir, string pattern, int[] deltas, int chunks, int chunkTokens, int threads, string outCsv)
    {
        using LabModel lab = LabModel.Load(modelPath, threads);
        ContextBuilder corpus = ContextBuilder.FromDirectory(corpusDir, pattern, lab.Tokenizer, chunkTokens);
        var encoder = new ChunkEncoder(lab.Fingerprint);
        var rope = RopeParameters.FromConfig(lab.Config);
        int headDim = lab.Config.HeadDim;
        int stride = lab.Geometry.KvStrideOf(0);
        var ci = CultureInfo.InvariantCulture;
        Console.WriteLine($"RoPE {rope}; head dim {headDim}; layer-0 stride {stride}");

        using var csv = new StreamWriter(outCsv) { AutoFlush = true };
        csv.WriteLine("Chunk,Delta,MaxAbsErr,MeanAbsErr,MaxAbsK,MaxRelErrPerHead,UnshiftedMaxAbsDiff");
        var rng = new Random(12345);
        for (int c = 0; c < chunks; c++)
        {
            int idx = rng.Next(corpus.Chunks.Count);
            int[] ids = corpus.Chunks[idx].TokenIds;
            ChunkKvEntry entry = encoder.Encode(lab.Model, ids);
            float[] k0 = entry.KeysPerLayer[0];
            foreach (int delta in deltas)
            {
                using SimpleKvCache cache = lab.NewCache(delta + ids.Length);
                lab.Prefill(cache, ids, delta, ids.Length); // one forward at Δ..Δ+n-1; rows < Δ are never written
                ReadOnlySpan<float> model = cache.KeysSpan(0).Slice(delta * stride, ids.Length * stride);

                float[] shifted = (float[])k0.Clone();
                RopeShift.Apply(shifted, headDim, delta, in rope);

                double maxAbs = 0, sumAbs = 0, maxK = 0, maxRel = 0, unshifted = 0;
                for (int h = 0; h < shifted.Length; h += headDim)
                {
                    double headErr = 0, headNorm = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        double e = Math.Abs(shifted[h + d] - model[h + d]);
                        headErr = Math.Max(headErr, e);
                        sumAbs += e;
                        headNorm += (double)model[h + d] * model[h + d];
                        maxK = Math.Max(maxK, Math.Abs(model[h + d]));
                        unshifted = Math.Max(unshifted, Math.Abs(k0[h + d] - model[h + d]));
                    }
                    maxAbs = Math.Max(maxAbs, headErr);
                    if (headNorm > 0) maxRel = Math.Max(maxRel, headErr / Math.Sqrt(headNorm));
                }
                double meanAbs = sumAbs / shifted.Length;
                csv.WriteLine(string.Join(',', idx.ToString(ci), delta.ToString(ci), maxAbs.ToString("G6", ci), meanAbs.ToString("G6", ci),
                    maxK.ToString("G6", ci), maxRel.ToString("G6", ci), unshifted.ToString("G6", ci)));
                Console.WriteLine($"chunk {idx} Δ={delta,6}: max|err| {maxAbs:G4}  mean|err| {meanAbs:G4}  max|K| {maxK:G4}  "
                    + $"max err/‖head‖ {maxRel:G4}  (unshifted max diff {unshifted:G4})");
            }
        }
        return 0;
    }
}
