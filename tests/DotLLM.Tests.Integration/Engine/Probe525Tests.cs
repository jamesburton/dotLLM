using System;
using System.Collections.Generic;
using System.Linq;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>TEMPORARY probe for issue #525 — delete before merge.</summary>
[Collection("SmallModel")]
public class Probe525Tests
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _out;

    public Probe525Tests(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _out = output;
    }

    private static string? ModelOverride => Environment.GetEnvironmentVariable("PROBE525_GGUF");

    private (TransformerModel model, GgufFile gguf) LoadModel()
    {
        var gguf = GgufFile.Open(ModelOverride ?? _fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config);
        return (model, gguf);
    }

    /// <summary>K/V snapshot for every layer, row-major [pos, kvStride].</summary>
    private sealed record Snapshot(float[][] K, float[][] V, float[] LastLogits);

    private unsafe Snapshot RunArm(TransformerModel model, int[] tokens, int[][] chunks)
    {
        int numLayers = model.Config.NumLayers;
        using var kv = new SimpleKvCache(numLayers, model.Config.NumKvHeads, model.Config.HeadDim, 64);
        float[] last = Array.Empty<float>();
        int pos = 0;
        foreach (var chunk in chunks)
        {
            var ids = chunk;
            var positions = Enumerable.Range(pos, ids.Length).ToArray();
            pos += ids.Length;
            using ITensor logits = model.Forward(ids, positions, -1, kv, null);
            int rows = logits.Shape[0];
            int vocab = model.Config.VocabSize;
            last = new Span<float>((float*)logits.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
        }

        var ks = new float[numLayers][];
        var vs = new float[numLayers][];
        int stride = model.Config.NumKvHeads * model.Config.HeadDim;
        int len = tokens.Length;
        for (int l = 0; l < numLayers; l++)
        {
            var kr = kv.GetKeysRef(l);
            var vr = kv.GetValuesRef(l);
            ks[l] = new Span<float>((float*)kr.DataPointer, len * stride).ToArray();
            vs[l] = new Span<float>((float*)vr.DataPointer, len * stride).ToArray();
        }
        return new Snapshot(ks, vs, last);
    }

    private static double MaxAbs(float[] a, float[] b, int from, int count)
    {
        double m = 0;
        for (int i = from; i < from + count; i++)
            m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
        return m;
    }

    /// <summary>Per-layer float hidden state, indexed [layer][globalPos][hiddenSize].</summary>
    private unsafe float[][][] RunArmHidden(TransformerModel model, int[] tokens, int[][] chunks)
    {
        int numLayers = model.Config.NumLayers;
        int hs = model.Config.HiddenSize;
        var caps = new float[numLayers][][];
        for (int l = 0; l < numLayers; l++) caps[l] = new float[tokens.Length][];

        using var kv = new SimpleKvCache(numLayers, model.Config.NumKvHeads, model.Config.HeadDim, 64);
        int pos = 0;
        int chunkStart = 0;
        model.DebugLayerHidden = (layer, seqLen, ptr) =>
        {
            for (int t = 0; t < seqLen; t++)
                caps[layer][chunkStart + t] = new Span<float>((float*)ptr + (long)t * hs, hs).ToArray();
        };
        foreach (var chunk in chunks)
        {
            chunkStart = pos;
            var positions = Enumerable.Range(pos, chunk.Length).ToArray();
            pos += chunk.Length;
            using ITensor logits = model.Forward(chunk, positions, -1, kv, null);
        }
        model.DebugLayerHidden = null;
        return caps;
    }

    private unsafe Dictionary<string, float[]> RunArmTensors(TransformerModel model, int[] tokens, int[][] chunks, int layer)
    {
        var caps = new Dictionary<string, float[]>(StringComparer.Ordinal);
        using var kv = new SimpleKvCache(model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, 64);
        int pos = 0, chunkStart = 0;
        model.DebugTensor = (l, label, rows, cols, ptr) =>
        {
            if (l != layer) return;
            for (int t = 0; t < rows; t++)
                caps[$"{label}#p{chunkStart + t}"] = new Span<float>((float*)ptr + (long)t * cols, cols).ToArray();
        };
        foreach (var chunk in chunks)
        {
            chunkStart = pos;
            var positions = Enumerable.Range(pos, chunk.Length).ToArray();
            pos += chunk.Length;
            using ITensor logits = model.Forward(chunk, positions, -1, kv, null);
        }
        model.DebugTensor = null;
        return caps;
    }

    /// <summary>
    /// Kernel-level isolation: the SAME query row attending over the SAME K/V prefix must give the
    /// same output whether the score row is seqKv=3 (chunked: cache holds 3 positions) or seqKv=5
    /// (single-pass: cache holds 5, positions 3..4 causally masked out for query row 2).
    /// </summary>
    /// <summary>Is the seed in the softmax row sum, or in WeightedValues?</summary>
    [Fact]
    public void Probe_SoftmaxRowLengthInvariance()
    {
        var rng = new Random(7);
        int mismatches = 0;
        for (int trial = 0; trial < 200; trial++)
        {
            int visible = rng.Next(2, 6);
            int padTo = visible + rng.Next(1, 5);
            var shortRow = new float[visible];
            var longRow = new float[padTo];
            for (int i = 0; i < visible; i++)
            {
                float x = (float)(rng.NextDouble() * 20 - 10);
                shortRow[i] = x; longRow[i] = x;
            }
            for (int i = visible; i < padTo; i++) longRow[i] = float.NegativeInfinity;

            var a = new float[visible];
            var b = new float[padTo];
            DotLLM.Cpu.Kernels.Softmax.ExecuteFast(shortRow, a);
            DotLLM.Cpu.Kernels.Softmax.ExecuteFast(longRow, b);
            for (int i = 0; i < visible; i++)
                if (a[i] != b[i]) { mismatches++; break; }
        }
        _out.WriteLine($"  Softmax.ExecuteFast: {mismatches}/200 trials differ when the row is padded with masked (-inf) entries");
    }

    [Fact]
    public unsafe void Probe_AttentionSeqKvInvariance()
    {
        const int numHeads = 9, numKvHeads = 3, headDim = 64;
        int qStride = numHeads * headDim, kvStride = numKvHeads * headDim;
        var rng = new Random(1234);
        int maxKv = 5;
        var q = new float[3 * qStride];
        var k = new float[maxKv * kvStride];
        var v = new float[maxKv * kvStride];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        float scale = 1f / MathF.Sqrt(headDim);

        var outShort = new float[3 * qStride];
        var outLong = new float[3 * qStride];
        fixed (float* qp = q, kp = k, vp = v, os = outShort, ol = outLong)
        {
            // chunked: cache has exactly 3 positions, query rows 0..2 at offset 0
            DotLLM.Cpu.Kernels.Attention.Execute(qp, kp, vp, os, 3, 3, numHeads, numKvHeads, headDim, 0, null);
            // single-pass: cache has 5 positions; rows 0..2 mask out kv 3..4
            DotLLM.Cpu.Kernels.Attention.Execute(qp, kp, vp, ol, 3, 5, numHeads, numKvHeads, headDim, 0, null);
        }
        for (int r = 0; r < 3; r++)
        {
            double m = 0;
            for (int i = 0; i < qStride; i++)
                m = Math.Max(m, Math.Abs((double)outShort[r * qStride + i] - outLong[r * qStride + i]));
            _out.WriteLine($"  row{r}: seqKv=3 vs seqKv=5 maxAbs = {m:E3}");
        }
    }

    [Fact]
    public void Probe_IntraLayerBisect()
    {
        var (model, gguf) = LoadModel();
        using var _ = gguf;
        using var __ = model;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = tokenizer.Encode("The capital of France is");
        int[][] Split(params int[] sizes)
        {
            var res = new List<int[]>();
            int p = 0;
            foreach (int s in sizes) { res.Add(tokens[p..(p + s)]); p += s; }
            return res.ToArray();
        }
        int layer = int.TryParse(Environment.GetEnvironmentVariable("PROBE525_LAYER"), out int lv) ? lv : 21;
        var baseline = RunArmTensors(model, tokens, Split(5), layer);
        foreach (var (name, chunks) in new (string, int[][])[]
        {
            ("[5] control", Split(5)),
            ("[3,2] BAD", Split(3, 2)),
            ("[2,3] GOOD", Split(2, 3)),
        })
        {
            var arm = RunArmTensors(model, tokens, chunks, layer);
            _out.WriteLine($"=== L{layer} {name} ===");
            foreach (var key in baseline.Keys.OrderBy(x => x, StringComparer.Ordinal))
            {
                double d = MaxAbs(baseline[key], arm[key], 0, baseline[key].Length);
                if (d != 0) _out.WriteLine($"  {key}: {d:E3}");
            }
        }
    }

    [Fact]
    public void Probe_PerLayerHiddenBisect()
    {
        var (model, gguf) = LoadModel();
        using var _ = gguf;
        using var __ = model;
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = tokenizer.Encode("The capital of France is");
        int[][] Split(params int[] sizes)
        {
            var res = new List<int[]>();
            int p = 0;
            foreach (int s in sizes) { res.Add(tokens[p..(p + s)]); p += s; }
            return res.ToArray();
        }

        var baseline = RunArmHidden(model, tokens, Split(5));
        foreach (var (name, chunks) in new (string, int[][])[]
        {
            ("[5] control", Split(5)),
            ("[3,2] BAD", Split(3, 2)),
            ("[2,3] GOOD", Split(2, 3)),
            ("[1,1,1,1,1] BAD", Split(1, 1, 1, 1, 1)),
        })
        {
            var arm = RunArmHidden(model, tokens, chunks);
            _out.WriteLine($"=== {name} ===");
            for (int l = 0; l < model.Config.NumLayers; l++)
            {
                var parts = new List<string>();
                for (int p = 0; p < tokens.Length; p++)
                {
                    double d = MaxAbs(baseline[l][p], arm[l][p], 0, baseline[l][p].Length);
                    if (d != 0) parts.Add($"p{p}={d:E2}");
                }
                if (parts.Count > 0) _out.WriteLine($"  L{l,2}: {string.Join(" ", parts)}");
            }
        }
    }

    [Fact]
    public void Probe_PerLayerKvBisect()
    {
        var (model, gguf) = LoadModel();
        using var _ = gguf;
        using var __ = model;

        // "The capital of France is" tokenized by the fixture tokenizer in the original repro.
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] tokens = tokenizer.Encode("The capital of France is").ToArray();
        _out.WriteLine($"tokens = [{string.Join(",", tokens)}]  n={tokens.Length}");
        Assert.Equal(5, tokens.Length);

        int[][] Split(params int[] sizes)
        {
            var res = new List<int[]>();
            int p = 0;
            foreach (int s in sizes) { res.Add(tokens[p..(p + s)]); p += s; }
            return res.ToArray();
        }

        var baseline = RunArm(model, tokens, Split(5));
        var arms = new (string name, int[][] chunks)[]
        {
            ("[5]  (repeat, determinism control)", Split(5)),
            ("[3,2] pos3=row0 n=2", Split(3, 2)),
            ("[2,3] pos3=row1 n=3", Split(2, 3)),
            ("[2,2,1] pos3=row1 n=2", Split(2, 2, 1)),
            ("[4,1] pos3=row3 n=4", Split(4, 1)),
            ("[1,1,1,1,1]", Split(1, 1, 1, 1, 1)),
        };

        int stride = model.Config.NumKvHeads * model.Config.HeadDim;
        foreach (var (name, chunks) in arms)
        {
            var arm = RunArm(model, tokens, chunks);
            _out.WriteLine($"=== {name} ===");
            double logitDelta = MaxAbs(baseline.LastLogits, arm.LastLogits, 0, baseline.LastLogits.Length);
            _out.WriteLine($"  last-row logits maxAbs = {logitDelta:E3}");
            for (int l = 0; l < model.Config.NumLayers; l++)
            {
                double k012 = MaxAbs(baseline.K[l], arm.K[l], 0, 3 * stride);
                double k3 = MaxAbs(baseline.K[l], arm.K[l], 3 * stride, stride);
                double k4 = MaxAbs(baseline.K[l], arm.K[l], 4 * stride, stride);
                double v3 = MaxAbs(baseline.V[l], arm.V[l], 3 * stride, stride);
                if (k012 != 0 || k3 != 0 || k4 != 0 || v3 != 0)
                    _out.WriteLine($"  L{l,2}: K[0..2]={k012:E3} K[3]={k3:E3} V[3]={v3:E3} K[4]={k4:E3}");
            }
        }
    }
}
