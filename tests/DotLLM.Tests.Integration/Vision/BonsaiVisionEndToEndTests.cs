using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vision;

/// <summary>Loads the 27B text model once for the end-to-end image tests (CPU, mmap'd, never copied).</summary>
public sealed class BonsaiTextFixture : IDisposable
{
    private GgufFile? _gguf;
    private Qwen3HybridDenseTransformerModel? _model;
    private readonly object _gate = new();

    internal Qwen3HybridDenseTransformerModel Model(string path)
    {
        lock (_gate)
        {
            if (_model is null)
            {
                _gguf = GgufFile.Open(path);
                var cfg = GgufModelConfigExtractor.Extract(_gguf.Metadata);
                _model = Qwen3HybridDenseTransformerModel.LoadFromGguf(_gguf, cfg, ThreadingConfig.Auto);
            }
            return _model;
        }
    }

    public void Dispose() { _model?.Dispose(); _gguf?.Dispose(); }
}

/// <summary>
/// Image prompt end to end (issue #2073): dotLLM CPU text logits after an image prompt against llama.cpp b12780 CPU (<c>-fa off</c>), teacher-forced over the
/// oracle's own greedy continuation, on the upstream Q2_0 file both engines load. The oracle records the exact token ids mtmd feeds (so tokenizer differences
/// cannot contaminate it) and the full-vocabulary F32 logits of the prompt's last position and of every continuation step.
/// </summary>
[Trait("Category", "Heavy")]
public sealed class BonsaiVisionEndToEndTests : IClassFixture<BonsaiTextFixture>
{
    private const int ImagePad = 248056;   // <|image_pad|>: only the count matters, the embeddings replace it
    private readonly BonsaiTextFixture _fx;
    private readonly ITestOutputHelper _out;

    public BonsaiVisionEndToEndTests(BonsaiTextFixture fx, ITestOutputHelper output) { _fx = fx; _out = output; }

    private static unsafe float[] Row(ITensor t, int row, int vocab)
    {
        var a = new float[vocab];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * t.Shape[1], vocab).CopyTo(a);
        return a;
    }

    /// <summary>KL(oracle || ours) over the oracle's vocabulary, in double precision.</summary>
    internal static (double Kl, bool SameTop) FullKl(ReadOnlySpan<float> oracle, ReadOnlySpan<float> ours)
    {
        static double Lse(ReadOnlySpan<float> x, out int arg)
        {
            float max = float.NegativeInfinity; arg = 0;
            for (int i = 0; i < x.Length; i++) if (x[i] > max) { max = x[i]; arg = i; }
            double s = 0;
            for (int i = 0; i < x.Length; i++) s += Math.Exp(x[i] - max);
            return max + Math.Log(s);
        }
        int n = oracle.Length;
        double lo = Lse(oracle, out int ao), lu = Lse(ours[..n], out int au);
        double kl = 0;
        for (int i = 0; i < n; i++)
        {
            double lp = oracle[i] - lo;
            kl += Math.Exp(lp) * (lp - (ours[i] - lu));
        }
        return (kl, ao == au);
    }

    internal sealed record Score(double MeanKl, double MaxKl, int Top1, int Rows, int GreedyMismatch);

    /// <summary>Runs the image prompt (chunked as asked) then teacher-forces the oracle's continuation; returns the logits rows and the greedy mismatches.</summary>
    internal static (float[][] Rows, int Mismatch) RunRows(BonsaiVisionFixtures.Oracle o, Qwen3HybridDenseTransformerModel model, ImageEmbeddings emb, int chunk = int.MaxValue)
    {
        var prompt = o.Before.Concat(Enumerable.Repeat(ImagePad, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        var cfg = model.Config;
        using var kv = new SimpleKvCache(model.AttentionLayerCount, cfg.NumKvHeads, cfg.HeadDim, prompt.Length + o.Gen.Length + 8);
        int V = o.NVocab, mism = 0;
        float[] ours = [];
        for (int at = 0; at < prompt.Length; at += chunk)
        {
            int len = Math.Min(chunk, prompt.Length - at);
            using var t = model.ForwardWithImages(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, spans, kv, lastTokenLogitsOnly: true);
            if (at + len == prompt.Length) ours = Row(t, 0, cfg.VocabSize);
        }
        var rows = new List<float[]>();
        for (int r = 0; r < o.Gen.Length + 1; r++)
        {
            rows.Add(ours);
            if (r < o.Gen.Length)
            {
                int arg = 0; for (int i = 1; i < V; i++) if (ours[i] > ours[arg]) arg = i;
                if (arg != o.Gen[r]) mism++;
                using var step = model.Forward([o.Gen[r]], [prompt.Length + r], -1, kv, lastTokenLogitsOnly: true);
                ours = Row(step, 0, cfg.VocabSize);
            }
        }
        return ([.. rows], mism);
    }

    internal static Score ScoreRows(BonsaiVisionFixtures.Oracle o, float[][] rows, int mism)
    {
        int V = o.NVocab;
        double sum = 0, max = 0; int top = 0;
        for (int r = 0; r < rows.Length; r++)
        {
            var (kl, same) = FullKl(o.Logits.AsSpan(r * V, V), rows[r]);
            sum += kl; max = Math.Max(max, kl); if (same) top++;
        }
        return new Score(sum / rows.Length, max, top, rows.Length, mism);
    }

    private Qwen3HybridDenseTransformerModel Model() => _fx.Model(BonsaiVisionFixtures.ResolveTextQ2_0());

    [SkippableTheory]
    [InlineData("rainbow80")]
    [InlineData("news224x168")]
    [InlineData("news640x488")]
    public void Report_EncoderArms_EndToEnd(string name)
    {
        var o = BonsaiVisionFixtures.LoadOracle(name);
        var model = Model();
        string mm = BonsaiVisionFixtures.ResolveMmproj();
        // oracle embeddings themselves (an exact-encoder control: isolates the text side)
        {
            using var oracleEmb = new ImageEmbeddings(Grid(o).Gw, Grid(o).Gh, 5120, "oracle");
            for (int i = 0; i < oracleEmb.Rows.Length; i++) oracleEmb.Rows[i] = (Half)o.Embd[i];
            var (rows, mism) = RunRows(o, model, oracleEmb);
            var s = ScoreRows(o, rows, mism);
            _out.WriteLine($"{name} [oracle embeddings, F16]: mean KL {s.MeanKl:E2} max {s.MaxKl:E2} top1 {s.Top1}/{s.Rows} greedy mismatches {s.GreedyMismatch}/{o.Gen.Length}");
        }
        foreach (var prec in new[] { VisionEmbedPrecision.F32, VisionEmbedPrecision.F16, VisionEmbedPrecision.BF16 })
        {
            using var enc = QwenVlVisionEncoder.Load(mm, ThreadingConfig.Auto, embedPrecision: prec);
            using var emb = enc.Encode(o.Image);
            var (rows, mism) = RunRows(o, model, emb);
            var s = ScoreRows(o, rows, mism);
            _out.WriteLine($"{name} [our encoder embd={prec}]: mean KL {s.MeanKl:E2} max {s.MaxKl:E2} top1 {s.Top1}/{s.Rows} greedy mismatches {s.GreedyMismatch}/{o.Gen.Length}");
        }
    }

    private static (int Gw, int Gh) Grid(BonsaiVisionFixtures.Oracle o)
    {
        var (w, h) = QwenVlImagePreprocessor.PlanSize(o.Image.Width, o.Image.Height);
        return (w / 32, h / 32);
    }
}
