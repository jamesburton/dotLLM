using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Image prompt end to end (issue #2035): dotLLM's text logits after an image prompt against llama.cpp b12780 CPU (<c>-fa off</c>, mtmd in F32 attention),
/// teacher-forced over the oracle's own greedy continuation. The oracle (<c>tools/inkling2-vision-oracle/oracle.c</c>) records the exact token ids mtmd
/// feeds (so tokenizer differences cannot contaminate it), the image, and the FULL-vocabulary F32 logits of the prompt and of every continuation step.
/// Heavy: loads the 17.6 GB NVFP4 model on the CPU.
/// </summary>
public sealed class Inkling2VisionEndToEndTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private GgufFile? _file;
    private Inkling2TransformerModel? _model;
    private Inkling2VisionEncoder? _encoder;

    public Inkling2VisionEndToEndTests(ITestOutputHelper output) => _output = output;

    public void Dispose() { _encoder?.Dispose(); _model?.Dispose(); _file?.Dispose(); }

    private sealed record Oracle(DecodedImage Image, int[] Before, int[] After, int ImageTokens, int[] Gen, float[] Logits, int NVocab);

    private static Oracle? LoadOracle(string name)
    {
        string dir = Path.Combine(Inkling2VisionFixtures.OracleDir, name);
        string meta = Path.Combine(dir, "meta.txt"), ppm = Path.Combine(dir, "image.ppm"), lg = Path.Combine(dir, "logits.f32");
        if (!File.Exists(meta) || !File.Exists(ppm) || !File.Exists(lg)) return null;
        int[]? before = null, after = null, gen = null; int imageTokens = 0, nVocab = 0;
        foreach (string line in File.ReadAllLines(meta))
        {
            var t = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (t[0] == "text") { var ids = t.Skip(2).Select(int.Parse).ToArray(); if (before is null) before = ids; else after = ids; }
            else if (t[0] == "image") imageTokens = int.Parse(t[1]);
            else if (t[0] == "n_vocab") nVocab = int.Parse(t[1]);
            else if (t[0] == "gen") gen = t.Skip(1).Select(int.Parse).ToArray();
        }
        var bytes = File.ReadAllBytes(lg);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return new Oracle(ImageDecoder.Load(ppm), before!, after!, imageTokens, gen!, f, nVocab);
    }

    private (Inkling2TransformerModel Model, Inkling2VisionEncoder Encoder) Models()
    {
        string? gguf = Inkling2RealModelTests.FindGguf();
        Skip.If(gguf is null, "Inkling-2-Mini-NVFP4.gguf not in the HF cache");
        string? mm = Inkling2VisionFixtures.FindMmproj();
        Skip.If(mm is null, "mmproj-Inkling-2-Mini-Q8_0.gguf not in the HF cache");
        if (_model is null)
        {
            _file = GgufFile.Open(gguf!);
            var cfg = GgufModelConfigExtractor.Extract(_file.Metadata);
            _model = Inkling2TransformerModel.LoadFromGguf(_file, cfg, ThreadingConfig.Auto, Inkling2Options.FromEnvironment());
            _encoder = Inkling2VisionEncoder.Load(mm!, ThreadingConfig.Auto);
        }
        return (_model, _encoder!);
    }

    private static unsafe float[] Row(ITensor t, int row, int vocab)
    {
        var a = new float[vocab];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)row * t.Shape[1], vocab).CopyTo(a);
        return a;
    }

    /// <summary>KL(oracle || ours) over the oracle's vocabulary, in double precision (full distribution, no truncation).</summary>
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

    /// <summary>Runs the image prompt then teacher-forces the oracle's continuation, scoring every row.</summary>
    private Score Run(Oracle o, Inkling2TransformerModel model, Inkling2VisionEncoder encoder, int? placeholder = null)
    {
        int ph = placeholder ?? 200054;   // <|unused_200054|>: only the ids' count matters, the embeddings replace them
        using var emb = encoder.Encode(o.Image);
        Assert.Equal(o.ImageTokens, emb.Tokens);
        var prompt = o.Before.Concat(Enumerable.Repeat(ph, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        var state = model.CreateState();
        int V = o.NVocab;
        double sum = 0, max = 0; int top = 0, mism = 0, rows = o.Gen.Length + 1;
        // Prefill in 32-token calls (positions continue the state), like the text gate, so carried KV/conv state is exercised.
        float[] ours = [];
        for (int at = 0; at < prompt.Length; at += 32)
        {
            int len = Math.Min(32, prompt.Length - at);
            using var t = model.ForwardWithImagesAndState(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, state, spans, lastTokenLogitsOnly: true);
            if (at + len == prompt.Length) ours = Row(t, 0, model.Config.VocabSize);
        }
        for (int r = 0; r < rows; r++)
        {
            var (kl, same) = FullKl(o.Logits.AsSpan(r * V, V), ours);
            sum += kl; max = Math.Max(max, kl); if (same) top++;
            if (r < o.Gen.Length)
            {
                int arg = 0; for (int i = 1; i < V; i++) if (ours[i] > ours[arg]) arg = i;
                if (arg != o.Gen[r]) mism++;
                using var step = model.ForwardWithImagesAndState([o.Gen[r]], [prompt.Length + r], -1, state, spans, lastTokenLogitsOnly: true);
                ours = Row(step, 0, model.Config.VocabSize);
            }
        }
        return new Score(sum / rows, max, top, rows, mism);
    }

    [SkippableTheory]
    [InlineData("rainbow190x110")]
    [InlineData("rainbow80")]
    [InlineData("rainbow640")]
    public void ImagePrompt_Logits_MatchLlamaCpp(string name)
    {
        var o = LoadOracle(name);
        Skip.If(o is null, "vision oracle (vision/<name>/logits.f32) not generated");
        var (model, enc) = Models();
        var s = Run(o!, model, enc);
        _output.WriteLine($"{name}: mean KL {s.MeanKl:E2}, max KL {s.MaxKl:E2}, top-1 {s.Top1}/{s.Rows}, teacher-forced greedy mismatches {s.GreedyMismatch}/{o!.Gen.Length}");
        Assert.True(s.MeanKl < Gate, $"mean KL {s.MeanKl:E2} over {s.Rows} positions (gate {Gate:E1})");
        Assert.True(s.Top1 >= s.Rows - 2, $"top-1 agreement {s.Top1}/{s.Rows}");
    }

    /// <summary>Set from the unmutated scores (printed by the test above) with ~3x headroom, well below every mutant (see <see cref="Mutants_AreKilled"/>).</summary>
    private const double Gate = 0.05;

    public static TheoryData<string> Mutants => new()
    {
        "NormalizeImageEmbeddings", "Bgr", "PosembAlignCorners", "PosembTransposed", "RopeAxesSwapped", "RopeNeox", "MergeOrder", "MergeSkipped", "NoLayerScale", "NoCls",
        "WrongPlaceholderRows",
    };

    /// <summary>"Prove the test can fail": the end-to-end gate on the non-square image against deliberately wrong implementations.</summary>
    [SkippableTheory]
    [MemberData(nameof(Mutants))]
    public void Mutants_AreKilled(string mutant)
    {
        var o = LoadOracle("rainbow190x110");
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        model.Mutant = Inkling2Mutant.None; enc.Mutant = Inkling2VisionMutant.None; enc.SwapRgb = false;
        try
        {
            switch (mutant)
            {
                case "NormalizeImageEmbeddings": model.Mutant = Inkling2Mutant.NormalizeImageEmbeddings; break;
                case "Bgr": enc.SwapRgb = true; break;
                case "WrongPlaceholderRows": break;   // handled below: image rows replaced by the placeholder token embedding (embeddings not injected)
                default: enc.Mutant = Enum.Parse<Inkling2VisionMutant>(mutant); break;
            }
            Score s;
            if (mutant == "WrongPlaceholderRows")
            {
                // The injection ignored: the placeholder token's own embedding sits at every image position.
                var spans = Array.Empty<ImageSpan>();
                var prompt = o!.Before.Concat(Enumerable.Repeat(200054, o.ImageTokens)).Concat(o.After).ToArray();
                var state = model.CreateState();
                float[] ours = [];
                for (int at = 0; at < prompt.Length; at += 32)
                {
                    int len = Math.Min(32, prompt.Length - at);
                    using var t = model.ForwardWithImagesAndState(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, state, spans, lastTokenLogitsOnly: true);
                    if (at + len == prompt.Length) ours = Row(t, 0, model.Config.VocabSize);
                }
                var (kl, same) = FullKl(o.Logits.AsSpan(0, o.NVocab), ours);
                s = new Score(kl, kl, same ? 1 : 0, 1, 0);
            }
            else s = Run(o!, model, enc);
            _output.WriteLine($"{mutant}: mean KL {s.MeanKl:E2}, max KL {s.MaxKl:E2}, top-1 {s.Top1}/{s.Rows}, greedy mismatches {s.GreedyMismatch}");
            Assert.True(s.MeanKl > Gate, $"mutant {mutant} survived the end-to-end gate: mean KL {s.MeanKl:E2} (gate {Gate:E1})");
        }
        finally { model.Mutant = Inkling2Mutant.None; enc.Mutant = Inkling2VisionMutant.None; enc.SwapRgb = false; }
    }
}
