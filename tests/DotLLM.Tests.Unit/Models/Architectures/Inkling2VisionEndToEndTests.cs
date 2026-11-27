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

    /// <summary>
    /// Runs the image prompt then teacher-forces the oracle's continuation; returns our full-vocabulary logits of every scored row (the prompt's last
    /// position, then one per continuation token) and how many greedy picks differ from the oracle's continuation.
    /// </summary>
    private (float[][] Rows, int GreedyMismatch) RunRows(Oracle o, Inkling2TransformerModel model, Inkling2VisionEncoder encoder, int? placeholder = null, int chunk = 32)
    {
        int ph = placeholder ?? 200054;   // <|unused_200054|>: only the ids' count matters, the embeddings replace them
        using var emb = encoder.Encode(o.Image);
        Assert.Equal(o.ImageTokens, emb.Tokens);
        var prompt = o.Before.Concat(Enumerable.Repeat(ph, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        var state = model.CreateState();
        int V = o.NVocab, mism = 0;
        var rows = new List<float[]>();
        // Prefill in 32-token calls (positions continue the state), like the text gate, so carried KV/conv state is exercised.
        float[] ours = [];
        for (int at = 0; at < prompt.Length; at += chunk)
        {
            int len = Math.Min(chunk, prompt.Length - at);
            using var t = model.ForwardWithImagesAndState(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, state, spans, lastTokenLogitsOnly: true);
            if (at + len == prompt.Length) ours = Row(t, 0, model.Config.VocabSize);
        }
        for (int r = 0; r < o.Gen.Length + 1; r++)
        {
            rows.Add(ours);
            if (r < o.Gen.Length)
            {
                int arg = 0; for (int i = 1; i < V; i++) if (ours[i] > ours[arg]) arg = i;
                if (arg != o.Gen[r]) mism++;
                using var step = model.ForwardWithImagesAndState([o.Gen[r]], [prompt.Length + r], -1, state, spans, lastTokenLogitsOnly: true);
                ours = Row(step, 0, model.Config.VocabSize);
            }
        }
        return ([.. rows], mism);
    }

    private static Score ScoreRows(Oracle o, float[][] rows, int mism)
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

    /// <summary>Runs the image prompt then teacher-forces the oracle's continuation, scoring every row.</summary>
    private Score Run(Oracle o, Inkling2TransformerModel model, Inkling2VisionEncoder encoder, int? placeholder = null)
    {
        var (rows, mism) = RunRows(o, model, encoder, placeholder);
        return ScoreRows(o, rows, mism);
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

    /// <summary>
    /// Review finding: the chunk-relative-index mutant of <c>TryImageRow</c> (image rows looked up by the position inside the chunk instead of the absolute
    /// position) was killed only by rainbow640. This is the exact, noise-free check on two more images: prefill in chunks of 4, 5 and 7 tokens (the image rows
    /// straddle chunk boundaries at offsets different from the 32-token run) and compare the <c>embed</c> trace of every chunk with the rows it must hold: an
    /// image position holds the encoder's row (F16 to F32, no norm), a text position the RMS-normalised token embedding (the other chunking's rows).
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80", 5)]
    [InlineData("rainbow190x110", 7)]
    [InlineData("rainbow190x110", 4)]
    public void ChunkedPrefill_ImageRows_AreAddressedByAbsolutePosition(string name, int chunk)
    {
        var o = LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        using var emb = enc.Encode(o!.Image);
        var prompt = o.Before.Concat(Enumerable.Repeat(200054, emb.Tokens)).Concat(o.After).ToArray();
        var spans = new[] { new ImageSpan(o.Before.Length, emb) };
        int H = model.Config.HiddenSize;

        float[] Embeds(int ch)
        {
            var all = new float[prompt.Length * H];
            int at = 0;
            model.Trace = (n, data, rows, cols) =>
            {
                if (n == "embed") { data.CopyTo(all.AsSpan(at * H, rows * cols)); }
            };
            try
            {
                var state = model.CreateState();
                for (at = 0; at < prompt.Length; at += ch)
                {
                    int len = Math.Min(ch, prompt.Length - at);
                    using var t = model.ForwardWithImagesAndState(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, state, spans, lastTokenLogitsOnly: true);
                }
            }
            finally { model.Trace = null; }
            return all;
        }
        var reference = Embeds(prompt.Length);   // one chunk: no boundary at all
        var chunked = Embeds(chunk);
        Assert.Equal(reference, chunked);        // identical, bit for bit, at every position
        var row = new float[H];
        for (int i = 0; i < emb.Tokens; i++)
        {
            emb.ReadRow(i, row);
            Assert.Equal(row, chunked.AsSpan((o.Before.Length + i) * H, H).ToArray());   // image positions hold the encoder rows, not token rows
        }
        Assert.NotEqual(chunked.AsSpan(0, H).ToArray(), chunked.AsSpan(o.Before.Length * H, H).ToArray());
    }

    /// <summary>
    /// MEASUREMENT (an open finding, not a gate): how much the final-position logits of an IMAGE prompt depend on the prefill chunk size, against the text-only
    /// control (the same prompt with the placeholder token's own embedding instead of the image rows) chunked the same way. The text control is chunk-size
    /// invariant (KL 3e-5 to 2e-4); the image prompt is not (KL 2e-3 to 1.3e-1, same top-1). The embedding rows are identical (see the exact test), so the
    /// difference arises downstream, plausibly from the large-norm image rows making the Q8_0 activation quantisation sensitive to summation-order drift
    /// between the GEMV / GEMM paths; unresolved, reported to the user.
    /// </summary>
    [SkippableTheory]
    [InlineData("rainbow80", 5)]
    [InlineData("rainbow190x110", 7)]
    [InlineData("rainbow190x110", 4)]
    public void Report_ChunkSizeDependenceOfImagePrefill(string name, int chunk)
    {
        var o = LoadOracle(name);
        Skip.If(o is null, "vision oracle not generated");
        var (model, enc) = Models();
        var reference = RunRows(o!, model, enc).Rows[0];
        var chunked = RunRows(o!, model, enc, chunk: chunk).Rows[0];
        var (kl, same) = FullKl(reference, chunked);
        float[] TextLast(int ch)
        {
            var prompt = o!.Before.Concat(Enumerable.Repeat(200054, o.ImageTokens)).Concat(o.After).ToArray();
            var state = model.CreateState();
            float[] last = [];
            for (int at = 0; at < prompt.Length; at += ch)
            {
                int len = Math.Min(ch, prompt.Length - at);
                using var t = model.ForwardWithState(prompt.AsSpan(at, len), Enumerable.Range(at, len).ToArray(), -1, state, lastTokenLogitsOnly: true);
                if (at + len == prompt.Length) last = Row(t, 0, model.Config.VocabSize);
            }
            return last;
        }
        var (controlKl, _) = FullKl(TextLast(32), TextLast(chunk));
        _output.WriteLine($"{name} chunk {chunk}: image KL vs chunk 32 {kl:E2} (same top-1 {same}); text-only control KL {controlKl:E2}");
        Assert.True(same, "the top-1 token must not depend on the chunk size");
    }

    /// <summary>
    /// The measurement for the user's decision on <c>patch_embd</c> / <c>position_embd</c> storage (docs/INKLING2.md "Embedding precision"): the end-to-end
    /// teacher-forced KL of the text logits on the three test images for each arm (F32 = as stored, F16, BF16), against llama.cpp's logits and against our
    /// own F32 arm (which isolates the arm from the reference's noise). Asserts only that the arms differ (a no-op conversion would be identical) and that the
    /// shipped gate still passes for the arm under test; the table is the output.
    /// </summary>
    [SkippableFact]
    public void Report_EmbedPrecisionArms_EndToEndKl()
    {
        var (model, _) = Models();
        string mm = Inkling2VisionFixtures.FindMmproj()!;
        var lines = new List<string> { "image | arm | mean KL vs llama.cpp | max KL | top-1 | greedy mismatches | mean KL vs our F32 arm | max KL vs F32 arm" };
        foreach (string name in new[] { "rainbow80", "rainbow190x110", "rainbow640" })
        {
            var o = LoadOracle(name);
            Skip.If(o is null, "vision oracle not generated");
            float[][]? baseline = null;
            foreach (var arm in new[] { VisionEmbedPrecision.F32, VisionEmbedPrecision.F16, VisionEmbedPrecision.BF16 })
            {
                using var enc = Inkling2VisionEncoder.Load(mm, ThreadingConfig.Auto, embedPrecision: arm);
                var (rows, mism) = RunRows(o!, model, enc);
                baseline ??= rows;
                var s = ScoreRows(o!, rows, mism);
                double self = 0, selfMax = 0;
                for (int r = 0; r < rows.Length; r++)
                {
                    var (kl, _) = FullKl(baseline[r], rows[r]);
                    self += kl; selfMax = Math.Max(selfMax, kl);
                }
                lines.Add($"{name,-14} | {arm,-4} | {s.MeanKl:E2} | {s.MaxKl:E2} | {s.Top1}/{s.Rows} | {s.GreedyMismatch}/{o!.Gen.Length} | {self / rows.Length:E2} | {selfMax:E2}");
                Assert.True(s.MeanKl < Gate, $"{name} {arm}: mean KL {s.MeanKl:E2} over the gate {Gate:E1}");
                if (arm != VisionEmbedPrecision.F32) Assert.True(self > 0, $"{name} {arm}: identical to the F32 arm, the conversion did nothing");
            }
        }
        foreach (string l in lines) _output.WriteLine(l);
    }

    /// <summary>
    /// Unmutated mean KL: 1.8e-3 (80), 1.07e-2 (190x110), 9.6e-3 (640), against a 4e-3 text-only floor. The gate sits 1.4x above the worst of them: thin on
    /// purpose, because the position-embedding mutants (3.2e-2 align_corners, 1.95e-2 transposed) are only 2-3x the floor end to end (they are sharp at
    /// the checkpoint gate); every other mutant is 0.06 to 1.7.
    /// </summary>
    private const double Gate = 0.015;

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
