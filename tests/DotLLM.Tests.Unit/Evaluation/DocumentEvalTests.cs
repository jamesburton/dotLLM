using System.Text.Json;
using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Per-document NLL / KL scoring (<c>dotllm perplexity --docs-file</c>, #1706). The oracles here are independent scalar
/// re-derivations (plain probabilities, hand-computed distributions), not calls back into <see cref="DocumentScoring"/>.
/// </summary>
public sealed class DocumentEvalTests
{
    private const int Vocab = 40;

    // Position-dependent, deterministic logits so a wrong row/position mapping changes the number.
    private static float[] Row(int position, int vocab)
    {
        var row = new float[vocab];
        for (int j = 0; j < vocab; j++) row[j] = (float)(3.0 * Math.Sin((position + 1) * (j + 2) * 0.173));
        return row;
    }

    private static FakePerplexityModel Model(bool allRows = true, int context = 64)
        => new(Vocab, context, allRows, Row);

    // Independent oracle: probabilities by plain (unshifted) softmax, which is exact at |logit| <= 3.
    private static double OracleNll(int[] seq)
    {
        double nll = 0;
        for (int i = 0; i < seq.Length - 1; i++)          // row i predicts token i+1
        {
            float[] x = Row(i, Vocab);
            double z = 0;
            for (int j = 0; j < Vocab; j++) z += Math.Exp(x[j]);
            nll += -Math.Log(Math.Exp(x[seq[i + 1]]) / z);
        }
        return nll;
    }

    private static int[] Doc(int n, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => rng.Next(Vocab)).ToArray();
    }

    // ---- JSONL parser / writer ----

    [Fact]
    public void ParseDocumentLine_ParsesAndRejects()
    {
        Assert.Equal(new[] { 1, 5, 9 }, DocumentJsonl.ParseDocumentLine(" [1, 5,9] "));
        Assert.Empty(DocumentJsonl.ParseDocumentLine("[]"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[1, -2]"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[1, 2.5]"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[1, \"a\"]"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("{\"a\":1}"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[1, 2"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[1] [2]"));
        Assert.Throws<FormatException>(() => DocumentJsonl.ParseDocumentLine("[3000000000]"));
    }

    [Fact]
    public void ReadDocuments_SkipsBlankLinesWithoutConsumingAnIndex()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "[1,2]\n\n[3]\n   \n[4,5,6]\n");
            var docs = DocumentJsonl.ReadDocuments(path).ToList();
            Assert.Equal(3, docs.Count);
            Assert.Equal(new[] { 4, 5, 6 }, docs[2]);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void ResultLine_RoundTripsAndNamesTheFields()
    {
        var r = new DocResult(7, 12, new DocumentScore(11, 23.5, 0.25, "full"));
        string line = DocumentJsonl.FormatResultLine(r);
        using var d = JsonDocument.Parse(line);
        var o = d.RootElement;
        Assert.Equal(7, o.GetProperty("doc").GetInt32());
        Assert.Equal(11, o.GetProperty("n_tokens_scored").GetInt32());
        Assert.Equal(23.5, o.GetProperty("nll_sum").GetDouble());
        Assert.Equal(23.5 / 11, o.GetProperty("nll_mean").GetDouble());
        Assert.Equal(0.25 / 11, o.GetProperty("kl_mean").GetDouble());
        Assert.Equal(r, DocumentJsonl.ParseResultLine(line));
        Assert.DoesNotContain('\n', line);

        // No scored token: mean is null, not NaN (NaN is not valid JSON).
        using var e = JsonDocument.Parse(DocumentJsonl.FormatResultLine(new DocResult(0, 1, new DocumentScore(0, 0, null, null))));
        Assert.Equal(JsonValueKind.Null, e.RootElement.GetProperty("nll_mean").ValueKind);
        Assert.False(e.RootElement.TryGetProperty("kl_sum", out _));
    }

    [Fact]
    public void RefLine_FullAndTopK_RoundTrip_AndInferTail()
    {
        var full = RefDocument.Parse(RefDocument.FormatFullLine(0, [[-0.5f, -1.5f, float.NegativeInfinity]]));
        Assert.Equal("full", full.Mode);
        Assert.Equal(1, full.Positions);

        // No tail_logprob: inferred as log(1 - sum p).
        var topk = RefDocument.Parse(RefDocument.FormatTopKLine(0, [[2, 0]], [[(float)Math.Log(0.5), (float)Math.Log(0.25)]], null));
        Assert.Equal("topk_tail_lb", topk.Mode);
        Assert.Throws<FormatException>(() => RefDocument.Parse("{\"doc\":0,\"format\":\"topk\",\"ids\":[[1,1]],\"logprobs\":[[-1,-2]]}"));
        Assert.Throws<FormatException>(() => RefDocument.Parse("{\"doc\":0,\"format\":\"wat\"}"));
        Assert.Throws<FormatException>(() => RefDocument.Parse("{\"format\":\"full\",\"logprobs\":[]}"));
    }

    // ---- NLL maths vs the independent oracle ----

    [Theory]
    [InlineData(2, 1)]
    [InlineData(9, 2)]
    [InlineData(31, 3)]
    public void Nll_MatchesIndependentOracle_AndCountsNMinusOneTargets(int n, int seed)
    {
        using var model = Model();
        int[] doc = Doc(n, seed);
        var sw = new StringWriter();
        var res = DocumentEvaluator.Run(model, [doc], default, sw);

        Assert.Single(res);
        Assert.Equal(n - 1, res[0].Score.TokensScored);                 // first token never scored
        Assert.Equal(OracleNll(doc), res[0].Score.NllSum, 9);
        Assert.Equal(res[0].Score.NllSum / (n - 1), res[0].Score.NllMean!.Value, 12);
        Assert.Single(model.ForwardCalls);                              // one pass for the whole document
        Assert.Equal(res[0], DocumentJsonl.ParseResultLine(sw.ToString().Trim()));
    }

    [Fact]
    public void Nll_IsInNatsNotBits()
    {
        using var model = new FakePerplexityModel(Vocab, 64, true, FakePerplexityModel.Uniform);
        var res = DocumentEvaluator.Run(model, [Doc(5, 1)], default, new StringWriter());
        Assert.Equal(4 * Math.Log(Vocab), res[0].Score.NllSum, 12);      // uniform: -ln(1/V) per target
        Assert.NotEqual(4 * Math.Log2(Vocab), res[0].Score.NllSum, 3);
    }

    [Fact]
    public void Nll_StableForHugeLogits_WithoutMaxSubtractionItWouldOverflow()
    {
        // Logits ~ 1000: exp(1000) overflows a double, so an unshifted softmax gives inf/NaN.
        // Reference values derived by hand: [1000, 990] -> -log(1/(1+e^-10)); [1000, 1000] -> log 2.
        float[] a = [1000f, 990f];
        Assert.Equal(Math.Log(1 + Math.Exp(-10)), -DocumentScoring.TargetLogProb(a, 0), 12);
        Assert.Equal(10 + Math.Log(1 + Math.Exp(-10)), -DocumentScoring.TargetLogProb(a, 1), 12);
        Assert.Equal(Math.Log(2), -DocumentScoring.TargetLogProb([1000f, 1000f], 0), 12);
    }

    [Fact]
    public void Nll_LastRowOnlyBackend_MatchesAllRowsBackend()
    {
        using var allRows = Model(allRows: true);
        using var lastRow = Model(allRows: false);
        int[] doc = Doc(12, 5);
        var a = DocumentEvaluator.Run(allRows, [doc], default, new StringWriter());
        var b = DocumentEvaluator.Run(lastRow, [doc], default, new StringWriter());
        Assert.Equal(a[0].Score.NllSum, b[0].Score.NllSum, 12);
        Assert.Equal(11, lastRow.ForwardCalls.Count);                    // one growing-prefix forward per target
    }

    [Fact]
    public void Bos_Prepended_OnlyWhenAsked_AndIsTheUnscoredFirstToken()
    {
        int[] doc = Doc(6, 4);
        using var m0 = Model();
        var plain = DocumentEvaluator.Run(m0, [doc], default, new StringWriter());
        Assert.Equal(6, plain[0].NTokens);
        Assert.Equal(5, plain[0].Score.TokensScored);

        using var m1 = Model();
        var bos = DocumentEvaluator.Run(m1, [doc], new DocumentEvalOptions(BosTokenId: 3), new StringWriter());
        Assert.Equal(7, bos[0].NTokens);
        Assert.Equal(6, bos[0].Score.TokensScored);                      // every document token is now a target
        Assert.Equal(3, m1.ForwardCalls[0][0]);
        Assert.Equal(OracleNll([3, .. doc]), bos[0].Score.NllSum, 9);
    }

    [Fact]
    public void Context_OverLimit_IsAnError_AndMaxDocTokensTruncates()
    {
        using var model = Model(context: 8);
        int[] doc = Doc(20, 6);
        var ex = Assert.Throws<ArgumentException>(() => DocumentEvaluator.Run(model, [doc], default, new StringWriter()));
        Assert.Contains("Document 0", ex.Message);
        Assert.Contains("--max-doc-tokens", ex.Message);

        var res = DocumentEvaluator.Run(model, [doc], new DocumentEvalOptions(MaxDocTokens: 8), new StringWriter());
        Assert.Equal(8, res[0].NTokens);
        Assert.Equal(OracleNll(doc[..8]), res[0].Score.NllSum, 9);

        // Truncation counts the BOS.
        var withBos = DocumentEvaluator.Run(model, [doc], new DocumentEvalOptions(BosTokenId: 1, MaxDocTokens: 8), new StringWriter());
        Assert.Equal(8, withBos[0].NTokens);
    }

    [Fact]
    public void ShortDocuments_ScoreNothing_AndOutOfVocabIsAnError()
    {
        using var model = Model();
        var res = DocumentEvaluator.Run(model, [[], [5], [5, 6]], default, new StringWriter());
        Assert.Equal(new[] { 0, 0, 1 }, res.Select(r => r.Score.TokensScored));
        Assert.Single(model.ForwardCalls);                              // only the 2-token document ran a forward
        Assert.Throws<ArgumentException>(() => DocumentEvaluator.Run(model, [[1, Vocab]], default, new StringWriter()));
    }

    // ---- KL ----

    private static float[] LogSoftmax(float[] x)
    {
        double z = x.Sum(v => Math.Exp(v));
        return x.Select(v => (float)(v - Math.Log(z))).ToArray();
    }

    [Fact]
    public void Kl_OfADistributionWithItself_IsZero_FullAndTopK()
    {
        float[] x = Row(3, Vocab);
        Assert.Equal(0, DocumentScoring.KlFull(LogSoftmax(x), x), 6);

        // Top-K = the whole vocabulary: no tail, still zero.
        int[] all = Enumerable.Range(0, Vocab).ToArray();
        Assert.Equal(0, DocumentScoring.KlTopKTail(all, LogSoftmax(x), double.NegativeInfinity, x), 6);
    }

    [Fact]
    public void Kl_IsAsymmetric_AndHasTheTeacherToModelDirection()
    {
        // p = [0.9, 0.1], q = [0.5, 0.5]. KL(p||q) = 0.9 ln 1.8 + 0.1 ln 0.2 ; KL(q||p) = 0.5 ln(0.5/0.9) + 0.5 ln 5.
        float[] logP = [(float)Math.Log(0.9), (float)Math.Log(0.1)];
        float[] modelLogits = [0f, 0f];                                  // q uniform
        double klPq = 0.9 * Math.Log(1.8) + 0.1 * Math.Log(0.2);
        double klQp = 0.5 * Math.Log(0.5 / 0.9) + 0.5 * Math.Log(5.0);
        Assert.NotEqual(klPq, klQp, 3);
        Assert.Equal(klPq, DocumentScoring.KlFull(logP, modelLogits), 6);          // teacher = reference file, model = scored
        Assert.Equal(klQp, DocumentScoring.KlFull(LogSoftmax(modelLogits), logP), 6);   // swapped roles give the other number
    }

    [Fact]
    public void KlTopKTail_IsALowerBoundOfFull_AndExactWhenTailFollowsTheModel()
    {
        float[] teacher = Row(5, Vocab);
        float[] model = Row(9, Vocab);
        float[] lpT = LogSoftmax(teacher);
        double full = DocumentScoring.KlFull(lpT, model);

        int[] top = Enumerable.Range(0, Vocab).OrderByDescending(i => lpT[i]).Take(8).ToArray();
        float[] topLp = top.Select(i => lpT[i]).ToArray();
        double tail = Math.Log(1 - topLp.Sum(v => Math.Exp(v)));
        double lb = DocumentScoring.KlTopKTail(top, topLp, tail, model);
        Assert.True(lb <= full + 1e-9, $"lower bound {lb} must not exceed full KL {full}");
        Assert.True(lb > 0.5 * full, "bound should be informative here");

        // Hand value for K=1 on a 3-vocab: teacher [.6,.3,.1], top-1 = id 0, tail .4; model q = [.2,.5,.3].
        float[] m3 = [(float)Math.Log(0.2), (float)Math.Log(0.5), (float)Math.Log(0.3)];
        double expect = 0.6 * Math.Log(0.6 / 0.2) + 0.4 * Math.Log(0.4 / 0.8);
        Assert.Equal(expect, DocumentScoring.KlTopKTail([0], [(float)Math.Log(0.6)], Math.Log(0.4), m3), 5);
    }

    [Fact]
    public void Evaluator_KlAgainstItself_IsZero_AndAgainstAnotherModelMatchesTheOracle()
    {
        int[] doc = Doc(10, 8);
        // Reference = this model's own distributions (full format) -> KL ~ 0.
        var selfRows = Enumerable.Range(0, 9).Select(i => LogSoftmax(Row(i, Vocab))).ToList();
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, RefDocument.FormatFullLine(0, selfRows) + "\n");
            using var model = Model();
            using var refs = new RefLogProbsReader(path);
            var res = DocumentEvaluator.Run(model, [doc], default, new StringWriter(), refs);
            Assert.Equal("full", res[0].Score.KlMode);
            Assert.Equal(0, res[0].Score.KlSum!.Value, 5);

            // A different reference (shifted positions): KL must equal the sum of independent per-row KLs.
            var otherRows = Enumerable.Range(0, 9).Select(i => LogSoftmax(Row(i + 4, Vocab))).ToList();
            File.WriteAllText(path, RefDocument.FormatFullLine(0, otherRows) + "\n");
            using var refs2 = new RefLogProbsReader(path);
            using var model2 = Model();
            var res2 = DocumentEvaluator.Run(model2, [doc], default, new StringWriter(), refs2);
            double oracle = 0;
            for (int i = 0; i < 9; i++)
            {
                double zq = Enumerable.Range(0, Vocab).Sum(j => Math.Exp(Row(i, Vocab)[j]));
                for (int v = 0; v < Vocab; v++)
                {
                    double p = Math.Exp(otherRows[i][v]);
                    double q = Math.Exp(Row(i, Vocab)[v]) / zq;
                    oracle += p * Math.Log(p / q);
                }
            }
            Assert.True(oracle > 0.01);
            Assert.Equal(oracle, res2[0].Score.KlSum!.Value, 5);
            Assert.Equal(oracle / 9, res2[0].Score.KlMean!.Value, 6);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void Evaluator_RejectsMismatchedReferences()
    {
        int[] doc = Doc(6, 2);
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, RefDocument.FormatFullLine(0, [LogSoftmax(Row(0, Vocab))]) + "\n");   // 1 position, doc needs 5
            using var model = Model();
            using var refs = new RefLogProbsReader(path);
            Assert.Throws<ArgumentException>(() => DocumentEvaluator.Run(model, [doc], default, new StringWriter(), refs));

            File.WriteAllText(path, RefDocument.FormatFullLine(1, [LogSoftmax(Row(0, Vocab))]) + "\n");   // wrong doc index
            using var refs2 = new RefLogProbsReader(path);
            Assert.Throws<FormatException>(() => refs2.Next());

            File.WriteAllText(path, "");                                                                    // missing entry
            using var refs3 = new RefLogProbsReader(path);
            Assert.Throws<ArgumentException>(() => DocumentEvaluator.Run(model, [doc], default, new StringWriter(), refs3));
        }
        finally { File.Delete(path); }
    }
}
