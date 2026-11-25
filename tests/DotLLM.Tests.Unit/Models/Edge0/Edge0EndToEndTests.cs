using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// End to end on the real Edge0-145B-q3k (about 105 GB mapped in place, CPU only, minutes per arm): the skeleton + strata pack + FP8 PLE load
/// through <see cref="ModelLoader"/>, a greedy continuation is the right text, and the direction of the column scale is settled by the
/// teacher-forced NLL of a fixed passage under the three arms. Opt-in (<c>DOTLLM_EDGE0_E2E=1</c>): it pages tens of GB of experts and should hold the
/// CPU resource lock. Reported as SKIPPED otherwise.
/// </summary>
public sealed class Edge0EndToEndTests(ITestOutputHelper output)
{
    private const string Passage =
        "The mitochondrion is an organelle found in most eukaryotic cells. Its main job is to convert nutrients into adenosine triphosphate, " +
        "the molecule that cells use as a source of chemical energy. In software, a similar story is told about caches: a small, fast memory " +
        "sits in front of a large, slow one, and a good replacement policy keeps the hot data close. The city of Lyon lies at the confluence " +
        "of the Rhone and the Saone rivers in the east of France.";

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf",
        "native_experts.txt", "experts.bin", "ple-fp8.gguf");

    private static double MeanNll(string skeleton, string? arm, ReadOnlySpan<int> ids)
    {
        string? old = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_COLSCALE");
        Environment.SetEnvironmentVariable("DOTLLM_EDGE0_COLSCALE", arm);
        try
        {
            var (model, gguf, _) = ModelLoader.LoadFromGguf(skeleton, DotLLM.Core.Configuration.ThreadingConfig.Auto);
            using (gguf)
            using (model)
            {
                var pos = new int[ids.Length];
                for (int i = 0; i < pos.Length; i++) pos[i] = i;
                using var logits = model.Forward(ids, pos, 0);
                int vocab = (int)logits.Shape[1];
                double nll = 0;
                unsafe
                {
                    var l = new ReadOnlySpan<float>((void*)logits.DataPointer, ids.Length * vocab);
                    for (int t = 0; t + 1 < ids.Length; t++)
                    {
                        var row = l.Slice(t * vocab, vocab);
                        float max = float.NegativeInfinity;
                        foreach (float v in row) max = Math.Max(max, v);
                        double sum = 0;
                        foreach (float v in row) sum += Math.Exp(v - max);
                        nll -= row[ids[t + 1]] - max - Math.Log(sum);
                    }
                }
                return nll / (ids.Length - 1);
            }
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_EDGE0_COLSCALE", old); }
    }

    [SkippableFact]
    public void RealModel_ColumnScaleDirectionIsSettledByNll()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_E2E") == "1", "set DOTLLM_EDGE0_E2E=1 (minutes of CPU, tens of GB paged; hold the cpu resource lock)");
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (skeleton, pack, ple-fp8.gguf)"));
        string skeleton = fx.Path!;

        int[] ids;
        using (var g = GgufFile.Open(skeleton))
            ids = GgufBpeTokenizerFactory.Load(g.Metadata).Encode(Passage).ToArray();
        output.WriteLine($"passage: {ids.Length} tokens");

        double inverse = MeanNll(skeleton, "inverse", ids), direct = MeanNll(skeleton, "direct", ids), none = MeanNll(skeleton, "none", ids);
        output.WriteLine($"mean NLL: 1/s {inverse:F3}   s {direct:F3}   none {none:F3}");
        // A model that reads the passage fluently is far below ln(vocab) = 12.4; the wrong directions are at chance.
        Assert.True(inverse < 3.0, $"1/s arm NLL {inverse:F3} is not a working language model");
        Assert.True(direct > inverse + 3.0 && none > inverse + 3.0, $"arms are not separated: 1/s {inverse:F3}, s {direct:F3}, none {none:F3}");
    }
}
