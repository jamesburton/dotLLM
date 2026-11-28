using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;

namespace DotLLM.Tests.Integration.Vision;

/// <summary>Locations and oracle loading shared by the Bonsai 2.5 vision tests (issue #2073).</summary>
internal static class BonsaiVisionFixtures
{
    internal const string Org = "prism-ml", Repo = "Ternary-Bonsai-2.5-27B-GGUF";
    internal static readonly string[] Images = ["rainbow80", "news224x168", "news640x488"];

    /// <summary>The oracle cache (<c>tools/bonsai25-vision-oracle</c>): <c>DOTLLM_BONSAI25_VISION_ORACLE_DIR</c> or the default test cache.</summary>
    internal static string OracleDir => Environment.GetEnvironmentVariable("DOTLLM_BONSAI25_VISION_ORACLE_DIR") is { Length: > 0 } d
        ? d
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "token-oracles", "bonsai25-vision");

    internal static string ResolveMmproj()
    {
        var f = TestFixtureResolver.ResolveFile("DOTLLM_BONSAI25_MMPROJ_GGUF", Org, Repo, "Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf");
        Skip.If(!f.Found, f.SkipMessage("Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf"));
        return f.Path!;
    }

    /// <summary>The upstream-ggml Q2_0 file: stock llama.cpp b12780 loads it, so it is the file the end-to-end oracle ran on.</summary>
    internal static string ResolveTextQ2_0()
    {
        var f = TestFixtureResolver.ResolveFile("DOTLLM_BONSAI25_Q2_0_GGUF", Org, Repo, "Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf");
        Skip.If(!f.Found, f.SkipMessage("Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf"));
        return f.Path!;
    }

    internal static string ResolveTextPq2_0()
    {
        var f = TestFixtureResolver.ResolveFile("DOTLLM_BONSAI25_PQ2_0_GGUF", Org, Repo, "Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf");
        Skip.If(!f.Found, f.SkipMessage("Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf"));
        return f.Path!;
    }

    internal sealed record Oracle(string Name, DecodedImage Image, int[] Before, int[] After, int ImageTokens, int[] Gen, float[] Embd, float[] Logits, int NVocab);

    internal static Oracle LoadOracle(string name)
    {
        string dir = Path.Combine(OracleDir, name);
        string meta = Path.Combine(dir, "meta.txt"), ppm = Path.Combine(dir, "image.ppm"), lg = Path.Combine(dir, "logits.f32"), em = Path.Combine(dir, "embd.f32");
        Skip.If(!File.Exists(meta) || !File.Exists(ppm) || !File.Exists(lg) || !File.Exists(em),
            $"Bonsai 2.5 vision oracle '{name}' missing under {OracleDir} (generate it with tools/bonsai25-vision-oracle).");
        int[]? before = null, after = null, gen = null; int imageTokens = 0, nVocab = 0;
        foreach (string line in File.ReadAllLines(meta))
        {
            var t = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (t.Length == 0) continue;
            if (t[0] == "text") { var ids = t.Skip(2).Select(int.Parse).ToArray(); if (before is null) before = ids; else after = ids; }
            else if (t[0] == "image") imageTokens = int.Parse(t[1]);
            else if (t[0] == "n_vocab") nVocab = int.Parse(t[1]);
            else if (t[0] == "gen") gen = t.Skip(1).Select(int.Parse).ToArray();
        }
        return new Oracle(name, ImageDecoder.Load(ppm), before!, after!, imageTokens, gen!, ReadF32(em), ReadF32(lg), nVocab);
    }

    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    /// <summary>relative RMS error of <paramref name="got"/> (F16 rows) against the oracle's F32 rows, and the smallest per-row cosine.</summary>
    internal static (double RelRms, double MinRowCosine) Compare(ImageEmbeddings got, float[] oracle)
    {
        int dim = got.Dimension;
        double num = 0, den = 0, minCos = 1;
        var row = new float[dim];
        for (int t = 0; t < got.Tokens; t++)
        {
            got.ReadRow(t, row);
            double dot = 0, a2 = 0, b2 = 0;
            for (int i = 0; i < dim; i++)
            {
                double o = oracle[t * dim + i], g = row[i];
                num += (g - o) * (g - o); den += o * o;
                dot += g * o; a2 += g * g; b2 += o * o;
            }
            minCos = Math.Min(minCos, dot / Math.Sqrt(a2 * b2));
        }
        return (Math.Sqrt(num / den), minCos);
    }
}
