using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan <c>granite</c> / <c>granitemoe</c> (GGUF) against the independent double-precision
/// <see cref="GraniteGgufReference"/> (#764, #313) — all four Granite scalars != 1 and mutually distinct. The reference
/// (not the CPU backend) is the oracle for the MoE fixture: the CPU routed-MoE path uses Q8-quantized activations like
/// llama.cpp, so on Q8_0 experts it sits ~1% from the exact value, whereas the Vulkan F32 arms keep F32 activations
/// (measured on gfx1151: Vulkan vs exact reference 4.5e-3, CPU vs exact reference 7e-2 on this fixture).
/// <para>
/// Since #912 (e8252373) the generic MoE layer routes decode-sized row counts (&lt;= <see cref="VulkanTransformerModel.GenericMoeMmvqMaxRows"/>)
/// onto the indexed dp4a MMVQ arm, which quantizes the activations to Q8_1 exactly like the CPU path, so a 1-token cached decode no longer sits
/// 4.5e-3 from the exact value (measured 3.6e-2 on this fixture, #1674). The cached-decode step therefore checks the F32 (scalar) arm against the
/// exact reference, then the default MMVQ arm against the F32 arm with a Q8_1-activation bound, and proves by the layer counters that each arm
/// actually ran. The 20-token cacheless forward is above the arm's row ceiling and stays on F32 activations.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public sealed class GraniteVulkanReferenceTests
{
    private readonly ITestOutputHelper _output;

    public GraniteVulkanReferenceTests(ITestOutputHelper output) => _output = output;

    private static readonly int[] Prompt = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 63, 21, 40, 6, 33, 78, 14, 90, 27, 3];

    /// <summary>
    /// Q8_1-activation bound for the MMVQ decode arm against the F32 arm of the same model: the activation quantizer rounds each 32-wide block
    /// to 8 bits (relative step 1/127 of the block max), the dp4a sum is exact in int32, so the deviation is a sum of K/32 blocks' rounding
    /// noise through three projections per layer, amplified by <c>LogitScale</c> at the end. The sibling MMVQ gate
    /// (<see cref="VulkanTransformerModelGenericMoeMmvqDecodeTests"/>) bounds the same arm at 5% of the logit max-abs + 0.02; the Granite fixture's
    /// scalars (<c>LogitScale</c>=5, <c>EmbeddingScale</c>=3) make the rows larger, so the same relative form is used. Measured on gfx1151 (#1674):
    /// 3.6e-2 absolute on a row of max-abs ~1 (the deliberately wrong kernels of the #1674 mutant run land at 0.4 .. 2+, see the PR).
    /// </summary>
    private const double MmvqRelativeTolerance = 0.05, MmvqAbsoluteTolerance = 0.02;

    public static IEnumerable<object[]> Fixtures()
    {
        yield return new object[] { new SyntheticGraniteConfig() };
        yield return new object[] { new SyntheticGraniteConfig { Hidden = 256, Heads = 4, KvHeads = 4, Tied = true, Layers = 2 } };
        yield return new object[] { new SyntheticGraniteConfig { Arch = "granitemoe", ExpertsQ8_0 = true } };
    }

    private static unsafe float[] Last(ITensor t)
    {
        int rows = t.Shape[0], v = t.Shape[1];
        var r = new float[v];
        new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * v, v).CopyTo(r);
        return r;
    }

    private static double Max(float[] a, double[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static string SpvDir()
    {
        string dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "spv"));
        if (!Directory.Exists(dir) || Directory.GetFiles(dir, "*.spv").Length == 0)
            dir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        return dir;
    }

    [SkippableTheory]
    [MemberData(nameof(Fixtures))]
    public void Vulkan_MatchesIndependentReference_PrefillCachedDecodeAndAblations(SyntheticGraniteConfig cfg)
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device available on this host.");

        var wts = SyntheticGraniteGguf.BuildWeights(cfg);
        string path = Path.Combine(Path.GetTempPath(), $"granite_vk_ref_{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGraniteGguf.Serialize(wts));
        try
        {
            using var gguf = GgufFile.Open(path);
            ModelConfig config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.True(config.HasGraniteScalars);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, SpvDir());

            // Cacheless full forward -> last-row logits.
            double[] reference = GraniteGgufReference.LastTokenLogits(wts, Prompt);
            double dCacheless;
            using (ITensor l = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), -1, kvCache: null))
                dCacheless = Max(Last(l), reference);
            _output.WriteLine($"{cfg.Arch}: cacheless Vulkan vs reference = {dCacheless:E3}");
            Assert.True(dCacheless < 0.03, $"cacheless Vulkan vs reference = {dCacheless:E3}");

            // KV-cache prefill + single-token decode of the final position. On the MoE fixture a 1-token forward is decode-sized, so the
            // generic MoE layer takes the Q8_1-activation MMVQ arm by default (#912): check the F32 (scalar) arm against the exact reference
            // first, then the MMVQ arm against the F32 arm. Both arms run on the SAME loaded model and cache; the layer counters prove which
            // arm each forward took (a silent fallback or a silent re-route would fail here even with acceptable logits).
            int split = Prompt.Length - 1;
            bool mmvq = cfg.IsMoe && model.GenericMoeMmvqAvailable;
            float[] decodeMmvq;
            using (var cache = model.CreateKvCache(maxSeqLen: Prompt.Length + 1))
            {
                using (model.Forward(Prompt[..split], Enumerable.Range(0, split).ToArray(), -1, cache)) { }

                model.GenericMoeMmvqEnabled = false;
                long mm0 = model.GenericMoeMmvqLayerCount, sc0 = model.GenericMoeScalarDecodeLayerCount;
                float[] decodeF32;
                using (ITensor l = model.Forward([Prompt[split]], [split], -1, cache)) decodeF32 = Last(l);
                Assert.Equal(0, model.GenericMoeMmvqLayerCount - mm0);
                if (cfg.IsMoe) Assert.Equal(cfg.Layers, model.GenericMoeScalarDecodeLayerCount - sc0);
                double dF32 = Max(decodeF32, reference);
                _output.WriteLine($"{cfg.Arch}: cached decode (F32 arm) Vulkan vs reference = {dF32:E3}");
                Assert.True(dF32 < 0.03, $"cached decode (F32 arm) Vulkan vs reference = {dF32:E3}");

                model.GenericMoeMmvqEnabled = true;
                mm0 = model.GenericMoeMmvqLayerCount; sc0 = model.GenericMoeScalarDecodeLayerCount;
                using (ITensor l = model.Forward([Prompt[split]], [split], -1, cache)) decodeMmvq = Last(l);
                if (mmvq)
                {
                    Assert.Equal(cfg.Layers, model.GenericMoeMmvqLayerCount - mm0);
                    Assert.Equal(0, model.GenericMoeScalarDecodeLayerCount - sc0);
                }
                double maxAbs = Math.Max(decodeF32.Max(MathF.Abs), 1e-3);
                double dArms = 0;
                for (int i = 0; i < decodeF32.Length; i++) dArms = Math.Max(dArms, Math.Abs(decodeMmvq[i] - decodeF32[i]));
                double dMmvqRef = Max(decodeMmvq, reference);
                _output.WriteLine($"{cfg.Arch}: cached decode (default arm, mmvq={mmvq}) vs F32 arm = {dArms:E3}, vs reference = {dMmvqRef:E3}, logit maxAbs = {maxAbs:G4}");
                double bound = mmvq ? MmvqRelativeTolerance * maxAbs + MmvqAbsoluteTolerance : 0;   // without the arm both forwards are the same dispatch: bit-identical
                Assert.True(dArms <= bound, $"cached decode: MMVQ arm vs F32 arm max|diff| = {dArms:E3} > {bound:E3} (logit maxAbs {maxAbs:G4})");
                Assert.Equal(ArgMax(decodeF32), ArgMax(decodeMmvq));
            }

            // Each scalar must matter: the ablated reference is farther from the Vulkan result than the tolerance — for the F32 cacheless row
            // AND for the default-arm decode row (so the MMVQ arm's Q8_1 noise cannot hide a dropped scalar either).
            var ablations = new (string Name, GraniteGgufReference.Options Opt)[]
            {
                ("embedding scale", new() { EmbeddingScale = false }),
                ("attention scale", new() { AttentionScale = false }),
                ("residual scale", new() { ResidualScale = false }),
                ("logit scale", new() { LogitScale = false }),
            };
            using (ITensor l = model.Forward(Prompt, Enumerable.Range(0, Prompt.Length).ToArray(), -1, kvCache: null))
            {
                float[] gpu = Last(l);
                foreach (var (name, opt) in ablations)
                {
                    double[] ablated = GraniteGgufReference.LastTokenLogits(wts, Prompt, opt);
                    double d = Max(gpu, ablated), dDecode = Max(decodeMmvq, ablated);
                    _output.WriteLine($"{cfg.Arch}: without {name}: cacheless {d:E3}, decode {dDecode:E3}");
                    Assert.True(d > 0.1, $"{cfg.Arch}: Vulkan result is within {d:E3} of the reference WITHOUT the {name} (not sensitive enough).");
                    Assert.True(dDecode > 0.1, $"{cfg.Arch}: Vulkan decode result is within {dDecode:E3} of the reference WITHOUT the {name} (not sensitive enough).");
                }
            }
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort */ }
        }
    }
}
