using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Flint;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-expert MoE layers on the generic CUDA host (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>, #1187) on the hermetic
/// <see cref="SyntheticFlintGguf"/> with routed Q4_1, shared Q8_0 and F32 attention / dense (MoE layers 2-5, layer 4 a
/// NoPE full layer, ungated shared expert, sinks, window 4 over 16 tokens). The CPU leg runs the CPU model's own routed
/// kernels (Q8_1 activations) on the mmap, so against the CPU model on the same file the remaining difference is CUDA's
/// FP16 residual stream and how that rounds the Q8_1 expert inputs. Prefill plus 6 decode steps through the CUDA KV
/// cache. A fully resident CUDA run is printed for reference: it runs the routed experts on F32 activations, so its distance to the CPU model is the Q8_1 activation noise
/// (and, on this random-weight model, an occasional routing near-tie).
/// </summary>
/// <remarks>
/// The bound can see a dropped or misplaced routed contribution: <c>CudaFlintParitySyntheticTests</c> shows the top-k
/// renormalisation alone moves these logits by more than 5x the bound, and renormalisation only acts through the routed sum.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaFlintCpuMoeTests : IDisposable
{
    /// <summary>
    /// CPU-expert CUDA vs the CPU model. Both quantise the expert inputs to Q8_1, but CUDA's inputs went through the FP16
    /// residual first, so the rounding does not cancel: measured up to 7.1e-2 (RTX 3060). Each Flint feature moves these
    /// logits by 3.0-9.1 (<c>CudaFlintParitySyntheticTests</c>), 30x above this bound.
    /// </summary>
    private const float AbsTol = 1e-1f;

    private const int PrefillTokens = 10;

    private static readonly int[] Ids = FlintSyntheticGoldenTests.Ids;

    private readonly ITestOutputHelper _out;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"dotllm-flint-cpumoe-{Guid.NewGuid():N}.gguf");

    public CudaFlintCpuMoeTests(ITestOutputHelper output)
    {
        _out = output;
        File.WriteAllBytes(_path, SyntheticFlintGguf.Build(FlintSyntheticGoldenTests.Quantized));
    }

    public void Dispose()
    {
        try { File.Delete(_path); } catch { /* best effort */ }
    }

    /// <summary>Placements: every MoE layer on the CPU, a split (layers 2-3 CPU, 4-5 device), and <c>--moe-offload on</c>.</summary>
    public static TheoryData<string> Placements => new() { "cpu-moe", "n-cpu-moe-4", "offload-on" };

    [SkippableTheory]
    [MemberData(nameof(Placements))]
    public void CpuExpertLayers_MatchCpuModel_PrefillAndDecode(string placement)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] cpu;
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded))
        using (ITensor t = cpuModel.Forward(Ids, Enumerable.Range(0, Ids.Length).ToArray(), deviceId: -1))
            cpu = Copy(t);
        List<float[]> resident = RunCuda(gguf, cfg, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, null);
        int vocab = SyntheticFlintGguf.Vocab;

        var options = placement switch
        {
            "cpu-moe" => new MoeOffloadOptions { CpuMoeLayers = MoeOffloadOptions.AllLayers, CpuThreads = 3 },
            "n-cpu-moe-4" => new MoeOffloadOptions { CpuMoeLayers = 4, CpuThreads = 1 },
            _ => new MoeOffloadOptions { Mode = MoeOffloadMode.On, CpuThreads = 2 },
        };
        int expectedCpuLayers = placement == "n-cpu-moe-4" ? 2 : 4;

        List<float[]> offloaded = RunCuda(gguf, cfg, options, expectedCpuLayers);
        for (int r = 0; r < offloaded.Count; r++)
        {
            int at = PrefillTokens - 1 + r;
            string what = $"{placement} {(r == 0 ? "prefill" : "decode")}@{at}";
            _out.WriteLine($"{what,-28} resident vs CPU model {MaxAbs(cpu.AsSpan(at * vocab, vocab), resident[r]):E3}, "
                + $"vs CPU experts {MaxAbs(resident[r], offloaded[r]):E3}");
            AssertMatch(what, cpu.AsSpan(at * vocab, vocab).ToArray(), offloaded[r]);
        }
    }

    /// <summary>Prefill of <see cref="PrefillTokens"/> then one-token decodes; returns the last row of each call.</summary>
    private static List<float[]> RunCuda(GgufFile gguf, ModelConfig cfg, MoeOffloadOptions options, int? expectedCpuLayers)
    {
        var rows = new List<float[]>();
        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0, moeOffload: options);
        using (model)
        {
            var cuda = Assert.IsType<CudaTransformerModel>(model);
            Assert.Equal(expectedCpuLayers is null, cuda.CpuMoeLayers == 0);
            using var kv = kvFactory(64);
            int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
            using (ITensor p = cuda.Forward(Ids.AsSpan(0, PrefillTokens), pos.AsSpan(0, PrefillTokens), deviceId: -1, kv))
                rows.Add(Copy(p));
            Assert.Equal(expectedCpuLayers is not null, cuda.CpuMoeExpertsComputed > 0);
            for (int i = PrefillTokens; i < Ids.Length; i++)
            {
                long before = cuda.CpuMoeExpertsComputed;
                using ITensor d = cuda.Forward(Ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, kv);
                rows.Add(Copy(d));

                // One decode token routes TopK distinct experts in each CPU layer.
                Assert.Equal((expectedCpuLayers ?? 0) * SyntheticFlintGguf.TopK, cuda.CpuMoeExpertsComputed - before);
            }
        }
        return rows;
    }

    private static float MaxAbs(ReadOnlySpan<float> a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < b.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Plan_ExplicitOnOffAndAutoFit()
    {
        long[] routed = [0, 0, 100, 100, 100, 100];
        var auto = MoeOffloadOptions.Default;
        Assert.Equal(3, CudaTransformerModel.PlanCpuMoeLayers(auto with { CpuMoeLayers = 3 }, routed, true, 1000, 10, 0));
        Assert.Equal(MoeOffloadOptions.AllLayers,
            CudaTransformerModel.PlanCpuMoeLayers(auto with { Mode = MoeOffloadMode.On }, routed, true, 500, 10_000, 0));
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto with { Mode = MoeOffloadMode.Off }, routed, true, 500, 10, 0));

        // Fits: nothing moves. Over by 150: layers 2 and 3 move (the first layer that frees enough is index 3, so N = 4).
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 600, 50));
        Assert.Equal(4, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 400, 50));
        Assert.Equal(3, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 450, 50));

        // Even every expert on the CPU does not fit; auto still moves all of them. Not eligible: auto never moves any.
        Assert.Equal(MoeOffloadOptions.AllLayers, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 50, 50));
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, false, 500, 50, 50));
    }

    private void AssertMatch(string what, float[] cpu, float[] offloaded)
    {
        Assert.Equal(cpu.Length, offloaded.Length);
        Assert.All(offloaded, v => Assert.True(float.IsFinite(v), $"{what}: non-finite logit"));
        float diff = MaxAbs(cpu, offloaded);
        _out.WriteLine($"{what,-28} max |CPU model - CPU experts| = {diff:E3}");
        Assert.True(diff <= AbsTol, $"{what}: max |CPU model - CPU experts| = {diff:E3} > {AbsTol:E3}");
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }
}
