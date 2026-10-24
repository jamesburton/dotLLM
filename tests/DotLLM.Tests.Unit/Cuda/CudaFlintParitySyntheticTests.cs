using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Flint;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-vs-CUDA parity for Recursive Flint (#1172, F6) on the hermetic <see cref="SyntheticFlintGguf"/>: GQA 8:2, sliding
/// window 4 over 16 tokens, NoPE on the full layers (layer 4 is a NoPE MoE layer at positions up to 15), non-zero learned
/// sinks on every layer, two leading dense layers, sigmoid top-3 of 8 with renormalisation and one ungated shared expert.
/// Loads through <see cref="CudaModelLoader.CreateFromGguf"/>, the dispatch whose Flint refusal this issue lifted.
/// </summary>
/// <remarks>
/// <para>
/// Variants: everything Q4_1 (the official file's layout: dense, attention, routed and shared experts all run the in-place
/// Q4_1 kernels) and routed Q4_1 + shared Q8_0 + F32 dense. Prefill compares the last row; decode prefills 10 tokens and
/// then decodes 6 through the CUDA KV cache (the fused RoPE + KV write with rope_dim 0 on NoPE layers, the grouped Q4_1
/// GEMV, the window clipping at positions past 4) against the CPU batched forward's rows.
/// </para>
/// <para>
/// <see cref="EveryFlintFeatureMovesLogitsFarAboveParityTolerance"/> measures each feature's isolated effect on the CPU so
/// the parity bound is known to be able to see CUDA dropping it (the <c>CudaGptOssParitySyntheticTests</c> pattern).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaFlintParitySyntheticTests : IDisposable
{
    /// <summary>
    /// Exact-F32 CPU vs the FP16 CUDA loop (F16 residual / activations / KV). Measured on an RTX 3060 (2026-10-23): worst
    /// 1.6e-2 (all Q4_1) and 2.7e-2 (Q8_0 shared expert, decode) over every compared row; the smallest feature effect below
    /// is the window off-by-one probe.
    /// </summary>
    private const float AbsTol = 5e-2f;

    /// <summary>Each feature must move the CPU logits by at least this multiple of <see cref="AbsTol"/>.</summary>
    private const float MinEffectSizeMultiple = 5f;

    private const int PrefillTokens = 10;

    private static readonly int[] Ids = FlintSyntheticGoldenTests.Ids;

    private static readonly SyntheticFlintGguf.Options AllQ4_1 = new()
    {
        DenseQuant = QuantizationType.Q4_1,
        RoutedQuant = QuantizationType.Q4_1,
        SharedQuant = QuantizationType.Q4_1,
    };

    private static readonly SyntheticFlintGguf.Options RoutedQ4_1SharedQ8_0 = FlintSyntheticGoldenTests.Quantized;

    private readonly ITestOutputHelper _out;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-flint-cuda-{Guid.NewGuid():N}");

    public CudaFlintParitySyntheticTests(ITestOutputHelper output)
    {
        _out = output;
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, true); } catch { /* best effort */ }
    }

    public static TheoryData<string> Variants => new() { "all-q4_1", "routed-q4_1-shared-q8_0" };

    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void Prefill_LastToken_MatchesCpu(string variant)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(Write(Options(variant), variant));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Flint, cfg.Architecture);

        float[] cpu = ExactCpuRows(variant);
        int vocab = SyntheticFlintGguf.Vocab;
        float[] cpuLast = cpu.AsSpan((Ids.Length - 1) * vocab, vocab).ToArray();

        var (model, _) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0);
        using (model)
        {
            Assert.IsType<CudaTransformerModel>(model);
            using ITensor t = model.Forward(Ids, Enumerable.Range(0, Ids.Length).ToArray(), deviceId: -1);
            AssertMatch($"{variant} prefill", cpuLast, Copy(t));
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Variants))]
    public void Decode_ThroughKvCache_MatchesCpuRows(string variant)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(Write(Options(variant), variant));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] cpu = ExactCpuRows(variant);
        int vocab = SyntheticFlintGguf.Vocab;

        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0);
        using (model)
        {
            using DotLLM.Core.Attention.IKvCache kv = kvFactory(64);
            int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
            using (ITensor p = model.Forward(Ids.AsSpan(0, PrefillTokens), pos.AsSpan(0, PrefillTokens), deviceId: -1, kv))
                AssertMatch($"{variant} prefill@{PrefillTokens - 1}", cpu.AsSpan((PrefillTokens - 1) * vocab, vocab).ToArray(), Copy(p));
            for (int i = PrefillTokens; i < Ids.Length; i++)
            {
                using ITensor d = model.Forward(Ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, kv);
                AssertMatch($"{variant} decode@{i}", cpu.AsSpan(i * vocab, vocab).ToArray(), Copy(d));
            }
        }
    }

    /// <summary>CPU-only: NoPE, the sliding window, the shared expert, renormalisation and the sinks each move the logits far above <see cref="AbsTol"/>.</summary>
    [Fact]
    public void EveryFlintFeatureMovesLogitsFarAboveParityTolerance()
    {
        using var gguf = GgufFile.Open(Write(AllQ4_1, "eff"));
        using var noSinks = GgufFile.Open(Write(AllQ4_1 with { Sinks = false }, "eff-nosinks"));
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] baseline = CpuRows(gguf, cfg);
        var mutants = new (string Name, float[] Rows)[]
        {
            ("NoPE on full layers", CpuRows(gguf, cfg with { NoRopeLayers = null })),
            ("sliding window 4", CpuRows(gguf, cfg with
            {
                PerLayerSlidingWindow = cfg.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : 4096).ToArray(),
            })),
            ("window off by one (5)", CpuRows(gguf, cfg with
            {
                PerLayerSlidingWindow = cfg.PerLayerSlidingWindow!.Select(w => w is null ? (int?)null : w + 1).ToArray(),
            })),
            ("ungated shared expert", CpuRows(gguf, cfg with { Moe = cfg.Moe! with { SharedExpertIntermediateSize = null, NumSharedExperts = 0 } })),
            ("top-k renormalisation", CpuRows(gguf, cfg with { Moe = cfg.Moe! with { NormalizeExpertWeights = false } })),
            ("learned sinks", CpuRows(noSinks, cfg)),
        };
        float bar = AbsTol * MinEffectSizeMultiple;
        foreach (var (name, rows) in mutants)
        {
            float effect = MaxAbs(baseline, rows);
            _out.WriteLine($"{name,-26} effect {effect:E3} ({effect / AbsTol:F1}x AbsTol)");
            Assert.True(effect > bar, $"{name} moves the logits by only {effect:E3} <= {bar:E3}: parity could not see CUDA drop it.");
        }
    }

    /// <summary>
    /// The CPU oracle runs the same weights dequantised to F32 (<see cref="SyntheticFlintGguf.Options.F32Roundtrip"/>), so
    /// the comparison is exact math vs CUDA's in-place Q4_1 kernels; the CPU's own Q4_1 MoE path quantises activations to
    /// Q8_1 (#1158) and would add its noise to the bound.
    /// </summary>
    private float[] ExactCpuRows(string variant)
    {
        using var exact = GgufFile.Open(Write(Options(variant) with { F32Roundtrip = true }, variant + "-f32"));
        return CpuRows(exact, GgufModelConfigExtractor.Extract(exact.Metadata));
    }

    private static SyntheticFlintGguf.Options Options(string variant)
        => variant == "all-q4_1" ? AllQ4_1 : RoutedQ4_1SharedQ8_0;

    private string Write(SyntheticFlintGguf.Options o, string name)
    {
        string p = Path.Combine(_scratch, name + ".gguf");
        File.WriteAllBytes(p, SyntheticFlintGguf.Build(o));
        return p;
    }

    private static float[] CpuRows(GgufFile gguf, ModelConfig cfg)
    {
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded);
        using ITensor t = model.Forward(Ids, Enumerable.Range(0, Ids.Length).ToArray(), deviceId: -1);
        return Copy(t);
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private void AssertMatch(string what, float[] cpu, float[] cuda)
    {
        Assert.Equal(cpu.Length, cuda.Length);
        Assert.All(cuda, v => Assert.True(float.IsFinite(v), $"{what}: non-finite CUDA logit"));
        float diff = MaxAbs(cpu, cuda);
        _out.WriteLine($"{what,-36} max |cpu - cuda| = {diff:E3}");
        Assert.True(diff <= AbsTol, $"{what}: max |cpu - cuda| = {diff:E3} > {AbsTol:E3}");
    }
}
