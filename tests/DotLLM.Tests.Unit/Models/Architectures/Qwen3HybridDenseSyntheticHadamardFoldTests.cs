using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// CPU-side guards for the synthetic Hadamard-fold fixture that the CUDA fold parity test
/// (<c>CudaQwen3HybridDenseHadamardFoldParityTests</c>, issue #479) compares against.
/// </summary>
/// <remarks>
/// A cross-backend parity test only discriminates a dropped or reordered transform if that
/// transform actually moves the output on the fixture. These tests pin that premise on the CPU
/// oracle, where they can run anywhere: the fold as a whole, the explicit signs, and — the one
/// that needs the non-default GDN geometry — the <c>gdn_v_grouped</c> permute each change the
/// logits by far more than the parity tolerance.
/// </remarks>
public sealed class Qwen3HybridDenseSyntheticHadamardFoldTests : IDisposable
{
    // Comfortably above the CUDA parity test's tolerance (abs 1.5e-3 + rel 5e-3), so "changes the
    // logits" here means "a parity test would see it".
    private const float MinDiscriminatingDiff = 2e-2f;

    private static readonly int[] Prompt = [1, 3, 5, 7, 9];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public Qwen3HybridDenseSyntheticHadamardFoldTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-fold-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Fixture_HasNonDegenerateGdnGeometry_AndFoldValidates()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        var gdn = config.GdnConfig!.Value;
        Assert.Equal(SyntheticHadamardFold.GdnKeyHeads, gdn.NKHead);
        Assert.Equal(SyntheticHadamardFold.GdnValueHeads, gdn.NVHead);
        Assert.True(gdn.NKHead >= 2 && gdn.VHeadsPerKHead >= 2,
            "the tiled->grouped permute is the identity unless both nKHead and rep are >= 2");

        // Construction runs ValidateQwen35FoldSet: the declaration must match the fixed sites.
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(
            gguf, config with { HadamardFold = SyntheticHadamardFold.For(config) }, ThreadingConfig.SingleThreaded);
        Assert.True(model.SupportsMtp);
    }

    /// <summary>
    /// Issue #485: the PQ2_0 variant of the fold fixture (the CUDA dp4a S=3-vs-S=1 test runs on it)
    /// must load with the fold declaration, stay finite, and actually be changed by the fold.
    /// </summary>
    [Fact]
    public void Pq2_0Fixture_FoldValidates_AndChangesLogits()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold-pq2.gguf"), withMtp: false,
            pq2_0Projections: true);
        float[] plain = Forward(path, fold: null);
        float[] folded = Forward(path, c => SyntheticHadamardFold.For(c));
        AssertAllFinite(folded);
        AssertDiffers(plain, folded, "PQ2_0 fixture: fold vs no fold");
    }

    [Fact]
    public void Fold_ChangesLogits()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        float[] plain = Forward(path, fold: null);
        float[] folded = Forward(path, c => SyntheticHadamardFold.For(c));
        AssertAllFinite(folded);
        AssertDiffers(plain, folded, "fold vs no fold");
    }

    [Fact]
    public void ExplicitSigns_ChangeLogits()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        float[] noSigns = Forward(path, c => SyntheticHadamardFold.For(c, withSigns: false));
        float[] signs = Forward(path, c => SyntheticHadamardFold.For(c, withSigns: true));
        AssertDiffers(noSigns, signs, "signs vs identity signs");
    }

    /// <summary>
    /// The model-level premise for catching a dropped <c>ssm_out</c> permute on another backend:
    /// with <c>NKHead = 2</c> the flag must matter. (On the default single-key-head fixture it
    /// cannot, which is exactly why the fold fixture overrides the geometry.)
    /// </summary>
    [Fact]
    public void GdnVGroupedPermute_ChangesLogits()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        float[] tiled = Forward(path, c => SyntheticHadamardFold.For(c, gdnVGrouped: false));
        float[] grouped = Forward(path, c => SyntheticHadamardFold.For(c, gdnVGrouped: true));
        AssertDiffers(tiled, grouped, "gdn_v_grouped on vs off");
    }

    [Fact]
    public void GdnVGroupedPermute_IsIdentity_OnDefaultSingleKeyHeadFixture()
    {
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "default.gguf"), withMtp: false);
        float[] tiled = Forward(path, c => SyntheticHadamardFold.For(c, gdnVGrouped: false));
        float[] grouped = Forward(path, c => SyntheticHadamardFold.For(c, gdnVGrouped: true));
        Assert.Equal(tiled, grouped);
    }

    /// <summary>
    /// The MTP head's two fold sites (trunk-embedding fallback, trunk lm_head fallback) must move
    /// its draft logits too.
    /// </summary>
    [Fact]
    public void Fold_ChangesMtpDraftLogits()
    {
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        float[] plain = Draft(path, fold: null);
        float[] folded = Draft(path, c => SyntheticHadamardFold.For(c));
        AssertAllFinite(folded);
        AssertDiffers(plain, folded, "MTP draft, fold vs no fold");
    }

    private static unsafe float[] Forward(string path, Func<ModelConfig, HadamardFoldConfig>? fold)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (fold is not null) config = config with { HadamardFold = fold(config) };

        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        int[] positions = Enumerable.Range(0, Prompt.Length).ToArray();
        using ITensor logits = model.Forward(Prompt, positions, deviceId: -1);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * config.VocabSize).ToArray();
    }

    private static unsafe float[] Draft(string path, Func<ModelConfig, HadamardFoldConfig>? fold)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (fold is not null) config = config with { HadamardFold = fold(config) };

        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        int[] positions = Enumerable.Range(0, Prompt.Length).ToArray();
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using var mtp = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(Prompt, positions, deviceId: -1, kv, adapter: null, mtp)) { }
        using ITensor draft = model.ForwardMtp(mtp, tokenId: 4, position: Prompt.Length);
        return new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize).ToArray();
    }

    private void AssertDiffers(float[] a, float[] b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        _out.WriteLine($"{what}: max |diff| = {max:E3}");
        Assert.True(max > MinDiscriminatingDiff,
            $"{what}: max |diff| {max:E3} <= {MinDiscriminatingDiff} — the fixture would not discriminate this transform.");
    }

    private static void AssertAllFinite(float[] v)
    {
        foreach (float x in v) Assert.True(float.IsFinite(x));
    }
}
