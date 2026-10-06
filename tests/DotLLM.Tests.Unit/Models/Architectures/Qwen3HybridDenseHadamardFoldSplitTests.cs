using System.Collections.Frozen;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #481: a CPU/GPU partial-offload split of a Hadamard-folded <c>qwen35</c> checkpoint.
/// <see cref="Qwen3HybridDenseTransformerModel.LoadTailFromGguf"/> validated its SLICED layer count
/// (with LOCAL block indices) against the full-trunk <c>prism.hadamard.weight_names</c> declaration,
/// so every folded checkpoint was refused. Each half now validates the blocks it owns against the
/// full declaration, and both still refuse any declared name that no block of the model rotates.
/// </summary>
/// <remarks>
/// Everything here is host-only and runs anywhere; the device-side twin is
/// <c>HybridQwen3HybridDenseHadamardFoldSplitParityTests</c> (CUDA). The fixture is the 4-block
/// <c>[GDN, Attn, GDN, Attn]</c> trunk with the non-degenerate GDN head geometry. The tail is tested
/// at every split point: at an even split, local and global block indices have the same parity, so
/// the old local-index validation would have computed the right layer KINDS by accident; odd splits
/// are the discriminating case for kind resolution.
/// </remarks>
public sealed class Qwen3HybridDenseHadamardFoldSplitTests : IDisposable
{
    private const int Blocks = 4;
    private const int Interval = SyntheticQwen35HybridDenseMtpGguf.FullAttnInterval;

    // Same bar as Qwen3HybridDenseSyntheticHadamardFoldTests: well above any parity tolerance.
    private const float MinDiscriminatingDiff = 2e-2f;

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public Qwen3HybridDenseHadamardFoldSplitTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-fold-split-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Fixture(bool withMtp = false) =>
        SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, $"fold-{Guid.NewGuid():N}.gguf"),
            withMtp: withMtp, blockCount: Blocks);

    private static HadamardFoldConfig Without(HadamardFoldConfig fold, string name)
    {
        Assert.True(fold.FoldedWeights.Contains(name), $"fixture declaration lacks {name}");
        return fold with { FoldedWeights = fold.FoldedWeights.Where(n => n != name).ToFrozenSet(StringComparer.Ordinal) };
    }

    private static HadamardFoldConfig With(HadamardFoldConfig fold, string name) =>
        fold with { FoldedWeights = fold.FoldedWeights.Append(name).ToFrozenSet(StringComparer.Ordinal) };

    // ── The regression ────────────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Tail_LoadsFoldedCheckpoint_AtEverySplit(int startLayer)
    {
        using var gguf = GgufFile.Open(Fixture());
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Blocks, config.NumLayers);
        var fold = SyntheticHadamardFold.For(config);

        using var tail = Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config with { HadamardFold = fold }, startLayer, ThreadingConfig.SingleThreaded);
        Assert.Equal(Blocks - startLayer, tail.Config.NumLayers);
    }

    /// <summary>The MTP-bearing variant: the tail slice never owns the MTP block, and must still load.</summary>
    [Fact]
    public void Tail_LoadsFoldedCheckpoint_WithMtpBlockPresent()
    {
        using var gguf = GgufFile.Open(Fixture(withMtp: true));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var tail = Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config with { HadamardFold = SyntheticHadamardFold.For(config) }, 2, ThreadingConfig.SingleThreaded);
        Assert.False(tail.SupportsMtp);
    }

    // ── The tail validates what it owns, and refuses what nobody owns ─────────────────────────

    [Theory]
    [InlineData(1, "blk.3.ffn_down.weight")]   // last tail block
    [InlineData(1, "blk.1.attn_q.weight")]     // first tail block, an ATTENTION block at an odd split
    [InlineData(2, "blk.2.attn_qkv.weight")]   // first tail block, a GDN block
    [InlineData(3, "output.weight")]           // the tail runs the lm_head
    public void Tail_RefusesDeclarationMissingATailOwnedName(int startLayer, string missing)
    {
        using var gguf = GgufFile.Open(Fixture());
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = Without(SyntheticHadamardFold.For(config), missing);

        var ex = Assert.Throws<NotSupportedException>(() => Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config with { HadamardFold = fold }, startLayer, ThreadingConfig.SingleThreaded));
        _out.WriteLine(ex.Message);
        Assert.Contains(missing, ex.Message);
    }

    /// <summary>
    /// The half-ownership rule, not a blanket relaxation: a name only the GPU head rotates is the
    /// head's to require. (The composition's up-front whole-model check still refuses this
    /// declaration — see <see cref="Composition_RefusesUncoveredDeclaration_BeforeTouchingCuda"/>.)
    /// </summary>
    [Theory]
    [InlineData(2, "blk.0.ffn_down.weight")]
    [InlineData(2, "blk.1.attn_output.weight")]
    [InlineData(3, "blk.2.ssm_out.weight")]
    public void Tail_DoesNotRequireHeadOwnedNames(int startLayer, string headOwned)
    {
        using var gguf = GgufFile.Open(Fixture());
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = Without(SyntheticHadamardFold.For(config), headOwned);

        using var tail = Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config with { HadamardFold = fold }, startLayer, ThreadingConfig.SingleThreaded);
    }

    [Theory]
    [InlineData("blk.4.attn_q.weight")]      // the MTP block's own projection: no backend rotates it
    [InlineData("blk.2.ssm_alpha.weight")]   // not folded in Bonsai 2; the fixed sites never rotate it
    [InlineData("blk.9.ffn_up.weight")]      // beyond the trunk
    public void Tail_RefusesDeclaredNameNoBlockRotates(string extra)
    {
        using var gguf = GgufFile.Open(Fixture(withMtp: true));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = With(SyntheticHadamardFold.For(config), extra);

        var ex = Assert.Throws<NotSupportedException>(() => Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config with { HadamardFold = fold }, 2, ThreadingConfig.SingleThreaded));
        Assert.Contains(extra, ex.Message);
    }

    // ── The head's half of the rule (what CudaHadamardRotation.Create runs for LoadHeadFromGguf) ──

    [Fact]
    public void HeadOwnership_RequiresOnlyItsPrefix_AndNotTheLmHead()
    {
        var config = ExtractConfig();
        var fold = SyntheticHadamardFold.For(config);
        var gdn = config.GdnConfig!.Value;

        void ValidateHead(HadamardFoldConfig f) =>
            new HadamardActivationRotator(f, gdn).ValidateQwen35FoldSet(
                Blocks, Interval, ownedFirstLayer: 0, ownedLayerCount: 2, ownsLmHead: false);

        ValidateHead(fold);
        ValidateHead(Without(fold, "output.weight"));          // the tail's
        ValidateHead(Without(fold, "blk.3.ffn_down.weight"));  // the tail's

        Assert.Throws<NotSupportedException>(() => ValidateHead(Without(fold, "blk.1.attn_q.weight")));
        Assert.Throws<NotSupportedException>(() => ValidateHead(Without(fold, "blk.0.attn_gate.weight")));
        Assert.Throws<NotSupportedException>(() => ValidateHead(With(fold, "blk.4.attn_q.weight")));
    }

    /// <summary>The whole-model defaults stay exact set equality (the pre-#481 contract).</summary>
    [Fact]
    public void WholeModelDefaults_AreStillExactSetEquality()
    {
        var config = ExtractConfig();
        var fold = SyntheticHadamardFold.For(config);
        var rotator = new HadamardActivationRotator(fold, config.GdnConfig!.Value);
        rotator.ValidateQwen35FoldSet(Blocks, Interval);

        foreach (string name in new[] { "output.weight", "blk.0.ffn_up.weight", "blk.3.attn_v.weight" })
            Assert.Throws<NotSupportedException>(() =>
                new HadamardActivationRotator(Without(fold, name), config.GdnConfig!.Value)
                    .ValidateQwen35FoldSet(Blocks, Interval));
    }

    [Fact]
    public void OwnedRangeOutsideTrunk_Throws()
    {
        var config = ExtractConfig();
        var rotator = new HadamardActivationRotator(SyntheticHadamardFold.For(config), config.GdnConfig!.Value);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            rotator.ValidateQwen35FoldSet(Blocks, Interval, ownedFirstLayer: 3, ownedLayerCount: 2));
    }

    // ── The composition refuses what neither half covers, before any CUDA call ────────────────

    /// <summary>
    /// <see cref="HybridQwen3HybridDenseTransformerModel.LoadFromGguf"/> runs the whole-model check
    /// host-side before creating a CUDA context, so this passes on a machine with no NVIDIA GPU —
    /// which is also the proof that the check precedes every device call.
    /// </summary>
    [Theory]
    [InlineData("+blk.4.attn_q.weight")]
    [InlineData("-blk.0.ffn_down.weight")]
    [InlineData("-output.weight")]
    public void Composition_RefusesUncoveredDeclaration_BeforeTouchingCuda(string edit)
    {
        using var gguf = GgufFile.Open(Fixture(withMtp: true));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = SyntheticHadamardFold.For(config);
        string name = edit[1..];
        fold = edit[0] == '+' ? With(fold, name) : Without(fold, name);

        var ex = Assert.Throws<NotSupportedException>(() => HybridQwen3HybridDenseTransformerModel.LoadFromGguf(
            gguf, config with { HadamardFold = fold }, numGpuLayers: 2, deviceId: 0, ThreadingConfig.SingleThreaded));
        Assert.Contains(name, ex.Message);
    }

    // ── The tail applies the rotations ────────────────────────────────────────────────────────

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Tail_AppliesFold(int startLayer)
    {
        string path = Fixture();
        float[] plain = TailForward(path, startLayer, fold: false);
        float[] folded = TailForward(path, startLayer, fold: true);

        Assert.Equal(plain.Length, folded.Length);
        float max = 0;
        for (int i = 0; i < plain.Length; i++)
        {
            Assert.True(float.IsFinite(folded[i]));
            max = MathF.Max(max, MathF.Abs(plain[i] - folded[i]));
        }
        _out.WriteLine($"tail@{startLayer} fold vs no fold: max |diff| = {max:E3}");
        Assert.True(max > MinDiscriminatingDiff,
            $"max |diff| {max:E3} <= {MinDiscriminatingDiff}: the tail did not apply the fold.");
    }

    private static unsafe float[] TailForward(string path, int startLayer, bool fold)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (fold) config = config with { HadamardFold = SyntheticHadamardFold.For(config) };

        using var tail = Qwen3HybridDenseTransformerModel.LoadTailFromGguf(
            gguf, config, startLayer, ThreadingConfig.SingleThreaded);

        const int seqLen = 3;
        int hidden = config.HiddenSize;
        var rng = new Random(481);
        var h = new float[seqLen * hidden];
        for (int i = 0; i < h.Length; i++) h[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] positions = [0, 1, 2];

        using ITensor logits = tail.ForwardFromHiddenState(h, positions, deviceId: -1, kvCache: null, gdnState: null);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * config.VocabSize).ToArray();
    }

    private ModelConfig ExtractConfig()
    {
        using var gguf = GgufFile.Open(Fixture());
        return GgufModelConfigExtractor.Extract(gguf.Metadata);
    }
}
