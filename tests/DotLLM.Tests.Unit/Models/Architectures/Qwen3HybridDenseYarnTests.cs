using DotLLM.Core.Configuration;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1101: the CPU Qwen3.5/3.8 hybrid host (the cross-backend oracle) honours <c>--rope-scaling yarn</c>. Before the
/// fix it read only <c>RoPEConfig.Theta</c>, so the override was silently ignored and the window stayed at the native size.
/// </summary>
/// <remarks>
/// Mutant runs (recorded in the PR): building the tables with the plain theta routine again fails
/// <see cref="YarnOverride_TablesMatchHfReference"/> and <see cref="YarnOverride_ChangesLogits_AndRaisesWindow"/>; the
/// pre-#1101 mscale convention (plain <c>AttnFactor</c>) fails the table test at every position.
/// </remarks>
public sealed class Qwen3HybridDenseYarnTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-yarn-{Guid.NewGuid():N}");

    public Qwen3HybridDenseYarnTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>The host's own cos/sin tables equal <c>cos/sin(pos · invFreq_HF) · attention_factor_HF</c> (HF constants).</summary>
    [Fact]
    public void YarnOverride_TablesMatchHfReference()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        using var gguf = GgufFile.Open(path);
        var config = HybridYarnParity.Config(gguf, yarn: true);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        float[] inv = RoPEYarnQwen35ReferenceTests.HfInvFreqFixtureBase1e4Orig512;
        float mscale = RoPEYarnQwen35ReferenceTests.HfAttentionFactorFactor4;
        var cos = model.RopeCosTableForTest;
        var sin = model.RopeSinTableForTest;
        Assert.Equal(config.MaxSequenceLength * 32, cos.Length);
        foreach (int pos in new[] { 0, 1, 511, 512, 1000, config.MaxSequenceLength - 1 })
        {
            for (int i = 0; i < 32; i++)
            {
                // The host forms the angle as a float32 product, as here; what remains is cos/sin rounding plus a few ulps of
                // inverse-frequency difference (MathF.Pow vs torch), which grows with the angle.
                float angleF = pos * inv[i];
                double angle = angleF;
                float bar = 2e-6f * (1 + angleF);
                Assert.True(MathF.Abs((float)(Math.Cos(angle) * mscale) - cos[pos * 32 + i]) <= bar, $"cos pos {pos} pair {i}");
                Assert.True(MathF.Abs((float)(Math.Sin(angle) * mscale) - sin[pos * 32 + i]) <= bar, $"sin pos {pos} pair {i}");
            }
        }
    }

    /// <summary>
    /// YaRN raises the window to factor × native, and changes the logits far beyond any cross-backend tolerance (so the GPU
    /// parity tests can see a host that skips it).
    /// </summary>
    [Fact]
    public void YarnOverride_ChangesLogits_AndRaisesWindow()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        var off = Run(path, yarn: false, promptLen: 500);
        var on = Run(path, yarn: true, promptLen: 500);

        float diff = HybridYarnParity.MaxAbsDiff(off.Trunk, on.Trunk);
        Assert.True(diff > HybridYarnParity.SignalFloor,
            $"YaRN moved the logits by only {diff:E3}: the override is not reaching the rotation.");

        // The mscale alone must be visible too: cancelling it (attention_factor 1/1.1386, so cos/sin are unscaled while the
        // ramped frequencies stay) is the pre-#1101 SmolLM3/Llama convention a host could silently fall back to.
        var noMscale = Run(path, yarn: true, promptLen: 500,
            attnFactor: 1.0f / RoPEYarnQwen35ReferenceTests.HfAttentionFactorFactor4);
        float mscaleDiff = HybridYarnParity.MaxAbsDiff(noMscale.Trunk, on.Trunk);
        Assert.True(mscaleDiff > HybridYarnParity.SignalFloor,
            $"Dropping the YaRN mscale moved the logits by only {mscaleDiff:E3}: the parity tests could not see it.");

        using var gguf = GgufFile.Open(path);
        Assert.Equal(HybridYarnParity.DiscriminatingNativeContext, HybridYarnParity.Config(gguf, yarn: false).MaxSequenceLength);
        Assert.Equal(4 * HybridYarnParity.DiscriminatingNativeContext, HybridYarnParity.Config(gguf, yarn: true).MaxSequenceLength);
    }

    /// <summary>
    /// The Bonsai-RoPE cacheless probe used by the GPU tests at positions past 262,144 can see YaRN: on vs off (both inside
    /// the native window, at 200,000) differ by more than the parity tests' signal floor.
    /// </summary>
    [Fact]
    public void BonsaiRope_Cacheless_YarnIsVisible()
    {
        string path = HybridYarnParity.WriteBonsaiLike(_scratch);
        List<float[]> Run16(bool yarn)
        {
            using var gguf = GgufFile.Open(path);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, HybridYarnParity.Config(gguf, yarn));
            return HybridYarnParity.RunUncached(model, 200_000, 16);
        }

        float diff = HybridYarnParity.MaxAbsDiff(Run16(false), Run16(true));
        Assert.True(diff > HybridYarnParity.SignalFloor, $"YaRN moved the cacheless logits by only {diff:E3}.");
    }

    /// <summary>Positions past the native window are accepted with YaRN (and refused without it).</summary>
    [Fact]
    public void YarnOverride_RunsPastNativeWindow()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        var on = Run(path, yarn: true, promptLen: 600, decodeSteps: 2, mtpSteps: 2);
        Assert.Equal(3, on.Trunk.Count);
        Assert.Equal(2, on.Mtp.Count);
        Assert.All(on.Trunk, row => Assert.All(row, v => Assert.True(float.IsFinite(v))));

        Assert.ThrowsAny<ArgumentException>(() => Run(path, yarn: false, promptLen: 600));
    }

    /// <summary>
    /// With YaRN off nothing changes: the tables are bit-identical to the plain theta tables the host built before #1101.
    /// </summary>
    [Fact]
    public void YarnOff_TablesAreThePlainThetaTables()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        using var gguf = GgufFile.Open(path);
        var config = HybridYarnParity.Config(gguf, yarn: false);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var cos = new float[config.MaxSequenceLength * 32];
        var sin = new float[config.MaxSequenceLength * 32];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(config.MaxSequenceLength, 64, config.RoPEConfig!.Value.Theta, cos, sin);
        Assert.True(model.RopeCosTableForTest.SequenceEqual(cos));
        Assert.True(model.RopeSinTableForTest.SequenceEqual(sin));
    }

    private static HybridYarnParity.Result Run(string path, bool yarn, int promptLen, int decodeSteps = 0, int mtpSteps = 0,
        float? attnFactor = null)
    {
        using var gguf = GgufFile.Open(path);
        var config = HybridYarnParity.Config(gguf, yarn, attnFactor);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        return HybridYarnParity.RunCached(model, kv, promptLen, decodeSteps, mtpSteps);
    }
}
