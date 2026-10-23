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
/// The mscale is checked on its own as well (<see cref="YarnOverride_ChangesLogits_AndRaisesWindow"/> cancels it with
/// <c>--yarn-attn-factor</c>): a host that kept the plain <c>AttnFactor</c> convention would move the logits past the parity
/// tests' tolerance.
/// </remarks>
public sealed class Qwen3HybridDenseYarnTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35-yarn-{Guid.NewGuid():N}");

    public Qwen3HybridDenseYarnTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// The host's rotation is HF's (inverse frequencies and attention factor checked in as constants), and it keeps no
    /// context-sized table: a 2,048-token YaRN window costs the same 128 bytes of frequencies as a 512-token one.
    /// </summary>
    [Fact]
    public void YarnOverride_RotationMatchesHfReference()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        using var gguf = GgufFile.Open(path);
        var config = HybridYarnParity.Config(gguf, yarn: true);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);

        float[] inv = RoPEYarnQwen35ReferenceTests.HfInvFreqFixtureBase1e4Orig512;
        var rope = model.RopeForTest;
        Assert.Equal(RoPEYarnQwen35ReferenceTests.HfAttentionFactorFactor4, rope.Mscale);
        Assert.Equal(inv.Length, rope.InverseFrequencies.Length);
        for (int i = 0; i < inv.Length; i++)
            Assert.True(MathF.Abs(inv[i] - rope.InverseFrequencies[i]) <= 4 * 1.1920929e-7f * inv[i], $"invFreq[{i}]");
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
    /// With YaRN off nothing changes: the host's rotation rows are bit-identical to the plain theta table it built before
    /// #1101.
    /// </summary>
    [Fact]
    public void YarnOff_RotationIsThePlainThetaTable()
    {
        string path = HybridYarnParity.WriteDiscriminating(_scratch);
        using var gguf = GgufFile.Open(path);
        var config = HybridYarnParity.Config(gguf, yarn: false);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        int n = config.MaxSequenceLength;
        var tableCos = new float[n * 32];
        var tableSin = new float[n * 32];
        DotLLM.Cpu.Kernels.RoPE.PrecomputeFrequencyTable(n, 64, config.RoPEConfig!.Value.Theta, tableCos, tableSin);
        var cos = new float[n * 32];
        var sin = new float[n * 32];
        model.RopeForTest.FillRows(Enumerable.Range(0, n).ToArray(), cos, sin);
        Assert.True(cos.AsSpan().SequenceEqual(tableCos));
        Assert.True(sin.AsSpan().SequenceEqual(tableSin));
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
