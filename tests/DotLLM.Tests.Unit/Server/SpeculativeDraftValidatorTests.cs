using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1319: <c>serve ... --speculative-model &lt;qwen4exp gguf&gt;</c> used to load the whole (87 GiB) target first, then die in the dense draft
/// loader with "--device vulkan was requested ... Architecture Qwen4Exp does not use the dense Llama-style GGUF tensor naming", blaming the
/// target and the device. The draft is now validated before the target loads and the message says what to do instead.
/// </summary>
public sealed class SpeculativeDraftValidatorTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-1319-" + Guid.NewGuid().ToString("N"));

    public SpeculativeDraftValidatorTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static ModelConfig ConfigFor(Architecture arch) => new()
    {
        Architecture = arch,
        VocabSize = 32,
        HiddenSize = 8,
        IntermediateSize = 16,
        NumLayers = 1,
        NumAttentionHeads = 2,
        NumKvHeads = 2,
        HeadDim = 4,
        MaxSequenceLength = 16,
        NormEpsilon = 1e-5f,
    };

    [Theory]
    [InlineData(Architecture.Llama)]
    [InlineData(Architecture.Qwen)]
    [InlineData(Architecture.Mistral)]
    public void DenseDraft_IsAccepted(Architecture arch)
        => SpeculativeDraftValidator.Validate(ConfigFor(arch), "draft.gguf", ConfigFor(Architecture.Llama));

    [Theory]
    [InlineData(Architecture.NemotronH)]
    [InlineData(Architecture.Qwen3MoeHybrid)]
    [InlineData(Architecture.Qwen3HybridDense)]
    [InlineData(Architecture.Lfm2Moe)]
    public void HybridDraft_IsRejectedAsAnOptionError(Architecture arch)
    {
        var ex = Assert.Throws<InvalidServerOptionException>(
            () => SpeculativeDraftValidator.Validate(ConfigFor(arch), "draft.gguf", ConfigFor(Architecture.Llama)));
        Assert.Contains("--speculative-model", ex.Message, StringComparison.Ordinal);
        Assert.Contains(arch.ToString(), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Qwen4ExpDraft_PointsAtMtpHead()
    {
        var ex = Assert.Throws<InvalidServerOptionException>(
            () => SpeculativeDraftValidator.Validate(
                ConfigFor(Architecture.Qwen4Exp), "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", ConfigFor(Architecture.Qwen4Exp)));
        Assert.Contains("--mtp-head", ex.Message, StringComparison.Ordinal);
        Assert.Contains("mtp-Qwen3.8-Flash-Next-Q8_0.gguf", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("cannot be honoured", ex.Message, StringComparison.Ordinal);   // not a device failure
    }

    /// <summary>
    /// The regression itself, through the real server load entry point: with a qwen4exp target and a qwen4exp draft the load must fail with the
    /// option error BEFORE the target is loaded. Before the fix the synthetic target loaded and the dense draft loader threw
    /// <see cref="NotSupportedException"/> from <c>TransformerWeights.LoadFromGguf</c>.
    /// </summary>
    [Fact]
    public void LoadModel_Qwen4ExpDraft_FailsEarlyWithOptionError()
    {
        string target = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "trunk.gguf"));
        string draft = SyntheticQwen4ExpGguf.WriteMtpOnly(Path.Combine(_dir, "mtp-head.gguf"));
        var options = ServerOptions.Parse(["--model", target, "--device", "cpu", "--speculative-model", draft]);

        var ex = Assert.Throws<InvalidServerOptionException>(() => ServerStartup.LoadModel(target, options));
        Assert.Contains("--mtp-head", ex.Message, StringComparison.Ordinal);
        Assert.Contains("nothing was loaded", ex.Message, StringComparison.Ordinal);
    }
}
