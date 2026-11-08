using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Adapters.Ternary;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Tests.Unit.Vulkan.Ternary;
using Xunit;
using Fixture = DotLLM.Tests.Unit.Vulkan.VulkanHybridLegacyQuantForwardTests.LegacyQuantFixture;

namespace DotLLM.Tests.Unit.Adapters;

/// <summary>
/// Entry-point behaviour of <c>--ternary-adapter</c> (#1711): the shared attach path wires every model that has a residual hook, and REFUSES (hard,
/// specific error) every model that does not: a silently ignored adapter would hand the user base-model output.
/// </summary>
public sealed class TernaryAdapterAttachTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("nsattach").FullName;
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    /// <summary>A model with no residual hook (stands for any unwired architecture/backend).</summary>
    private sealed class HooklessModel : IModel
    {
        public ModelConfig Config { get; } = new() { Architecture = Architecture.Mamba3, VocabSize = 8, HiddenSize = 256, NumLayers = 2, IntermediateSize = 1, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 1, MaxSequenceLength = 8 };
        public long ComputeMemoryBytes => 0;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    private string Adapter(int d, int layer)
    {
        string path = Path.Combine(_dir, $"a{layer}.nsadapter");
        VulkanTernaryAdapterKernelTests.Synth(4, d, 5, 1f, 0f, layer).Save(path);
        return path;
    }

    [Fact]
    public void ModelWithoutHook_IsRefused_WithASpecificError_NotSilentlyIgnored()
    {
        var m = new HooklessModel();
        string why = Assert.IsType<string>(TernaryAdapterAttach.Unsupported(m, m.Config));
        Assert.Contains(nameof(HooklessModel), why);
        Assert.Contains("Mamba3", why);
        Assert.Contains("silently ignored", why);
        var ex = Assert.Throws<NotSupportedException>(() => TernaryAdapterAttach.Attach(m, m.Config, [Adapter(256, 0)]));
        Assert.Contains("--ternary-adapter", ex.Message);
    }

    [Fact]
    public void Server_RefusesAHooklessModel_AndDisposesIt()
    {
        string gp = Path.Combine(_dir, "tiny.gguf");
        File.WriteAllBytes(gp, DotLLM.Tests.Unit.ChunkReuse.SyntheticDenseGguf.Build("llama"));
        var gguf = GgufFile.Open(gp);
        var m = new HooklessModel();
        var ex = Assert.Throws<InvalidServerOptionException>(() =>
            ServerStartup.AttachTernaryAdapters(m, m.Config, gguf, new ServerOptions { Model = "x", TernaryAdapterPaths = [Adapter(256, 0)] }));
        Assert.Contains("no residual-modifier hook", ex.Message);
    }

    [Fact]
    public void HybridDenseAndMoeHybrid_AreWired_AndHandleDisposeDetaches()
    {
        using var fx = Fixture.Build(1718, QuantizationType.Q4_1);
        int d = fx.DenseConfig.HiddenSize;
        using var dense = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32, fx.OutputWeightPtr, QuantizationType.Q4_1, 8, d);
        using var moe = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fx.MoeConfig, fx.MoeLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32, fx.OutputWeightPtr, QuantizationType.Q4_1, 8, d);
        string a = Adapter(d, 1);
        Assert.Null(TernaryAdapterAttach.Unsupported(dense, fx.DenseConfig));
        Assert.Null(TernaryAdapterAttach.Unsupported(moe, fx.MoeConfig));
        using (var h = TernaryAdapterAttach.Attach(dense, fx.DenseConfig, [a]))
        {
            Assert.Equal("cpu", h.Where);
            Assert.IsType<TernaryResidualHook>(dense.ResidualModifier);
        }
        Assert.Null(dense.ResidualModifier);
        using (TernaryAdapterAttach.Attach(moe, fx.MoeConfig, [a])) Assert.NotNull(moe.ResidualModifier);
        Assert.Null(moe.ResidualModifier);
        // geometry mismatch is an error too (layer 5 does not exist in the 2-layer model)
        Assert.Throws<InvalidOperationException>(() => TernaryAdapterAttach.Attach(dense, fx.DenseConfig, [Adapter(d, 5)]));
        Assert.Null(dense.ResidualModifier);
    }
}
