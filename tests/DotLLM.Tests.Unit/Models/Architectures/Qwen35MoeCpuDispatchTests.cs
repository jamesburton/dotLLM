using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression coverage for issue #135: the CLI / server CPU path routed <c>qwen35moe</c>
/// (Qwen3.6 Gated-DeltaNet + sparse-MoE hybrid) GGUFs to the plain
/// <see cref="TransformerModel"/> loader, which fails with
/// "The given key 'blk.0.attn_output.weight' was not present" — GDN layers have no attention
/// output projection. These tests run the SAME shared dispatch the CLI (<c>run</c>/<c>chat</c>),
/// the server, and <c>ModelLoader.LoadFromGguf</c> now all use
/// (<see cref="ModelLoader.CreateCpuModelFromGguf"/>) against a tiny synthetic
/// <c>qwen35moe</c> GGUF (<see cref="SyntheticQwen35MoeGguf"/>) whose layer 0 is GDN —
/// exactly the shape that discriminates broken from fixed dispatch: pre-fix these throw
/// <see cref="KeyNotFoundException"/>, post-fix they return a
/// <see cref="Qwen3MoeHybridTransformerModel"/>.
/// </summary>
public sealed class Qwen35MoeCpuDispatchTests : IDisposable
{
    private readonly string _scratch;

    public Qwen35MoeCpuDispatchTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-qwen35moe-dispatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string WriteFixture() =>
        SyntheticQwen35MoeGguf.Write(Path.Combine(_scratch, "qwen35moe-tiny.gguf"));

    [Fact]
    public void LoadFromGguf_Qwen35Moe_DispatchesToHybridModel()
    {
        string path = WriteFixture();

        var (model, gguf, config) = ModelLoader.LoadFromGguf(path);
        using var _g = gguf;
        using var _m = model;

        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
    }

    [Fact]
    public void CreateCpuModelFromGguf_Qwen35Moe_LoadsAndForwards()
    {
        string path = WriteFixture();

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);
        Assert.NotNull(config.HybridLayout);
        Assert.Equal(HybridLayerKind.GatedDeltaNet, config.HybridLayout!.LayerKind[0]);
        Assert.Equal(HybridLayerKind.Attention, config.HybridLayout.LayerKind[1]);

        // The same dispatch call the CLI run/chat commands and the server use for CPU.
        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        var hybrid = Assert.IsType<Qwen3MoeHybridTransformerModel>(model);

        // Forward smoke: the fixture must be architecturally coherent end-to-end.
        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];
        using var kvCache = new SimpleKvCache(
            hybrid.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);

        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(config.VocabSize, logits.Shape[1]);
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * config.VocabSize);
            foreach (float v in span)
                Assert.True(float.IsFinite(v), $"non-finite logit: {v}");
        }
    }

    /// <summary>
    /// Real-model smoke, env-gated: set <c>DOTLLM_QWEN36_35B_A3B_GGUF</c> to a local
    /// Qwen3.6-35B-A3B GGUF path to verify dispatch on the real file (mmap load only;
    /// no forward — the 35B forward is the CLI smoke's job).
    /// </summary>
    [SkippableFact]
    public void CreateCpuModelFromGguf_RealQwen36_35B_DispatchesToHybridModel()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN36_35B_A3B_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path),
            "DOTLLM_QWEN36_35B_A3B_GGUF not set / file missing — real-model dispatch smoke skipped.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3MoeHybrid, config.Architecture);

        using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.IsType<Qwen3MoeHybridTransformerModel>(model);
    }
}
