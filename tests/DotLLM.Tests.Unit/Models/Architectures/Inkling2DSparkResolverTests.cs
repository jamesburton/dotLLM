using DotLLM.Models.Architectures.DSpark;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Sibling auto-detection of the Inkling 2 DSpark drafter (issue #2039); no model needed.</summary>
public sealed class Inkling2DSparkResolverTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dspark-resolver-" + Guid.NewGuid().ToString("N"));
    public Inkling2DSparkResolverTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch (IOException) { } }

    [Fact]
    public void Find_PicksTheLargestDsparkSibling_InTheModelDirectoryOrTheOneAbove()
    {
        string sub = Path.Combine(_dir, "quant"); Directory.CreateDirectory(sub);
        string model = Path.Combine(sub, "Inkling-2-Mini-NVFP4.gguf");
        File.WriteAllBytes(model, [1]);
        Assert.Null(Inkling2DSparkResolver.Find(model));
        File.WriteAllBytes(Path.Combine(_dir, "dspark-small.gguf"), new byte[10]);
        File.WriteAllBytes(Path.Combine(_dir, "dspark-big.gguf"), new byte[100]);
        File.WriteAllBytes(Path.Combine(_dir, "mtp-other.gguf"), new byte[1000]);   // another family's head is not a DSpark drafter
        Assert.Equal(Path.Combine(_dir, "dspark-big.gguf"), Inkling2DSparkResolver.Find(model));
        File.WriteAllBytes(Path.Combine(sub, "dspark-near.gguf"), new byte[5]);        // the model's own directory wins over the one above
        Assert.Equal(Path.Combine(sub, "dspark-near.gguf"), Inkling2DSparkResolver.Find(model));
    }

    [Fact]
    public void AGarbageFile_IsNotADsparkDrafter()
    {
        string f = Path.Combine(_dir, "dspark-garbage.gguf");
        File.WriteAllBytes(f, [0, 1, 2, 3, 4, 5, 6, 7]);
        Assert.False(Inkling2DSparkDrafter.IsDSparkFile(f));
        Assert.False(Inkling2DSparkDrafter.IsDSparkFile(Path.Combine(_dir, "missing.gguf")));
    }

    /// <summary>
    /// Review of #2057: an explicit <c>--dspark</c> on a model that cannot take it (any model but the CPU Inkling 2 one, e.g. the Vulkan Inkling 2 model
    /// under <c>--device auto</c>) refuses to start instead of being ignored (#2017's explicit-path rule); without the flag that model is left alone.
    /// Kills: serve's <c>TryAttachSeparateMtpHead</c> consulting <c>DSparkPath</c> only for the CPU Inkling 2 model (the explicit path was then dropped silently).
    /// </summary>
    [Fact]
    public void ExplicitPath_OnAModelThatCannotTakeIt_RefusesToStart()
    {
        using var model = new PlainModel();
        string draft = Path.Combine(_dir, "dspark-x.gguf");
        File.WriteAllBytes(draft, [1]);
        Assert.Throws<NotSupportedException>(() => Inkling2DSparkResolver.TryAttach(model, _dir, draft));
        Assert.Null(Inkling2DSparkResolver.TryAttach(model, _dir));   // auto-detect: other models are left alone
        Assert.Throws<NotSupportedException>(() => DotLLM.Server.ServerStartup.TryAttachSeparateMtpHead(model, Path.Combine(_dir, "m.gguf"),
            new DotLLM.Server.ServerOptions { Model = "x", DSparkPath = draft }));
        DotLLM.Server.ServerStartup.TryAttachSeparateMtpHead(model, Path.Combine(_dir, "m.gguf"), new DotLLM.Server.ServerOptions { Model = "x" });   // no flag: no-op
    }

    private sealed class PlainModel : DotLLM.Core.Models.IModel
    {
        public DotLLM.Core.Models.ModelConfig Config { get; } = new()
        {
            VocabSize = 8, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 4, IntermediateSize = 8, HeadDim = 4, MaxSequenceLength = 16,
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };
        public long ComputeMemoryBytes => 0;
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => throw new NotSupportedException();
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, DotLLM.Core.Attention.IKvCache? kv) => throw new NotSupportedException();
        public void Dispose() { }
    }

    /// <summary>
    /// #2046: the resolver, the CLI and the server key on <see cref="IInkling2DSparkHost"/>, not on the CPU model class, so the Vulkan Inkling 2 model (or any backend implementing the
    /// marker) takes an auto-detected sibling and an explicit path. Kills: a resolver that type-checks <c>Inkling2TransformerModel</c> (the Vulkan model was then refused or skipped).
    /// </summary>
    [Fact]
    public void AnyHostImplementingTheMarker_TakesTheDrafter_AutoAndExplicit()
    {
        string model = Path.Combine(_dir, "Inkling-2-Mini-NVFP4.gguf");
        File.WriteAllBytes(model, [1]);
        string draft = Path.Combine(_dir, "dspark-x.gguf");
        File.WriteAllBytes(draft, [1]);
        using var auto = new FakeHost();
        Assert.Equal(draft, Inkling2DSparkResolver.TryAttach(auto, model));
        Assert.Equal(draft, auto.Attached);
        using var explicitHost = new FakeHost();
        Assert.Equal(draft, Inkling2DSparkResolver.TryAttach(explicitHost, model, draft));
        using var already = new FakeHost { Attached = "x" };
        Assert.Null(Inkling2DSparkResolver.TryAttach(already, model));                                               // auto: an attached drafter is left alone
        Assert.Throws<NotSupportedException>(() => Inkling2DSparkResolver.TryAttach(already, model, draft));         // explicit: refuses
        using var viaServer = new FakeHost();
        DotLLM.Server.ServerStartup.TryAttachSeparateMtpHead(viaServer, model, new DotLLM.Server.ServerOptions { Model = "x", DSparkPath = draft });
        Assert.Equal(draft, viaServer.Attached);
    }

    private sealed class FakeHost : DotLLM.Core.Models.IModel, IInkling2DSparkHost
    {
        public string? Attached { get; set; }
        public bool HasDFlashDrafter => Attached is not null;
        public void AttachDFlashDrafter(string path) => Attached = path;
        public DotLLM.Core.Models.ModelConfig Config { get; } = new()
        {
            VocabSize = 8, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 4, IntermediateSize = 8, HeadDim = 4, MaxSequenceLength = 16,
            Architecture = DotLLM.Core.Configuration.Architecture.Inkling2,
        };
        public long ComputeMemoryBytes => 0;
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => throw new NotSupportedException();
        public DotLLM.Core.Tensors.ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, DotLLM.Core.Attention.IKvCache? kv) => throw new NotSupportedException();
        public void Dispose() { }
    }
}
