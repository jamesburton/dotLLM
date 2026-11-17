using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issues #1836 / #1841 (absorbed by #1848): the dense CUDA host serves no more context than its paged pool holds, and every
/// endpoint a client sizes its window from reports the served window, not the checkpoint's.
/// </summary>
public sealed class CudaServedContextTests
{
    [Theory]
    // No --ctx-size: the model's window, capped to the pool (TwIL-8B on a 12 GB 3060: ~17K of 131K).
    [InlineData(null, 17_408, 131_072, 17_408, false)]
    [InlineData(null, 17_408, 8_192, 8_192, false)]
    // --ctx-size inside the pool is honoured.
    [InlineData(8_192, 17_408, 131_072, 8_192, false)]
    // --ctx-size past the pool: capped, with a warning (a request past the pool cannot run).
    [InlineData(65_536, 17_408, 131_072, 17_408, true)]
    // --ctx-size past the model's window: the window, which still fits the pool.
    [InlineData(200_000, 262_144, 131_072, 131_072, false)]
    public void PagedServedContext_IsCappedToThePool(int? requested, int pool, int native, int expected, bool warns)
    {
        Assert.Equal(expected, ServerStartup.ChoosePagedServedContext(requested, pool, native, out string? warning));
        Assert.Equal(warns, warning is not null);
    }

    private sealed class StubModel : IModel
    {
        public ModelConfig Config { get; init; } = new()
        {
            Architecture = Architecture.Llama, VocabSize = 8, HiddenSize = 8, NumLayers = 1, IntermediateSize = 8,
            NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 8, MaxSequenceLength = 131_072,
        };
        public long ComputeMemoryBytes => 0;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    [Fact]
    public void Status_Health_AndModelsMeta_ReportTheServedWindow()
    {
        // The model still advertises its native 131,072 window; the served config (after the fit) is 17,408. Every client-facing
        // figure must be the served one.
        var model = new StubModel();
        var served = model.Config with { MaxSequenceLength = 17_408 };
        using var state = new ServerState
        {
            Options = new ServerOptions { Model = "", ModelId = "twil", UsePaged = true, PromptCacheEnabled = false },
            Model = model,
            Config = served,
            IsReady = true,
        };

        Assert.Equal(17_408, state.GetStatus().NCtx);
        Assert.Equal(17_408, HealthEndpoint.BuildHealth(state).MaxContext);
        Assert.Equal(17_408, ModelMeta.Build(state.Config, gguf: null)!.NCtx);
    }

    [Fact]
    public void Status_ReportsNoWindow_WithoutAModel()
    {
        using var state = new ServerState { Options = new ServerOptions { Model = "", ModelId = "none" } };
        Assert.Null(state.GetStatus().NCtx);
    }

    [Theory]
    // #1862: no flag on the dense CUDA non-paged f16 path -> the default chunk.
    [InlineData(null, true, ServerStartup.DefaultCudaDensePrefillChunkTokens)]
    // No flag anywhere else (CPU, Vulkan, HIP, CUDA paged or quantized KV) -> one pass, unchanged.
    [InlineData(null, false, 0)]
    // An explicit value always wins, on either path, and 0 is an explicit opt-out back to one pass.
    [InlineData(512, true, 512)]
    [InlineData(4096, true, 4096)]
    [InlineData(0, true, 0)]
    [InlineData(256, false, 256)]
    [InlineData(-5, true, 0)]
    public void PrefillChunk_DefaultsOnDenseCudaOnly_AndAnExplicitValueWins(int? requested, bool cudaDense, int expected)
        => Assert.Equal(expected, ServerStartup.ResolvePrefillChunk(requested, cudaDense));

    [Fact]
    public void PrefillChunkDefault_IsBoundedByTheRetainedG3Plane()
    {
        // The default's first chunk must keep G3 inside the plane the fit always charges, for a 32-head model like TwIL-8B (#1862).
        Assert.True(DotLLM.Cuda.CudaG3Attention.SquareScratchElems(ServerStartup.DefaultCudaDensePrefillChunkTokens, 32)
                    <= DotLLM.Cuda.CudaG3Attention.RetainedMaxElems);
    }

    [Fact]
    public void PrefillChunkDefault_AppliesOnlyToTheDenseCudaHostsNonPagedF16Cache()
    {
        // No driver is needed: only the model's type is inspected, so an uninitialised instance stands in for a loaded one.
        var cuda = (IModel)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(DotLLM.Cuda.CudaTransformerModel));
        var f16 = new KvCacheConfig(KvCacheDType.F32, KvCacheDType.F32);
        var q8 = new KvCacheConfig(KvCacheDType.Q8_0, KvCacheDType.Q8_0);
        Assert.True(ServerStartup.IsCudaDenseF16NonPaged(cuda, cudaPagedPool: false, f16));
        Assert.False(ServerStartup.IsCudaDenseF16NonPaged(cuda, cudaPagedPool: true, f16));
        Assert.False(ServerStartup.IsCudaDenseF16NonPaged(cuda, cudaPagedPool: false, q8));
        // Any other model (CPU, Vulkan, HIP) keeps one-pass prefill, whatever its cache.
        Assert.False(ServerStartup.IsCudaDenseF16NonPaged(new StubModel(), cudaPagedPool: false, f16));
    }

    [Fact]
    public void ServeOptions_WithoutTheFlag_LeaveTheChunkUnset_SoTheBackendDefaultApplies()
    {
        Assert.Null(ServerOptions.Parse(new[] { "--model", "m.gguf" }).PrefillChunkSize);
        Assert.Equal(0, ServerOptions.Parse(new[] { "--model", "m.gguf", "--prefill-chunk-size", "0" }).PrefillChunkSize);
    }
}
