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
}
