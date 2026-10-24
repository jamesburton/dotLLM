using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cli.Helpers;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

public sealed class CudaLfm2KvPolicyTests
{
    [Theory]
    [InlineData(KvCacheDType.Q8_0, KvCacheDType.F32, 0)]
    [InlineData(KvCacheDType.F32, KvCacheDType.Q4_0, 0)]
    [InlineData(KvCacheDType.F32, KvCacheDType.F32, 128)]
    public void UnsupportedKvFlagsRejectedBeforeDeviceAccess(KvCacheDType key, KvCacheDType value, int window)
        => Assert.Throws<NotSupportedException>(() => CudaLfm2MoeTransformerModel.ValidateKvStorage(key, value, window));

    [Fact]
    public void CliClassifiesOwnedCacheAndRejectsQuantization()
    {
        var model = (CudaLfm2MoeTransformerModel)RuntimeHelpers.GetUninitializedObject(typeof(CudaLfm2MoeTransformerModel));
        Assert.Equal(KvFlagTarget.OwnCacheUnsupported, KvStorageCli.Classify(model, false));
        Assert.True(KvStorageCli.TryBuild("q8_0", null, null, null, 0, out var request, out _));
        Assert.Throws<NotSupportedException>(() => KvStorageCli.Apply(model, request));
        CudaLfm2MoeTransformerModel.ValidateKvStorage(KvCacheDType.F32, KvCacheDType.F32);
    }
}

[Collection(CudaCollection.Name)]
[Trait("Category", "GPU")]
public sealed class CudaLfm2ServerDispatchTests
{
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void ServerStartupUsesLfmOwnedCacheAndRunsForward(bool paged)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "CUDA unavailable");
        string path = Path.Combine(Path.GetTempPath(), "dotllm-lfm-server-" + Guid.NewGuid().ToString("N") + ".gguf");
        File.WriteAllBytes(path, CudaLfm2ParityTests.Build());
        try
        {
            using var state = ServerStartup.LoadModel(path, new ServerOptions
            {
                Model = path, ModelId = "tiny-cuda-lfm", Device = "gpu", UsePaged = paged,
                Warmup = WarmupOptions.Disabled, PromptCacheEnabled = false
            });
            Assert.IsType<CudaLfm2MoeTransformerModel>(state.Model);
            Assert.Null(state.PagedFactory); Assert.Null(state.CudaPagedFactory);
            using var kv = state.KvCacheFactory!(state.Config!, 16);
            Assert.IsType<CudaHybridKvCacheHandle>(kv);
            using var logits = state.Model!.Forward([1, 4], [0, 1], -1, kv);
            Assert.Equal(48, logits.Shape[1]); Assert.Equal(2, kv.CurrentLength);
        }
        finally { File.Delete(path); }
    }
}
