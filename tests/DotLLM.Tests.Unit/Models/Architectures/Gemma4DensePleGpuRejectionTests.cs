using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #730: the dense Gemma-4 variant (E2B/E4B: PLE + shared KV, no MoE) is CPU-only.
/// CUDA used to fail mid-load with "LoadLayer called without Moe config"; both GPU backends now
/// reject it up front (before any device work), so these run without a GPU.
/// </summary>
public sealed class Gemma4DensePleGpuRejectionTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-g4rej-{Guid.NewGuid():N}");

    public Gemma4DensePleGpuRejectionTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (GgufFile Gguf, ModelConfig Config) Open(SyntheticGemma4Config? cfg)
    {
        string path = Path.Combine(_scratch, $"{Guid.NewGuid():N}.gguf");
        SyntheticGemma4Gguf.WriteGemma4(path, cfg);
        var gguf = GgufFile.Open(path);
        return (gguf, GgufModelConfigExtractor.Extract(gguf.Metadata));
    }

    [Fact]
    public void Predicate_TrueForDensePle_FalseForMoe()
    {
        var (g1, dense) = Open(SyntheticGemma4Gguf.E4BLike);
        using (g1) Assert.True(dense.IsGemma4DensePle);

        var (g2, moe) = Open(null);
        using (g2)
        {
            Assert.NotNull(moe.Moe);
            Assert.False(moe.IsGemma4DensePle);
        }
    }

    [Fact]
    public void Cuda_RejectsDensePle_EarlyWithActionableMessage()
    {
        var (gguf, cfg) = Open(SyntheticGemma4Gguf.E4BLike);
        using (gguf)
        {
            var ex = Assert.Throws<NotSupportedException>(() => CudaTransformerModel.LoadFromGguf(gguf, cfg));
            Assert.Contains("CUDA", ex.Message);
            Assert.Contains("--device cpu", ex.Message);
        }
    }

    [Fact]
    public void Message_LinksPriorityIssue()
    {
        string msg = ModelConfig.Gemma4DensePleUnsupportedMessage("CUDA");
        Assert.Contains("issues/734", msg);
    }
}
