using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads Microsoft BitNet b1.58 2B4T (I2_S ternary GGUF, ~1.2 GB) for BitNet integration tests.
/// Exercises the ternary-weight (I2_S) path, the squared-ReLU FFN and the attention/FFN Sub-LN
/// end-to-end. Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class BitNetModelFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync(
            "microsoft/bitnet-b1.58-2B-4T-gguf", "ggml-model-i2_s.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("BitNetModel")]
public class BitNetModelCollection : ICollectionFixture<BitNetModelFixture>;
