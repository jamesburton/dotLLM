using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Downloads SmolLM-135M Q5_0 GGUF (~100 MB). Q5_0 is a second, structurally different
/// quantization for tests that must not be single-format (e.g. the #530 chunk-invariance sweep).
/// Cached in <c>~/.dotllm/test-cache/</c> across test runs.
/// </summary>
public sealed class Q5_0ModelFixture : IAsyncLifetime
{
    /// <summary>Full local path to the downloaded GGUF file.</summary>
    public string FilePath { get; private set; } = string.Empty;

    public async Task InitializeAsync() =>
        FilePath = await TestModelDownloader.EnsureModelAsync("QuantFactory/SmolLM-135M-GGUF", "SmolLM-135M.Q5_0.gguf");

    public Task DisposeAsync() => Task.CompletedTask;
}

[CollectionDefinition("Q5_0Model")]
public class Q5_0ModelCollection : ICollectionFixture<Q5_0ModelFixture>;
