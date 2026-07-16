using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Regression tests for issue #146 — transient <c>VK_ERROR_MEMORY_MAP_FAILED</c> at model
/// load under memory pressure. On Windows/WDDM a <c>vkMapMemory</c> must make the entire
/// allocation host-resident, so mapping the GB-scale weight-staging buffer (sized
/// vocab×hidden×4 for the token-embed dequant) transiently failed ~2/15 heavy back-to-back
/// runs on the Strix Halo UMA box; a retry always succeeded. The fix routes every map (and
/// <c>vkAllocateMemory</c>) through a bounded retry-with-backoff
/// (<see cref="VulkanDevice.MapMemoryWithRetry"/>, <c>DOTLLM_VULKAN_MEM_RETRIES</c>).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMemoryPressureRetryTests
{
    private readonly ITestOutputHelper _output;
    public VulkanMemoryPressureRetryTests(ITestOutputHelper output) => _output = output;

    // ---- Pure policy tests (no GPU required) -------------------------------------------

    [Theory]
    [InlineData(-1, true)]   // VK_ERROR_OUT_OF_HOST_MEMORY
    [InlineData(-2, true)]   // VK_ERROR_OUT_OF_DEVICE_MEMORY
    [InlineData(-5, true)]   // VK_ERROR_MEMORY_MAP_FAILED
    [InlineData(-3, false)]  // VK_ERROR_INITIALIZATION_FAILED — API/driver, not pressure
    [InlineData(-4, false)]  // VK_ERROR_DEVICE_LOST — never retry a lost device
    [InlineData(-13, false)] // VK_ERROR_UNKNOWN
    [InlineData(0, false)]   // VK_SUCCESS
    public void IsTransientMemoryResult_ClassifiesPressureResultsOnly(int vkResult, bool expected)
        => Assert.Equal(expected, VulkanDevice.IsTransientMemoryResult(vkResult));

    [Fact]
    public void RetryBackoff_IsExponentialAndBounded()
    {
        Assert.Equal(25, VulkanDevice.RetryBackoffMs(0));
        Assert.Equal(100, VulkanDevice.RetryBackoffMs(1));
        Assert.Equal(400, VulkanDevice.RetryBackoffMs(2));
        Assert.Equal(1600, VulkanDevice.RetryBackoffMs(3));
        // Shift is clamped — no overflow / absurd sleeps for high attempt counts.
        Assert.Equal(25 << 12, VulkanDevice.RetryBackoffMs(100));
    }

    // ---- Device tests -------------------------------------------------------------------

    [SkippableFact]
    public void MapMemoryWithRetry_RoundTrips_OnHostVisibleBuffer()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        using var buf = device.Allocate(4096);
        nint mapped = device.MapMemoryWithRetry(buf.Memory, 0, 4096, "test map");
        try
        {
            unsafe
            {
                var span = new Span<byte>((void*)mapped, 4096);
                span.Fill(0xAB);
                Assert.Equal(0xAB, span[4095]);
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(device.Handle, buf.Memory);
        }
    }

    /// <summary>
    /// Env-gated stress variant: repeated full model load/dispose cycles — the #146
    /// reproduction shape (each load allocates + repeatedly maps the GB-scale staging
    /// buffer). Set <c>DOTLLM_VULKAN_STRESS_LOAD_CYCLES=N</c> to enable; run it while
    /// the box is under memory pressure (or back-to-back with other GPU work) to
    /// exercise the retry path. Any surviving transient failure fails the test.
    /// </summary>
    [SkippableFact]
    public void ModelLoadDisposeCycles_SurviveMemoryPressure()
    {
        string? cyclesEnv = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_STRESS_LOAD_CYCLES");
        Skip.If(string.IsNullOrEmpty(cyclesEnv), "Set DOTLLM_VULKAN_STRESS_LOAD_CYCLES=N to run the load/dispose stress.");
        int cycles = int.Parse(cyclesEnv!);

        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_STRESS_MODEL")
            ?? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Stress model not found at {modelPath}");

        for (int i = 0; i < cycles; i++)
        {
            using var gguf = GgufFile.Open(modelPath);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = VulkanTransformerModel.LoadFromGguf(gguf, config);
            using var kv = model.CreateKvCache(128);
            _output.WriteLine($"cycle {i}: OK");
        }
    }
}
