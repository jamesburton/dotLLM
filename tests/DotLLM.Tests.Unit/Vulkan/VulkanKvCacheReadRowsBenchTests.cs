using System.Diagnostics;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1551: <see cref="VulkanKvCache.ReadRows"/> / <see cref="VulkanKvCache.WriteRows"/> throughput. The correctness test runs
/// always (round trip, F32 and F16); the timing is opt-in (<c>DOTLLM_KVIO_BENCH=1</c>) and prints MB/s for one 512-row chunk of a
/// Qwen3-4B-shaped cache (stride 1024, K and V).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanKvCacheReadRowsBenchTests
{
    private readonly ITestOutputHelper _out;

    /// <summary>Initializes the test with the xUnit output sink.</summary>
    /// <param name="output">Sink.</param>
    public VulkanKvCacheReadRowsBenchTests(ITestOutputHelper output) => _out = output;

    /// <summary>Write then read 512 rows at an offset: exact for F32, half-rounded for F16, K and V not swapped.</summary>
    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void ReadRows_RoundTripsLargeChunk(VulkanKvStorage storage)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        const int stride = 1024, rows = 512, layers = 2;
        using var cache = new VulkanKvCache(dev, new[] { stride, stride }, rows + 64, storage);
        var rng = new Random(1544);
        var k = new float[rows * stride]; var v = new float[rows * stride];
        for (int i = 0; i < k.Length; i++) { k[i] = (float)(rng.NextDouble() - 0.5); v[i] = (float)(rng.NextDouble() + 3.0); }
        for (int l = 0; l < layers; l++) cache.WriteRows(l, 64, rows, k, v);
        var rk = new float[k.Length]; var rv = new float[v.Length];
        for (int l = 0; l < layers; l++)
        {
            cache.ReadRows(l, 64, rows, rk, rv);
            for (int i = 0; i < k.Length; i++)
            {
                float ek = storage == VulkanKvStorage.F16 ? (float)(Half)k[i] : k[i];
                float ev = storage == VulkanKvStorage.F16 ? (float)(Half)v[i] : v[i];
                if (rk[i] != ek || rv[i] != ev) Assert.Fail($"layer {l} element {i}: got ({rk[i]}, {rv[i]}) expected ({ek}, {ev})");
            }
        }
    }

    /// <summary>Opt-in timing: <c>DOTLLM_KVIO_BENCH=1</c>. Hold the GPU locks first.</summary>
    [SkippableFact]
    public void Bench_ReadRows_WriteRows()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_KVIO_BENCH") == "1", "DOTLLM_KVIO_BENCH=1 to enable.");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device");
        using var dev = VulkanDevice.Create();
        const int stride = 1024, rows = 512, layers = 36;
        using var cache = new VulkanKvCache(dev, Enumerable.Repeat(stride, layers).ToArray(), rows, VulkanKvStorage.F32);
        var k = new float[rows * stride]; var v = new float[rows * stride];
        new Random(1).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(k.AsSpan()));
        for (int i = 0; i < k.Length; i++) { if (!float.IsFinite(k[i])) k[i] = 0.5f; v[i] = k[i]; }
        double mib = layers * 2.0 * rows * stride * 4 / 1048576.0;
        for (int rep = 0; rep < 5; rep++)
        {
            long t0 = Stopwatch.GetTimestamp();
            for (int l = 0; l < layers; l++) cache.WriteRows(l, 0, rows, k, v);
            double w = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            t0 = Stopwatch.GetTimestamp();
            for (int l = 0; l < layers; l++) cache.ReadRows(l, 0, rows, k, v);
            double r = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
            _out.WriteLine($"rep {rep}: {layers} layers x K+V x {rows} rows = {mib:F0} MiB: WriteRows {w:F0} ms ({mib * 1000 / w:F0} MiB/s), ReadRows {r:F0} ms ({mib * 1000 / r:F0} MiB/s)");
        }
    }
}
