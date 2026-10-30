using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Engine.Samplers;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1434: the per-row-parameter Gumbel kernel must give, for EVERY row, exactly the CPU <see cref="GumbelNoise.Sample"/> with that row's own
/// seed, token index and 1/temperature (rows of one launch belong to different sequences).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGumbelArgMaxRowsKernelTests
{
    [SkippableTheory]
    [InlineData(151_936, 2)]
    [InlineData(151_936, 7)]
    [InlineData(248_320, 33)]
    public void EveryRow_MatchesTheCpuReference_WithItsOwnParameters(int vocab, int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxRowsKernel.IsAvailable(spvDir), "gumbel rows shader not compiled");
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxRowsKernel.Create(device, spvDir);
        using var logits = device.AllocateHostReadback((long)rows * vocab * sizeof(float));
        using var ids = device.AllocateHostReadback(rows * sizeof(uint));
        using var rowParams = device.Allocate((long)rows * GumbelArgMaxRowsKernel.RowParamBytes);
        var rng = new Random(1434);
        var host = new float[(long)rows * vocab];
        var idHost = new int[rows];
        float[] temps = [1.0f, 0.7f, 0.55f, 1.3f];
        for (int launch = 0; launch < 20; launch++)
        {
            var packed = new uint[rows * 8];
            var seeds = new ulong[rows]; var idx = new int[rows]; var invT = new float[rows];
            for (int r = 0; r < rows; r++)
            {
                var row = host.AsSpan(r * vocab, vocab);
                for (int i = 0; i < vocab; i++) row[i] = (float)(rng.NextDouble() * 6 - 3);
                for (int k = 0; k < 8; k++) row[rng.Next(vocab)] = 6f + (float)rng.NextDouble();
                for (int k = 0; k < vocab / 50; k++) row[rng.Next(vocab)] = float.NegativeInfinity;
                seeds[r] = (ulong)rng.NextInt64();
                idx[r] = rng.Next(0, 100_000);
                float t = temps[rng.Next(temps.Length)];
                invT[r] = t == 1.0f ? 1f : 1f / t;
                GumbelArgMaxRowsKernel.PackRow(packed.AsSpan(r * 8, 8), seeds[r], idx[r], invT[r]);
            }
            device.Upload(host, logits);
            device.Upload(MemoryMarshal.AsBytes(packed.AsSpan()), rowParams);
            kernel.Launch(logits, ids, rowParams, rows, vocab, vocab);
            device.Download(ids, MemoryMarshal.Cast<int, float>(idHost.AsSpan()));
            var seen = new HashSet<int>();
            for (int r = 0; r < rows; r++)
            {
                var prepared = new float[vocab];
                TensorPrimitives.Multiply(host.AsSpan(r * vocab, vocab), invT[r], prepared);
                int cpu = GumbelNoise.Sample(prepared, seeds[r], idx[r]);
                Assert.True(cpu == idHost[r], $"launch {launch} row {r}: cpu {cpu} gpu {idHost[r]}");
                seen.Add(idHost[r]);
            }
            Assert.True(seen.Count > 1, "all rows drew the same token: per-row parameters are not being applied");
        }
    }

    // #1435: top-k / min-p on the device must pick exactly what SamplerPipeline.PrepareDistribution + GumbelNoise.Sample pick.
    // quantStep > 0 rounds the logits to that grid, so many entries tie exactly at the k-th value (the index-order tie rule decides).
    [SkippableTheory]
    [InlineData(151_936, 0f, 40, 0f, 0.7f)]
    [InlineData(151_936, 0f, 0, 0.05f, 0.8f)]
    [InlineData(151_936, 0f, 20, 0.1f, 1.0f)]
    [InlineData(151_936, 0f, 1, 0f, 0.9f)]
    [InlineData(3_000, 0.25f, 50, 0f, 1.0f)]
    [InlineData(3_000, 0.25f, 7, 0.02f, 0.6f)]
    [InlineData(3_000, 1.0f, 300, 0f, 1.0f)]
    [InlineData(700, 0.5f, 100_000, 0.05f, 0.8f)]
    [InlineData(248_320, 0f, 64, 0.03f, 0.7f)]
    public void FilteredDraw_MatchesTheHostPipeline(int vocab, float quantStep, int topK, float minP, float temperature)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxRowsKernel.IsAvailable(spvDir), "gumbel rows shader not compiled");
        const int Rows = 6;
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxRowsKernel.Create(device, spvDir);
        using var logits = device.AllocateHostReadback((long)Rows * vocab * sizeof(float));
        using var ids = device.AllocateHostReadback(Rows * sizeof(uint));
        using var rowParams = device.Allocate((long)Rows * GumbelArgMaxRowsKernel.RowParamBytes);
        var rng = new Random(topK * 31 + vocab);
        var host = new float[(long)Rows * vocab];
        var idHost = new int[Rows];
        int distinctChoices = 0;
        for (int launch = 0; launch < 12; launch++)
        {
            var packed = new uint[Rows * 8];
            var expected = new int[Rows];
            for (int r = 0; r < Rows; r++)
            {
                var row = host.AsSpan(r * vocab, vocab);
                for (int i = 0; i < vocab; i++)
                {
                    float v = (float)(rng.NextDouble() * 6 - 3);
                    row[i] = quantStep > 0 ? MathF.Round(v / quantStep) * quantStep : v;
                }
                for (int k = 0; k < 8; k++) row[rng.Next(vocab)] = 6f + (float)rng.NextDouble();
                if (quantStep == 0) for (int k = 0; k < vocab / 100; k++) row[rng.Next(vocab)] = float.NegativeInfinity;
                var opts = new DotLLM.Core.Configuration.InferenceOptions
                {
                    Temperature = temperature, TopK = topK, MinP = minP, Seed = 1000 * launch + r, GumbelSampling = true,
                };
                var pipeline = new SamplerPipeline(opts);
                var spec = pipeline.DeviceGumbelSpec ?? throw new InvalidOperationException("pipeline not device eligible");
                int tokenIndex = rng.Next(0, 5000);
                GumbelArgMaxRowsKernel.PackRow(packed.AsSpan(r * 8, 8), pipeline.GumbelSeed, tokenIndex, spec.InvTemperature, spec.Filter.TopK, spec.Filter.MinP);
                var prepared = row.ToArray();
                pipeline.PrepareDistribution(prepared, new List<int>());
                expected[r] = GumbelNoise.Sample(prepared, pipeline.GumbelSeed, tokenIndex);
            }
            device.Upload(host, logits);
            device.Upload(MemoryMarshal.AsBytes(packed.AsSpan()), rowParams);
            kernel.Launch(logits, ids, rowParams, Rows, vocab, vocab);
            device.Download(ids, MemoryMarshal.Cast<int, float>(idHost.AsSpan()));
            for (int r = 0; r < Rows; r++)
            {
                Assert.True(expected[r] == idHost[r], $"launch {launch} row {r}: host {expected[r]} device {idHost[r]}");
                distinctChoices++;
            }
        }
        Assert.Equal(12 * Rows, distinctChoices);
    }

    // The tie rule specifically: constant row, k < vocab. TopKSampler keeps the FIRST k indices; with a flat row Gumbel then picks among them.
    [SkippableFact]
    public void FilteredDraw_AllTied_KeepsOnlyTheFirstKIndices()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxRowsKernel.IsAvailable(spvDir), "gumbel rows shader not compiled");
        const int V = 5000, K = 17, Launches = 200;
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxRowsKernel.Create(device, spvDir);
        using var logits = device.AllocateHostReadback((long)V * sizeof(float));
        using var ids = device.AllocateHostReadback(sizeof(uint));
        using var rowParams = device.Allocate(GumbelArgMaxRowsKernel.RowParamBytes);
        var host = new float[V];
        Array.Fill(host, 0.5f);
        device.Upload(host, logits);
        var idHost = new int[1];
        for (int t = 0; t < Launches; t++)
        {
            var packed = new uint[8];
            GumbelArgMaxRowsKernel.PackRow(packed, seed: 7, tokenIndex: t, invTemperature: 1f, topK: K);
            device.Upload(MemoryMarshal.AsBytes(packed.AsSpan()), rowParams);
            kernel.Launch(logits, ids, rowParams, 1, V, V);
            device.Download(ids, MemoryMarshal.Cast<int, float>(idHost.AsSpan()));
            Assert.InRange(idHost[0], 0, K - 1);
        }
    }
}
