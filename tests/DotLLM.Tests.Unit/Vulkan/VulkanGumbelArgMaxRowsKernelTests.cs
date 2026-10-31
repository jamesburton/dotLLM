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
            var packed = new uint[rows * 4];
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
                GumbelArgMaxRowsKernel.PackRow(packed.AsSpan(r * 4, 4), seeds[r], idx[r], invT[r]);
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
}
