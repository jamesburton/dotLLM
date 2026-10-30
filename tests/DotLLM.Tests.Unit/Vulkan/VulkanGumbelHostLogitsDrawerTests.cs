using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Samplers;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>#1435: the generic host-logits drawer (hosts with no in-place LM-head hook) draws exactly what the CPU draws for each tensor's row.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGumbelHostLogitsDrawerTests
{
    [SkippableTheory]
    [InlineData(0, 0f)]
    [InlineData(40, 0f)]
    [InlineData(0, 0.05f)]
    [InlineData(25, 0.1f)]
    public unsafe void DrawBatch_MatchesTheCpuReference_ForEveryTensor(int topK, float minP)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelHostLogitsDrawer.IsAvailable(spvDir), "gumbel rows shader not compiled");
        const int Vocab = 32_000, N = 70;   // > MaxRows: exercises the chunking
        using var device = VulkanDevice.Create();
        using var drawer = new GumbelHostLogitsDrawer(device, spvDir);
        var rng = new Random(topK * 7 + 1435);
        var tensors = new List<ITensor>();
        var draws = new GumbelDraw[N];
        var expected = new int[N];
        try
        {
            for (int r = 0; r < N; r++)
            {
                var t = UnmanagedTensor.Allocate(new TensorShape(1, Vocab), DType.Float32, deviceId: -1);
                var row = new Span<float>((void*)t.DataPointer, Vocab);
                for (int i = 0; i < Vocab; i++) row[i] = (float)(rng.NextDouble() * 6 - 3);
                for (int k = 0; k < 6; k++) row[rng.Next(Vocab)] = 6f + (float)rng.NextDouble();
                tensors.Add(t);
                var pipeline = new SamplerPipeline(new DotLLM.Core.Configuration.InferenceOptions
                {
                    Temperature = 0.8f, TopK = topK, MinP = minP, Seed = 50 + r, GumbelSampling = true,
                });
                var spec = pipeline.DeviceGumbelSpec!.Value;
                int idx = rng.Next(0, 4000);
                draws[r] = new GumbelDraw(pipeline.GumbelSeed, idx, spec.InvTemperature, spec.Filter.TopK, spec.Filter.MinP);
                var prepared = row.ToArray();
                pipeline.PrepareDistribution(prepared, new List<int>());
                expected[r] = GumbelNoise.Sample(prepared, pipeline.GumbelSeed, idx);
            }
            var got = new int[N];
            drawer.DrawBatch(tensors, Vocab, draws, got);
            Assert.Equal(expected, got);
            Assert.True(drawer.Draws >= 2);
        }
        finally
        {
            foreach (var t in tensors) t.Dispose();
        }
    }
}
