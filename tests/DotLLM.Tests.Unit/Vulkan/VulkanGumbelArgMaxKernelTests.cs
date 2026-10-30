using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Engine.Samplers;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1382: the device Gumbel-max sampler must be the SAME function as the CPU reference (<see cref="GumbelNoise"/>), bit for bit, because
/// a speculative verify row (device) and a plain decode step (host) have to pick the same token from the same logits. Proven three ways:
/// (1) the noise over the whole 2^24 uniform domain and over Philox counters, (2) the sampled id on real-vocabulary rows against
/// <see cref="GumbelNoise.Sample"/>, (3) a chi-square test of the device draw against softmax.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGumbelArgMaxKernelTests
{
    [Fact]
    public void CpuLog_IsAccurate_AndExactAtTheEdges()
    {
        // Relative error of the deterministic log against the double-precision log over the F32 domain the sampler uses.
        double worst = 0;
        var rng = new Random(3);
        for (int i = 0; i < 2_000_000; i++)
        {
            float x = i < 1000 ? 1f + (i - 500) * 6e-8f : (float)Math.Exp((rng.NextDouble() - 0.5) * 40);
            if (x <= 0 || !float.IsNormal(x) || x == 1f) continue;
            double rel = Math.Abs(GumbelNoise.Log(x) - Math.Log(x)) / Math.Abs(Math.Log(x));
            worst = Math.Max(worst, rel);
        }
        Assert.True(worst < 4e-7, $"worst relative error {worst:E2}");
        Assert.Equal(0f, GumbelNoise.Log(1f));
        Assert.Equal(float.NegativeInfinity, GumbelNoise.Log(0f));
        Assert.True(float.IsPositiveInfinity(GumbelNoise.FromUniform(1f)));   // the 32-bit mutant's u = 1 still maps to +inf
    }

    [SkippableFact]
    public void Noise_OverTheWholeUniformDomain_IsBitExactWithTheCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxKernel.IsAvailable(spvDir), "gumbel shaders not compiled");
        using var device = VulkanDevice.Create();
        using var dump = new GumbelNoiseDumpKernel(device, spvDir);
        const int Chunk = 1 << 22;
        using var buf = device.Allocate((long)Chunk * sizeof(float));
        var gpu = new float[Chunk];
        long mismatches = 0;
        for (int first = 0; first < (1 << 24); first += Chunk)
        {
            dump.Launch(buf, Chunk, first, tokenIndex: 0, seed: 0, uniformSweep: true);
            device.Download(buf, gpu);
            for (int i = 0; i < Chunk; i++)
            {
                uint word = (uint)(first + i) << 8;                       // Uniform24(word) = max(m, 1) * 2^-24 for m = first + i
                float cpu = GumbelNoise.FromUniform(GumbelNoise.Uniform24(word));
                if (BitConverter.SingleToInt32Bits(cpu) != BitConverter.SingleToInt32Bits(gpu[i]))
                {
                    if (mismatches++ < 5) Assert.Fail($"uniform mantissa {first + i}: cpu {cpu:G9} ({BitConverter.SingleToInt32Bits(cpu):X8}) gpu {gpu[i]:G9} ({BitConverter.SingleToInt32Bits(gpu[i]):X8})");
                }
            }
        }
        Assert.Equal(0, mismatches);
    }

    [SkippableTheory]
    [InlineData(0UL, 0, 0)]
    [InlineData(0x123456789ABCDEF0UL, 1234567, 200_000)]
    [InlineData(ulong.MaxValue, int.MaxValue, 151_000)]
    public void Noise_OverPhiloxCounters_IsBitExactWithTheCpu(ulong seed, int tokenIndex, int firstIndex)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxKernel.IsAvailable(spvDir), "gumbel shaders not compiled");
        using var device = VulkanDevice.Create();
        using var dump = new GumbelNoiseDumpKernel(device, spvDir);
        const int Count = 1 << 20;
        using var buf = device.Allocate((long)Count * sizeof(float));
        var gpu = new float[Count];
        dump.Launch(buf, Count, firstIndex, tokenIndex, seed);
        device.Download(buf, gpu);
        for (int i = 0; i < Count; i++)
        {
            float cpu = GumbelNoise.Noise(seed, tokenIndex, firstIndex + i);
            Assert.True(BitConverter.SingleToInt32Bits(cpu) == BitConverter.SingleToInt32Bits(gpu[i]),
                $"seed {seed:X} n {tokenIndex} i {firstIndex + i}: cpu {cpu:G9} gpu {gpu[i]:G9}");
        }
    }

    [SkippableTheory]
    [InlineData(151_936, 1.0f)]
    [InlineData(151_936, 0.7f)]
    [InlineData(248_320, 0.55f)]
    public void Sample_MatchesTheCpuReference_OnRealVocabularyRows(int vocab, float temperature)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxKernel.IsAvailable(spvDir), "gumbel shaders not compiled");
        const int Rows = 8, Launches = 40;
        float invT = 1f / temperature;
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxKernel.Create(device, spvDir);
        using var logits = device.AllocateHostReadback((long)Rows * vocab * sizeof(float));
        using var ids = device.AllocateHostReadback(Rows * sizeof(uint));
        var rng = new Random(1382);
        var host = new float[Rows * vocab];
        var idHost = new int[Rows];
        int agree = 0, total = 0;
        for (int launch = 0; launch < Launches; launch++)
        {
            for (int r = 0; r < Rows; r++)
            {
                var row = host.AsSpan(r * vocab, vocab);
                // a peaked head plus a long flat tail, some -inf (as a mask would leave), so Gumbel noise decides between near-ties
                for (int i = 0; i < vocab; i++) row[i] = (float)(rng.NextDouble() * 6 - 3);
                for (int k = 0; k < 8; k++) row[rng.Next(vocab)] = 6f + (float)rng.NextDouble();
                for (int k = 0; k < vocab / 50; k++) row[rng.Next(vocab)] = float.NegativeInfinity;
            }
            device.Upload(host, logits);
            ulong seed = (ulong)rng.NextInt64();
            int first = rng.Next(0, 100_000);
            kernel.Launch(logits, ids, Rows, vocab, vocab, seed, first, invT);
            device.Download(ids, MemoryMarshal.Cast<int, float>(idHost.AsSpan()));
            for (int r = 0; r < Rows; r++)
            {
                var prepared = new float[vocab];
                TensorPrimitives.Multiply(host.AsSpan(r * vocab, vocab), invT, prepared);
                int cpu = GumbelNoise.Sample(prepared, seed, first + r);
                total++;
                if (cpu == idHost[r]) agree++;
                else Assert.Fail($"launch {launch} row {r}: cpu {cpu} gpu {idHost[r]}");
            }
        }
        Assert.Equal(total, agree);
    }

    [SkippableFact]
    public void Sample_AllMasked_FallsBackToIndexZero_LikeTheCpuArgMax()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxKernel.IsAvailable(spvDir), "gumbel shaders not compiled");
        const int V = 700;
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxKernel.Create(device, spvDir);
        using var logits = device.Allocate(2L * V * sizeof(float));
        using var ids = device.Allocate(2 * sizeof(uint));
        var host = new float[2 * V];
        Array.Fill(host, float.NegativeInfinity);
        host[V + 5] = 0f;                       // row 1: a single finite entry always wins
        device.Upload(host, logits);
        kernel.Launch(logits, ids, 2, V, V, seed: 9, firstTokenIndex: 3, invTemperature: 1f);
        var got = new int[2];
        device.Download(ids, MemoryMarshal.Cast<int, float>(got.AsSpan()));
        Assert.Equal(GumbelNoise.Sample(host.AsSpan(0, V), 9, 3), got[0]);
        Assert.Equal(5, got[1]);
    }

    [SkippableFact]
    public void DeviceDraw_MatchesSoftmax_ChiSquare()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(GumbelArgMaxKernel.IsAvailable(spvDir), "gumbel shaders not compiled");
        const int V = 16, Rows = 4096, Launches = 15;   // 61,440 independent draws (one token index each)
        float temperature = 0.8f;
        float invT = 1f / temperature;
        using var device = VulkanDevice.Create();
        using var kernel = GumbelArgMaxKernel.Create(device, spvDir);
        using var logits = device.Allocate((long)Rows * V * sizeof(float));
        using var ids = device.Allocate(Rows * sizeof(uint));
        var raw = new float[V];
        for (int i = 0; i < V; i++) raw[i] = 2.5f * MathF.Sin(i * 1.7f + 0.3f);
        raw[3] = float.NegativeInfinity;        // a masked token must never win
        var host = new float[Rows * V];
        for (int r = 0; r < Rows; r++) raw.CopyTo(host, r * V);
        device.Upload(host, logits);

        var counts = new int[V];
        var got = new int[Rows];
        for (int launch = 0; launch < Launches; launch++)
        {
            kernel.Launch(logits, ids, Rows, V, V, seed: 0xC0FFEEUL + (ulong)launch, firstTokenIndex: launch * Rows, invTemperature: invT);
            device.Download(ids, MemoryMarshal.Cast<int, float>(got.AsSpan()));
            foreach (int t in got) counts[t]++;
        }

        // expected = softmax(raw * invT)
        double max = double.NegativeInfinity;
        var p = new double[V];
        for (int i = 0; i < V; i++) { p[i] = float.IsNegativeInfinity(raw[i]) ? 0 : Math.Exp((double)(raw[i] * invT)); }
        double z = p.Sum();
        int n = Rows * Launches;
        double chi = 0;
        int df = -1;
        for (int i = 0; i < V; i++)
        {
            p[i] /= z;
            if (p[i] == 0) { Assert.Equal(0, counts[i]); continue; }
            double e = p[i] * n;
            chi += (counts[i] - e) * (counts[i] - e) / e;
            df++;
        }
        _ = max;
        // df = 14 here (15 live bins); p = 0.001 critical value 36.12. Fixed seeds, so the outcome is deterministic.
        Assert.Equal(14, df);
        Assert.True(chi < 36.12, $"chi-square {chi:F2} for df {df}");
    }
}
