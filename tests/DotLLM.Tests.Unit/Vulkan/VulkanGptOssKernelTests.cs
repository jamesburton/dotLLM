using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical parity for the gpt-oss-specific Vulkan kernels (#737): MXFP4 indexed expert matmul,
/// clamped OAI SwiGLU, per-expert bias add, raw-top-k router and sink attention. Each is checked
/// against an independent scalar reference (or the CPU engine kernel).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGptOssKernelTests
{
    private static readonly int[] KValues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    private static float E8M0Half(byte e)
    {
        uint bits = e < 2 ? 0x00200000u << e : (uint)(e - 1) << 23;
        return BitConverter.UInt32BitsToSingle(bits);
    }

    [SkippableTheory]
    [InlineData(3, 4, 8, 32, 3)]
    [InlineData(40, 5, 33, 96, 4)]
    [InlineData(6, 32, 64, 2880, 6)]   // gpt-oss-20b K (90 blocks; 17-byte rows are not 4-byte aligned)
    public void Mxfp4IndexedMatmul_MatchesDequantReference(int n, int numExperts, int m, int k, int activeExperts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x737 + n * 31 + numExperts * 17 + m * 11 + k);
        int blocksPerRow = k / 32;
        int rowBytes = blocksPerRow * 17;
        byte[] bank = new byte[(long)numExperts * m * rowBytes];
        float[] deq = new float[(long)numExperts * m * k];
        for (long row = 0; row < (long)numExperts * m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                long off = row * rowBytes + b * 17;
                byte e = (byte)rng.Next(118, 130);        // scales around 2^-9 .. 2^2
                bank[off] = e;
                float d = E8M0Half(e);
                for (int j = 0; j < 16; j++)
                {
                    byte q = (byte)rng.Next(256);
                    bank[off + 1 + j] = q;
                    deq[row * k + b * 32 + j] = KValues[q & 0xF] * d;
                    deq[row * k + b * 32 + j + 16] = KValues[q >> 4] * d;
                }
            }
        }

        float[] x = Random(rng, n * k);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = rng.Next(Math.Min(activeExperts, numExperts));

        var expected = new float[n * m];
        for (int r = 0; r < n; r++)
            for (int o = 0; o < m; o++)
            {
                double acc = 0;
                long wb = ((long)indices[r] * m + o) * k;
                for (int j = 0; j < k; j++) acc += deq[wb + j] * x[r * k + j];
                expected[r * m + o] = (float)acc;
            }

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedMatmulMxfp4F32Kernel.Create(device, spvDir);
        using var bankBuf = device.Allocate(bank.Length);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));
        device.Upload(bank, bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

        kernel.Launch(bankBuf, xBuf, idxBuf, yBuf, m, k, n, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float bar = 1e-3f + 2e-4f * MathF.Abs(expected[i]) + 1e-5f * k;
            Assert.True(diff <= bar, $"row={i / m}, col={i % m}: ref={expected[i]:F6} vs vulkan={actual[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void SwiGluOai_MatchesCpuKernel_IncludingClampRegion()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(11);
        int n = 4097;
        float[] gate = new float[n], up = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = (float)(rng.NextDouble() * 30 - 15);   // exercises min(gate, 7)
            up[i] = (float)(rng.NextDouble() * 30 - 15);     // exercises clamp(up, -7, 7)
        }
        float[] expected = new float[n];
        MoeQuantSwiGluMlp.SwiGluOai(gate, up, expected);

        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var gBuf = device.Allocate((long)n * 4);
        using var uBuf = device.Allocate((long)n * 4);
        using var rBuf = device.Allocate((long)n * 4);
        device.Upload(gate, gBuf);
        device.Upload(up, uBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordSwiGluOai(ctx.CommandBuffer, gBuf, uBuf, rBuf, n);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[n];
        device.Download(rBuf, actual);
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f + 1e-5f * MathF.Abs(expected[i]),
                $"i={i} gate={gate[i]} up={up[i]}: cpu={expected[i]} vulkan={actual[i]}");
    }

    [SkippableFact]
    public void ExpertBiasAdd_AddsTheRoutedExpertsRow()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(5);
        int rows = 19, dim = 37, experts = 6;
        float[] y = Random(rng, rows * dim);
        float[] bias = Random(rng, experts * dim);
        int[] idx = new int[rows];
        for (int i = 0; i < rows; i++) idx[i] = rng.Next(experts);
        float[] expected = (float[])y.Clone();
        for (int r = 0; r < rows; r++)
            for (int i = 0; i < dim; i++) expected[r * dim + i] += bias[idx[r] * dim + i];

        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var yBuf = device.Allocate((long)y.Length * 4);
        using var bBuf = device.Allocate((long)bias.Length * 4);
        using var iBuf = device.Allocate((long)idx.Length * 4);
        device.Upload(y, yBuf);
        device.Upload(bias, bBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(idx), iBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordExpertBiasAdd(ctx.CommandBuffer, yBuf, bBuf, iBuf, rows, dim, experts);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[y.Length];
        device.Download(yBuf, actual);
        for (int i = 0; i < y.Length; i++) Assert.Equal(expected[i], actual[i], 6);
    }

    [SkippableTheory]
    [InlineData(5, 32, 4)]
    [InlineData(3, 128, 8)]
    [InlineData(1, 32, 4)]
    public void RawTopKSoftmax_SelectsOnRawLogitsThenSoftmaxesSelected(int seqLen, int numExperts, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(23 + seqLen);
        float[] logits = new float[seqLen * numExperts];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 12 - 6);

        var expIdx = new int[seqLen * k];
        var expW = new float[seqLen * k];
        for (int t = 0; t < seqLen; t++)
        {
            var order = Enumerable.Range(0, numExperts)
                .OrderByDescending(e => logits[t * numExperts + e]).ThenBy(e => e).Take(k).ToArray();
            float mx = logits[t * numExperts + order[0]];
            double sum = 0;
            for (int s = 0; s < k; s++) sum += Math.Exp(logits[t * numExperts + order[s]] - mx);
            for (int s = 0; s < k; s++)
            {
                expIdx[t * k + s] = order[s];
                expW[t * k + s] = (float)(Math.Exp(logits[t * numExperts + order[s]] - mx) / sum);
            }
        }

        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var lBuf = device.Allocate((long)logits.Length * 4);
        using var iBuf = device.Allocate((long)expIdx.Length * 4);
        using var wBuf = device.Allocate((long)expW.Length * 4);
        device.Upload(logits, lBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordTopKRawSoftmax(ctx.CommandBuffer, lBuf, iBuf, wBuf, seqLen, numExperts, k);
            ctx.SubmitAndWait();
        }
        int[] aIdx = new int[expIdx.Length];
        float[] aW = new float[expW.Length];
        device.Download(iBuf, MemoryMarshal.Cast<int, float>(aIdx.AsSpan()));
        device.Download(wBuf, aW);
        Assert.Equal(expIdx, aIdx);
        for (int i = 0; i < aW.Length; i++) Assert.Equal(expW[i], aW[i], 5);
    }

    [SkippableTheory]
    [InlineData(1, 40, 0)]      // decode, no window
    [InlineData(1, 40, 8)]      // decode, window
    [InlineData(17, 17, 0)]     // prefill dense
    [InlineData(17, 17, 5)]     // prefill windowed
    [InlineData(4, 36, 6)]      // chunked prefill with a KV prefix
    public void SinkAttention_MatchesCpuKernel(int seqQ, int seqKv, int window)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int numHeads = 8, numKvHeads = 2, headDim = 64;
        var rng = new Random(77 + seqQ * 3 + seqKv + window);
        float[] q = Random(rng, seqQ * numHeads * headDim);
        float[] k = Random(rng, seqKv * numKvHeads * headDim);
        float[] v = Random(rng, seqKv * numKvHeads * headDim);
        float[] sinks = new float[numHeads];
        for (int h = 0; h < numHeads; h++) sinks[h] = (float)(rng.NextDouble() * 6 - 3);
        int posOff = seqKv - seqQ;
        float scale = 1f / MathF.Sqrt(headDim);

        float[] expected = new float[seqQ * numHeads * headDim];
        Attention.Execute(q, k, v, expected, seqQ, seqKv, numHeads, numKvHeads, headDim, posOff, scale,
            window > 0 ? window : null, 0f, sinks);

        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        using var qBuf = device.Allocate((long)q.Length * 4);
        using var kBuf = device.Allocate((long)k.Length * 4);
        using var vBuf = device.Allocate((long)v.Length * 4);
        using var oBuf = device.Allocate((long)expected.Length * 4);
        using var sBuf = device.Allocate((long)sinks.Length * 4);
        device.Upload(q, qBuf); device.Upload(k, kBuf); device.Upload(v, vBuf); device.Upload(sinks, sBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordAttentionSinks(ctx.CommandBuffer, qBuf, kBuf, vBuf, oBuf, sBuf,
                seqQ, seqKv, numHeads, numKvHeads, headDim, posOff, window);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[expected.Length];
        device.Download(oBuf, actual);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 2e-4f + 1e-3f * MathF.Abs(expected[i]),
                $"i={i}: cpu={expected[i]} vulkan={actual[i]}");

        // Discrimination: the sink must matter on this fixture (otherwise a sink-ignoring
        // kernel would pass). Re-run the CPU reference without sinks and require a real gap.
        float[] noSink = new float[expected.Length];
        Attention.Execute(q, k, v, noSink, seqQ, seqKv, numHeads, numKvHeads, headDim, posOff, scale,
            window > 0 ? window : null, 0f);
        float maxGap = 0;
        for (int i = 0; i < noSink.Length; i++) maxGap = MathF.Max(maxGap, MathF.Abs(noSink[i] - expected[i]));
        Assert.True(maxGap > 5e-3f, $"fixture too weak: sink changes output by only {maxGap}");
    }

    private static float[] Random(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
