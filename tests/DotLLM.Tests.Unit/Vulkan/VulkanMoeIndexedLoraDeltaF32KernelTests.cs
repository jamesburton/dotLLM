using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Runtime.InteropServices;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical parity for the indexed MoE LoRA delta kernel. The kernel must
/// update only rows routed to the requested expert.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedLoraDeltaF32KernelTests
{
    [SkippableFact]
    public void Launch_UpdatesOnlyMatchingExpertRows()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int rows = 5;
        const int inputDim = 4;
        const int outputDim = 3;
        const int rank = 2;
        const int expert = 1;

        float[] x =
        [
            0.1f, -0.2f, 0.3f, 0.4f,
            -0.5f, 0.6f, -0.7f, 0.8f,
            0.9f, 1.0f, -1.1f, 1.2f,
            -1.3f, 1.4f, 1.5f, -1.6f,
            1.7f, -1.8f, 1.9f, -2.0f,
        ];
        int[] indices = [0, 1, 2, 1, 0];
        float[] b =
        [
            0.11f, -0.12f, 0.13f, -0.14f,
            -0.21f, 0.22f, -0.23f, 0.24f,
        ];
        float[] a =
        [
            0.31f, -0.32f,
            -0.41f, 0.42f,
            0.51f, 0.52f,
        ];
        float[] y =
        [
            1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f,
            1.6f, 1.7f, 1.8f,
            1.9f, 2.0f, 2.1f,
            2.2f, 2.3f, 2.4f,
        ];
        float[] expected = (float[])y.Clone();
        ApplyReference(x, indices, b, a, expected, rows, inputDim, outputDim, rank, expert);

        using var device = VulkanDevice.Create();
        using var kernel = MoeIndexedLoraDeltaF32Kernel.Create(device, spvDir);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var bBuf = device.Allocate((long)b.Length * sizeof(float));
        using var aBuf = device.Allocate((long)a.Length * sizeof(float));
        using var yBuf = device.Allocate((long)y.Length * sizeof(float));

        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        device.Upload(b, bBuf);
        device.Upload(a, aBuf);
        device.Upload(y, yBuf);

        kernel.Launch(xBuf, idxBuf, bBuf, aBuf, yBuf,
            rows, inputDim, outputDim, rank, expert);

        float[] actual = new float[y.Length];
        device.Download(yBuf, actual);

        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-5f,
                $"i={i} actual={actual[i]} expected={expected[i]} diff={actual[i] - expected[i]}");
    }

    private static void ApplyReference(
        float[] x,
        int[] indices,
        float[] b,
        float[] a,
        float[] y,
        int rows,
        int inputDim,
        int outputDim,
        int rank,
        int expert)
    {
        for (int row = 0; row < rows; row++)
        {
            if (indices[row] != expert) continue;
            for (int m = 0; m < outputDim; m++)
            {
                float acc = 0f;
                for (int r = 0; r < rank; r++)
                {
                    float tmp = 0f;
                    for (int k = 0; k < inputDim; k++)
                        tmp += b[r * inputDim + k] * x[row * inputDim + k];
                    acc += a[m * rank + r] * tmp;
                }
                y[row * outputDim + m] += acc;
            }
        }
    }
}
