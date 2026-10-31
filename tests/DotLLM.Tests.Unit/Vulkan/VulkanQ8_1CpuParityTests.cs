using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ8_1CpuParityTests
{
    private readonly ITestOutputHelper _output;

    public VulkanQ8_1CpuParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void RowsQuantizer_MatchesCpuQ8Codes()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var quantizer = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        const int rowCount = 128;
        const int k = 8192;
        var random = new Random(1254);
        var input = new float[rowCount * k];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(random.NextDouble() * 2.0 - 1.0);

        var cpu = new byte[rowCount * (k / 32) * 36];
        var cpuScalar = new byte[cpu.Length];
        fixed (float* inputPtr = input)
        fixed (byte* cpuPtr = cpu)
        fixed (byte* scalarPtr = cpuScalar)
            for (int row = 0; row < rowCount; row++)
            {
                MatMul.QuantizeF32ToQ8_1(inputPtr + row * k, cpuPtr + row * (k / 32) * 36, k);
                MatMul.QuantizeF32ToQ8_1Scalar(inputPtr + row * k, scalarPtr + row * (k / 32) * 36, k);
            }

        using var inputBuffer = device.Allocate((long)input.Length * sizeof(float));
        using var qBuffer = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(rowCount, k));
        using var dsBuffer = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(rowCount, k));
        device.Upload(input, inputBuffer);
        using (var context = device.CreateSubmitContext())
        {
            context.Begin();
            quantizer.Record(context.CommandBuffer, inputBuffer, qBuffer, dsBuffer, rowCount, k);
            context.SubmitAndWait();
        }

        var gpuWords = new uint[rowCount * k / 4];
        var gpuScales = new float[rowCount * (k / 32) * 2];
        device.Download(qBuffer, MemoryMarshal.Cast<uint, float>(gpuWords.AsSpan()));
        device.Download(dsBuffer, gpuScales);

        int mismatches = 0;
        int scalarMismatches = 0;
        int cpuImplementationMismatches = 0;
        int maxCodeDifference = 0;
        var firstMismatches = new List<string>();
        for (int i = 0; i < rowCount * k; i++)
        {
            int packed = (int)((gpuWords[i / 4] >> ((i & 3) * 8)) & 0xff);
            sbyte gpuCode = unchecked((sbyte)packed);
            int row = i / k;
            int withinRow = i % k;
            int cpuOffset = row * (k / 32) * 36 + (withinRow / 32) * 36 + 4 + withinRow % 32;
            sbyte cpuCode = unchecked((sbyte)cpu[cpuOffset]);
            sbyte scalarCode = unchecked((sbyte)cpuScalar[cpuOffset]);
            if (gpuCode != cpuCode) mismatches++;
            if (gpuCode != scalarCode) scalarMismatches++;
            if (cpuCode != scalarCode) cpuImplementationMismatches++;
            maxCodeDifference = Math.Max(maxCodeDifference, Math.Abs(gpuCode - cpuCode));
            if (gpuCode != cpuCode && firstMismatches.Count < 8)
            {
                float maxAbs = 0;
                int blockStart = row * k + (withinRow / 32) * 32;
                for (int e = 0; e < 32; e++) maxAbs = Math.Max(maxAbs, Math.Abs(input[blockStart + e]));
                float cpuScale = maxAbs / 127.0f;
                float cpuInverse = 1.0f / cpuScale;
                float gpuScale = gpuScales[(row * (k / 32) + withinRow / 32) * 2];
                float gpuInverse = 1.0f / gpuScale;
                firstMismatches.Add($"flat={i}, row={row}, element={withinRow}, CPU-AVX2={cpuCode}, CPU-scalar={scalarCode}, Vulkan={gpuCode}, x={input[i]:R}, " +
                    $"scaleCPU={cpuScale:R}/0x{BitConverter.SingleToInt32Bits(cpuScale):X8}, scaleGPU={gpuScale:R}/0x{BitConverter.SingleToInt32Bits(gpuScale):X8}, " +
                    $"invCPU={cpuInverse:R}/0x{BitConverter.SingleToInt32Bits(cpuInverse):X8}, invGPU-CPU={gpuInverse:R}/0x{BitConverter.SingleToInt32Bits(gpuInverse):X8}, " +
                    $"productCPU={input[i] * cpuInverse:R}, productGPU-CPU={input[i] * gpuInverse:R}");
            }
        }

        _output.WriteLine(
            $"{device.DeviceName} (vendor 0x{device.VendorId:X4}), seed=1254, values={input.Length}: " +
            $"CPU-AVX2/Vulkan mismatches={mismatches}, CPU-scalar/Vulkan mismatches={scalarMismatches}, " +
            $"max code difference={maxCodeDifference}; " + string.Join(" | ", firstMismatches));
        Assert.Equal(0, cpuImplementationMismatches);
        Assert.True(mismatches <= input.Length / 100_000,
            $"{device.DeviceName} CPU/Vulkan Q8_1 mismatch rate exceeded 1e-5: {mismatches}/{input.Length}.");
        Assert.Equal(mismatches, scalarMismatches);
        Assert.True(maxCodeDifference <= 1,
            $"{device.DeviceName} Q8_1 activation quantization drift exceeded one code: {maxCodeDifference}.");
    }
}
