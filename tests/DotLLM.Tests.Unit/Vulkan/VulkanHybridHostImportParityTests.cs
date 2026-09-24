using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity for the #508 zero-copy import on the <b>hybrid</b> upload path —
/// <see cref="VulkanQwen3MoeHybridWeights.UploadProjectionMatrix"/>, which is how Bonsai
/// 2 27B, Qwen3-MoE-hybrid and the Qwen3 hybrid dense stack load, and whose shape
/// Nemotron-H duplicates. <see cref="VulkanHostImportParityTests"/> does the same job for
/// <c>VulkanWeights</c>.
/// </summary>
/// <remarks>
/// The import arm and the staging arm must put the SAME bytes on the device: the import
/// is a pure allocation change, so a greedy decode over an imported model has to be
/// byte-identical to one over a staged model. The two arms are selected with
/// <c>DOTLLM_VULKAN_DISABLE_HOST_IMPORT</c>, and the test asserts which arm it actually
/// got (via <see cref="VulkanDevice.Buffer.IsHostImported"/>) rather than trusting the
/// env var — comparing two staging uploads would pass vacuously.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanHybridHostImportParityTests
{
    private const string DisableEnv = "DOTLLM_VULKAN_DISABLE_HOST_IMPORT";
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    [SkippableFact]
    public unsafe void Q8_0Projection_ImportAndStagingPutIdenticalBytesOnDevice()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not expose VK_EXT_external_memory_host on this host.");

        // m even so the packed size is a 4-byte multiple and Download's float span fits.
        const int m = 64;
        const int k = 256;
        int rowBytes = k / Q8_0GroupSize * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        Assert.Equal(0, totalBytes % 4);

        ulong pageAlignment = Math.Max(4096UL, device.MinImportedHostPointerAlignment);
        nuint allocBytes = (nuint)((((ulong)totalBytes + pageAlignment - 1) & ~(pageAlignment - 1)) + pageAlignment);
        void* host = NativeMemory.AlignedAlloc(allocBytes, (nuint)pageAlignment);

        string? original = Environment.GetEnvironmentVariable(DisableEnv);
        try
        {
            new Span<byte>(host, (int)allocBytes).Clear();

            var rng = new Random(0x508);
            float[] weightsF32 = new float[m * k];
            for (int i = 0; i < weightsF32.Length; i++)
                weightsF32[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.1);
            fixed (float* srcPtr = weightsF32)
            {
                for (int row = 0; row < m; row++)
                    MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, (byte*)host + (long)row * rowBytes, k);
            }

            float[] x = new float[k];
            for (int i = 0; i < k; i++) x[i] = (float)((rng.NextDouble() * 2.0 - 1.0));

            // The comparison runs a kernel over each arm's buffer and compares the OUTPUT,
            // exactly as VulkanHostImportParityTests does. Downloading the weight buffer
            // itself would put the imported allocation's host mapping on trial rather than
            // the bytes the shader actually reads.
            using var kernel = MatMulQ8_0Kernel.Create(device, spvDir);
            float[] staged = new float[m];
            float[] imported = new float[m];

            // Arm 1 — staging, forced.
            Environment.SetEnvironmentVariable(DisableEnv, "1");
            using (var staging = VulkanStagingBuffer.Create(device, totalBytes))
            {
                using var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                    device, staging, (nint)host, QuantizationType.Q8_0, m, k,
                    forceF32: false, out var stagedQt, out long stagedBytes);
                staging.WaitAll();

                Assert.False(buf.IsHostImported, "arm 1 must be the staging path");
                Assert.Equal(QuantizationType.Q8_0, stagedQt);
                Assert.Equal(totalBytes, stagedBytes);

                using var bufX = device.Allocate((long)k * sizeof(float));
                using var bufY = device.Allocate((long)m * sizeof(float));
                device.Upload(x, bufX);
                kernel.Launch(buf, bufX, bufY, m, k);
                device.Download(bufY, staged);
            }

            // Arm 2 — import allowed.
            Environment.SetEnvironmentVariable(DisableEnv, null);
            using (var staging = VulkanStagingBuffer.Create(device, totalBytes))
            {
                using var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                    device, staging, (nint)host, QuantizationType.Q8_0, m, k,
                    forceF32: false, out var importedQt, out long importedBytes);
                staging.WaitAll();

                Skip.IfNot(buf.IsHostImported,
                    "Driver accepts VK_EXT_external_memory_host but refused this import " +
                    "(a discrete GPU is refused by design — issue #507).");
                Assert.Equal(QuantizationType.Q8_0, importedQt);
                Assert.Equal(totalBytes, importedBytes);

                using var bufX = device.Allocate((long)k * sizeof(float));
                using var bufY = device.Allocate((long)m * sizeof(float));
                device.Upload(x, bufX);
                kernel.Launch(buf, bufX, bufY, m, k);
                device.Download(bufY, imported);
            }

            for (int i = 0; i < m; i++)
            {
                Assert.True(
                    BitConverter.SingleToInt32Bits(staged[i]) == BitConverter.SingleToInt32Bits(imported[i]),
                    $"row {i} differs: staged={staged[i]} imported={imported[i]}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(DisableEnv, original);
            NativeMemory.AlignedFree(host);
        }
    }

    /// <summary>
    /// A widened (host-dequantised) projection must NEVER import — its device image is
    /// an F32 expansion, not the source bytes, so aliasing the source would feed the
    /// kernel quant blocks reinterpreted as floats.
    /// </summary>
    [SkippableFact]
    public unsafe void WidenedProjection_NeverImports()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();

        const int m = 8;
        const int k = 256;
        int totalBytes = m * (k / Q8_0GroupSize) * Q8_0BlockBytes;
        ulong pageAlignment = Math.Max(4096UL, device.MinImportedHostPointerAlignment);
        void* host = NativeMemory.AlignedAlloc((nuint)(pageAlignment * 2), (nuint)pageAlignment);
        try
        {
            new Span<byte>(host, (int)(pageAlignment * 2)).Clear();
            Assert.True(totalBytes <= (int)pageAlignment);

            using var staging = VulkanStagingBuffer.Create(device, (long)m * k * sizeof(float));
            using var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                device, staging, (nint)host, QuantizationType.Q8_0, m, k,
                forceF32: true, out var qt, out long bytes);
            staging.WaitAll();

            Assert.False(buf.IsHostImported);
            Assert.Equal(QuantizationType.F32, qt);
            Assert.Equal((long)m * k * sizeof(float), bytes);
        }
        finally
        {
            NativeMemory.AlignedFree(host);
        }
    }
}
