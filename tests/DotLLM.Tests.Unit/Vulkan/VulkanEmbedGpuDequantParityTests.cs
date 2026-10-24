using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Discriminating end-to-end parity test for the issue-#147 GPU-side token-embed
/// dequant: uploads a real GGUF's Q4_K/Q6_K token-embed table twice — once via
/// the device dequant shader, once via the legacy CPU dequant
/// (<c>DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT=1</c>) — and requires the
/// device-resident F32 tables to be BIT-IDENTICAL over a window that spans the
/// kernel's 32768-workgroup dispatch-chunk boundary. Bit parity is what keeps
/// first-forward logits identical to the pre-#147 load path.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanEmbedGpuDequantParityTests
{
    private const string DisableEnv = "DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT";

    [SkippableFact]
    public void EmbedTable_GpuDequant_BitIdenticalToCpuPath()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string modelPath = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_EMBED_PARITY_MODEL")
            ?? "C:/Development/gguf-cache/Llama-3.2-3B-Instruct-IQ4_XS.gguf"; // Q6_K token_embd
        Skip.If(!File.Exists(modelPath), $"Parity model not found at {modelPath}");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config);

        using var device = VulkanDevice.Create();

        // Compare a window that covers > MaxBlocksPerDispatch super-blocks so a
        // broken chunk offset cannot pass: 40960 blocks × 256 elements.
        long tableElems = (long)weights.VocabSize * weights.HiddenSize;
        int compareElems = (int)Math.Min(tableElems, 40960L * 256);

        string? prev = Environment.GetEnvironmentVariable(DisableEnv);
        try
        {
            Environment.SetEnvironmentVariable(DisableEnv, null);
            using var gpuWeights = VulkanWeights.Upload(device, weights, numLayers: 1, spvDir: spvDir);
            Assert.StartsWith("gpu-", VulkanWeights.LastTokenEmbedDequantPath, StringComparison.Ordinal);
            float[] gpuTable = new float[compareElems];
            device.Download(gpuWeights.TokenEmbedding, gpuTable);

            Environment.SetEnvironmentVariable(DisableEnv, "1");
            using var cpuWeights = VulkanWeights.Upload(device, weights, numLayers: 1, spvDir: spvDir);
            Assert.Equal("cpu", VulkanWeights.LastTokenEmbedDequantPath);
            float[] cpuTable = new float[compareElems];
            device.Download(cpuWeights.TokenEmbedding, cpuTable);

            for (int i = 0; i < compareElems; i++)
            {
                if (BitConverter.SingleToInt32Bits(cpuTable[i]) != BitConverter.SingleToInt32Bits(gpuTable[i]))
                    Assert.Fail($"Embed table mismatch at element {i} (row {i / weights.HiddenSize}): " +
                                $"cpu={cpuTable[i]:G9} gpu={gpuTable[i]:G9}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(DisableEnv, prev);
        }
    }
}
