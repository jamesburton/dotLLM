using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Verifies the SPIR-V storage-buffer writes-mask reflection (issue #144) that
/// feeds the hazard-scoped barrier tracker, against the repo's real compiled
/// shaders. No GPU required — pure blob parsing.
/// </summary>
public class SpirvReflectionTests
{
    private static string? FindSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    private static uint MaskOf(string spvDir, string name)
        => SpirvReflection.ComputeStorageWritesMask(File.ReadAllBytes(Path.Combine(spvDir, name)));

    [SkippableTheory]
    // GLSL: b0 weights readonly, b1 xq readonly, b2 xds readonly, b3 y writeonly.
    [InlineData("matmul_q8_0_mmvq.spv", 0b1000u)]
    // GLSL: b0 x readonly, b1 xq writeonly, b2 xds writeonly.
    [InlineData("quantize_q8_1.spv", 0b110u)]
    // GLSL: b0 q read-write (in-place), b1 k read-write, b2 positions readonly.
    [InlineData("rope_f32.spv", 0b011u)]
    // GLSL: b0 in readonly, b1 gamma readonly, b2 out writeonly.
    [InlineData("rmsnorm_f32.spv", 0b100u)]
    [InlineData("rmsnorm_f32_sg.spv", 0b100u)]
    // GLSL: single in-place read-write binding.
    [InlineData("silu_inplace_f32.spv", 0b1u)]
    // GLSL: b0 a readonly, b1 b readonly, b2 out writeonly.
    [InlineData("add.spv", 0b100u)]
    // Split-KV attention: b0 q, b1 k, b2 v readonly; b3 partials, b4 meta written.
    [InlineData("attention_f32_splitkv.spv", 0b11000u)]
    public void KnownShaders_ReflectExpectedWriteMasks(string shader, uint expected)
    {
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V blobs not found — build native/vulkan first.");
        Skip.IfNot(File.Exists(Path.Combine(spvDir!, shader)), $"{shader} not present.");
        Assert.Equal(expected, MaskOf(spvDir!, shader));
    }

    [SkippableFact]
    public void AllShaders_ParseAndDeclareAtLeastOneWrite()
    {
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V blobs not found — build native/vulkan first.");

        int reflected = 0;
        foreach (string path in Directory.GetFiles(spvDir!, "*.spv"))
        {
            uint mask = SpirvReflection.ComputeStorageWritesMask(File.ReadAllBytes(path));
            // Every compute kernel writes at least one buffer; a zero mask
            // would mean the hazard tracker misses the kernel's output.
            Assert.True(mask != 0, $"{Path.GetFileName(path)} reflected an all-readonly mask.");
            if (mask != ~0u) reflected++;
        }
        // The reflection must actually be working (not falling back to the
        // conservative all-writable mask everywhere): the overwhelming
        // majority of shaders declare readonly inputs.
        Assert.True(reflected > 50, $"Only {reflected} shaders reflected a non-conservative mask.");
    }

    [Theory]
    [InlineData(new byte[0])]
    [InlineData(new byte[] { 1, 2, 3, 4 })] // wrong magic
    [InlineData(new byte[] { 1, 2, 3 })]    // not word-aligned
    public void GarbageInput_YieldsConservativeMask(byte[] blob)
        => Assert.Equal(~0u, SpirvReflection.ComputeStorageWritesMask(blob));
}
