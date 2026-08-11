using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Real-bytes parity test for the Vulkan Q5_0 → FP32 dequant kernel
/// (<see cref="Q5_0DequantF32Kernel"/>) against the CPU oracle anchored in
/// <c>DequantizeQ5_0AnchorTests</c> (#344). Sister test to the Q3_K real-GGUF
/// parity gate in <see cref="RealGgufVulkanParityTests"/>: that test caught
/// #311 (Q3_K's bit layout was transposed) only because it ran against real
/// tensor bytes rather than a self-authored fixture, since the fixture
/// generator shared the same bug as the kernel. This test reads every Q5_0
/// tensor out of a real GGUF and dequantizes it both ways, comparing
/// element-for-element with 0 ULP tolerance (both paths do one multiply per
/// element, no reduction, so any difference is a real divergence).
/// </summary>
[Trait("Category", "GPU")]
public sealed class RealGgufQ5_0ParityTests
{
    private readonly ITestOutputHelper _output;

    public RealGgufQ5_0ParityTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public unsafe void SmolLM135M_Q5_0_VulkanDequant_MatchesCpuOracle()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        string path = fixture.Path!;

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q5_0_dequant_f32.spv")),
            "q5_0_dequant_f32.spv not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {path}");

        using var gguf = GgufFile.Open(path);
        var q5_0Tensors = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.Q5_0)
            .ToList();

        if (q5_0Tensors.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {path} contains no Q5_0 tensors. Types present: "
                + string.Join(", ", typesPresent));
            return;
        }

        // Allocate device buffers ONCE at the maximum tensor size and reuse
        // them across the loop below. DescriptorSetCache is keyed on raw
        // Vulkan buffer handles, which Vulkan recycles — per-iteration
        // alloc/free can bind a dead smaller buffer and silently read/write
        // zeros, masquerading as a kernel truncation bug.
        long maxElements = q5_0Tensors.Max(t => t.Shape.ElementCount);
        long maxBlocks = maxElements / Q5_0DequantF32Kernel.Q5_0GroupSize;
        long maxSrcBytes = maxBlocks * Q5_0DequantF32Kernel.Q5_0BlockBytes;
        long maxSrcBytesAligned = (maxSrcBytes + 3) & ~3L;
        long maxDstBytes = maxElements * sizeof(float);

        using var device = VulkanDevice.Create();
        using var kernel = Q5_0DequantF32Kernel.Create(device, spvDir);
        using var bufSrc = device.Allocate(maxSrcBytesAligned);
        using var bufDst = device.Allocate(maxDstBytes);

        long totalBlocksChecked = 0;
        int tensorsChecked = 0;

        foreach (var tensor in q5_0Tensors)
        {
            long elements = tensor.Shape.ElementCount;
            Assert.Equal(0, elements % Q5_0DequantF32Kernel.Q5_0GroupSize);
            int totalBlocks = checked((int)(elements / Q5_0DequantF32Kernel.Q5_0GroupSize));
            long srcBytes = (long)totalBlocks * Q5_0DequantF32Kernel.Q5_0BlockBytes;

            nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            float[] expected = new float[elements];
            Dequantize.ToFloat32(tensorPtr, elements, QuantizationType.Q5_0, expected);

            var srcSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)srcBytes));
            device.Upload(srcSpan, bufSrc);
            kernel.Launch(bufSrc, bufDst, totalBlocks);

            float[] actual = new float[elements];
            device.Download(bufDst, actual);

            for (int i = 0; i < elements; i++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                    Assert.Fail(
                        $"[{tensor.Name}] Q5_0 dequant mismatch at element {i} "
                        + $"(block {i / Q5_0DequantF32Kernel.Q5_0GroupSize}): "
                        + $"cpu={expected[i]:G9} gpu={actual[i]:G9}");
            }

            totalBlocksChecked += totalBlocks;
            tensorsChecked++;
            _output.WriteLine($"[{tensor.Name}] {totalBlocks} blocks, {elements} elements — match.");
        }

        _output.WriteLine($"Checked {tensorsChecked} Q5_0 tensors, {totalBlocksChecked} blocks total.");
        Assert.True(tensorsChecked > 0);
    }

    private static void SkipIfVulkanUnavailable(out string spvDir)
    {
        Skip.IfNot(IsVulkanRuntimeAvailable(),
            "Vulkan runtime not available on this host (vulkan-1.dll missing or no compatible device).");
        spvDir = ResolveSpvDir();
        Skip.If(spvDir is null || !Directory.Exists(spvDir),
            $"Vulkan SPV directory not found (resolved: {spvDir ?? "null"}).");
    }

    private static bool IsVulkanRuntimeAvailable()
    {
        try
        {
            using var d = VulkanDevice.Create();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static string ResolveSpvDir()
    {
        // The repo ships SPV blobs at native/vulkan/spv/ relative to the
        // repo root. Tests run from bin/Debug/net10.0/, so walk up to the
        // repo root.
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "vulkan", "spv");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return null!;
    }
}
