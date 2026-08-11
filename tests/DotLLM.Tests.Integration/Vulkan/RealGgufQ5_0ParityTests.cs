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

    /// <summary>
    /// Real-bytes parity test for the Vulkan Q5_0 decode-path GEMV
    /// (<see cref="MatMulQ5_0GemvF32Kernel"/>) against a CPU reference computed
    /// from the trusted CPU dequant oracle (#344 Task 5). Mirrors the dequant
    /// parity test above but exercises the matmul kernel: for every 2D Q5_0
    /// weight tensor in a real GGUF, dequantize the row with the CPU oracle,
    /// dot it against a random x in double precision, and compare against the
    /// Vulkan GEMV output relative to the row's magnitude. Sister test to the
    /// Q3_K real-GGUF GEMV parity gate that caught #311 (Q3_K's bit layout was
    /// transposed) only because it ran against real tensor bytes rather than a
    /// self-authored fixture, since the fixture generator shared the same bug
    /// as the kernel — real bytes are the point, not a synthetic block.
    /// </summary>
    [SkippableFact]
    public unsafe void SmolLM135M_Q5_0_RealGgufBytes_VulkanGemv_MatchesCpuReference()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_SMOLLM135M_Q5_0_GGUF", "QuantFactory", "SmolLM-135M-GGUF",
            "SmolLM-135M.Q5_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("SmolLM-135M Q5_0 GGUF"));
        string path = fixture.Path!;

        SkipIfVulkanUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "matmul_q5_0_f32_gemv.spv")),
            "matmul_q5_0_f32_gemv.spv not compiled (glslc / Vulkan SDK required).");

        _output.WriteLine($"gguf: {path}");

        using var gguf = GgufFile.Open(path);
        var q5_0MatmulTensors = gguf.Tensors
            .Where(t => t.QuantizationType == QuantizationType.Q5_0
                     && t.Shape.Rank == 2
                     && t.Shape[0] % MatMulQ5_0GemvF32Kernel.Q5_0GroupSize == 0)
            .ToList();

        if (q5_0MatmulTensors.Count == 0)
        {
            var typesPresent = gguf.Tensors
                .Select(t => t.QuantizationType)
                .Distinct()
                .OrderBy(t => t.ToString(), StringComparer.Ordinal)
                .ToList();
            Assert.Fail(
                $"Fixture at {path} contains no 2D Q5_0 matmul tensors. Types present: "
                + string.Join(", ", typesPresent));
            return;
        }

        // Allocate device buffers ONCE at the maximum tensor size and reuse
        // them across the loop below — see the note on the dequant test above
        // for why per-iteration alloc/free is unsafe with DescriptorSetCache.
        int maxK = q5_0MatmulTensors.Max(t => t.Shape[0]);
        int maxM = q5_0MatmulTensors.Max(t => t.Shape[1]);
        long maxRowBytes = (long)(maxK / MatMulQ5_0GemvF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemvF32Kernel.Q5_0BlockBytes;
        long maxWeightBytes = maxM * maxRowBytes;
        long maxWeightBytesAligned = (maxWeightBytes + 3) & ~3L;
        long maxXBytes = ((long)maxK * sizeof(float) + 3) & ~3L;
        long maxYBytes = ((long)maxM * sizeof(float) + 3) & ~3L;

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ5_0GemvF32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(maxWeightBytesAligned);
        using var bufX = device.Allocate(maxXBytes);
        using var bufY = device.Allocate(maxYBytes);

        var rng = new Random(344);
        int tensorsChecked = 0;
        long rowsChecked = 0;

        foreach (var tensor in q5_0MatmulTensors)
        {
            int k = tensor.Shape[0];
            int m = tensor.Shape[1];
            long elements = (long)k * m;

            nint tensorPtr = gguf.DataBasePointer + (nint)tensor.DataOffset;

            float[] wDequant = new float[elements];
            Dequantize.ToFloat32(tensorPtr, elements, QuantizationType.Q5_0, wDequant);

            float[] x = new float[k];
            for (int i = 0; i < k; i++)
                x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            long rowBytes = (long)(k / MatMulQ5_0GemvF32Kernel.Q5_0GroupSize) * MatMulQ5_0GemvF32Kernel.Q5_0BlockBytes;
            long weightBytes = (long)m * rowBytes;
            var wSpan = new ReadOnlySpan<byte>((void*)tensorPtr, checked((int)weightBytes));
            device.Upload(wSpan, bufW);
            device.Upload(x.AsSpan(), bufX);

            kernel.Launch(bufW, bufX, bufY, m, k);

            float[] actual = new float[m];
            device.Download(bufY, actual);

            for (int row = 0; row < m; row++)
            {
                double refDot = 0.0;
                double sumAbs = 0.0;
                long rowBase = (long)row * k;
                for (int col = 0; col < k; col++)
                {
                    double term = (double)wDequant[rowBase + col] * x[col];
                    refDot += term;
                    sumAbs += Math.Abs(term);
                }

                // Denominator is the row's L1 magnitude (sum of |w*x| terms), not
                // |refDot| itself: with K=576 random +/-1 x values and small Q5_0
                // weights, refDot is a signed sum that can cancel down to ~1e-6
                // while sumAbs stays ~1e-2 — dividing by the cancelled value would
                // blow up a perfectly good (sub-ULP-scale) absolute difference into
                // a spurious large "relative" error.
                double rowMagnitude = Math.Max(sumAbs, 1e-6);
                double relErr = Math.Abs(actual[row] - refDot) / rowMagnitude;
                Assert.True(relErr <= 2e-2,
                    $"[{tensor.Name}] row {row}: cpu={refDot:G9} gpu={actual[row]:G9} sumAbs={sumAbs:G9} relErr={relErr:G6}");
            }

            rowsChecked += m;
            tensorsChecked++;
            _output.WriteLine($"[{tensor.Name}] m={m} k={k} — {m} rows match within 2e-2 relative.");
        }

        _output.WriteLine($"Checked {tensorsChecked} Q5_0 matmul tensors, {rowsChecked} rows total.");
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
