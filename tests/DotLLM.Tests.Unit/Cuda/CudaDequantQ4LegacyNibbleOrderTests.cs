using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Nibble-order regression tests for the CUDA Q4_0 / Q4_1 dequant kernels (issue #254).
///
/// GGUF packs a 32-element 4-bit block as two halves, not as interleaved pairs:
/// element <c>j</c> is the low nibble of <c>qs[j]</c> and element <c>j + 16</c> is the high
/// nibble of <c>qs[j]</c>. The CUDA kernels used to read <c>out[2j] = lo(qs[j])</c>,
/// <c>out[2j+1] = hi(qs[j])</c>, which permutes every weight inside a block and silently
/// destroys model quality (perplexity ~1e10 instead of ~21).
///
/// The fixtures below deliberately give each byte a low nibble that differs from its high
/// nibble, so the correct and the interleaved layouts produce different element vectors —
/// a test built on a uniform pattern would pass against either kernel.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaDequantQ4LegacyNibbleOrderTests
{
    private const int BlockSize = 32;
    private const int BlockCount = 3;
    private const int ElementCount = BlockSize * BlockCount;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the test class with the xUnit output helper.</summary>
    /// <param name="output">Sink for diagnostic output.</param>
    public CudaDequantQ4LegacyNibbleOrderTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Q4_0: element <c>j</c> = low nibble of <c>qs[j]</c>, element <c>j + 16</c> = high nibble,
    /// dequantized as <c>d * (nibble - 8)</c>.
    /// </summary>
    [SkippableFact]
    public void Q4_0_Dequant_UsesTwoHalvesNibbleOrder()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 18;
        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[ElementCount];

        for (int b = 0; b < BlockCount; b++)
        {
            int blockBase = b * BlockBytes;
            float d = 0.5f / (1 << b); // Exactly representable in FP16.
            BitConverter.GetBytes((Half)d).CopyTo(packed, blockBase);

            for (int j = 0; j < 16; j++)
            {
                int lo = (j + b) & 0x0F;
                int hi = (j + b + 7) & 0x0F; // Never equal to lo → layout-discriminating.
                packed[blockBase + 2 + j] = (byte)((hi << 4) | lo);

                expected[b * BlockSize + j] = d * (lo - 8);
                expected[b * BlockSize + j + 16] = d * (hi - 8);
            }
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.Q4_0);
        AssertMatches(expected, actual, "Q4_0");
    }

    /// <summary>
    /// Q4_1: same two-halves nibble order as Q4_0, but dequantized as <c>d * nibble + m</c>
    /// (unsigned nibble plus a per-block minimum, no bias of 8).
    /// </summary>
    [SkippableFact]
    public void Q4_1_Dequant_UsesTwoHalvesNibbleOrderAndAppliesMin()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");

        const int BlockBytes = 20;
        byte[] packed = new byte[BlockBytes * BlockCount];
        float[] expected = new float[ElementCount];

        for (int b = 0; b < BlockCount; b++)
        {
            int blockBase = b * BlockBytes;
            float d = 0.5f / (1 << b);
            float m = -1.25f + b; // Exactly representable in FP16.
            BitConverter.GetBytes((Half)d).CopyTo(packed, blockBase);
            BitConverter.GetBytes((Half)m).CopyTo(packed, blockBase + 2);

            for (int j = 0; j < 16; j++)
            {
                int lo = (j + b) & 0x0F;
                int hi = (j + b + 7) & 0x0F;
                packed[blockBase + 4 + j] = (byte)((hi << 4) | lo);

                expected[b * BlockSize + j] = (d * lo) + m;
                expected[b * BlockSize + j + 16] = (d * hi) + m;
            }
        }

        float[] actual = DequantOnGpu(packed, QuantizationType.Q4_1);
        AssertMatches(expected, actual, "Q4_1");
    }

    private static float[] DequantOnGpu(byte[] packed, QuantizationType quantType)
    {
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());

        nint dSrc = 0;
        nint dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)packed.Length).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(ElementCount * sizeof(ushort))).ThrowOnError();

            fixed (byte* p = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)p, (nuint)packed.Length).ThrowOnError();

            kernels.LaunchDequantToF16(dSrc, quantType, dDst, ElementCount, stream.Handle);
            stream.Synchronize();

            ushort[] f16 = new ushort[ElementCount];
            fixed (ushort* p = f16)
            {
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(ElementCount * sizeof(ushort)))
                    .ThrowOnError();
            }

            float[] result = new float[ElementCount];
            for (int i = 0; i < ElementCount; i++)
                result[i] = (float)BitConverter.UInt16BitsToHalf(f16[i]);

            return result;
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }
    }

    private void AssertMatches(float[] expected, float[] actual, string label)
    {
        // Every value is exactly representable in FP16, so the comparison can be tight.
        const float Tolerance = 1e-3f;

        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (MathF.Abs(expected[i] - actual[i]) > Tolerance)
            {
                mismatches++;
                if (firstMismatch < 0) firstMismatch = i;
            }
        }

        _output.WriteLine($"{label}: mismatches={mismatches}/{expected.Length}");
        if (firstMismatch >= 0)
        {
            _output.WriteLine(
                $"  first mismatch at {firstMismatch}: expected={expected[firstMismatch]:F6}, actual={actual[firstMismatch]:F6}");
        }

        Assert.True(
            mismatches == 0,
            $"{label} dequant nibble order wrong: {mismatches} of {expected.Length} elements differ "
            + (firstMismatch >= 0
                ? $"(first at index {firstMismatch}: expected {expected[firstMismatch]}, got {actual[firstMismatch]})"
                : string.Empty));
    }

    private static string ResolvePtxDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("DOTLLM_PTX_DIR");
        if (envDir is not null && Directory.Exists(envDir))
            return envDir;

        string repoRoot = Path.GetDirectoryName(typeof(CudaDequantQ4LegacyNibbleOrderTests).Assembly.Location)!;
        while (repoRoot.Length > 3 && !File.Exists(Path.Combine(repoRoot, "dotLLM.slnx")))
            repoRoot = Path.GetDirectoryName(repoRoot)!;

        return Path.Combine(repoRoot, "native", "ptx");
    }
}
