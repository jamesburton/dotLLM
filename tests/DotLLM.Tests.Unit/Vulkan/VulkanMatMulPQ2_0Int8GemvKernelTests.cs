using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #496 — parity tests for the int8-activation PQ2_0 GEMV
/// (<see cref="MatMulPQ2_0Int8GemvKernel"/>) and its activation quantizer
/// (<see cref="QuantizePQ2_0Int8Kernel"/>), the Vulkan twin of CUDA #485.
/// </summary>
/// <remarks>
/// <para>
/// Three oracles, each catching a different class of bug:
/// </para>
/// <list type="number">
///   <item><b>The quantizer against the CPU's own W2A8 activation quantizer</b>
///     (<c>MatMul.QuantizeF32ToQ8_0</c>), byte by byte after un-permuting. This is the only test
///     that can see a wrong <c>scale</c>-vs-<c>d</c> choice or a broken 4x4 byte transpose in
///     isolation; a GEMV parity test would hide both inside the tolerance.</item>
///   <item><b>The GEMV against the CPU W2A8 GEMV</b> (<c>MatMul.GemvPQ2_0</c>) — the same tier,
///     so the only legitimate divergence is fp32 summation order and the quantizer's own ULP
///     differences from the CPU one.</item>
///   <item><b>The GEMV against the float-activation Vulkan kernel</b>: argmax-exact and
///     RMS-bounded, AND asserted to be <b>not bit-identical</b>. Bit-identical means the float
///     shader ran — the int8 path cannot reproduce it — which is how a stale <c>.spv</c> copied
///     into <c>bin/</c> or an unhonoured flag is caught.</item>
/// </list>
/// <para>
/// Weights use codes 0..3 (values −1, 0, +1, <b>+2</b>), not just ternary: code 3 is the one the
/// <c>-Σq</c> accumulator seed and the <c>&amp; 0x03030303</c> mask are easiest to get wrong on.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulPQ2_0Int8GemvKernelTests
{
    private const int GroupSize = 128;
    private const int GroupBytes = 34;
    private const int QBlock = 32;

    /// <summary>
    /// Creates a PQ2_0 GEMV kernel with the int8 path forced on, or skips when the device has no
    /// integer dot product / the SPIR-V is missing.
    /// </summary>
    private static MatMulPQ2_0GemvF32Kernel CreateInt8(VulkanDevice device, string spvDir)
    {
        bool? saved = MatMulPQ2_0Int8GemvKernel.EnabledGlobalOverride;
        MatMulPQ2_0Int8GemvKernel.EnabledGlobalOverride = true;
        try
        {
            var kernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);
            if (!kernel.UsesInt8)
            {
                kernel.Dispose();
                throw new SkipException(
                    "Device does not advertise VK_KHR_shader_integer_dot_product, or the int8 SPIR-V is missing.");
            }
#pragma warning disable IDISP011 // ownership transfers to the caller; the disposed path throws instead of returning.
            return kernel;
#pragma warning restore IDISP011
        }
        finally
        {
            MatMulPQ2_0Int8GemvKernel.EnabledGlobalOverride = saved;
        }
    }

    /// <summary>
    /// The quantizer against <c>MatMul.QuantizeF32ToQ8_0</c>, the CPU W2A8 activation quantizer.
    /// Reports the int8 mismatch rate; ±1 at an exact rounding tie is tolerated (GLSL has no
    /// correctly-rounded divide), a wrong value is not.
    /// </summary>
    [SkippableTheory]
    [InlineData(128, 1)]
    [InlineData(1024, 1)]
    [InlineData(2560, 3)]
    [InlineData(512, 8)]
    public unsafe void Quantizer_MatchesCpuQ8_0ActivationQuantizer(int k, int columns)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x1249 ^ (k * 31 + columns));
        float[] x = new float[k * columns];
        for (int i = 0; i < x.Length; i++)
        {
            // Heavy-tailed: a few large outliers per row is the case per-32 blocking exists for.
            float u = rng.NextSingle() * 2f - 1f;
            x[i] = rng.Next(64) == 0 ? u * 40f : u;
        }

        using var device = VulkanDevice.Create();
        using var quant = QuantizePQ2_0Int8Kernel.TryCreate(device, spvDir)
            ?? throw new SkipException("quantize_pq2_0_int8.spv missing.");

        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufXq = device.Allocate(QuantizePQ2_0Int8Kernel.PackedBytes(k, columns));
        using var bufMeta = device.Allocate(QuantizePQ2_0Int8Kernel.MetaBytes(k, columns));
        device.Upload(x, bufX);

        quant.Launch(bufX, bufXq, bufMeta, k, columns);

        // VulkanDevice.Download only speaks float; reinterpret (the buffers are 4-byte multiples).
        float[] qRaw = new float[QuantizePQ2_0Int8Kernel.PackedBytes(k, columns) / sizeof(float)];
        device.Download(bufXq, qRaw);
        byte[] gpuQ = System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(qRaw).ToArray();

        float[] metaRaw = new float[QuantizePQ2_0Int8Kernel.MetaBytes(k, columns) / sizeof(float)];
        device.Download(bufMeta, metaRaw);
        uint[] meta = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(metaRaw).ToArray();

        int blocks = k / QBlock;
        int mismatches = 0;
        long compared = 0;
        int dMismatches = 0;
        double dMaxRel = 0;
        int qMaxAbs = 0;

        byte[] cpu = new byte[(long)blocks * columns * 34];
        fixed (float* px = x)
        fixed (byte* pc = cpu)
        {
            for (int s = 0; s < columns; s++)
                MatMul.QuantizeF32ToQ8_0(px + (long)s * k, pc + (long)s * blocks * 34, k);
        }

        for (int s = 0; s < columns; s++)
        {
            for (int b = 0; b < blocks; b++)
            {
                int cpuBlock = (s * blocks + b) * 34;
                float cpuD = (float)BitConverter.ToHalf(cpu, cpuBlock);
                float gpuD = BitConverter.UInt32BitsToSingle(meta[2 * (s * blocks + b)]);
                if (cpuD != gpuD)
                {
                    dMismatches++;
                    double rel = cpuD != 0 ? Math.Abs(cpuD - gpuD) / cpuD : Math.Abs(gpuD);
                    dMaxRel = Math.Max(dMaxRel, rel);
                }

                int gpuSum = 0;
                for (int j = 0; j < QBlock; j++)
                {
                    sbyte expected = (sbyte)cpu[cpuBlock + 2 + j];
                    // Un-permute: element 16c + 4*jj + i sits at byte 16c + 4*i + jj.
                    int c = j / 16, r = j % 16, jj = r / 4, i = r % 4;
                    int srcByte = (s * k) + b * QBlock + 16 * c + 4 * i + jj;
                    sbyte actual = (sbyte)gpuQ[srcByte];
                    gpuSum += actual;
                    compared++;
                    if (expected != actual)
                    {
                        mismatches++;
                        qMaxAbs = Math.Max(qMaxAbs, Math.Abs(expected - actual));
                    }
                }
                // The block sum must match the bytes this kernel actually wrote — the GEMV seeds
                // its dot chain with it, so a disagreement here is a silent wrong answer.
                Assert.Equal(gpuSum, unchecked((int)meta[2 * (s * blocks + b) + 1]));
            }
        }

        double rate = (double)mismatches / compared;
        string report =
            $"int8 mismatches {mismatches}/{compared} ({rate:G3}), max |Δq| {qMaxAbs}; " +
            $"scale mismatches {dMismatches}/{blocks * columns}, max rel {dMaxRel:G3}";

        // MEASURED on gfx1151: 0/13312 int8 mismatches and 0/404 scale mismatches — the quantizer
        // is byte-exact with the CPU. (It was NOT before `halfRoundEven`: packHalf2x16 disagreed
        // on ~40% of blocks by up to one fp16 ULP, worth ~3e-4 relative RMS on the GEMV.) The
        // bounds below leave room for a driver whose 2.5-ULP divide lands a value on the other
        // side of a rounding boundary; a wrong permutation, rounding mode or scale/d choice blows
        // straight through them.
        Assert.True(qMaxAbs <= 1, report);
        Assert.True(rate <= 1e-3, report);
        Assert.True(dMaxRel <= 1.0 / 1024.0, report);   // at most one fp16 ULP
    }

    /// <summary>
    /// The int8 GEMV against the CPU W2A8 GEMV (<c>MatMul.GemvPQ2_0</c>) — the same numeric tier,
    /// so only fp32 summation order should separate them.
    /// </summary>
    [SkippableTheory]
    [InlineData(64, 128)]
    [InlineData(256, 512)]
    [InlineData(1024, 2560)]
    [InlineData(577, 1024)]     // ragged M: not a multiple of the 4 rows per workgroup
    public unsafe void Int8Gemv_MatchesCpuW2A8(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x496 ^ (m * 13 + k));
        byte[] weights = PackPQ2_0Codes(RandomCodes(rng, m * k), m, k, RandomScales(rng, m, k));
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        float[] cpu = new float[m];
        fixed (byte* pw = weights)
        fixed (float* px = x)
        fixed (float* py = cpu)
            MatMul.GemvPQ2_0(pw, px, py, m, k, null);

        using var device = VulkanDevice.Create();
        using var kernel = CreateInt8(device, spvDir);

        using var bufW = device.Allocate((weights.Length + 3) & ~3);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);
        float[] gpu = new float[m];
        device.Download(bufY, gpu);

        AssertAgrees(cpu, gpu, rmsBound: 1e-4);
    }

    /// <summary>
    /// The int8 GEMV against the float-activation Vulkan kernel, for every compiled column width
    /// 1..8. Argmax-exact and RMS-bounded — and NOT bit-identical, which is what proves the int8
    /// shader ran rather than the float one.
    /// </summary>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void Int8Gemv_AgreesWithFloatKernel_ButIsNotBitIdentical(int columns)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int m = 512, k = 1024;
        var rng = new Random(0x496_C0 ^ columns);
        byte[] weights = PackPQ2_0Codes(RandomCodes(rng, m * k), m, k, RandomScales(rng, m, k));
        float[] x = new float[(long)k * columns];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        using var device = VulkanDevice.Create();
        using var bufW = device.Allocate((weights.Length + 3) & ~3);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)m * columns * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(x, bufX);

        float[] floatOut = new float[m * columns];
        using (var floatKernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir))
        {
            Assert.False(floatKernel.UsesInt8, "The float arm must not have picked up the int8 path.");
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            floatKernel.RecordColumns(ctx.CommandBuffer, bufW, bufX, bufY, m, k, columns);
            ctx.SubmitAndWait();
            device.Download(bufY, floatOut);
        }

        float[] int8Out = new float[m * columns];
        using (var int8Kernel = CreateInt8(device, spvDir))
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            int8Kernel.RecordColumns(ctx.CommandBuffer, bufW, bufX, bufY, m, k, columns);
            ctx.SubmitAndWait();
            device.Download(bufY, int8Out);
        }

        Assert.False(floatOut.AsSpan().SequenceEqual(int8Out),
            "int8 output is bit-identical to the float kernel — the int8 shader did not run " +
            "(stale .spv in bin/, or the flag was not honoured).");

        for (int s = 0; s < columns; s++)
        {
            var f = floatOut.AsSpan(s * m, m).ToArray();
            var q = int8Out.AsSpan(s * m, m).ToArray();
            // Measured 2.1e-3 .. 3.8e-3 across the eight widths on gfx1151. That gap is the
            // int8 ACTIVATION quantization itself, not this kernel: the same arrangement against
            // the CPU W2A8 GEMV — which quantizes identically — agrees to 1e-4
            // (Int8Gemv_MatchesCpuW2A8). 6e-3 leaves headroom without hiding a real defect, which
            // would move this by orders of magnitude, not by a factor of two.
            AssertAgrees(f, q, rmsBound: 6e-3);
        }
    }

    /// <summary>
    /// A multi-column dispatch must equal <c>columns</c> single-column dispatches of the same
    /// kernel — the dead-column clamp and the per-column metadata indexing are what this catches.
    /// </summary>
    [SkippableFact]
    public void Int8Gemv_MultiColumn_MatchesPerColumnDispatches()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int m = 256, k = 512, columns = 5;
        var rng = new Random(0x496_5C);
        byte[] weights = PackPQ2_0Codes(RandomCodes(rng, m * k), m, k, RandomScales(rng, m, k));
        float[] x = new float[(long)k * columns];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        using var device = VulkanDevice.Create();
        using var kernel = CreateInt8(device, spvDir);
        using var bufW = device.Allocate((weights.Length + 3) & ~3);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)m * columns * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weights), bufW);
        device.Upload(x, bufX);

        float[] batched = new float[m * columns];
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordColumns(ctx.CommandBuffer, bufW, bufX, bufY, m, k, columns);
            ctx.SubmitAndWait();
        }
        device.Download(bufY, batched);

        float[] looped = new float[m * columns];
        for (int s = 0; s < columns; s++)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            kernel.Record(ctx.CommandBuffer, bufW, bufX, bufY, m, k,
                xOffsetElements: s * k, yOffsetElements: s * m);
            ctx.SubmitAndWait();
        }
        device.Download(bufY, looped);

        for (int i = 0; i < looped.Length; i++)
            Assert.Equal(looped[i], batched[i], 4);
    }

    /// <summary>Argmax-exact plus a relative-RMS bound.</summary>
    private static void AssertAgrees(float[] expected, float[] actual, double rmsBound)
    {
        Assert.Equal(expected.Length, actual.Length);

        int ai = 0, bi = 0;
        double se = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] > expected[ai]) ai = i;
            if (actual[i] > actual[bi]) bi = i;
            double d = (double)expected[i] - actual[i];
            se += d * d;
            norm += (double)expected[i] * expected[i];
        }
        Assert.Equal(ai, bi);

        double rms = Math.Sqrt(se / expected.Length);
        double rel = norm > 0 ? rms / Math.Sqrt(norm / expected.Length) : rms;
        Assert.True(rel <= rmsBound, $"relative RMS {rel:G4} exceeds {rmsBound:G4}");
    }

    /// <summary>Random PQ2_0 codes 0..3 — values −1, 0, +1, +2. Code 3 is deliberately included.</summary>
    private static byte[] RandomCodes(Random rng, int count)
    {
        byte[] codes = new byte[count];
        for (int i = 0; i < count; i++) codes[i] = (byte)rng.Next(4);
        return codes;
    }

    private static Half[] RandomScales(Random rng, int m, int k)
    {
        var scales = new Half[m * (k / GroupSize)];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);
        return scales;
    }

    /// <summary>
    /// Packs raw codes (0..3) into the PQ2_0 layout: 34-byte groups, fp16 scale then 32 code
    /// bytes, byte <c>b</c> holding the four CONSECUTIVE elements <c>4b..4b+3</c> at ascending
    /// bit offsets {0,2,4,6}.
    /// </summary>
    private static byte[] PackPQ2_0Codes(byte[] codes, int m, int k, Half[] scales)
    {
        int groups = k / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * rowBytes];
        for (int r = 0; r < m; r++)
        {
            for (int g = 0; g < groups; g++)
            {
                int groupByteBase = r * rowBytes + g * GroupBytes;
                BitConverter.GetBytes(scales[r * groups + g]).CopyTo(buf, groupByteBase);
                for (int p = 0; p < GroupSize; p++)
                    buf[groupByteBase + 2 + p / 4] |= (byte)(codes[r * k + g * GroupSize + p] << (2 * (p % 4)));
            }
        }
        return buf;
    }
}
