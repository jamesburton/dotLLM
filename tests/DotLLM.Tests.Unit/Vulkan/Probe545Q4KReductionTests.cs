using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Runtime.CompilerServices;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #545 — whether the Q4_K prefill GEMM carries the same reduction-depth
/// accuracy gap #544 found and fixed in Q8_0, and whether fixing it works.
/// </summary>
/// <remarks>
/// <para>
/// #544 established for Q8_0 that <c>matmul_q8_0_mmq.comp</c> summed all K/32
/// block products sequentially (O(n) error growth) while the decode GEMV splits
/// them across a subgroup and finishes with <c>subgroupAdd</c> (O(log n)) — so
/// prefill was 1.65-2.37x less accurate than decode, widening with K.
/// </para>
/// <para>
/// Q4_K has the same shape on paper: <c>matmul_q4_k_mmq.comp</c> accumulates
/// <c>blocksPerRow x 2 x SUBS_PER_CHUNK</c> = 8 per super-block sequentially,
/// and <c>matmul_q4_k_mmvq.comp</c> finishes with a <c>subgroupAdd</c>. But that
/// is a reading, not a measurement: the ratio is measured here rather than
/// carried over from Q8_0, because the two families differ in unpack, in scale
/// handling, and in how the min term is distributed across lanes.
/// </para>
/// <para>
/// <b>The oracle is validated before it is used.</b> It accumulates in
/// <c>double</c> the same quantity both kernels compute, from the same bytes:
/// the Q4_K weights as the GPU sees them, and the Q8_1 activation
/// <i>downloaded from the GPU quantizer</i> rather than re-quantized on the CPU
/// (re-quantizing would introduce rounding that is not the kernel's). Its first
/// assertion is against <b>MMVQ</b>: if the oracle does not agree with the
/// shallow kernel to ~1E-07, the oracle's algebra is wrong — sign of the min
/// term, or the footing of the Q8_1 <c>s</c> term — and nothing measured against
/// it afterwards means anything. Q3_K's dequant was transposed in every backend
/// for months, so nothing here is assumed.
/// </para>
/// <para>Runs by default; ~1 s, anchored to a CPU oracle rather than to another GPU kernel.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe545Q4KReductionTests
{
    private const int Q4KGroupSize = 256;   // super-block
    private const int Q4KBlockBytes = 144;
    private const int SubBlockSize = 32;

    /// <summary>MMVQ must agree with the f64 oracle this closely, or the oracle is wrong.</summary>
    private const double OracleValidationRms = 1e-5;

    /// <summary>
    /// Relative-RMS bound the Q4_K prefill GEMM must hold against the f64 oracle.
    /// <para>
    /// Deliberately NOT a ratio against MMVQ, which is what #544 used for Q8_0.
    /// After #545's two-level accumulation, Q4_K MMQ is <i>better</i> than MMVQ
    /// (0.49-0.72x its error), so a ratio gate would be satisfied by MMQ getting
    /// worse as long as MMVQ got worse too. An absolute bound holds the gain.
    /// </para>
    /// <para>
    /// Measured post-fix: 1.286E-07 / 1.352E-07 / 1.563E-07. Pre-fix: 1.931E-07 /
    /// 1.968E-07 / 3.344E-07. 1.75E-07 separates them on every shape. The margin
    /// is modest but these kernels are deterministic — repeat runs reproduce the
    /// figures exactly — so it is a bound, not a tolerance.
    /// </para>
    /// </summary>
    private const double MmqRelRmsBound = 1.75e-7;

    /// <summary>
    /// Q4_K MMVQ's own relative RMS, held so the decode path cannot quietly
    /// regress while attention is on prefill. It is now the WEAKER of the two —
    /// see the follow-up noted in the test body.
    /// </summary>
    private const double MmvqRelRmsBound = 3.6e-7;

    private readonly ITestOutputHelper _out;
    public Probe545Q4KReductionTests(ITestOutputHelper output) => _out = output;

    /// <summary>(m, k) — Llama/Qwen projection shapes; k spans one and four super-blocks per 2048.</summary>
    private static readonly (int m, int k)[] Shapes =
    [
        (2048, 2048),
        (8192, 2048),
        (2048, 8192),   // 32 super-blocks = 256 sequential accumulations in MMQ
    ];

    [SkippableFact]
    public void Q4K_Mmq_ReductionDepth_VsMmvqAndF64Oracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — neither Q4_K integer-dot path exists here.");

        using var quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmvq = MatMulQ4KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmvq.spv missing or unsupported.");
        using var mmq = MatMulQ4KMmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmq.spv missing or unsupported.");

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} (VendorId 0x{device.VendorId:X4})");
        sb.AppendLine("Both kernels scored against an f64 accumulation of the SAME quantities.");
        sb.AppendLine();

        foreach ((int m, int k) in Shapes)
        {
            var rng = new Random(0x545 + m * 31 + k);
            float[] weightsF32 = Q4KFixture.RandomFloats(rng, m * k, 0.1f);
            byte[] weightsQ4K = Q4KFixture.QuantizeRows(weightsF32, m, k);
            float[] x = Q4KFixture.RandomFloats(rng, k, 1.0f);

            using var bufW = device.Allocate(((long)weightsQ4K.Length + 3) & ~3L);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
            using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
            using var bufV = device.Allocate((long)m * sizeof(float));
            using var bufQ = device.Allocate((long)m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(weightsQ4K), bufW);
            device.Upload(x, bufX);

            // One quantized activation, shared by both arms and by the oracle, so
            // the only thing varying between them is the matmul.
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quantRows.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, 1, k);
                ctx.SubmitAndWait();
            }

            var xqWords = new float[QuantizeQ8_1RowsKernel.PackedBytes(1, k) / sizeof(float)];
            var xds = new float[QuantizeQ8_1RowsKernel.ScaleBytes(1, k) / sizeof(float)];
            device.Download(bufXq, xqWords);
            device.Download(bufXds, xds);

            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufV, m, k);
                mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufQ, m, k, 1);
                ctx.SubmitAndWait();
            }

            var got_v = new float[m];
            var got_q = new float[m];
            device.Download(bufV, got_v);
            device.Download(bufQ, got_q);

            double[] oracle = OracleF64(weightsQ4K, xqWords, xds, m, k);
            (double rmsV, double maxV) = ScoreAgainst(oracle, got_v);
            (double rmsQ, double maxQ) = ScoreAgainst(oracle, got_q);

            // Relative RMS, since Q4_K output magnitudes differ from Q8_0's and an
            // absolute bound would not transfer between shapes.
            double scale = RootMeanSquare(oracle);
            double relV = rmsV / scale, relQ = rmsQ / scale;

            sb.AppendLine($"  m={m,5} k={k,5}   super-blocks={k / Q4KGroupSize,3}  MMQ sequential accumulations={8 * k / Q4KGroupSize,4}");
            sb.AppendLine($"    |oracle| rms={scale:E3}");
            sb.AppendLine($"    MMVQ  rms={rmsV:E3} max={maxV:E3}  rel={relV:E3}");
            sb.AppendLine($"    MMQ   rms={rmsQ:E3} max={maxQ:E3}  rel={relQ:E3}   ratio(MMQ/MMVQ)={rmsQ / rmsV:F3}");

            // ORACLE VALIDATION, before any conclusion is drawn from it. MMVQ is the
            // shallow kernel; if the oracle disagrees with it by more than int8
            // activation quantization can explain, the oracle's algebra is wrong
            // (min-term sign, or the footing of the Q8_1 s term) and the MMQ number
            // below is meaningless.
            Assert.True(relV < OracleValidationRms,
                $"the f64 oracle does not agree with MMVQ at m={m} k={k} (relative rms {relV:E3} "
                + $"> {OracleValidationRms:E0}). The ORACLE is the suspect here, not the kernel — "
                + "check the Q4_K min-term sign and whether xds.y is d*sum(xq) or sum(xq)."
                + Environment.NewLine + sb);

            // #545 REGRESSION GATES.
            Assert.True(relQ < MmqRelRmsBound,
                $"Q4_K MMQ (prefill) accuracy has regressed at m={m} k={k}: relative rms {relQ:E3} "
                + $">= {MmqRelRmsBound:E3}. #545 brought this to 1.29-1.56E-07 via the two-level "
                + "accumulation in matmul_q4_k_mmq.comp; the pre-fix values were 1.93-3.34E-07."
                + Environment.NewLine + sb);

            Assert.True(relV < MmvqRelRmsBound,
                $"Q4_K MMVQ (decode) accuracy has regressed at m={m} k={k}: relative rms {relV:E3} "
                + $">= {MmvqRelRmsBound:E3}." + Environment.NewLine + sb);
        }

        // Finding worth carrying: after #545, Q4_K MMQ is 0.49-0.72x MMVQ's error,
        // i.e. the DECODE kernel is now the weaker of the two for this quant. MMVQ
        // accumulates 2 terms per super-block sequentially per lane before its
        // subgroupAdd, so it has the same kind of depth left in it. That is a
        // separate piece of work from this issue, and the reverse of Q8_0's
        // situation — do not assume which side is weaker without measuring.

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// f64 reference: Q4_K weight rows times one Q8_1 activation row, accumulated
    /// in double. Unpacks exactly as <c>Q4KFixture.CpuGemvQ4K</c> does
    /// (<c>w = d*sc[j]*nib - dmin*mn[j]</c>) but dots against the activation the
    /// GPU quantizer produced, so the only difference from either kernel is the
    /// order and precision of the summation.
    /// </summary>
    /// <summary>
    /// Exposed so <see cref="Probe545GenericOracleAgreementTests"/> can anchor the
    /// family-agnostic oracle to this one, which is the only one validated against
    /// a GPU kernel directly.
    /// </summary>
    internal static double[] OracleF64ForCrossCheck(byte[] weightsQ4K, float[] xqWords, float[] xds, int m, int k)
        => OracleF64(weightsQ4K, xqWords, xds, m, k);

    private static unsafe double[] OracleF64(byte[] weightsQ4K, float[] xqWords, float[] xds, int m, int k)
    {
        // Unpack the Q8_1 activation: 4 int8 per 32-bit word, (d, s) per 32 elements.
        var xInt = new sbyte[k];
        for (int w = 0; w < xqWords.Length; w++)
        {
            int bits = BitConverter.SingleToInt32Bits(xqWords[w]);
            for (int b = 0; b < 4; b++) xInt[w * 4 + b] = (sbyte)((bits >> (b * 8)) & 0xFF);
        }
        var xF = new double[k];
        for (int i = 0; i < k; i++) xF[i] = xds[(i / SubBlockSize) * 2] * (double)xInt[i];

        int blocksPerRow = k / Q4KGroupSize;
        int rowBytes = blocksPerRow * Q4KBlockBytes;
        var result = new double[m];

        Span<byte> scBuf = stackalloc byte[8];
        Span<byte> mnBuf = stackalloc byte[8];

        fixed (byte* wPtr = weightsQ4K)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                double sum = 0.0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q4KBlockBytes;
                    double d = (float)Unsafe.ReadUnaligned<Half>(block);
                    double dmin = (float)Unsafe.ReadUnaligned<Half>(block + 2);
                    fixed (byte* sc = scBuf)
                    fixed (byte* mn = mnBuf)
                    {
                        DotLLM.Cpu.Kernels.Dequantize.UnpackQ4Q5Scales(block + 4, sc, mn);
                    }
                    byte* qs = block + 16;
                    int xBase = b * Q4KGroupSize;

                    for (int j = 0; j < 8; j++)
                    {
                        double scF = d * scBuf[j];
                        double mnF = dmin * mnBuf[j];
                        int pairIdx = j / 2;
                        int nibbleHalf = j % 2;
                        int outBase = xBase + j * SubBlockSize;
                        for (int i = 0; i < SubBlockSize; i++)
                        {
                            int qsByte = pairIdx * SubBlockSize + i;
                            int nib = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                            sum += (scF * nib - mnF) * xF[outBase + i];
                        }
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    private static (double rms, double max) ScoreAgainst(double[] oracle, float[] got)
    {
        double se = 0, mx = 0;
        for (int i = 0; i < oracle.Length; i++)
        {
            double d = Math.Abs(got[i] - oracle[i]);
            se += d * d;
            mx = Math.Max(mx, d);
        }
        return (Math.Sqrt(se / oracle.Length), mx);
    }

    private static double RootMeanSquare(double[] v)
    {
        double se = 0;
        foreach (double d in v) se += d * d;
        return Math.Sqrt(se / v.Length);
    }
}
