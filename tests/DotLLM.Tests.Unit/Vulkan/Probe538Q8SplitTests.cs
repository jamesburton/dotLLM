using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #538, defect B — the <c>seqLen == 1</c> MMVQ/GEMV vs MMQ/GEMM
/// split on Q8_0, localised to the stage that actually differs.
/// </summary>
/// <remarks>
/// <para>
/// A 1-token chunk and a multi-token chunk of the same prompt produce different
/// KV on a Q8_0 model, seeded at layer <b>0</b> — before any attention runs, so
/// it is the projection matmul, not #533. <c>RecordMatmul</c> sends
/// <c>seqLen == 1</c> to <c>QuantizeQ8_1Kernel</c> + <c>matmul_q8_0_mmvq</c> and
/// everything else to <c>QuantizeQ8_1RowsKernel</c> + <c>matmul_q8_0_mmq</c>.
/// </para>
/// <para>
/// Two stages can differ and they need separating, because they have different
/// fixes: the ACTIVATION QUANTIZER may emit different <c>xq</c>/<c>xds</c> bytes
/// for the same row (a real numerical difference — the two would be computing
/// different inputs), or the quantizers may agree and only the MATMUL reduction
/// order differ (in which case there is nothing wrong with either kernel and the
/// question is what bound to accept). So this probe:
/// </para>
/// <list type="number">
///   <item>runs both quantizers on the SAME single row and compares the packed
///     bytes and the (d, s) scale pairs <b>bitwise</b>;</item>
///   <item>feeds ONE quantized buffer — whichever the comparison licenses — to
///     both matmuls and compares their outputs, so any gap measured there is the
///     matmul alone and cannot be inherited from the quantizer.</item>
/// </list>
/// <para>
/// Both stages carry a negative control that perturbs one input element by one
/// ULP and re-runs; a comparison whose count does not move is not looking at
/// anything, and that is how a "clean" verdict gets published wrongly.
/// </para>
/// <para>Enable with <c>DOTLLM_538_PROBE=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe538Q8SplitTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    private readonly ITestOutputHelper _out;
    public Probe538Q8SplitTests(ITestOutputHelper output) => _out = output;

    private static bool Enabled =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_538_PROBE"), "1", StringComparison.Ordinal);

    /// <summary>Llama-3.2-1B projection shapes, which is where #538 was observed.</summary>
    public static TheoryData<int, int> Shapes => new()
    {
        { 2048, 2048 },   // q / k / v / o projection  (m, k)
        { 8192, 2048 },   // gate / up
        { 2048, 8192 },   // down
    };

    [SkippableFact]
    public void Q8_1Quantizers_AgreeAtN1_AndMatmulsCompared()
    {
        Skip.IfNot(Enabled, "DOTLLM_538_PROBE=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — the MMVQ/MMQ split does not exist here.");

        using var q1 = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var qRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing.");
        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing.");

        var sb = new StringBuilder();
        sb.AppendLine($"Device: {device.DeviceName} (VendorId 0x{device.VendorId:X4})");
        sb.AppendLine("Stage 1 = quantizer bytes (bitwise). Stage 2 = matmul, fed ONE shared quantized buffer.");
        sb.AppendLine();

        long totalQuantDiff = 0;
        long totalMatmulDiff = 0;

        foreach (object[] row in Shapes)
        {
            int m = (int)row[0], k = (int)row[1];
            var rng = new Random(0x538 + m * 31 + k);

            float[] weightsF32 = RandomFloats(rng, m * k, 0.1f);
            byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
            float[] x = RandomFloats(rng, k, 1.0f);

            long wBytes = ((long)weightsQ8.Length + 3) & ~3L;
            using var bufW = device.Allocate(wBytes);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq1 = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
            using var bufXds1 = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
            using var bufXqR = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
            using var bufXdsR = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
            using var bufCv = device.Allocate((long)m * sizeof(float));
            using var bufCq = device.Allocate((long)m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);

            (long qDiff, long dsDiff, long mmDiff, float mmMaxAbs, float mmMaxRel) Run(float[] input)
            {
                device.Upload(input, bufX);

                // Stage 1 — both quantizers on the same row, one submit.
                using (var ctx = device.CreateSubmitContext())
                {
                    ctx.Begin();
                    q1.Record(ctx.CommandBuffer, bufX, bufXq1, bufXds1, k);
                    qRows.Record(ctx.CommandBuffer, bufX, bufXqR, bufXdsR, 1, k);
                    ctx.SubmitAndWait();
                }

                // Download() only exposes a float span; the packed int8 stream is
                // a multiple of 4 bytes by construction, so read it as 32-bit words
                // and compare BITWISE — never as float values, which would treat a
                // NaN bit pattern as equal to itself incorrectly and hide bytes.
                var xq1 = new float[QuantizeQ8_1Kernel.PackedBytes(k) / sizeof(float)];
                var xqR = new float[QuantizeQ8_1RowsKernel.PackedBytes(1, k) / sizeof(float)];
                var xds1 = new float[QuantizeQ8_1Kernel.ScaleBytes(k) / sizeof(float)];
                var xdsR = new float[QuantizeQ8_1RowsKernel.ScaleBytes(1, k) / sizeof(float)];
                device.Download(bufXq1, xq1);
                device.Download(bufXqR, xqR);
                device.Download(bufXds1, xds1);
                device.Download(bufXdsR, xdsR);

                long qd = 0;
                for (int i = 0; i < xq1.Length; i++)
                    if (BitConverter.SingleToInt32Bits(xq1[i]) != BitConverter.SingleToInt32Bits(xqR[i])) qd++;
                long dsd = 0;
                for (int i = 0; i < xds1.Length; i++)
                    if (BitConverter.SingleToInt32Bits(xds1[i]) != BitConverter.SingleToInt32Bits(xdsR[i])) dsd++;

                // Stage 2 — ONE quantized buffer into both matmuls. If stage 1 is
                // clean the choice is immaterial; if it is not, this still isolates
                // the matmul by construction rather than measuring the sum of both.
                using (var ctx = device.CreateSubmitContext())
                {
                    ctx.Begin();
                    mmvq.Record(ctx.CommandBuffer, bufW, bufXq1, bufXds1, bufCv, m, k);
                    mmq.Record(ctx.CommandBuffer, bufW, bufXq1, bufXds1, bufCq, m, k, 1);
                    ctx.SubmitAndWait();
                }

                var cv = new float[m];
                var cq = new float[m];
                device.Download(bufCv, cv);
                device.Download(bufCq, cq);

                long md = 0; float maxAbs = 0, maxRel = 0;
                for (int i = 0; i < m; i++)
                {
                    if (BitConverter.SingleToInt32Bits(cv[i]) == BitConverter.SingleToInt32Bits(cq[i])) continue;
                    md++;
                    float a = MathF.Abs(cv[i] - cq[i]);
                    maxAbs = MathF.Max(maxAbs, a);
                    float denom = MathF.Max(MathF.Abs(cv[i]), MathF.Abs(cq[i]));
                    if (denom > 0) maxRel = MathF.Max(maxRel, a / denom);
                }
                return (qd, dsd, md, maxAbs, maxRel);
            }

            var r = Run(x);

            // CONTROLS. The first version of this probe perturbed one input
            // element by a ULP and re-ran — which is DEGENERATE here: both
            // matmuls see the same perturbation, so the difference BETWEEN them
            // is unchanged (it reported the identical count, 1678/1678). It
            // could not have failed, which is exactly the flaw #532's probe had.
            //
            // The claim that needs a control is stage 1's ZERO. So: quantize a
            // DIFFERENT row and check the same bitwise comparator reports a
            // difference. If it cannot see two genuinely different rows apart,
            // its 0 above means nothing.
            float[] other = RandomFloats(new Random(0xC0FFEE + m + k), k, 1.0f);
            (long qDiffXX, long dsDiffXX) = QuantizeBothAndCompareAgainst(
                device, q1, qRows, bufX, bufXq1, bufXds1, bufXqR, bufXdsR, k, x, other);

            // Stage 2 needs the opposite control: is the gap a systematic kernel
            // difference, or just run-to-run nondeterminism? Same kernel twice
            // must be bitwise identical.
            long mmvqSelfDiff = MmvqSelfConsistency(device, mmvq, bufW, bufXq1, bufXds1, bufCv, bufCq, m, k);

            // Which of the two is RIGHT? A difference between two kernels says
            // nothing about that on its own, and "accept a bound" is only the
            // honest disposition if neither is systematically worse. Score both
            // against an f64 CPU oracle over the SAME Q8_0 weights and the SAME
            // dequantized Q8_1 activation, so the only thing that varies is the
            // order of the float accumulation.
            double[] oracle = OracleF64(weightsQ8, xqBytesOf(device, bufXq1, k), xdsOf(device, bufXds1, k), m, k);
            var cvF = new float[m];
            var cqF = new float[m];
            device.Download(bufCv, cvF);   // still holds MMVQ from the self-consistency run
            RunMmqInto(device, mmq, bufW, bufXq1, bufXds1, bufCq, m, k);
            device.Download(bufCq, cqF);
            (double rmsV, double maxV) = ScoreAgainst(oracle, cvF);
            (double rmsQ, double maxQ) = ScoreAgainst(oracle, cqF);

            sb.AppendLine($"  m={m,5} k={k,5}");
            sb.AppendLine($"    stage 1  quantizer xq words differing = {r.qDiff}/{QuantizeQ8_1Kernel.PackedBytes(k) / sizeof(float)}"
                        + $"   xds (d,s) floats differing = {r.dsDiff}/{QuantizeQ8_1Kernel.ScaleBytes(k) / sizeof(float)}");
            sb.AppendLine($"      control: same comparator on a DIFFERENT row -> xq {qDiffXX}, xds {dsDiffXX} differing"
                        + "  (must be > 0, else the zero above is unfalsifiable)");
            sb.AppendLine($"    stage 2  MMVQ vs MMQ differing = {r.mmDiff}/{m}"
                        + $"   maxAbs={r.mmMaxAbs:E3}  maxRel={r.mmMaxRel:E3}");
            sb.AppendLine($"      control: MMVQ vs MMVQ (same kernel twice) -> {mmvqSelfDiff}/{m} differing"
                        + "  (must be 0, else the gap above is nondeterminism not a kernel difference)");
            sb.AppendLine($"    vs f64 oracle   MMVQ rms={rmsV:E3} max={maxV:E3}   MMQ rms={rmsQ:E3} max={maxQ:E3}"
                        + $"   ratio(MMQ/MMVQ rms)={rmsQ / rmsV:F3}");

            Assert.True(qDiffXX > 0 && dsDiffXX > 0,
                $"the quantizer comparator cannot distinguish two different rows at m={m} k={k}; "
                + "its 0-differing verdict proves nothing.");
            Assert.True(mmvqSelfDiff == 0,
                $"MMVQ is not deterministic across two dispatches at m={m} k={k} ({mmvqSelfDiff} differing); "
                + "the MMVQ-vs-MMQ gap cannot be attributed to the kernel difference.");

            totalQuantDiff += r.qDiff + r.dsDiff;
            totalMatmulDiff += r.mmDiff;

            // Accuracy gates. Neither kernel is wrong — both sit ~1E-07 RMS from
            // an f64 accumulation of the identical quantities — so the MMVQ/MMQ
            // difference is two valid float orderings, not a defect, and there is
            // nothing to collapse. What IS worth holding is that neither drifts:
            Assert.True(rmsV < 1e-5 && maxV < 1e-4,
                $"MMVQ has drifted from the f64 oracle at m={m} k={k}: rms={rmsV:E3} max={maxV:E3}");
            Assert.True(rmsQ < 1e-5 && maxQ < 1e-4,
                $"MMQ has drifted from the f64 oracle at m={m} k={k}: rms={rmsQ:E3} max={maxQ:E3}");

            // ...and that the PREFILL path does not fall further behind the decode
            // path. MMQ is measured 1.65x / 1.73x / 2.37x the RMS error of MMVQ on
            // these three shapes — prefill is the less accurate of the two, which
            // is the reason the fix for #538 must NOT be "route n==1 through MMQ".
            // 5x leaves room for the measured spread while still catching a real
            // regression in the prefill accumulation.
            Assert.True(rmsQ < 5.0 * rmsV,
                $"MMQ (prefill) accuracy has regressed relative to MMVQ (decode) at m={m} k={k}: "
                + $"rms {rmsQ:E3} vs {rmsV:E3} (ratio {rmsQ / rmsV:F2}, bound 5.00)");

        }

        _out.WriteLine(sb.ToString());

        // Reported, not asserted, for the matmul: the disposition of a non-zero
        // reduction-order gap is a decision (collapse to one path vs accept a
        // bound), and MMVQ is the hot decode GEMV. The QUANTIZER is different:
        // the two arms must compute the same input, so that one is a gate.
        Assert.True(totalQuantDiff == 0,
            $"the two Q8_1 quantizers emit different bytes for the same row ({totalQuantDiff} differing); "
            + "the n==1 and n>1 matmul arms are not even seeing the same activation." + Environment.NewLine + sb);
    }

    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Control for stage 1: quantizes <paramref name="a"/> with the n==1 kernel
    /// and <paramref name="b"/> with the rows kernel, and returns how many words
    /// the SAME bitwise comparator reports. Two genuinely different rows must
    /// come back non-zero, or a 0 on identical input means nothing.
    /// </summary>
    private static (long xq, long xds) QuantizeBothAndCompareAgainst(
        VulkanDevice device, QuantizeQ8_1Kernel q1, QuantizeQ8_1RowsKernel qRows,
        VulkanDevice.Buffer bufX,
        VulkanDevice.Buffer bufXq1, VulkanDevice.Buffer bufXds1,
        VulkanDevice.Buffer bufXqR, VulkanDevice.Buffer bufXdsR,
        int k, float[] a, float[] b)
    {
        device.Upload(a, bufX);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q1.Record(ctx.CommandBuffer, bufX, bufXq1, bufXds1, k);
            ctx.SubmitAndWait();
        }
        device.Upload(b, bufX);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            qRows.Record(ctx.CommandBuffer, bufX, bufXqR, bufXdsR, 1, k);
            ctx.SubmitAndWait();
        }

        var xq1 = new float[QuantizeQ8_1Kernel.PackedBytes(k) / sizeof(float)];
        var xqR = new float[QuantizeQ8_1RowsKernel.PackedBytes(1, k) / sizeof(float)];
        var xds1 = new float[QuantizeQ8_1Kernel.ScaleBytes(k) / sizeof(float)];
        var xdsR = new float[QuantizeQ8_1RowsKernel.ScaleBytes(1, k) / sizeof(float)];
        device.Download(bufXq1, xq1);
        device.Download(bufXqR, xqR);
        device.Download(bufXds1, xds1);
        device.Download(bufXdsR, xdsR);

        long qd = 0, dsd = 0;
        for (int i = 0; i < xq1.Length; i++)
            if (BitConverter.SingleToInt32Bits(xq1[i]) != BitConverter.SingleToInt32Bits(xqR[i])) qd++;
        for (int i = 0; i < xds1.Length; i++)
            if (BitConverter.SingleToInt32Bits(xds1[i]) != BitConverter.SingleToInt32Bits(xdsR[i])) dsd++;
        return (qd, dsd);
    }

    /// <summary>
    /// Control for stage 2: runs MMVQ twice into two buffers. Must be bitwise
    /// identical, otherwise the MMVQ-vs-MMQ gap could be nondeterminism rather
    /// than a difference between the two kernels.
    /// </summary>
    private static long MmvqSelfConsistency(
        VulkanDevice device, MatMulQ8_0MmvqKernel mmvq,
        VulkanDevice.Buffer bufW, VulkanDevice.Buffer bufXq, VulkanDevice.Buffer bufXds,
        VulkanDevice.Buffer bufA, VulkanDevice.Buffer bufB, int m, int k)
    {
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufA, m, k);
            ctx.SubmitAndWait();
        }
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufB, m, k);
            ctx.SubmitAndWait();
        }
        var a = new float[m];
        var b = new float[m];
        device.Download(bufA, a);
        device.Download(bufB, b);
        long d = 0;
        for (int i = 0; i < m; i++)
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) d++;
        return d;
    }

    private static float[] xqBytesOf(VulkanDevice device, VulkanDevice.Buffer buf, int k)
    {
        var w = new float[QuantizeQ8_1Kernel.PackedBytes(k) / sizeof(float)];
        device.Download(buf, w);
        return w;
    }

    private static float[] xdsOf(VulkanDevice device, VulkanDevice.Buffer buf, int k)
    {
        var w = new float[QuantizeQ8_1Kernel.ScaleBytes(k) / sizeof(float)];
        device.Download(buf, w);
        return w;
    }

    private static void RunMmqInto(
        VulkanDevice device, MatMulQ8_0MmqKernel mmq,
        VulkanDevice.Buffer bufW, VulkanDevice.Buffer bufXq, VulkanDevice.Buffer bufXds,
        VulkanDevice.Buffer dst, int m, int k)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, dst, m, k, 1);
        ctx.SubmitAndWait();
    }

    /// <summary>
    /// f64 reference for one Q8_0 row-major weight matrix times one Q8_1 activation
    /// row, accumulated in double. Both GPU kernels compute exactly this quantity;
    /// they differ only in the order and precision of the accumulation, so the
    /// distance from this oracle says which ordering is more accurate.
    /// </summary>
    private static double[] OracleF64(byte[] weightsQ8, float[] xqWords, float[] xds, int m, int k)
    {
        int blocks = k / Q8_0GroupSize;
        int rowBytes = blocks * Q8_0BlockBytes;

        // Unpack the Q8_1 activation: xq is 4 int8 per 32-bit word; xds is (d, s)
        // per 32-element block.
        var xInt = new sbyte[k];
        for (int w = 0; w < xqWords.Length; w++)
        {
            int bits = BitConverter.SingleToInt32Bits(xqWords[w]);
            for (int b = 0; b < 4; b++) xInt[w * 4 + b] = (sbyte)((bits >> (b * 8)) & 0xFF);
        }

        var outp = new double[m];
        for (int row = 0; row < m; row++)
        {
            int baseOff = row * rowBytes;
            double acc = 0.0;
            for (int blk = 0; blk < blocks; blk++)
            {
                int off = baseOff + blk * Q8_0BlockBytes;
                double dW = (double)BitConverter.UInt16BitsToHalf(
                    (ushort)(weightsQ8[off] | (weightsQ8[off + 1] << 8)));
                double dX = xds[blk * 2];
                long dot = 0;
                for (int e = 0; e < Q8_0GroupSize; e++)
                    dot += (sbyte)weightsQ8[off + 2 + e] * (long)xInt[blk * Q8_0GroupSize + e];
                acc += dW * dX * dot;
            }
            outp[row] = acc;
        }
        return outp;
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

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int rowBytes = (k / Q8_0GroupSize) * Q8_0BlockBytes;
        var dst = new byte[m * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
        }
        return dst;
    }
}
