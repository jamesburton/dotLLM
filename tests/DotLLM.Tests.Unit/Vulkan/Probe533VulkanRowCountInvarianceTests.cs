using System.Text;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #533 — row-count invariance of every Vulkan op a chunked
/// prefill exercises.
/// </summary>
/// <remarks>
/// <para>
/// A row-independent op must produce bit-identical results for row <c>r</c>
/// whatever the total row count <c>n</c> of the dispatch (as long as
/// <c>r &lt; n</c>). Chunked prefill is exactly this: chunk <c>[0..3)</c> of a
/// 6-token prompt must write the same KV rows as rows 0..2 of a single
/// 6-token pass. #533 measured that it does not, with an even/odd parity
/// structure, on BOTH Q8_0 and F32 weights.
/// </para>
/// <para>
/// Each arm below dispatches the SAME input buffer at <c>n = 1..8</c> and
/// compares rows bitwise against the <c>n = 8</c> dispatch. The op that breaks
/// names itself. This is deliberately a BITWISE test, not a tolerance test —
/// the existing MMQ parity test tolerates <c>3e-2</c> against a CPU oracle,
/// which is the same magnitude class as the defect and is why it has been
/// invisible.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe533VulkanRowCountInvarianceTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int MaxN = 8;

    private readonly ITestOutputHelper _out;
    public Probe533VulkanRowCountInvarianceTests(ITestOutputHelper output) => _out = output;

    // ─────────────────────────────────────────────────────────────
    // Matmul family
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void MatMul_RowInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        const int m = 512;
        const int k = 2048;

        var rng = new Random(533);
        float[] weightsF32 = RandomFloats(rng, m * k, 0.1f);
        float[] inputB = RandomFloats(rng, MaxN * k, 1.0f);
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        var sb = new StringBuilder();
        sb.AppendLine($"=== #533 matmul row-count invariance (m={m} k={k}) ===");

        using var bufWq8 = device.Allocate(((long)weightsQ8.Length + 3) & ~3L);
        using var bufWf32 = device.Allocate((long)m * k * sizeof(float));
        using var bufB = device.Allocate((long)MaxN * k * sizeof(float));
        using var bufC = device.Allocate((long)MaxN * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufWq8);
        device.Upload(weightsF32, bufWf32);
        device.Upload(inputB, bufB);

        // --- F32 weights: matmul_f32.comp (this is what the #533 F32 arm runs) ---
        using (var f32 = MatMulF32Kernel.Create(device, spvDir))
        {
            Sweep(sb, "matmul_f32", m, n =>
            {
                f32.Launch(bufWf32, bufB, bufC, m, k, n);
                return DownloadRows(device, bufC, n, m);
            });
        }

        // --- Q8_0 scalar F32-in GEMM ---
        using (var gemm = MatMulQ8_0GemmKernel.Create(device, spvDir))
        {
            Sweep(sb, "q8_0_f32gemm", m, n =>
            {
                gemm.Launch(bufWq8, bufB, bufC, m, k, n);
                return DownloadRows(device, bufC, n, m);
            });
        }

        // --- Q8_0 coopmat GEMM ---
        MatMulQ8_0GemmCoopmatKernel? coop = null;
        try { coop = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
        catch (Exception ex) { sb.AppendLine($"  q8_0_coopmat: unavailable ({ex.GetType().Name})"); }
        if (coop is not null)
        {
            using (coop)
            {
                Sweep(sb, "q8_0_coopmat", m, n =>
                {
                    coop.Launch(bufWq8, bufB, bufC, m, k, n);
                    return DownloadRows(device, bufC, n, m);
                });
            }
        }

        // --- Q8_0 dp4a MMQ (quantize_q8_1_rows + matmul_q8_0_mmq) — the DEFAULT prefill path ---
        var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir);
        var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir);
        if (quant is null || mmq is null)
        {
            sb.AppendLine("  q8_0_mmq: unavailable (no integer-dot / spv)");
        }
        else
        {
            using (quant)
            using (mmq)
            using (var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(MaxN, k)))
            using (var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(MaxN, k)))
            {
                Sweep(sb, "q8_0_mmq (quant+mmq)", m, n =>
                {
                    using (var ctx = device.CreateSubmitContext())
                    {
                        ctx.Begin();
                        quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
                        KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                        mmq.Record(ctx.CommandBuffer, bufWq8, bufXq, bufXds, bufC, m, k, n);
                        ctx.SubmitAndWait();
                    }
                    return DownloadRows(device, bufC, n, m);
                });

                // The quantized activation itself — isolate quantize_q8_1_rows.
                int xqFloatsPerRow = k / 4;   // int8 [n,k] viewed as 4 bytes / "float"
                Sweep(sb, "quantize_q8_1_rows(xq bits)", xqFloatsPerRow, n =>
                {
                    using (var ctx = device.CreateSubmitContext())
                    {
                        ctx.Begin();
                        quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
                        ctx.SubmitAndWait();
                    }
                    var got = new float[(long)MaxN * xqFloatsPerRow];
                    device.Download(bufXq, got);
                    return got;
                });
            }
        }

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────
    // RMSNorm
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void RmsNorm_RowInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        const int k = 2048;
        var rng = new Random(5331);
        float[] input = RandomFloats(rng, MaxN * k, 1.0f);
        float[] weight = RandomFloats(rng, k, 1.0f);

        using var bufIn = device.Allocate((long)MaxN * k * sizeof(float));
        using var bufW = device.Allocate((long)k * sizeof(float));
        using var bufOut = device.Allocate((long)MaxN * k * sizeof(float));
        device.Upload(input, bufIn);
        device.Upload(weight, bufW);

        using var norm = RmsNormF32Kernel.Create(device, spvDir);
        var sb = new StringBuilder();
        sb.AppendLine($"=== #533 rmsnorm row-count invariance (k={k}) ===");
        Sweep(sb, "rmsnorm_f32", k, n =>
        {
            norm.Launch(bufIn, bufW, bufOut, n, k, 1e-5f);
            return DownloadRows(device, bufOut, n, k);
        });
        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────
    // RoPE
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Rope_RowInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        // Llama-3.2-1B: 32 q heads, 8 kv heads, headDim 64.
        const int numHeads = 32, numKvHeads = 8, headDim = 64;
        int qRow = numHeads * headDim, kRow = numKvHeads * headDim;

        var rng = new Random(5332);
        float[] q0 = RandomFloats(rng, MaxN * qRow, 1.0f);
        float[] k0 = RandomFloats(rng, MaxN * kRow, 1.0f);
        int[] positions = new int[MaxN];
        for (int i = 0; i < MaxN; i++) positions[i] = i;

        using var bufQ = device.Allocate((long)MaxN * qRow * sizeof(float));
        using var bufK = device.Allocate((long)MaxN * kRow * sizeof(float));
        using var bufPos = device.Allocate((long)MaxN * sizeof(int));
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.Cast<int, float>(positions), bufPos);

        using var rope = RopeF32Kernel.Create(device, spvDir);
        var sb = new StringBuilder();
        sb.AppendLine("=== #533 rope row-count invariance ===");
        Sweep(sb, "rope_f32 (Q)", qRow, n =>
        {
            device.Upload(q0, bufQ);
            device.Upload(k0, bufK);
            rope.Launch(bufQ, bufK, bufPos, n, numHeads, numKvHeads, headDim,
                        ropeDim: headDim, theta: 500000f);
            return DownloadRows(device, bufQ, n, qRow);
        });
        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────
    // Attention on the seqQ axis (NOT covered by #532's KV-length sweep)
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void Attention_SeqQInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        const int numHeads = 32, numKvHeads = 8, headDim = 64;
        const int seqKv = MaxN;
        int qRow = numHeads * headDim, kvRow = numKvHeads * headDim;

        var rng = new Random(5333);
        float[] q = RandomFloats(rng, MaxN * qRow, 1.0f);
        float[] kk = RandomFloats(rng, seqKv * kvRow, 1.0f);
        float[] vv = RandomFloats(rng, seqKv * kvRow, 1.0f);

        using var bufQ = device.Allocate((long)MaxN * qRow * sizeof(float));
        using var bufK = device.Allocate((long)seqKv * kvRow * sizeof(float));
        using var bufV = device.Allocate((long)seqKv * kvRow * sizeof(float));
        using var bufO = device.Allocate((long)MaxN * qRow * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(kk, bufK);
        device.Upload(vv, bufV);

        var sb = new StringBuilder();
        sb.AppendLine("=== #533 attention seqQ-count invariance (seqKv fixed = 8, posOff=0) ===");

        using (var dense = AttentionF32Kernel.Create(device, spvDir))
        {
            Sweep(sb, "attention_f32 (dense)", qRow, n =>
            {
                dense.Launch(bufQ, bufK, bufV, bufO, n, seqKv, numHeads, numKvHeads, headDim);
                return DownloadRows(device, bufO, n, qRow);
            });
        }

        var flash = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        if (flash is null) sb.AppendLine("  flash: unavailable");
        else
            using (flash)
            {
                Sweep(sb, "attention_flash_f32", qRow, n =>
                {
                    flash.Launch(bufQ, bufK, bufV, bufO, n, seqKv, numHeads, numKvHeads, headDim);
                    return DownloadRows(device, bufO, n, qRow);
                });
            }

        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// THE #533 ARM. Production chunked-prefill shape: <c>seqQ == seqKv == n</c>,
    /// <c>positionOffset == 0</c>. Under a causal mask query row <c>r</c> sees keys
    /// <c>0..r</c> whatever <c>n</c> is, so row <c>r</c> must be bit-identical for
    /// every <c>n &gt; r</c>. The previous arm held <c>seqKv</c> fixed at 8 and swept
    /// only <c>seqQ</c> — that is the axis #532 measured, and it is clean.
    /// </summary>
    [SkippableFact]
    public void Attention_SquarePrefillInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        const int numHeads = 32, numKvHeads = 8, headDim = 64;
        int qRow = numHeads * headDim, kvRow = numKvHeads * headDim;

        var rng = new Random(5334);
        float[] q = RandomFloats(rng, MaxN * qRow, 1.0f);
        float[] kk = RandomFloats(rng, MaxN * kvRow, 1.0f);
        float[] vv = RandomFloats(rng, MaxN * kvRow, 1.0f);

        using var bufQ = device.Allocate((long)MaxN * qRow * sizeof(float));
        using var bufK = device.Allocate((long)MaxN * kvRow * sizeof(float));
        using var bufV = device.Allocate((long)MaxN * kvRow * sizeof(float));
        using var bufO = device.Allocate((long)MaxN * qRow * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(kk, bufK);
        device.Upload(vv, bufV);

        var sb = new StringBuilder();
        sb.AppendLine("=== #533 attention SQUARE prefill invariance (seqQ == seqKv == n, posOff=0) ===");

        using (var dense = AttentionF32Kernel.Create(device, spvDir))
        {
            Sweep(sb, "attention_f32 (dense)", qRow, n =>
            {
                dense.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
                return DownloadRows(device, bufO, n, qRow);
            });
        }

        var flash = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        if (flash is null) sb.AppendLine("  flash scalar: unavailable");
        else
            using (flash)
            {
                Sweep(sb, "attention_flash_f32 (scalar)", qRow, n =>
                {
                    flash.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
                    return DownloadRows(device, bufO, n, qRow);
                });
            }

        var coop = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir);
        if (coop is null) sb.AppendLine("  flash coopmat: unavailable on this device");
        else
            using (coop)
            {
                Sweep(sb, "attention_flash_f32_coopmat (PRODUCTION DEFAULT)", qRow, n =>
                {
                    coop.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
                    return DownloadRows(device, bufO, n, qRow);
                });

                // Which axis carries the parity? rowsInTile (= seqQ) held at 8,
                // seqKv swept. Note the causal early-exit clamps kvEnd to
                // min(seqKv, lastPosQ+1), so in the square arm above tileLen == n
                // too — this arm separates rowsInTile from tileLen.
                Sweep(sb, "coopmat seqQ=8 FIXED, seqKv=n", qRow, n =>
                {
                    coop.Launch(bufQ, bufK, bufV, bufO, MaxN, n, numHeads, numKvHeads, headDim);
                    return DownloadRows(device, bufO, n, qRow);
                });
            }

        _out.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Does the effect survive at realistic prompt lengths, or is it a
    /// single-KV-tile artifact? BC = 64, so a prefill of length L ends in a tile
    /// of length <c>((L-1) % 64) + 1</c>. Sweeps L across a tile boundary with a
    /// long, even reference.
    /// </summary>
    [SkippableFact]
    public void Attention_Coopmat_LongPrefillInvariance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var coop = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir);
        Skip.If(coop is null, "coopmat FA unavailable on this device.");

        const int numHeads = 32, numKvHeads = 8, headDim = 64, refN = 160;
        int qRow = numHeads * headDim, kvRow = numKvHeads * headDim;

        var rng = new Random(5335);
        float[] q = RandomFloats(rng, refN * qRow, 1.0f);
        float[] kk = RandomFloats(rng, refN * kvRow, 1.0f);
        float[] vv = RandomFloats(rng, refN * kvRow, 1.0f);

        using var bufQ = device.Allocate((long)refN * qRow * sizeof(float));
        using var bufK = device.Allocate((long)refN * kvRow * sizeof(float));
        using var bufV = device.Allocate((long)refN * kvRow * sizeof(float));
        using var bufO = device.Allocate((long)refN * qRow * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(kk, bufK);
        device.Upload(vv, bufV);

        using (coop)
        {
            float[] Run(int n)
            {
                coop!.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
                var all = new float[(long)refN * qRow];
                device.Download(bufO, all);
                return all;
            }

            float[] reference = Run(refN);
            var sb = new StringBuilder();
            sb.AppendLine($"=== #533 coopmat FA long-prefill invariance (reference L={refN}) ===");
            foreach (int n in new[] { 61, 62, 63, 64, 65, 66, 67, 68, 126, 127, 128, 129, 130 })
            {
                float[] got = Run(n);
                long diff = 0; float maxAbs = 0; int firstRow = -1, lastRow = -1;
                for (int r = 0; r < n; r++)
                    for (int i = 0; i < qRow; i++)
                    {
                        int idx = r * qRow + i;
                        if (BitConverter.SingleToInt32Bits(reference[idx]) != BitConverter.SingleToInt32Bits(got[idx]))
                        {
                            diff++;
                            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[idx] - got[idx]));
                            if (firstRow < 0) firstRow = r;
                            lastRow = r;
                        }
                    }
                sb.AppendLine($"  L={n,3} ({(n % 2 == 0 ? "even" : "odd ")}) lastTileLen={((n - 1) % 64) + 1,2}: " +
                              $"differing={diff,7} maxAbs={maxAbs:E3} rows[{firstRow}..{lastRow}]");
            }
            _out.WriteLine(sb.ToString());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Dispatches <paramref name="run"/> at n = MaxN then n = 1..MaxN-1 and
    /// reports, per n, how many of the first n rows differ bitwise from the
    /// n = MaxN result.
    /// </summary>
    private static void Sweep(StringBuilder sb, string label, int rowLen, Func<int, float[]> run)
    {
        float[] reference = run(MaxN);
        sb.AppendLine($"  {label}:");
        for (int n = 1; n < MaxN; n++)
        {
            float[] got = run(n);
            long diff = 0; float maxAbs = 0; int firstRow = -1;
            for (int r = 0; r < n; r++)
                for (int i = 0; i < rowLen; i++)
                {
                    int idx = r * rowLen + i;
                    if (BitConverter.SingleToInt32Bits(reference[idx]) != BitConverter.SingleToInt32Bits(got[idx]))
                    {
                        diff++;
                        maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[idx] - got[idx]));
                        if (firstRow < 0) firstRow = r;
                    }
                }
            sb.AppendLine($"    n={n} ({(n % 2 == 0 ? "even" : "odd ")}): differing={diff,8}/{(long)n * rowLen,-8} " +
                          $"maxAbs={maxAbs:E3} firstDiffRow={firstRow}");
        }
    }

    private static float[] DownloadRows(VulkanDevice device, VulkanDevice.Buffer buf, int n, int rowLen)
    {
        var all = new float[(long)MaxN * rowLen];
        device.Download(buf, all);
        return all;
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
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
