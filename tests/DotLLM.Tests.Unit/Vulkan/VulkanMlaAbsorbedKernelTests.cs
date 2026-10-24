using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the absorbed-MLA kernels (issue #913): per-head Q8_0 GEMM (+ q_pe rope pack), latent-cache row
/// builder, and attention over the latent cache (unsplit and split-KV + merge). References are self-contained scalar CPU
/// implementations reading the SAME Q8_0 bytes the GPU sees. Shapes are chosen to defeat degenerate coincidences: head
/// counts that are not a multiple of the row tile (3, 5), row counts not a multiple of 128, token counts not a multiple of 8,
/// multi-tile KV lengths with a non-zero position offset, and causal masking inside a tile.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMlaAbsorbedKernelTests
{
    private const int GroupSize = 32;
    private const int BlockBytes = 34;

    // ───────────────────────── head GEMM ─────────────────────────

    [SkippableTheory]
    [InlineData(200, 96, 3, 11, false)]     // M not a multiple of 128, T not a multiple of 8, odd head count
    [InlineData(512, 192, 2, 1, false)]     // k_b shape at decode
    [InlineData(256, 512, 2, 9, false)]     // v_b shape, 9 tokens (one full token tile + 1)
    [InlineData(512, 192, 5, 17, true)]     // absorb + rope pack, 5 heads, 3 token tiles
    [InlineData(130, 64, 3, 8, true)]       // M just over one row tile
    public void HeadGemm_MatchesCpuReference(int m, int k, int numHeads, int seqLen, bool rope)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int RopeDim = 64;
        const float Theta = 10000f;
        var rng = new Random(0x913 + m + k * 3 + numHeads * 7 + seqLen);

        // x head row = [k contraction floats | (rope ? 64 q_pe floats : 0)]
        int xHead = k + (rope ? RopeDim : 0);
        int outHead = m + (rope ? RopeDim : 0);
        float[] x = Rand(rng, seqLen * numHeads * xHead, 1f);
        float[] wF = Rand(rng, numHeads * m * k, 0.2f);
        byte[] w = QuantizeRows(wF, numHeads * m, k);
        int[] pos = Enumerable.Range(7, seqLen).ToArray();     // non-zero start so a missing position offset shows

        float[] expected = new float[seqLen * numHeads * outHead];
        for (int t = 0; t < seqLen; t++)
            for (int h = 0; h < numHeads; h++)
            {
                for (int r = 0; r < m; r++)
                    expected[t * numHeads * outHead + h * outHead + r] =
                        DotQ8(w, (h * m + r), k, x, t * numHeads * xHead + h * xHead);
                if (!rope) continue;
                for (int p = 0; p < RopeDim / 2; p++)
                {
                    double freq = 1.0 / Math.Pow(Theta, 2.0 * p / RopeDim);
                    double ang = pos[t] * freq;
                    float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                    int src = t * numHeads * xHead + h * xHead + k + 2 * p;
                    int dst = t * numHeads * outHead + h * outHead + m + 2 * p;
                    expected[dst] = x[src] * c - x[src + 1] * s;
                    expected[dst + 1] = x[src] * s + x[src + 1] * c;
                }
            }

        using var device = VulkanDevice.Create();
        using var kernel = MlaHeadGemmQ8Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(Padded(w.Length));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));
        using var bufPos = device.Allocate(Math.Max(16, pos.Length * 4));
        device.Upload(PadBytes(w), bufW);
        device.Upload(x, bufX);
        device.Upload(new float[expected.Length], bufOut);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pos.AsSpan()), bufPos);

        long before = MlaAbsorbedKernelBase.TotalDispatches;
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.Record(ctx.CommandBuffer, bufW, bufX, bufOut, bufPos, m, k, numHeads, seqLen,
                numHeads * xHead, xHead, numHeads * outHead, outHead,
                rope ? RopeDim : 0, k, m, Theta);
            ctx.SubmitAndWait();
        }
        Assert.Equal(before + 1, MlaAbsorbedKernelBase.TotalDispatches);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);
        AssertClose(expected, actual, 1e-3f, 2e-3f, $"headgemm m={m} k={k} h={numHeads} t={seqLen} rope={rope}");
    }

    // ───────────────────────── kv prep ─────────────────────────

    [SkippableTheory]
    [InlineData(1, 0)]
    [InlineData(5, 0)]
    [InlineData(5, 11)]     // non-zero cache row offset
    public void KvPrep_MatchesCpuReference(int seqLen, int rowOffset)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int Lora = 512, Rope = 64, C = Lora + Rope;
        const float Eps = 1e-5f, Theta = 1e6f;
        var rng = new Random(0x914 + seqLen);
        float[] kvA = Rand(rng, seqLen * C, 2f);
        float[] nw = Rand(rng, Lora, 1f);
        int[] pos = Enumerable.Range(rowOffset, seqLen).ToArray();
        int rows = rowOffset + seqLen + 3;
        float[] expected = new float[rows * C];
        Array.Fill(expected, -7f);                     // untouched rows must stay untouched
        for (int t = 0; t < seqLen; t++)
        {
            double ss = 0;
            for (int i = 0; i < Lora; i++) ss += (double)kvA[t * C + i] * kvA[t * C + i];
            float rinv = (float)(1.0 / Math.Sqrt(ss / Lora + Eps));
            int o = (rowOffset + t) * C;
            for (int i = 0; i < Lora; i++) expected[o + i] = kvA[t * C + i] * rinv * nw[i];
            for (int p = 0; p < Rope / 2; p++)
            {
                double ang = pos[t] * (1.0 / Math.Pow(Theta, 2.0 * p / Rope));
                float c = (float)Math.Cos(ang), s = (float)Math.Sin(ang);
                float v0 = kvA[t * C + Lora + 2 * p], v1 = kvA[t * C + Lora + 2 * p + 1];
                expected[o + Lora + 2 * p] = v0 * c - v1 * s;
                expected[o + Lora + 2 * p + 1] = v0 * s + v1 * c;
            }
        }

        using var device = VulkanDevice.Create();
        using var kernel = MlaKvPrepKernel.Create(device, spvDir);
        using var bufKv = device.Allocate((long)kvA.Length * 4);
        using var bufW = device.Allocate((long)nw.Length * 4);
        using var bufPos = device.Allocate(Math.Max(16, pos.Length * 4));
        using var bufC = device.Allocate((long)expected.Length * 4);
        device.Upload(kvA, bufKv);
        device.Upload(nw, bufW);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(pos.AsSpan()), bufPos);
        var init = new float[expected.Length];
        Array.Fill(init, -7f);
        device.Upload(init, bufC);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.Record(ctx.CommandBuffer, bufKv, bufW, bufPos, bufC, seqLen, rowOffset, Lora, Rope, Eps, Theta);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[expected.Length];
        device.Download(bufC, actual);
        AssertClose(expected, actual, 1e-3f, 2e-4f, $"kvprep t={seqLen} off={rowOffset}");
    }

    // ───────────────────────── attention ─────────────────────────

    [SkippableTheory]
    //   seqQ seqKv heads posOff splits
    [InlineData(1, 1, 5, 0, 1)]            // first decode token
    [InlineData(1, 200, 5, 199, 1)]        // decode, 4 key tiles, 5 heads (one partial row tile), unsplit
    [InlineData(1, 200, 5, 199, 3)]        // same, split-KV + merge
    [InlineData(1, 777, 20, 776, 12)]      // GLM-4.7-Flash head count, many splits
    [InlineData(3, 90, 5, 87, 2)]          // tiny verify chunk: 3 tokens, causal inside the chunk, split
    [InlineData(70, 70, 3, 0, 1)]          // prefill, causal ramp across tile boundaries
    [InlineData(37, 150, 20, 113, 1)]      // chunked prefill with history, 20 heads (rows straddle tokens)
    public void Attention_MatchesCpuReference(int seqQ, int seqKv, int numHeads, int posOff, int numSplits)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int Lat = 512, C = 576;
        const float Scale = 0.0625f;
        var rng = new Random(0x915 + seqQ * 13 + seqKv + numHeads * 31);
        float[] q = Rand(rng, seqQ * numHeads * C, 1f);
        float[] cache = Rand(rng, seqKv * C, 1f);
        float[] expected = CpuAbsorbedAttention(q, cache, seqQ, seqKv, numHeads, posOff, Scale);

        using var device = VulkanDevice.Create();
        using var kernel = MlaAbsorbedAttentionKernel.Create(device, spvDir);
        int splitRows = Math.Max(4, seqQ) * numHeads;
        using var bufQ = device.Allocate((long)q.Length * 4);
        using var bufC = device.Allocate((long)cache.Length * 4 + 4096);
        using var bufOut = device.Allocate((long)expected.Length * 4);
        using var bufPO = device.Allocate((long)numSplits * splitRows * Lat * 4);
        using var bufPML = device.Allocate((long)numSplits * splitRows * 2 * 4);
        device.Upload(q, bufQ);
        device.Upload(cache, bufC);
        device.Upload(new float[expected.Length], bufOut);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.Record(ctx.CommandBuffer, bufQ, bufC, bufOut, bufPO, bufPML, seqQ, seqKv, numHeads, posOff, Scale,
                numSplits, splitRows);
            if (numSplits > 1)
            {
                KernelSupportBarrier(ctx.CommandBuffer);
                kernel.RecordMerge(ctx.CommandBuffer, bufPO, bufPML, bufOut, seqQ * numHeads, numSplits, splitRows);
            }
            ctx.SubmitAndWait();
        }
        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);
        AssertClose(expected, actual, 2e-3f, 2e-4f, $"attn q={seqQ} kv={seqKv} h={numHeads} off={posOff} splits={numSplits}");
    }

    private static void KernelSupportBarrier(nint cmd)
    {
        // internal helper; reflection keeps the test project free of InternalsVisibleTo assumptions
        var t = typeof(MlaAbsorbedKernelBase).Assembly.GetType("DotLLM.Vulkan.Kernels.KernelSupport")!;
        t.GetMethod("ComputeToComputeBarrier", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic
            | System.Reflection.BindingFlags.Public)!.Invoke(null, [cmd]);
    }

    private static float[] CpuAbsorbedAttention(float[] q, float[] c, int seqQ, int seqKv, int heads, int posOff, float scale)
    {
        const int Lat = 512, C = 576;
        var o = new float[seqQ * heads * Lat];
        var sc = new double[seqKv];
        for (int t = 0; t < seqQ; t++)
            for (int h = 0; h < heads; h++)
            {
                int qb = (t * heads + h) * C;
                double mx = double.NegativeInfinity;
                for (int s = 0; s < seqKv; s++)
                {
                    if (s > posOff + t) { sc[s] = double.NegativeInfinity; continue; }
                    double d = 0;
                    for (int i = 0; i < C; i++) d += (double)q[qb + i] * c[s * C + i];
                    sc[s] = d * scale;
                    mx = Math.Max(mx, sc[s]);
                }
                double z = 0;
                for (int s = 0; s < seqKv; s++) { sc[s] = double.IsNegativeInfinity(sc[s]) ? 0 : Math.Exp(sc[s] - mx); z += sc[s]; }
                for (int i = 0; i < Lat; i++)
                {
                    double a = 0;
                    for (int s = 0; s < seqKv; s++) a += sc[s] * c[s * C + i];
                    o[(t * heads + h) * Lat + i] = (float)(a / z);
                }
            }
        return o;
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float[] Rand(Random rng, int n, float range)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * range);
        return a;
    }

    private static unsafe byte[] QuantizeRows(float[] src, int rows, int k)
    {
        int rowBytes = k / GroupSize * BlockBytes;
        var dst = new byte[rows * rowBytes];
        fixed (float* s = src)
        fixed (byte* d = dst)
            for (int r = 0; r < rows; r++)
                MatMul.QuantizeF32ToQ8_0(s + (long)r * k, d + (long)r * rowBytes, k);
        return dst;
    }

    private static unsafe float DotQ8(byte[] w, int row, int k, float[] x, int xOff)
    {
        int rowBytes = k / GroupSize * BlockBytes;
        float sum = 0;
        fixed (byte* wp = w)
        {
            byte* rb = wp + (long)row * rowBytes;
            for (int b = 0; b < k / GroupSize; b++)
            {
                byte* blk = rb + b * BlockBytes;
                float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(blk);
                sbyte* qs = (sbyte*)(blk + 2);
                float bs = 0;
                for (int j = 0; j < GroupSize; j++) bs += qs[j] * x[xOff + b * GroupSize + j];
                sum += d * bs;
            }
        }
        return sum;
    }

    private const int Poison = 4096;
    private static long Padded(int bytes) => (((long)bytes + 3) & ~3L) + Poison;
    private static byte[] PadBytes(byte[] w)
    {
        var p = new byte[Padded(w.Length)];
        w.CopyTo(p, 0);
        p.AsSpan(w.Length).Fill(0x5A);     // an over-read of the last block shows up as a wrong result
        return p;
    }

    private static void AssertClose(float[] e, float[] a, float rel, float abs, string what)
    {
        Assert.Equal(e.Length, a.Length);
        int bad = 0; float worst = 0; int wi = -1;
        for (int i = 0; i < e.Length; i++)
        {
            float err = MathF.Abs(e[i] - a[i]);
            if (err > abs + rel * MathF.Abs(e[i]))
            {
                bad++;
                if (err > worst) { worst = err; wi = i; }
            }
        }
        Assert.True(bad == 0, $"{what}: {bad}/{e.Length} elements out of tolerance; worst |err|={worst:G4} at {wi} (exp {(wi >= 0 ? e[wi] : 0)}, got {(wi >= 0 ? a[wi] : 0)})");
    }
}
