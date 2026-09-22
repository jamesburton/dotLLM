using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan PQ2_0 (PrismML Bonsai ternary) GEMV kernel against a
/// scalar ground-truth reference. The reference is computed directly from the unpacked ternary
/// values with each 128-element group scaled by its own fp16 scale
/// (<c>y[r] = Σ_g scale(r,g) · Σ_{c∈g} ternary[r,c] · x[c]</c>), so a passing test proves the GPU
/// kernel decodes the PQ2_0 bit-layout (byte <c>b</c> packs 4 CONSECUTIVE group-relative
/// positions {4b,4b+1,4b+2,4b+3} at ASCENDING bit offsets {0,2,4,6}, value = code−1; see issue
/// #271 — this is PrismML's real format, not I2_S's strided {gp,gp+32,gp+64,gp+96} interleave),
/// reads each group's leading fp16 scale from its 34-byte group header, applies it per-group (not
/// once at the end, unlike I2_S), and reduces correctly. Mirrors
/// <c>VulkanMatMulI2SGemvF32KernelTests</c>.
/// </summary>
/// <remarks>
/// Tolerance — PQ2_0 codes are exact ternary (no per-element quant error beyond the fp16 group
/// scale's own rounding), so divergence from the sequential CPU reference is fp16-scale rounding
/// plus reduction order (GPU 128-thread tree reduce vs scalar sum). The same 5e-3 / 1e-3
/// tolerances as the I2_S / K-quant GEMV parity tests cover it.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulPQ2_0GemvF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    [SkippableTheory]
    [InlineData(1, 128)]      // one group -> 34 bytes packed
    [InlineData(8, 128)]
    [InlineData(4, 256)]      // two groups, distinct per-group scales
    [InlineData(16, 768)]
    [InlineData(2048, 256)]
    [InlineData(2560, 2560)]
    [InlineData(576, 1024)]
    public void Launch_MatchesScalarReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_50 ^ (m * 7 + k * 11));

        int groups = k / GroupSize;

        // Random ternary weights {-1,0,+1}, a random fp16 scale per (row, group), and a
        // random activation vector.
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsPQ2_0 = PackPQ2_0(ternary, scales, m, k);

        // Ground-truth reference: y[r] = Σ_g scale(r,g) · Σ_{c∈g} ternary[r,c] · x[c].
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            int rowBase = r * k;
            for (int g = 0; g < groups; g++)
            {
                double groupScale = (float)scales[r * groups + g];
                double groupAcc = 0;
                int groupBase = g * GroupSize;
                for (int c = 0; c < GroupSize; c++)
                    groupAcc += ternary[rowBase + groupBase + c] * (double)x[groupBase + c];
                acc += groupScale * groupAcc;
            }
            expected[r] = (float)acc;
        }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsPQ2_0.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsPQ2_0), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertClose(expected, actual, m, k);
    }

    /// <summary>
    /// Issue #446 — the offset-taking <c>Record</c> overload that makes a real looped GEMV
    /// possible: <c>n</c> dispatches over one <c>[n, K]</c> activation buffer into one
    /// <c>[n, M]</c> output buffer, each addressing its own row through push constants because
    /// <see cref="VulkanDevice.Buffer"/> has no offset view.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This test discriminates, which the zero-offset cases cannot.</b> Every token gets a
    /// DIFFERENT activation row, so a kernel that ignored <c>xOff</c> would compute token 0's
    /// answer <c>n</c> times and every row but the first would fail; a kernel that ignored
    /// <c>yOff</c> would leave rows 1..n-1 at zero and overwrite row 0. Both are the exact
    /// failure modes the timing proxy in <c>VulkanPQ2_0GemmBench</c> was silently exhibiting,
    /// which is why that proxy was never shippable.
    /// </para>
    /// <para>
    /// Two dispatches also share one command buffer with no barrier between them, matching what
    /// <see cref="PQ2_0SmallNDispatch"/> records — the writes are to disjoint <c>y</c> ranges, so
    /// if that reasoning were wrong this test would be where it shows up.
    /// </para>
    /// </remarks>
    [SkippableTheory]
    [InlineData(2, 5, 256)]      // ragged m, several tokens
    [InlineData(4, 4, 128)]      // exactly the shipped DefaultGemvLoopMaxN
    [InlineData(3, 320, 640)]
    public void LoopedGemvWithRowOffsets_MatchesScalarReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x2A_46 ^ (n * 13 + m * 7 + k * 11));
        int groups = k / GroupSize;

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        Half[] scales = new Half[m * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        // One [n, k] activation batch — every token's row is distinct, so an ignored xOff shows.
        float[] x = new float[(long)n * k];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsPQ2_0 = PackPQ2_0(ternary, scales, m, k);

        float[] expected = new float[(long)n * m];
        for (int t = 0; t < n; t++)
        {
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                int rowBase = r * k;
                for (int g = 0; g < groups; g++)
                {
                    double groupScale = (float)scales[r * groups + g];
                    double groupAcc = 0;
                    int groupBase = g * GroupSize;
                    for (int c = 0; c < GroupSize; c++)
                        groupAcc += ternary[rowBase + groupBase + c] * (double)x[t * k + groupBase + c];
                    acc += groupScale * groupAcc;
                }
                expected[t * m + r] = (float)acc;
            }
        }

        using var device = VulkanDevice.Create();
        using var kernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        using var bufW = device.Allocate(((long)weightsPQ2_0.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)n * k * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsPQ2_0), bufW);
        device.Upload(x, bufX);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            for (int t = 0; t < n; t++)
                kernel.Record(ctx.CommandBuffer, bufW, bufX, bufY, m, k,
                    xOffsetElements: t * k, yOffsetElements: t * m);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[(long)n * m];
        device.Download(bufY, actual);

        for (int t = 0; t < n; t++)
            AssertClose(expected.AsSpan(t * m, m).ToArray(), actual.AsSpan(t * m, m).ToArray(), m, k);
    }

    /// <summary>
    /// <see cref="PQ2_0SmallNDispatch"/> must produce the same answer whichever kernel its
    /// threshold picks — the point of the crossover is that it is a pure performance choice.
    /// </summary>
    /// <remarks>
    /// Runs the same <c>[n, K]</c> batch through the dispatcher and through the GEMM directly and
    /// requires agreement, at every <c>n</c> that straddles the shipped threshold of 4. Without
    /// this, an off-by-one in the loop's offsets would only ever surface as a quality regression
    /// on a real model.
    /// </remarks>
    [SkippableTheory]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(8)]
    public void SmallNDispatch_AgreesWithTheGemmItReplaces(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int M = 260, K = 384;       // ragged M so the GEMM's boundary path runs too
        var rng = new Random(0x2A_47 ^ n);
        int groups = K / GroupSize;

        sbyte[] ternary = new sbyte[M * K];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        Half[] scales = new Half[M * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);
        float[] x = new float[(long)n * K];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsPQ2_0 = PackPQ2_0(ternary, scales, M, K);

        using var device = VulkanDevice.Create();
        using var gemv = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);
        using var gemm = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir);

        using var bufW = device.Allocate(((long)weightsPQ2_0.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)n * K * sizeof(float));
        using var bufDispatch = device.Allocate((long)n * M * sizeof(float));
        using var bufGemm = device.Allocate((long)n * M * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsPQ2_0), bufW);
        device.Upload(x, bufX);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            PQ2_0SmallNDispatch.Record(ctx.CommandBuffer, gemv, gemm, bufW, bufX, bufDispatch, M, K, n);
            ctx.SubmitAndWait();
        }
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gemm.Record(ctx.CommandBuffer, bufW, bufX, bufGemm, M, K, n);
            ctx.SubmitAndWait();
        }

        float[] viaDispatch = new float[(long)n * M];
        float[] viaGemm = new float[(long)n * M];
        device.Download(bufDispatch, viaDispatch);
        device.Download(bufGemm, viaGemm);

        for (int t = 0; t < n; t++)
            AssertClose(viaGemm.AsSpan(t * M, M).ToArray(), viaDispatch.AsSpan(t * M, M).ToArray(), M, K);
    }

    /// <summary>
    /// Issue #470 — the multi-column GEMV must match the per-token loop it replaces to within
    /// float rounding, for every column count 1..8, including the runtime tails (3, 5, 6, 7) that
    /// run a wider compiled variant with dead columns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why not bit-exact.</b> The multi-column shader keeps the single-column kernel's
    /// per-lane partition and tree reduce, so each column performs the same operations in the
    /// same order. Measured on gfx1151, though, the two pipelines disagree by one ULP on some
    /// outputs (0.23716736 vs 0.23716734): the driver fuses multiply-adds differently in the two
    /// shaders. The tolerance here, 1e-5 absolute plus 1e-5 relative on outputs of magnitude
    /// ~0.1-1, is 500x tighter than the scalar-reference tests. Every bug this test exists to
    /// catch (a wrong x row, an ignored offset, a column written to the wrong y row) moves an
    /// output by the full size of a dot product.
    /// </para>
    /// <para>
    /// Every token row is distinct, so reading the wrong x row fails. Nonzero base offsets make
    /// ignoring <c>xOff</c>/<c>yOff</c> fail. The output buffer is pre-filled with a sentinel
    /// that must survive past the last live column, so writing a dead column fails. K = 384 gives
    /// an odd group count, so groups straddle uint words differently row to row. M = 67 is ragged.
    /// </para>
    /// </remarks>
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    public void RecordColumns_MatchesTheLoopedGemv(int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int M = 67, K = 384;
        const int XBase = 3 * K, YBase = 2 * M;       // nonzero base offsets
        const float Sentinel = -12345.5f;
        var rng = new Random(0x2A_70 ^ n);
        int groups = K / GroupSize;

        sbyte[] ternary = new sbyte[M * K];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        Half[] scales = new Half[M * groups];
        for (int i = 0; i < scales.Length; i++) scales[i] = (Half)(rng.NextSingle() * 0.05f + 0.01f);

        // One extra activation row past the batch, so a clamp or stride bug that reads row n
        // gets real data, not zeros.
        long xLen = XBase + (long)(n + 1) * K;
        float[] x = new float[xLen];
        for (int i = 0; i < x.Length; i++) x[i] = rng.NextSingle() * 2f - 1f;

        byte[] weightsPQ2_0 = PackPQ2_0(ternary, scales, M, K);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        long yLen = YBase + (long)(MatMulPQ2_0GemvF32Kernel.MaxColumns + 1) * M;
        float[] sentinel = new float[yLen];
        Array.Fill(sentinel, Sentinel);

        using var bufW = device.Allocate(((long)weightsPQ2_0.Length + 3) & ~3L);
        using var bufX = device.Allocate(xLen * sizeof(float));
        using var bufLoop = device.Allocate(yLen * sizeof(float));
        using var bufCols = device.Allocate(yLen * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsPQ2_0), bufW);
        device.Upload(x, bufX);
        device.Upload(sentinel, bufLoop);
        device.Upload(sentinel, bufCols);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            for (int t = 0; t < n; t++)
                kernel.Record(ctx.CommandBuffer, bufW, bufX, bufLoop, M, K,
                    xOffsetElements: XBase + t * K, yOffsetElements: YBase + t * M);
            ctx.SubmitAndWait();
        }
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordColumns(ctx.CommandBuffer, bufW, bufX, bufCols, M, K, n,
                xOffsetElements: XBase, yOffsetElements: YBase);
            ctx.SubmitAndWait();
        }

        float[] viaLoop = new float[yLen];
        float[] viaCols = new float[yLen];
        device.Download(bufLoop, viaLoop);
        device.Download(bufCols, viaCols);

        for (long i = 0; i < yLen; i++)
        {
            bool live = i >= YBase && i < YBase + (long)n * M;
            if (!live)
                Assert.True(viaCols[i] == Sentinel, $"element {i} outside the {n} live columns was written: {viaCols[i]}");
            else
                Assert.True(MathF.Abs(viaLoop[i] - viaCols[i]) <= 1e-5f + 1e-5f * MathF.Abs(viaLoop[i]),
                    $"n={n} column {(i - YBase) / M} row {(i - YBase) % M}: loop {viaLoop[i]:R} vs multi-column {viaCols[i]:R}");
        }
    }

    /// <summary>
    /// Packs a row-major <c>[m, k]</c> ternary matrix + per-(row,group) fp16 scales into the
    /// PQ2_0 byte layout the kernel decodes: each 128-element group is 34 bytes — a leading
    /// little-endian fp16 scale, then 32 bytes where byte <c>b</c> holds 4 CONSECUTIVE
    /// group-relative positions {4b, 4b+1, 4b+2, 4b+3} at ASCENDING bit offsets {0, 2, 4, 6}
    /// (stored code = value + 1 ∈ {0,1,2}). Row stride = (k/128)·34 bytes. See issue #271 —
    /// this is PrismML's real format, verified byte-for-byte against their reference
    /// dequantize_row_q2_0, not I2_S's strided {gp,gp+32,gp+64,gp+96} interleave this helper
    /// used to (wrongly) assume.
    /// </summary>
    private static byte[] PackPQ2_0(sbyte[] ternary, Half[] scales, int m, int k)
    {
        int groups = k / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)m * rowBytes];
        for (int r = 0; r < m; r++)
        {
            int rowBase = r * k;
            int rowByteBase = r * rowBytes;
            for (int g = 0; g < groups; g++)
            {
                int groupByteBase = rowByteBase + g * GroupBytes;
                BitConverter.GetBytes(scales[r * groups + g]).CopyTo(buf, groupByteBase);
                int codeBase = groupByteBase + 2;
                int groupElemBase = g * GroupSize;
                for (int p = 0; p < GroupSize; p++)
                {
                    int code = ternary[rowBase + groupElemBase + p] + 1;   // {-1,0,1} -> {0,1,2}
                    int byteInGroup = p / 4;
                    int shift = 2 * (p % 4);                               // 4b+0->0, 4b+1->2, 4b+2->4, 4b+3->6
                    buf[codeBase + byteInGroup] |= (byte)(code << shift);
                }
            }
        }
        return buf;
    }

    private static void AssertClose(float[] expected, float[] actual, int m, int k)
    {
        Assert.Equal(m, actual.Length);
        for (int r = 0; r < m; r++)
        {
            float diff = MathF.Abs(expected[r] - actual[r]);
            float tol = AbsTol + RelTol * MathF.Abs(expected[r]);
            Assert.True(diff <= tol,
                $"row {r} (m={m}, k={k}): expected {expected[r]:G9}, got {actual[r]:G9}, |Δ|={diff:G9} > tol {tol:G9}");
        }
    }
}
