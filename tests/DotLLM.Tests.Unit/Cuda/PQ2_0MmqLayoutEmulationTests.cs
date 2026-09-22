using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-only emulation of the packed PQ2_0 prefill GEMM (issue #490,
/// <c>native/kernels/pq2_0_mmq_dp4a.cu</c>), ported index formula for index formula: the same grid
/// order, the same <c>threadIdx → (tIdM, tIdN)</c> split, the same cooperative staging loops into the
/// same shared arrays (<c>wsm[block][row]</c> transposed, <c>xsm[col][word]</c>), the same
/// clamp-rows / zero-fill-dead-columns tail rules, and the same per-thread dp4a inner loop.
/// </summary>
/// <remarks>
/// <para>
/// This runs without a GPU, so a transposed shared index, an off-by-one activation chunk, a dropped
/// K-group or a wrong tail mask is caught before the PTX exists — the only executable check available
/// on a machine with no nvcc and no NVIDIA card. It is also the exact-integer oracle the GPU parity
/// test compares against, so kernel and oracle cannot drift apart.
/// </para>
/// <para>
/// Both instantiations are covered: the wide default tile (BN=32, TM=TN=4, BM=128) and the narrow one
/// for <c>S ≤ 16</c> (BN=16, TN=2, TM=8, BM=256). Shapes deliberately make every dimension ragged —
/// <c>n</c> not a multiple of BM, <c>S</c> not a multiple of BN — because a tile-aligned shape cannot
/// see a tail bug.
/// </para>
/// </remarks>
public sealed class PQ2_0MmqLayoutEmulationTests
{
    private readonly ITestOutputHelper _out;
    public PQ2_0MmqLayoutEmulationTests(ITestOutputHelper output) => _out = output;

    public const int TileWide = 32;
    public const int TileNarrow = 16;

    public static TheoryData<int, int, int, int> Shapes()
    {
        var data = new TheoryData<int, int, int, int>();   // columns, n, k, tile
        foreach (int tile in new[] { TileWide, TileNarrow })
        {
            data.Add(9, 37, 128, tile);      // every dimension a tail; one K group
            data.Add(13, 128, 256, tile);
            data.Add(16, 13, 384, tile);     // n far below BM
            data.Add(37, 129, 256, tile);
            data.Add(33, 257, 128, tile);    // S = BN + 1 on the wide tile
            data.Add(64, 256, 512, tile);    // aligned for the wide tile
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Emulation_MatchesExactIntegerReference(int columns, int n, int k, int tile)
    {
        var rng = new Random(490 + 31 * columns + 7 * n + k + tile);
        byte[] codes = RandomCodes(rng, n, k);                    // [n, k/4] packed 2-bit codes
        float[] ws = RandomScales(rng, n * (k / 128));            // fp16-rounded group scales
        byte[] split = BuildSplitLayout(codes, ws, n, k);

        sbyte[] q = RandomInt8(rng, columns * k);
        (float[] xd, int[] xsum) = BlockMeta(q, rng, columns, k);
        sbyte[] permuted = Permute(q);

        float[] y = Emulate(split, permuted, xd, xsum, n, k, columns, tile);
        (double[] refY, double[] absSum) = Reference(codes, ws, q, xd, n, k, columns);

        double worst = 0;
        int worstIdx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double tol = 1e-5 * absSum[i] + 1e-6;
            double ratio = Math.Abs(refY[i] - y[i]) / tol;
            if (ratio > worst) { worst = ratio; worstIdx = i; }
        }
        _out.WriteLine($"S={columns} n={n} k={k} BN={tile}: worst |diff|/tol = {worst:F4} " +
                       $"(col {worstIdx / n}, row {worstIdx % n})");
        Assert.True(worst <= 1.0,
            $"S={columns} n={n} k={k} BN={tile}: column {worstIdx / n} row {worstIdx % n}: " +
            $"expected {refY[worstIdx]:R}, emulated {y[worstIdx]:R} (Σ|term| {absSum[worstIdx]:E3})");
    }

    /// <summary>
    /// Outputs outside the <c>[columns, n]</c> rectangle must never be written: the emulation fills
    /// the destination with a sentinel and checks it survives wherever a tile overhangs.
    /// </summary>
    [Theory]
    [InlineData(33, 129, TileWide)]
    [InlineData(17, 37, TileNarrow)]
    public void Emulation_NeverWritesOutsideTheOutputRectangle(int columns, int n, int tile)
    {
        const int k = 256;
        var rng = new Random(4901 + columns + n + tile);
        byte[] codes = RandomCodes(rng, n, k);
        float[] ws = RandomScales(rng, n * (k / 128));
        byte[] split = BuildSplitLayout(codes, ws, n, k);
        sbyte[] q = RandomInt8(rng, columns * k);
        (float[] xd, int[] xsum) = BlockMeta(q, rng, columns, k);

        // One extra column and one extra row of slack around the real output.
        int padded = (columns + 1) * n + n;
        var y = new float[padded];
        Array.Fill(y, float.NaN);
        Emulate(split, Permute(q), xd, xsum, n, k, columns, tile, y);

        for (int i = columns * n; i < padded; i++)
            Assert.True(float.IsNaN(y[i]), $"slack element {i} was written (tile overhang stored past column {columns - 1})");
    }

    /// <summary>
    /// The kernel's magic-number int→float (<c>__int_as_float(0x4B400000 + v) − 12582912f</c>) must be
    /// bit-identical to a plain conversion over the whole range a block dot can reach
    /// (<c>|isum| ≤ 3·127·32 + 127·32 = 16256</c>), so <c>PQ2M_MAGIC_I2F</c> is a pure speed switch.
    /// </summary>
    [Fact]
    public void MagicIntToFloat_IsExactOverTheBlockDotRange()
    {
        for (int v = -(1 << 20); v <= (1 << 20); v++)
            Assert.Equal(BitConverter.SingleToInt32Bits((float)v), BitConverter.SingleToInt32Bits(MagicI2F(v)));
    }

    private static float MagicI2F(int v)
        => BitConverter.Int32BitsToSingle(unchecked(0x4B400000 + v)) - 12582912.0f;

    // ───────────────────────── the emulation ─────────────────────────

    /// <summary>
    /// The emulated kernel output. Exposed so the GPU parity test can assert the compiled kernel is
    /// BIT-identical to this port on a small shape: the accumulation order, the exact int-to-float
    /// conversion and the unfused <c>d*ws</c> product are the same on both sides, so any difference
    /// means nvcc compiled something other than the reviewed source (FMA contraction, say).
    /// </summary>
    internal static float[] Emulate(byte[] split, sbyte[] permuted, float[] xd, int[] xsum,
        int n, int k, int columns, int tileColumns)
    {
        var y = new float[(long)columns * n];
        Emulate(split, permuted, xd, xsum, n, k, columns, tileColumns, y);
        return y;
    }

    /// <summary>
    /// Instruction-for-instruction port of <c>pq2_0_mmq_dp4a_body&lt;BN, TN, TM&gt;</c>. Phases are
    /// separated exactly where the kernel has <c>__syncthreads()</c>.
    /// </summary>
    private static void Emulate(byte[] split, sbyte[] permuted, float[] xd, int[] xsum,
        int n, int k, int columns, int tileColumns, float[] y)
    {
        const int threads = 256, threadsM = 32;
        int tn = tileColumns == TileNarrow ? 2 : 4;
        int tm = tileColumns == TileNarrow ? 8 : 4;
        int bn = tileColumns, bm = threadsM * tm;

        int gpr = k / 128, bpr = k / 32;
        long totalGroups = (long)n * gpr;
        long codesBase = ((totalGroups * 2) + 31) & ~31L;   // pq2m_codes_base_offset

        int gridX = (columns + bn - 1) / bn;                // x = column tile (L2 reuse order)
        int gridY = (n + bm - 1) / bm;

        // Shared memory of one block.
        var wsm = new ulong[4, bm];      // [block][row] — transposed, as in the kernel
        var wscale = new float[bm];
        var xsm = new uint[bn, 32];      // [col][word]; uint4 chunk w = words 4w..4w+3
        var smXd = new float[bn, 4];
        var smXs = new int[bn, 4];

        for (int by = 0; by < gridY; by++)
        for (int bx = 0; bx < gridX; bx++)
        {
            int rowBase = by * bm, colBase = bx * bn;
            var acc = new float[threads, tm, tn];

            for (int g = 0; g < gpr; g++)
            {
                // ── staging (every thread, then a barrier) ──
                for (int tid = 0; tid < threads; tid++)
                {
                    for (int i = tid; i < bm * 2; i += threads)
                    {
                        int r = i >> 1, h = i & 1;
                        int rowG = Math.Min(rowBase + r, n - 1);
                        long baseOff = codesBase + ((long)rowG * gpr + g) * 32 + 16L * h;
                        wsm[2 * h, r] = BitConverter.ToUInt64(split, (int)baseOff);
                        wsm[2 * h + 1, r] = BitConverter.ToUInt64(split, (int)baseOff + 8);
                    }
                    for (int i = tid; i < bm; i += threads)
                    {
                        int rowG = Math.Min(rowBase + i, n - 1);
                        ushort bits = BitConverter.ToUInt16(split, (int)(((long)rowG * gpr + g) * 2));
                        wscale[i] = (float)BitConverter.UInt16BitsToHalf(bits);
                    }
                    for (int i = tid; i < bn * 8; i += threads)
                    {
                        int c = i >> 3, w = i & 7;
                        int colG = colBase + c;
                        for (int j = 0; j < 4; j++)
                        {
                            uint word = 0;
                            if (colG < columns)
                            {
                                long b0 = (long)colG * k + (long)g * 128 + 16L * w + 4L * j;
                                word = (uint)(byte)permuted[b0]
                                     | ((uint)(byte)permuted[b0 + 1] << 8)
                                     | ((uint)(byte)permuted[b0 + 2] << 16)
                                     | ((uint)(byte)permuted[b0 + 3] << 24);
                            }
                            xsm[c, 4 * w + j] = word;
                        }
                    }
                    for (int i = tid; i < bn * 4; i += threads)
                    {
                        int c = i >> 2, b = i & 3;
                        int colG = colBase + c;
                        bool live = colG < columns;
                        smXd[c, b] = live ? xd[(long)colG * bpr + g * 4 + b] : 0.0f;
                        smXs[c, b] = live ? xsum[(long)colG * bpr + g * 4 + b] : 0;
                    }
                }

                // ── compute (every thread, then a barrier) ──
                for (int tid = 0; tid < threads; tid++)
                {
                    int tIdM = tid & (threadsM - 1), tIdN = tid / threadsM;
                    int colLocal = tIdN * tn;

                    var ws = new float[tm];
                    for (int r = 0; r < tm; r++) ws[r] = wscale[tIdM + threadsM * r];

                    for (int b = 0; b < 4; b++)
                    {
                        var a = new uint[tn, 8];
                        var d = new float[tn];
                        var xs = new int[tn];
                        for (int c = 0; c < tn; c++)
                        {
                            for (int w = 0; w < 8; w++) a[c, w] = xsm[colLocal + c, 8 * b + w];
                            d[c] = smXd[colLocal + c, b];
                            xs[c] = smXs[colLocal + c, b];
                        }

                        for (int r = 0; r < tm; r++)
                        {
                            ulong cw = wsm[b, tIdM + threadsM * r];
                            uint cwx = (uint)cw, cwy = (uint)(cw >> 32);
                            var wp = new uint[8];
                            for (int i = 0; i < 4; i++)
                            {
                                wp[i] = (cwx >> (2 * i)) & 0x03030303u;
                                wp[4 + i] = (cwy >> (2 * i)) & 0x03030303u;
                            }
                            for (int c = 0; c < tn; c++)
                            {
                                int isum = -xs[c];
                                for (int w = 0; w < 8; w++) isum = Dp4a(wp[w], a[c, w], isum);
                                acc[tid, r, c] = MathF.FusedMultiplyAdd(isum, d[c] * ws[r], acc[tid, r, c]);
                            }
                        }
                    }
                }
            }

            // ── store ──
            for (int tid = 0; tid < threads; tid++)
            {
                int tIdM = tid & (threadsM - 1), tIdN = tid / threadsM;
                for (int r = 0; r < tm; r++)
                {
                    int row = rowBase + tIdM + threadsM * r;
                    if (row >= n) continue;
                    for (int c = 0; c < tn; c++)
                    {
                        int col = colBase + tIdN * tn + c;
                        if (col < columns) y[(long)col * n + row] = acc[tid, r, c];
                    }
                }
            }
        }
    }

    /// <summary>CUDA <c>__dp4a(int, int, int)</c>: signed byte-wise dot plus accumulator.</summary>
    private static int Dp4a(uint a, uint b, int c)
    {
        for (int i = 0; i < 4; i++)
            c += (sbyte)(a >> (8 * i)) * (sbyte)(b >> (8 * i));
        return c;
    }

    // ───────────────────────── reference and fixtures ─────────────────────────

    /// <summary>
    /// Independent reference: <c>Σ_blocks (Σ (code−1)·q) · (d·ws)</c> straight off the unpermuted
    /// codes and int8 values, accumulated in double. Shares no index math with the emulation.
    /// </summary>
    private static (double[] Y, double[] AbsSum) Reference(byte[] codes, float[] ws, sbyte[] q, float[] xd,
        int n, int k, int columns)
    {
        int gpr = k / 128, bpr = k / 32;
        var y = new double[(long)columns * n];
        var abs = new double[(long)columns * n];
        for (int r = 0; r < n; r++)
        for (int c = 0; c < columns; c++)
        {
            double sum = 0, a = 0;
            for (int blk = 0; blk < bpr; blk++)
            {
                int g = blk / 4;
                int isum = 0;
                for (int i = 0; i < 32; i++)
                {
                    long e = (long)blk * 32 + i;
                    int code = (codes[(long)r * (k / 4) + e / 4] >> (int)(2 * (e % 4))) & 3;
                    isum += (code - 1) * q[(long)c * k + e];
                }
                double term = (double)isum * (float)(xd[(long)c * bpr + blk] * ws[(long)r * gpr + g]);
                sum += term;
                a += Math.Abs(term);
            }
            y[(long)c * n + r] = sum;
            abs[(long)c * n + r] = a;
        }
        return (y, abs);
    }

    /// <summary>The SPLIT layout the kernel reads: all <c>n·gpr</c> fp16 scales (padded to 32 bytes), then 32 code bytes per group.</summary>
    private static byte[] BuildSplitLayout(byte[] codes, float[] ws, int n, int k)
    {
        int gpr = k / 128;
        long totalGroups = (long)n * gpr;
        long codesBase = (totalGroups * 2 + 31) & ~31L;
        var split = new byte[codesBase + totalGroups * 32];
        for (long g = 0; g < totalGroups; g++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)ws[g]);
            split[g * 2] = (byte)bits;
            split[g * 2 + 1] = (byte)(bits >> 8);
        }
        for (int r = 0; r < n; r++)
            for (int g = 0; g < gpr; g++)
                Array.Copy(codes, (long)r * (k / 4) + (long)g * 32,
                           split, codesBase + ((long)r * gpr + g) * 32, 32);
        return split;
    }

    private static byte[] RandomCodes(Random rng, int n, int k)
    {
        var codes = new byte[(long)n * (k / 4)];
        for (long i = 0; i < codes.LongLength; i++)
        {
            int v = 0;
            for (int j = 0; j < 4; j++)
            {
                int u = rng.Next(10);
                v |= (u < 3 ? 0 : u < 6 ? 1 : u < 9 ? 2 : 3) << (2 * j);   // code 3 (+2) included
            }
            codes[i] = (byte)v;
        }
        return codes;
    }

    private static float[] RandomScales(Random rng, long count)
    {
        var ws = new float[count];
        for (long i = 0; i < count; i++) ws[i] = (float)(Half)(0.01f + rng.NextSingle() * 0.05f);
        return ws;
    }

    private static sbyte[] RandomInt8(Random rng, long count)
    {
        var q = new sbyte[count];
        for (long i = 0; i < count; i++) q[i] = (sbyte)rng.Next(-127, 128);
        return q;
    }

    /// <summary>Per-32 metadata the quantizer emits: a half-rounded scale and the exact sum of the block's int8 values.</summary>
    private static (float[] D, int[] Sum) BlockMeta(sbyte[] q, Random rng, int columns, int k)
    {
        int blocks = columns * (k / 32);
        var d = new float[blocks];
        var sum = new int[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = (float)(Half)(rng.NextSingle() * 0.05f);
            int s = 0;
            for (int i = 0; i < 32; i++) s += q[(long)b * 32 + i];
            sum[b] = s;
        }
        return (d, sum);
    }

    /// <summary>The quantizer's in-chunk permutation: element <c>16c + 4j + i</c> lands at byte <c>16c + 4i + j</c>.</summary>
    private static sbyte[] Permute(sbyte[] q)
    {
        var p = new sbyte[q.Length];
        for (int e = 0; e < q.Length; e++)
        {
            int chunk = e & ~15, r = e & 15;
            p[chunk + 4 * (r & 3) + (r >> 2)] = q[e];
        }
        return p;
    }
}
