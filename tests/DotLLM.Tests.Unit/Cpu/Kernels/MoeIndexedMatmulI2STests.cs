using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Numerical gate for the indexed ternary (I2_S) MoE matmul
/// (<see cref="MatMul.MoeIndexedMatmulI2_S"/>, issue #116).
///
/// <para>The kernel fuses per-expert base+stride addressing with the trit-unpack +
/// per-expert-α dequant inner loop of the proven dense I2_S GEMM. These tests prove the
/// indexed path is correct against that dense path with <b>no training and no GPU</b>: we
/// build N identical expert clones of one weight matrix, route tokens to those clones, and
/// assert the indexed output equals the dense <see cref="MatMul.GemmI2_S(byte*, float*, float*, int, int, int, DotLLM.Cpu.Threading.ComputeThreadPool?)"/>
/// output on the same weights.</para>
///
/// <para>Tolerance: on the no-AVX2 float-fallback path both kernels run the identical scalar
/// dot with the identical scale, so the result is bit-exact. On AVX2/VNNI boxes both take the
/// W2A8 path and quantise activations identically, so they still match to well within the
/// <c>1e-4</c> envelope asserted here.</para>
/// </summary>
public sealed unsafe class MoeIndexedMatmulI2STests
{
    /// <summary>
    /// Test-side reference packer for the dense I2_S layout: ternary {-1,0,+1} → codes {0,1,2},
    /// 4 per byte in 128-element blocks, followed by a single float32 per-tensor scale at the
    /// tail (byte offset n/4). Mirrors I2STests.PackI2S. Caller frees.
    /// </summary>
    private static byte* PackI2SWithTailScale(sbyte[] ternary, float scale)
    {
        int n = ternary.Length;
        byte* buf = (byte*)NativeMemory.AllocZeroed((nuint)(n / 4 + 4));
        for (int e = 0; e < n; e++)
        {
            int block = e / 128, j = e % 128;
            int groupIdx = j / 32, groupPos = j % 32;
            int code = ternary[e] + 1; // -1→0, 0→1, +1→2
            buf[block * 32 + groupPos] |= (byte)(code << (6 - 2 * groupIdx));
        }
        *(float*)(buf + n / 4) = scale;
        return buf;
    }

    /// <summary>Packs ternary {-1,0,+1} into the I2_S payload only (no trailing scale).</summary>
    private static void PackI2SPayloadOnly(sbyte[] ternary, byte* dest)
    {
        int n = ternary.Length;
        // dest must be zeroed by the caller.
        for (int e = 0; e < n; e++)
        {
            int block = e / 128, j = e % 128;
            int groupIdx = j / 32, groupPos = j % 32;
            int code = ternary[e] + 1;
            dest[block * 32 + groupPos] |= (byte)(code << (6 - 2 * groupIdx));
        }
    }

    private static void AssertWithinTolerance(float expected, float actual)
    {
        float tol = 1e-4f + 1e-4f * MathF.Abs(expected);
        Assert.True(MathF.Abs(expected - actual) <= tol,
            $"expected {expected}, got {actual}, |Δ|={MathF.Abs(expected - actual)} > tol {tol}");
    }

    /// <summary>
    /// Core gate: identical expert clones + a given routing must reproduce the dense I2_S GEMM
    /// over all tokens. <paramref name="allToZero"/> selects between "all tokens → expert 0"
    /// and uniform "token t → expert t % numExperts" routing.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void IdenticalCloneExperts_MatchDenseI2SGemm(bool allToZero)
    {
        var rng = new Random(2026);
        const int m = 7;            // output features (weight rows)
        const int k = 256;          // input dim (2 I2_S blocks)
        const int n = 12;           // tokens / rows
        const int numExperts = 4;
        const float scale = 0.031f;

        // One random ternary weight matrix [m × k].
        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);

        // Random activations [n × k].
        float[] bAct = new float[n * k];
        for (int i = 0; i < bAct.Length; i++) bAct[i] = rng.NextSingle() * 2f - 1f;

        // ── Reference: dense I2_S GEMM over all tokens on the single weight matrix. ──
        byte* dense = PackI2SWithTailScale(ternary, scale);
        float[] cDense = new float[n * m];

        // ── Indexed: N identical expert clones (packed payload only) + scale vector. ──
        long payloadBytes = (long)m * k / 4;
        byte* banks = (byte*)NativeMemory.AllocZeroed((nuint)(payloadBytes * numExperts));
        float[] expertScales = new float[numExperts];
        int[] rowExpertIds = new int[n];
        float[] cIndexed = new float[n * m];

        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                PackI2SPayloadOnly(ternary, banks + e * payloadBytes);
                expertScales[e] = scale;
            }
            for (int t = 0; t < n; t++)
                rowExpertIds[t] = allToZero ? 0 : t % numExperts;

            fixed (float* bp = bAct)
            fixed (float* cdp = cDense)
            fixed (float* cip = cIndexed)
            fixed (float* scalesPtr = expertScales)
            fixed (int* rowPtr = rowExpertIds)
            {
                MatMul.GemmI2_S(dense, bp, cdp, m, k, n, threadPool: null);

                MatMul.MoeIndexedMatmulI2_S(
                    banks, payloadBytes,
                    new ReadOnlySpan<float>(scalesPtr, numExperts),
                    bp, cip, m, k, n,
                    new ReadOnlySpan<int>(rowPtr, n),
                    threadPool: null);
            }

            for (int i = 0; i < n * m; i++)
                AssertWithinTolerance(cDense[i], cIndexed[i]);
        }
        finally
        {
            NativeMemory.Free(dense);
            NativeMemory.Free(banks);
        }
    }

    /// <summary>
    /// Distinct per-expert scales must be honoured: cloning the same trits but scaling expert e
    /// by α_e, with token t routed to expert t%E, yields dense-output × (α_e / α_ref) per row.
    /// Discriminates against a bug that ignored the per-expert scale vector (e.g. always used
    /// expert 0's α or a tail scale).
    /// </summary>
    [Fact]
    public void PerExpertScaleVector_IsHonoured()
    {
        var rng = new Random(77);
        const int m = 5;
        const int k = 128;          // one I2_S block
        const int n = 9;
        const int numExperts = 3;
        const float baseScale = 0.05f;

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] bAct = new float[n * k];
        for (int i = 0; i < bAct.Length; i++) bAct[i] = rng.NextSingle() * 2f - 1f;

        // Dense reference at unit scale (1.0) so we can rescale per expert analytically.
        byte* denseUnit = PackI2SWithTailScale(ternary, 1.0f);
        float[] cUnit = new float[n * m];

        long payloadBytes = (long)m * k / 4;
        byte* banks = (byte*)NativeMemory.AllocZeroed((nuint)(payloadBytes * numExperts));
        float[] expertScales = new float[numExperts];
        int[] rowExpertIds = new int[n];
        float[] cIndexed = new float[n * m];

        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                PackI2SPayloadOnly(ternary, banks + e * payloadBytes);
                expertScales[e] = baseScale * (e + 1); // distinct: 0.05, 0.10, 0.15
            }
            for (int t = 0; t < n; t++) rowExpertIds[t] = t % numExperts;

            fixed (float* bp = bAct)
            fixed (float* cup = cUnit)
            fixed (float* cip = cIndexed)
            fixed (float* scalesPtr = expertScales)
            fixed (int* rowPtr = rowExpertIds)
            {
                MatMul.GemmI2_S(denseUnit, bp, cup, m, k, n, threadPool: null);
                MatMul.MoeIndexedMatmulI2_S(
                    banks, payloadBytes,
                    new ReadOnlySpan<float>(scalesPtr, numExperts),
                    bp, cip, m, k, n,
                    new ReadOnlySpan<int>(rowPtr, n),
                    threadPool: null);
            }

            for (int t = 0; t < n; t++)
            {
                float expectedScale = expertScales[t % numExperts];
                for (int r = 0; r < m; r++)
                    AssertWithinTolerance(cUnit[t * m + r] * expectedScale, cIndexed[t * m + r]);
            }
        }
        finally
        {
            NativeMemory.Free(denseUnit);
            NativeMemory.Free(banks);
        }
    }

    /// <summary>
    /// Ragged K (issue #206): the indexed path used to guard <c>k % 128 == 0</c> explicitly, but
    /// it always delegates to <see cref="MatMul.GemmI2_S(byte*, float*, float*, int, int, int, float, DotLLM.Cpu.Threading.ComputeThreadPool?)"/>
    /// per touched expert, which is now ragged-safe — so the guard was simply removed. Proves the
    /// indexed path still matches the dense reference when k is not a multiple of 128.
    ///
    /// <para>m is chosen so <c>m*k</c> (the total per-expert element count) is itself an exact
    /// multiple of 128 (gcd(200,128)=8, so m must be a multiple of 16) — matching how every real
    /// I2_S GGUF tensor is shaped (the flattened-block packing never leaves an unwritten partial
    /// tail at the very end of the tensor; a total that isn't 128-aligned would need its own
    /// "implicit zero tail" handling, out of this issue's scope since no real checkpoint needs
    /// it — see PackI2S's byte-sizing note in I2STests.cs for why this matters for the test
    /// packer specifically).</para>
    /// </summary>
    [Fact]
    public void RaggedK_MatchesDenseI2SGemm()
    {
        var rng = new Random(206);
        const int m = 16;   // m*k = 3200 = 25*128 (exact — see remarks above)
        const int k = 200;   // NOT a multiple of 128 (200 % 128 == 72)
        const int n = 9;
        const int numExperts = 3;
        const float scale = 0.031f;

        sbyte[] ternary = new sbyte[m * k];
        for (int i = 0; i < ternary.Length; i++) ternary[i] = (sbyte)(rng.Next(3) - 1);
        float[] bAct = new float[n * k];
        for (int i = 0; i < bAct.Length; i++) bAct[i] = rng.NextSingle() * 2f - 1f;

        byte* dense = PackI2SWithTailScale(ternary, scale);
        float[] cDense = new float[n * m];

        long payloadBytes = (long)m * k / 4;
        byte* banks = (byte*)NativeMemory.AllocZeroed((nuint)(payloadBytes * numExperts));
        float[] expertScales = new float[numExperts];
        int[] rowExpertIds = new int[n];
        float[] cIndexed = new float[n * m];

        try
        {
            for (int e = 0; e < numExperts; e++)
            {
                PackI2SPayloadOnly(ternary, banks + e * payloadBytes);
                expertScales[e] = scale;
            }
            for (int t = 0; t < n; t++) rowExpertIds[t] = t % numExperts;

            fixed (float* bp = bAct)
            fixed (float* cdp = cDense)
            fixed (float* cip = cIndexed)
            fixed (float* scalesPtr = expertScales)
            fixed (int* rowPtr = rowExpertIds)
            {
                MatMul.GemmI2_S(dense, bp, cdp, m, k, n, threadPool: null);

                MatMul.MoeIndexedMatmulI2_S(
                    banks, payloadBytes,
                    new ReadOnlySpan<float>(scalesPtr, numExperts),
                    bp, cip, m, k, n,
                    new ReadOnlySpan<int>(rowPtr, n),
                    threadPool: null);
            }

            for (int i = 0; i < n * m; i++)
                AssertWithinTolerance(cDense[i], cIndexed[i]);
        }
        finally
        {
            NativeMemory.Free(dense);
            NativeMemory.Free(banks);
        }
    }
}
