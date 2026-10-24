using System;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Verifies the blockwise normalized Walsh-Hadamard transform used by PrismML Hadamard-folded
/// checkpoints (<c>prism.hadamard.*</c>, Bonsai 2).
/// </summary>
/// <remarks>
/// The oracle is the dense rotation matrix the PrismML llama.cpp fork materializes in
/// <c>llama_model::load_tensors</c>:
/// <code>
/// data[row*n + col] = popcount_parity(row &amp; col) ? -1/sqrt(n) : +1/sqrt(n)
/// </code>
/// The fast path must equal a dense multiply by that matrix — that equivalence is the whole
/// correctness argument for skipping the matrix at runtime, so it is tested directly rather than
/// via a round-trip (which an all-zeros or sign-flipped implementation would also pass).
/// </remarks>
public sealed class HadamardTests
{
    /// <summary>
    /// Builds the fork's dense rotation matrix for a block of <paramref name="n"/> elements.
    /// </summary>
    private static float[,] DenseRotation(int n)
    {
        float scale = 1f / MathF.Sqrt(n);
        var h = new float[n, n];
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                uint parity = (uint)(row & col);
                parity ^= parity >> 16;
                parity ^= parity >> 8;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                h[row, col] = (parity & 1) != 0 ? -scale : scale;
            }
        }
        return h;
    }

    /// <summary>Dense reference: applies <see cref="DenseRotation"/> blockwise to a row.</summary>
    private static float[] DenseBlockwise(ReadOnlySpan<float> src, int blockSize)
    {
        var h = DenseRotation(blockSize);
        var dst = new float[src.Length];
        for (int b = 0; b < src.Length; b += blockSize)
        {
            for (int row = 0; row < blockSize; row++)
            {
                double acc = 0;
                for (int col = 0; col < blockSize; col++)
                    acc += (double)h[row, col] * src[b + col];
                dst[b + row] = (float)acc;
            }
        }
        return dst;
    }

    private static float[] RandomRow(int length, int seed)
    {
        var rng = new Random(seed);
        var row = new float[length];
        for (int i = 0; i < length; i++)
            row[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return row;
    }

    private static sbyte[] RandomSigns(int length, int seed)
    {
        var rng = new Random(seed);
        var signs = new sbyte[length];
        for (int i = 0; i < length; i++)
            signs[i] = (sbyte)(rng.Next(2) == 0 ? -1 : 1);
        return signs;
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(64)]
    [InlineData(1024)] // the Bonsai 2 block size
    public void ForwardRow_WithoutSigns_MatchesDenseRotation(int blockSize)
    {
        var src = RandomRow(blockSize, seed: 0x4AD0 + blockSize);
        var expected = DenseBlockwise(src, blockSize);

        var actual = new float[blockSize];
        Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, actual, blockSize);

        AssertClose(expected, actual);
    }

    [Fact]
    public void ForwardRow_MultiBlockRow_TransformsEachBlockIndependently()
    {
        // 5120 is the Bonsai 2 hidden width: 5 independent blocks of 1024.
        const int blockSize = 1024;
        const int width = 5120;
        var src = RandomRow(width, seed: 0x5120);
        var expected = DenseBlockwise(src, blockSize);

        var actual = new float[width];
        Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, actual, blockSize);

        AssertClose(expected, actual);
    }

    [Fact]
    public void ForwardRow_AppliesSignsBeforeRotation()
    {
        const int blockSize = 64;
        var src = RandomRow(blockSize, seed: 0x519);
        var signs = RandomSigns(blockSize, seed: 0x51A);

        // Reference: flip first, then rotate.
        var flipped = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
            flipped[i] = signs[i] < 0 ? -src[i] : src[i];
        var expected = DenseBlockwise(flipped, blockSize);

        var actual = new float[blockSize];
        Hadamard.ForwardRow(src, signs, actual, blockSize);

        AssertClose(expected, actual);
    }

    [Fact]
    public void InverseRow_AppliesSignsAfterRotation_AndIsNotTheForwardOrder()
    {
        const int blockSize = 64;
        var src = RandomRow(blockSize, seed: 0x11B);
        var signs = RandomSigns(blockSize, seed: 0x11C);

        // Reference: rotate first, then flip.
        var rotated = DenseBlockwise(src, blockSize);
        var expected = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
            expected[i] = signs[i] < 0 ? -rotated[i] : rotated[i];

        var actual = new float[blockSize];
        Hadamard.InverseRow(src, signs, actual, blockSize);

        AssertClose(expected, actual);

        // Discriminates against the forward order: with random signs the two orders differ, so a
        // copy-paste of ForwardRow into InverseRow fails here rather than silently corrupting
        // embeddings at runtime.
        var forwardOrder = new float[blockSize];
        Hadamard.ForwardRow(src, signs, forwardOrder, blockSize);
        Assert.False(
            MaxAbsDiff(expected, forwardOrder) < 1e-4f,
            "sign-before-rotation and sign-after-rotation must not agree, or the test cannot detect the swap");
    }

    [Fact]
    public void ForwardThenInverse_RoundTripsToIdentity()
    {
        const int blockSize = 1024;
        var src = RandomRow(blockSize * 2, seed: 0x5711);
        var signs = RandomSigns(blockSize * 2, seed: 0x5712);

        var rotated = new float[src.Length];
        Hadamard.ForwardRow(src, signs, rotated, blockSize);

        // Forward is S then H; the inverse of that is H then S, which is exactly InverseRow.
        var restored = new float[src.Length];
        Hadamard.InverseRow(rotated, signs, restored, blockSize);

        AssertClose(src, restored);
    }

    [Fact]
    public void ForwardRow_IsOrthogonal_PreservesL2Norm()
    {
        const int blockSize = 1024;
        var src = RandomRow(blockSize, seed: 0x0B7);

        var dst = new float[blockSize];
        Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, dst, blockSize);

        double before = 0, after = 0;
        for (int i = 0; i < blockSize; i++) { before += (double)src[i] * src[i]; after += (double)dst[i] * dst[i]; }
        Assert.Equal(Math.Sqrt(before), Math.Sqrt(after), 3);
    }

    [Fact]
    public void ForwardRow_SupportsInPlaceAliasing()
    {
        const int blockSize = 256;
        var src = RandomRow(blockSize, seed: 0xA11A5);
        var expected = new float[blockSize];
        Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, expected, blockSize);

        var inPlace = (float[])src.Clone();
        Hadamard.ForwardRow(inPlace, ReadOnlySpan<sbyte>.Empty, inPlace, blockSize);

        AssertClose(expected, inPlace);
    }

    [Theory]
    [InlineData(3)]    // not a power of two
    [InlineData(0)]
    public void ForwardRow_RejectsNonPowerOfTwoBlockSize(int blockSize)
    {
        var src = new float[16];
        var dst = new float[16];
        Assert.Throws<ArgumentException>(() =>
            Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, dst, blockSize));
    }

    [Fact]
    public void ForwardRow_RejectsWidthNotMultipleOfBlockSize()
    {
        var src = new float[100];
        var dst = new float[100];
        Assert.Throws<ArgumentException>(() =>
            Hadamard.ForwardRow(src, ReadOnlySpan<sbyte>.Empty, dst, 64));
    }

    /// <summary>
    /// The GDN permute must move whole <c>dState</c> head blocks from tiled
    /// <c>vh = k + nKHead·r</c> to grouped <c>vh' = r + rep·k</c> — the Bonsai 2 geometry is
    /// dState 128, nKHead 16, rep 3 (6144 wide).
    /// </summary>
    [Fact]
    public void PermuteTiledToGrouped_MovesHeadsToGroupedOrder()
    {
        const int dState = 4, nKHead = 16, rep = 3;
        int width = dState * nKHead * rep;

        // Tag each head block with its own head index so misplacement is unambiguous.
        var src = new float[width];
        for (int vh = 0; vh < nKHead * rep; vh++)
            for (int d = 0; d < dState; d++)
                src[vh * dState + d] = vh * 100 + d;

        var dst = new float[width];
        Hadamard.PermuteTiledToGrouped(src, dst, dState, nKHead, rep);

        for (int k = 0; k < nKHead; k++)
        {
            for (int r = 0; r < rep; r++)
            {
                int tiled = k + nKHead * r;
                int grouped = r + rep * k;
                for (int d = 0; d < dState; d++)
                    Assert.Equal(tiled * 100 + d, dst[grouped * dState + d]);
            }
        }
    }

    /// <summary>
    /// With <c>rep &gt; 1</c> the permutation must actually move data — a no-op implementation
    /// would pass a round-trip test but silently feed the fold the wrong basis.
    /// </summary>
    [Fact]
    public void PermuteTiledToGrouped_IsNotIdentity_WhenRepExceedsOne()
    {
        const int dState = 128, nKHead = 16, rep = 3;
        int width = dState * nKHead * rep; // 6144, the Bonsai 2 ssm_out input width

        var src = RandomRow(width, seed: 0x6144);
        var dst = new float[width];
        Hadamard.PermuteTiledToGrouped(src, dst, dState, nKHead, rep);

        Assert.True(MaxAbsDiff(src, dst) > 1e-6f, "permute must not be the identity for rep > 1");

        // And it must be a pure permutation: same multiset of values.
        var a = (float[])src.Clone();
        var b = (float[])dst.Clone();
        Array.Sort(a);
        Array.Sort(b);
        Assert.Equal(a, b);
    }

    /// <summary>
    /// Asserts a relative match against the dense reference.
    /// </summary>
    /// <remarks>
    /// A fixed decimal-place tolerance is the wrong instrument here: the butterfly and the dense
    /// f64 reference differ by float rounding that scales with the block width (intermediates reach
    /// <c>sqrt(n)</c> times the input magnitude before the normalization settles), so an absolute
    /// bound that fits n=64 fails at n=1024 for purely numerical reasons. A relative bound holds
    /// across every block size while still rejecting a genuinely wrong transform, which is off by
    /// order 1, not order 1e-5.
    /// </remarks>
    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float relTol = 2e-5f)
    {
        float scale = 0;
        for (int i = 0; i < expected.Length; i++)
            scale = MathF.Max(scale, MathF.Abs(expected[i]));
        scale = MathF.Max(scale, 1e-9f);

        float worst = 0;
        int worstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float rel = MathF.Abs(expected[i] - actual[i]) / scale;
            if (rel > worst) { worst = rel; worstIdx = i; }
        }

        Assert.True(worst <= relTol,
            $"max relative error {worst:E3} at index {worstIdx} exceeds {relTol:E3} " +
            $"(expected {(worstIdx >= 0 ? expected[worstIdx] : 0)}, actual {(worstIdx >= 0 ? actual[worstIdx] : 0)}, ref scale {scale:E3})");
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0;
        for (int i = 0; i < a.Length; i++)
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }
}
