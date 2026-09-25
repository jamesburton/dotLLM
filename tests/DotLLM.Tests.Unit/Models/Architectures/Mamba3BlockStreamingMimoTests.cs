using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Block-level regression tests for Mamba-3 MIMO streaming decode — the
/// MIMO analog of <see cref="Mamba3BlockStreamingTests"/>. Runs a one-shot
/// T-token <see cref="Mamba3Block.ForwardMimo(Mamba3ForwardScratch, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, Span{float}, Span{float}, Span{float}, Span{float}, int, int, int, int, int, int, int, int, int, float, float)"/>
/// forward and compares it to the same T tokens processed through multiple
/// split schedules, with the canonical <c>k_state [R, H, N]</c> /
/// <c>v_state [H, P]</c> / <c>ssm_state [H, P, N]</c> / <c>cum_angle [H, S]</c>
/// buffers threaded between chunks. Both trajectories must coincide within
/// F32-reorder noise — the streaming MIMO SSD closes the shifted_γ[T-1]=0
/// chunk-edge gap by replaying <c>v_state · (Σ_r k_state[r]) · DT[0] · (1-trap[0])</c>
/// into the SSM state at the start of the next chunk.
/// </summary>
/// <remarks>
/// Inputs are deterministic per-token ramps, weights are deterministic
/// per-index ramps — identical across all schedules. The synthetic fixture
/// exercises rank R=3 to catch rank-coupling bugs a R=2 fixture might miss.
/// </remarks>
public sealed class Mamba3BlockStreamingMimoTests
{
    // Small-but-architecturally-valid MIMO dims.
    private const int SeqLen = 6;
    private const int DModel = 8;
    private const int NHead = 4;
    private const int HeadDim = 4;
    private const int DInner = NHead * HeadDim;     // 16
    private const int DState = 8;
    private const int NumBcHeads = 1;
    private const int NumRopeAngles = 2;
    private const int MimoRank = 3;
    private const int BcPerToken = DState * NumBcHeads * MimoRank;
    private const int DInProj = 2 * DInner + 2 * BcPerToken + 3 * NHead + NumRopeAngles;
    private const float AFloor = 1e-4f;

    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    private readonly ITestOutputHelper _output;

    public Mamba3BlockStreamingMimoTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(new[] { 6 })]
    [InlineData(new[] { 3, 3 })]
    [InlineData(new[] { 2, 2, 2 })]
    [InlineData(new[] { 1, 1, 1, 1, 1, 1 })]
    [InlineData(new[] { 4, 2 })]
    [InlineData(new[] { 2, 4 })]
    [InlineData(new[] { 1, 2, 3 })]
    [InlineData(new[] { 5, 1 })]
    public void StateThreading_MimoStreamingMatchesOneShot(int[] schedule)
    {
        int sum = 0; foreach (int s in schedule) sum += s;
        Assert.Equal(SeqLen, sum);

        // Deterministic per-seed fixtures.
        float[] u = Ramp(SeqLen * DModel, 0.01f, 0);
        float[] inProj = Ramp(DInProj * DModel, 0.003f, 1);
        float[] outProj = Ramp(DModel * DInner, 0.005f, 2);
        float[] dtBias = Ramp(NHead, 0.02f, 3);
        float[] bNormW = Ramp(DState, 0.5f, 4);
        float[] cNormW = Ramp(DState, 0.5f, 5);
        // B_bias / C_bias for MIMO have shape [H, R, N].
        float[] bBias = Ramp(NHead * MimoRank * DState, 0.02f, 6);
        float[] cBias = Ramp(NHead * MimoRank * DState, 0.02f, 7);
        float[] d = Ramp(NHead, 0.1f, 8);
        // mimo_z / mimo_o: shape [H, R, P].
        float[] mimoZ = Ramp(NHead * MimoRank * HeadDim, 0.3f, 9);
        float[] mimoO = Ramp(NHead * MimoRank * HeadDim, 0.3f, 10);

        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, MimoRank);

        // ── One-shot: full T tokens in a single ForwardMimo call ──────────
        float[] yOneShot = new float[SeqLen * DModel];
        float[] ssm1 = new float[NHead * HeadDim * DState];
        float[] cum1 = new float[NHead * NumRopeAngles];
        float[] kSt1 = new float[MimoRank * NHead * DState];
        float[] vSt1 = new float[NHead * HeadDim];
        Mamba3Block.ForwardMimo(
            scratch, u, inProj, outProj,
            dtBias, bNormW, cNormW, bBias, cBias, d, mimoZ, mimoO,
            yOneShot, ssm1, cum1, kSt1, vSt1,
            SeqLen, DModel, DInner, NHead, HeadDim, DState,
            NumBcHeads, NumRopeAngles, MimoRank, AFloor);

        // ── Streaming: per-chunk ForwardMimo with state threaded ─────────
        float[] ySplit = new float[SeqLen * DModel];
        float[] ssm2 = new float[NHead * HeadDim * DState];
        float[] cum2 = new float[NHead * NumRopeAngles];
        float[] kSt2 = new float[MimoRank * NHead * DState];
        float[] vSt2 = new float[NHead * HeadDim];

        int offset = 0;
        foreach (int chunkLen in schedule)
        {
            float[] uChunk = new float[chunkLen * DModel];
            Array.Copy(u, offset * DModel, uChunk, 0, chunkLen * DModel);
            float[] yChunk = new float[chunkLen * DModel];
            Mamba3Block.ForwardMimo(
                scratch, uChunk, inProj, outProj,
                dtBias, bNormW, cNormW, bBias, cBias, d, mimoZ, mimoO,
                yChunk, ssm2, cum2, kSt2, vSt2,
                chunkLen, DModel, DInner, NHead, HeadDim, DState,
                NumBcHeads, NumRopeAngles, MimoRank, AFloor);
            Array.Copy(yChunk, 0, ySplit, offset * DModel, chunkLen * DModel);
            offset += chunkLen;
        }
        Assert.Equal(SeqLen, offset);

        string label = string.Join("+", schedule);
        (float yMaxAbs, float yMaxRel) = Drift(yOneShot, ySplit);
        (float ssmMaxAbs, float ssmMaxRel) = Drift(ssm1, ssm2);
        (float kMaxAbs, float kMaxRel) = Drift(kSt1, kSt2);
        (float vMaxAbs, float vMaxRel) = Drift(vSt1, vSt2);
        _output.WriteLine(
            $"{label}  y: abs={yMaxAbs:E3} rel={yMaxRel:E3}  "
            + $"ssm: abs={ssmMaxAbs:E3} rel={ssmMaxRel:E3}  "
            + $"k_state: abs={kMaxAbs:E3} rel={kMaxRel:E3}  "
            + $"v_state: abs={vMaxAbs:E3} rel={vMaxRel:E3}");

        AssertClose($"{label} y", yOneShot, ySplit);
        AssertClose($"{label} ssm_state", ssm1, ssm2);
        AssertClose($"{label} k_state", kSt1, kSt2);
        AssertClose($"{label} v_state", vSt1, vSt2);
    }

    [Fact]
    public void OneShotCall_WithEmptyStreamingBuffers_MatchesBaseOverload()
    {
        // The ForwardMimo overload that omits kState/vState must produce the
        // same output as the streaming overload called with empty spans —
        // this guards the base-overload dispatch wiring.
        float[] u = Ramp(SeqLen * DModel, 0.01f, 0);
        float[] inProj = Ramp(DInProj * DModel, 0.003f, 1);
        float[] outProj = Ramp(DModel * DInner, 0.005f, 2);
        float[] dtBias = Ramp(NHead, 0.02f, 3);
        float[] bNormW = Ramp(DState, 0.5f, 4);
        float[] cNormW = Ramp(DState, 0.5f, 5);
        float[] bBias = Ramp(NHead * MimoRank * DState, 0.02f, 6);
        float[] cBias = Ramp(NHead * MimoRank * DState, 0.02f, 7);
        float[] d = Ramp(NHead, 0.1f, 8);
        float[] mimoZ = Ramp(NHead * MimoRank * HeadDim, 0.3f, 9);
        float[] mimoO = Ramp(NHead * MimoRank * HeadDim, 0.3f, 10);

        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, MimoRank);

        float[] yBase = new float[SeqLen * DModel];
        float[] ssmBase = new float[NHead * HeadDim * DState];
        float[] cumBase = new float[NHead * NumRopeAngles];
        Mamba3Block.ForwardMimo(
            scratch, u, inProj, outProj,
            dtBias, bNormW, cNormW, bBias, cBias, d, mimoZ, mimoO,
            yBase, ssmBase, cumBase,
            SeqLen, DModel, DInner, NHead, HeadDim, DState,
            NumBcHeads, NumRopeAngles, MimoRank, AFloor);

        float[] yStream = new float[SeqLen * DModel];
        float[] ssmStream = new float[NHead * HeadDim * DState];
        float[] cumStream = new float[NHead * NumRopeAngles];
        Mamba3Block.ForwardMimo(
            scratch, u, inProj, outProj,
            dtBias, bNormW, cNormW, bBias, cBias, d, mimoZ, mimoO,
            yStream, ssmStream, cumStream,
            kState: Span<float>.Empty, vState: Span<float>.Empty,
            SeqLen, DModel, DInner, NHead, HeadDim, DState,
            NumBcHeads, NumRopeAngles, MimoRank, AFloor);

        // Byte-for-byte: both paths go through the same code.
        for (int i = 0; i < yBase.Length; i++) Assert.Equal(yBase[i], yStream[i]);
        for (int i = 0; i < ssmBase.Length; i++) Assert.Equal(ssmBase[i], ssmStream[i]);
        for (int i = 0; i < cumBase.Length; i++) Assert.Equal(cumBase[i], cumStream[i]);
    }

    private static float[] Ramp(int count, float amplitude, int seed)
    {
        float[] r = new float[count];
        for (int i = 0; i < count; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            r[i] = amplitude * MathF.Cos(phi);
        }
        return r;
    }

    private static (float maxAbs, float maxRel) Drift(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float maxAbs = 0f, maxRel = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            float rel = diff / (MathF.Abs(a[i]) + 1e-12f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
        }
        return (maxAbs, maxRel);
    }

    private static void AssertClose(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float absDiff = MathF.Abs(e - a);
            float relDiff = absDiff / (MathF.Abs(e) + 1e-12f);
            if (absDiff > AbsTol && relDiff > RelTol)
            {
                Assert.Fail(
                    $"{label}[{i}]: expected={e:F8} actual={a:F8} "
                    + $"abs={absDiff:E3} rel={relDiff:E3} "
                    + $"(tol abs={AbsTol:E0} rel={RelTol:E0})");
            }
        }
    }
}
