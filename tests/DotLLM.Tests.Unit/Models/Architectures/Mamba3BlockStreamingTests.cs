using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Kernel-level regression tests for Mamba-3 SISO streaming decode:
/// runs a one-shot T-token block forward and compares it to the same T tokens
/// processed through multiple split schedules, with the canonical k_state /
/// v_state / ssm_state / cum_angle buffers threaded between chunks. Both
/// trajectories must coincide within F32-reorder noise — the streaming SSD
/// closes the shifted_γ[T-1]=0 chunk-edge gap that previously desynced the
/// state trajectory.
/// </summary>
/// <remarks>
/// Unlike the model-level tests (<see cref="Mamba3TransformerModelDecodeTests"/>)
/// these run a SINGLE block in isolation, so they isolate the streaming
/// kernel behaviour from the wider per-layer residual / LM-head pipeline.
/// The inputs are deterministic per-token ramps, the weights are deterministic
/// per-index ramps — identical across all schedules.
/// </remarks>
public sealed class Mamba3BlockStreamingTests
{
    // Small-but-architecturally-realistic dims: n_head=4, head_dim=4,
    // d_state=8, S=2 — matches the Mamba3TransformerModelDecode fixture.
    private const int SeqLen = 6;
    private const int DModel = 8;
    private const int NHead = 4;
    private const int HeadDim = 4;
    private const int DInner = NHead * HeadDim;      // 16
    private const int DState = 8;
    private const int NumBcHeads = 1;
    private const int NumRopeAngles = 2;
    private const int BcPerToken = DState * NumBcHeads;
    private const int DInProj = 2 * DInner + 2 * BcPerToken + 3 * NHead + NumRopeAngles;
    private const float AFloor = 1e-4f;

    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    private readonly ITestOutputHelper _output;

    public Mamba3BlockStreamingTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(new[] { 2, 2, 2 })]
    [InlineData(new[] { 1, 1, 1, 1, 1, 1 })]
    [InlineData(new[] { 3, 3 })]
    [InlineData(new[] { 4, 2 })]
    [InlineData(new[] { 2, 4 })]
    [InlineData(new[] { 1, 2, 3 })]
    [InlineData(new[] { 5, 1 })]
    public void StateThreading_StreamingMatchesOneShot(int[] schedule)
    {
        int sum = 0; foreach (int s in schedule) sum += s;
        Assert.Equal(SeqLen, sum);

        // Random-looking but deterministic per-seed fixtures.
        float[] u = Ramp(SeqLen * DModel, 0.01f, 0);
        float[] inProj = Ramp(DInProj * DModel, 0.003f, 1);
        float[] outProj = Ramp(DModel * DInner, 0.005f, 2);
        float[] dtBias = Ramp(NHead, 0.02f, 3);
        float[] bNormW = Ramp(DState, 0.5f, 4);
        float[] cNormW = Ramp(DState, 0.5f, 5);
        float[] bBias = Ramp(NHead * NumBcHeads * DState, 0.02f, 6);
        float[] cBias = Ramp(NHead * NumBcHeads * DState, 0.02f, 7);
        float[] d = Ramp(NHead, 0.1f, 8);

        using var scratch = Mamba3ForwardScratch.FromDimensions(
            DInner, NHead, DState, NumBcHeads, NumRopeAngles, mimoRank: 1);

        // One-shot: full T tokens in a single Forward.
        float[] yOneShot = new float[SeqLen * DModel];
        float[] ssm1 = new float[NHead * HeadDim * DState];
        float[] cum1 = new float[NHead * NumRopeAngles];
        float[] kSt1 = new float[NHead * DState];
        float[] vSt1 = new float[NHead * HeadDim];
        Mamba3Block.Forward(
            scratch, u, inProj, outProj,
            dtBias, bNormW, cNormW, bBias, cBias, d,
            yOneShot, ssm1, cum1, kSt1, vSt1,
            SeqLen, DModel, DInner, NHead, HeadDim, DState,
            NumBcHeads, NumRopeAngles, AFloor);

        // Streaming: per-chunk Forwards with state threaded.
        float[] ySplit = new float[SeqLen * DModel];
        float[] ssm2 = new float[NHead * HeadDim * DState];
        float[] cum2 = new float[NHead * NumRopeAngles];
        float[] kSt2 = new float[NHead * DState];
        float[] vSt2 = new float[NHead * HeadDim];

        int offset = 0;
        foreach (int chunkLen in schedule)
        {
            float[] uChunk = new float[chunkLen * DModel];
            Array.Copy(u, offset * DModel, uChunk, 0, chunkLen * DModel);
            float[] yChunk = new float[chunkLen * DModel];
            Mamba3Block.Forward(
                scratch, uChunk, inProj, outProj,
                dtBias, bNormW, cNormW, bBias, cBias, d,
                yChunk, ssm2, cum2, kSt2, vSt2,
                chunkLen, DModel, DInner, NHead, HeadDim, DState,
                NumBcHeads, NumRopeAngles, AFloor);
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
            $"{label}  y: max_abs={yMaxAbs:E3} max_rel={yMaxRel:E3}  "
            + $"ssm: max_abs={ssmMaxAbs:E3} max_rel={ssmMaxRel:E3}  "
            + $"k_state: max_abs={kMaxAbs:E3} max_rel={kMaxRel:E3}  "
            + $"v_state: max_abs={vMaxAbs:E3} max_rel={vMaxRel:E3}");

        AssertClose($"{label} y", yOneShot, ySplit);
        AssertClose($"{label} ssm_state", ssm1, ssm2);
        AssertClose($"{label} k_state", kSt1, kSt2);
        AssertClose($"{label} v_state", vSt1, vSt2);
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
