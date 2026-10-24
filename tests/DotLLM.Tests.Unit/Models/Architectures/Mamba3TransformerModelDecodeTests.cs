using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Verifies persistent <see cref="Mamba3State"/> threading across
/// <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
/// calls on a synthetic tiny-but-architecturally-valid checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// <b>What these tests assert.</b>
/// </para>
/// <list type="bullet">
///   <item><description>
///     Fresh state is zero; <see cref="Mamba3State.Reset"/> returns it to zero.
///   </description></item>
///   <item><description>
///     State threading is <em>deterministic</em>: two runs of the same
///     chunk schedule against a fresh state produce identical logits down
///     to F32 tolerances.
///   </description></item>
///   <item><description>
///     State actually advances across calls: a zero-state decode of a
///     token does NOT match a state-threaded decode of the same token.
///     This rules out the "state reset every call" bug.
///   </description></item>
///   <item><description>
///     Prefill + per-token decode reproduces a one-shot prefill to F32
///     noise (<see cref="PrefillThenDecode_BitEqualsOneShot"/>).
///   </description></item>
/// </list>
/// <para>
/// <b>Why streaming decode reproduces one-shot.</b> The canonical Mamba-3
/// scan's per-token state update uses
/// <c>scale[t] = γ[t] + shifted_γ[t]</c> with
/// <c>shifted_γ[t] = DT[t+1]·(1 − trap[t+1])</c> — a 1-token lookahead baked
/// into each <c>h_t</c> update. At the last token of a chunk the lookahead
/// evaluates to zero (the next token is in the next call). The streaming
/// decode plumbing persists the previous chunk's last-token post-RoPE K
/// and V on <see cref="Mamba3State.KState(int)"/> /
/// <see cref="Mamba3State.VState(int)"/> and, at the START of the next
/// chunk, adds the deferred <c>ssm += v · k · DT[0] · (1 − trap[0])</c>
/// term to the SSM state BEFORE the scan — reproducing what a one-shot
/// forward would have folded in at that position. Matches
/// <c>mamba3_siso_fwd.py:341-352</c>.
/// </para>
/// </remarks>
public sealed class Mamba3TransformerModelDecodeTests : IDisposable
{
    // Matches Mamba3TransformerModelTests fixture dims verbatim so the shared
    // fixture builder stays unchanged.
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // Determinism tolerance: F32 kernels with identical operand scheduling
    // should be bit-identical, but hold a small slop for future kernel reorders.
    private const float DeterminismAbsTol = 1e-6f;
    private const float DeterminismRelTol = 1e-5f;

    // Split-vs-one-shot tolerance — F32-reorder noise only once
    // streaming-decode is plumbed (k_state + v_state + chunk-boundary
    // adjustment). Observed on the tiny 2-layer synthetic fixture: 2+2
    // splits drift 1e-8, 2+1+1 and 1+1+1+1 drift ~2e-6 (two or three F32
    // accumulation reorders across chunk edges). Tolerance scoped to that
    // envelope — two+ orders of magnitude tighter than the pre-P3 ceiling.
    private const float SplitAbsTol = 1e-5f;
    private const float SplitRelTol = 1e-4f;

    private readonly string _scratch;
    private readonly ITestOutputHelper _output;

    public Mamba3TransformerModelDecodeTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-m3dec-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string Scratch(string name) => Path.Combine(_scratch, name);

    private static ModelConfig BuildConfig()
    {
        var m3 = new Mamba3Config
        {
            StateSize = StateSize,
            NumHeads = NumHeads,
            HeadDim = HeadDim,
            Expand = Expand,
            NumGroups = 1,
            ChunkSize = 2,
            IsMimo = false,
            MimoRank = 4,
            AFloor = 1e-4f,
            DtInitFloor = 1e-4f,
            DtMin = 1e-3f,
            DtMax = 0.1f,
            UseL2Warp = false,
            RopeFraction = 0.5f,
            IsOutProjNorm = false,
            RescalePrenormResidual = true,
            ResidualInFp32 = true,
        };
        return new ModelConfig
        {
            Architecture = Architecture.Mamba3,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = 0,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 32,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.None,
            RoPEConfig = null,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            SlidingWindowSize = null,
            MlaConfig = null,
            HybridLayout = null,
            SsmConfig = null,
            Mamba3Config = m3,
            ChatTemplate = null,
        };
    }

    private void WriteSmallWeightFixture(string path)
    {
        var b = new SafetensorsFixtureBuilder();

        SmallRamp(b, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], 0.05f, 0);
        SmallRamp(b, Mamba3TensorMapping.FinalNorm, [HiddenSize], 0.5f, 1);
        SmallRamp(b, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], 0.05f, 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            SmallRamp(b, Mamba3TensorMapping.LayerNorm(i), [HiddenSize], 0.5f, sBase + 0);
            SmallRamp(b, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], 0.02f, sBase + 1);
            SmallRamp(b, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], 0.05f, sBase + 2);
            SmallRamp(b, Mamba3TensorMapping.BNorm(i), [StateSize], 0.5f, sBase + 3);
            SmallRamp(b, Mamba3TensorMapping.CNorm(i), [StateSize], 0.5f, sBase + 4);
            SmallRamp(b, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize], 0.02f, sBase + 5);
            SmallRamp(b, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize], 0.02f, sBase + 6);
            SmallRamp(b, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, sBase + 7);
            SmallRamp(b, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, sBase + 8);
        }

        b.WriteTo(path);
    }

    private static void SmallRamp(SafetensorsFixtureBuilder b, string name,
                                  int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        b.AddFloat32(name, shape, values);
    }

    // ────────────────────────────────────────────────────────────────────
    // State hygiene
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Mamba3State_ZeroedOnConstruct()
    {
        ModelConfig cfg = BuildConfig();
        using var state = new Mamba3State(cfg);

        Assert.Equal(NumLayers, state.NumLayers);
        Assert.Equal(NumHeads * HeadDim * StateSize, state.SsmStateElementsPerLayer);
        Assert.Equal(NumHeads * NumRopeAngles, state.CumAngleElementsPerLayer);
        Assert.Equal(NumHeads * StateSize, state.KStateElementsPerLayer);
        Assert.Equal(NumHeads * HeadDim, state.VStateElementsPerLayer);

        for (int layer = 0; layer < NumLayers; layer++)
        {
            foreach (float v in state.SsmState(layer)) Assert.Equal(0f, v);
            foreach (float v in state.CumAngle(layer)) Assert.Equal(0f, v);
            foreach (float v in state.KState(layer)) Assert.Equal(0f, v);
            foreach (float v in state.VState(layer)) Assert.Equal(0f, v);
        }

        // Scribble non-zero values, then Reset → zero again.
        for (int layer = 0; layer < NumLayers; layer++)
        {
            state.SsmState(layer).Fill(1.25f);
            state.CumAngle(layer).Fill(-2.5f);
            state.KState(layer).Fill(3.0f);
            state.VState(layer).Fill(-4.0f);
        }
        state.Reset();
        for (int layer = 0; layer < NumLayers; layer++)
        {
            foreach (float v in state.SsmState(layer)) Assert.Equal(0f, v);
            foreach (float v in state.CumAngle(layer)) Assert.Equal(0f, v);
            foreach (float v in state.KState(layer)) Assert.Equal(0f, v);
            foreach (float v in state.VState(layer)) Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void Mamba3State_MimoKStateIsRankExpanded()
    {
        // Canonical mamba3.py:434-445: when is_mimo=True the decode cache's
        // k_state carries a rank axis ([R, H, N]); the other buffers are
        // rank-free. A SISO config (covered by Mamba3State_ZeroedOnConstruct)
        // keeps k_state at [H, N].
        const int mimoRank = 3;
        ModelConfig cfg = BuildConfig() with
        {
            Mamba3Config = BuildConfig().Mamba3Config! with
            {
                IsMimo = true,
                MimoRank = mimoRank,
            },
        };
        using var state = new Mamba3State(cfg);

        Assert.Equal(NumHeads * HeadDim * StateSize, state.SsmStateElementsPerLayer);
        Assert.Equal(NumHeads * NumRopeAngles, state.CumAngleElementsPerLayer);
        Assert.Equal(mimoRank * NumHeads * StateSize, state.KStateElementsPerLayer);
        Assert.Equal(NumHeads * HeadDim, state.VStateElementsPerLayer);

        for (int layer = 0; layer < NumLayers; layer++)
        {
            Assert.Equal(mimoRank * NumHeads * StateSize, state.KState(layer).Length);
            foreach (float v in state.KState(layer)) Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void StateThreading_KAndVStateAdvance_AtChunkEnd()
    {
        // After a prefill, k_state / v_state must be non-zero (the previous
        // chunk's last-token post-RoPE K and V have been persisted). Before
        // any forward pass they must be zero. This regression-guards the
        // block's step 6.5 CopyTo into kState/vState.
        string path = Scratch("kv-advance.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig cfg = BuildConfig();
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, cfg);

        using var state = new Mamba3State(cfg);
        // All buffers zero pre-forward — already covered by ZeroedOnConstruct.
        using ITensor _ = model.Forward(new int[] { 0, 1 }, new int[] { 0, 1 }, -1, state);

        for (int layer = 0; layer < cfg.NumLayers; layer++)
        {
            bool kAny = false, vAny = false;
            foreach (float v in state.KState(layer)) if (MathF.Abs(v) > 1e-12f) { kAny = true; break; }
            foreach (float v in state.VState(layer)) if (MathF.Abs(v) > 1e-12f) { vAny = true; break; }
            Assert.True(kAny, $"k_state at layer {layer} is still all-zero after prefill — the block isn't persisting the last-token post-RoPE K.");
            Assert.True(vAny, $"v_state at layer {layer} is still all-zero after prefill — the block isn't persisting the last-token V.");
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Determinism + state evolution
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void StateThreading_IsDeterministic_AcrossRuns()
    {
        string path = Scratch("determinism.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig cfg = BuildConfig();
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, cfg);

        int[] tokens = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        // Run A: 2 + 1 + 1 split with state threaded.
        float[] runA = RunSplit(model, tokens, positions, [2, 1, 1]);

        // Run B: same schedule, fresh state.
        float[] runB = RunSplit(model, tokens, positions, [2, 1, 1]);

        var drift = DriftStats(runA, runB);
        _output.WriteLine($"2+1+1 determinism: max_abs={drift.maxAbs:E3} max_rel={drift.maxRel:E3}");
        AssertClose("2+1+1 determinism", runA, runB,
            absTol: DeterminismAbsTol, relTol: DeterminismRelTol);
    }

    [Fact]
    public void StateThreading_ActuallyAdvances_NotZeroedPerCall()
    {
        // If the state were reset to zero at every call (the bug this whole
        // feature exists to fix), a single-token decode at t=1 would produce
        // the same logits as a single-token decode from a fresh state — i.e.
        // position 1 in a threaded run would equal position 0 in a zero-state
        // run. Verify the two DIFFER.
        string path = Scratch("advance.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig cfg = BuildConfig();
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, cfg);

        int[] tokens = [0, 1];
        int[] positions = [0, 1];

        // Threaded path: prefill [0], decode [1] — state carries [0]'s influence.
        using var state = new Mamba3State(cfg);
        using ITensor _ = model.Forward(tokens.AsSpan(0, 1), positions.AsSpan(0, 1), -1, state);
        using ITensor threaded = model.Forward(tokens.AsSpan(1, 1), positions.AsSpan(1, 1), -1, state);
        float[] threadedLogits = TensorToArray(threaded);

        // Zero-state path: forward [1] from a fresh state.
        using var zeroState = new Mamba3State(cfg);
        using ITensor zero = model.Forward(tokens.AsSpan(1, 1), positions.AsSpan(1, 1), -1, zeroState);
        float[] zeroLogits = TensorToArray(zero);

        var drift = DriftStats(threadedLogits, zeroLogits);
        _output.WriteLine(
            $"threaded-vs-zero-state single token: max_abs={drift.maxAbs:E3} max_rel={drift.maxRel:E3}");

        // State must have influenced the result. The tiny-ramp synthetic
        // weights produce small activations, so the difference is O(1e-8) but
        // still well above the bitwise-equal noise floor (== 0.0f exactly when
        // state is truly ignored, as the two forwards are then identical modulo
        // the fresh-zero state). Threshold 1e-12 — anything measurably non-zero
        // means the state buffer is being read.
        Assert.True(drift.maxAbs > 1e-12f,
            $"Decode logits are bitwise-equal to zero-state logits (max_abs={drift.maxAbs:E3}); "
            + "state is not being threaded into the forward pass.");
    }

    // ────────────────────────────────────────────────────────────────────
    // Split-prefill equivalence to one-shot (post-streaming-decode)
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void PrefillThenDecode_BitEqualsOneShot()
    {
        // With streaming-decode plumbing (k_state + v_state + chunk-boundary
        // adjustment in Mamba3Block.Forward) a split prefill+decode MUST
        // reproduce one-shot logits to F32-reorder noise only. Previously
        // this asserted a drift bound — the canonical scan's
        // shifted_γ[t] = DT[t+1]·(1-trap[t+1]) lookahead dropped to 0 at
        // chunk edges, so every chunk boundary injected a small per-token
        // discrepancy. Closing the gap means persisting the previous chunk's
        // last-token post-RoPE K and V and replaying the deferred
        // `v · k · DT[0] · (1 - trap[0])` term at the next chunk's start.
        string path = Scratch("split-approx.safetensors");
        WriteSmallWeightFixture(path);

        using var sf = SafetensorsFile.Open(path);
        ModelConfig cfg = BuildConfig();
        using var model = Mamba3TransformerModel.LoadFromSafetensors(sf, cfg);

        int[] tokens = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];
        using ITensor fullLogits = model.Forward(tokens, positions, deviceId: -1);
        float[] full = TensorToArray(fullLogits);

        foreach (int[] schedule in new[] { new[] { 2, 2 }, new[] { 2, 1, 1 }, new[] { 1, 1, 1, 1 } })
        {
            float[] split = RunSplit(model, tokens, positions, schedule);
            var drift = DriftStats(full, split);
            _output.WriteLine(
                $"one-shot vs {string.Join("+", schedule)}: "
                + $"max_abs={drift.maxAbs:E3} max_rel={drift.maxRel:E3} at idx {drift.worstIdx}");
            AssertClose($"one-shot vs {string.Join("+", schedule)}", full, split,
                absTol: SplitAbsTol, relTol: SplitRelTol);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static float[] RunSplit(Mamba3TransformerModel model, int[] tokens,
                                    int[] positions, int[] schedule)
    {
        ModelConfig cfg = model.Config;
        using var state = new Mamba3State(cfg);
        float[] logits = new float[tokens.Length * cfg.VocabSize];
        int tokenOffset = 0;
        foreach (int chunkLen in schedule)
        {
            using ITensor chunk = model.Forward(
                tokens.AsSpan(tokenOffset, chunkLen),
                positions.AsSpan(tokenOffset, chunkLen),
                deviceId: -1, state);
            TensorToArray(chunk).AsSpan()
                .CopyTo(logits.AsSpan(tokenOffset * cfg.VocabSize, chunkLen * cfg.VocabSize));
            tokenOffset += chunkLen;
        }
        if (tokenOffset != tokens.Length)
            throw new InvalidOperationException(
                $"Schedule sum ({tokenOffset}) does not match token count ({tokens.Length}).");
        return logits;
    }

    private static unsafe float[] TensorToArray(ITensor t)
    {
        int total = 1;
        for (int i = 0; i < t.Shape.Rank; i++) total *= t.Shape[i];
        float[] result = new float[total];
        new ReadOnlySpan<float>((void*)t.DataPointer, total).CopyTo(result);
        return result;
    }

    private static (float maxAbs, float maxRel, int worstIdx) DriftStats(float[] expected, float[] actual)
    {
        float maxAbs = 0f, maxRel = 0f;
        int worstIdx = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float absDiff = MathF.Abs(expected[i] - actual[i]);
            float relDiff = absDiff / (MathF.Abs(expected[i]) + 1e-12f);
            if (absDiff > maxAbs) { maxAbs = absDiff; worstIdx = i; }
            if (relDiff > maxRel) maxRel = relDiff;
        }
        return (maxAbs, maxRel, worstIdx);
    }

    private static void AssertClose(string label, float[] expected, float[] actual,
                                    float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float absDiff = MathF.Abs(e - a);
            float relDiff = absDiff / (MathF.Abs(e) + 1e-12f);
            if (absDiff > absTol && relDiff > relTol)
            {
                Assert.Fail(
                    $"{label}[{i}]: expected={e:F8} actual={a:F8} absDiff={absDiff:E3} "
                    + $"relDiff={relDiff:E3} (abs_tol={absTol:E0} rel_tol={relTol:E0})");
            }
        }
    }
}
