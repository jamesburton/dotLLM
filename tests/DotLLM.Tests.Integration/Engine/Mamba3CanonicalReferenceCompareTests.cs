using System.Text.Json;
using DotLLM.Cpu.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Cross-reference the <b>canonical</b> Mamba-3 kernel additions
/// (<see cref="Mamba3DataRoPE"/>'s <c>ExecuteCanonical</c> and
/// <see cref="Mamba3CanonicalSsd"/>) against the <c>state-spaces/mamba</c>
/// reference emitted by
/// <c>tests/DotLLM.Tests.Integration/Fixtures/Mamba3/capture_fixtures_canonical.py</c>.
/// Kept deliberately side-by-side with <see cref="Mamba3ReferenceCompareTests"/>
/// (minimal-reference) — the minimal suite stays as kernel-regression net until
/// Stage P2b retires the minimal kernels entirely.
/// </summary>
public class Mamba3CanonicalReferenceCompareTests
{
    // Pure-python canonical scan reconstructs the Triton kernel's inner loop
    // algebraically, in F32; observed drift vs the C# port is comparable to
    // the minimal-reference comparator (~1e-7 in most tensors). Hold the
    // minimal-reference tolerance (1e-6 / 1e-5) — if fixture noise rises, we
    // loosen here with a comment, not the minimal suite.
    private const float AbsTol = 1e-6f;
    private const float RelTol = 1e-5f;

    private readonly ITestOutputHelper _output;

    public Mamba3CanonicalReferenceCompareTests(ITestOutputHelper output) => _output = output;

    private static string FixturePath(string name) =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Mamba3", name);

    private static Fixture LoadFixture(string name)
    {
        string fullPath = Path.GetFullPath(FixturePath(name));
        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"Canonical fixture not found at {fullPath}. " +
                "Run tests/.../Fixtures/Mamba3/capture_fixtures_canonical.py first.");
        using var stream = File.OpenRead(fullPath);
        var f = JsonSerializer.Deserialize<Fixture>(stream, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        }) ?? throw new InvalidDataException("Canonical fixture deserialised to null.");
        return f;
    }

    // ------------------------------------------------------------------------
    // DataRoPE canonical — SISO (rotate_pairwise=True)
    // ------------------------------------------------------------------------
    [SkippableFact]
    public void DataRoPE_Canonical_Siso_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath("fixture_canonical.json"))),
            "fixture_canonical.json missing — run capture_fixtures_canonical.py");

        var f = LoadFixture("fixture_canonical.json");
        int seqlen = f.Config["seqlen"].GetInt32();
        int nheads = f.Config["nheads"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();
        int numRopeAngles = f.Config["num_rope_angles"].GetInt32();
        int effectiveRank = f.Config["effective_rank"].GetInt32();
        Assert.False(f.Config["is_mimo"].GetBoolean());
        Assert.Equal(1, effectiveRank);

        // Canonical's B_biased / C_biased are post-RMSnorm+bias, shape [B, L, R, H, N]
        // with R=1 here — flat layout matches [T, 1, nHead, dState] for the kernel.
        float[] bIn = (float[])f.PostDerivation["B_biased"].Data.Clone();
        float[] cIn = (float[])f.PostDerivation["C_biased"].Data.Clone();
        float[] anglesRaw = f.PostSplitRaw["angles_raw"].Data;   // [B, L, S]
        float[] dt = f.PostDerivation["DT"].Data;                // [B, L, H]

        Mamba3DataRoPE.ExecuteCanonical(
            bIn, cIn,
            anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty,
            cumAngleOut: Span<float>.Empty,
            seqLen: seqlen,
            nRank: effectiveRank,
            nHead: nheads,
            dState: dState,
            numRopeAngles: numRopeAngles,
            mode: Mamba3RoPEMode.Pairwise);

        _output.WriteLine(DriftStats("B_post_rope (SISO)", f.PostDerivation["B_post_rope"].Data, bIn));
        _output.WriteLine(DriftStats("C_post_rope (SISO)", f.PostDerivation["C_post_rope"].Data, cIn));

        AssertClose("B_post_rope (SISO)", f.PostDerivation["B_post_rope"].Data, bIn);
        AssertClose("C_post_rope (SISO)", f.PostDerivation["C_post_rope"].Data, cIn);
    }

    // ------------------------------------------------------------------------
    // DataRoPE canonical — MIMO (rotate_pairwise=False / halved)
    // ------------------------------------------------------------------------
    [SkippableFact]
    public void DataRoPE_Canonical_Mimo_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath("fixture_canonical_mimo.json"))),
            "fixture_canonical_mimo.json missing — run capture_fixtures_canonical.py");

        var f = LoadFixture("fixture_canonical_mimo.json");
        int seqlen = f.Config["seqlen"].GetInt32();
        int nheads = f.Config["nheads"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();
        int numRopeAngles = f.Config["num_rope_angles"].GetInt32();
        int effectiveRank = f.Config["effective_rank"].GetInt32();
        Assert.True(f.Config["is_mimo"].GetBoolean());
        Assert.Equal(2, effectiveRank);

        float[] bIn = (float[])f.PostDerivation["B_biased"].Data.Clone();
        float[] cIn = (float[])f.PostDerivation["C_biased"].Data.Clone();
        float[] anglesRaw = f.PostSplitRaw["angles_raw"].Data;
        float[] dt = f.PostDerivation["DT"].Data;

        Mamba3DataRoPE.ExecuteCanonical(
            bIn, cIn,
            anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty,
            cumAngleOut: Span<float>.Empty,
            seqLen: seqlen,
            nRank: effectiveRank,
            nHead: nheads,
            dState: dState,
            numRopeAngles: numRopeAngles,
            mode: Mamba3RoPEMode.Halved);

        _output.WriteLine(DriftStats("B_post_rope (MIMO)", f.PostDerivation["B_post_rope"].Data, bIn));
        _output.WriteLine(DriftStats("C_post_rope (MIMO)", f.PostDerivation["C_post_rope"].Data, cIn));

        AssertClose("B_post_rope (MIMO)", f.PostDerivation["B_post_rope"].Data, bIn);
        AssertClose("C_post_rope (MIMO)", f.PostDerivation["C_post_rope"].Data, cIn);
    }

    // ------------------------------------------------------------------------
    // Canonical SSD scan — SISO
    // ------------------------------------------------------------------------
    [SkippableFact]
    public void CanonicalSsd_Siso_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath("fixture_canonical.json"))),
            "fixture_canonical.json missing — run capture_fixtures_canonical.py");

        var f = LoadFixture("fixture_canonical.json");
        int seqlen = f.Config["seqlen"].GetInt32();
        int nHead = f.Config["nheads"].GetInt32();
        int headDim = f.Config["headdim"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();

        // Inputs (already-derived tensors from fixture).
        float[] v = f.PostSplitRaw["x_raw"].Data;                 // [B, L, d_inner] row-major == [T, H, P] flat
        float[] qRoped = f.PostDerivation["C_post_rope"].Data;    // [B, L, 1, H, N] → [T, H, N]
        float[] kRoped = f.PostDerivation["B_post_rope"].Data;
        float[] qkPre = f.PostDerivation["qk_pre_dot_per_rank"].Data; // [B, L, 1, H] → [T, H]  (SISO rank=1)
        float[] scale = f.PostDerivation["scale"].Data;           // [B, L, H]
        float[] gamma = f.PostDerivation["gamma"].Data;
        float[] adt = f.PostDerivation["ADT"].Data;
        float[] d = f.Inputs["D"].Data;
        float[] z = f.PostSplitRaw["z_raw"].Data;                 // [T, H, P]

        float[] state = new float[nHead * headDim * dState];
        float[] y = new float[seqlen * nHead * headDim];

        Mamba3CanonicalSsd.ExecuteSiso(
            state, v, qRoped, kRoped, qkPre, scale, gamma, adt, d, z, y,
            seqlen, nHead, headDim, dState);

        // Expected y from canonical (pre-outproj, same flat layout as our y).
        _output.WriteLine(DriftStats("y_pre_outproj (SISO)", f.Outputs["y_pre_outproj"].Data, y));
        _output.WriteLine(DriftStats("ssm_state (SISO)", f.SsmState["ssm_state_out"].Data, state));

        AssertClose("y_pre_outproj (SISO)", f.Outputs["y_pre_outproj"].Data, y);
        AssertClose("ssm_state (SISO)", f.SsmState["ssm_state_out"].Data, state);
    }

    // ------------------------------------------------------------------------
    // Canonical SSD scan — MIMO
    // ------------------------------------------------------------------------
    [SkippableFact]
    public void CanonicalSsd_Mimo_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath("fixture_canonical_mimo.json"))),
            "fixture_canonical_mimo.json missing — run capture_fixtures_canonical.py");

        var f = LoadFixture("fixture_canonical_mimo.json");
        int seqlen = f.Config["seqlen"].GetInt32();
        int nHead = f.Config["nheads"].GetInt32();
        int headDim = f.Config["headdim"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();
        int nRank = f.Config["effective_rank"].GetInt32();
        Assert.Equal(2, nRank);

        float[] v = f.PostSplitRaw["x_raw"].Data;                 // [T, H, P]
        float[] qRoped = f.PostDerivation["C_post_rope"].Data;    // [T, R, H, N]
        float[] kRoped = f.PostDerivation["B_post_rope"].Data;
        float[] qkPreSum = f.PostDerivation["qk_pre_dot_sum"].Data; // [T, H]
        float[] scale = f.PostDerivation["scale"].Data;
        float[] gamma = f.PostDerivation["gamma"].Data;
        float[] adt = f.PostDerivation["ADT"].Data;
        float[] d = f.Inputs["D"].Data;
        float[] z = f.PostSplitRaw["z_raw"].Data;
        float[] mimoZ = f.Inputs["mimo_z"].Data;                  // [H, R, P]
        float[] mimoO = f.Inputs["mimo_o"].Data;

        float[] state = new float[nHead * headDim * dState];
        float[] y = new float[seqlen * nHead * headDim];
        float[] yPerRank = new float[seqlen * nRank * nHead * headDim];

        Mamba3CanonicalSsd.ExecuteMimo(
            state, v, qRoped, kRoped, qkPreSum, scale, gamma, adt, d, z,
            mimoZ, mimoO, y, yPerRank,
            seqlen, nRank, nHead, headDim, dState);

        _output.WriteLine(DriftStats("y_pre_contract (MIMO)", f.Outputs["y_pre_contract"].Data, yPerRank));
        _output.WriteLine(DriftStats("y_pre_outproj (MIMO)", f.Outputs["y_pre_outproj"].Data, y));
        _output.WriteLine(DriftStats("ssm_state (MIMO)", f.SsmState["ssm_state_out"].Data, state));

        AssertClose("y_pre_contract (MIMO)", f.Outputs["y_pre_contract"].Data, yPerRank);
        AssertClose("y_pre_outproj (MIMO)", f.Outputs["y_pre_outproj"].Data, y);
        AssertClose("ssm_state (MIMO)", f.SsmState["ssm_state_out"].Data, state);
    }

    // ------------------------------------------------------------------------
    // QkNorm on the canonical per-(R, G) slice layout. Since Mamba3QkNorm
    // RMS-normalizes every trailing [dState] slice independently, passing
    // [T*R*G, dState] flat through it reproduces canonical's
    // RMSNormGated(d_state) applied per (R, G) slice. Confirm here.
    // ------------------------------------------------------------------------
    [SkippableFact]
    public void QkNorm_Canonical_Mimo_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath("fixture_canonical_mimo.json"))),
            "fixture_canonical_mimo.json missing — run capture_fixtures_canonical.py");

        var f = LoadFixture("fixture_canonical_mimo.json");
        int seqlen = f.Config["seqlen"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();
        int effectiveRank = f.Config["effective_rank"].GetInt32();
        int numBcHeads = f.Config["num_bc_heads"].GetInt32();
        const float eps = 1e-5f;

        // B_raw slice in the fixture is shape [B, L, R*G*dState] flat from in_proj.
        // After the canonical rearrange `(b l (r g n) -> b l r g n)` the
        // row-major memory is already [T, R, G, dState]. Treat the whole thing
        // as a flat T*R*G x dState tensor for QkNorm.
        float[] bIn = (float[])f.PostSplitRaw["B_raw"].Data.Clone();
        float[] cIn = (float[])f.PostSplitRaw["C_raw"].Data.Clone();
        float[] bWeight = f.Inputs["B_norm_weight"].Data;
        float[] cWeight = f.Inputs["C_norm_weight"].Data;

        int sliceCount = seqlen * effectiveRank * numBcHeads;
        Mamba3QkNorm.Execute(bIn, bWeight, eps, sliceCount, nGroup: 1, dState);
        Mamba3QkNorm.Execute(cIn, cWeight, eps, sliceCount, nGroup: 1, dState);

        _output.WriteLine(DriftStats("B_post_norm", f.PostDerivation["B_post_norm"].Data, bIn));
        _output.WriteLine(DriftStats("C_post_norm", f.PostDerivation["C_post_norm"].Data, cIn));

        AssertClose("B_post_norm", f.PostDerivation["B_post_norm"].Data, bIn);
        AssertClose("C_post_norm", f.PostDerivation["C_post_norm"].Data, cIn);
    }

    // ────────────────────────────────────────────────────────────────────────
    // helpers
    // ────────────────────────────────────────────────────────────────────────
    private static string DriftStats(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f, maxRel = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]);
            float r = d / (MathF.Abs(expected[i]) + 1e-12f);
            if (d > maxAbs) maxAbs = d;
            if (r > maxRel) maxRel = r;
        }
        return $"{label}: max_abs={maxAbs:E3} max_rel={maxRel:E3}";
    }

    private static void AssertClose(string label, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float absDiff = MathF.Abs(e - a);
            float relDiff = absDiff / (MathF.Abs(e) + 1e-12f);
            if (absDiff > AbsTol && relDiff > RelTol)
            {
                Assert.Fail($"{label}[{i}]: expected={e:F8} actual={a:F8} absDiff={absDiff:E3} relDiff={relDiff:E3} "
                            + $"(tol abs={AbsTol:E0} rel={RelTol:E0})");
            }
        }
    }

    // ---- fixture DTOs (canonical fixture schema differs from minimal) ------
    private sealed class Fixture
    {
        public Dictionary<string, JsonElement> Config { get; set; } = default!;
        public Dictionary<string, Tensor> Inputs { get; set; } = default!;
        public Dictionary<string, Tensor> PostSplitRaw { get; set; } = default!;
        public Dictionary<string, Tensor> PostDerivation { get; set; } = default!;
        public Dictionary<string, Tensor> SsmState { get; set; } = default!;
        public Dictionary<string, Tensor> Outputs { get; set; } = default!;
    }

    private sealed class Tensor
    {
        public int[] Shape { get; set; } = [];
        public float[] Data { get; set; } = [];
    }
}
