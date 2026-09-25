using System.Text.Json;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Minimal-reference regression net for the kernels that survive the
/// canonical pivot (<see cref="Mamba3QkNorm"/>). The minimal fixtures
/// (<c>fixture.json</c>, produced by <c>capture_fixtures.py</c>) exercise
/// the simplified VikramKarLex algebra — the kernels that still exist
/// after Stage P2b are a superset of what those fixtures test, so this
/// suite reads as "does the kept kernel still match the minimal ground
/// truth on the slice it originally validated?"
/// </summary>
/// <remarks>
/// The bulk of Mamba-3 correctness validation lives in
/// <see cref="Mamba3CanonicalReferenceCompareTests"/> against the
/// <c>state-spaces/mamba</c> canonical capture. Do not re-add
/// Discretize / SelectiveScan / minimal-Block comparators here — those
/// kernels were deleted in Stage P2b.
/// </remarks>
public class Mamba3ReferenceCompareTests
{
    private const float AbsTol = 1e-6f;
    private const float RelTol = 1e-5f;

    private static string FixturePath() =>
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "Mamba3", "fixture.json");

    private static Fixture LoadFixture()
    {
        string fullPath = Path.GetFullPath(FixturePath());
        if (!File.Exists(fullPath))
            throw new FileNotFoundException(
                $"Mamba-3 minimal fixture not found at {fullPath}. " +
                "Run tests/DotLLM.Tests.Integration/Fixtures/Mamba3/capture_fixtures.py first.");
        using var stream = File.OpenRead(fullPath);
        return JsonSerializer.Deserialize<Fixture>(stream, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        }) ?? throw new InvalidDataException("Fixture deserialised to null.");
    }

    [SkippableFact]
    public void QkNorm_MatchesReference()
    {
        Skip.IfNot(File.Exists(Path.GetFullPath(FixturePath())),
            "fixture.json missing — run capture_fixtures.py");

        var f = LoadFixture();
        int seqlen = f.Config["seqlen"].GetInt32();
        int dState = f.Config["d_state"].GetInt32();
        const float eps = 1e-5f;

        // Reference QK-norm uses a single RMSNorm(bc_dim) layer before group split;
        // in SISO bc_dim == d_state and there is no group dim, so our [T, nGroup=1, dState] layout matches.
        int nGroup = 1;
        float[] bIn = (float[])f.Inputs["B_raw"].Data.Clone();
        float[] cIn = (float[])f.Inputs["C_raw"].Data.Clone();
        float[] bWeight = f.Inputs["B_norm_weight"].Data;
        float[] cWeight = f.Inputs["C_norm_weight"].Data;

        Mamba3QkNorm.Execute(bIn, bWeight, eps, seqlen, nGroup, dState);
        Mamba3QkNorm.Execute(cIn, cWeight, eps, seqlen, nGroup, dState);

        AssertCloseElementwise("B_qkn", f.Expected["B_qkn"].Data, bIn);
        AssertCloseElementwise("C_qkn", f.Expected["C_qkn"].Data, cIn);
    }

    private static void AssertCloseElementwise(string label, float[] expected, float[] actual)
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
                            + $"(tolerance abs={AbsTol:E0} rel={RelTol:E0})");
            }
        }
    }

    // ── Fixture DTOs ──────────────────────────────────────────────────────────

    private sealed class Fixture
    {
        public Dictionary<string, JsonElement> Config { get; set; } = default!;
        public Dictionary<string, Tensor> Inputs { get; set; } = default!;
        public Dictionary<string, Tensor> Activated { get; set; } = default!;
        public Dictionary<string, Tensor> Expected { get; set; } = default!;
    }

    private sealed class Tensor
    {
        public int[] Shape { get; set; } = [];
        public float[] Data { get; set; } = [];
    }
}
