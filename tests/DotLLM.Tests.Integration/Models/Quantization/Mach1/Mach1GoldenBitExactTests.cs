using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Models.Quantization.Mach1;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Quantization.Mach1;

/// <summary>
/// Validates the Phase A codec decoder (issue #266) against the vendor's own
/// golden tensor: <c>goldens/L0_e0_fp32.safetensors</c> from
/// <c>SyzygyResearch/Mach-1-Additive-35B</c>. Resolved via
/// <c>DOTLLM_MACH1_35B_DIR</c>, falling back to the conventional
/// <c>~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B/</c> cache
/// path; skips (does not fail) if the fixture is absent, per this project's
/// fixture-storage conventions — no weights are ever committed to the repo.
/// </summary>
public sealed class Mach1GoldenBitExactTests
{
    [SkippableFact]
    public void DecodeExpertLayer0Expert0_MatchesVendorGolden_BitExact()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null,
            "Mach-1-Additive-35B fixture not found. Set DOTLLM_MACH1_35B_DIR or populate " +
            "~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B/ (see docs/QUANTIZATION.md).");

        string codecJsonPath = Path.Combine(root!, "packed", "experts", "codec.json");
        using JsonDocument codecDoc = JsonDocument.Parse(File.ReadAllText(codecJsonPath));
        JsonElement cbParamsEl = codecDoc.RootElement.GetProperty("cb_params");
        var cb = new Mach1CbParams(
            K: cbParamsEl.GetProperty("K").GetDouble(),
            L: cbParamsEl.GetProperty("L").GetInt32(),
            V: cbParamsEl.GetProperty("V").GetInt32(),
            TlutBits: cbParamsEl.GetProperty("tlut_bits").GetInt32(),
            TdX: cbParamsEl.GetProperty("td_x").GetInt32(),
            TdY: cbParamsEl.GetProperty("td_y").GetInt32());

        using var codebookFile = SafetensorsFile.Open(Path.Combine(root!, "packed", "experts", "codebook.safetensors"));
        var smallTlut = MemoryMarshal.Cast<byte, float>(codebookFile.GetTensorSpan("tlut"));

        using var layerFile = SafetensorsFile.Open(Path.Combine(root!, "packed", "experts", "L00.safetensors"));
        Assert.True(Mach1ExpertLayerDecoderV3T.IsV3TContainer(layerFile.Metadata),
            "L00.safetensors is expected to be the chunked v3t container (fields metadata contains 'wave_gamma').");

        var decoder = new Mach1ExpertLayerDecoderV3T(layerFile, smallTlut, cb);

        using var goldenFile = SafetensorsFile.Open(Path.Combine(root!, "goldens", "L0_e0_fp32.safetensors"));

        AssertProjectionMatchesGolden(decoder, goldenFile, "gate", m0: 512, n0: 2048, goldenKey: "L0.e0.gate.fp32");
        AssertProjectionMatchesGolden(decoder, goldenFile, "up", m0: 512, n0: 2048, goldenKey: "L0.e0.up.fp32");
        AssertProjectionMatchesGolden(decoder, goldenFile, "down", m0: 2048, n0: 512, goldenKey: "L0.e0.down.fp32");
    }

    private static void AssertProjectionMatchesGolden(
        Mach1ExpertLayerDecoderV3T decoder, SafetensorsFile goldenFile, string proj, int m0, int n0, string goldenKey)
    {
        var decoded = new float[m0 * n0];
        decoder.DecodeExpertProjection(expertIndex: 0, proj, m0, n0, decoded);

        var golden = MemoryMarshal.Cast<byte, float>(goldenFile.GetTensorSpan(goldenKey));
        Assert.Equal(golden.Length, decoded.Length);

        int mismatches = 0;
        double maxAbsDiff = 0;
        int firstMismatchIndex = -1;
        for (int i = 0; i < decoded.Length; i++)
        {
            float expected = golden[i];
            float actual = decoded[i];
            if (BitConverter.SingleToInt32Bits(expected) != BitConverter.SingleToInt32Bits(actual))
            {
                mismatches++;
                if (firstMismatchIndex < 0)
                    firstMismatchIndex = i;
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs((double)expected - actual));
            }
        }

        Assert.True(mismatches == 0,
            $"{proj}: {mismatches}/{decoded.Length} elements are not bit-exact vs the golden " +
            $"(max abs diff {maxAbsDiff:g9}, first mismatch at flat index {firstMismatchIndex}: " +
            $"expected {golden[Math.Max(firstMismatchIndex, 0)]}, got {decoded[Math.Max(firstMismatchIndex, 0)]}).");
    }

    private static string? ResolveFixtureRoot()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MACH1_35B_DIR");
        if (!string.IsNullOrWhiteSpace(env) && IsValidFixtureRoot(env))
            return env;

        string conventional = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "SyzygyResearch", "Mach-1-Additive-35B");
        return IsValidFixtureRoot(conventional) ? conventional : null;
    }

    private static bool IsValidFixtureRoot(string path)
    {
        return Directory.Exists(path)
            && File.Exists(Path.Combine(path, "packed", "experts", "codec.json"))
            && File.Exists(Path.Combine(path, "packed", "experts", "codebook.safetensors"))
            && File.Exists(Path.Combine(path, "packed", "experts", "L00.safetensors"))
            && File.Exists(Path.Combine(path, "goldens", "L0_e0_fp32.safetensors"));
    }
}
