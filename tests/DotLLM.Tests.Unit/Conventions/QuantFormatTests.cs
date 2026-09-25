using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Conventions;

/// <summary>
/// Anchors the <see cref="QuantFormat"/> table (issue #351) so the single source
/// of truth is itself pinned, not merely single: each format's block bytes are
/// restated field-by-field from llama.cpp's <c>ggml/src/ggml-common.h</c> block
/// structs, the group sizes from its <c>QK*</c> constants, and the row-stride
/// arithmetic is cross-checked against the CPU seed
/// (<see cref="Dequantize.RowByteSize"/>). A source-scan test additionally
/// cross-checks the GLSL shader constants — which cannot consume C# — against
/// the same table.
/// </summary>
public class QuantFormatTests
{
    /// <summary>
    /// llama.cpp block layouts, restated as sums of their struct fields
    /// (f16 = 2 bytes). A mismatch here means either the table or this
    /// transcription is wrong — resolve against ggml-common.h, never by
    /// editing one side to match the other.
    /// </summary>
    [Theory]
    // legacy 32-element blocks
    [InlineData(QuantizationType.Q4_0, 2 + 16, 32)]                 // d + qs[QK4_0/2]
    [InlineData(QuantizationType.Q4_1, 2 + 2 + 16, 32)]             // d + m + qs
    [InlineData(QuantizationType.Q5_0, 2 + 4 + 16, 32)]             // d + qh[4] + qs
    [InlineData(QuantizationType.Q5_1, 2 + 2 + 4 + 16, 32)]         // d + m + qh + qs
    [InlineData(QuantizationType.Q8_0, 2 + 32, 32)]                 // d + qs[QK8_0]
    [InlineData(QuantizationType.MXFP4, 1 + 16, 32)]                // e8m0 scale + qs
    [InlineData(QuantizationType.IQ4_NL, 2 + 16, 32)]               // d + qs
    // K-quant / IQ 256-element super-blocks
    [InlineData(QuantizationType.Q2_K, 16 + 64 + 2 + 2, 256)]       // scales[QK_K/16] + qs[QK_K/4] + d + dmin
    [InlineData(QuantizationType.Q3_K, 32 + 64 + 12 + 2, 256)]      // hmask[QK_K/8] + qs[QK_K/4] + scales[12] + d
    [InlineData(QuantizationType.Q4_K, 2 + 2 + 12 + 128, 256)]      // d + dmin + scales[K_SCALE_SIZE] + qs[QK_K/2]
    [InlineData(QuantizationType.Q5_K, 2 + 2 + 12 + 32 + 128, 256)] // d + dmin + scales + qh[QK_K/8] + qs[QK_K/2]
    [InlineData(QuantizationType.Q6_K, 128 + 64 + 16 + 2, 256)]     // ql[QK_K/2] + qh[QK_K/4] + scales[QK_K/16] + d
    [InlineData(QuantizationType.IQ1_S, 2 + 32 + 16, 256)]          // d + qs[QK_K/8] + qh[QK_K/32 × u16]
    [InlineData(QuantizationType.IQ2_XXS, 2 + 64, 256)]             // d + qs[QK_K/8 × u16]
    [InlineData(QuantizationType.IQ2_XS, 2 + 64 + 8, 256)]          // d + qs[QK_K/8 × u16] + scales[QK_K/32]
    [InlineData(QuantizationType.IQ2_S, 2 + 32 + 8 + 32 + 8, 256)]  // d + qs[QK_K/8] + qh[QK_K/32] + signs(qs hi)[QK_K/8] + scales
    [InlineData(QuantizationType.IQ3_XXS, 2 + 96, 256)]             // d + qs[3·QK_K/8]
    [InlineData(QuantizationType.IQ3_S, 2 + 64 + 8 + 32 + 4, 256)]  // d + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] + scales[IQ3S_N_SCALE]
    [InlineData(QuantizationType.IQ4_XS, 2 + 2 + 4 + 128, 256)]     // d + scales_h(u16) + scales_l[QK_K/64] + qs[QK_K/2]
    // ternary / 2-bit 128-code groups
    [InlineData(QuantizationType.I2_S, 128 / 4, 128)]               // 4 codes/byte; f32 scale at TENSOR tail, not in-block
    [InlineData(QuantizationType.PQ2_0, 2 + 128 / 4, 128)]          // f16 scale in-block + 4 codes/byte
    public void Info_MatchesLlamaCppBlockLayout(QuantizationType type, int expectedBlockBytes, int expectedGroupSize)
    {
        var info = QuantFormat.TryGetInfo(type);
        Assert.NotNull(info);
        Assert.Equal(expectedBlockBytes, info!.Value.BlockBytes);
        Assert.Equal(expectedGroupSize, info.Value.GroupSize);
    }

    [Fact]
    public void BlockFormats_IsExhaustiveOverTheTable_AndNonBlockTypesReturnNull()
    {
        foreach (var t in QuantFormat.BlockFormats)
            Assert.NotNull(QuantFormat.TryGetInfo(t));
        Assert.Equal(QuantFormat.BlockFormats.Length,
            QuantFormat.BlockFormats.ToArray().Distinct().Count());

        Assert.Null(QuantFormat.TryGetInfo(QuantizationType.F32));
        Assert.Null(QuantFormat.TryGetInfo(QuantizationType.F16));
        Assert.Null(QuantFormat.TryGetInfo(QuantizationType.BF16));
    }

    /// <summary>
    /// The table's row-stride arithmetic must agree with the CPU seed it
    /// centralises (<see cref="Dequantize.RowByteSize"/>) for every block format,
    /// at several element counts including odd block counts.
    /// </summary>
    [Fact]
    public void RowByteSize_AgreesWithCpuDequantizeSeed()
    {
        foreach (var t in QuantFormat.BlockFormats)
        {
            var info = QuantFormat.TryGetInfo(t)!.Value;
            foreach (int blocks in new[] { 1, 3, 7, 128 })
            {
                long elems = (long)blocks * info.GroupSize;
                Assert.Equal(Dequantize.RowByteSize(elems, t), info.RowByteSize(elems));
            }
        }
    }

    [Theory]
    [InlineData(QuantizationType.Q8_0, 8.5)]     // 34·8/32
    [InlineData(QuantizationType.Q4_K, 4.5)]     // 144·8/256
    [InlineData(QuantizationType.IQ2_XXS, 2.0625)] // 66·8/256
    [InlineData(QuantizationType.I2_S, 2.0)]     // 32·8/128
    public void BitsPerWeight_MatchesKnownFigures(QuantizationType type, double expected)
    {
        Assert.Equal(expected, QuantFormat.TryGetInfo(type)!.Value.BitsPerWeight, precision: 10);
    }

    /// <summary>
    /// Cross-checks the GLSL shader sources' block-byte constants against the
    /// table (#351 acceptance: shaders keep their own literals — they cannot
    /// consume C# — so a scan pins them instead). Named constants
    /// (<c>Q4K_BLOCK_BYTES</c>…) map by name; the handful of bare
    /// <c>BLOCK_BYTES</c> map by the shader's filename.
    /// </summary>
    [SkippableFact]
    public void VulkanShaderSources_BlockByteConstants_MatchTable()
    {
        string shaderDir = Path.Combine(FindRepoRoot(), "native", "vulkan", "shaders");
        Skip.If(!Directory.Exists(shaderDir), $"shader source dir not found: {shaderDir}");

        // normalized constant name -> expected bytes
        var byName = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            ["Q4_0"] = QuantFormat.Q4_0BlockBytes, ["Q4_1"] = QuantFormat.Q4_1BlockBytes,
            ["Q5_0"] = QuantFormat.Q5_0BlockBytes, ["Q5_1"] = QuantFormat.Q5_1BlockBytes,
            ["Q8_0"] = QuantFormat.Q8_0BlockBytes, ["Q8_1"] = QuantFormat.Q8_1BlockBytes,
            ["Q2K"] = QuantFormat.Q2_KBlockBytes, ["Q3K"] = QuantFormat.Q3_KBlockBytes,
            ["Q4K"] = QuantFormat.Q4_KBlockBytes, ["Q5K"] = QuantFormat.Q5_KBlockBytes,
            ["Q6K"] = QuantFormat.Q6_KBlockBytes, ["Q8K"] = QuantFormat.Q8_KBlockBytes,
            ["IQ1_S"] = QuantFormat.IQ1_SBlockBytes, ["IQ2_XXS"] = QuantFormat.IQ2_XXSBlockBytes,
            ["IQ2_XS"] = QuantFormat.IQ2_XSBlockBytes, ["IQ2_S"] = QuantFormat.IQ2_SBlockBytes,
            ["IQ3_XXS"] = QuantFormat.IQ3_XXSBlockBytes, ["IQ3_S"] = QuantFormat.IQ3_SBlockBytes,
            ["IQ4_NL"] = QuantFormat.IQ4_NLBlockBytes, ["IQ4_XS"] = QuantFormat.IQ4_XSBlockBytes,
            ["I2S"] = QuantFormat.I2_SBlockBytes, ["PQ2_0"] = QuantFormat.PQ2_0BlockBytes,
            ["MXFP4"] = QuantFormat.Mxfp4BlockBytes,
        };
        // filename fragment -> expected bytes, for bare `BLOCK_BYTES`
        var byFile = new (string Fragment, int Bytes)[]
        {
            ("q8_0", QuantFormat.Q8_0BlockBytes), ("q8_1", QuantFormat.Q8_1BlockBytes),
            ("q5_0", QuantFormat.Q5_0BlockBytes), ("q5_1", QuantFormat.Q5_1BlockBytes),
            ("q4_0", QuantFormat.Q4_0BlockBytes), ("q4_1", QuantFormat.Q4_1BlockBytes),
            ("iq4_nl", QuantFormat.IQ4_NLBlockBytes), ("mxfp4", QuantFormat.Mxfp4BlockBytes),
        };

        var re = new Regex(@"const uint (?<name>[A-Z0-9_]*?)_?BLOCK_BYTES\s*=\s*(?<val>\d+)",
            RegexOptions.Compiled, TimeSpan.FromSeconds(10));
        var mismatches = new List<string>();
        int checkedCount = 0;
        foreach (string file in Directory.EnumerateFiles(shaderDir, "*.comp"))
        {
            string text = File.ReadAllText(file);
            string name = Path.GetFileName(file);
            foreach (Match m in re.Matches(text))
            {
                string constName = m.Groups["name"].Value;
                int val = int.Parse(m.Groups["val"].Value);
                int? expected = null;
                if (constName.Length > 0 && byName.TryGetValue(constName, out int e))
                    expected = e;
                else if (constName.Length == 0)
                {
                    // Order matters: check longer fragments first is handled by
                    // the tuple order above (iq4_nl before nothing clashes here).
                    foreach (var (frag, bytes) in byFile)
                        if (name.Contains(frag, StringComparison.OrdinalIgnoreCase)) { expected = bytes; break; }
                }
                if (expected is null)
                {
                    mismatches.Add($"{name}: unmapped constant '{constName}_BLOCK_BYTES = {val}'");
                    continue;
                }
                checkedCount++;
                if (val != expected)
                    mismatches.Add($"{name}: {constName}_BLOCK_BYTES = {val}, table says {expected}");
            }
        }

        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
        Assert.True(checkedCount >= 80, // ~90 constants exist today; guard the scan itself against silently matching nothing
            $"shader scan only matched {checkedCount} constants — regex or layout drift, scan is no longer covering the sources");
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "dotLLM.slnx")))
                return dir.FullName;
            dir = dir.Parent!;
        }
        throw new InvalidOperationException("repo root not found from " + AppContext.BaseDirectory);
    }
}
