using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Loader / dispatch pairing (#921). <see cref="VulkanQwen3MoeHybridWeights"/> is shared by the MoE-hybrid and hybrid-dense hosts (and,
/// through <c>Q4RecordMatmul</c>, by LFM2-MoE / LFM3): a keep-packed arm there without a <c>RecordMatmul</c> arm in EITHER host is an
/// inference-time throw or - worse - a fall-through to the F32 `default` that reinterprets packed blocks as floats (that is how PQ2_0
/// broke once). Every quantization the loader keeps packed must therefore name a <c>case QuantizationType.X:</c> in both hosts' dispatch
/// tables. Source-text check, no GPU; it fails the moment a Keep* arm lands alone (verified by removing the Q5_1 arm from one host).
/// </summary>
public sealed class VulkanHybridKeepPackedDispatchTests
{
    private static readonly string[] Hosts =
    [
        Path.Combine("src", "DotLLM.Vulkan", "VulkanQwen3MoeHybridTransformerModel.cs"),
        Path.Combine("src", "DotLLM.Vulkan", "VulkanQwen3HybridDenseTransformerModel.cs"),
    ];

    [Fact]
    public void EveryKeptQuant_HasADispatchArmInBothHybridHosts()
    {
        string root = FindRepoRoot();
        var kept = new List<QuantizationType>();
        foreach (QuantizationType qt in Enum.GetValues<QuantizationType>())
        {
            // 256 and 1024 are multiples of every block / group size in the table (32, 128, 256), so alignment never masks a kept type.
            if (VulkanQwen3MoeHybridWeights.DeviceQuantTypeForProjection(qt, 1024) == qt && qt != QuantizationType.F32)
                kept.Add(qt);
        }
        Assert.Contains(QuantizationType.Q5_1, kept);   // the #921 arms are in the table
        Assert.Contains(QuantizationType.Q4_0, kept);
        Assert.Contains(QuantizationType.Q4_K, kept);

        var missing = new List<string>();
        foreach (string rel in Hosts)
        {
            string text = File.ReadAllText(Path.Combine(root, rel));
            // Strip the comments so a mention in prose does not count as an arm.
            text = Regex.Replace(text, @"//[^\n]*", "", RegexOptions.None, TimeSpan.FromSeconds(10));
            foreach (var qt in kept)
                if (!Regex.IsMatch(text, $@"case\s+QuantizationType\.{Regex.Escape(qt.ToString())}\s*:", RegexOptions.None, TimeSpan.FromSeconds(10)))
                    missing.Add($"{Path.GetFileName(rel)} lacks `case QuantizationType.{qt}:`");
        }
        Assert.True(missing.Count == 0, string.Join("; ", missing));
    }

    private static string FindRepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "CLAUDE.md")) && Directory.Exists(Path.Combine(d.FullName, "src", "DotLLM.Vulkan")))
                return d.FullName;
        throw new DirectoryNotFoundException("repo root (CLAUDE.md + src/DotLLM.Vulkan) not found above " + AppContext.BaseDirectory);
    }
}
