using DotLLM.Core.Attention;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #1926 (P1): the driver's post-compile VGPR / SGPR / LDS / scratch numbers for every KV twin of the v2 grouped
/// split-KV decode pass (<c>attention_f32_splitkv_gqa_v2*.spv</c>), via <c>VK_AMD_shader_info</c>.
/// </summary>
/// <remarks>
/// <para>
/// Each module is compiled with the body's group specialization constant (<c>constant_id = 0</c>, #1926 E1b) unset and at 4 and 8, so
/// a row shows what the group bound costs in registers. A module that declares no such constant reports the same numbers on every
/// row (Vulkan ignores map entries for ids a module does not use). TurboQuant twins are compiled for tq3 and tq4 at head_dim 128.
/// </para>
/// <para>
/// <c>DOTLLM_1926_EXTRA_SPV</c> names a directory of extra candidate <c>.spv</c> files (any name containing the twin suffix, e.g.
/// <c>v2_kvq4_e1a.spv</c>) to probe beside the shipped ones without a rebuild.
/// </para>
/// <para>Enable with <c>DOTLLM_1926_SHADER_INFO=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe1926SplitKvV2ShaderInfoTests
{
    private readonly ITestOutputHelper _out;
    public Probe1926SplitKvV2ShaderInfoTests(ITestOutputHelper output) => _out = output;

    private const int SplitPushConstantBytes = 14 * sizeof(uint);

    [SkippableFact]
    public void V2Twins_ShaderStatistics()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_1926_SHADER_INFO"), "1", StringComparison.Ordinal),
            "DOTLLM_1926_SHADER_INFO=1 to enable this diagnostic.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasShaderInfoAmd, "Device/driver does not advertise VK_AMD_shader_info.");

        var files = new List<string>();
        foreach (string suffix in new[] { "", "_kvf16", "_kvq8", "_kvq4", "_kvq8q4", "_kvq4q8", "_kvtq" })
            files.Add(Path.Combine(spvDir, "attention_f32_splitkv_gqa_v2" + suffix + ".spv"));
        string? extra = Environment.GetEnvironmentVariable("DOTLLM_1926_EXTRA_SPV");
        if (!string.IsNullOrEmpty(extra) && Directory.Exists(extra))
            files.AddRange(Directory.GetFiles(extra, "*.spv").OrderBy(f => f, StringComparer.Ordinal));

        _out.WriteLine($"Device: {device.DeviceName}");
        _out.WriteLine("| module | variant | GROUP | VGPR | SGPR | LDS used | scratch |");
        _out.WriteLine("|---|---|---:|---:|---:|---:|---:|");
        foreach (string path in files)
        {
            string name = Path.GetFileNameWithoutExtension(path);
            if (!File.Exists(path)) { _out.WriteLine($"| {name} | missing | | | | | |"); continue; }
            TurboQuantLayout?[] layouts = name.Contains("kvtq", StringComparison.Ordinal)
                ? [new TurboQuantLayout(128, 8, 3, TurboQuantLayout.DefaultNorm), new TurboQuantLayout(128, 8, 4, TurboQuantLayout.DefaultNorm)]
                : [null];
            using var module = VulkanModule.LoadFromFile(device, path);
            foreach (TurboQuantLayout? tq in layouts)
            {
                foreach (uint? group in new uint?[] { null, 4u, 8u })
                {
                    ReadOnlySpan<uint> spec = group is { } g ? [g] : default;
                    string variant = tq is { } l ? $"tq{l.MseBits}" : "-";
                    try
                    {
                        using ComputePipeline p = TqKvTwin.CreatePipeline(module, 5, SplitPushConstantBytes, tq, spec: spec);
                        var s = device.GetShaderStatisticsAmd(p.Pipeline);
                        _out.WriteLine($"| {name} | {variant} | {(group?.ToString() ?? "unset")} | {s.resourceUsage.numUsedVgprs} | " +
                                       $"{s.resourceUsage.numUsedSgprs} | {s.resourceUsage.ldsUsageSizeInBytes} | {s.resourceUsage.scratchMemUsageInBytes} |");
                    }
                    catch (Exception ex)
                    {
                        _out.WriteLine($"| {name} | {variant} | {group} | FAILED ({ex.GetType().Name}: {ex.Message}) | | | |");
                    }
                }
            }
        }
    }
}
