using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #241 step 1 — MEASURE BEFORE CHANGING ANYTHING. Reports the wave
/// (subgroup) width the driver ACTUALLY compiled the Q8_0 MMQ prefill pipeline
/// for, via <c>VK_KHR_pipeline_executable_properties</c>.
/// </summary>
/// <remarks>
/// <para>
/// Subgroup size is invisible to the three instruments the #384-#391 MMQ
/// investigation already exhausted: black-box timing A/Bs, SPIR-V disassembly,
/// and <c>VK_AMD_shader_info</c>'s statistics (which report VGPR/SGPR/LDS/scratch
/// but not wave width). Whatever this prints is a standing unknown closed —
/// including a null result.
/// </para>
/// <para>
/// The wave32-pinned Q8_0 <em>MMVQ</em> decode pipeline is queried alongside as a
/// POSITIVE CONTROL: it is pinned to 32 by <see cref="Wave32SubgroupControl"/>, so
/// if the query does not report 32 for it, the query itself is not trustworthy and
/// the MMQ number means nothing.
/// </para>
/// <para>Enable with <c>DOTLLM_MMQ_SUBGROUP_BENCH=1</c>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMmqSubgroupSizeBench
{
    private const string Wave32EnvVar = "DOTLLM_VULKAN_DISABLE_WAVE32";

    private readonly ITestOutputHelper _output;
    public VulkanMmqSubgroupSizeBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Bench_MmqCompiledSubgroupSize()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_MMQ_SUBGROUP_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_MMQ_SUBGROUP_BENCH=1 to enable this diagnostic.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device lacks integer-dot-product support — MMQ unavailable.");
        Skip.IfNot(device.HasPipelineExecutableProperties,
            "Device does not advertise VK_KHR_pipeline_executable_properties — cannot read the compiled wave width.");

        _output.WriteLine($"Device: {device.DeviceName}");
        _output.WriteLine($"  VkPhysicalDeviceSubgroupProperties.subgroupSize (device default): {device.SubgroupSize}");
        _output.WriteLine($"  subgroup-size-control: supported={device.HasSubgroupSizeControl} min={device.MinSubgroupSize} max={device.MaxSubgroupSize}");
        _output.WriteLine("");

        // ── The measurement: MMQ, untouched, driver's own choice ──────────
        using (var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported."))
        {
            Report("matmul_q8_0_mmq.comp (production, requiredSubgroupSize UNSET)", device, mmq.PipelineHandle);
        }

        // ── Positive control: the SAME kernel with the pin on and off ─────
        // Wave32SubgroupControl pins matmul_q8_0_mmvq to 32; DOTLLM_VULKAN_DISABLE_WAVE32=1
        // is its documented opt-out. If the pinned and unpinned builds of this
        // kernel are indistinguishable to an instrument, that instrument cannot
        // answer the MMQ question either — #54/#330 measured a real, repeatable
        // decode delta from exactly this pin, so it demonstrably takes effect.
        string? saved = Environment.GetEnvironmentVariable(Wave32EnvVar);
        try
        {
            Environment.SetEnvironmentVariable(Wave32EnvVar, "1");
            using (var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir))
                if (mmvq is not null)
                    Report("matmul_q8_0_mmvq.comp (32-thread wg, pin UNSET — driver default)", device, mmvq.PipelineHandle);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Wave32EnvVar, saved);
        }
        foreach (uint pin in new uint[] { 32, 64 })
        {
            using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir, pin);
            if (mmvq is not null)
                Report($"matmul_q8_0_mmvq.comp (32-thread wg, requiredSubgroupSize={pin})", device, mmvq.PipelineHandle);
        }

        // ── Explicit pins, to confirm the pin takes on THIS kernel ────────
        foreach (uint pin in new uint[] { 32, 64 })
        {
            if (!device.SupportsRequiredSubgroupSize(pin, DotLLM.Vulkan.Interop.VkShaderStageFlags.Compute))
            {
                _output.WriteLine($"matmul_q8_0_mmq.comp pinned to {pin}: NOT SUPPORTED on this device.");
                continue;
            }
            using var pinned = MatMulQ8_0MmqKernel.TryCreate(device, spvDir, pin);
            if (pinned is not null)
                Report($"matmul_q8_0_mmq.comp (requiredSubgroupSize={pin})", device, pinned.PipelineHandle);
        }
    }

    private void Report(string label, VulkanDevice device, nint pipeline)
    {
        var execs = device.GetPipelineSubgroupSizes(pipeline);
        _output.WriteLine(label);
        if (execs.Count == 0)
        {
            _output.WriteLine("  (driver reported zero executables)");
        }
        else
        {
            foreach (var e in execs)
                _output.WriteLine($"  pipeline-executable subgroupSize = {e.SubgroupSize}  (name '{e.Name}')");
        }

        // VK_KHR_pipeline_executable_properties turns out to report the WORKGROUP
        // size on this driver (256 for a 16x16 kernel — not a legal wave width),
        // so it cannot answer the question on its own. The AMD ISA disassembly
        // carries the real wavefront size in its PAL metadata.
        if (!device.HasShaderInfoAmd) return;
        string disasm;
        try { disasm = device.GetShaderDisassemblyAmd(pipeline); }
        catch (Exception ex) { _output.WriteLine($"  (disassembly unavailable: {ex.Message})"); return; }

        // VK_AMD_shader_info statistics: a wave32 vs wave64 compile of the same
        // shader almost always differs in VGPR allocation, so a change here is
        // independent corroboration that the pin reached the ISA backend.
        var st = device.GetShaderStatisticsAmd(pipeline).resourceUsage;
        _output.WriteLine($"  stats: VGPR={st.numUsedVgprs} SGPR={st.numUsedSgprs} LDS={st.ldsUsageSizeInBytes} scratch={st.scratchMemUsageInBytes}");

        // RDNA wave32 vs wave64 is decidable straight from the ISA text: the exec
        // mask is 32-bit (`exec_lo`, `s_*_b32 exec_lo`) in wave32 and 64-bit
        // (`exec`, `s_*_b64 exec`) in wave64. Counting both tells us the wave
        // width without needing any metadata field the driver may not populate.
        int execLo = Count(disasm, "exec_lo");
        int execFull = Count(disasm, "exec") - execLo;
        int b64 = Count(disasm, "_b64");
        int b32 = Count(disasm, "_b32");
        _output.WriteLine($"  disassembly: {disasm.Length} chars, sha1={Sha1(disasm)}");
        _output.WriteLine($"    exec_lo={execLo} bare-exec={execFull}  _b32={b32} _b64={b64}"
            + $"  => wave{(execLo > 0 && execFull == 0 ? "32" : execFull > 0 && execLo == 0 ? "64" : "?? (mixed/inconclusive)")}");

        string? outDir = Environment.GetEnvironmentVariable("DOTLLM_MMQ_SUBGROUP_DUMP_DIR");
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
            string file = Path.Combine(outDir,
                string.Concat(label.Split(Path.GetInvalidFileNameChars())) + ".isa.txt");
            File.WriteAllText(file, disasm);
            _output.WriteLine($"    dumped -> {file}");
        }
    }

    private static int Count(string haystack, string needle)
    {
        int n = 0, i = 0;
        while ((i = haystack.IndexOf(needle, i, StringComparison.Ordinal)) >= 0) { n++; i += needle.Length; }
        return n;
    }

    private static string Sha1(string s) =>
        Convert.ToHexString(System.Security.Cryptography.SHA1.HashData(System.Text.Encoding.UTF8.GetBytes(s)))[..12];
}
