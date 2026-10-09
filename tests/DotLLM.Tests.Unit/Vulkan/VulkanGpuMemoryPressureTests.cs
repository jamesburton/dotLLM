using System.Diagnostics;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #880: the residency gate must see GPU memory held by OTHER processes. <c>VK_EXT_memory_budget</c> cannot (measured: a second
/// process holding 20 GiB left this process's budget/usage untouched), so the gate reads the OS's per-process GPU counters.
/// Pure arithmetic and parsing here; the GPU tests below use a real second process.
/// </summary>
public sealed class VulkanGpuMemoryPressureArithmeticTests
{
    private const long GiB = 1L << 30;

    [Theory]
    [InlineData("pid_10840_luid_0x00000000_0x00010d52_phys_0", 10840, 0x00010d52UL)]
    [InlineData("pid_4_luid_0x00000001_0x0000ABCD_phys_1", 4, 0x000000010000ABCDUL)]
    public void ParsesProcessMemoryInstanceNames(string name, int pid, ulong luid)
    {
        Assert.True(VulkanGpuMemoryCounters.TryParseInstance(name, out int p, out ulong l));
        Assert.Equal(pid, p);
        Assert.Equal(luid, l);
    }

    [Theory]
    [InlineData("luid_0x00000000_0x00010d52_phys_0")]   // adapter-level instance, no pid
    [InlineData("pid_x_luid_0x0_0x1_phys_0")]
    [InlineData("")]
    public void RejectsMalformedInstanceNames(string name)
        => Assert.False(VulkanGpuMemoryCounters.TryParseInstance(name, out _, out _));

    [Fact]
    public void ExcludesOwnProcess_AndSortsLargestFirst()
    {
        var p = new VulkanGpuMemoryPressure(
        [
            new GpuProcessUsage(100, 10 * GiB, 0, "me"),
            new GpuProcessUsage(200, 5 * GiB, 1 * GiB, "ollama"),
            new GpuProcessUsage(300, 30 * GiB, 0, "llama-server"),
        ], ownPid: 100, integrated: true);
        Assert.Equal(36 * GiB, p.OtherBytes);
        Assert.Equal(300, p.Others[0].Pid);
        Assert.DoesNotContain(p.Others, o => o.Pid == 100);
        Assert.Contains("llama-server", p.DescribeCulprits(), StringComparison.Ordinal);
    }

    [Fact]
    public void DiscreteGpu_CountsDedicatedUsageOnly()
    {
        // On a dGPU the resident capacity is VRAM; another process's system-RAM "shared" usage does not compete with it.
        var procs = new[] { new GpuProcessUsage(200, 40 * GiB, 3 * GiB, "other") };
        Assert.Equal(3 * GiB, new VulkanGpuMemoryPressure(procs, 1, integrated: false).OtherBytes);
        Assert.Equal(43 * GiB, new VulkanGpuMemoryPressure(procs, 1, integrated: true).OtherBytes);
    }

    [Fact]
    public void Plan_RefusesWhenOthersPushItOverBudget_AndNamesShortfallAndCulprits()
    {
        var alone = new Qwen4ExpResidencyPlan(69 * GiB, 2 * GiB, 0, 104 * GiB, 6 * GiB, 127 * GiB);
        Assert.True(alone.Fits);
        var crowded = alone with { OtherProcessBytes = 70 * GiB, OtherProcessDetail = "pid 7 (dotllm) 70.0 GiB" };
        Assert.False(crowded.Fits);
        Assert.Equal(71 * GiB + 70 * GiB - 98 * GiB, crowded.ShortfallBytes);
        string text = crowded.DescribeShortfall();
        Assert.Contains("short by 43.0 GiB", text, StringComparison.Ordinal);
        Assert.Contains("pid 7 (dotllm)", text, StringComparison.Ordinal);
        foreach (string culprit in new[] { "llama.cpp", "Lemonade", "Docker", "ollama", "browser" })
            Assert.Contains(culprit, text, StringComparison.Ordinal);
        Assert.Contains("VK_ERROR_DEVICE_LOST", text, StringComparison.Ordinal);
    }

    [Fact]
    public void PostUploadShortfall_IsOursPlusOthersAgainstCapacityMinusHeadroom()
    {
        Assert.Equal(0, Qwen4ExpResidencyPlan.PostUploadShortfall(70 * GiB, 5 * GiB, 104 * GiB, 6 * GiB));
        Assert.Equal(2 * GiB, Qwen4ExpResidencyPlan.PostUploadShortfall(70 * GiB, 30 * GiB, 104 * GiB, 6 * GiB));
    }
}

/// <summary>#880 end to end on the GPU: a real second process holds device memory and the loader refuses with a clear error.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGpuMemoryPressureGpuTests
{
    private const long GiB = 1L << 30;
    private readonly ITestOutputHelper _out;
    public VulkanGpuMemoryPressureGpuTests(ITestOutputHelper output) => _out = output;

    private static string? FindRepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "tools", "vk-alloc-probe", "vk_alloc_probe.cs"))) return d.FullName;
        return null;
    }

    /// <summary>Builds <c>tools/vk-alloc-probe</c> and starts its <c>hold</c> mode as a separate process holding <paramref name="gib"/> GiB.</summary>
    private static Process StartHolder(string repo, string outDir, int gib)
    {
        var build = Process.Start(new ProcessStartInfo("dotnet",
            $"build \"{Path.Combine(repo, "tools", "vk-alloc-probe", "vk_alloc_probe.cs")}\" -o \"{outDir}\" -v q")
        { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false })!;
        build.StandardOutput.ReadToEnd(); build.StandardError.ReadToEnd();
        build.WaitForExit();
        Skip.IfNot(build.ExitCode == 0, "vk-alloc-probe did not build");
        var psi = new ProcessStartInfo(Path.Combine(outDir, "vk_alloc_probe.exe"),
            $"--mode hold --chunk 1024 --count {gib} --hold-sec 90")
        { RedirectStandardOutput = true, UseShellExecute = false };
        var holder = Process.Start(psi)!;
        string? line;
        var deadline = DateTime.UtcNow.AddSeconds(60);
        while ((line = holder.StandardOutput.ReadLine()) is not null)
        {
            if (line.Contains("held for", StringComparison.Ordinal)) return holder;
            if (DateTime.UtcNow > deadline) break;
        }
        holder.Kill(true);
        throw new InvalidOperationException("holder process did not report its hold");
    }

    [SkippableFact]
    public void SecondProcessHoldingMemory_IsVisible_AndGateRefusesInsteadOfDeviceLost()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(OperatingSystem.IsWindows(), "PDH GPU counters are Windows-only");
        string? repo = FindRepoRoot();
        Skip.If(repo is null, "tools/vk-alloc-probe not found");

        string dir = Path.Combine(Path.GetTempPath(), "dotllm-880-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "m.gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.F32));
        string? prior = Environment.GetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT");
        Process? holder = null;
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            Skip.IfNot(device.DeviceLuid() is not null, "driver reports no adapter LUID");
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", null);

            // 8 GiB capacity - 6 GiB headroom = a 2 GiB budget. The tiny model needs ~1 GiB (scratch allowance) so it fits ALONE...
            const long capacity = 8 * GiB;
            using (var alone = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, capacity,
                       otherPressureProbe: () => new VulkanGpuMemoryPressure([], 0, true)))
                Assert.NotNull(alone);

            // ...but not beside a second process that really holds 3 GiB of GPU memory.
            holder = StartHolder(repo!, Path.Combine(dir, "probe"), gib: 3);
            var seen = device.ReadOtherProcessPressure();
            Assert.NotNull(seen);
            var h = seen!.Others.FirstOrDefault(o => o.Pid == holder.Id);
            Assert.True(h.TotalBytes >= 2 * GiB, $"holder pid {holder.Id} not visible with >= 2 GiB; saw {seen.DescribeCulprits()}");
            Assert.True(seen.OtherBytes >= 3 * GiB);

            var ex = Assert.Throws<NotSupportedException>(() =>
                VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, capacity));
            _out.WriteLine(ex.Message);
            Assert.Contains("short by", ex.Message, StringComparison.Ordinal);
            Assert.Contains("vk_alloc_probe", ex.Message, StringComparison.Ordinal);
            Assert.Contains("DOTLLM_VK_ALLOW_OVERCOMMIT", ex.Message, StringComparison.Ordinal);
            Assert.DoesNotContain("DEVICE_LOST", ex.Message.Replace("VK_ERROR_DEVICE_LOST", ""), StringComparison.Ordinal);

            // The documented escape hatch still works.
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", "1");
            using var forced = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, capacity);
            Assert.NotNull(forced);
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", prior);
            try { holder?.Kill(true); holder?.WaitForExit(10_000); } catch (InvalidOperationException) { }
            holder?.Dispose();
            try { Directory.Delete(dir, true); } catch (Exception e) when (e is IOException or UnauthorizedAccessException) { }
        }
    }

    [SkippableFact]
    public void PressureAppearingDuringUpload_IsCaughtByThePostUploadCheck()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), "dotllm-880b-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, "m.gguf");
        File.WriteAllBytes(path, Qwen4ExpRandomGguf.Build(Qwen4ExpRandomGguf.Tiny, Q4eQuant.F32));
        string? prior = Environment.GetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT");
        try
        {
            using var gguf = GgufFile.Open(path);
            var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", null);
            int calls = 0;
            VulkanGpuMemoryPressure? Probe() => calls++ == 0
                ? new VulkanGpuMemoryPressure([], 0, true)                                                  // quiet before the load
                : new VulkanGpuMemoryPressure([new GpuProcessUsage(4242, 6 * GiB, 0, "llama-server")], 0, true);   // a neighbour arrived meanwhile
            var ex = Assert.Throws<NotSupportedException>(() =>
                VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir, 8 * GiB, Probe));
            _out.WriteLine(ex.Message);
            Assert.Equal(2, calls);
            Assert.Contains("after the qwen4exp upload", ex.Message, StringComparison.Ordinal);
            Assert.Contains("llama-server", ex.Message, StringComparison.Ordinal);
            Assert.Equal(0, device.LiveBytesTotal());   // the refused load released everything it uploaded
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT", prior);
            try { Directory.Delete(dir, true); } catch (IOException) { }
        }
    }
}
