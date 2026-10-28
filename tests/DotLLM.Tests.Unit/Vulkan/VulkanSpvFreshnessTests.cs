using System.Collections.Concurrent;
using System.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Guards the tracked <c>native/vulkan/spv/*.spv</c> blobs (#1364): every one must be byte-identical to what the
/// pinned glslc (<c>native/vulkan/glslc.version</c>) produces from its <c>.comp</c> with
/// <c>--target-env=vulkan1.2 -O</c>. Needs no GPU; skips when glslc is missing or is not the pinned version
/// (SPIR-V bytes depend on the glslc version, so any other toolchain would report false drift).
/// </summary>
public sealed class VulkanSpvFreshnessTests
{
    [SkippableFact]
    public void TrackedSpv_MatchesPinnedGlslcOutput()
    {
        string vulkanDir = FindVulkanNativeDir();
        string pin = File.ReadAllLines(Path.Combine(vulkanDir, "glslc.version"))[0].Trim();
        Assert.False(string.IsNullOrEmpty(pin), "native/vulkan/glslc.version is empty");

        string? version = RunGlslc("--version");
        Skip.If(version is null, "glslc not on PATH");
        Skip.IfNot(version!.StartsWith(pin, StringComparison.Ordinal),
            $"glslc '{version}' is not the pinned '{pin}'");

        string tmp = Path.Combine(Path.GetTempPath(), "dotllm-spv-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmp);
        try
        {
            var stale = new ConcurrentBag<string>();
            Parallel.ForEach(Directory.GetFiles(Path.Combine(vulkanDir, "shaders"), "*.comp"), comp =>
            {
                string name = Path.GetFileNameWithoutExtension(comp);
                string outPath = Path.Combine(tmp, name + ".spv");
                string? err = RunGlslc($"--target-env=vulkan1.2 -O -o \"{outPath}\" \"{comp}\"");
                string tracked = Path.Combine(vulkanDir, "spv", name + ".spv");
                if (err is null || !File.Exists(outPath) || !File.Exists(tracked)
                    || !File.ReadAllBytes(outPath).AsSpan().SequenceEqual(File.ReadAllBytes(tracked)))
                    stale.Add(name);
            });
            Assert.True(stale.IsEmpty,
                "Tracked .spv out of date vs .comp (run `dotnet msbuild src/DotLLM.Vulkan -t:RegenerateVulkanShaders` with the pinned glslc and commit): "
                + string.Join(", ", stale.Order()));
        }
        finally { try { Directory.Delete(tmp, true); } catch (IOException) { } }
    }

    private static string FindVulkanNativeDir()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            string c = Path.Combine(d.FullName, "native", "vulkan");
            if (File.Exists(Path.Combine(c, "glslc.version"))) return c;
        }
        throw new DirectoryNotFoundException("native/vulkan/glslc.version not found above " + AppContext.BaseDirectory);
    }

    /// <summary>Returns the first output line on success, null on failure to start or non-zero exit.</summary>
    private static string? RunGlslc(string args)
    {
        try
        {
            using var p = Process.Start(new ProcessStartInfo("glslc", args)
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            })!;
            string o = p.StandardOutput.ReadToEnd();
            string e = p.StandardError.ReadToEnd();
            p.WaitForExit();
            if (p.ExitCode != 0) return null;
            return (o.Length > 0 ? o : e.Length > 0 ? e : "ok").Split('\n')[0].Trim();
        }
        catch (System.ComponentModel.Win32Exception) { return null; }
    }
}
