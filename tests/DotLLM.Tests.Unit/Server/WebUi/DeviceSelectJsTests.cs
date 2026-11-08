using System.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Server.WebUi;

/// <summary>
/// #1704: the web UI's device-choice logic (<c>wwwroot/device-select.js</c>) is pure JavaScript, tested by running
/// <c>device-select.test.js</c> under node. Skips (visibly) where node is not installed; CI runners ship node.
/// </summary>
public sealed class DeviceSelectJsTests
{
    private static string? RepoRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "src", "DotLLM.Server", "wwwroot", "device-select.js"))) return d.FullName;
        return null;
    }

    [SkippableFact]
    public void DeviceSelectLogic_PassesItsNodeTests()
    {
        string? root = RepoRoot();
        Assert.NotNull(root);
        string script = Path.Combine(root!, "tests", "DotLLM.Tests.Unit", "Server", "WebUi", "device-select.test.js");
        string module = Path.Combine(root!, "src", "DotLLM.Server", "wwwroot", "device-select.js");

        Process p;
        try
        {
            p = Process.Start(new ProcessStartInfo("node", [script, module])
            { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false })!;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            Skip.If(true, "node is not installed.");
            return;
        }
        using (p)
        {
            string stdout = p.StandardOutput.ReadToEnd(), stderr = p.StandardError.ReadToEnd();
            p.WaitForExit();
            Assert.True(p.ExitCode == 0, stdout + stderr);
        }
    }
}
