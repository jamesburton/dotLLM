using DotLLM.Vulkan.DirectStorage;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

[Collection(CliConsoleCollection.Name)]
public sealed class DirectStorageInstallCommandTests
{
    [Fact]
    public void InstallHelp_IsRegisteredAndDescribesLicenseAcceptance()
    {
        var result = CliCommandModel.Run("directstorage", "install", "--help");
        Assert.Equal(0, result.ExitCode);
        Assert.Contains("license acceptance", result.Output, StringComparison.Ordinal);
    }

    [SkippableFact]
    public void ExistingRuntime_DoesNotRequireAnotherDownloadOrLicensePrompt()
    {
        Skip.IfNot(DirectStorageInstaller.IsAvailable(), "Existing DirectStorage runtime required.");
        Assert.Equal(0, CliCommandModel.Run("directstorage", "install").ExitCode);
    }
}
