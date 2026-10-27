using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Interop;

/// <summary>#1336: the Windows HIP runtime search order (user-space ROCm first, legacy 5.7 shim last).</summary>
public sealed class HipRuntimeCandidateTests
{
    private static readonly string Sdk = Path.Combine("C:", "rocm");

    [Fact]
    public void NoEnvironment_VersionedDriverDllsBeforeTheLegacyBareName()
    {
        var list = HipLibraryResolver.WindowsRuntimeCandidates(_ => null, _ => false, "").ToList();
        Assert.Equal(["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"], list);
    }

    [Fact]
    public void UserSpaceRocm_IsTriedBeforeSystemDlls_NewestFirst()
    {
        string bin = Path.Combine(Sdk, "bin");
        var present = new HashSet<string>
        {
            Path.Combine(bin, "amdhip64_7.dll"),
            Path.Combine(bin, "amdhip64_6.dll"),
        };
        var list = HipLibraryResolver.WindowsRuntimeCandidates(
            k => k == "HIP_PATH" ? Sdk : null, present.Contains, "").ToList();
        Assert.Equal(Path.Combine(bin, "amdhip64_7.dll"), list[0]);
        Assert.Equal(Path.Combine(bin, "amdhip64_6.dll"), list[1]);
        Assert.Equal(["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"], list.Skip(2));
    }

    [Fact]
    public void DotllmOverride_WinsOverHipPath()
    {
        string a = Path.Combine("C:", "a"), b = Path.Combine("C:", "b");
        var present = new HashSet<string> { Path.Combine(a, "bin", "amdhip64_7.dll"), Path.Combine(b, "bin", "amdhip64_7.dll") };
        var list = HipLibraryResolver.WindowsRuntimeCandidates(
            k => k == "DOTLLM_HIP_PATH" ? a : k == "HIP_PATH" ? b : null, present.Contains, "").ToList();
        Assert.Equal(Path.Combine(a, "bin", "amdhip64_7.dll"), list[0]);
        Assert.Equal(Path.Combine(b, "bin", "amdhip64_7.dll"), list[1]);
    }

    [Fact]
    public void MissingFiles_AreSkipped()
    {
        var list = HipLibraryResolver.WindowsRuntimeCandidates(k => k == "ROCM_PATH" ? Sdk : null, _ => false, "").ToList();
        Assert.Equal(3, list.Count);
    }
}
