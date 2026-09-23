using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The #508/#438 composition rule, as a pure CPU test — no Vulkan device.
/// </summary>
/// <remarks>
/// #508 (alias the mmap'd pages) and #438 (stage, then unmap the GGUF) reach the same
/// ~7 GiB by opposite routes, and <b>you cannot unmap pages you have imported</b>. The
/// two are therefore mutually exclusive per tensor, and the whole point of routing every
/// model through one policy object is that the release decision can be read off the
/// import ledger instead of being guessed. These tests pin exactly that read:
/// <see cref="VulkanWeightImportPolicy.MayReleaseWholeHostMapping"/> must go false the
/// instant one tensor imports, and the staged ranges must remain individually
/// releasable regardless. A policy that let both be "on" would either lose the memory
/// win or unmap live device memory.
/// </remarks>
// The policy is process-wide static state that every weights load mutates, so this
// class must not run concurrently with the GPU tests that load models.
[Collection("VulkanKernels")]
public class VulkanWeightImportPolicyTests : IDisposable
{
    public VulkanWeightImportPolicyTests() => VulkanWeightImportPolicy.Reset();

    public void Dispose() => VulkanWeightImportPolicy.Reset();

    /// <summary>
    /// A load where nothing imported — the dGPU case, since #507 refuses the import
    /// there — is the case #438 may release wholesale.
    /// </summary>
    [Fact]
    public void AllStaged_PermitsWholeMappingRelease()
    {
        VulkanWeightImportPolicy.NoteStaged(0x1000, 4096);
        VulkanWeightImportPolicy.NoteStaged(0x2000, 8192);

        Assert.True(VulkanWeightImportPolicy.MayReleaseWholeHostMapping);
        Assert.Equal(2, VulkanWeightImportPolicy.StagedTensorCount);
        Assert.Equal(4096L + 8192L, VulkanWeightImportPolicy.StagedBytes);
        Assert.Equal(2, VulkanWeightImportPolicy.StagedSourceRanges.Count);
    }

    /// <summary>
    /// The discriminating assertion. One import is enough to make
    /// <c>gguf.Dispose()</c> unsafe for the whole file — the imported pages ARE device
    /// memory. Only the individually-listed staged ranges stay releasable.
    /// </summary>
    [Fact]
    public void AnySingleImport_BlocksWholeMappingRelease_ButNotTheStagedRanges()
    {
        VulkanWeightImportPolicy.NoteStaged(0x1000, 4096);
        Assert.True(VulkanWeightImportPolicy.MayReleaseWholeHostMapping);

        VulkanWeightImportPolicy.NoteImported(1 << 20);

        Assert.False(VulkanWeightImportPolicy.MayReleaseWholeHostMapping);
        Assert.Equal(1, VulkanWeightImportPolicy.ImportedTensorCount);
        Assert.Equal(1L << 20, VulkanWeightImportPolicy.ImportedBytes);

        // The staged tensor's source pages are still dead and still individually
        // releasable — the two mechanisms compose per tensor, not per file.
        Assert.Single(VulkanWeightImportPolicy.StagedSourceRanges);
        Assert.Equal(((nint)0x1000, 4096L), VulkanWeightImportPolicy.StagedSourceRanges[0]);
    }

    /// <summary>
    /// The ledger is per load. Without this, the second model loaded in a process would
    /// inherit the first's imports and #438 would refuse a release that is in fact safe.
    /// </summary>
    [Fact]
    public void Reset_ClearsTheLedgerBetweenLoads()
    {
        VulkanWeightImportPolicy.NoteImported(123);
        VulkanWeightImportPolicy.NoteStaged(0x3000, 456);
        Assert.False(VulkanWeightImportPolicy.MayReleaseWholeHostMapping);

        VulkanWeightImportPolicy.Reset();

        Assert.True(VulkanWeightImportPolicy.MayReleaseWholeHostMapping);
        Assert.Equal(0, VulkanWeightImportPolicy.ImportedTensorCount);
        Assert.Equal(0, VulkanWeightImportPolicy.StagedTensorCount);
        Assert.Empty(VulkanWeightImportPolicy.StagedSourceRanges);
        Assert.Equal(string.Empty, VulkanWeightImportPolicy.LastFallbackReason);
    }

    /// <summary>
    /// A staged tensor with no usable source pointer (a managed <c>float[]</c> norm
    /// vector) is counted but contributes no release candidate — offering pages we do
    /// not own would be a use-after-free waiting to happen.
    /// </summary>
    [Fact]
    public void StagedWithoutASourcePointer_IsNotAReleaseCandidate()
    {
        VulkanWeightImportPolicy.NoteStaged(0, 2048, "not_source_bytes");

        Assert.Equal(1, VulkanWeightImportPolicy.StagedTensorCount);
        Assert.Empty(VulkanWeightImportPolicy.StagedSourceRanges);
        Assert.Equal("not_source_bytes", VulkanWeightImportPolicy.LastFallbackReason);
    }

    /// <summary>The env escape hatch must be read per call — tests and the microbench toggle it.</summary>
    [Fact]
    public void DisableEnvVar_IsReadPerCall()
    {
        const string env = "DOTLLM_VULKAN_DISABLE_HOST_IMPORT";
        string? original = Environment.GetEnvironmentVariable(env);
        try
        {
            Environment.SetEnvironmentVariable(env, null);
            Assert.False(VulkanWeightImportPolicy.IsDisabled);
            Environment.SetEnvironmentVariable(env, "1");
            Assert.True(VulkanWeightImportPolicy.IsDisabled);
            Environment.SetEnvironmentVariable(env, "0");
            Assert.False(VulkanWeightImportPolicy.IsDisabled);
        }
        finally
        {
            Environment.SetEnvironmentVariable(env, original);
        }
    }
}
