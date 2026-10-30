using DotLLM.HuggingFace;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Store-root resolution and existing-file adoption for <see cref="TestModelDownloader"/> (#1279).
/// Deliberately declares no <c>[Collection]</c> so it pulls in no model fixture and downloads
/// nothing — these are path and temp-file assertions.
/// </summary>
public class TestModelDownloaderTests
{
    // #1279: downloads land in the CLI's model store, not ~/.dotllm/test-cache.
    private static string DefaultCacheDir => HuggingFaceDownloader.DefaultModelsDirectory;

    [Fact]
    public void DefaultCacheDir_IsNotTheLegacyTestCache()
    {
        Assert.NotEqual(
            Path.GetFullPath(TestFixtureResolver.LegacyTestCacheDirectory),
            Path.GetFullPath(TestModelDownloader.ResolveCacheDirectory(null)));
    }

    [Fact]
    public void TryUseExisting_FileInAnotherStore_HardlinksIntoTargetWithoutDownloading()
    {
        using var temp = new TempDir();
        string store = Path.Combine(temp.Path, "models");
        string elsewhere = Path.Combine(temp.Path, "elsewhere");
        Directory.CreateDirectory(elsewhere);
        string source = Path.Combine(elsewhere, "probe-1279.gguf");
        File.WriteAllBytes(source, [1, 2, 3, 4]);

        string? path = TestModelDownloader.TryUseExisting(
            "dotllm-test-1279/no-such-repo", "probe-1279.gguf", store, [elsewhere]);

        string expected = Path.Combine(store, "dotllm-test-1279", "no-such-repo", "probe-1279.gguf");
        Assert.Equal(expected, path);

        // Same volume (both under the temp dir), so this is a hardlink: a write through one name
        // is visible through the other, which a copy would not show.
        File.WriteAllBytes(source, [9, 9, 9, 9]);
        Assert.Equal(new byte[] { 9, 9, 9, 9 }, File.ReadAllBytes(expected));
    }

    [Fact]
    public void TryUseExisting_FileAlreadyInTarget_ReturnsItUnchanged()
    {
        using var temp = new TempDir();
        string expected = Path.Combine(temp.Path, "o", "r", "f.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(expected)!);
        File.WriteAllBytes(expected, [1]);

        Assert.Equal(expected, TestModelDownloader.TryUseExisting("o/r", "f.gguf", temp.Path, extraDirectories: null));
    }

    [Fact]
    public void TryUseExisting_NoStoreHoldsTheFile_ReturnsNullSoTheCallerDownloads()
    {
        using var temp = new TempDir();

        Assert.Null(TestModelDownloader.TryUseExisting(
            "dotllm-test-1279/no-such-repo", "missing-1279.gguf", temp.Path, [temp.Path]));
    }

    [Fact]
    public void AdoptExisting_WhenLinkImpossible_UsesSourceInPlace()
    {
        using var temp = new TempDir();
        string source = Path.Combine(temp.Path, "src.gguf");
        File.WriteAllBytes(source, [1]);

        // An occupied target makes the hardlink fail, standing in for a cross-volume link: the
        // source must come back as-is, with nothing copied over the target.
        string target = Path.Combine(temp.Path, "dst.gguf");
        File.WriteAllBytes(target, [2]);

        Assert.Equal(source, TestModelDownloader.AdoptExisting(source, target));
        Assert.Equal(new byte[] { 2 }, File.ReadAllBytes(target));
    }

    [Fact]
    public void ResolveFile_GlobPattern_MatchesInDotLlmStoreDirectory()
    {
        using var temp = new TempDir();
        string file = Path.Combine(temp.Path, "SmolLM-135M.Q3_K_M.gguf");
        File.WriteAllBytes(file, [1]);

        FixtureLocation loc = TestFixtureResolver.ResolveFile(
            [], "dotllm-test-1279", "no-such-repo", ["*Q3_K_M*.gguf"], [temp.Path]);

        Assert.Equal(file, loc.Path);
    }

    [Fact]
    public void ResolveCacheDirectory_WhenUnset_UsesDefault()
    {
        Assert.Equal(DefaultCacheDir, TestModelDownloader.ResolveCacheDirectory(null));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\t")]
    public void ResolveCacheDirectory_WhenBlank_UsesDefault(string envValue)
    {
        Assert.Equal(DefaultCacheDir, TestModelDownloader.ResolveCacheDirectory(envValue));
    }

    [Fact]
    public void ResolveCacheDirectory_WhenSet_UsesOverride()
    {
        string target = Path.Combine(Path.GetTempPath(), "dotllm-test-cache-override");

        Assert.Equal(Path.GetFullPath(target), TestModelDownloader.ResolveCacheDirectory(target));
    }

    [Fact]
    public void ResolveCacheDirectory_WhenRelative_IsMadeAbsolute()
    {
        string resolved = TestModelDownloader.ResolveCacheDirectory("some-relative-cache");

        Assert.True(Path.IsPathFullyQualified(resolved));
        Assert.Equal(Path.GetFullPath("some-relative-cache"), resolved);
    }

    [Fact]
    public void ResolveCacheDirectory_Override_DoesNotAppendTestCacheSegment()
    {
        // Pointing the override at ~/.dotllm/models must use that directory verbatim,
        // so the layout stays interchangeable with the CLI's model directory.
        string models = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "models");

        Assert.Equal(models, TestModelDownloader.ResolveCacheDirectory(models));
    }

    [Fact]
    public void CacheDirectory_ReflectsEnvironmentVariable()
    {
        // Reads the ambient value rather than mutating it, so this stays safe under
        // xUnit's parallel collection execution.
        string expected = TestModelDownloader.ResolveCacheDirectory(
            Environment.GetEnvironmentVariable(TestModelDownloader.CacheDirEnvVar));

        Assert.Equal(expected, TestModelDownloader.CacheDirectory);
    }

    /// <summary>Self-deleting temp directory.</summary>
    private sealed class TempDir : IDisposable
    {
        public TempDir()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "dotllm-1279-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try
            {
                Directory.Delete(Path, recursive: true);
            }
            catch (IOException)
            {
                // best effort: a leftover temp dir is harmless
            }
        }
    }
}
