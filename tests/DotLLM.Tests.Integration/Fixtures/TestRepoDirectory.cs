namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Locates a Hugging Face repo directory (<c>owner/repo</c>) for suites that download small
/// fixtures on demand: an existing copy in any store known to <see cref="TestFixtureResolver"/>,
/// otherwise the directory the download should populate under the shared model store.
/// </summary>
internal static class TestRepoDirectory
{
    /// <summary>
    /// Returns the directory already holding <paramref name="requiredEntries"/> for
    /// <paramref name="repoId"/>, or the download target under
    /// <see cref="TestModelDownloader.CacheDirectory"/> when none does.
    /// </summary>
    /// <param name="repoId">Hugging Face repo id (<c>owner/repo</c>).</param>
    /// <param name="requiredEntries">Entry names that must all exist in an existing copy.</param>
    /// <returns>Absolute directory path (may not exist yet).</returns>
    public static string Resolve(string repoId, params string[] requiredEntries)
    {
        string[] parts = repoId.Split('/');
        return TestFixtureResolver.ResolveDirectory([], parts[0], parts[1], requiredEntries, extraDirectories: null).Path
            ?? Path.Combine(TestModelDownloader.CacheDirectory, parts[0], parts[1]);
    }
}
