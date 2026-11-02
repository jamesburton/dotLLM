using System.Security.Cryptography;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// The copied wire-contract fixtures are byte-identical to the contract set (ChunkedkvCache
/// <c>docs/contracts/chunk-cache-v1/</c>, fixtures at 14e38cc, rulings at 3deddf4).
/// </summary>
public sealed class ContractFixtureTests
{
    /// <summary>
    /// SHA-256 of <c>fixtures.sha256</c> itself, pinned so a fixture edited together with its list entry still fails
    /// here. Changing it means the contract changed: re-copy the whole set from the contract folder.
    /// </summary>
    private const string ListSha256 = "6c1cef3241f7aad0194ce6110c0d935e31b0ef467244326efebe4e1a7b8e0c0a";

    [Fact]
    public void FixtureList_IsTheContractList()
    {
        Assert.Equal(ListSha256, Sha256Hex(Fixtures.Bytes("fixtures.sha256")));
    }

    [Fact]
    public void FixtureCopy_MatchesContractSha256()
    {
        string[] lines = File.ReadAllLines(Fixtures.PathOf("fixtures.sha256"));
        var listed = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string line in lines.Where(l => l.Length > 0))
        {
            string[] f = line.Split("  ", 2);
            listed.Add(f[1], f[0]);
        }

        Assert.Equal(12, listed.Count);
        foreach ((string name, string hex) in listed)
            Assert.True(string.Equals(hex, Sha256Hex(Fixtures.Bytes(name)), StringComparison.Ordinal), $"{name}: bytes differ from the contract");

        string[] onDisk = Directory.GetFiles(Fixtures.Dir, "*.json").Select(Path.GetFileName).OfType<string>()
            .Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(listed.Keys.Order(StringComparer.Ordinal), onDisk);
    }

    private static string Sha256Hex(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}
