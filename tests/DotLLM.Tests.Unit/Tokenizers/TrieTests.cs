using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers;

public sealed class TrieTests
{
    [Fact]
    public void TryMatchLongest_ReturnsLongestPrefix()
    {
        var trie = new Trie();
        trie.Add("a", tokenId: 1, score: -2f);
        trie.Add("ab", tokenId: 2, score: -1f);
        trie.Add("abc", tokenId: 3, score: -0.5f);

        bool matched = trie.TryMatchLongest("abcd", out int tokenId, out float score, out int length);

        Assert.True(matched);
        Assert.Equal(3, tokenId);
        Assert.Equal(-0.5f, score);
        Assert.Equal(3, length);
    }

    [Fact]
    public void TryMatchLongest_ReturnsFalseWhenNoPrefixMatches()
    {
        var trie = new Trie();
        trie.Add("abc", tokenId: 7, score: 1f);

        bool matched = trie.TryMatchLongest("zzz", out int tokenId, out float score, out int length);

        Assert.False(matched);
        Assert.Equal(-1, tokenId);
        Assert.Equal(0f, score);
        Assert.Equal(0, length);
    }

    [Fact]
    public void TryMatchLongest_PreservesTerminalPrefixWhenPathContinues()
    {
        var trie = new Trie();
        trie.Add("tok", tokenId: 10, score: 0.1f);
        trie.Add("token", tokenId: 11, score: 0.2f);
        trie.Add("tokens", tokenId: 12, score: 0.3f);

        bool matched = trie.TryMatchLongest("tokenize", out int tokenId, out float score, out int length);

        Assert.True(matched);
        Assert.Equal(11, tokenId);
        Assert.Equal(0.2f, score);
        Assert.Equal(5, length);
    }

    [Fact]
    public void Add_CanOverwriteExistingToken()
    {
        var trie = new Trie();
        trie.Add("dup", tokenId: 5, score: -3f);
        trie.Add("dup", tokenId: 6, score: -1f);

        bool matched = trie.TryMatchLongest("dup", out int tokenId, out float score, out int length);

        Assert.True(matched);
        Assert.Equal(6, tokenId);
        Assert.Equal(-1f, score);
        Assert.Equal(3, length);
    }
}
