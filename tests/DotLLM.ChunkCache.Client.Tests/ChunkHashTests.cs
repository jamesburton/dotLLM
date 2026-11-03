using System.Text;
using System.Text.Json.Nodes;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="ChunkHash"/> against the contract fixtures (spec 5.2, 5.3 item 1).</summary>
public sealed class ChunkHashTests
{
    /// <summary>
    /// Every annotation in the fixtures whose span the contract accepts carries <c>ChunkHash.Of(span text)</c>. The span
    /// text is cut by the wire rule (UTF-8 byte offsets into the decoded part), and <c>nonascii-crlf.json</c> stores its
    /// text as <c>é</c> / surrogate-pair escapes, so hashing the wire bytes or UTF-16 would not match.
    /// </summary>
    [Theory]
    [InlineData("inline.json")]
    [InlineData("manifest.json")]
    [InlineData("nonascii-crlf.json")]
    [InlineData("hash-mismatch.json")]
    public void Hash_MatchesFixtures(string fixture)
    {
        var annotations = Fixtures.Annotations(Fixtures.Json(fixture));
        JsonArray accepted = Fixtures.Json("expected.json")["requests"]![fixture]!["accepted_ids"]!.AsArray();

        string[] matching = annotations.Where(a => string.Equals(ChunkHash.Of(a.SpanText), a.Hash, StringComparison.Ordinal)).Select(a => a.Id).ToArray();

        Assert.NotEmpty(matching);
        Assert.Equal(accepted.Select(n => (string)n!), matching);
    }

    /// <summary>
    /// The hash-mismatch fixture's defective span is a CRLF part annotated with the hash of its LF-normalised text:
    /// a hash that normalised line endings would accept it. The contract says it must not.
    /// </summary>
    [Fact]
    public void Hash_CrlfPart_DoesNotMatchItsLfHash()
    {
        var a = Fixtures.Annotations(Fixtures.Json("hash-mismatch.json"))[1];
        Assert.Contains("\r\n", a.SpanText, StringComparison.Ordinal);
        Assert.NotEqual(a.Hash, ChunkHash.Of(a.SpanText));
        Assert.Equal(a.Hash, ChunkHash.Of(a.SpanText.Replace("\r\n", "\n", StringComparison.Ordinal)));
    }

    /// <summary><c>expected.json</c>'s <c>lf_variant</c>: a CRLF chunk misses an LF-indexed entry.</summary>
    [Fact]
    public void Hash_LfVariant_DiffersFromCrlf()
    {
        var a = Fixtures.Annotations(Fixtures.Json("nonascii-crlf.json"))[1];
        string lfHash = (string)Fixtures.Json("expected.json")["requests"]!["nonascii-crlf.json"]!["lf_variant"]!["lf_hash"]!;

        Assert.Equal(lfHash, ChunkHash.Of(a.SpanText.Replace("\r\n", "\n", StringComparison.Ordinal)));
        Assert.NotEqual(lfHash, ChunkHash.Of(a.SpanText));
    }

    /// <summary>The fixture really carries escapes, so the test above is about decoded text, not wire bytes.</summary>
    [Fact]
    public void NonAsciiFixture_StoresEscapes()
    {
        string raw = Encoding.UTF8.GetString(Fixtures.Bytes("nonascii-crlf.json"));
        Assert.Contains("\\u00e9", raw, StringComparison.Ordinal);
        Assert.Contains("\\ud83d\\ude00", raw, StringComparison.Ordinal);
    }

    /// <summary>Known-answer vectors (FIPS 180-2 for "abc"): prefix, lowercase hex, 64 digits.</summary>
    [Theory]
    [InlineData("", "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")]
    [InlineData("abc", "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")]
    public void Hash_KnownAnswers(string text, string expected) => Assert.Equal(expected, ChunkHash.Of(text));

    /// <summary>"é" hashes as its UTF-8 bytes C3 A9, not as UTF-16 (E9 00).</summary>
    [Fact]
    public void Hash_IsOverUtf8()
    {
        static string Hex(byte[] b) => "sha256:" + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(b)).ToLowerInvariant();
        Assert.Equal(Hex([0xC3, 0xA9]), ChunkHash.Of("é"));
        Assert.NotEqual(Hex([0xE9, 0x00]), ChunkHash.Of("é"));
    }

    /// <summary>No normalisation: a BOM and trailing whitespace change the hash.</summary>
    [Fact]
    public void Hash_DoesNotNormalise()
    {
        Assert.NotEqual(ChunkHash.Of("x"), ChunkHash.Of("﻿x"));
        Assert.NotEqual(ChunkHash.Of("x"), ChunkHash.Of("x "));
    }
}
