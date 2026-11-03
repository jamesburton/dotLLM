using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Content-addressed chunk key (<see cref="ChunkKey"/>, scheme <c>chunk-key/v4</c>) and store namespaces
/// (<see cref="ChunkNamespace"/>). Pure: no store, no model. The store-side checks (a changed key is a miss) are in
/// <c>ChunkStoreTests</c>.
/// </summary>
public sealed class ChunkKeyTests
{
    internal const string Fingerprint = "fp-model-a";
    internal const string Tokenizer = "tok-a";
    internal const string Backend = "cpu";
    internal const KvDtype Dtype = KvDtype.F32;
    internal static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    internal static readonly int[] Tokens = [3, 17, 5, 29, 11, 40, 8];

    [Theory]
    [InlineData("model")]
    [InlineData("tokenizer")]
    [InlineData("rope-dim")]
    [InlineData("rope-theta")]
    [InlineData("rope-type")]
    [InlineData("kv-output-version")]
    [InlineData("backend")]
    [InlineData("dtype")]
    [InlineData("tokens")]
    [InlineData("positioning")]
    public void Key_DiffersWhenModelTokenizerRopeOrEngineVersionChanges(string field)
        => Assert.NotEqual(BaseKey(), Variant(field));

    /// <summary>The default key with one input changed (shared with ChunkStoreTests' store-miss check).</summary>
    internal static ChunkKey Variant(string field) => field switch
    {
        "model" => ChunkKey.Create("fp-model-b", Tokenizer, Rope, Backend, Dtype, Tokens),
        "tokenizer" => ChunkKey.Create(Fingerprint, "tok-b", Rope, Backend, Dtype, Tokens),
        "rope-dim" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { RopeDim = 64 }, Backend, Dtype, Tokens),
        "rope-theta" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Theta = 1_000_000f }, Backend, Dtype, Tokens),
        "rope-type" => ChunkKey.Create(Fingerprint, Tokenizer, Rope with { Type = RoPEType.Norm }, Backend, Dtype, Tokens),
        "kv-output-version" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, ChunkReuseSchema.KvOutputVersion + 1),
        "backend" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, "vulkan", Dtype, Tokens),
        "dtype" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, KvDtype.F16, Tokens),
        "tokens" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [3, 17, 5, 29, 11, 40, 9]),
        "positioning" => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, positioning: ChunkPositioning.Contextual),
        _ => throw new ArgumentOutOfRangeException(nameof(field)),
    };

    /// <summary>The key of the default inputs.</summary>
    internal static ChunkKey BaseKey() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);

    [Fact]
    public void Key_FieldBoundariesCannotBeForged()
    {
        // Length prefixes: moving characters between adjacent string fields must change the key.
        ChunkKey a = ChunkKey.Create("ab", "c", Rope, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create("a", "bc", Rope, Backend, Dtype, Tokens);
        Assert.NotEqual(a, b);
        // The token count is encoded: a trailing zero token is not the same chunk.
        Assert.NotEqual(ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1, 0]),
                        ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, [1]));
    }

    // chunk-key/v3 (before the namespace field) of the default inputs; v3 stores are rebuilt, never read.
    private const string GoldenV3Key = "b0f13424a15549bb98aeff929c34318dbb47586ed3056d1dc177a0e6ef1c80de";

    [Fact]
    public void Key_DefaultNamespace_GoldenV4()
    {
        // Pins the canonical encoding (scheme string, field order, widths, endianness). An accidental change would orphan every
        // persistent store (misses, not wrong hits); a deliberate one bumps the scheme string and updates this value.
        // chunk-key/v4 hashes the namespace (length-prefixed, right after the scheme) even when it is the default "", so the
        // default namespace never collapses to the v3 encoding.
        ChunkKey key = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        // Both values cross-checked when pinned by an independent re-implementation of the encoding (Python hashlib/struct,
        // 2026-11-02), which also reproduces GoldenV3Key under the v3 encoding.
        Assert.Equal("8be78abfcdcf2194b9cf662cdc6e1384d1d0808c5f492b720d65ce487935cd69", key.Hex);
        Assert.Equal(key, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.Default));
        Assert.Equal(key, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ""));
        Assert.NotEqual(GoldenV3Key, key.Hex);
        Assert.Equal("2c4b35eac74ee8c32a65c5974f92fba80a4880b8024b21a935b8b79621be3a41",
            ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-a").Hex);
    }

    [Fact]
    public void Key_DiffersByNamespace()
    {
        ChunkKey def = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-a");
        ChunkKey b = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "tenant-b");
        ChunkKey k1 = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.ForApiKey("key-1"));
        ChunkKey k2 = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.ForApiKey("key-2"));
        Assert.Equal(5, new[] { def, a, b, k1, k2 }.Distinct().Count());
        // Same namespace, same key.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: new string("tenant-a".AsSpan())));

        // Moving characters between the namespace and the fingerprint (the next field) changes the key. (This pair would differ
        // even without the namespace's length prefix, because the fingerprint carries its own; a real prefix-free forgery would
        // need the fingerprint's 4 length bytes to be valid namespace characters, i.e. a fingerprint of >= 0x2d2d2d2d bytes. The
        // prefix is pinned by the golden values instead.)
        Assert.NotEqual(ChunkKey.Create("b", Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: "a"),
                        ChunkKey.Create("ab", Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ""));

        Assert.Throws<ArgumentNullException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: null!));
    }

    [Fact]
    public void Namespace_LoneSurrogates_NeverShareAKey()
    {
        // The hazard: the default UTF-8 encoder replaces every lone surrogate with U+FFFD, so two different namespace strings
        // would hash to the same bytes and share every key.
        Assert.Equal(System.Text.Encoding.UTF8.GetBytes("\uD800"), System.Text.Encoding.UTF8.GetBytes("\uDC00"));
        foreach (string ns in new[] { "\uD800", "\uDC00", "a\uD800", "a\uDC00" })
        {
            Assert.False(ChunkNamespace.IsValid(ns));
            Assert.Throws<ArgumentException>(() => ChunkNamespace.Validate(ns));
            Assert.ThrowsAny<ArgumentException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns));
        }
    }

    [Theory]
    [InlineData("")]
    [InlineData("a")]
    [InlineData("tenant-a")]
    [InlineData("a.b_c-9")]
    [InlineData("k-0123456789abcdef")]
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")] // 64
    public void Namespace_Valid_IsAccepted(string ns)
    {
        Assert.True(ChunkNamespace.IsValid(ns));
        ChunkNamespace.Validate(ns);
        Assert.Equal(64, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns).Hex.Length);
    }

    [Theory]
    [InlineData("Tenant")]
    [InlineData("K-0123456789ABCDEF")]
    [InlineData("a/b")]
    [InlineData("a\\b")]
    [InlineData("a b")]
    [InlineData(" ")]
    [InlineData("tenant\n")]
    [InlineData("a\0")]
    [InlineData("é")]
    // Lone surrogates are in Namespace_LoneSurrogates_NeverShareAKey: xUnit serialises them identically, so as InlineData they
    // would collapse into one theory case.
    [InlineData("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")] // 65
    [InlineData("default")] // reserved (Phase I Task 4b): the index file name tag of the empty Default namespace
    public void Namespace_Invalid_IsRefusedEverywhere(string ns)
    {
        // Rule: ^(|k-[0-9a-f]{16}|[a-z0-9._-]{1,64})$ (empty = Default). Enforced by ChunkNamespace and by ChunkKey.Create, so a
        // namespace that reaches a key can also go into a file name, a JSON ledger or a log line unescaped.
        Assert.False(ChunkNamespace.IsValid(ns));
        Assert.Throws<ArgumentException>(() => ChunkNamespace.Validate(ns));
        Assert.ThrowsAny<ArgumentException>(() => ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ns));
    }

    [Fact]
    public void Namespace_Null_IsNotValid()
    {
        Assert.False(ChunkNamespace.IsValid(null));
        Assert.Throws<ArgumentNullException>(() => ChunkNamespace.Validate(null!));
        Assert.True(ChunkNamespace.IsValid(ChunkNamespace.Default));
        Assert.True(ChunkNamespace.IsValid(ChunkNamespace.ForApiKey("sk-test-123")));
    }

    [Fact]
    public void ForApiKey_IsStable_AndNeverEqualsDefault()
    {
        // "k-" + the first 16 lowercase hex of SHA-256(UTF-8 key); golden cross-checked with Python hashlib (2026-11-02).
        Assert.Equal("k-e0dbaa0c6455768b", ChunkNamespace.ForApiKey("sk-test-123"));
        Assert.Equal(ChunkNamespace.ForApiKey("sk-test-123"), ChunkNamespace.ForApiKey(new string("sk-test-123".AsSpan())));
        Assert.Equal("k-2bb80d537b1da3e3", ChunkNamespace.ForApiKey("secret"));
        Assert.Equal("", ChunkNamespace.Default);
        foreach (string apiKey in new[] { "sk-test-123", "secret", " ", "é" })
        {
            string ns = ChunkNamespace.ForApiKey(apiKey);
            Assert.NotEqual(ChunkNamespace.Default, ns);
            Assert.Matches("^k-[0-9a-f]{16}$", ns); // a digest, never the key itself
        }
        Assert.NotEqual(ChunkNamespace.ForApiKey("sk-test-123"), ChunkNamespace.ForApiKey("sk-test-124"));
        Assert.Throws<ArgumentNullException>(() => ChunkNamespace.ForApiKey(null!));
        Assert.Throws<ArgumentException>(() => ChunkNamespace.ForApiKey("")); // no key: the caller uses Default
    }

    [Fact]
    public void Key_SameInputs_SameKey()
    {
        ChunkKey a = ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens);
        ChunkKey b = ChunkKey.Create(new string(Fingerprint.AsSpan()), new string(Tokenizer.AsSpan()),
            new RopeParameters(128, 5_000_000f, RoPEType.NeoX), new string(Backend.AsSpan()), KvDtype.F32, (int[])Tokens.Clone());
        Assert.Equal(a, b);
        // The default schema version is the current constant.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, ChunkReuseSchema.KvOutputVersion));
        // The default positioning is Isolated.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, positioning: ChunkPositioning.Isolated));
        // The default namespace is ChunkNamespace.Default.
        Assert.Equal(a, ChunkKey.Create(Fingerprint, Tokenizer, Rope, Backend, Dtype, Tokens, @namespace: ChunkNamespace.Default));
        Assert.Equal(64, a.Hex.Length);
        Assert.Matches("^[0-9a-f]{64}$", a.Hex);
    }
}
