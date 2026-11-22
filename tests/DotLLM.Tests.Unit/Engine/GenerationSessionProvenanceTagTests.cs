using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1919 T1: <see cref="CacheProvenanceTag"/> equality / diff and the order-stable modifier digest.</summary>
public sealed class GenerationSessionProvenanceTagTests
{
    private static CacheProvenanceTag Tag() => new("model-a", "tok-1", "f16", "mods-0", "snap-1");

    public static TheoryData<string, Func<CacheProvenanceTag, CacheProvenanceTag>> FieldMutations => new()
    {
        { nameof(CacheProvenanceTag.ModelDigest), t => t with { ModelDigest = "model-b" } },
        { nameof(CacheProvenanceTag.TokenizerRevision), t => t with { TokenizerRevision = "tok-2" } },
        { nameof(CacheProvenanceTag.Precision), t => t with { Precision = "q8_0" } },
        { nameof(CacheProvenanceTag.ActiveModifiersDigest), t => t with { ActiveModifiersDigest = "mods-1" } },
        { nameof(CacheProvenanceTag.KnowledgeSnapshotId), t => t with { KnowledgeSnapshotId = "snap-2" } },
    };

    [Fact]
    public void EqualTags_HaveNoDiff()
    {
        Assert.Equal(Tag(), Tag());
        Assert.Empty(Tag().Diff(Tag()));
    }

    [Theory]
    [MemberData(nameof(FieldMutations))]
    public void Diff_NamesExactlyTheDifferingField(string field, Func<CacheProvenanceTag, CacheProvenanceTag> mutate)
    {
        var changed = mutate(Tag());
        Assert.NotEqual(Tag(), changed);
        Assert.Equal([field], Tag().Diff(changed));
        Assert.Equal([field], changed.Diff(Tag()));
    }

    [Fact]
    public void Diff_ListsEveryDifferingField()
    {
        var other = new CacheProvenanceTag("x", "tok-1", "y", "mods-0", "z");
        Assert.Equal(
            [nameof(CacheProvenanceTag.ModelDigest), nameof(CacheProvenanceTag.Precision), nameof(CacheProvenanceTag.KnowledgeSnapshotId)],
            Tag().Diff(other));
    }

    [Fact]
    public void Constructor_RejectsNullFields()
    {
        Assert.Throws<ArgumentNullException>(() => new CacheProvenanceTag(null!, "a", "b", "c", "d"));
        Assert.Throws<ArgumentNullException>(() => new CacheProvenanceTag("a", "b", "c", "d", null!));
    }

    [Fact]
    public void Digest_IsOrderStable()
    {
        (string, string, string)[] a = [("lora", "x", "d1"), ("overlay", "y", "d2"), ("adapter", "z", "d3")];
        (string, string, string)[] b = [a[2], a[0], a[1]];
        Assert.Equal(ActiveModifiersDigest.Compute(a), ActiveModifiersDigest.Compute(b));
    }

    [Fact]
    public void Digest_IsSensitiveToEveryComponentAndMembership()
    {
        var baseline = ActiveModifiersDigest.Compute([("lora", "x", "d1")]);
        Assert.NotEqual(baseline, ActiveModifiersDigest.Compute([("lora2", "x", "d1")]));
        Assert.NotEqual(baseline, ActiveModifiersDigest.Compute([("lora", "x2", "d1")]));
        Assert.NotEqual(baseline, ActiveModifiersDigest.Compute([("lora", "x", "d2")]));
        Assert.NotEqual(baseline, ActiveModifiersDigest.Compute([("lora", "x", "d1"), ("lora", "y", "d1")]));
        Assert.NotEqual(baseline, ActiveModifiersDigest.Compute([]));
    }

    [Fact]
    public void Digest_LengthPrefixing_PreventsFieldBoundaryCollisions()
    {
        Assert.NotEqual(
            ActiveModifiersDigest.Compute([("ab", "c", "d")]),
            ActiveModifiersDigest.Compute([("a", "bc", "d")]));
    }

    [Fact]
    public void Digest_IsLowercaseSha256Hex()
    {
        string d = ActiveModifiersDigest.Compute([("a", "b", "c")]);
        Assert.Equal(64, d.Length);
        Assert.Matches("^[0-9a-f]{64}$", d);
    }
    /// <summary>#1931 review M-5: a <c>with</c> expression cannot smuggle a null past the constructor's checks.</summary>
    [Fact]
    public void WithExpression_NullField_IsRejected()
    {
        var tag = new CacheProvenanceTag("m", "t", "p", "d", "k");
        Assert.Throws<ArgumentNullException>(() => tag with { ModelDigest = null! });
        Assert.Throws<ArgumentNullException>(() => tag with { TokenizerRevision = null! });
        Assert.Throws<ArgumentNullException>(() => tag with { Precision = null! });
        Assert.Throws<ArgumentNullException>(() => tag with { ActiveModifiersDigest = null! });
        Assert.Throws<ArgumentNullException>(() => tag with { KnowledgeSnapshotId = null! });
        Assert.Equal(tag, tag with { ModelDigest = "m" });
        Assert.Equal(new[] { "ModelDigest" }, tag.Diff(tag with { ModelDigest = "x" }));
    }
}
