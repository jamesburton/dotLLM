using DotLLM.Engine;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1919 T4: a session refuses every operation while the live provenance differs from the creation tag in any field.</summary>
public sealed class GenerationSessionProvenanceEnforcementTests
{
    private static readonly string[] Fields =
    [
        nameof(CacheProvenanceTag.ModelDigest),
        nameof(CacheProvenanceTag.TokenizerRevision),
        nameof(CacheProvenanceTag.Precision),
        nameof(CacheProvenanceTag.ActiveModifiersDigest),
        nameof(CacheProvenanceTag.KnowledgeSnapshotId),
    ];

    private static CacheProvenanceTag Mutate(CacheProvenanceTag t, string field) => field switch
    {
        nameof(CacheProvenanceTag.ModelDigest) => t with { ModelDigest = "model-b" },
        nameof(CacheProvenanceTag.TokenizerRevision) => t with { TokenizerRevision = "tok-2" },
        nameof(CacheProvenanceTag.Precision) => t with { Precision = "q8_0" },
        nameof(CacheProvenanceTag.ActiveModifiersDigest) => t with { ActiveModifiersDigest = "mods-1" },
        nameof(CacheProvenanceTag.KnowledgeSnapshotId) => t with { KnowledgeSnapshotId = "snap-2" },
        _ => throw new ArgumentOutOfRangeException(nameof(field)),
    };

    public static TheoryData<string> FieldNames { get { var d = new TheoryData<string>(); foreach (string f in Fields) d.Add(f); return d; } }

    private sealed class Live(CacheProvenanceTag tag) { public CacheProvenanceTag Tag = tag; }

    private static (GenerationSessionFixture Fx, GenerationSession S, Live Live) Open(string family = Llama)
    {
        var fx = Create(family);
        var live = new Live(Tag());
        var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), () => live.Tag);
        return (fx, s, live);
    }

    private static void AssertNamesOnly(string field, InvalidOperationException ex)
    {
        Assert.Contains(field, ex.Message);
        foreach (string other in Fields.Where(f => f != field))
            Assert.DoesNotContain(other, ex.Message);
    }

    /// <summary>One field at a time, every operation: each refuses, naming exactly the differing field, and changes nothing.</summary>
    [Theory]
    [MemberData(nameof(FieldNames))]
    public async Task AnyDifferingField_RefusesEveryOperation(string field)
    {
        var (fx, s, live) = Open();
        using (fx)
        await using (s)
        {
            await s.PrefillAsync(fx.Prompt);
            var before = new float[s.VocabSize];
            s.CopyNextLogits(before);
            int pos = s.CurrentPosition;

            live.Tag = Mutate(Tag(), field);

            AssertNamesOnly(field, await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.PrefillAsync(fx.Observation)));
            AssertNamesOnly(field, await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.AppendObservationTokensAsync(fx.Observation)));
            AssertNamesOnly(field, await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(2))));
            AssertNamesOnly(field, Assert.Throws<InvalidOperationException>(() => s.NextTokenArgMax()));
            AssertNamesOnly(field, Assert.Throws<InvalidOperationException>(() => s.CopyNextLogits(new float[s.VocabSize])));
            AssertNamesOnly(field, Assert.Throws<InvalidOperationException>(() => s.RollbackTo(0)));
            AssertNamesOnly(field, Assert.Throws<InvalidOperationException>(() => s.CanRollbackTo(0)));
            AssertNamesOnly(field, Assert.Throws<InvalidOperationException>(() => s.ValidateProvenance(Mutate(Tag(), field))));

            Assert.Equal(pos, s.CurrentPosition);   // refusals touched nothing
            Assert.Equal(pos, s.KvLength);

            // Restoring the live view un-blocks the very same session: state was never dropped or stale-reused.
            live.Tag = Tag();
            var after = new float[s.VocabSize];
            s.CopyNextLogits(after);
            Assert.Equal(before, after);
            await s.AppendObservationTokensAsync(fx.Observation);
        }
    }

    [Fact]
    public async Task EqualLiveTag_AndNoProvider_AreAccepted()
    {
        var (fx, s, _) = Open();
        using (fx)
        await using (s)
        {
            await s.PrefillAsync(fx.Prompt);
            s.ValidateProvenance(Tag());
            s.ValidateProvenance(Tag() with { });
        }
        using var fx2 = Create(Llama);
        await using var plain = fx2.NewGenerator().CreateSession(Greedy(), Tag());   // no provider: nothing to compare against
        await plain.PrefillAsync(fx2.Prompt);
    }

    [Fact]
    public async Task ValidateProvenance_ListsAllDifferingFields_AndRejectsNull()
    {
        var (fx, s, _) = Open();
        using (fx)
        await using (s)
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                s.ValidateProvenance(Tag() with { ModelDigest = "x", KnowledgeSnapshotId = "y" }));
            Assert.Contains(nameof(CacheProvenanceTag.ModelDigest), ex.Message);
            Assert.Contains(nameof(CacheProvenanceTag.KnowledgeSnapshotId), ex.Message);
            Assert.DoesNotContain(nameof(CacheProvenanceTag.Precision), ex.Message);
            Assert.Throws<ArgumentNullException>(() => s.ValidateProvenance(null!));
        }
    }

    [Fact]
    public async Task ProviderReturningNull_IsRejected()
    {
        using var fx = Create(Llama);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), () => null!);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.PrefillAsync(fx.Prompt));
    }

    /// <summary>The provider is consulted per operation, so a change between two operations is caught on the second.</summary>
    [Fact]
    public async Task ChangeBetweenOperations_IsCaughtOnTheNextOne()
    {
        var (fx, s, live) = Open();
        using (fx)
        await using (s)
        {
            await s.PrefillAsync(fx.Prompt);
            await s.GenerateUntilBoundaryAsync(new StopAfter(2));
            live.Tag = Tag() with { ActiveModifiersDigest = ActiveModifiersDigest.Compute([("lora", "adapter-a", "d1")]) };
            await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(2)));
        }
    }
}
