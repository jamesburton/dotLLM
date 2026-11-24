using System.Text.Json;
using DotLLM.Core.Constraints;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Core.Sampling;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;
using Xunit;
using static DotLLM.Tests.Unit.Engine.GenerationSessionFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1921: mid-session <see cref="GenerationSession.SetConstraint"/>, asserted on every backend (shared suite).</summary>
public abstract partial class GenerationSessionBackendSuite
{
    internal const string ObjSchema = """
        {"type":"object","properties":{"a":{"enum":["a"]}},"required":["a"],"additionalProperties":false}
        """;
    private const string Gbnf = "root ::= \"}a:a\"";

    private static void AssertValidObj(GenerationSessionFixture fx, GenerationSegmentResult r)
    {
        Assert.True(r.FinishReason == FinishReason.ConstraintComplete, $"{r.FinishReason}: [{fx.ConstraintTokenizer.Decode(r.TokenIds.ToArray())}] n={r.TokenIds.Count}");
        string text = fx.ConstraintTokenizer.Decode(r.TokenIds.ToArray());
        using var doc = JsonDocument.Parse(text);
        var root = doc.RootElement;
        Assert.Equal(JsonValueKind.Object, root.ValueKind);
        Assert.Equal("a", Assert.Single(root.EnumerateObject()).Name);
        Assert.Equal("a", root.GetProperty("a").GetString());
    }

    private static JsonSchemaConstraint Json(GenerationSessionFixture fx) => new(fx.ConstraintTokenizer, ObjSchema);

    /// <summary>free (null) -> pause -> schema -> valid JSON -> null -> observation -> resume; position and KV length move monotonically in step.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SetConstraint_MultiPhase_EndToEnd(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        Assert.Null(s.ActiveConstraint);
        var free = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Equal(3, free.TokenIds.Count);
        int p1 = s.CurrentPosition;

        var json = Json(fx);
        s.SetConstraint(json);
        Assert.Same(json, s.ActiveConstraint);
        Assert.Equal(p1, s.CurrentPosition);
        var seg = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        AssertValidObj(fx, seg);
        Assert.True(json.IsComplete());
        int p2 = s.CurrentPosition;
        Assert.True(p2 > p1);
        Assert.Equal(p2, s.KvLength);

        s.SetConstraint(null);
        Assert.Null(s.ActiveConstraint);
        await s.AppendObservationTokensAsync(fx.Observation);
        Assert.Equal(p2 + fx.Observation.Length, s.CurrentPosition);
        var resumed = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        Assert.Equal(3, resumed.TokenIds.Count);
        Assert.NotEqual(FinishReason.ConstraintComplete, resumed.FinishReason);
        Assert.Equal(s.CurrentPosition, s.KvLength);
    }

    /// <summary>SetConstraint touches no model state: position, KV length and logits are identical; every committed token is forwarded exactly once (CPU: counted).</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SetConstraint_PreservesPrefix_NoReprefill(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        int pos = s.CurrentPosition, kv = s.KvLength;
        var before = Logits(s);
        fx.Counter.Reset();

        s.SetConstraint(Json(fx));
        s.SetConstraint(null);
        s.SetConstraint(Json(fx));
        Assert.Equal(pos, s.CurrentPosition);
        Assert.Equal(kv, s.KvLength);
        Assert.Equal(before, Logits(s));
        if (fx.CountsForwards) { Assert.Equal(0, fx.Counter.Forwards); Assert.Equal(0, fx.Counter.Tokens); }

        var seg = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        AssertValidObj(fx, seg);
        Assert.Equal(pos + seg.TokenIds.Count, s.CurrentPosition);
        Assert.Equal(s.CurrentPosition, s.KvLength);
        // One forward per committed token and not one more: the prefix was never re-prefilled.
        if (fx.CountsForwards) { Assert.Equal(seg.TokenIds.Count, fx.Counter.Tokens); Assert.Equal(seg.TokenIds.Count, fx.Counter.Forwards); }
    }

    /// <summary>Json / Grammar / null across turns, each turn matches only its own constraint; reusing a finished instance leaks no parser state.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SetConstraint_MultiTurn_Isolation(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);

        var json = Json(fx);
        s.SetConstraint(json);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));

        s.SetConstraint(new GrammarConstraint(fx.ConstraintTokenizer, Gbnf));
        var g = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        Assert.Equal(FinishReason.ConstraintComplete, g.FinishReason);
        Assert.Equal("}a:a", fx.ConstraintTokenizer.Decode(g.TokenIds.ToArray()));

        s.SetConstraint(null);
        var free = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        Assert.Equal(4, free.TokenIds.Count);
        Assert.NotEqual(FinishReason.ConstraintComplete, free.FinishReason);

        // The same, already-complete instance again: Reset on set means it starts a fresh document.
        Assert.True(json.IsComplete());
        s.SetConstraint(json);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));

        s.SetConstraint(new GrammarConstraint(fx.ConstraintTokenizer, Gbnf));
        var g2 = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        Assert.Equal("}a:a", fx.ConstraintTokenizer.Decode(g2.TokenIds.ToArray()));

        // Re-setting a half-used constraint restarts it cleanly too.
        s.SetConstraint(Json(fx));
        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        var half = s.ActiveConstraint!;
        s.SetConstraint(half);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
    }

    /// <summary>A rollback that discards tokens resets the active constraint: the replayed segment is a whole document again.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_ResetsTheActiveConstraint(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int boundary = s.CurrentPosition;
        var json = Json(fx);
        s.SetConstraint(json);
        var part = await s.GenerateUntilBoundaryAsync(new StopAfter(4));   // mid-document
        Assert.False(json.IsComplete());
        s.RollbackTo(boundary);
        Assert.Same(json, s.ActiveConstraint);
        var seg = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        AssertValidObj(fx, seg);
        Assert.Equal(part.TokenIds.ToArray(), seg.TokenIds.Take(part.TokenIds.Count).ToArray());   // greedy: same prefix again, from a clean parse
    }

    /// <summary>SetConstraint is refused during an operation, after provenance drift and after disposal; the active constraint is unchanged.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task SetConstraint_IsRefusedWhileBusyDriftedOrDisposed(string family)
    {
        using var fx = Open(family);
        var live = Tag();
        var s = fx.NewGenerator().CreateSession(Greedy(), Tag(), () => live);
        await s.PrefillAsync(fx.Prompt);
        var json = Json(fx);
        s.SetConstraint(json);

        Exception? inFlight = null;
        await s.GenerateUntilBoundaryAsync(new ReentrantStop(() =>
        {
            try { s.SetConstraint(null); } catch (Exception e) { inFlight = e; }
        }));
        Assert.IsType<InvalidOperationException>(inFlight);
        Assert.Same(json, s.ActiveConstraint);

        live = live with { ActiveModifiersDigest = "mods-1" };
        Assert.Throws<InvalidOperationException>(() => s.SetConstraint(null));
        Assert.Same(json, s.ActiveConstraint);
        live = Tag();
        s.SetConstraint(null);

        await s.DisposeAsync();
        Assert.Throws<ObjectDisposedException>(() => s.SetConstraint(json));
    }

    private sealed class ReentrantStop(Action onCall) : IStopCondition
    {
        private bool _done;
        public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
        {
            if (!_done) { _done = true; onCall(); }
            return generatedTokens.Count >= 2 ? StopResult.StopInclude : StopResult.Continue;
        }
    }

    /// <summary>Observation (and prompt) tokens never advance the constraint: its mask is the initial one after an append and the next document is valid.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task ObservationTokens_DoNotAdvanceTheConstraint(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        var json = Json(fx);
        s.SetConstraint(json);
        long[] initial = json.GetAllowedTokens().AsSpan().ToArray();

        // Observation made of characters this constraint could never accept first: advancing on them would corrupt (or assert in) its parse state.
        var ct = fx.ConstraintTokenizer;
        int[] obs = [.. Enumerable.Range(0, ct.VocabSize).Where(i => ct.DecodeToken(i) is "}" or ":" or "a")];
        Assert.NotEmpty(obs);
        await s.AppendObservationTokensAsync(obs);
        Assert.Equal(initial, json.GetAllowedTokens().AsSpan().ToArray());
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
    }

    // -- #1937 review I-1: rollback inside a constrained span keeps the constraint consistent ------------------------------

    private static void AssertDocument(GenerationSessionFixture fx, IEnumerable<int> sampled)
    {
        string text = fx.ConstraintTokenizer.Decode(sampled.ToArray());
        using var doc = JsonDocument.Parse(text);
        var root = doc.RootElement;
        Assert.Equal(JsonValueKind.Object, root.ValueKind);
        Assert.Equal("a", Assert.Single(root.EnumerateObject()).Name);
        Assert.Equal("a", root.GetProperty("a").GetString());
    }

    /// <summary>Document paused in three segments (3+2+2 tokens, minimum document is 9), rolled back to a mid-document boundary, then continued: the concatenation is one valid document.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_MidDocumentBoundary_ReplaysTheConstraint(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        s.SetConstraint(Json(fx));
        var a = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        var b = await s.GenerateUntilBoundaryAsync(new StopAfter(2));
        int mid = s.CurrentPosition;
        var c = await s.GenerateUntilBoundaryAsync(new StopAfter(2));
        Assert.True(s.CanRollbackTo(mid));

        s.RollbackTo(mid);
        Assert.Equal(mid, s.CurrentPosition);
        var rest = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        Assert.Equal(FinishReason.ConstraintComplete, rest.FinishReason);
        AssertDocument(fx, a.TokenIds.Concat(b.TokenIds).Concat(rest.TokenIds));
        // Greedy and exact: the re-generated tail starts as the undisturbed run did.
        Assert.Equal(c.TokenIds.ToArray(), rest.TokenIds.Take(c.TokenIds.Count).ToArray());
    }

    /// <summary>Rollback to the current position is a no-op for the constraint (nothing discarded).</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_CurrentPosition_LeavesTheConstraintAlone(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        s.SetConstraint(Json(fx));
        var a = await s.GenerateUntilBoundaryAsync(new StopAfter(4));
        s.RollbackTo(s.CurrentPosition);
        var rest = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        AssertDocument(fx, a.TokenIds.Concat(rest.TokenIds));
    }

    /// <summary>Attention models: rollback to an arbitrary position inside the document (not a boundary).</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_ArbitraryPositionInDocument_Attention(string family)
    {
        using var fx = Open(family);
        Skip.If(fx.IsRecurrent, "recurrent models roll back to boundaries only");
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int p0 = s.CurrentPosition;
        s.SetConstraint(Json(fx));
        var a = await s.GenerateUntilBoundaryAsync(new StopAfter(7));
        s.RollbackTo(p0 + 4);   // between boundaries
        var rest = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        Assert.Equal(FinishReason.ConstraintComplete, rest.FinishReason);
        AssertDocument(fx, a.TokenIds.Take(4).Concat(rest.TokenIds));
    }

    /// <summary>Rollback to the attach point (and to 0, with a prompt before the attach point) leaves a fresh constraint: a whole new document follows.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_AttachPointOrZero_ResetsTheConstraint(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int p0 = s.CurrentPosition;
        s.SetConstraint(Json(fx));
        await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        s.RollbackTo(p0);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));

        s.SetConstraint(Json(fx));   // attaches at the end of that document
        await s.GenerateUntilBoundaryAsync(new StopAfter(2));
        s.RollbackTo(0);
        await s.PrefillAsync(fx.Prompt);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
    }

    /// <summary>An observation inside the span is not sampled: rollback past it re-advances only the sampled tokens.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_AfterObservationInSpan_DoesNotAdvanceForObservation(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        s.SetConstraint(Json(fx));
        var a = await s.GenerateUntilBoundaryAsync(new StopAfter(3));
        var ct = fx.ConstraintTokenizer;
        int[] obs = [.. Enumerable.Range(0, ct.VocabSize).Where(i => ct.DecodeToken(i) is "}" or ":" or "a")];
        await s.AppendObservationTokensAsync(obs);
        var b = await s.GenerateUntilBoundaryAsync(new StopAfter(2));
        int mid = s.CurrentPosition;
        await s.GenerateUntilBoundaryAsync(new StopAfter(2));

        s.RollbackTo(mid);
        var rest = await s.GenerateUntilBoundaryAsync(new StopAfter(60));
        Assert.Equal(FinishReason.ConstraintComplete, rest.FinishReason);
        AssertDocument(fx, a.TokenIds.Concat(b.TokenIds).Concat(rest.TokenIds));
    }

    /// <summary>A faulted session (failed forward, CPU only) recovers with RollbackTo(0), which also resets the constraint.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task RollbackTo_Zero_AfterFault_ResetsTheConstraint(string family)
    {
        using var fx = Open(family);
        Skip.IfNot(fx.CountsForwards, "fault injection hooks the CPU forward");
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        s.SetConstraint(Json(fx));
        await s.GenerateUntilBoundaryAsync(new StopAfter(2));
        fx.Counter.FailNextForward();
        try { await s.GenerateUntilBoundaryAsync(new StopAfter(5)); } catch (IOException) { }
        fx.Counter.Reset();
        s.RollbackTo(0);
        await s.PrefillAsync(fx.Prompt);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
    }

    /// <summary>A constraint that throws in Advance after its token was committed is marked broken: the next constrained sample refuses with a clear error, and rollback repairs it.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families), MemberType = typeof(GenerationSessionFixture))]
    public async Task ConstraintAdvanceFailure_RefusesNextSample_UntilRolledBack(string family)
    {
        using var fx = Open(family);
        await using var s = fx.NewGenerator().CreateSession(Greedy(), Tag());
        await s.PrefillAsync(fx.Prompt);
        int p0 = s.CurrentPosition;
        var c = new ThrowingConstraint(Json(fx)) { ThrowOnAdvanceNumber = 3 };
        s.SetConstraint(c);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
        Assert.Equal(p0 + 3, s.CurrentPosition);   // the token whose Advance threw is committed
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () => await s.GenerateUntilBoundaryAsync(new StopAfter(2)));
        Assert.Contains("stale", ex.Message);
        Assert.Equal(p0 + 3, s.CurrentPosition);   // refused before sampling anything
        c.ThrowOnAdvanceNumber = 0;
        s.RollbackTo(p0);
        AssertValidObj(fx, await s.GenerateUntilBoundaryAsync(new StopAfter(60)));
    }

    /// <summary>EOG ids other than EOS count as terminal for a complete constraint.</summary>
    [Fact]
    public void ConstraintIsTerminal_CountsEveryEndOfGenerationId()
    {
        using var fx = GenerationSessionFixture.Create(GenerationSessionFixture.Llama);
        var tok = new ExtraEogTokenizer(fx.ConstraintTokenizer, extraEog: 5);
        var gen = new TextGenerator(fx.Model, tok);
        Assert.True(gen.ConstraintIsTerminal(new FixedMaskConstraint(tok.VocabSize, complete: true, allowed: [5])));
        Assert.True(gen.ConstraintIsTerminal(new FixedMaskConstraint(tok.VocabSize, complete: true, allowed: [2, 5])));
        Assert.False(gen.ConstraintIsTerminal(new FixedMaskConstraint(tok.VocabSize, complete: true, allowed: [5, 6])));
        Assert.False(gen.ConstraintIsTerminal(new FixedMaskConstraint(tok.VocabSize, complete: false, allowed: [5])));
    }

    /// <summary>The session's ResponseFormat constraint and Generate's come from the one factory.</summary>
    [Fact]
    public async Task ResponseFormatFactory_IsSharedAndCoversEveryFormat()
    {
        using var fx = GenerationSessionFixture.Create(GenerationSessionFixture.Llama);
        var gen = new TextGenerator(fx.Model, fx.ConstraintTokenizer);
        Assert.Null(gen.CreateResponseConstraint(null));
        Assert.Null(gen.CreateResponseConstraint(new ResponseFormat.Text()));
        Assert.IsType<JsonConstraint>(gen.CreateResponseConstraint(new ResponseFormat.JsonObject()));
        Assert.IsType<JsonSchemaConstraint>(gen.CreateResponseConstraint(new ResponseFormat.JsonSchema { Schema = ObjSchema }));
        Assert.IsType<GrammarConstraint>(gen.CreateResponseConstraint(new ResponseFormat.Grammar { GbnfGrammar = Gbnf }));
        await using var s = gen.CreateSession(Greedy() with { ResponseFormat = new ResponseFormat.JsonSchema { Schema = ObjSchema } }, Tag());
        Assert.IsType<JsonSchemaConstraint>(s.ActiveConstraint);
    }

    private sealed class ThrowingConstraint(IDecodingConstraint inner) : IDecodingConstraint
    {
        public int ThrowOnAdvanceNumber;
        private int _n;
        public void Advance(int tokenId) { if (++_n == ThrowOnAdvanceNumber) throw new InvalidOperationException("boom"); inner.Advance(tokenId); }
        public TokenMask GetAllowedTokens() => inner.GetAllowedTokens();
        public bool IsComplete() => inner.IsComplete();
        public void Reset() { _n = 0; inner.Reset(); }
        public IDecodingConstraint Clone() => new ThrowingConstraint(inner.Clone()) { ThrowOnAdvanceNumber = ThrowOnAdvanceNumber };
    }

    private sealed class FixedMaskConstraint(int vocab, bool complete, int[] allowed) : IDecodingConstraint
    {
        public void Advance(int tokenId) { }
        public TokenMask GetAllowedTokens()
        {
            var m = new TokenMask(vocab);
            foreach (int a in allowed) m.Allow(a);
            return m;
        }
        public bool IsComplete() => complete;
        public void Reset() { }
        public IDecodingConstraint Clone() => this;
    }

    private sealed class ExtraEogTokenizer(ITokenizer inner, int extraEog) : ITokenizer
    {
        public IReadOnlyList<int> ExtraEndOfGenerationTokenIds => [extraEog];
        public int[] Encode(string text) => inner.Encode(text);
        public string Decode(ReadOnlySpan<int> tokenIds) => inner.Decode(tokenIds);
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => inner.Decode(tokenIds, stripBosSpace);
        public string DecodeToken(int tokenId) => inner.DecodeToken(tokenId);
        public int BosTokenId => inner.BosTokenId;
        public int EosTokenId => inner.EosTokenId;
        public int VocabSize => inner.VocabSize;
        public int CountTokens(string text) => inner.CountTokens(text);
    }
}
