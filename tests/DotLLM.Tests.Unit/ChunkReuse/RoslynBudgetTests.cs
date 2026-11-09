using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Roslyn work is bounded in size and in time (#1664, from the #1665 review): some small inputs take minutes in the parser or the
/// binder (a long chain of <c>await</c>, hundreds of thousands of statements on one line) and a file with hundreds of thousands
/// of members allocates gigabytes. A file over <see cref="RoslynParsing.MaxFileChars"/> is not handed to Roslyn; a file that
/// overruns its budget is cancelled and treated like a too-deep one; the rest of the corpus is unaffected.
/// </summary>
public sealed class RoslynBudgetTests
{
    private const string Ok = "class Lib\n{\n    public static int Add(int a, int b) => a + b;\n}\n\nclass Use\n{\n    int F() => Lib.Add(1, 2);\n}\n";

    /// <summary>A legitimate file with many members: parsing it takes far longer than a millisecond.</summary>
    private static string ManyMembers(int n) =>
        "class Big\n{\n" + string.Concat(Enumerable.Range(0, n).Select(i => "    int M" + i + "(int a) => a + " + i + " + Lib.Add(a, 1);\n")) + "}\n";

    private static string OverTheSizeCap() => "// " + new string('x', RoslynParsing.MaxFileChars) + "\nclass Huge { }\n";

    // ---- RunWithLargeStack

    [Fact]
    public async Task RunWithLargeStack_CooperativeWorkOverTheTimeout_ThrowsTimeoutException_Promptly()
    {
        // No clock: the work parks on the token and can only end because the timeout cancelled it. The grace is an hour, so a
        // call that returned through the grace path (waiting for a thread that never stops) would not return at all.
        using var sawCancellation = new ManualResetEventSlim();
        using var threadEnded = new ManualResetEventSlim();

        Exception? thrown = await HangGuard.ReturnsAsync(Task.Run(() => Record.Exception(() => RoslynParsing.RunWithLargeStack<int>(ct =>
        {
            try
            {
                ct.WaitHandle.WaitOne();
                sawCancellation.Set();
                ct.ThrowIfCancellationRequested();
                return 0;
            }
            finally { threadEnded.Set(); }
        }, TimeSpan.FromMilliseconds(100), grace: TimeSpan.FromHours(1)))), nameof(RoslynParsing.RunWithLargeStack));

        TimeoutException timeout = Assert.IsType<TimeoutException>(thrown);
        Assert.Contains("exceeded", timeout.Message, StringComparison.Ordinal); // the budget path, not the abandoned-thread path
        Assert.True(sawCancellation.IsSet, "the work was cancelled through its token");
        Assert.True(threadEnded.IsSet, "the work thread was joined, not abandoned");
    }

    [Fact]
    public async Task RunWithLargeStack_WorkThatIgnoresTheToken_IsAbandonedAfterTheGrace()
    {
        // No clock: the work stays blocked on `release` until the test lets it go, so the call can only return by abandoning it.
        using var started = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var workFinished = new ManualResetEventSlim();
        try
        {
            Exception? thrown = await HangGuard.ReturnsAsync(Task.Run(() => Record.Exception(() => RoslynParsing.RunWithLargeStack<int>(_ =>
            {
                started.Set();
                release.Wait();
                workFinished.Set();
                return 1;
            }, TimeSpan.FromMilliseconds(50), grace: TimeSpan.FromMilliseconds(100)))), nameof(RoslynParsing.RunWithLargeStack));

            TimeoutException timeout = Assert.IsType<TimeoutException>(thrown);
            Assert.Contains("abandoned", timeout.Message, StringComparison.Ordinal);
            Assert.False(workFinished.IsSet, "the call returned while the work was still blocked, so it did not wait for it");
            started.Wait(); // the abandoned thread is the work thread: it did start
        }
        finally
        {
            release.Set(); // lets the abandoned thread finish
        }
    }

    [Fact]
    public void RunWithLargeStack_WorkThatFinishesAfterTheBudgetButWithinTheGrace_StillCountsAsAnOverrun()
    {
        // Roslyn's parser polls the token only now and then: a result that arrives late is not accepted.
        Assert.Throws<TimeoutException>(() => RoslynParsing.RunWithLargeStack(_ => { Thread.Sleep(300); return 1; },
            TimeSpan.FromMilliseconds(50), grace: TimeSpan.FromSeconds(30)));
    }

    [Fact]
    public void RunWithLargeStack_WorkWithinTheTimeout_ReturnsItsResult_AndRethrowsItsOwnExceptions()
    {
        Assert.Equal(7, RoslynParsing.RunWithLargeStack(_ => 7, TimeSpan.FromSeconds(30)));
        Assert.Throws<InvalidOperationException>(() => RoslynParsing.RunWithLargeStack<int>(_ => throw new InvalidOperationException("own"), TimeSpan.FromSeconds(30)));
    }

    // ---- size cap

    [Fact]
    public void SymbolChunker_FileOverTheSizeCap_IsChunkedByLines_NotParsed()
    {
        var warnings = new List<string>();
        string text = OverTheSizeCap();
        Assert.True(RoslynParsing.TooLarge(text, out _), "control: the file is over the cap");

        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096, warnings.Add).Chunk(new SourceFile("huge.cs", text));

        Assert.NotEmpty(chunks);
        Assert.Contains(warnings, w => w.Contains("characters exceeds", StringComparison.Ordinal));
    }

    [Fact]
    public void SymbolChunker_FileAtTheSizeCap_IsStillParsed()
    {
        var warnings = new List<string>();
        string text = "// " + new string('x', RoslynParsing.MaxFileChars - 40) + "\nclass AtTheCap { }\n";
        Assert.False(RoslynParsing.TooLarge(text, out _), "control: the file is under the cap");

        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096, warnings.Add).Chunk(new SourceFile("cap.cs", text));

        Assert.NotEmpty(chunks);
        Assert.DoesNotContain(warnings, w => w.Contains("characters exceeds", StringComparison.Ordinal));
    }

    [Fact]
    public void CreateCompilation_FileOverTheSizeCap_LeavesTheFileOut_AndKeepsTheRest()
    {
        var (compilation, trees) = Corpus.CreateCompilation([new SourceFile("ok.cs", Ok), new SourceFile("huge.cs", OverTheSizeCap())], implicitUsings: false);

        Assert.Empty(trees["huge.cs"].GetRoot().DescendantNodes());
        Assert.NotNull(compilation.GetTypeByMetadataName("Lib"));
        Assert.Null(compilation.GetTypeByMetadataName("Huge"));
    }

    [Fact]
    public void SymbolLocatorFind_FileOverTheSizeCap_IsNotSearched_AndTheOthersAre()
    {
        var files = new[] { new SourceFile("huge.cs", OverTheSizeCap() + "class Target { }\n"), new SourceFile("ok.cs", "class Target { }\n") };

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find(files, ["Target"]);

        Assert.DoesNotContain(matches, m => m.FilePath == "huge.cs");
        Assert.Contains(matches, m => m.FilePath == "ok.cs");
    }

    // ---- time budget

    [Fact]
    public void SymbolLocatorFind_ParseOverTheBudget_SkipsTheFile_NoThrow()
    {
        var files = new[] { new SourceFile("a.cs", "class Target { }\n") };
        Assert.NotEmpty(SymbolLocator.Find(files, ["Target"]).Matches); // control: the default budget finds it

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find(files, ["Target"], TimeSpan.Zero);

        Assert.Empty(matches);
    }

    [Fact]
    public void SymbolChunker_ParseOverTheBudget_FallsBackToLines()
    {
        var warnings = new List<string>();
        string text = ManyMembers(20_000);
        var file = new SourceFile("big.cs", text);
        Assert.NotEmpty(new SymbolChunker(new CharTokenizer(), 4096, warnings.Add).Chunk(file));
        Assert.Empty(warnings);

        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 4096, warnings.Add) { FileBudget = TimeSpan.Zero }.Chunk(file);

        Assert.NotEmpty(chunks);
        Assert.Contains(warnings, w => w.Contains("budget", StringComparison.Ordinal));
    }

    [Fact]
    public void CreateCompilation_ParseOverTheBudget_LeavesTheFileOut_AndKeepsTheRest()
    {
        var files = new[] { new SourceFile("ok.cs", Ok), new SourceFile("big.cs", ManyMembers(20_000)) };
        var (control, controlTrees) = Corpus.CreateCompilation(files, implicitUsings: false);
        Assert.NotEmpty(controlTrees["big.cs"].GetRoot().DescendantNodes()); // control: the default budget parses it

        var (compilation, trees) = Corpus.CreateCompilation(files, implicitUsings: false, fileBudget: TimeSpan.Zero);

        // The budget applies to every file of the call, so a zero budget leaves them all out; the point is that nothing throws.
        Assert.Empty(trees["big.cs"].GetRoot().DescendantNodes());
        Assert.Null(compilation.GetTypeByMetadataName("Big"));
        Assert.NotNull(control.GetTypeByMetadataName("Big"));
        Assert.NotNull(control.GetTypeByMetadataName("Lib"));
    }

    [Fact]
    public void GraphBuilder_BindingOverTheBudget_GivesThatFileNoEdges_AndTheOthersKeepTheirs()
    {
        var tok = new CharTokenizer();
        SourceFile[] files = [new("ok.cs", Ok)];
        Chunk[] chunks = files.SelectMany(new SymbolChunker(tok, 4096).Chunk).ToArray();
        Assert.Contains(new RoslynGraphBuilder(tok).Build(files, chunks).QueryEdges("ok.cs", 0, Ok.Length), e => e.Kind == SymbolEdgeKind.Call); // control

        ReferenceGraph graph = new RoslynGraphBuilder(tok) { FileBudget = TimeSpan.Zero }.Build(files, chunks);

        Assert.Empty(graph.QueryEdges("ok.cs", 0, Ok.Length));
    }

    [Fact]
    public void CallResolutionSites_BindingOverTheBudget_SkipsThatFile()
    {
        ProjectCompilation p = ProjectCompilation.BclOnly([new SourceFile("ok.cs", Ok)]);
        Assert.NotEmpty(CallResolution.Sites([p])); // control

        Assert.Empty(CallResolution.SitesWithin([p], TimeSpan.Zero));
    }

    // ---- git reader tasks

    [Fact]
    public void Observe_BothReadersFailing_ReturnsTheFirstFailure_AndLeavesNoTaskUnobserved()
    {
        var unobserved = new List<AggregateException>();
        void OnUnobserved(object? s, UnobservedTaskExceptionEventArgs e) { lock (unobserved) unobserved.Add(e.Exception); }
        TaskScheduler.UnobservedTaskException += OnUnobserved;
        try
        {
            Exception? first = Observe(new GitException("stdout over the cap"), new GitException("stderr over the cap"));

            Assert.Equal("stdout over the cap", first?.Message);
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            lock (unobserved) Assert.DoesNotContain(unobserved, e => e.InnerExceptions.Any(i => i.Message.EndsWith("over the cap", StringComparison.Ordinal)));
        }
        finally
        {
            TaskScheduler.UnobservedTaskException -= OnUnobserved;
        }
    }

    // Not inlined: the tasks must be unreachable once Observe returns, so the GC can finalise an unobserved one.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Exception? Observe(Exception stdoutError, Exception stderrError) =>
        GitCliInfo.Observe(Task.FromException<string>(stdoutError), Task.FromException<string>(stderrError));
}
