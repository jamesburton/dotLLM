using DotLLM.ChunkReuse.CodeCorpus;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Hostile-input and mutation-killing tests for the symbol chunker and the code corpus (PR #1646 review): deep nesting must
/// not overflow the stack inside Roslyn, the directory loader must not follow links out of its root.
/// </summary>
public sealed class CorpusHardeningTests
{
    private static string DeepParens(int depth) =>
        "class C\n{\n    int Shallow() => 0;\n\n    int M()\n    {\n        return " + new string('(', depth) + "1" + new string(')', depth)
        + ";\n    }\n}\n";

    /// <summary>
    /// ~2000 nested parentheses overflowed the stack inside Roslyn's parser on a default-size thread, killing the process
    /// (a StackOverflowException cannot be caught). A crash here aborts the test host, which the runner reports as a failure.
    /// </summary>
    [Fact]
    public void SymbolChunker_5000NestedParens_DoesNotKillTheProcess()
    {
        string text = DeepParens(5000);
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(new SourceFile("deep.cs", text));
        CorpusTests.AssertCoversWithoutOverlap(text, chunks);
        CorpusTests.AssertOffsetsTile(text, chunks);
    }

    [Fact]
    public void Corpus_Compilation_5000NestedParens_DoesNotKillTheProcess()
    {
        var files = new[] { new SourceFile("deep.cs", DeepParens(5000)), new SourceFile("ok.cs", "class Ok { int M() => 1; }\n") };
        (Microsoft.CodeAnalysis.CSharp.CSharpCompilation compilation, Dictionary<string, Microsoft.CodeAnalysis.SyntaxTree> trees) =
            Corpus.CreateCompilation(files, implicitUsings: false);
        Assert.Equal(2, trees.Count);
        Assert.NotNull(compilation.GetTypeByMetadataName("Ok"));
    }
}
