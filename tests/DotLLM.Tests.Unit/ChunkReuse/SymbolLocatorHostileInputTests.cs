using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Tooling;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Hostile corpus text must not take the process down. Roslyn's parser recurses with the nesting depth, so a file with thousands of
/// nested brackets overflows the stack (a stack overflow cannot be caught). <see cref="SymbolLocator"/> parses candidate files
/// itself, so it follows the same rule as the corpus and the symbol chunker: parse on the large-stack thread, and skip a file nested
/// deeper than <see cref="RoslynParsing.MaxNestingDepth"/>.
/// </summary>
public sealed class SymbolLocatorHostileInputTests
{
    private static string DeeplyNested(int depth) =>
        "class A\n{\n    void Target()\n    {\n        int x = " + new string('(', depth) + "1" + new string(')', depth) + ";\n    }\n}\n";

    [Fact]
    public void Find_FileNestedBeyondTheCap_IsSkipped_NotACrash()
    {
        string deep = DeeplyNested(5000);
        Assert.True(RoslynParsing.TooDeep(deep, out int depth), "control: the input really is over the cap");
        Assert.True(depth >= 5000);
        var files = new[]
        {
            new SourceFile("deep.cs", deep),
            new SourceFile("ok.cs", "class B\n{\n    void Target() { }\n}\n"),
        };

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find(files, ["Target"]);

        Assert.Equal(["ok.cs"], matches.Select(m => m.FilePath));
    }

    /// <summary>Offsets are UTF-16 positions into the file text, also after characters outside the BMP and non-ASCII ones.</summary>
    [Fact]
    public void Find_OffsetsAreUtf16_AfterNonBmpAndNonAsciiText()
    {
        const string emoji = "\U0001F600";
        Assert.Equal(2, emoji.Length); // control: the escape reached the string (one code point, two UTF-16 units)
        string text = "// " + emoji + " éè\nclass C\n{\n    string s = \"" + emoji + emoji + "\";\n    void Target() { }\n}\n";

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find([new SourceFile("u.cs", text)], ["Target"]);

        SymbolDefinition d = Assert.Single(matches);
        Assert.Equal("Target", text.Substring(d.NameOffset, "Target".Length));
        Assert.StartsWith("void Target()", text.Substring(d.StartOffset), StringComparison.Ordinal);
        Assert.EndsWith("{ }", text.Substring(d.StartOffset, d.EndOffset - d.StartOffset), StringComparison.Ordinal);
        Assert.Equal(text[..d.NameOffset].Count(ch => ch == '\n') + 1, d.Line);
    }

    [Fact]
    public void Find_FileNestedAtTheCap_IsStillSearched()
    {
        // The cap is on brackets: 'class A {' and 'void Target() {' and the cast / call parens already count, so stay well under it.
        string nested = DeeplyNested(100);
        Assert.False(RoslynParsing.TooDeep(nested, out _));

        (IReadOnlyList<SymbolDefinition> matches, _) = SymbolLocator.Find([new SourceFile("nested.cs", nested)], ["Target"]);

        Assert.Equal(["nested.cs"], matches.Select(m => m.FilePath));
    }
}
