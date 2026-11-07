using DotLLM.ChunkReuse.CodeCorpus;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>
/// Finds declarations by (a dotted suffix of) their qualified name, by syntax: types, delegates, methods, constructors,
/// properties (including positional record parameters), events, fields, enum members. Files are parsed with
/// <see cref="SymbolChunker.ParseOptions"/> (no preprocessor symbols, so a declaration inside an inactive <c>#if</c> region is
/// not found); only files whose text contains the symbol's last name are parsed.
/// </summary>
/// <remarks>
/// Not supported, because their name is not an identifier a query can spell: indexers (<c>this[…]</c>), operators and
/// conversion operators, finalizers (<c>~T</c>; the type itself is found). Identifiers written with Unicode escapes
/// (<c>A</c>) are matched by their value but the text prefilter looks for the literal name, so such a file is skipped.
/// Local functions and locals are not declarations here.
/// </remarks>
public static class SymbolLocator
{
    /// <summary>Splits <paramref name="symbol"/> into name segments (see <see cref="SelectionRequest.Symbol"/>).</summary>
    /// <exception cref="ArgumentException">Empty or malformed.</exception>
    public static string[] ParseQuery(string symbol)
    {
        ArgumentNullException.ThrowIfNull(symbol);
        string s = symbol.Trim();
        if (s.StartsWith("global::", StringComparison.Ordinal)) s = s["global::".Length..];
        int paren = s.IndexOf('(');
        if (paren >= 0)
        {
            if (!s.EndsWith(')')) throw new ArgumentException($"Malformed symbol '{symbol}': unbalanced parameter list.", nameof(symbol));
            s = s[..paren].TrimEnd();
        }
        // Drop generic argument lists (balanced) and arity suffixes.
        var sb = new System.Text.StringBuilder();
        int depth = 0;
        foreach (char c in s)
        {
            if (c == '<') depth++;
            else if (c == '>')
            {
                if (--depth < 0) throw new ArgumentException($"Malformed symbol '{symbol}': unbalanced '>'.", nameof(symbol));
            }
            else if (depth == 0) sb.Append(c);
        }
        if (depth != 0) throw new ArgumentException($"Malformed symbol '{symbol}': unbalanced '<'.", nameof(symbol));
        string[] segments = sb.ToString().Split('.').Select(x => x.Trim()).Select(x => x.Contains('`') ? x[..x.IndexOf('`')] : x).ToArray();
        if (segments.Length == 0 || segments.Any(x => x.Length == 0 || !SyntaxFacts.IsValidIdentifier(x.TrimStart('@'))))
            throw new ArgumentException($"Malformed symbol '{symbol}': expected Name or A.B.Name.", nameof(symbol));
        string? keyword = segments.FirstOrDefault(x => !x.StartsWith('@') && SyntaxFacts.GetKeywordKind(x) != SyntaxKind.None);
        if (keyword is not null)
            throw new ArgumentException($"Symbol '{symbol}': '{keyword}' is a C# keyword. Indexers, operators and finalizers cannot be selected by name; select their type, or write @{keyword} for an identifier.", nameof(symbol));
        return segments.Select(x => x.TrimStart('@')).ToArray();
    }

    /// <summary>
    /// Declarations in <paramref name="files"/> whose qualified name ends with <paramref name="segments"/> (ordinal), and
    /// suggestions: qualified names of declarations whose last name equals the query's ignoring case but that do not match.
    /// Both ordered by path, then offset.
    /// </summary>
    public static (IReadOnlyList<SymbolDefinition> Matches, IReadOnlyList<string> Suggestions) Find(IReadOnlyList<SourceFile> files, string[] segments)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(segments);
        string name = segments[^1];
        var matches = new List<SymbolDefinition>();
        var suggestions = new List<string>();
        foreach (SourceFile file in files.OrderBy(f => f.Path, StringComparer.Ordinal))
        {
            if (!file.Text.Contains(name, StringComparison.OrdinalIgnoreCase)) continue;
            // Roslyn's parser recurses with the nesting depth and a stack overflow cannot be caught: a file nested deeper than the
            // cap (the corpus and the symbol chunker apply the same rule) is skipped, and the rest are parsed on the large-stack thread.
            if (RoslynParsing.TooDeep(file.Text, out int bracketDepth))
            {
                System.Diagnostics.Trace.TraceWarning($"{file.Path}: bracket nesting {bracketDepth} exceeds {RoslynParsing.MaxNestingDepth}; not searched.");
                continue;
            }
            SyntaxTree tree = RoslynParsing.Parse(file.Text, file.Path);
            // Nesting without brackets (generic angles, '.' chains, ternaries, unary operators) passes the bracket cap: measure the tree.
            int syntaxDepth = RoslynParsing.SyntaxDepth(tree.GetRoot());
            if (syntaxDepth > RoslynParsing.MaxSyntaxDepth)
            {
                System.Diagnostics.Trace.TraceWarning($"{file.Path}: syntax depth {syntaxDepth} exceeds {RoslynParsing.MaxSyntaxDepth}; not searched.");
                continue;
            }
            foreach ((string[] qualified, string kind, SyntaxToken id, TextSpan span) in Declarations(tree.GetRoot()))
            {
                if (!qualified[^1].Equals(name, StringComparison.OrdinalIgnoreCase)) continue;
                if (EndsWith(qualified, segments))
                {
                    int line = tree.GetLineSpan(id.Span).StartLinePosition.Line + 1;
                    matches.Add(new SymbolDefinition(string.Join('.', qualified), kind, file.Path, line, id.SpanStart, span.Start, span.End));
                }
                else
                {
                    suggestions.Add(string.Join('.', qualified));
                }
            }
        }
        return (matches, suggestions.Distinct(StringComparer.Ordinal).ToArray());
    }

    private static bool EndsWith(string[] qualified, string[] segments)
    {
        if (qualified.Length < segments.Length) return false;
        int offset = qualified.Length - segments.Length;
        for (int i = 0; i < segments.Length; i++)
            if (!string.Equals(qualified[offset + i], segments[i], StringComparison.Ordinal)) return false;
        return true;
    }

    private readonly record struct TextSpan(int Start, int End);

    private static IEnumerable<(string[] Qualified, string Kind, SyntaxToken Identifier, TextSpan Span)> Declarations(SyntaxNode root)
    {
        // Member bodies hold no declarations we look for (local functions are not members).
        foreach (SyntaxNode node in root.DescendantNodes(n => n is not (BlockSyntax or ArrowExpressionClauseSyntax or EqualsValueClauseSyntax)))
        {
            switch (node)
            {
                case BaseTypeDeclarationSyntax t:
                    yield return (Qualify(t, t.Identifier.ValueText), "type", t.Identifier, Span(t));
                    // Positional record parameters declare properties (record Box(int Size) → Box.Size).
                    if (t is RecordDeclarationSyntax { ParameterList: { } positional })
                        foreach (ParameterSyntax p in positional.Parameters)
                            yield return (Qualify(p, p.Identifier.ValueText), "property", p.Identifier, Span(p));
                    break;
                case DelegateDeclarationSyntax d:
                    yield return (Qualify(d, d.Identifier.ValueText), "delegate", d.Identifier, Span(d));
                    break;
                case MethodDeclarationSyntax m:
                    yield return (Qualify(m, m.Identifier.ValueText), "method", m.Identifier, Span(m));
                    break;
                case ConstructorDeclarationSyntax c:
                    yield return (Qualify(c, c.Identifier.ValueText), "constructor", c.Identifier, Span(c));
                    break;
                case PropertyDeclarationSyntax p:
                    yield return (Qualify(p, p.Identifier.ValueText), "property", p.Identifier, Span(p));
                    break;
                case EventDeclarationSyntax e:
                    yield return (Qualify(e, e.Identifier.ValueText), "event", e.Identifier, Span(e));
                    break;
                case EnumMemberDeclarationSyntax em:
                    yield return (Qualify(em, em.Identifier.ValueText), "enum-member", em.Identifier, Span(em));
                    break;
                case BaseFieldDeclarationSyntax f:
                    foreach (VariableDeclaratorSyntax v in f.Declaration.Variables)
                        yield return (Qualify(f, v.Identifier.ValueText), f is EventFieldDeclarationSyntax ? "event" : "field", v.Identifier, Span(f));
                    break;
            }
        }
    }

    private static TextSpan Span(SyntaxNode n) => new(n.SpanStart, n.Span.End);

    private static string[] Qualify(SyntaxNode node, string name)
    {
        var parts = new List<string> { name };
        for (SyntaxNode? p = node.Parent; p is not null; p = p.Parent)
        {
            switch (p)
            {
                case BaseTypeDeclarationSyntax t:
                    parts.Add(t.Identifier.ValueText);
                    break;
                case BaseNamespaceDeclarationSyntax ns:
                    string[] nsParts = ns.Name.ToString().Split('.', StringSplitOptions.TrimEntries);
                    for (int i = nsParts.Length - 1; i >= 0; i--) parts.Add(nsParts[i]);
                    break;
            }
        }
        parts.Reverse();
        return parts.ToArray();
    }
}
