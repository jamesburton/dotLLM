using System.Runtime.ExceptionServices;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Guards Roslyn against hostile nesting. Roslyn's parser recurses per nesting level, and a deep enough file (about 2000
/// nested parentheses on a default 1–1.5 MB thread) overflows the stack, which no <c>catch</c> can stop: the process dies.
/// Three defences: files nested deeper than <see cref="MaxNestingDepth"/> brackets are not parsed at all (callers fall back); a
/// parsed tree deeper than <see cref="MaxSyntaxDepth"/> is not bound; and the parse and the binding that do run use a dedicated
/// thread with a <see cref="ParseThreadStackBytes"/> stack. Callers that bind (the graph builder) must use the second and third.
/// The first defence alone is a PARSE guard: it does not make binding safe.
/// </summary>
internal static class RoslynParsing
{
    /// <summary>Stack size of the parse thread (256 MiB, reserved, committed only as used).</summary>
    public const int ParseThreadStackBytes = 256 * 1024 * 1024;

    /// <summary>
    /// Deepest bracket nesting (<c>( [ {</c>) a file may have and still be parsed. Hand-written code stays far below it. This
    /// guards the PARSE only: nesting that uses no brackets is not counted, see <see cref="MaxSyntaxDepth"/>.
    /// </summary>
    public const int MaxNestingDepth = 512;

    /// <summary>
    /// Deepest <c>( [ {</c> nesting of <paramref name="text"/>, counted naively: brackets inside strings, chars and comments
    /// count too. That can only over-count a file with hundreds of unbalanced openers in literals, which then merely takes the
    /// fallback path.
    /// </summary>
    public static int NestingDepth(string text)
    {
        int depth = 0, max = 0;
        foreach (char c in text)
        {
            if (c is '(' or '[' or '{')
            {
                if (++depth > max) max = depth;
            }
            else if (c is ')' or ']' or '}')
            {
                if (depth > 0) depth--;
            }
        }
        return max;
    }

    /// <summary>
    /// Deepest syntax-tree depth a file may have and still be bound. The bracket cap does not see nesting that uses no brackets
    /// (generic angles, '.' chains, ternaries, unary operators, 'else if' chains), and Roslyn's binder recurses per level. Hand-written
    /// code stays far below this; a 1000-term sum is about 1000 deep.
    /// </summary>
    public const int MaxSyntaxDepth = 2048;

    /// <summary>
    /// Deepest syntax-tree depth a file may have and still go into a PUBLIC compilation (<see cref="Corpus.Compilation"/>,
    /// <c>ProjectCompilation.BclOnly</c>). Callers bind those on their own, ordinary stack (about 1 MiB), where Roslyn's
    /// binder overflowed between 400 and 600 levels (nested lambdas, nested switch expressions), so the cap sits well under that.
    /// <see cref="MaxSyntaxDepth"/> is only safe where this library controls the stack (<see cref="RunWithLargeStack{T}"/>).
    /// </summary>
    public const int MaxCompilationSyntaxDepth = 256;

    /// <summary>
    /// Depth of the deepest node under <paramref name="root"/> (a lone node is 1), computed without recursion so a hostile tree
    /// cannot overflow the stack while it is measured. Parse the tree first (on the large-stack thread).
    /// </summary>
    public static int SyntaxDepth(SyntaxNode root)
    {
        var stack = new Stack<(SyntaxNode Node, int Depth)>();
        stack.Push((root, 1));
        int max = 0;
        while (stack.Count > 0)
        {
            (SyntaxNode node, int depth) = stack.Pop();
            if (depth > max) max = depth;
            foreach (SyntaxNode child in node.ChildNodes()) stack.Push((child, depth + 1));
        }
        return max;
    }

    /// <summary>True when <paramref name="text"/> is nested too deeply to hand to Roslyn.</summary>
    public static bool TooDeep(string text, out int depth)
    {
        depth = NestingDepth(text);
        return depth > MaxNestingDepth;
    }

    /// <summary>Parses <paramref name="text"/> on a <see cref="ParseThreadStackBytes"/> thread.</summary>
    public static SyntaxTree Parse(string text, string path) =>
        RunWithLargeStack(() => CSharpSyntaxTree.ParseText(text, SymbolChunker.ParseOptions, path));

    /// <summary>Runs <paramref name="work"/> on a fresh thread with a <see cref="ParseThreadStackBytes"/> stack and returns its
    /// result; an exception is rethrown on the caller with its original stack trace.</summary>
    public static T RunWithLargeStack<T>(Func<T> work)
    {
        T result = default!;
        ExceptionDispatchInfo? error = null;
        var thread = new Thread(() =>
        {
            try { result = work(); }
            catch (Exception ex) { error = ExceptionDispatchInfo.Capture(ex); }
        }, ParseThreadStackBytes) { IsBackground = true, Name = "dotllm-roslyn-parse" };
        thread.Start();
        thread.Join();
        error?.Throw();
        return result;
    }
}
