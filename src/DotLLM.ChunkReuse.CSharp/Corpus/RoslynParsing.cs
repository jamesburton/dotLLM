using System.Runtime.ExceptionServices;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>
/// Guards Roslyn against hostile nesting. Roslyn's parser recurses per nesting level, and a deep enough file (about 2000
/// nested parentheses on a default 1–1.5 MB thread) overflows the stack, which no <c>catch</c> can stop: the process dies.
/// Two defences: files nested deeper than <see cref="MaxNestingDepth"/> are not parsed at all (callers fall back), and the
/// parse that does run uses a dedicated thread with a <see cref="ParseThreadStackBytes"/> stack.
/// </summary>
internal static class RoslynParsing
{
    /// <summary>Stack size of the parse thread (256 MiB, reserved, committed only as used).</summary>
    public const int ParseThreadStackBytes = 256 * 1024 * 1024;

    /// <summary>
    /// Deepest bracket nesting (<c>( [ {</c>) a file may have and still be parsed. Hand-written code stays far below it; the
    /// limit is low enough that later Roslyn stages (binding a compilation's trees on an ordinary thread) are safe too.
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
