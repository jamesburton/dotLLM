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
    /// <see cref="MaxSyntaxDepth"/> is only safe where this library controls the stack (<c>RunWithLargeStack</c>).
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

    /// <summary>
    /// Longest file (in UTF-16 chars) handed to Roslyn. Hand-written files stay far below it; a larger one (a generated blob, or
    /// a hostile file with hundreds of thousands of members) takes the fallback path (chunked by lines, left out of the compilation,
    /// not searched) instead of allocating gigabytes in the parser.
    /// </summary>
    public const int MaxFileChars = 2 * 1024 * 1024;

    /// <summary>True when <paramref name="text"/> is too long to hand to Roslyn.</summary>
    public static bool TooLarge(string text, out int length)
    {
        length = text.Length;
        return length > MaxFileChars;
    }

    /// <summary>
    /// Time one file may take to parse (or to bind, in the graph builder and call resolution). Some small inputs take minutes in
    /// Roslyn (a long chain of <c>await</c>, hundreds of thousands of statements on one line): the work is cancelled at the budget
    /// and the file is treated like a too-deep one.
    /// </summary>
    public static readonly TimeSpan DefaultFileBudget = TimeSpan.FromSeconds(30);

    /// <summary>How long after cancellation a runner still waits for the work to stop before it abandons the thread.</summary>
    public static readonly TimeSpan CancelGrace = TimeSpan.FromSeconds(10);

    /// <summary>Parses <paramref name="text"/> on a <see cref="ParseThreadStackBytes"/> thread; throws <see cref="TimeoutException"/> after <paramref name="budget"/> (default <see cref="DefaultFileBudget"/>).</summary>
    public static SyntaxTree Parse(string text, string path, TimeSpan? budget = null) =>
        RunWithLargeStack(ct => CSharpSyntaxTree.ParseText(text, SymbolChunker.ParseOptions, path, cancellationToken: ct), budget ?? DefaultFileBudget);

    /// <summary>Runs <paramref name="work"/> on a fresh thread with a <see cref="ParseThreadStackBytes"/> stack and returns its
    /// result; an exception is rethrown on the caller with its original stack trace. No time limit.</summary>
    public static T RunWithLargeStack<T>(Func<T> work) => RunWithLargeStack(_ => work(), Timeout.InfiniteTimeSpan);

    /// <summary>
    /// As <see cref="RunWithLargeStack{T}(Func{T})"/> with a time limit: <paramref name="work"/> gets a token that is cancelled
    /// after <paramref name="timeout"/> (Roslyn's parse and binding calls honour it). When the work still has not stopped
    /// <paramref name="grace"/> (default <see cref="CancelGrace"/>) later, its thread is abandoned (it is a background thread) and
    /// the caller gets a <see cref="TimeoutException"/> either way.
    /// </summary>
    public static T RunWithLargeStack<T>(Func<CancellationToken, T> work, TimeSpan timeout, TimeSpan? grace = null)
    {
        T result = default!;
        ExceptionDispatchInfo? error = null;
        using var cts = new CancellationTokenSource();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        if (timeout == TimeSpan.Zero) cts.Cancel();
        else if (timeout != Timeout.InfiniteTimeSpan) cts.CancelAfter(timeout);
        var thread = new Thread(() =>
        {
            try { result = work(cts.Token); }
            catch (Exception ex) { error = ExceptionDispatchInfo.Capture(ex); }
        }, ParseThreadStackBytes) { IsBackground = true, Name = "dotllm-roslyn-parse" };
        thread.Start();
        if (timeout == Timeout.InfiniteTimeSpan)
        {
            thread.Join();
        }
        else if (!thread.Join(timeout + (grace ?? CancelGrace)))
        {
            cts.Cancel();
            throw new TimeoutException($"Roslyn work did not stop {(grace ?? CancelGrace).TotalSeconds:F0} s after its {timeout.TotalSeconds:F0} s budget; the thread was abandoned.");
        }
        // Roslyn's parser polls the token only now and then, so work can finish after its budget ran out: that is an overrun too.
        if (timeout != Timeout.InfiniteTimeSpan && (clock.Elapsed > timeout || cts.IsCancellationRequested) && (error is null || error.SourceException is OperationCanceledException))
            throw new TimeoutException($"Roslyn work exceeded its {timeout.TotalSeconds:F0} s budget.");
        error?.Throw();
        return result;
    }
}
