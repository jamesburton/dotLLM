using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Conventions;

/// <summary>
/// Guard against the "silent no-op pass" anti-pattern (issues #421 and #307): a hardware- or
/// fixture-gated test that logs a skip reason and then <c>return;</c>s. xunit sees a method that
/// ran to completion, so the case reports <b>passed</b> while having asserted nothing.
/// </summary>
/// <remarks>
/// <para><b>Why a source scan rather than reflection or a Roslyn analyzer.</b> Reflection over the
/// test assemblies cannot see control flow — a silent-skip method and a real one have identical
/// metadata, and decompiling IL to spot an early <c>ret</c> guarded by a log call is far more
/// fragile than reading the source. A Roslyn analyzer would be precise, but needs a new analyzer
/// project, package plumbing and per-project wiring before it fails anything; this scan runs inside
/// the suite that already runs everywhere, needs no new build infrastructure, and reports the exact
/// <c>file:line</c>. The anti-pattern is textual (a log call followed by a bare <c>return;</c>), so
/// a textual check is proportionate.</para>
/// <para><b>The rule.</b> Inside a <c>[Fact]</c>/<c>[Theory]</c>/<c>[SkippableFact]</c>/
/// <c>[SkippableTheory]</c> method body, a bare <c>return;</c> must not be preceded — within
/// <see cref="LookbackLines"/> lines, and with no intervening <c>Skip.*</c>, <c>Assert.*</c> or
/// <c>throw</c> (see <see cref="Barrier"/>) — by a test-output write. Use
/// <c>Skip.If</c>/<c>Skip.IfNot</c> instead, which reports
/// <b>skipped</b>. A deliberate exception can be annotated with a
/// <c>// silent-skip-ok: &lt;reason&gt;</c> comment on the <c>return;</c> line or the line above it.</para>
/// <para><b>Known gap.</b> This rule requires a log line, so it does not catch the *unlogged*
/// variant — <c>if (!Avx2.IsSupported) return;</c> and friends, which bail before asserting
/// anything and say nothing at all. A survey while fixing #307 counted ~46 of those, nearly all
/// hardware-capability gates under <c>tests/DotLLM.Tests.Unit/Cpu/Kernels/</c>. They are silent
/// only on hardware lacking the ISA, which is a different (and much larger) mechanical change
/// than the fixture gates #307 is about, so they are left for a follow-up rather than widened
/// into here — a rule that fails on 46 pre-existing sites could not be landed.</para>
/// </remarks>
public sealed class SilentSkipConventionTests
{
    /// <summary>Comment marker opting a specific <c>return;</c> out of the rule.</summary>
    private const string SuppressionMarker = "silent-skip-ok";

    /// <summary>How many lines above a <c>return;</c> are searched for a preceding log call.</summary>
    private const int LookbackLines = 8;

    private static readonly Regex TestAttribute = new(
        @"\[\s*(?:SkippableFact|SkippableTheory|Fact|Theory)\b",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    private static readonly Regex BareReturn = new(
        @"^\s*return\s*;\s*$", RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    private static readonly Regex OutputWrite = new(
        @"(?:_output|Output|_testOutput|testOutput|Console)\s*\.\s*Write",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    /// <summary>
    /// Statements that prove the test is not a silent no-op, and therefore stop the look-back:
    /// a <c>Skip.*</c> call (which throws <c>SkipException</c>), any <c>throw</c>, or an
    /// <c>Assert.*</c> — if an assertion ran between the log line and the <c>return;</c>,
    /// the case genuinely exercised something.
    /// </summary>
    private static readonly Regex Barrier = new(
        @"\bSkip\s*\.\s*(?:If|IfNot|Always)\b|\bthrow\b|\bAssert\s*\.",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    private readonly ITestOutputHelper _output;

    public SilentSkipConventionTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void NoTestReturnsSilentlyAfterLoggingASkip()
    {
        string? testsRoot = FindTestsRoot();
        Skip.If(testsRoot is null,
            "Repository source tree not reachable from the test output directory; "
            + "this convention scan only runs from a source build.");

        var violations = new List<string>();
        int filesScanned = 0;

        foreach (string file in Directory.EnumerateFiles(testsRoot!, "*.cs", SearchOption.AllDirectories))
        {
            if (IsBuildOutput(file, testsRoot!))
                continue;

            filesScanned++;
            foreach ((int line, string logLine) in FindViolations(File.ReadAllLines(file)))
            {
                violations.Add(string.Create(CultureInfo.InvariantCulture,
                    $"{Path.GetRelativePath(testsRoot!, file)}:{line} — bare 'return;' after: {logLine.Trim()}"));
            }
        }

        _output.WriteLine($"scanned {filesScanned} .cs files under {testsRoot}");
        Assert.True(filesScanned > 0, $"Convention scan found no source files under '{testsRoot}'.");

        if (violations.Count == 0)
            return; // silent-skip-ok: success path of the convention scan itself, not a gated test.

        var sb = new StringBuilder();
        sb.Append(CultureInfo.InvariantCulture,
            $"{violations.Count} test(s) log a skip reason and then 'return;', which xunit reports as PASSED ")
          .AppendLine("even though nothing executed (see issues #421 / #307).")
          .AppendLine("Replace each with Skip.If(...)/Skip.IfNot(...), or annotate the line with")
          .AppendLine($"'// {SuppressionMarker}: <reason>' if the early return is genuinely intentional.")
          .AppendLine();
        foreach (string v in violations)
            sb.AppendLine(v);

        Assert.Fail(sb.ToString());
    }

    /// <summary>
    /// The detector, factored out so <see cref="DetectorFlagsTheAntiPattern"/> can prove it
    /// discriminates rather than trivially passing.
    /// </summary>
    internal static IEnumerable<(int Line, string LogLine)> FindViolations(IReadOnlyList<string> lines)
    {
        ArgumentNullException.ThrowIfNull(lines);

        int i = 0;
        while (i < lines.Count)
        {
            if (!TestAttribute.IsMatch(lines[i]))
            {
                i++;
                continue;
            }

            (int start, int end) = MethodBodySpan(lines, i);
            if (start < 0)
            {
                i++;
                continue;
            }

            for (int k = start; k <= end; k++)
            {
                if (!BareReturn.IsMatch(lines[k]) || IsSuppressed(lines, k))
                    continue;

                for (int b = k - 1; b >= Math.Max(start, k - LookbackLines); b--)
                {
                    if (Barrier.IsMatch(lines[b]))
                        break;
                    if (OutputWrite.IsMatch(lines[b]))
                    {
                        yield return (k + 1, lines[b]);
                        break;
                    }
                }
            }

            i = end + 1;
        }
    }

    /// <summary>
    /// Discrimination test: the detector must flag the broken form and accept the fixed one.
    /// Without this, a detector that silently matched nothing would still make the scan green.
    /// </summary>
    [Fact]
    public void DetectorFlagsTheAntiPattern()
    {
        string[] broken =
        [
            "    [SkippableFact]",
            "    public void Gated()",
            "    {",
            "        if (path is null)",
            "        {",
            "            _output.WriteLine(\"[SKIP] fixture not found\");",
            "            return;",
            "        }",
            "        Assert.True(true);",
            "    }",
        ];

        string[] fixedForm =
        [
            "    [SkippableFact]",
            "    public void Gated()",
            "    {",
            "        Skip.If(path is null, \"fixture not found\");",
            "        Assert.True(true);",
            "    }",
        ];

        string[] suppressed =
        [
            "    [SkippableFact]",
            "    public void Gated()",
            "    {",
            "        _output.WriteLine(\"diagnostic only\");",
            $"        return; // {SuppressionMarker}: diagnostic harness, nothing to assert.",
            "    }",
        ];

        // An assertion between the log line and the return proves the case exercised something.
        string[] assertedThenReturned =
        [
            "    [SkippableFact]",
            "    public void Gated()",
            "    {",
            "        _output.WriteLine(\"[VK head] observed ...\");",
            "        Assert.True(vkParis, \"expected Paris\");",
            "        return;",
            "    }",
        ];

        Assert.Single(FindViolations(broken));
        Assert.Empty(FindViolations(fixedForm));
        Assert.Empty(FindViolations(suppressed));
        Assert.Empty(FindViolations(assertedThenReturned));
    }

    // ── Helpers ──

    private static bool IsSuppressed(IReadOnlyList<string> lines, int index)
        => lines[index].Contains(SuppressionMarker, StringComparison.Ordinal)
           || (index > 0 && lines[index - 1].Contains(SuppressionMarker, StringComparison.Ordinal));

    /// <summary>
    /// Brace-counts from the attribute at <paramref name="attributeIndex"/> to the end of the
    /// method body, returning the inclusive line range of the body (or <c>(-1, -1)</c> for an
    /// expression-bodied or otherwise brace-less member).
    /// </summary>
    private static (int Start, int End) MethodBodySpan(IReadOnlyList<string> lines, int attributeIndex)
    {
        int depth = 0;
        int start = -1;

        for (int j = attributeIndex; j < lines.Count; j++)
        {
            string line = lines[j];
            if (start < 0 && line.Contains('{', StringComparison.Ordinal))
                start = j;

            depth += Count(line, '{') - Count(line, '}');

            if (start >= 0 && depth <= 0)
                return (start, j);
        }

        return (-1, -1);
    }

    private static int Count(string s, char c)
    {
        int n = 0;
        foreach (char ch in s)
        {
            if (ch == c)
                n++;
        }

        return n;
    }

    private static bool IsBuildOutput(string file, string root)
    {
        string rel = Path.GetRelativePath(root, file);
        return rel.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
            || rel.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
            || rel.StartsWith($"obj{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
            || rel.StartsWith($"bin{Path.DirectorySeparatorChar}", StringComparison.Ordinal);
    }

    /// <summary>
    /// Walks up from the test output directory looking for the repository root (identified by
    /// <c>dotLLM.slnx</c> alongside a <c>tests</c> directory), then returns that <c>tests</c> tree.
    /// Returns <c>null</c> when running outside a source checkout.
    /// </summary>
    private static string? FindTestsRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string tests = Path.Combine(dir.FullName, "tests");
            if (File.Exists(Path.Combine(dir.FullName, "dotLLM.slnx")) && Directory.Exists(tests))
                return tests;
            dir = dir.Parent;
        }

        return null;
    }
}
